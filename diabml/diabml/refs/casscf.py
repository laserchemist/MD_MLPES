"""State-averaged CASSCF scans with orbital continuity, plus per-state character diagnostics.

Designed for multistate scans of small molecules such as CH2OO, where:
  * several singlet states are needed at once (state averaging),
  * orbitals must be carried smoothly from geometry to geometry (projected from the nearest finished point),
  * we want to know *what* each state is: charge separation (zwitterion) and open-shell (biradical) character.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..units import HARTREE_EV


@dataclass
class CASSpec:
    ncas: int = 6
    nelecas: int = 8
    nroots: int = 4
    basis: str = '6-31g*'
    symmetry: str | None = 'Cs'
    wfnsym: str | None = "A'"
    spin_ss: float = 0.0          # <S^2> enforced on every root (0 = singlets)
    max_cycle: int = 100


def _mol(atoms, spec: CASSpec, charge=0, spin=0):
    from pyscf import gto
    return gto.M(atom=[(a, tuple(x)) for a, x in atoms], basis=spec.basis, symmetry=spec.symmetry or False,
                 charge=charge, spin=spin, verbose=0, unit='Angstrom')


def run_sa_casscf(atoms, spec: CASSpec, guess=None):
    """One SA-CASSCF calculation. guess = (mo_coeff, mol) from a nearby geometry, projected onto this one."""
    from pyscf import scf, mcscf
    mol = _mol(atoms, spec)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASSCF(mf, spec.ncas, spec.nelecas).state_average_([1.0 / spec.nroots] * spec.nroots)
    if spec.wfnsym:
        mc.fcisolver.wfnsym = spec.wfnsym
    mc.fix_spin_(ss=spec.spin_ss)
    mc.max_cycle_macro = spec.max_cycle
    mo = mcscf.project_init_guess(mc, guess[0], guess[1]) if guess is not None else None
    mc.kernel(mo)
    return mol, mc


def state_diagnostics(mol, mc, fragment_atoms):
    """Per-state charge on a fragment (Mulliken, from the state's own density) and effective unpaired electrons.

    fragment_atoms: atom indices whose summed Mulliken charge is reported (e.g. the terminal O of CH2OO).
    Effective unpaired electrons n_u = sum_i min(n_i, 2 - n_i) over the state's active natural occupations
    (0 for a closed shell, 2 for a perfect singlet biradical)."""
    ncore, ncas = mc.ncore, mc.ncas
    C = mc.mo_coeff
    S = mol.intor('int1e_ovlp')
    Cc, Ca = C[:, :ncore], C[:, ncore:ncore + ncas]
    dm_core = 2 * Cc @ Cc.T
    charges, nu = [], []
    for d1 in mc.fcisolver.states_make_rdm1(mc.ci, ncas, mc.nelecas):
        occ = np.linalg.eigvalsh(d1)
        nu.append(float(np.sum(np.minimum(occ, 2 - occ))))
        dm = dm_core + Ca @ d1 @ Ca.T
        pop = np.einsum('ij,ji->i', dm, S)
        q = mol.atom_charges().astype(float)
        for i, (b0, b1) in enumerate(mol.aoslice_by_atom()[:, 2:4]):
            q[i] -= pop[b0:b1].sum()
        charges.append(float(q[list(fragment_atoms)].sum()))
    return np.array(charges), np.array(nu)


def active_irreps(mol, mc):
    from pyscf import symm
    if not mol.symmetry:
        return []
    lab = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mc.mo_coeff)
    return list(lab[mc.ncore:mc.ncore + mc.ncas])


def nevpt2_states(mc, spec: CASSpec):
    """Strongly contracted NEVPT2 for every root of a converged SA-CASSCF.

    PySCF's NEVPT2 needs a multi-root CASCI, so the state-averaged orbitals are reused in a CASCI with the same
    roots, symmetry and spin, and each root is corrected separately (state-specific SC-NEVPT2). Returns
    (E_nevpt2, E_casci) in hartree. Near avoided crossings a multistate method (QD-NEVPT2, XMS-CASPT2) is the
    better choice; compare with the CASSCF curves before trusting state-specific corrections there."""
    from pyscf import mcscf, mrpt
    ci = mcscf.CASCI(mc._scf, spec.ncas, spec.nelecas)
    ci.fcisolver.nroots = spec.nroots
    if spec.wfnsym:
        ci.fcisolver.wfnsym = spec.wfnsym
    ci.fix_spin_(ss=spec.spin_ss)
    ci.verbose = 0
    ci.kernel(mc.mo_coeff)
    e_cas = np.atleast_1d(np.asarray(ci.e_tot, float))
    e_corr = np.array([mrpt.NEVPT(ci, root=i).set(verbose=0).kernel() for i in range(spec.nroots)])
    return e_cas + e_corr, e_cas


def _visit_order(z):
    """Greedy nearest-neighbour path through standardised points, starting at point 0."""
    order, left = [0], set(range(1, len(z)))
    while left:
        last = z[order[-1]]
        nxt = min(left, key=lambda j: np.sum((z[j] - last) ** 2))
        order.append(nxt); left.remove(nxt)
    return order


def scan(points, atoms_fn, spec: CASSpec, fragment_atoms=(2,), log=print, start_guess=None,
         checkpoint=None, dynamic=None):
    """Run SA-CASSCF over a list of internal-coordinate tuples, carrying orbitals from the nearest finished point.

    points: (N, d) array of internal coordinates; atoms_fn(*point) -> atom list.
    The visiting order is greedy nearest-neighbour (in standardised coordinates) from the first point, so each
    calculation starts from orbitals of a geometry close by. Returns a dict of arrays in the original order.

    checkpoint: optional .npz path. Results and orbitals are written after every point, and a scan restarted with
    the same points and checkpoint continues where it stopped (orbital continuity is kept across the restart).
    dynamic: None or 'nevpt2' (adds SC-NEVPT2 energies for every root, see nevpt2_states)."""
    import json
    import os
    pts = np.asarray(points, float)
    N = len(pts)
    z = (pts - pts.mean(0)) / (pts.std(0) + 1e-12)
    order = _visit_order(z)
    R = dict(E_hartree=np.full((N, spec.nroots), np.nan), charge=np.full((N, spec.nroots), np.nan),
             n_unpaired=np.full((N, spec.nroots), np.nan), converged=np.zeros(N, bool),
             active_space_consistent=np.zeros(N, bool), done=np.zeros(N, bool), seconds=np.zeros(N))
    if dynamic == 'nevpt2':
        R['E_nevpt2_hartree'] = np.full((N, spec.nroots), np.nan)
        R['casci_mismatch_hartree'] = np.full(N, np.nan)
    mo_store, ref_irr = None, None
    if checkpoint and os.path.exists(checkpoint):
        ck = np.load(checkpoint, allow_pickle=False)
        if ck['points'].shape == pts.shape and np.allclose(ck['points'], pts):
            for k in R:
                if k in ck.files:
                    R[k] = ck[k].copy()
            mo_store = ck['mo'].copy() if 'mo' in ck.files else None
            ref_irr = json.loads(str(ck['ref_irr'])) if 'ref_irr' in ck.files else None
            if log:
                log(f'  resuming: {int(R["done"].sum())}/{N} points already done')
        elif log:
            log('  checkpoint belongs to a different grid; starting fresh')
    live = {}                                    # point -> (mo_coeff, mol) computed in this session

    def guess_from(j):
        if j in live:
            return live[j]
        return mo_store[j], _mol(atoms_fn(*pts[j]), spec)

    def save():
        if not checkpoint:
            return
        tmp = str(checkpoint) + '.tmp.npz'
        extra = {} if mo_store is None else {'mo': mo_store}
        np.savez(tmp, points=pts, ref_irr=np.array(json.dumps(ref_irr)), **R, **extra)
        os.replace(tmp, checkpoint)

    t0 = time.time()
    n_new = 0
    for k, i in enumerate(order):
        if R['done'][i]:
            continue
        t1 = time.time()
        finished = np.where(R['done'])[0]
        if len(finished):
            j = int(finished[np.argmin(np.sum((z[finished] - z[i]) ** 2, 1))])
            guess = guess_from(j)
        else:
            guess = start_guess
        mol, mc = run_sa_casscf(atoms_fn(*pts[i]), spec, guess)
        if not mc.converged and len(finished):   # retry from the first point's orbitals
            mol, mc = run_sa_casscf(atoms_fn(*pts[i]), spec, guess_from(order[0]))
        R['E_hartree'][i] = mc.e_states
        R['charge'][i], R['n_unpaired'][i] = state_diagnostics(mol, mc, fragment_atoms)
        R['converged'][i] = mc.converged
        irr = sorted(active_irreps(mol, mc))
        ref_irr = irr if ref_irr is None else ref_irr
        R['active_space_consistent'][i] = irr == ref_irr
        if dynamic == 'nevpt2':
            e_pt, e_ci = nevpt2_states(mc, spec)
            R['E_nevpt2_hartree'][i] = e_pt
            R['casci_mismatch_hartree'][i] = float(np.max(np.abs(e_ci - np.asarray(mc.e_states))))
        if mo_store is None:
            mo_store = np.full((N,) + mc.mo_coeff.shape, np.nan)
        mo_store[i] = mc.mo_coeff
        live[i] = (mc.mo_coeff, mol)
        R['done'][i] = True
        R['seconds'][i] = time.time() - t1
        n_new += 1
        save()
        if log and (n_new % 10 == 1 or R['done'].all()):
            log(f'  {int(R["done"].sum()):4d}/{N}  point {np.round(pts[i], 3)}  converged={mc.converged}  '
                f'{R["seconds"][i]:.1f} s/pt  {time.time() - t0:.0f} s this session')
    return R


def to_ev(E_hartree, zero_hartree):
    return (np.asarray(E_hartree) - zero_hartree) * HARTREE_EV
