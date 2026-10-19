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


def scan(points, atoms_fn, spec: CASSpec, fragment_atoms=(2,), log=print, start_guess=None):
    """Run SA-CASSCF over a list of internal-coordinate tuples, carrying orbitals from the nearest finished point.

    points: (N, d) array of internal coordinates; atoms_fn(*point) -> atom list.
    The visiting order is greedy nearest-neighbour (in standardised coordinates) from the first point, so each
    calculation starts from orbitals of a geometry close by. Returns a dict of arrays in the original order."""
    pts = np.asarray(points, float)
    N = len(pts)
    z = (pts - pts.mean(0)) / (pts.std(0) + 1e-12)
    done, order = {}, [0]
    left = set(range(1, N))
    while left:
        last = z[order[-1]]
        nxt = min(left, key=lambda j: np.sum((z[j] - last) ** 2))
        order.append(nxt); left.remove(nxt)
    E = np.full((N, spec.nroots), np.nan)
    Q = np.full((N, spec.nroots), np.nan)
    NU = np.full((N, spec.nroots), np.nan)
    conv = np.zeros(N, bool)
    irr_ok = np.zeros(N, bool)
    ref_irr = None
    t0 = time.time()
    for k, i in enumerate(order):
        if done:
            j = min(done, key=lambda m: np.sum((z[m] - z[i]) ** 2))
            guess = done[j]
        else:
            guess = start_guess
        mol, mc = run_sa_casscf(atoms_fn(*pts[i]), spec, guess)
        if not mc.converged and done:            # retry from the first point's orbitals
            mol, mc = run_sa_casscf(atoms_fn(*pts[i]), spec, done[order[0]])
        E[i] = mc.e_states
        Q[i], NU[i] = state_diagnostics(mol, mc, fragment_atoms)
        conv[i] = mc.converged
        irr = sorted(active_irreps(mol, mc))
        ref_irr = irr if ref_irr is None else ref_irr
        irr_ok[i] = irr == ref_irr
        done[i] = (mc.mo_coeff, mol)
        if log and (k % 10 == 0 or k == N - 1):
            log(f'  {k + 1:4d}/{N}  point {np.round(pts[i], 3)}  converged={mc.converged}  {time.time() - t0:.0f} s')
    return dict(E_hartree=E, charge=Q, n_unpaired=NU, converged=conv, active_space_consistent=irr_ok)


def to_ev(E_hartree, zero_hartree):
    return (np.asarray(E_hartree) - zero_hartree) * HARTREE_EV
