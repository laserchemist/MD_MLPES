"""Per-state dipole moments from finite electric fields: SA-CASSCF and SC-NEVPT2 with PySCF.

    mu_k = -dE_k/dF  ~  (E_k(-F) - E_k(+F)) / 2F        (central difference, one pair of runs per field direction)

These are *relaxed* dipoles (orbitals, CI and the perturbation all respond to the field), i.e. the derivative of the
same energies a surface is trained on, which is what IR intensities need. For a planar molecule in Cs symmetry the
field is applied in the molecular plane only (x and y; the dipole has no z component), which keeps the mirror plane,
so the A' root selection of the reference data still applies. PySCF does not reorient the molecule, so the dipoles
are in the frame of the input coordinates (for CH2OO from diabml.geometry.ch2oo_xyz: O1 at the origin, C on -x,
O2 in the xy plane with y > 0). ORCA cannot be used here: with symmetry on it applies every field along its own z
axis, perpendicular to a Cs mirror plane (checked in orca_tests/ff_test).

Checked (orca_tests/ff_test, CH2OO SA4-CASSCF(8,6)/cc-pVDZ): field-free energies equal ORCA's to 1-3 uEh, the
state-averaged dipole equals ORCA's to 1e-4 a.u., the mean of the four finite-field CASSCF dipoles equals the
analytic state-averaged dipole (SA-CASSCF is variational for the average), and F = 0.002 vs 0.001 a.u. changes the
dipoles by < 0.001 D.
"""
from __future__ import annotations

import numpy as np

DEBYE = 2.541746          # Debye per atomic unit


def _run(atoms, spec, field, guess=None, nevpt=True):
    """SA-CASSCF (+ SC-NEVPT2 per root) in a uniform field (a.u.). guess = (mo_coeff, mol) to project from."""
    from pyscf import gto, mcscf, mrpt, scf
    mol = gto.M(atom=[(a, tuple(x)) for a, x in atoms], basis=spec['basis'], symmetry=spec.get('symmetry') or False,
                verbose=0, unit='Angstrom')
    mol.set_common_orig((0.0, 0.0, 0.0))
    F = np.asarray(field, float)
    r = mol.intor('int1e_r')
    h = mol.intor('int1e_kin') + mol.intor('int1e_nuc') + np.einsum('x,xij->ij', F, r)   # electrons: +F.r
    enuc = mol.energy_nuc() - float(F @ (mol.atom_charges() @ mol.atom_coords()))        # nuclei: -Z R.F
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *a, **k: h
    mf.energy_nuc = lambda *a, **k: enuc
    mf.kernel()
    n = int(spec['nroots'])
    mc = mcscf.CASSCF(mf, int(spec['ncas']), int(spec['nelecas'])).state_average_([1.0 / n] * n)
    if spec.get('wfnsym'):
        mc.fcisolver.wfnsym = spec['wfnsym']
    mc.fix_spin_(ss=float(spec.get('spin_ss', 0.0)))
    mc.max_cycle_macro = int(spec.get('max_cycle', 100))
    mc.conv_tol = 1e-10
    mo = mcscf.project_init_guess(mc, guess[0], guess[1]) if guess is not None else None
    mc.kernel(mo)
    e_cas = np.array(mc.e_states, float)
    e_pt = None
    if nevpt:
        ci = mcscf.CASCI(mf, int(spec['ncas']), int(spec['nelecas']))
        ci.fcisolver.nroots = n
        if spec.get('wfnsym'):
            ci.fcisolver.wfnsym = spec['wfnsym']
        ci.fix_spin_(ss=float(spec.get('spin_ss', 0.0)))
        ci.verbose = 0
        ci.kernel(mc.mo_coeff)
        e_ci = np.atleast_1d(np.asarray(ci.e_tot, float))
        e_pt = e_ci + np.array([mrpt.NEVPT(ci, root=i).set(verbose=0).kernel() for i in range(n)])
    dm = mc.make_rdm1()
    mu_sa = -np.einsum('xij,ji->x', r, dm) + mol.atom_charges() @ mol.atom_coords()
    return {'mol': mol, 'mo': mc.mo_coeff, 'e_cas': e_cas, 'e_pt': e_pt, 'mu_sa': mu_sa,
            'converged': bool(mc.converged)}


def state_dipoles(atoms, spec, F=0.002, guess=None, nevpt=True, axes=(0, 1)):
    """Finite-field dipoles of every SA-CASSCF root (and its SC-NEVPT2 energy) at one geometry.

    atoms: [(symbol, xyz Angstrom)]; spec: the reference data's CASSpec as a dict (basis, ncas, nelecas, nroots,
    symmetry, wfnsym, spin_ss). axes: field directions (0 = x, 1 = y, 2 = z); a planar molecule in the xy plane
    needs only x and y. Returns a dict:
      e_cas, e_pt         field-free energies per root (hartree; roots in SA-CASSCF / CASCI order)
      mu_cas, mu_pt       (n_roots, 3) dipoles in a.u., input frame (components not in `axes` are 0)
      mu_sa               analytic dipole of the state-averaged density (a.u.), the check on the field sign and frame
      min_gap             smallest gap between neighbouring CASSCF roots (hartree): below ~5 mEh the field can
                          reorder roots and the finite difference of those two roots is not meaningful
      mo, mol             field-free orbitals (to start a neighbour from)"""
    r0 = _run(atoms, spec, (0, 0, 0), guess=guess, nevpt=nevpt)
    g = (r0['mo'], r0['mol'])
    n = len(r0['e_cas'])
    mu_cas, mu_pt = np.zeros((n, 3)), np.zeros((n, 3)) if nevpt else None
    conv = r0['converged']
    for ax in axes:
        f = np.zeros(3); f[ax] = F
        rp = _run(atoms, spec, f, guess=g, nevpt=nevpt)
        rm = _run(atoms, spec, -f, guess=g, nevpt=nevpt)
        conv &= rp['converged'] and rm['converged']
        mu_cas[:, ax] = (rm['e_cas'] - rp['e_cas']) / (2 * F)
        if nevpt:
            mu_pt[:, ax] = (rm['e_pt'] - rp['e_pt']) / (2 * F)
    ec = np.sort(r0['e_cas'])
    return {'e_cas': r0['e_cas'], 'e_pt': r0['e_pt'], 'mu_cas': mu_cas, 'mu_pt': mu_pt, 'mu_sa': r0['mu_sa'],
            'min_gap': float(np.min(np.diff(ec))) if n > 1 else np.inf, 'converged': bool(conv),
            'mo': r0['mo'], 'mol': r0['mol']}
