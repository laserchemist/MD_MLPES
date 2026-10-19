"""Full CI references for small systems (exact within the basis).

fci_states  : all-electron FCI, any charge/spin, several roots with <S^2> enforced (H, H2, H3 ...).
fc_fci      : frozen-core FCI (CASCI over all valence orbitals) with optional point-group symmetry, used for
              O, OH and H2O. Orbitals come from a closed-shell RHF of the same nuclei with a chosen reference
              charge, so fragments and molecules share one recipe.
"""
from __future__ import annotations

import numpy as np


def fci_states(xyz, charge, nelec, nroots=1, ss=None, basis='cc-pvdz', symbols=None):
    """Lowest nroots FCI energies (hartree). xyz: (n, 3) Angstrom. nelec = (n_alpha, n_beta).
    Orbitals are eigenvectors of the one-electron Hamiltonian; FCI in the full space does not depend on them."""
    from pyscf import gto, fci, ao2mo
    xyz = np.atleast_2d(xyz)
    symbols = symbols or ['H'] * len(xyz)
    mol = gto.M(atom=[(s, tuple(x)) for s, x in zip(symbols, xyz)], basis=basis, charge=charge,
                spin=nelec[0] - nelec[1], verbose=0)
    h = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    Li = np.linalg.inv(np.linalg.cholesky(mol.intor('int1e_ovlp')))
    _, Cp = np.linalg.eigh(Li @ h @ Li.T)
    C = Li.T @ Cp
    norb = C.shape[1]
    if sum(nelec) == 1:
        return np.linalg.eigvalsh(C.T @ h @ C)[:nroots] + mol.energy_nuc()
    eri = ao2mo.restore(1, ao2mo.kernel(mol, C), norb)
    solver = fci.direct_spin1.FCI()
    if ss is not None:
        solver = fci.addons.fix_spin_(solver, ss=ss, shift=0.5)
    solver.conv_tol = 1e-10
    e, _ = solver.kernel(C.T @ h @ C, eri, norb, nelec, ecore=mol.energy_nuc(), nroots=nroots)
    return np.atleast_1d(e)[:nroots]


def fc_fci(atom, charge, na, nb, nroots=1, ss=None, sym=None, wfnsym=None, ref_charge=None, basis='6-31g'):
    """Frozen-core (heavy-atom 1s) FCI. atom: PySCF atom string. Returns energies (hartree)."""
    from pyscf import gto, scf, mcscf, fci
    kw = dict(atom=atom, basis=basis, verbose=0, unit='Angstrom')
    if sym:
        kw.update(symmetry=True, symmetry_subgroup=sym)
    mol = gto.M(charge=charge, spin=na - nb, **kw)
    ref = gto.M(charge=ref_charge if ref_charge is not None else charge, spin=0, **kw)
    mf_ref = scf.RHF(ref).run()
    mf = scf.RHF(mol) if na == nb else scf.ROHF(mol)
    ncore = sum(1 for a in mol._atom if a[0] not in ('H', 'He'))
    mc = mcscf.CASCI(mf, mol.nao - ncore, (na - ncore, nb - ncore))
    mc.fcisolver.nroots = nroots
    if wfnsym is not None:
        mc.fcisolver.wfnsym = wfnsym
    if ss is not None:
        mc.fcisolver = fci.addons.fix_spin_(mc.fcisolver, ss=ss, shift=0.5)
    mc.fcisolver.conv_tol = 1e-10
    mc.verbose = 0
    return np.atleast_1d(mc.kernel(mf_ref.mo_coeff)[0])
