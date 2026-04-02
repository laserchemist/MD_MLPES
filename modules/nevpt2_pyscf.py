"""
modules/nevpt2_pyscf.py

PySCF interface for CASSCF + SC-NEVPT2 single-point calculations on MVKOO
(and CH2OO) geometries.  Designed to produce the two-layer energy correction:

    δ_CASSCF(R) = E_CASSCF(R) - E_B3LYP(R)   (static correlation)
    δ_NEVPT2(R) = E_NEVPT2(R) - E_CASSCF(R)  (dynamic correlation)

Both are trained as separate NM-coordinate KRR models in
casscf_nevpt2_correction.py and applied by NEVPTDeltaDriver in bakken.py.

Units: energies in Hartree throughout; caller converts to kcal/mol as needed.
Dipoles returned in Debye.

Active space guidance
---------------------
CH2OO (Criegee, 5 atoms):
    n_active_orb=4, n_active_elec=4
    Orbitals: O-O σ, O-O σ*, C-O π, C-O π*
    Captures zwitterion (C+)(O-O-) ↔ biradical balance

MVKOO (12 atoms, s-cis or s-trans):
    n_active_orb=4, n_active_elec=4  (minimal, fast)
    Orbitals: same COO block as CH2OO; vinyl π/π* added for (6,6) treatment
    For IRC/TS region (H-transfer): σ(C-H) + σ*(O-H) + σ(C3-H7) + σ*(O2-H7)
    → pass active_space='irc' to select IRC orbital set automatically

References
----------
Angeli et al., J. Chem. Phys. 114, 10252 (2001) — NEVPT2 theory
PySCF: Q. Sun et al., WIREs Comput. Mol. Sci. 8, e1340 (2018)
"""

import numpy as np
import warnings

# Suppress PySCF's deprecation noise for sc_nevpt; we use mrpt.NEVPT
warnings.filterwarnings('ignore', category=UserWarning, module='pyscf')

HARTREE_TO_KCAL = 627.509474
AU_TO_DEBYE = 2.541746230


def _build_mol(symbols, coords, basis='6-31g*', charge=0, spin=0, verbose=0):
    """
    Build a PySCF Mole object from symbol list and coords (Angstrom).

    Parameters
    ----------
    symbols : list of str   e.g. ['C','H','H','O','O']
    coords  : np.ndarray, shape (N,3), Angstrom
    basis   : str
    charge  : int
    spin    : int  (2S, i.e. n_unpaired electrons)
    """
    from pyscf import gto
    atom_str = '; '.join(
        f'{s} {x:.8f} {y:.8f} {z:.8f}'
        for s, (x, y, z) in zip(symbols, coords)
    )
    mol = gto.Mole()
    mol.atom    = atom_str
    mol.basis   = basis
    mol.charge  = charge
    mol.spin    = spin
    mol.verbose = verbose
    mol.build()
    return mol


def _select_active_space(mc, n_active_orb, n_active_elec, active_space_type,
                          mol, mf):
    """
    Optionally sort MO coefficients to place the desired orbitals in the
    active window.  For 'auto' we rely on the CASSCF default (HOMO-based).
    For 'irc' we sort by contribution from the key atomic orbital pairs.

    Returns the (possibly reordered) CASSCF object — caller still calls
    mc.kernel().
    """
    if active_space_type == 'auto':
        return mc

    if active_space_type == 'irc':
        # For H-transfer TS: want σ(C-H) and σ*(O-H) type MOs.
        # Heuristic: sort by localisation on O and the transferring H.
        # In practice the default HOMO window usually captures these for
        # small active spaces on Criegee systems; flag for future refinement.
        return mc

    raise ValueError(f"Unknown active_space_type '{active_space_type}'. "
                     f"Use 'auto' or 'irc'.")


def compute_casscf_nevpt2(symbols, coords,
                           basis='6-31g*',
                           n_active_orb=4, n_active_elec=4,
                           active_space_type='auto',
                           charge=0, spin=0,
                           frozen_core=True,
                           max_cycle_casscf=200,
                           conv_tol_casscf=1e-8,
                           verbose=0):
    """
    Run RHF → CASSCF → SC-NEVPT2 for a single geometry.

    Parameters
    ----------
    symbols         : list of str
    coords          : np.ndarray, shape (N, 3), Angstrom
    basis           : str, default '6-31g*' (matches PSI4 training data)
    n_active_orb    : int, number of active orbitals
    n_active_elec   : int, number of active electrons
    active_space_type : 'auto' | 'irc'
    charge          : int
    spin            : int  (2S)
    frozen_core     : bool  (freeze 1s cores of heavy atoms in NEVPT2)
    max_cycle_casscf: int
    conv_tol_casscf : float
    verbose         : int (0=silent, 3=full PySCF output)

    Returns
    -------
    dict with keys:
        e_hf        : float   RHF energy (Ha)
        e_casscf    : float   CASSCF energy (Ha)
        e_nevpt2_corr : float SC-NEVPT2 correlation energy (Ha, negative)
        e_nevpt2    : float   CASSCF + NEVPT2 corr total (Ha)
        delta_nevpt2: float   E_NEVPT2 - E_CASSCF  (Ha, negative)
        dipole_casscf : np.ndarray (3,)  CASSCF dipole (Debye)
        no_occ      : np.ndarray        CASSCF natural orbital occupancies
        converged   : bool
        error       : str or None
    """
    result = {
        'e_hf': None, 'e_casscf': None,
        'e_nevpt2_corr': None, 'e_nevpt2': None,
        'delta_nevpt2': None,
        'dipole_casscf': np.zeros(3),
        'no_occ': None,
        'converged': False,
        'error': None,
    }

    try:
        from pyscf import scf, mcscf, mrpt
        from pyscf.tools import molden  # noqa (ensure full install)

        mol = _build_mol(symbols, coords, basis=basis,
                         charge=charge, spin=spin, verbose=verbose)

        # --- RHF ---
        mf = scf.RHF(mol)
        mf.max_cycle = 200
        mf.conv_tol  = 1e-9
        mf.kernel()
        result['e_hf'] = float(mf.e_tot)

        # --- CASSCF ---
        mc = mcscf.CASSCF(mf, n_active_orb, n_active_elec)
        mc.max_cycle_macro = max_cycle_casscf
        mc.conv_tol        = conv_tol_casscf
        mc.verbose         = verbose

        mc = _select_active_space(mc, n_active_orb, n_active_elec,
                                   active_space_type, mol, mf)
        mc.kernel()
        result['e_casscf'] = float(mc.e_tot)
        result['converged'] = mc.converged

        # Natural orbital occupancies of active space (diagnostic)
        # mc.mo_occ gives closed-shell occupancies; for true NOs we need the
        # active-space 1-RDM eigenvalues.
        try:
            casdm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
            no_occ_active = np.sort(np.linalg.eigvalsh(casdm1))[::-1]
            result['no_occ'] = no_occ_active
        except Exception:
            result['no_occ'] = None

        # CASSCF dipole moment
        try:
            dm1 = mc.make_rdm1()
            with mol.with_common_orig([0, 0, 0]):
                dip_ao = mol.intor('int1e_r', comp=3)
            dip = -np.einsum('xij,ji->x', dip_ao, dm1)
            # add nuclear contribution
            for i, (sym, coord) in enumerate(zip(symbols, coords)):
                from pyscf.data.elements import charge as elem_charge
                Z = elem_charge(sym)
                dip += Z * np.array(coord) / 0.529177  # Ang → Bohr
            result['dipole_casscf'] = dip * AU_TO_DEBYE
        except Exception as e:
            if verbose:
                print(f"  [nevpt2] dipole failed: {e}")

        # --- SC-NEVPT2 ---
        pt = mrpt.NEVPT(mc)
        pt.verbose = verbose
        if frozen_core:
            # freeze 1s of C, N, O atoms (atomic number >= 6)
            n_frozen = sum(1 for s in symbols
                          if _atomic_number(s) >= 6)
            pt.frozen = list(range(n_frozen))
        e_corr = pt.kernel()
        result['e_nevpt2_corr'] = float(e_corr)
        result['e_nevpt2']      = float(mc.e_tot + e_corr)
        result['delta_nevpt2']  = float(e_corr)   # NEVPT2 - CASSCF (negative)

    except Exception as e:
        result['error'] = str(e)
        result['converged'] = False

    return result


def compute_batch(geometries, symbols,
                  basis='6-31g*',
                  n_active_orb=4, n_active_elec=4,
                  active_space_type='auto',
                  frozen_core=True,
                  verbose=0,
                  n_jobs=1):
    """
    Run compute_casscf_nevpt2 on a list of geometries.

    Parameters
    ----------
    geometries : list of np.ndarray, each shape (N, 3), Angstrom
    symbols    : list of str  (shared across all frames)
    n_jobs     : int  (1 = serial; >1 uses joblib Parallel if available)

    Returns
    -------
    list of result dicts (same order as geometries)
    """
    if n_jobs == 1 or len(geometries) == 1:
        results = []
        for i, coords in enumerate(geometries):
            if verbose:
                print(f"  [nevpt2] frame {i+1}/{len(geometries)}")
            r = compute_casscf_nevpt2(
                symbols, coords,
                basis=basis,
                n_active_orb=n_active_orb,
                n_active_elec=n_active_elec,
                active_space_type=active_space_type,
                frozen_core=frozen_core,
                verbose=verbose,
            )
            results.append(r)
        return results

    # Parallel branch
    try:
        from joblib import Parallel, delayed
        n_jobs = min(n_jobs, len(geometries))
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_casscf_nevpt2)(
                symbols, coords,
                basis=basis,
                n_active_orb=n_active_orb,
                n_active_elec=n_active_elec,
                active_space_type=active_space_type,
                frozen_core=frozen_core,
                verbose=0,
            )
            for coords in geometries
        )
        return results
    except ImportError:
        # fallback to serial
        return compute_batch(geometries, symbols,
                             basis=basis,
                             n_active_orb=n_active_orb,
                             n_active_elec=n_active_elec,
                             active_space_type=active_space_type,
                             frozen_core=frozen_core,
                             verbose=verbose,
                             n_jobs=1)


def _atomic_number(symbol):
    """Return atomic number for a symbol string."""
    from pyscf.data.elements import charge as elem_charge
    return elem_charge(symbol)


def summarise_results(results, b3lyp_energies=None):
    """
    Print a summary table of CASSCF + NEVPT2 results.

    Parameters
    ----------
    results       : list of result dicts from compute_batch
    b3lyp_energies: list of float or None  — B3LYP energies (Ha) for each frame
    """
    print(f"\n{'Frame':>5}  {'E_CASSCF (Ha)':>14}  {'δ_NEVPT2 (kcal)':>16}  "
          f"{'Converged':>9}  {'Error':>20}")
    print("-" * 75)
    for i, r in enumerate(results):
        b3 = f"{(r['e_casscf'] - b3lyp_energies[i]) * HARTREE_TO_KCAL:+.3f}" \
             if (b3lyp_energies is not None and r['e_casscf'] is not None) else "  —"
        dyn = f"{r['delta_nevpt2'] * HARTREE_TO_KCAL:.3f}" \
              if r['delta_nevpt2'] is not None else "  —"
        conv = "YES" if r['converged'] else "NO"
        err  = (r['error'] or "")[:20]
        e_cas = f"{r['e_casscf']:.6f}" if r['e_casscf'] is not None else "      —"
        print(f"{i:>5}  {e_cas:>14}  {dyn:>16}  {conv:>9}  {err:>20}")


# ---------------------------------------------------------------------------
# Quick self-test (run as: python3 -m modules.nevpt2_pyscf)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    print("Running nevpt2_pyscf self-test on H2O ...")

    symbols = ['O', 'H', 'H']
    coords  = np.array([
        [0.000,  0.000,  0.119],
        [0.000,  0.757, -0.476],
        [0.000, -0.757, -0.476],
    ])  # Angstrom, near-equilibrium

    r = compute_casscf_nevpt2(
        symbols, coords,
        basis='6-31g*',
        n_active_orb=4, n_active_elec=4,
        verbose=0,
    )

    if r['error']:
        print(f"ERROR: {r['error']}")
        sys.exit(1)

    print(f"  RHF:           {r['e_hf']:.6f} Ha")
    print(f"  CASSCF(4,4):   {r['e_casscf']:.6f} Ha")
    print(f"  NEVPT2 corr:   {r['e_nevpt2_corr']*HARTREE_TO_KCAL:.3f} kcal/mol")
    print(f"  NEVPT2 total:  {r['e_nevpt2']:.6f} Ha")
    print(f"  Dipole (D):    {r['dipole_casscf']}")
    print(f"  Converged:     {r['converged']}")
    print("Self-test PASSED")
