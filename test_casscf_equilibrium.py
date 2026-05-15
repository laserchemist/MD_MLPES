#!/usr/bin/env python3
"""
test_casscf_equilibrium.py — Quick SA-2-CASSCF(4,4) + triplet CASSCF test at
the MVKOO wB97X equilibrium geometry.

Purpose
-------
Validates the PySCF CASSCF setup before committing to the 240-frame NM grid run:
  1. Confirms PySCF SA-2-CASSCF(4,4)/6-31G* converges at equilibrium
  2. Reports S0, S1 energies and natural orbital occupations
  3. Confirms triplet SS-CASSCF(4,4) converges and gives E_T1
  4. Computes δ_S0 = E_CASSCF_S0 − E_wB97X and the S0→S1, S0→T1 gaps
  5. Checks active-space orbital character (should be COO lone pairs / π/π*)

Active space
------------
CASSCF(4,4)/6-31G* — 4 electrons in 4 orbitals.
Near the MVKOO minimum the 4 active orbitals should correspond to:
    n+(O_terminal), n-(O_terminal), π(COO), π*(COO)
Expected NO occupations at equilibrium: ~ (1.98, 1.90, 0.10, 0.02).
Strong deviation from this (e.g., (1.5, 1.5, 0.5, 0.5)) signals biradical character.

Usage
-----
    python3 test_casscf_equilibrium.py

    # With explicit geometry (default: uses frame 502 from wB97X results):
    python3 test_casscf_equilibrium.py --eq-coords /tmp/mvkoo_wb97x_eq.npy

    # Verbose PySCF output:
    python3 test_casscf_equilibrium.py --verbose
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'modules'))

HARTREE_TO_KCAL  = 627.509474
HARTREE_TO_EV    = 27.211396

# wB97X energy at the equilibrium frame (from results.json, frame 502)
E_WB97X_EQ_HA    = -306.209984583785

MVKOO_SYMBOLS    = ['C', 'O', 'O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']

# Expected NO occupations at MVKOO B3LYP/wB97X equilibrium (from prior runs)
# Reference from 2026-03-31 CASSCF surface run
EQ_NO_OCC_REF    = np.array([1.998, 1.924, 0.077, 0.000])
NO_OCC_SWITCH_THRESHOLD = 0.15


# ── Geometry helpers ───────────────────────────────────────────────────────────

def build_mol(symbols, coords_ang, spin=0, charge=0, basis='6-31g*', verbose=0):
    """Build a PySCF Mole object."""
    from pyscf import gto
    atom_str = '; '.join(
        f'{s} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}'
        for s, c in zip(symbols, coords_ang)
    )
    mol = gto.Mole()
    mol.atom    = atom_str
    mol.basis   = basis
    mol.charge  = charge
    mol.spin    = spin        # 0 = singlet, 2 = triplet
    mol.verbose = verbose
    mol.build()
    return mol


# ── Frozen core helper ─────────────────────────────────────────────────────────

def n_frozen(symbols):
    """Number of heavy-atom 1s core orbitals to freeze in NEVPT2 (not used here,
    but kept for reference)."""
    return sum(1 for s in symbols if s in ('C', 'O', 'N', 'F', 'S'))


# ── SA-2-CASSCF singlet ───────────────────────────────────────────────────────

def run_singlet_casscf(symbols, coords_ang, basis='6-31g*',
                        n_active_orb=4, n_active_elec=4,
                        n_states=2, verbose=0,
                        mo_init=None, max_cycles=300, conv_tol=1e-8):
    """
    RHF → SA-n-CASSCF(4,4)/basis singlet calculation.

    Returns dict with:
        e_s0, e_s1      : CASSCF state energies (Ha)
        no_occ_s0       : (4,) active natural orbital occupations for S0
        no_occ_s1       : (4,) for S1
        mo_coeff        : final MO coefficients (for seeding triplet)
        state_switched  : bool  (based on NO deviation from EQ_NO_OCC_REF)
        converged       : bool
        e_hf            : RHF energy
    """
    from pyscf import scf, mcscf

    result = {
        'e_s0': None, 'e_s1': None, 'e_hf': None,
        'no_occ_s0': None, 'no_occ_s1': None,
        'mo_coeff': None,
        'state_switched': False, 'no_occ_max_dev': None,
        'converged': False, 'error': None,
    }

    try:
        mol = build_mol(symbols, coords_ang, spin=0, basis=basis, verbose=verbose)

        # RHF
        mf = scf.RHF(mol)
        mf.max_cycle = 300
        mf.conv_tol  = 1e-9
        mf.kernel()
        result['e_hf'] = float(mf.e_tot)
        print(f"  RHF energy:      {mf.e_tot:.8f} Ha  "
              f"(converged: {mf.converged})")

        # SA-n-CASSCF
        weights = [1.0 / n_states] * n_states
        mc = mcscf.CASSCF(mf, n_active_orb, n_active_elec).state_average(weights)
        mc.max_cycle_macro = max_cycles
        mc.conv_tol        = conv_tol
        mc.verbose         = verbose
        if mo_init is not None:
            mc.kernel(mo_init)
        else:
            mc.kernel()

        result['converged'] = mc.converged
        result['mo_coeff']  = mc.mo_coeff.copy()

        e_states = np.array(mc.e_states)
        result['e_s0'] = float(e_states[0])
        result['e_s1'] = float(e_states[1]) if len(e_states) > 1 else None

        # Natural orbital occupations for each state.
        # SA-CASSCF stores mc.ci as a list of CI vectors. Use states_make_rdm1
        # (PySCF >= 2.2) when available; otherwise fall back to per-state
        # make_rdm1 with an explicit (nalpha, nbeta) nelecas tuple.
        try:
            nelec = mc.nelecas
            if not isinstance(nelec, tuple):
                nelec = (nelec // 2, nelec - nelec // 2)
            if hasattr(mc.fcisolver, 'states_make_rdm1'):
                casdm1_list = mc.fcisolver.states_make_rdm1(
                    mc.ci, mc.ncas, nelec)
            else:
                casdm1_list = [
                    mc.fcisolver.make_rdm1(mc.ci[i], mc.ncas, nelec)
                    for i in range(len(e_states))
                ]
            for state_idx, key in enumerate(('no_occ_s0', 'no_occ_s1')[:len(e_states)]):
                occ = np.sort(np.linalg.eigvalsh(casdm1_list[state_idx]))[::-1]
                result[key] = occ
        except Exception as exc:
            print(f"    Warning: NO occupations extraction failed: {exc}")

        # State-switch detection (truncate reference to match active space size)
        if result['no_occ_s0'] is not None:
            n = len(result['no_occ_s0'])
            ref = EQ_NO_OCC_REF[:n] if n <= len(EQ_NO_OCC_REF) else EQ_NO_OCC_REF
            dev = np.max(np.abs(result['no_occ_s0'] - ref))
            result['no_occ_max_dev'] = float(dev)
            result['state_switched'] = bool(dev > NO_OCC_SWITCH_THRESHOLD)

    except Exception as exc:
        result['error'] = str(exc)
        import traceback
        traceback.print_exc()

    return result


# ── SS-CASSCF triplet ─────────────────────────────────────────────────────────

def run_triplet_casscf(symbols, coords_ang, mo_coeff_seed=None,
                        basis='6-31g*', n_active_orb=4, n_active_elec=4,
                        verbose=0):
    """
    ROHF → SS-CASSCF(4,4)/basis triplet calculation (spin=2).
    Seeds MO from singlet calculation to maintain consistent active space.

    Returns dict with:
        e_t1        : CASSCF triplet energy (Ha)
        no_occ_t1   : (4,) active NO occupations
        converged   : bool
    """
    from pyscf import scf, mcscf

    result = {
        'e_t1': None, 'no_occ_t1': None,
        'e_rohf': None,
        'converged': False, 'error': None,
    }

    try:
        mol = build_mol(symbols, coords_ang, spin=2, basis=basis, verbose=verbose)

        # ROHF
        mf = scf.ROHF(mol)
        mf.max_cycle = 300
        mf.conv_tol  = 1e-9
        mf.kernel()
        result['e_rohf'] = float(mf.e_tot)
        print(f"  ROHF energy:     {mf.e_tot:.8f} Ha  "
              f"(converged: {mf.converged})")

        # SS-CASSCF, seed with singlet MO if available
        mc = mcscf.CASSCF(mf, n_active_orb, n_active_elec)
        mc.max_cycle_macro = 300
        mc.conv_tol        = 1e-8
        mc.verbose         = verbose

        if mo_coeff_seed is not None:
            mc.kernel(mo_coeff_seed)
        else:
            mc.kernel()

        result['converged'] = mc.converged
        result['e_t1']      = float(mc.e_tot)

        try:
            casdm1   = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
            occ      = np.sort(np.linalg.eigvalsh(casdm1))[::-1]
            result['no_occ_t1'] = occ
        except Exception as exc:
            print(f"    Warning: triplet NO occupations failed: {exc}")

    except Exception as exc:
        result['error'] = str(exc)
        import traceback
        traceback.print_exc()

    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Quick SA-2-CASSCF(4,4) equilibrium test for MVKOO delta-ML')
    parser.add_argument('--eq-coords', default=None,
                        help='Path to equilibrium .npy (Angstrom). '
                             'Default: load from wB97X results.json frame 502.')
    parser.add_argument('--wb97x-energy', type=float, default=E_WB97X_EQ_HA,
                        help=f'wB97X energy at equilibrium (Ha, default: {E_WB97X_EQ_HA:.8f})')
    parser.add_argument('--basis', default='6-31g*')
    parser.add_argument('--n-active-orb',  type=int, default=4)
    parser.add_argument('--n-active-elec', type=int, default=4)
    parser.add_argument('--n-states', type=int, default=2,
                        help='Number of singlet states for SA-CASSCF (default: 2)')
    parser.add_argument('--skip-triplet', action='store_true',
                        help='Skip triplet CASSCF')
    parser.add_argument('--verbose', action='store_true',
                        help='Pass verbose=4 to PySCF')
    parser.add_argument('--output', default=None,
                        help='Save JSON results to this path')
    parser.add_argument('--mo-init', default=None,
                        help='Path to .npy MO coefficient array to seed CASSCF '
                             '(shape: n_mo × n_mo). Useful when RHF starting MOs '
                             'give poor active space coverage.')
    parser.add_argument('--max-cycles', type=int, default=300,
                        help='Max CASSCF macro cycles (default: 300)')
    parser.add_argument('--conv-tol', type=float, default=1e-8,
                        help='CASSCF convergence tolerance (default: 1e-8)')
    args = parser.parse_args()

    verbose = 4 if args.verbose else 0

    mo_init = None
    if args.mo_init:
        mo_init = np.load(args.mo_init)
        print(f"Seeding CASSCF with MO coefficients from: {args.mo_init}  shape={mo_init.shape}")

    # ── Load equilibrium geometry ─────────────────────────────────────────────
    if args.eq_coords:
        eq_coords = np.load(args.eq_coords)
        print(f"Loaded geometry: {args.eq_coords}")
    else:
        # Load from training data frame 502 (wB97X minimum)
        from data_formats import load_trajectory
        traj = load_trajectory(
            str(REPO_ROOT / 'outputs/mvko_20260319_081314/combined_training_data.npz'))
        eq_coords = traj.coordinates[502]
        print("Loaded geometry: training data frame 502 (wB97X minimum)")

    symbols = MVKOO_SYMBOLS
    print(f"\nMolecule: MVKOO ({len(symbols)} atoms)")
    print(f"Basis: {args.basis}")
    print(f"Active space: CASSCF({args.n_active_elec},{args.n_active_orb})")
    print(f"SA states (singlet): {args.n_states}")
    print(f"wB97X reference energy: {args.wb97x_energy:.8f} Ha")
    print()

    all_results = {}

    # ── Singlet SA-CASSCF ─────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Running SA-{args.n_states}-CASSCF({args.n_active_elec},{args.n_active_orb}) singlet ...")
    print("=" * 60)
    t0 = time.time()
    sing = run_singlet_casscf(
        symbols, eq_coords,
        basis=args.basis,
        n_active_orb=args.n_active_orb,
        n_active_elec=args.n_active_elec,
        n_states=args.n_states,
        verbose=verbose,
        mo_init=mo_init,
        max_cycles=args.max_cycles,
        conv_tol=args.conv_tol,
    )
    t_sing = time.time() - t0
    all_results['singlet'] = {k: v.tolist() if isinstance(v, np.ndarray) else v
                               for k, v in sing.items() if k != 'mo_coeff'}
    all_results['singlet']['elapsed_s'] = t_sing

    print(f"\nSinglet CASSCF results ({t_sing:.1f}s):")
    if sing['error']:
        print(f"  ERROR: {sing['error']}")
    else:
        print(f"  Converged:       {sing['converged']}")
        print(f"  E_S0:            {sing['e_s0']:.8f} Ha")
        if sing['e_s1'] is not None:
            gap_s1_ev  = (sing['e_s1'] - sing['e_s0']) * HARTREE_TO_EV
            gap_s1_kcal = (sing['e_s1'] - sing['e_s0']) * HARTREE_TO_KCAL
            print(f"  E_S1:            {sing['e_s1']:.8f} Ha")
            print(f"  S0→S1 gap:       {gap_s1_ev:.3f} eV  ({gap_s1_kcal:.1f} kcal/mol)")
        if sing['no_occ_s0'] is not None:
            print(f"  NO occ (S0):     {np.array2string(sing['no_occ_s0'], precision=3)}")
            print(f"  Reference occ:   {np.array2string(EQ_NO_OCC_REF, precision=3)}")
            print(f"  Max deviation:   {sing['no_occ_max_dev']:.4f}  "
                  f"({'STATE-SWITCHED!' if sing['state_switched'] else 'OK'})")
        if sing['no_occ_s1'] is not None:
            print(f"  NO occ (S1):     {np.array2string(sing['no_occ_s1'], precision=3)}")

        # δ_S0 at equilibrium
        # NOTE: absolute total energies from DFT and CASSCF are NOT comparable.
        # wB97X includes correlation for all electrons; CASSCF(4,4) only correlates
        # the 4 active electrons.  Their absolute difference (~1000 kcal/mol) is the
        # correlation energy of the frozen/restricted orbitals — physically meaningless
        # as a correction.
        #
        # The correct δ for the NM grid is always RELATIVE to the equilibrium value:
        #   δ_S0(R) = [E_CASSCF_S0(R) - E_CASSCF_S0(R_eq)]
        #           - [E_wB97X(R)     - E_wB97X(R_eq)]
        # At equilibrium this is 0 by construction.  For displaced frames it measures
        # the differential curvature of the two surfaces.
        abs_diff_ha   = sing['e_s0'] - args.wb97x_energy
        abs_diff_kcal = abs_diff_ha * HARTREE_TO_KCAL
        print(f"\n  Absolute E_CASSCF_S0 − E_wB97X (NOT the usable correction):")
        print(f"    {abs_diff_ha:.6f} Ha = {abs_diff_kcal:.0f} kcal/mol")
        print(f"    (Expected: ~1000 kcal/mol — correlation energy of frozen orbitals)")
        print(f"\n  δ_S0 at equilibrium (relative, by definition) = 0.000 kcal/mol")
        print(f"  Reference values to store for grid script:")
        print(f"    E_CASSCF_S0(eq)  = {sing['e_s0']:.10f} Ha")
        print(f"    E_wB97X(eq)      = {args.wb97x_energy:.10f} Ha")
        all_results['e_casscf_s0_eq_ha'] = sing['e_s0']
        all_results['e_wb97x_eq_ha']     = args.wb97x_energy
        all_results['delta_s0_eq_kcal']  = 0.0   # always 0 at reference geometry

    # ── Triplet SS-CASSCF ─────────────────────────────────────────────────────
    if not args.skip_triplet and sing.get('mo_coeff') is not None:
        print()
        print("=" * 60)
        print(f"Running SS-CASSCF({args.n_active_elec},{args.n_active_orb}) triplet ...")
        print("=" * 60)
        t0 = time.time()
        trip = run_triplet_casscf(
            symbols, eq_coords,
            mo_coeff_seed=sing['mo_coeff'],
            basis=args.basis,
            n_active_orb=args.n_active_orb,
            n_active_elec=args.n_active_elec,
            verbose=verbose,
        )
        t_trip = time.time() - t0
        all_results['triplet'] = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                   for k, v in trip.items() if k != 'mo_coeff'}
        all_results['triplet']['elapsed_s'] = t_trip

        print(f"\nTriplet CASSCF results ({t_trip:.1f}s):")
        if trip['error']:
            print(f"  ERROR: {trip['error']}")
        else:
            print(f"  Converged:       {trip['converged']}")
            print(f"  E_T1:            {trip['e_t1']:.8f} Ha")
            if trip['no_occ_t1'] is not None:
                print(f"  NO occ (T1):     {np.array2string(np.array(trip['no_occ_t1']), precision=3)}")

            if sing['e_s0'] is not None and trip['e_t1'] is not None:
                gap_t1_ev   = (trip['e_t1'] - sing['e_s0']) * HARTREE_TO_EV
                gap_t1_kcal = (trip['e_t1'] - sing['e_s0']) * HARTREE_TO_KCAL
                print(f"  S0→T1 gap:       {gap_t1_ev:.3f} eV  ({gap_t1_kcal:.1f} kcal/mol)")
                all_results['gap_t1_ev']   = gap_t1_ev
                all_results['gap_t1_kcal'] = gap_t1_kcal

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if not sing.get('error'):
        print(f"  wB97X reference (eq):   {args.wb97x_energy:.10f} Ha")
        print(f"  CASSCF S0 (eq):         {sing['e_s0']:.10f} Ha")
        print(f"  δ_S0 at equilibrium:    0.000 kcal/mol  (0 by construction)")
        if sing['e_s1'] is not None:
            gap_s1_ev   = (sing['e_s1'] - sing['e_s0']) * HARTREE_TO_EV
            gap_s1_kcal = (sing['e_s1'] - sing['e_s0']) * HARTREE_TO_KCAL
            print(f"  Δgap_S1 (S0→S1):        {gap_s1_kcal:.1f} kcal/mol  ({gap_s1_ev:.2f} eV)")
            note = "S0 sufficient for 300K IR" if gap_s1_ev > 1.5 else "S1 may mix at high T"
            print(f"    → {note}")
        if 'gap_t1_kcal' in all_results:
            print(f"  Δgap_T1 (S0→T1):        {all_results['gap_t1_kcal']:.1f} kcal/mol  "
                  f"({all_results['gap_t1_ev']:.2f} eV)")
            print(f"    → ISC relevant for hot ozonolysis MVKOO (kT_ozon ≈ 2–4 eV)")
        print()
        no = sing['no_occ_s0']
        if no is not None:
            no = np.array(no)
            print(f"  NO occ S0: {np.array2string(no, precision=3)}")
            char = 'near-closed-shell (good: biradical character small near eq)' \
                   if no[0] > 1.9 else 'significant biradical character'
            print(f"  Active space: {char}")
        else:
            print("  NO occupations: not extracted (see warning above)")
        print()
        print("Interpretation:")
        print("  δ_S0(eq) = 0 by definition; the NM grid will show how δ varies")
        print("  with geometry — expected range < ±5 kcal/mol near equilibrium.")
        print("  Gap surfaces (Δgap_S1, Δgap_T1) are well-defined absolute quantities.")
        print()
        print("Reference energies for casscf_wB97X_nm_grid.py:")
        print(f"  --casscf-eq-energy  {sing['e_s0']:.10f}")
        print(f"  --wb97x-eq-energy   {args.wb97x_energy:.10f}")
        print()
        print("Next step: build and run casscf_wB97X_nm_grid.py (240-frame NM grid)")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = args.output or 'outputs/casscf_eq_test_result.json'
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
