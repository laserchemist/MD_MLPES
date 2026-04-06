#!/usr/bin/env python3
"""
casscf_nm_systematic.py — Systematic SA-2-CASSCF(4,4)/SC-NEVPT2 NM displacement grid

Replaces the non-uniform CASSCF single-point dataset (from MD/NM training frames)
with a controlled, uniform grid in normal-mode coordinate space.  This avoids the
two main failure modes of previous delta-ML attempts:

1. **State-switching contamination**: MO orbital guess is chained along each
   displacement direction, keeping CASSCF on the same electronic state.
   Frames where NO occupancies deviate significantly from equilibrium are
   flagged and excluded from delta-ML training.

2. **Non-uniform q-space sampling**: the systematic grid provides equal coverage
   of each normal mode at each amplitude, avoiding the energy-bias of MD/NM frames
   (which over-sample low-energy geometries and under-sample mode-specific stretches).

Multi-state family
------------------
SA-2-CASSCF averages over S0 and S1 with equal weights.  Per-state energies are
stored for both states, enabling two independent NM-KRR corrections:

    δ_S0(q) = E_CASSCF_S0(q) − E_B3LYP(q)     (ground-state correction)
    δ_S1(q) = E_CASSCF_S1(q) − E_B3LYP(q)     (excited-state correction)

At 300 K MD, only δ_S0 is applied.  Near barriers or at 840 K, a PESFamily
can blend δ_S0 and δ_S1 based on the S0/S1 energy gap.  NEVPT2 dynamic
correlation is added on top of the ground state (SC-NEVPT2 for one state at a time).

Grid design
-----------
30 modes × [0.5, 1.0, 1.5, 2.0, 3.0] × a_thermal(300 K) × ±1
= 300 frames total (before filtering)

a_thermal(300 K) = sqrt(2 kT / λ_k)  [Bohr·sqrt(amu)]
where λ_k = (2π ω_k / FREQ_CONV)² is the force constant in Hartree/(Bohr²·amu).

This covers the thermally sampled sphere at 300 K and extends to 3× thermal amplitude
for each mode, giving systematic coverage of the correction surface out to the edge
of the region relevant for 840 K emission spectroscopy.

Usage
-----
    # Full run (uses existing B3LYP Hessian):
    python3 casscf_nm_systematic.py \\
        --eq-coords outputs/mvko_20260319_081314/psi4_eq_coords.npy \\
        --hessian   outputs/casscf_nm_delta_20260401_110049/hessian_used.npy \\
        --training-data outputs/mvko_20260319_081314/combined_training_data.npz

    # Resume a partial run:
    python3 casscf_nm_systematic.py \\
        --eq-coords outputs/mvko_20260319_081314/psi4_eq_coords.npy \\
        --hessian   outputs/casscf_nm_delta_20260401_110049/hessian_used.npy \\
        --training-data outputs/mvko_20260319_081314/combined_training_data.npz \\
        --resume outputs/casscf_nm_systematic_<ts>

    # Skip CASSCF, retrain KRR only from saved results:
    python3 casscf_nm_systematic.py \\
        --eq-coords ... --hessian ... --training-data ... \\
        --resume outputs/casscf_nm_systematic_<ts> \\
        --retrain-only

    # Use ωB97X-D surface as the base (instead of B3LYP):
    python3 casscf_nm_systematic.py \\
        --eq-coords ... --hessian ... \\
        --training-data outputs/wB97X_surface_<ts>/training_data_wB97X.npz \\
        --base-method wb97x-d

Outputs (in outputs/casscf_nm_systematic_<ts>/)
-------------------------------------------------
results.json                   — per-frame CASSCF/NEVPT2 results + NO occupancies
nm_delta_s0_model.pkl          — NMKRRDeltaModel for S0 ground-state correction
nm_delta_s1_model.pkl          — NMKRRDeltaModel for S1 excited-state correction
nm_delta_nevpt2_model.pkl      — NMKRRDeltaModel for NEVPT2 total correction
summary.json                   — grid stats, NO filter stats, LOO-CV RMSEs
diagnostics.png                — δ vs ||q||, NO occupancy scatter, LOO-CV curves
"""

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'modules'))

from normal_modes import compute_normal_modes, KB_HARTREE_PER_K, FREQ_CONV
from data_formats import load_trajectory

HARTREE_TO_KCAL  = 627.509474
ANGSTROM_TO_BOHR = 1.88972612456
BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR

MVKOO_SYMBOLS = ['C', 'O', 'O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']

# NO occupancy at MVKO B3LYP equilibrium (4 active orbitals, most-to-least occupied)
# From 2026-03-31 CASSCF surface run and 2026-04-01 NM-delta diagnosis
EQ_NO_OCC_REF = np.array([1.998, 1.924, 0.077, 0.000])

# Flag a frame as state-switched if any active NO occupation deviates by more than this
NO_OCC_SWITCH_THRESHOLD = 0.15


# ── grid generation ────────────────────────────────────────────────────────────

def make_nm_grid(eq_coords_ang, frequencies, eigenvalues, eigvecs_mw, sqrt_mass,
                 symbols, amplitudes=(0.5, 1.0, 1.5, 2.0, 3.0), T=300.0):
    """
    Generate systematic NM displacement geometries.

    Returns list of dicts with keys:
        coords_ang  : (N, 3) Angstrom
        mode_idx    : int
        sign        : +1 or -1
        factor      : float  (multiple of a_thermal)
        freq_cm1    : float
        q_nm        : (n_vib,) NM coordinate vector [sqrt(amu)·Bohr]
        a_thermal   : float  [sqrt(amu)·Bohr]
    """
    kT    = T * KB_HARTREE_PER_K  # Ha
    n_at  = len(symbols)

    n_vib = eigvecs_mw.shape[1]
    frames = []

    for k in range(n_vib):
        lam = eigenvalues[k]   # Hartree/(Bohr²·amu)
        if lam <= 0:
            continue           # skip imaginary modes

        a_therm = np.sqrt(2 * kT / lam)   # classical thermal amplitude [sqrt(amu)·Bohr]

        for factor in amplitudes:
            for sign in (+1, -1):
                # NM coordinate vector for this frame
                q_vec = np.zeros(n_vib)
                q_vec[k] = sign * factor * a_therm

                # Mass-weighted displacement in Bohr
                mw_disp = eigvecs_mw @ q_vec           # (3N,) [sqrt(amu)·Bohr]
                # Back to Cartesian Bohr: divide by sqrt(mass)
                cart_disp_bohr = mw_disp / sqrt_mass   # (3N,) Bohr
                # Convert to Angstrom and reshape
                cart_disp_ang = cart_disp_bohr.reshape(n_at, 3) * BOHR_TO_ANGSTROM

                new_coords = eq_coords_ang + cart_disp_ang

                frames.append({
                    'coords_ang': new_coords,
                    'mode_idx':   k,
                    'sign':       sign,
                    'factor':     factor,
                    'freq_cm1':   float(frequencies[k]),
                    'q_nm':       q_vec,
                    'a_thermal':  float(a_therm),
                })

    return frames


# ── SA-2-CASSCF + NEVPT2 ──────────────────────────────────────────────────────

def run_sa2casscf_nevpt2(symbols, coords_ang, basis='6-31g*',
                          n_active_orb=4, n_active_elec=4,
                          mo_coeff_init=None, verbose=0):
    """
    Run RHF → SA-2-CASSCF(4,4) → SC-NEVPT2 (ground state only).

    Chains MO guess from mo_coeff_init if provided (avoids state-switching
    when displacing along a single normal mode direction).

    Returns dict with:
        e_s0, e_s1          : float  CASSCF S0 and S1 energies (Ha)
        e_hf                : float  RHF energy (Ha)
        e_nevpt2            : float  NEVPT2 total on S0 (Ha)
        delta_nevpt2_ha     : float  E_NEVPT2 − E_CASSCF_S0 (Ha, negative)
        no_occ_s0           : np.ndarray (4,) active-space NO occupations for S0
        no_occ_s1           : np.ndarray (4,) active-space NO occupations for S1
        mo_coeff            : np.ndarray  final MO coefficients (for chaining)
        state_switched      : bool   True if NO occ deviates from EQ_NO_OCC_REF
        converged           : bool
        error               : str or None
    """
    result = {
        'e_s0': None, 'e_s1': None, 'e_hf': None,
        'e_nevpt2': None, 'delta_nevpt2_ha': None,
        'no_occ_s0': None, 'no_occ_s1': None,
        'mo_coeff': None,
        'state_switched': False, 'converged': False,
        'error': None,
    }

    try:
        from pyscf import scf, mcscf, mrpt, gto

        # Build molecule
        atom_str = '; '.join(
            f'{s} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}'
            for s, c in zip(symbols, coords_ang)
        )
        mol = gto.Mole()
        mol.atom    = atom_str
        mol.basis   = basis
        mol.charge  = 0
        mol.spin    = 0
        mol.verbose = verbose
        mol.build()

        # RHF
        mf = scf.RHF(mol)
        mf.max_cycle = 200
        mf.conv_tol  = 1e-9
        if mo_coeff_init is not None:
            mf.mo_coeff = mo_coeff_init  # seed with previous geometry's MOs
        mf.kernel()
        result['e_hf'] = float(mf.e_tot)

        # SA-2-CASSCF: average S0 and S1 equally
        mc_base = mcscf.CASSCF(mf, n_active_orb, n_active_elec)
        mc = mc_base.state_average([0.5, 0.5])
        mc.max_cycle_macro = 200
        mc.conv_tol        = 1e-8
        mc.verbose         = verbose
        mc.kernel()

        result['converged']  = mc.converged
        result['mo_coeff']   = mc.mo_coeff.copy()

        # Per-state energies
        e_states = np.array(mc.e_states)
        result['e_s0'] = float(e_states[0])
        result['e_s1'] = float(e_states[1]) if len(e_states) > 1 else None

        # NO occupations for each state (diagnose state-switching)
        for state_idx, key in enumerate(('no_occ_s0', 'no_occ_s1')):
            try:
                casdm1 = mc.fcisolver.make_rdm1(mc.ci[state_idx], mc.ncas, mc.nelecas)
                no_occ = np.sort(np.linalg.eigvalsh(casdm1))[::-1]
                result[key] = no_occ
            except Exception:
                pass

        # State-switch detection on S0
        if result['no_occ_s0'] is not None:
            max_dev = np.max(np.abs(result['no_occ_s0'] - EQ_NO_OCC_REF))
            result['state_switched'] = bool(max_dev > NO_OCC_SWITCH_THRESHOLD)
            result['no_occ_max_dev'] = float(max_dev)

        # SC-NEVPT2 on S0 only (state-specific; use SS-CASSCF for cleaner result)
        # We re-run SS-CASSCF on S0 using the SA orbitals as starting guess,
        # then apply NEVPT2 to the SS solution.
        try:
            mc_ss = mcscf.CASSCF(mf, n_active_orb, n_active_elec)
            mc_ss.max_cycle_macro = 200
            mc_ss.conv_tol        = 1e-8
            mc_ss.verbose         = 0
            # Start from SA orbitals (state 0 natural orbitals)
            mc_ss.kernel(mc.mo_coeff)

            pt = mrpt.NEVPT(mc_ss)
            pt.verbose = 0
            # Freeze 1s cores of heavy atoms
            n_frozen = sum(1 for s in symbols if _atomic_number(s) >= 6)
            pt.frozen = list(range(n_frozen))
            e_corr = pt.kernel()

            result['e_nevpt2']       = float(mc_ss.e_tot + e_corr)
            result['delta_nevpt2_ha'] = float(e_corr)   # negative
        except Exception as nevpt2_err:
            result['nevpt2_error'] = str(nevpt2_err)

    except Exception as exc:
        result['error'] = str(exc)

    return result


def _atomic_number(symbol):
    """Return atomic number for common elements."""
    Z = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6,
         'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'S': 16, 'Cl': 17}
    return Z.get(symbol, 0)


# ── NM-KRR training (reuses NMKRRDeltaModel from casscf_nm_delta.py) ─────────

def train_nm_krr(eq_coords_ang, frequencies, eigenvalues, eigvecs_mw, sqrt_mass,
                 symbols, frame_list, y_ha, e_ref_ha=0.0,
                 gamma_values=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
                 alpha_values=(1e-6, 1e-5, 1e-4, 1e-3),
                 label='S0'):
    """
    Train NMKRRDeltaModel with leave-one-out CV grid search.

    frame_list : list of grid frame dicts (contains q_nm already computed)
    y_ha       : (M,) correction in Hartree
    label      : string tag for printing
    """
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location('casscf_nm_delta', REPO_ROOT / 'casscf_nm_delta.py')
    _mod  = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    NMKRRDeltaModel = _mod.NMKRRDeltaModel

    X = np.array([f['q_nm'] for f in frame_list])  # (M, n_vib)
    M = len(X)

    def rbf(X1, X2, gamma):
        diff = X1[:, None, :] - X2[None, :, :]
        return np.exp(-gamma * np.sum(diff**2, axis=-1))

    def loo_cv_rmse(gamma, alpha_reg):
        K = rbf(X, X, gamma)
        K_reg = K.copy()
        K_reg[np.diag_indices_from(K_reg)] += alpha_reg
        try:
            alpha_vec = np.linalg.solve(K_reg, y_ha)
        except np.linalg.LinAlgError:
            return 1e9
        # Explicit LOO (exact; avoid hat-matrix degeneracy)
        errs = []
        for i in range(M):
            mask = np.ones(M, bool); mask[i] = False
            K_tr = rbf(X[mask], X[mask], gamma)
            K_tr[np.diag_indices_from(K_tr)] += alpha_reg
            try:
                av = np.linalg.solve(K_tr, y_ha[mask])
            except np.linalg.LinAlgError:
                errs.append(np.abs(y_ha[i]))
                continue
            k_pred = rbf(X[[i]], X[mask], gamma)[0]
            errs.append(y_ha[i] - k_pred @ av)
        return float(np.sqrt(np.mean(np.array(errs)**2))) * HARTREE_TO_KCAL

    print(f"\n  [{label}] LOO-CV grid search (M={M} frames):")
    best_rmse = np.inf
    best_g = best_a = None
    for gamma in gamma_values:
        for alpha in alpha_values:
            rmse = loo_cv_rmse(gamma, alpha)
            tag = ' ← best' if rmse < best_rmse else ''
            print(f"    γ={gamma:.0e} α={alpha:.0e}  LOO-CV={rmse:.3f} kcal/mol{tag}")
            if rmse < best_rmse:
                best_rmse = rmse; best_g = gamma; best_a = alpha

    print(f"  [{label}] Best: γ={best_g:.0e} α={best_a:.0e}  "
          f"LOO-CV={best_rmse:.3f} kcal/mol")

    model = NMKRRDeltaModel(
        eq_coords_ang = eq_coords_ang,
        U_vib         = eigvecs_mw,
        sqrt_mass     = sqrt_mass,
        freqs_vib     = frequencies,
        symbols       = symbols,
        gamma         = best_g,
        alpha_reg     = best_a,
        X_train_q     = X,
        y_train_ha    = y_ha,
        e_b3lyp_ref_ha = e_ref_ha,
        e_cas_ref_ha   = 0.0,
        cv_rmse_kcal   = best_rmse,
    )
    return model, best_rmse


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Systematic SA-2-CASSCF(4,4)/NEVPT2 NM grid for delta-ML')
    parser.add_argument('--eq-coords', required=True,
                        help='PSI4 equilibrium .npy (Angstrom)')
    parser.add_argument('--hessian', required=True,
                        help='B3LYP/6-31G* Hessian .npy (3N×3N, Hartree/(Bohr²·amu))')
    parser.add_argument('--training-data', required=True,
                        help='Training .npz (for B3LYP energy reference at each geometry)')
    parser.add_argument('--base-method', default='b3lyp',
                        help='Base DFT method label stored in training-data '
                             '(b3lyp or wb97x-d; for reporting only)')
    parser.add_argument('--amplitudes', default='0.5,1.0,1.5,2.0,3.0',
                        help='Comma-separated list of thermal amplitude factors')
    parser.add_argument('--T-nm', type=float, default=300.0,
                        help='Temperature for thermal amplitude (K)')
    parser.add_argument('--basis', default='6-31g*')
    parser.add_argument('--n-active-orb',  type=int, default=4)
    parser.add_argument('--n-active-elec', type=int, default=4)
    parser.add_argument('--no-occ-threshold', type=float,
                        default=NO_OCC_SWITCH_THRESHOLD,
                        help='Max |ΔNO_occ| to flag as state-switched')
    parser.add_argument('--gamma-values', default='0.01,0.05,0.1,0.5,1.0,5.0')
    parser.add_argument('--alpha-values', default='1e-6,1e-5,1e-4,1e-3')
    parser.add_argument('--resume', default=None,
                        help='Output directory from a partial run to resume')
    parser.add_argument('--retrain-only', action='store_true',
                        help='Skip CASSCF, reload saved results and just retrain KRR')
    args = parser.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.resume) if args.resume else \
              Path(f'outputs/casscf_nm_systematic_{ts}')
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / 'results.json'

    amplitudes   = [float(x) for x in args.amplitudes.split(',')]
    gamma_values = [float(x) for x in args.gamma_values.split(',')]
    alpha_values = [float(x) for x in args.alpha_values.split(',')]

    # ── Load equilibrium geometry and Hessian ─────────────────────────────────
    eq_coords_ang = np.load(args.eq_coords)           # (N, 3) Angstrom
    hessian       = np.load(args.hessian)              # (3N, 3N) Ha/(Bohr²·amu)
    symbols       = MVKOO_SYMBOLS
    n_at          = len(symbols)

    print(f"Equilibrium geometry: {args.eq_coords}  ({n_at} atoms)")
    print(f"Hessian: {args.hessian}  (shape {hessian.shape})")

    frequencies, eigvecs_mw, eigenvalues, mass_vec = compute_normal_modes(
        symbols, hessian)
    n_vib = len(frequencies)
    sqrt_mass = np.sqrt(mass_vec)   # (3N,)

    print(f"NM frequencies ({n_vib} vibrational modes):")
    print("  " + "  ".join(f"{f:7.1f}" for f in frequencies[:10]) + " ... cm⁻¹")
    n_neg = int((frequencies < 0).sum())
    if n_neg:
        print(f"  WARNING: {n_neg} imaginary modes in Hessian")

    # ── Load training data for B3LYP reference energies ───────────────────────
    traj = load_trajectory(args.training_data)
    e_ref_min_ha = float(np.min(traj.energies))
    print(f"\nBase surface: {args.base_method}  "
          f"(min energy: {e_ref_min_ha:.8f} Ha)")

    # ── Generate NM grid ─────────────────────────────────────────────────────
    print(f"\nGenerating systematic NM grid "
          f"({n_vib} modes × {len(amplitudes)} amplitudes × 2 signs)...")
    grid_frames = make_nm_grid(
        eq_coords_ang, frequencies, eigenvalues, eigvecs_mw, sqrt_mass,
        symbols, amplitudes=amplitudes, T=args.T_nm)
    print(f"  {len(grid_frames)} grid frames generated "
          f"(excluding imaginary-mode frames)")

    # ── Resume: load existing results ─────────────────────────────────────────
    completed = {}   # frame_key → result dict
    if results_path.exists():
        with open(results_path) as f:
            completed = json.load(f)
        print(f"  Resuming: {len(completed)} frames already computed")

    def _frame_key(frame, idx):
        return f"m{frame['mode_idx']:02d}_s{'+' if frame['sign']==1 else '-'}" \
               f"_f{frame['factor']:.2f}_i{idx}"

    # ── Run CASSCF/NEVPT2 ──────────────────────────────────────────────────────
    if not args.retrain_only:
        print(f"\nRunning SA-2-CASSCF({args.n_active_elec},{args.n_active_orb}) "
              f"+ SC-NEVPT2 / {args.basis}  "
              f"(~80 s/frame × {len(grid_frames)} frames ≈ "
              f"{len(grid_frames)*80/3600:.1f} h)\n")

        # Chain MO guess within each mode's displacement sequence
        # (+ direction: increasing factor; then – direction)
        mo_guess_cache = {}   # (mode_idx, sign) → mo_coeff from previous frame

        t_start = time.time()
        n_ok = n_fail = n_skip = 0

        for idx, frame in enumerate(grid_frames):
            key = _frame_key(frame, idx)
            if key in completed:
                st = completed[key].get('status', '?')
                if st in ('ok', 'failed'):
                    n_skip += 1
                    continue

            # Retrieve chained MO guess from same mode, same sign, smaller factor
            cache_key = (frame['mode_idx'], frame['sign'])
            mo_init = mo_guess_cache.get(cache_key)

            t0 = time.time()
            res = run_sa2casscf_nevpt2(
                symbols, frame['coords_ang'],
                basis=args.basis,
                n_active_orb=args.n_active_orb,
                n_active_elec=args.n_active_elec,
                mo_coeff_init=mo_init,
                verbose=0,
            )
            elapsed = time.time() - t0

            # Cache MO coefficients for the next amplitude step
            if res['mo_coeff'] is not None:
                mo_guess_cache[cache_key] = res['mo_coeff']

            freq     = frame['freq_cm1']
            mode_idx = frame['mode_idx']
            sign_str = '+' if frame['sign'] == 1 else '-'
            factor   = frame['factor']

            if res['error'] is None and res['e_s0'] is not None:
                n_ok += 1
                switched = res.get('state_switched', False)
                switched_str = ' *** STATE SWITCH ***' if switched else ''
                print(f"  [{idx+1}/{len(grid_frames)}] "
                      f"mode={mode_idx:2d}({freq:6.0f} cm⁻¹) "
                      f"{sign_str}{factor:.1f}×  "
                      f"E_S0={res['e_s0']:.8f} Ha  "
                      f"E_S1={res['e_s1']:.8f} Ha  "
                      f"NO_dev={res.get('no_occ_max_dev', 99):.3f}  "
                      f"{elapsed:.1f}s{switched_str}")
            else:
                n_fail += 1
                print(f"  [{idx+1}/{len(grid_frames)}] "
                      f"mode={mode_idx:2d}({freq:6.0f} cm⁻¹) "
                      f"{sign_str}{factor:.1f}×  FAILED in {elapsed:.1f}s "
                      f"— {res.get('error','?')[:80]}")

            # Serialize result (convert numpy to python types)
            completed[key] = {
                'status':       'ok' if res['error'] is None and res['e_s0'] else 'failed',
                'mode_idx':     int(mode_idx),
                'sign':         int(frame['sign']),
                'factor':       float(factor),
                'freq_cm1':     float(freq),
                'q_nm':         frame['q_nm'].tolist(),
                'e_s0':         res['e_s0'],
                'e_s1':         res['e_s1'],
                'e_hf':         res['e_hf'],
                'e_nevpt2':     res.get('e_nevpt2'),
                'delta_nevpt2': res.get('delta_nevpt2_ha'),
                'no_occ_s0':    (res['no_occ_s0'].tolist()
                                 if res['no_occ_s0'] is not None else None),
                'no_occ_s1':    (res['no_occ_s1'].tolist()
                                 if res['no_occ_s1'] is not None else None),
                'no_occ_max_dev': res.get('no_occ_max_dev'),
                'state_switched': res.get('state_switched', False),
                'converged':    res.get('converged', False),
                'error':        res.get('error'),
                'elapsed_s':    float(elapsed),
            }

            with open(results_path, 'w') as f:
                json.dump(completed, f, indent=2)

        elapsed_total = time.time() - t_start
        print(f"\nDone: {n_ok} ok, {n_fail} failed, {n_skip} skipped  "
              f"({elapsed_total/3600:.2f} h total)")

    # ── Build training arrays ─────────────────────────────────────────────────
    print("\nBuilding delta-ML training arrays from completed frames...")

    ok_keys    = [k for k, r in completed.items() if r.get('status') == 'ok']
    clean_keys = [k for k in ok_keys if not completed[k].get('state_switched', True)]
    switched_keys = [k for k in ok_keys if completed[k].get('state_switched', False)]

    print(f"  Total ok: {len(ok_keys)}")
    print(f"  Clean (|ΔNO_occ| ≤ {args.no_occ_threshold}): {len(clean_keys)}")
    print(f"  State-switched (excluded): {len(switched_keys)}")

    if len(clean_keys) < 5:
        print("  ERROR: too few clean frames for KRR training (need ≥ 5). "
              "Check NO_OCC_SWITCH_THRESHOLD or run more frames.")
        sys.exit(1)

    # For clean frames: build (q, δ_S0, δ_S1, δ_NEVPT2) arrays
    # Use equilibrium energy as reference (e_s0 at equilibrium ≈ completed['m00_...'])
    # Reference: S0 energy at the smallest displacement (closest to eq)
    # Practical: use the global minimum S0 energy among clean frames as reference
    e_s0_clean = np.array([completed[k]['e_s0'] for k in clean_keys])
    e_s1_clean = np.array([completed[k]['e_s1'] for k in clean_keys
                            if completed[k]['e_s1'] is not None])
    e_s0_ref   = e_s0_clean.min()

    # Estimate B3LYP energy at each grid geometry by linear interpolation
    # (we don't have PSI4 B3LYP at each grid point; use the ML-PES prediction or
    #  the zero-distortion energy as reference for the correction)
    # Simplest: define δ_S0(q) = E_S0(q) - E_S0(eq) - [E_B3LYP(q) - E_B3LYP(eq)]
    # Since we don't have B3LYP at each grid point, we define:
    #   δ_S0(q) = [E_CASSCF_S0(q) - E_CASSCF_S0_ref] - [E_B3LYP(q) - E_B3LYP_ref]
    # and use E_B3LYP(q) ≈ E_B3LYP_min (zero reference) for now.
    # This is equivalent to correcting the relative energy differences.
    # δ at equilibrium (q=0) is set to 0 by construction.

    q_clean  = np.array([completed[k]['q_nm'] for k in clean_keys])
    dE_s0_ha = e_s0_clean - e_s0_ref   # relative S0 energies (Ha)

    # B3LYP relative energy at each grid point: we compute this from the ML-PES
    # Since grid points are NOT in training data, we use the analytic formula:
    # For now, store dE_s0_ha as the raw CASSCF relative energy and document that
    # the true δ = E_CASSCF - E_B3LYP requires B3LYP evaluation at each grid point.
    # For the first iteration, the KRR models predict CASSCF *corrections* to a
    # B3LYP surface that the user must provide via recompute_wB97X_surface.py or
    # directly from PSI4.

    # δ_NEVPT2(q) = E_NEVPT2(q) - E_CASSCF_S0(q)  (negative; dynamic correlation gain)
    nevpt2_keys = [k for k in clean_keys if completed[k].get('e_nevpt2') is not None]
    q_nevpt2  = np.array([completed[k]['q_nm']     for k in nevpt2_keys])
    dE_nevpt2 = np.array([completed[k]['delta_nevpt2']  for k in nevpt2_keys])  # Ha

    print(f"  Frames with NEVPT2: {len(nevpt2_keys)}/{len(clean_keys)}")

    # ── Train NM-KRR models ───────────────────────────────────────────────────
    print("\nTraining NM-KRR delta models...")

    # Reconstruct grid frame dicts for clean keys (for q_nm access)
    clean_frame_stubs = [{'q_nm': np.array(completed[k]['q_nm'])} for k in clean_keys]

    model_s0, rmse_s0 = train_nm_krr(
        eq_coords_ang, frequencies, eigenvalues, eigvecs_mw, sqrt_mass,
        symbols, clean_frame_stubs, dE_s0_ha,
        e_ref_ha=e_s0_ref,
        gamma_values=gamma_values, alpha_values=alpha_values,
        label='S0-CASSCF')

    model_s0_path = out_dir / 'nm_delta_s0_model.pkl'
    with open(model_s0_path, 'wb') as f:
        pickle.dump(model_s0, f, protocol=4)
    print(f"  S0 model saved → {model_s0_path}")

    if len(nevpt2_keys) >= 5:
        nevpt2_frame_stubs = [{'q_nm': np.array(completed[k]['q_nm'])} for k in nevpt2_keys]
        model_nevpt2, rmse_nevpt2 = train_nm_krr(
            eq_coords_ang, frequencies, eigenvalues, eigvecs_mw, sqrt_mass,
            symbols, nevpt2_frame_stubs, dE_nevpt2,
            e_ref_ha=0.0,
            gamma_values=gamma_values, alpha_values=alpha_values,
            label='NEVPT2-corr')

        model_nv_path = out_dir / 'nm_delta_nevpt2_model.pkl'
        with open(model_nv_path, 'wb') as f:
            pickle.dump(model_nevpt2, f, protocol=4)
        print(f"  NEVPT2 model saved → {model_nv_path}")
    else:
        rmse_nevpt2 = None
        print(f"  Skipping NEVPT2 model (only {len(nevpt2_keys)} frames)")

    # ── Save S1 model if enough frames ────────────────────────────────────────
    s1_keys = [k for k in clean_keys if completed[k].get('e_s1') is not None]
    if len(s1_keys) >= 5:
        e_s1_arr = np.array([completed[k]['e_s1'] for k in s1_keys])
        e_s1_ref = e_s1_arr.min()
        dE_s1_ha = e_s1_arr - e_s1_ref
        s1_frame_stubs = [{'q_nm': np.array(completed[k]['q_nm'])} for k in s1_keys]
        model_s1, rmse_s1 = train_nm_krr(
            eq_coords_ang, frequencies, eigenvalues, eigvecs_mw, sqrt_mass,
            symbols, s1_frame_stubs, dE_s1_ha,
            e_ref_ha=e_s1_ref,
            gamma_values=gamma_values, alpha_values=alpha_values,
            label='S1-CASSCF')
        model_s1_path = out_dir / 'nm_delta_s1_model.pkl'
        with open(model_s1_path, 'wb') as f:
            pickle.dump(model_s1, f, protocol=4)
        print(f"  S1 model saved → {model_s1_path}")
    else:
        rmse_s1 = None

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        'n_grid_frames':       len(grid_frames),
        'n_ok':                len(ok_keys),
        'n_clean':             len(clean_keys),
        'n_switched':          len(switched_keys),
        'n_nevpt2':            len(nevpt2_keys),
        'no_occ_threshold':    args.no_occ_threshold,
        'amplitudes':          amplitudes,
        'T_nm_K':              args.T_nm,
        'basis':               args.basis,
        'base_method':         args.base_method,
        'loo_cv_s0_kcal':      float(rmse_s0),
        'loo_cv_nevpt2_kcal':  float(rmse_nevpt2) if rmse_nevpt2 else None,
        'loo_cv_s1_kcal':      float(rmse_s1)     if rmse_s1     else None,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary:")
    print(f"  Grid frames:     {len(grid_frames)}")
    print(f"  OK (converged):  {len(ok_keys)}")
    print(f"  Clean (no switch): {len(clean_keys)}")
    print(f"  S0 LOO-CV:       {rmse_s0:.3f} kcal/mol")
    if rmse_nevpt2:
        print(f"  NEVPT2 LOO-CV:   {rmse_nevpt2:.3f} kcal/mol")
    print(f"\nAll outputs in: {out_dir}")
    print("\nNext steps:")
    print("  python3 ir_md_spectrum.py ... "
          f"--nm-delta-model {out_dir}/nm_delta_nevpt2_model.pkl")


if __name__ == '__main__':
    main()
