#!/usr/bin/env python3
"""
train_wB97X_nm_model.py — Train the wB97X ML-PES using normal-mode coordinates.

This replaces the Coulomb+RBF kernel (which produces imaginary modes and C-H
elongation artifacts) with a physically motivated NM-coordinate KRR.

In NM-coordinate space q = U_vib^T · M^{1/2} · (R − R_eq):
  - q = 0 exactly at the reference minimum → Hessian well-defined at origin
  - ||q||² grows monotonically with distortion → no descriptor clustering
  - RBF kernel is localised in physical displacement space
  - Analytic Hessian H_q is positive-definite near equilibrium for
    symmetric training data (±displacements along each mode)

Usage
-----
    python3 train_wB97X_nm_model.py \\
        --training-data outputs/wB97X_surface_20260406_223155/training_data_wB97X.npz \\
        --eq-coords     outputs/mvko_20260319_081314/psi4_eq_coords.npy \\
        --hessian       outputs/casscf_nm_delta_20260401_110049/hessian_used.npy \\
        --out-dir       outputs/wB97X_surface_20260406_223155

Outputs (in --out-dir):
    mlpes_wB97X_nm.pkl   — NMKRRPESModel for NMPESDriver
    nm_pes_diagnostics.png

Energy filter: only frames with ΔE_wB97X < --max-de kcal/mol are used
(default 50). This avoids the high-energy contamination pattern that
corrupts near-equilibrium Hessians.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'modules'))

from data_formats import load_trajectory
from modules.nm_pes import NMKRRPESModel

KCAL        = 627.509474
ANG2BOHR    = 1.88972612463
FREQ_CONV   = 5140.48   # cm⁻¹ / sqrt(Ha/(Bohr²·amu))
KB_HA_PER_K = 3.1668114e-6

# ── Atomic masses ─────────────────────────────────────────────────────────────
from modules.normal_modes import ATOMIC_MASSES


def loo_cv_rmse(X_q: np.ndarray, y: np.ndarray,
                gamma: float, alpha: float) -> float:
    """Leave-one-out cross-validation RMSE (kcal/mol)."""
    M = len(y)
    y_mean = float(np.mean(y))
    yc = y - y_mean
    A2 = np.sum(X_q ** 2, axis=1, keepdims=True)
    K  = np.exp(-gamma * (A2 + A2.T - 2.0 * X_q @ X_q.T))
    K_reg = K.copy()
    K_reg[np.diag_indices_from(K_reg)] += alpha
    beta = np.linalg.solve(K_reg, yc)            # dual coefficients
    preds = K @ beta + y_mean                    # in-sample predictions
    # LOO via hat matrix diagonal: LOO_i = (y_i - pred_i) / (1 - H_ii)
    # H = K K_reg^{-1}, H_ii = K[i] · K_reg^{-1}[i]
    K_reg_inv = np.linalg.inv(K_reg)
    H_diag = np.einsum('ij,ji->i', K, K_reg_inv)
    residuals = (y - preds) / np.maximum(1.0 - H_diag, 1e-10)
    return float(np.sqrt(np.mean(residuals ** 2))) * KCAL


def compute_nm_modes(symbols, eq_coords_ang, H_cart_ha_bohr2):
    """
    Compute NM eigenvectors from the Cartesian Hessian.

    Returns:
        freqs    : (n_vib,) cm⁻¹ (positive vibrational modes)
        U_vib    : (3N, n_vib) mass-weighted NM eigenvectors
        sqrt_mass: (3N,)
        eigenvalues: (n_vib,) Ha/(Bohr²·amu)
    """
    masses_amu = np.array([ATOMIC_MASSES[s] for s in symbols])
    sqrt_mass  = np.repeat(np.sqrt(masses_amu), 3)            # (3N,)
    H_mw = H_cart_ha_bohr2 / np.outer(sqrt_mass, sqrt_mass)  # mass-weighted
    evals_all, evecs_all = np.linalg.eigh(H_mw)

    n_vib = len(symbols) * 3 - 6
    pos_idx = np.where(evals_all > 0)[0]
    if len(pos_idx) >= n_vib:
        sort_idx = pos_idx[-n_vib:]
    else:
        sort_idx = np.argsort(np.abs(evals_all))[-n_vib:]
        sort_idx = sort_idx[np.argsort(evals_all[sort_idx])]

    eigenvalues = evals_all[sort_idx]
    U_vib       = evecs_all[:, sort_idx]
    freqs       = FREQ_CONV * np.sqrt(np.abs(eigenvalues)) * np.sign(eigenvalues)
    return freqs, U_vib, sqrt_mass, eigenvalues


def main():
    parser = argparse.ArgumentParser(
        description='Train wB97X ML-PES using NM-coordinate KRR')
    parser.add_argument('--training-data', required=True,
                        help='wB97X training .npz (from train_wB97X_model.py)')
    parser.add_argument('--eq-coords',     required=False, default=None,
                        help='(N,3) equilibrium geometry .npy, Angstrom. '
                             'If omitted, the minimum-energy training frame is used '
                             '(recommended for wB97X training data).')
    parser.add_argument('--hessian',       required=True,
                        help='(3N,3N) Cartesian Hessian .npy, Ha/Bohr²')
    parser.add_argument('--out-dir',       required=True,
                        help='Output directory (will be created if needed)')
    parser.add_argument('--max-de',        type=float, default=50.0,
                        help='Only include frames with ΔE_wB97X < this (kcal/mol). '
                             'Default 50. Prevents high-energy contamination.')
    parser.add_argument('--gamma-values',  default='0.01,0.05,0.1,0.2,0.5',
                        help='Comma-separated γ values for LOO-CV sweep')
    parser.add_argument('--alpha-values',  default='1e-6,1e-5,1e-4',
                        help='Comma-separated α values for LOO-CV sweep')
    parser.add_argument('--gamma',         type=float, default=None,
                        help='Fixed γ (skip LOO-CV sweep)')
    parser.add_argument('--alpha',         type=float, default=None,
                        help='Fixed α (skip LOO-CV sweep)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load training data ────────────────────────────────────────────────────
    print(f'Loading training data: {args.training_data}')
    traj    = load_trajectory(args.training_data)
    symbols = list(traj.symbols)
    coords  = np.array(traj.coordinates)    # (M, N, 3) Ang
    e_ha    = np.array(traj.energies)       # (M,) Ha
    print(f'  {len(coords)} frames, {len(symbols)} atoms')

    # Energy filter
    dE_kcal = (e_ha - e_ha.min()) * KCAL
    mask    = dE_kcal < args.max_de
    n_all   = len(coords)
    coords  = coords[mask]
    e_ha    = e_ha[mask]
    dE_kcal = dE_kcal[mask]
    print(f'  After ΔE < {args.max_de} kcal/mol filter: {len(coords)} / {n_all} frames')
    print(f'  ΔE range: {float(dE_kcal.min()):.2f} – {float(dE_kcal.max()):.2f} kcal/mol')

    # ── Load eq geometry + Hessian ────────────────────────────────────────────
    if args.eq_coords:
        print(f'\nLoading eq geometry: {args.eq_coords}')
        eq_coords = np.load(args.eq_coords)    # (N, 3) Ang
    else:
        # Use minimum-energy training frame as reference
        # (avoids imaginary modes when B3LYP and wB97X minima differ)
        min_idx   = int(np.argmin(e_ha))
        eq_coords = coords[min_idx].copy()
        print(f'\nUsing minimum-energy training frame (idx={min_idx}, '
              f'ΔE={float(dE_kcal[min_idx]):.3f} kcal/mol) as reference geometry.')
    print(f'Loading Hessian: {args.hessian}')
    H_cart    = np.load(args.hessian)      # (3N, 3N) Ha/Bohr²

    # ── Compute NM eigenvectors ───────────────────────────────────────────────
    print('\nComputing NM eigenvectors ...')
    freqs, U_vib, sqrt_mass, eigenvalues = compute_nm_modes(symbols, eq_coords, H_cart)
    n_vib = len(freqs)
    print(f'  {n_vib} vibrational modes')
    print(f'  Lowest 6  : {[int(f) for f in freqs[:6]]} cm⁻¹')
    print(f'  C-H range : {[int(f) for f in freqs if f > 2500]} cm⁻¹')

    # ── Project training frames to NM coordinates ─────────────────────────────
    print('\nProjecting training data to NM coordinates ...')
    # q[i] = U_vib^T · M^{1/2} · (R[i] - R_eq) in sqrt(amu)·Bohr
    delta_ang  = coords - eq_coords[None, :, :]            # (M, N, 3) Ang
    delta_bohr = delta_ang.reshape(len(coords), -1) * ANG2BOHR  # (M, 3N) Bohr
    delta_mw   = delta_bohr * sqrt_mass[None, :]           # (M, 3N)
    X_q        = delta_mw @ U_vib                          # (M, n_vib)
    print(f'  X_q shape: {X_q.shape}')
    q_norms    = np.linalg.norm(X_q, axis=1)
    print(f'  ||q|| range: {float(q_norms.min()):.3f} – {float(q_norms.max()):.3f} sqrt(amu)·Bohr')

    # Check thermal amplitude coverage
    KB_HA   = KB_HA_PER_K
    T_ref   = 300.0
    a_therm = np.sqrt(2.0 * T_ref * KB_HA / eigenvalues)  # (n_vib,) sqrt(amu)·Bohr
    print(f'  Thermal amplitude 300K: {float(a_therm.min()):.3f} – {float(a_therm.max()):.3f}')

    # ── Coordinate scaling: normalise to thermal amplitude ────────────────────
    # q_s = q / a_therm(300K)  →  all modes lie in ~[-2,+2] units of a_therm.
    # This makes the RBF width γ physically consistent for all modes: the kernel
    # decays when modes differ by 1/sqrt(γ) thermal amplitudes regardless of
    # frequency.  Without scaling, a single γ fits soft torsions OR stiff C-H
    # modes but not both (27× amplitude range).
    coord_scale = a_therm.copy()          # (n_vib,) sqrt(amu)·Bohr
    X_qs        = X_q / coord_scale       # (M, n_vib) dimensionless
    print(f'  Coord scale range: {coord_scale.min():.4f} – {coord_scale.max():.4f} sqrt(amu)·Bohr')
    print(f'  X_qs range (scaled): {float(X_qs.min()):.3f} – {float(X_qs.max()):.3f}')

    # ── Hyperparameter selection (in scaled q_s space) ────────────────────────
    if args.gamma is not None and args.alpha is not None:
        best_gamma = args.gamma
        best_alpha = args.alpha
        best_rmse  = loo_cv_rmse(X_qs, e_ha, best_gamma, best_alpha)
        print(f'\nUsing fixed γ={best_gamma}, α={best_alpha}')
        print(f'  LOO-CV RMSE (scaled): {best_rmse:.4f} kcal/mol')
    else:
        gammas = [float(g) for g in args.gamma_values.split(',')]
        alphas = [float(a) for a in args.alpha_values.split(',')]
        print(f'\nLOO-CV sweep (scaled coords): {len(gammas)} γ × {len(alphas)} α '
              f'= {len(gammas)*len(alphas)} combinations ...')

        best_rmse  = np.inf
        best_gamma = gammas[0]
        best_alpha = alphas[0]
        results    = []
        for gamma in gammas:
            for alpha in alphas:
                rmse = loo_cv_rmse(X_qs, e_ha, gamma, alpha)
                results.append((gamma, alpha, rmse))
                status = ' ← best' if rmse < best_rmse else ''
                print(f'  γ={gamma:.4g}  α={alpha:.1e}  LOO-CV={rmse:.4f} kcal/mol{status}')
                if rmse < best_rmse:
                    best_rmse  = rmse
                    best_gamma = gamma
                    best_alpha = alpha

        print(f'\nBest: γ={best_gamma}  α={best_alpha}  LOO-CV={best_rmse:.4f} kcal/mol')

    # ── Train final model ─────────────────────────────────────────────────────
    print(f'\nTraining final NMKRRPESModel on {len(X_q)} frames ...')
    model = NMKRRPESModel(
        eq_coords_ang = eq_coords,
        U_vib         = U_vib,
        sqrt_mass     = sqrt_mass,
        freqs_vib     = freqs,
        symbols       = symbols,
        gamma         = best_gamma,
        alpha_reg     = best_alpha,
        X_train_q     = X_q,
        y_train_ha    = e_ha,
        cv_rmse_kcal  = best_rmse,
        coord_scale   = coord_scale,
    )

    # In-sample RMSE (evaluate without wall to get clean KRR fit quality)
    y_pred = np.array([model.predict_ha(X_q[i]) for i in range(len(X_q))])
    insample_rmse = float(np.sqrt(np.mean((y_pred - e_ha) ** 2))) * KCAL
    print(f'  In-sample RMSE: {insample_rmse:.4f} kcal/mol')

    # ── NM frequency check via analytic Hessian ───────────────────────────────
    print('\nChecking NM frequencies via analytic Hessian at equilibrium ...')
    from modules.nm_pes import NMPESDriver, FREQ_CONV as FCONV
    model_path_tmp = str(out_dir / '_tmp_nm_check.pkl')
    model.save(model_path_tmp)
    driver = NMPESDriver(model_path_tmp)
    freqs_ml = driver.nm_frequencies(eq_coords)

    n_imag = int((freqs_ml < 0).sum())
    print(f'  Imaginary modes: {n_imag}')
    print(f'  Lowest 10: {[int(f) for f in sorted(freqs_ml)[:10]]} cm⁻¹')
    print(f'  C-H region (>2500): {[int(f) for f in freqs_ml if f > 2500]} cm⁻¹')
    import os; os.remove(model_path_tmp)

    # ── Save ──────────────────────────────────────────────────────────────────
    model_path = out_dir / 'mlpes_wB97X_nm.pkl'
    model.save(str(model_path))
    print(f'\nModel saved → {model_path}')

    # Save summary JSON
    summary = {
        'timestamp':    datetime.now().isoformat(),
        'n_frames':     int(len(X_q)),
        'n_atoms':      len(symbols),
        'n_vib':        int(n_vib),
        'gamma':        best_gamma,
        'alpha':        best_alpha,
        'loo_cv_rmse_kcal': float(best_rmse),
        'insample_rmse_kcal': float(insample_rmse),
        'n_imag_modes': n_imag,
        'freq_lowest_10': [float(f) for f in sorted(freqs_ml)[:10]],
        'freq_ch_region': [float(f) for f in freqs_ml if f > 2500],
        'model_path':   str(model_path),
    }
    with open(out_dir / 'nm_pes_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Diagnostics plot ─────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # (1) Parity plot: predicted vs actual energy
        ax = axes[0]
        dE_pred = (y_pred - y_pred.min()) * KCAL
        dE_act  = (e_ha   - e_ha.min())   * KCAL
        ax.scatter(dE_act, dE_pred, s=8, alpha=0.6, c=dE_act, cmap='viridis')
        lim = max(dE_act.max(), dE_pred.max())
        ax.plot([0, lim], [0, lim], 'r--', lw=1)
        ax.set_xlabel('DFT ΔE (kcal/mol)')
        ax.set_ylabel('ML ΔE (kcal/mol)')
        ax.set_title(f'NM-KRR parity  RMSE={insample_rmse:.3f} kcal/mol')

        # (2) NM frequencies: ML-KRR vs Hessian
        ax = axes[1]
        n_show = min(36, len(freqs))
        ax.bar(range(1, n_show+1), freqs[:n_show], color='grey', alpha=0.5,
               label='Hessian (input)')
        ax.bar(range(1, len(freqs_ml)+1), freqs_ml, color='blue', alpha=0.4,
               label='ML analytic Hessian')
        ax.axhline(0, color='red', lw=0.8)
        ax.set_xlabel('Mode index')
        ax.set_ylabel('Frequency (cm⁻¹)')
        ax.set_title(f'NM frequencies  n_imag={n_imag}')
        ax.legend(fontsize=7)

        # (3) ||q|| distribution
        ax = axes[2]
        ax.hist(q_norms, bins=40, color='steelblue', alpha=0.7)
        ax.set_xlabel('||q|| (sqrt(amu)·Bohr)')
        ax.set_ylabel('Count')
        ax.set_title(f'NM displacement distribution  (n={len(X_q)})')

        plt.tight_layout()
        plt.savefig(out_dir / 'nm_pes_diagnostics.png', dpi=120)
        print(f'Diagnostics → {out_dir}/nm_pes_diagnostics.png')
        plt.close()
    except Exception as exc:
        print(f'  [plot skipped: {exc}]')

    print(f'\nNext step — run IR spectrum with NM-PES model:')
    print(f'  python3 ir_md_spectrum.py \\')
    print(f'    --nm-pes-model {model_path} \\')
    print(f'    --training-data outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \\')
    print(f'    --nm-delta-model outputs/casscf_wB97X_nm_grid_20260407_184904/nm_delta_s0_model.pkl \\')
    print(f'    --steps 30000 --temp 300 --preminimize --zpe-min-freq 50 --zpe-max-freq 4000 \\')
    print(f'    --n-trajectories 5 --max-bond-extension 2.0 \\')
    print(f'    --output-dir outputs/ir_spectrum_wB97X_nm_300K')


if __name__ == '__main__':
    main()
