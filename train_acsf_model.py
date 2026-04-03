#!/usr/bin/env python3
"""
Train ML-PES using ACSF descriptors + KRR.

Loads the existing MVKO training dataset (same data as the Coulomb+KRR model),
replaces CoulombMatrixDescriptor with ACSFDescriptor, and retrains with a γ/α
grid search.  Results are saved to outputs/acsf_model_<ts>/ for direct
comparison with the Coulomb+KRR baseline.

The Coulomb+KRR model is NOT modified — it continues running the current IR
spectrum calculation.

Usage
-----
    python3 train_acsf_model.py \
        --training-data outputs/mvko_ch_retrain_20260403_101429/combined_training_data.npz \
        --de-cutoff 50 \
        --gamma-values 1e-5,3e-5,1e-4,3e-4,1e-3 \
        --alpha-values 1e-6,1e-5,1e-4 \
        --hessian outputs/casscf_nm_delta_20260401_110049/hessian_used.npy

The --de-cutoff flag (default 50 kcal/mol) limits training to near-equilibrium
frames for the initial comparison.  Re-run without the flag to include the
large-amplitude C-H stretch frames.
"""

import sys, argparse, pickle, json
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'modules'))

from data_formats import load_trajectory, TrajectoryData
from ml_pes import MLPESConfig, MLPESTrainer
from acsf_descriptor import ACSFDescriptor

HARTREE_TO_KCAL = 627.509474


class ACSFMLPESTrainer(MLPESTrainer):
    """
    MLPESTrainer subclass that uses ACSFDescriptor instead of CoulombMatrix.

    All other logic (StandardScaler, KRR, save/load) is identical to the
    parent class.  Overrides self.descriptor after super().__init__().
    """

    def __init__(self, config, acsf_descriptor):
        super().__init__(config)
        self.descriptor = acsf_descriptor   # replaces CoulombMatrixDescriptor


def gamma_alpha_grid_search(trainer_cls, acsf_desc, traj, gammas, alphas,
                             n_cv=5, random_seed=42):
    """
    Grid search over (γ, α) using k-fold CV RMSE.

    Returns list of (gamma, alpha, cv_rmse) sorted by cv_rmse ascending.
    """
    from sklearn.model_selection import KFold
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import StandardScaler

    print(f"\nComputing ACSF descriptors for {traj.n_frames} frames ...")
    X = acsf_desc.compute_batch(list(traj.symbols), traj.coordinates)
    y = traj.energies
    print(f"  Descriptor shape: {X.shape}")

    kf = KFold(n_splits=n_cv, shuffle=True, random_state=random_seed)
    results = []

    n_combos = len(gammas) * len(alphas)
    print(f"\nGrid search: {len(gammas)} γ × {len(alphas)} α = {n_combos} combos "
          f"({n_cv}-fold CV)")
    print(f"{'γ':>10}  {'α':>10}  {'CV RMSE (kcal/mol)':>20}")
    print("-" * 45)

    for gamma in gammas:
        for alpha in alphas:
            fold_errors = []
            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                sx = StandardScaler()
                sy = StandardScaler()
                X_tr_sc  = sx.fit_transform(X_tr)
                X_val_sc = sx.transform(X_val)
                y_tr_sc  = sy.fit_transform(y_tr.reshape(-1, 1)).flatten()

                m = KernelRidge(kernel='rbf', gamma=gamma, alpha=alpha)
                m.fit(X_tr_sc, y_tr_sc)
                y_pred_sc = m.predict(X_val_sc)
                y_pred = sy.inverse_transform(y_pred_sc.reshape(-1, 1)).flatten()
                err = (y_pred - y_val) * HARTREE_TO_KCAL
                fold_errors.append(np.sqrt((err**2).mean()))

            cv_rmse = np.mean(fold_errors)
            results.append((gamma, alpha, cv_rmse))
            print(f"{gamma:>10.1e}  {alpha:>10.1e}  {cv_rmse:>20.4f}")

    results.sort(key=lambda x: x[2])
    return results


def train_final_model(acsf_desc, traj, gamma, alpha, val_split=0.1, seed=42):
    """Train the final model on full dataset with best hyperparameters."""
    cfg = MLPESConfig(
        gamma=gamma,
        alpha=alpha,
        kernel='rbf',
        train_forces=False,
        tune_hyperparameters=False,
        validation_split=val_split,
        random_seed=seed,
        descriptor_type='acsf',
    )
    trainer = ACSFMLPESTrainer(cfg, acsf_desc)
    trainer.train(traj)
    return trainer


def check_hessian_frequencies(trainer, symbols, eq_coords, hessian_path=None):
    """
    Check Hessian NM frequencies by finite differences on the ML-PES.

    Uses the existing PSI4 Hessian geometry as the equilibrium point.
    Falls back to FD Hessian if analytic not available for ACSF.
    """
    from normal_modes import compute_normal_modes
    BOHR_TO_ANG = 0.52917721092

    print("\n--- Hessian frequency check (FD on ACSF ML-PES) ---")

    # Simple FD Hessian
    n = len(symbols)
    n3 = 3 * n
    delta = 0.005  # Å

    def energy(coords_flat):
        c = coords_flat.reshape(n, 3)
        return float(trainer.predict(symbols, c))

    coords_flat = eq_coords.flatten()
    e0 = energy(coords_flat)

    H = np.zeros((n3, n3))
    print(f"  Computing FD Hessian ({n3}×{n3}, δ={delta} Å) ...")
    for i in range(n3):
        for j in range(i, n3):
            cp = coords_flat.copy(); cp[i] += delta; cp[j] += delta
            cm = coords_flat.copy(); cm[i] -= delta; cm[j] -= delta
            cpm = coords_flat.copy(); cpm[i] += delta; cpm[j] -= delta
            cmp = coords_flat.copy(); cmp[i] -= delta; cmp[j] += delta
            H[i, j] = H[j, i] = (energy(cp) + energy(cm) - energy(cpm) - energy(cmp)) / (4 * delta**2)

    freqs, *_ = compute_normal_modes(symbols, H)

    print(f"\n  NM frequencies (ACSF ML-PES FD Hessian):")
    print(f"  {'Mode':>5}  {'freq (cm-1)':>12}")
    for i, f in enumerate(freqs):
        tag = '  <-- C-H (physical!)' if 2800 < f < 3500 else (
              '  <-- C-H (stiff)'     if f > 5000 else '')
        print(f"  {i:3d}  {f:12.1f}{tag}")

    return freqs


def main():
    parser = argparse.ArgumentParser(description='Train ACSF+KRR ML-PES for MVKO')
    parser.add_argument('--training-data', required=True,
                        help='Training data .npz (952-frame combined dataset)')
    parser.add_argument('--de-cutoff', type=float, default=50.0,
                        help='Max relative energy (kcal/mol) to include. '
                             'Use 50 for near-eq comparison, 500 for full range.')
    parser.add_argument('--gamma-values', default='1e-5,3e-5,1e-4,3e-4,1e-3',
                        help='Comma-separated γ values for grid search')
    parser.add_argument('--alpha-values', default='1e-6,1e-5,1e-4,1e-3',
                        help='Comma-separated α values for grid search')
    parser.add_argument('--n-cv', type=int, default=5,
                        help='Number of CV folds for grid search')
    parser.add_argument('--r-cut', type=float, default=6.0,
                        help='ACSF cutoff radius (Å)')
    parser.add_argument('--aggregate', default='sum_by_species',
                        choices=['concatenate', 'sum', 'sum_by_species'],
                        help='How to reduce per-atom ACSF vectors to a single descriptor. '
                             'sum_by_species (default): compact, permutation-invariant, '
                             'best KRR conditioning. concatenate: full info but too large.')
    parser.add_argument('--hessian', default=None,
                        help='PSI4 Hessian .npy for NM frequency check')
    parser.add_argument('--skip-grid-search', action='store_true',
                        help='Skip grid search, use best gamma/alpha directly')
    parser.add_argument('--gamma', type=float, default=None,
                        help='γ to use when --skip-grid-search')
    parser.add_argument('--alpha', type=float, default=None,
                        help='α to use when --skip-grid-search')
    args = parser.parse_args()

    gammas = [float(x) for x in args.gamma_values.split(',')]
    alphas = [float(x) for x in args.alpha_values.split(',')]

    # ── Load training data ────────────────────────────────────────────────────
    print(f"Loading training data: {args.training_data}")
    traj_full = load_trajectory(args.training_data)
    e_min = traj_full.energies.min()
    de_all = (traj_full.energies - e_min) * HARTREE_TO_KCAL

    if args.de_cutoff < max(de_all):
        mask = de_all < args.de_cutoff
        print(f"  Applying dE < {args.de_cutoff:.0f} kcal/mol filter: "
              f"{mask.sum()} / {traj_full.n_frames} frames kept")
        traj = TrajectoryData(
            symbols     = traj_full.symbols,
            coordinates = traj_full.coordinates[mask],
            energies    = traj_full.energies[mask],
            forces      = traj_full.forces[mask] if traj_full.forces is not None else None,
            dipoles     = None,
        )
    else:
        traj = traj_full
        print(f"  Using all {traj.n_frames} frames (dE up to {de_all.max():.0f} kcal/mol)")

    symbols = list(traj.symbols)
    eq_coords = traj.coordinates[np.argmin(traj.energies)]

    # ── Build ACSF descriptor ─────────────────────────────────────────────────
    print(f"\nBuilding ACSF descriptor (R_cut={args.r_cut} Å, aggregate={args.aggregate}) ...")
    acsf_desc = ACSFDescriptor(
        species   = sorted(set(symbols)),
        r_cut     = args.r_cut,
        aggregate = args.aggregate,
    )
    print(f"  {acsf_desc.describe()}")
    # Compute actual output size for one frame
    test_vec = acsf_desc.compute(symbols, eq_coords)
    print(f"  Full descriptor dimension: {len(test_vec)} "
          f"({len(symbols)} atoms × {acsf_desc.n_features_per_atom} features/atom)")

    # ── Grid search ───────────────────────────────────────────────────────────
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path('outputs') / f'acsf_model_{ts}'
    out.mkdir(parents=True, exist_ok=True)

    if args.skip_grid_search:
        best_gamma = args.gamma
        best_alpha = args.alpha
        grid_results = []
        print(f"\nSkipping grid search — using γ={best_gamma}, α={best_alpha}")
    else:
        grid_results = gamma_alpha_grid_search(
            ACSFMLPESTrainer, acsf_desc, traj, gammas, alphas,
            n_cv=args.n_cv,
        )
        best_gamma, best_alpha, best_cv = grid_results[0]
        print(f"\nBest: γ={best_gamma:.1e}, α={best_alpha:.1e}, "
              f"CV RMSE={best_cv:.4f} kcal/mol")

        # Save grid search results
        gs_path = out / 'grid_search_results.json'
        with open(gs_path, 'w') as f:
            json.dump([{'gamma': g, 'alpha': a, 'cv_rmse': r}
                       for g, a, r in grid_results], f, indent=2)
        print(f"Grid search results saved: {gs_path}")

    # ── Train final model ─────────────────────────────────────────────────────
    print(f"\nTraining final model (γ={best_gamma}, α={best_alpha}) ...")
    trainer = train_final_model(acsf_desc, traj, best_gamma, best_alpha)

    rmse = trainer.training_history.get('rmse_kcal', float('nan'))
    mae  = trainer.training_history.get('mae_kcal',  float('nan'))
    print(f"  Validation RMSE : {rmse:.4f} kcal/mol")
    print(f"  Validation MAE  : {mae:.4f} kcal/mol")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = out / 'mlpes_acsf.pkl'
    trainer.save(str(model_path))
    print(f"  Model saved: {model_path}")

    # Save metadata
    meta = {
        'descriptor': 'ACSF',
        'species': acsf_desc.species,
        'r_cut': acsf_desc.r_cut,
        'n_g2': len(acsf_desc.g2_params),
        'n_g4': len(acsf_desc.g4_params),
        'features_per_atom': acsf_desc.n_features_per_atom,
        'n_atoms': len(symbols),
        'total_features': len(test_vec),
        'n_training_frames': traj.n_frames,
        'de_cutoff_kcal': args.de_cutoff,
        'best_gamma': best_gamma,
        'best_alpha': best_alpha,
        'val_rmse_kcal': rmse,
        'val_mae_kcal': mae,
        'grid_results': [{'gamma': g, 'alpha': a, 'cv_rmse': r}
                         for g, a, r in grid_results],
    }
    with open(out / 'model_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # ── Hessian frequency check ───────────────────────────────────────────────
    if args.hessian:
        freqs = check_hessian_frequencies(trainer, symbols, eq_coords, args.hessian)
        np.save(out / 'hessian_frequencies.npy', freqs)
    else:
        print("\n(No --hessian provided; skipping frequency check)")
        print("  To check frequencies later:")
        print(f"  python3 train_acsf_model.py --training-data {args.training_data} "
              f"--skip-grid-search --gamma {best_gamma} --alpha {best_alpha} "
              f"--hessian outputs/casscf_nm_delta_20260401_110049/hessian_used.npy")

    # ── Comparison summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ACSF+KRR model summary")
    print(f"{'='*60}")
    print(f"  Output dir       : {out}")
    print(f"  Descriptor       : {acsf_desc.describe()}")
    print(f"  Training frames  : {traj.n_frames} (dE < {args.de_cutoff:.0f} kcal/mol)")
    print(f"  Best γ / α       : {best_gamma:.1e} / {best_alpha:.1e}")
    print(f"  Validation RMSE  : {rmse:.4f} kcal/mol")
    print(f"\n  Coulomb+KRR baseline (904 frames, γ=0.001, α=1e-5):")
    print(f"    Validation RMSE  : 0.2734 kcal/mol")
    print(f"    C-H Hessian modes: 9825–15005 cm-1 (stiff artifact)")
    print(f"\n  To run IR spectrum with ACSF model:")
    print(f"    python3 ir_md_spectrum.py \\")
    print(f"      --model {model_path} \\")
    print(f"      --training-data outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \\")
    print(f"      --steps 30000 --temp 300 --preminimize --zpe-min-freq 50 --zpe-max-freq 4000 \\")
    print(f"      --n-trajectories 5 --max-bond-extension 2.5")


if __name__ == '__main__':
    main()
