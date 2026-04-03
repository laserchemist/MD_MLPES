#!/usr/bin/env python3
"""
Combine existing 904-frame MVKO training data with new C-H stretch NM
displacement frames (filtered to dE < 500 kcal/mol) and retrain ML-PES.

Usage:
    python3 retrain_with_ch_stretch.py \
        --training-data outputs/mvko_20260319_081314/combined_training_data.npz \
        --ch-stretch    outputs/nm_ch_stretch_20260402_225350/nm_displacements.npz \
        --de-cutoff     500 \
        --gamma 0.001 --alpha 1e-5
"""

import sys, argparse, pickle
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'modules'))

from data_formats import load_trajectory, save_trajectory, TrajectoryData
from ml_pes import MLPESConfig, MLPESTrainer

HARTREE_TO_KCAL = 627.509474


def main():
    parser = argparse.ArgumentParser(description='Retrain MVKO ML-PES with C-H stretch frames')
    parser.add_argument('--training-data', required=True,
                        help='Existing 904-frame training data (.npz)')
    parser.add_argument('--ch-stretch', required=True,
                        help='New C-H stretch NM displacements (.npz)')
    parser.add_argument('--de-cutoff', type=float, default=500.0,
                        help='Max relative energy (kcal/mol) to include from CH-stretch set')
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=1e-5)
    parser.add_argument('--hessian', default=None,
                        help='Pre-computed Hessian .npy for NM frequency check (optional)')
    args = parser.parse_args()

    # ── Load existing training data ───────────────────────────────────────────
    print(f"Loading existing training data: {args.training_data}")
    base = load_trajectory(args.training_data)
    n_base = base.n_frames
    e_min_base = base.energies.min()
    print(f"  {n_base} frames, E_min = {e_min_base:.6f} Ha")

    # ── Load C-H stretch data ─────────────────────────────────────────────────
    print(f"\nLoading C-H stretch frames: {args.ch_stretch}")
    ch = load_trajectory(args.ch_stretch)
    n_ch_total = ch.n_frames
    print(f"  {n_ch_total} frames before filtering")

    # Filter by relative energy (using the global minimum from the base set)
    e_min = min(e_min_base, ch.energies.min())
    de_ch = (ch.energies - e_min) * HARTREE_TO_KCAL
    mask = de_ch < args.de_cutoff
    n_ch_kept = mask.sum()
    print(f"  dE < {args.de_cutoff:.0f} kcal/mol: {n_ch_kept} / {n_ch_total} kept")
    print(f"  Max dE kept: {de_ch[mask].max():.1f} kcal/mol")

    if n_ch_kept == 0:
        print("ERROR: no CH-stretch frames survive the energy filter."); sys.exit(1)

    # ── Summarise C-H bond coverage of new frames ─────────────────────────────
    symbols  = list(base.symbols)
    coords_k = ch.coordinates[mask]
    c_idx = [i for i, s in enumerate(symbols) if s == 'C']
    h_idx = [i for i, s in enumerate(symbols) if s == 'H']
    ch_dists = []
    for frame in coords_k:
        for hi in h_idx:
            nn = min(np.linalg.norm(frame[hi] - frame[ci]) for ci in c_idx)
            ch_dists.append(nn)
    ch_arr = np.array(ch_dists)
    print(f"\n  C-H bond coverage (kept frames):")
    print(f"    normal  < 1.3 Ang: {(ch_arr < 1.3).sum()}")
    print(f"    stretch 1.3-2.0  : {((ch_arr >= 1.3) & (ch_arr < 2.0)).sum()}")
    print(f"    dissoc  > 2.0    : {(ch_arr >= 2.0).sum()}")
    print(f"    max C-H          : {ch_arr.max():.2f} Ang")

    # ── Concatenate ───────────────────────────────────────────────────────────
    print(f"\nCombining {n_base} base + {n_ch_kept} CH-stretch = "
          f"{n_base + n_ch_kept} total frames")

    combined_coords   = np.concatenate([base.coordinates,  ch.coordinates[mask]],  axis=0)
    combined_energies = np.concatenate([base.energies,     ch.energies[mask]],     axis=0)

    # Forces: base set may have forces, CH set should too
    if base.forces is not None and ch.forces is not None:
        combined_forces = np.concatenate([base.forces, ch.forces[mask]], axis=0)
    else:
        combined_forces = None

    # Dipoles: base has none (combined_training_data.npz has no dipoles)
    #          keep as None — dipole surface trained separately
    combined_dipoles = None

    combined = TrajectoryData(
        symbols     = symbols,
        coordinates = combined_coords,
        energies    = combined_energies,
        forces      = combined_forces,
        dipoles     = combined_dipoles,
        metadata    = {
            'source':        'mvko_base_plus_ch_stretch',
            'n_base':        n_base,
            'n_ch_stretch':  int(n_ch_kept),
            'de_cutoff_kcal': args.de_cutoff,
            'gamma':         args.gamma,
            'alpha':         args.alpha,
        },
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path('outputs') / f'mvko_ch_retrain_{ts}'
    out.mkdir(parents=True, exist_ok=True)

    # Save combined dataset for reference
    combined_path = out / 'combined_training_data.npz'
    save_trajectory(combined, str(combined_path))
    print(f"Combined dataset saved: {combined_path}")

    print(f"\nTraining ML-PES (γ={args.gamma}, α={args.alpha}) ...")
    cfg = MLPESConfig(
        gamma=args.gamma,
        alpha=args.alpha,
        kernel='rbf',
        train_forces=False,
        tune_hyperparameters=False,
        validation_split=0.1,
        random_seed=42,
    )
    trainer = MLPESTrainer(cfg)
    trainer.train(combined)

    rmse = trainer.training_history.get('rmse_kcal', float('nan'))
    mae  = trainer.training_history.get('mae_kcal',  float('nan'))
    print(f"\nValidation RMSE : {rmse:.4f} kcal/mol")
    print(f"Validation MAE  : {mae:.4f} kcal/mol")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = out / 'mlpes_ch_retrained.pkl'
    trainer.save(str(model_path))
    print(f"Model saved: {model_path}")

    # ── Optional: NM frequency check ─────────────────────────────────────────
    if args.hessian:
        print(f"\n--- NM frequency check (analytic Hessian) ---")
        try:
            from normal_modes import compute_normal_modes
            from bakken import MLPESDriver

            hessian = np.load(args.hessian)
            eq_coords = base.coordinates[np.argmin(base.energies)]

            driver = MLPESDriver(trainer, symbols, eq_coords)
            freq_ml, _, _, _ = compute_normal_modes(symbols, driver.analytic_hessian(eq_coords))
            print(f"{'Mode':>5}  {'freq (cm-1)':>12}")
            for i, f in enumerate(freq_ml):
                tag = '  <-- C-H' if f > 2500 else ''
                print(f"  {i:3d}  {f:12.1f}{tag}")
        except Exception as e:
            print(f"  Could not compute NM frequencies: {e}")

    print(f"\nOutput directory: {out}")
    print(f"\nTo run IR spectrum with retrained model:")
    print(f"  python3 ir_md_spectrum.py \\")
    print(f"    --model {model_path} \\")
    print(f"    --training-data outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \\")
    print(f"    --steps 30000 --temp 300 --preminimize --zpe-min-freq 50 --zpe-max-freq 4000 \\")
    print(f"    --n-trajectories 5 --max-bond-extension 2.5")


if __name__ == '__main__':
    main()
