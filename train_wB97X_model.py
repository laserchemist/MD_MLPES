#!/usr/bin/env python3
"""
train_wB97X_model.py — Assemble training_data_wB97X.npz and train ML-PES
from an already-completed recompute_wB97X_surface.py results.json.

No PSI4 required — all single-points are already done.

Usage
-----
    python3 train_wB97X_model.py \
        --results  outputs/wB97X_surface_20260406_223155 \
        --training-data outputs/mvko_20260319_081314/combined_training_data.npz \
        --gamma 0.001 --alpha 1e-5
"""

import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'modules'))

from data_formats import TrajectoryData, load_trajectory, save_trajectory
from ml_pes import MLPESTrainer, MLPESConfig

HARTREE_TO_KCAL = 627.509474


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results',       required=True,
                        help='Directory with results.json from recompute_wB97X_surface.py')
    parser.add_argument('--training-data', required=True,
                        help='Original B3LYP training .npz (for coordinates)')
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=1e-5)
    args = parser.parse_args()

    out_dir = Path(args.results)

    # ── Load computed wB97X results ────────────────────────────────────────────
    print(f'Loading {out_dir}/results.json ...')
    with open(out_dir / 'results.json') as f:
        raw = json.load(f)
    completed = {int(k): v for k, v in raw.items()}
    ok_idx = sorted(i for i, r in completed.items() if r.get('status') == 'ok')
    print(f'  OK frames: {len(ok_idx)} / {len(completed)}')

    # ── Load original training data (coordinates) ──────────────────────────────
    print(f'Loading training data: {args.training_data} ...')
    traj = load_trajectory(args.training_data)
    symbols = list(traj.symbols)
    print(f'  {len(traj.coordinates)} frames, {len(symbols)} atoms')

    # ── Assemble wB97X TrajectoryData ──────────────────────────────────────────
    wb97x_e  = np.array([completed[i]['energy_ha']  for i in ok_idx])
    forces   = np.array([completed[i]['forces']      for i in ok_idx])
    dipoles  = np.array([completed[i]['dipole_D']    for i in ok_idx])
    coords   = traj.coordinates[ok_idx]

    dE = (wb97x_e - wb97x_e.min()) * HARTREE_TO_KCAL
    print(f'\nwB97X-D energy stats (relative, kcal/mol):')
    print(f'  min={dE.min():.3f}  max={dE.max():.3f}  mean={dE.mean():.3f}')

    traj_wb = TrajectoryData(
        symbols     = symbols,
        coordinates = coords,
        energies    = wb97x_e,
        forces      = forces,
        dipoles     = dipoles,
        metadata    = json.dumps({
            'method': 'wb97x-d', 'basis': '6-31G*',
            'source': str(args.training_data),
            'n_frames': len(ok_idx),
        }),
    )
    npz_path = out_dir / 'training_data_wB97X.npz'
    save_trajectory(traj_wb, npz_path)
    print(f'Saved {len(ok_idx)} frames → {npz_path}')

    # ── Train KRR ─────────────────────────────────────────────────────────────
    print(f'\nTraining KRR (γ={args.gamma}, α={args.alpha}) on {len(ok_idx)} frames ...')
    config = MLPESConfig(gamma=args.gamma, alpha=args.alpha, train_forces=False,
                         tune_hyperparameters=False)
    trainer = MLPESTrainer(config)
    trainer.train(traj_wb)
    # Extract best RMSE from trainer internals (already in kcal/mol)
    rmse = trainer.best_params.get('best_rmse_kcal',
           trainer.best_params.get('rmse_kcal', float('nan'))) \
           if hasattr(trainer, 'best_params') else float('nan')
    print(f'  Train RMSE: {rmse:.4f} kcal/mol')

    model_path = out_dir / 'mlpes_wB97X.pkl'
    trainer.save(str(model_path))
    print(f'  Model saved → {model_path}')

    # ── Quick NM frequency check ───────────────────────────────────────────────
    try:
        from modules.bakken import MLPESDriver
        from modules.normal_modes import compute_normal_modes
        eq_coords = coords[int(wb97x_e.argmin())]
        driver = MLPESDriver(trainer, symbols)
        hess = driver.analytic_hessian(eq_coords)
        freqs, *_ = compute_normal_modes(symbols, hess)
        n_imag = int((freqs < 0).sum())
        print(f'\nAnalytic Hessian NM frequencies:')
        print(f'  Imaginary modes: {n_imag}')
        print(f'  Lowest 6: {[int(f) for f in sorted(freqs)[:6]]} cm⁻¹')
        print(f'  C-H cluster: {[int(f) for f in freqs if f > 2500]} cm⁻¹')
    except Exception as exc:
        print(f'  NM check skipped: {exc}')

    print(f'\nDone. Next:')
    print(f'  python3 ir_md_spectrum.py \\')
    print(f'    --model {model_path} \\')
    print(f'    --training-data {npz_path} \\')
    print(f'    --nm-delta-model outputs/casscf_wB97X_nm_grid_20260407_184904/nm_delta_s0_model.pkl \\')
    print(f'    --steps 30000 --temp 300 --preminimize --zpe-min-freq 50 --zpe-max-freq 4000 \\')
    print(f'    --n-trajectories 5 --max-bond-extension 2.5')


if __name__ == '__main__':
    main()
