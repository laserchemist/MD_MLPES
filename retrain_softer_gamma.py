#!/usr/bin/env python3
"""
Retrain ML-PES with softer (smaller) gamma values to obtain physical
normal-mode frequencies from the numerical Hessian.

The current best model (gamma=0.01) produces unphysical Hessian curvature:
lowest mode 1846 cm⁻¹ (expected ~780 cm⁻¹ for CH₂OO). This script sweeps
gamma from 0.0001 upward, trains a model for each, computes the numerical
Hessian at the ML-PES minimum, and reports the resulting frequencies.

Usage:
    python3 retrain_softer_gamma.py \
        --training-data outputs/clean_psi410_20260308_203552/training_data.npz

Output:
    outputs/softer_gamma_TIMESTAMP/
        mlpes_gamma_<value>.pkl   — one model per gamma
        hessian_report.txt        — frequency table for all models
        best_model.pkl            — symlink to recommended model
"""

import sys
import os
import argparse
import pickle
import datetime
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'modules'))

from data_formats import load_trajectory
from ml_pes import MLPESTrainer, MLPESConfig
from bakken import MLPESDriver, minimize_geometry
from normal_modes import compute_normal_modes

# ── Constants ──────────────────────────────────────────────────────────────
HARTREE_TO_KCAL = 627.509474
ANG_TO_BOHR     = 1.88972612456
CM_INV_PER_AU   = 219474.63
FREQ_CONV        = 5140.48    # sqrt(Ha/(Bohr²·amu)) → cm⁻¹

# CH₂OO expected fundamentals (B3LYP/6-31G*, literature):
CH2OO_EXPECTED = [
    ('COO bend',      ~500),
    ('OO stretch',    ~880),
    ('CO stretch',    ~950),
    ('CH₂ wag',      ~1286),
    ('CH₂ scissors', ~1480),
    ('CH sym',       ~3010),
    ('CH asym',      ~3063),
]
PHYSICAL_FREQ_MAX = 4000.0   # cm⁻¹ — above this is unphysical for CH₂OO


def train_model(training_data_path: str, gamma: float,
                alpha_range=None, output_path: str = None) -> dict:
    """
    Train a KRR ML-PES with a fixed gamma, optimising alpha.

    Returns dict with keys: model_path, gamma, best_alpha, rmse_kcal.
    """
    if alpha_range is None:
        alpha_range = [1e-5, 1e-4, 1e-3, 1e-2, 0.1]

    config = MLPESConfig(
        gamma       = gamma,
        gamma_range = [gamma],          # fix gamma — only sweep alpha
        alpha_range = alpha_range,
    )
    trainer = MLPESTrainer(config)
    traj    = load_trajectory(training_data_path)

    history = trainer.train(
        traj.symbols,
        traj.coordinates,
        traj.energies,
        verbose=True,
    )

    if output_path:
        trainer.save(output_path)

    return {
        'model_path':  output_path,
        'gamma':       history.get('best_gamma', gamma),
        'best_alpha':  history.get('best_alpha'),
        'rmse_kcal':   history.get('best_rmse_kcal'),
        'trainer':     trainer,
        'symbols':     traj.symbols,
        'coords_min_energy': traj.coordinates[np.argmin(traj.energies)],
    }


def compute_hessian_frequencies(trainer, symbols, coords_start,
                                 preminimize=True) -> np.ndarray:
    """
    Save trainer to a temp pkl, load as MLPESDriver, pre-minimise, compute
    numerical Hessian frequencies.  Returns array of cm⁻¹ values.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tf:
        tmp_path = tf.name
    trainer.save(tmp_path)

    driver = MLPESDriver(tmp_path)
    os.unlink(tmp_path)

    coords = coords_start.copy()
    if preminimize:
        coords, _, _ = minimize_geometry(driver, coords,
                                         max_steps=300, force_tol=0.005,
                                         verbose=False)

    n_atoms = len(symbols)
    n_dof   = 3 * n_atoms
    delta   = 0.01   # Å, outer FD step

    H_ang2 = np.zeros((n_dof, n_dof))
    for i in range(n_dof):
        cp = coords.flatten().copy(); cp[i] += delta
        cm = coords.flatten().copy(); cm[i] -= delta
        F_p = driver.forces(cp.reshape(n_atoms, 3)).flatten()
        F_m = driver.forces(cm.reshape(n_atoms, 3)).flatten()
        H_ang2[:, i] = -(F_p - F_m) / (2.0 * delta)

    H_ang2  = 0.5 * (H_ang2 + H_ang2.T)
    H_bohr2 = H_ang2 * ANG_TO_BOHR ** 2

    frequencies, _, _, _ = compute_normal_modes(symbols, H_bohr2)
    return frequencies


def main():
    parser = argparse.ArgumentParser(
        description='Retrain ML-PES with softer gamma for physical Hessian')
    parser.add_argument('--training-data', required=True,
                        help='.npz training data file')
    parser.add_argument('--gamma-values', default='0.0001,0.0003,0.001,0.003,0.01',
                        help='Comma-separated gamma values to try (default: 0.0001–0.01)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (auto-timestamped if omitted)')
    parser.add_argument('--no-preminimize', action='store_true',
                        help='Skip geometry pre-minimisation before Hessian')
    args = parser.parse_args()

    ts      = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.output_dir) if args.output_dir else \
              Path('outputs') / f'softer_gamma_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)

    gammas = [float(g) for g in args.gamma_values.split(',')]

    print(f"\n{'='*70}")
    print(f"  SOFTER GAMMA ML-PES RETRAINING")
    print(f"{'='*70}")
    print(f"  Training data : {args.training_data}")
    print(f"  Gamma values  : {gammas}")
    print(f"  Output dir    : {out_dir}")
    print(f"  Pre-minimise  : {not args.no_preminimize}")

    # Load training data once to get starting geometry
    traj = load_trajectory(args.training_data)
    coords_start = traj.coordinates[np.argmin(traj.energies)]

    results = []

    for gamma in gammas:
        print(f"\n{'─'*70}")
        print(f"  Training  gamma = {gamma:.4f}")
        print(f"{'─'*70}")

        model_path = str(out_dir / f'mlpes_gamma_{gamma:.6f}.pkl')
        result = train_model(args.training_data, gamma,
                              output_path=model_path)

        print(f"  → gamma={gamma:.4f}  alpha={result['best_alpha']}  "
              f"RMSE={result['rmse_kcal']:.4f} kcal/mol")

        # Compute Hessian frequencies
        print(f"  Computing Hessian frequencies ...")
        freqs = compute_hessian_frequencies(
            result['trainer'], result['symbols'], coords_start,
            preminimize=not args.no_preminimize,
        )

        n_unphysical = int((np.abs(freqs) > PHYSICAL_FREQ_MAX).sum())
        freqs_physical = freqs[freqs < PHYSICAL_FREQ_MAX]
        freq_min = freqs_physical.min() if len(freqs_physical) > 0 else float('nan')
        freq_max = freqs_physical.max() if len(freqs_physical) > 0 else float('nan')

        print(f"  NM frequencies (cm⁻¹): "
              + "  ".join(f"{f:.0f}" for f in freqs))
        print(f"  Unphysical (>{PHYSICAL_FREQ_MAX:.0f}): {n_unphysical}/{len(freqs)}")

        result['frequencies'] = freqs
        result['n_unphysical'] = n_unphysical
        result['freq_min_physical'] = freq_min
        result['freq_max_physical'] = freq_max
        results.append(result)

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'gamma':>8}  {'alpha':>8}  {'RMSE (kcal)':>12}  "
          f"{'Unphysical':>10}  {'Freq range (cm⁻¹)':>20}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*20}")
    for r in results:
        fr = f"{r['freq_min_physical']:.0f} – {r['freq_max_physical']:.0f}"
        marker = '  ← best' if r['n_unphysical'] == 0 else ''
        print(f"  {r['gamma']:>8.4f}  {r['best_alpha']:>8.5f}  "
              f"{r['rmse_kcal']:>12.4f}  {r['n_unphysical']:>10}  "
              f"{fr:>20}{marker}")

    # ── Recommended model: fewest unphysical modes, then lowest RMSE ──────
    best = min(results, key=lambda r: (r['n_unphysical'], r['rmse_kcal']))
    best_link = out_dir / 'best_model.pkl'
    import shutil
    shutil.copy(best['model_path'], str(best_link))

    print(f"\n  Recommended model  : {best_link}")
    print(f"  gamma={best['gamma']:.4f}  RMSE={best['rmse_kcal']:.4f} kcal/mol  "
          f"unphysical modes={best['n_unphysical']}/{len(best['frequencies'])}")
    print(f"\n  NM frequencies:")
    for k, freq in enumerate(best['frequencies']):
        flag = '  ← UNPHYSICAL' if abs(freq) > PHYSICAL_FREQ_MAX else ''
        print(f"    Mode {k+1:2d}: {freq:8.1f} cm⁻¹{flag}")

    # ── Write text report ──────────────────────────────────────────────────
    report_path = out_dir / 'hessian_report.txt'
    with open(report_path, 'w') as fh:
        fh.write(f"Softer gamma retraining report — {ts}\n")
        fh.write(f"Training data: {args.training_data}\n\n")
        for r in results:
            fh.write(f"gamma={r['gamma']:.6f}  alpha={r['best_alpha']}  "
                     f"RMSE={r['rmse_kcal']:.4f} kcal/mol\n")
            fh.write("  NM frequencies (cm⁻¹): "
                     + "  ".join(f"{f:.1f}" for f in r['frequencies']) + "\n\n")
        fh.write(f"\nRecommended: gamma={best['gamma']:.6f}  "
                 f"model: {best['model_path']}\n")

    print(f"\n  Report saved       : {report_path}")
    print(f"\n  Next step — run IR workflow with the recommended model:")
    print(f"    python3 ir_md_spectrum.py \\")
    print(f"        --model {best_link} \\")
    print(f"        --training-data {args.training_data} \\")
    print(f"        --steps 20000 --temp 300 --timestep 0.5 --save-every 1 \\")
    print(f"        --preminimize --zpe-min-freq 50 --zpe-max-freq 4000")


if __name__ == '__main__':
    main()
