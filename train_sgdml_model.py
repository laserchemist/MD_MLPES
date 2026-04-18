#!/usr/bin/env python3
"""
train_sgdml_model.py — Train an sGDML ML-PES from existing npz training data.

Replaces the Coulomb-matrix KRR backend with sGDML, which:
  • Trains on forces (native), giving physically correct Hessian at equilibrium
  • Enforces molecular symmetries (CH3 permutations, etc.) automatically
  • Produces analytic forces in predict(); no FD needed during MD

Usage
-----
  # Sweep sig values, save best model:
  python3 train_sgdml_model.py \\
      --training-data outputs/wB97X_surface_20260406_223155/training_data_wB97X_aug.npz \\
      --output-dir    outputs/sgdml_wB97X_20260417 \\
      --sig-values    10,25,50,100,200 \\
      --n-train       900 --n-valid 80

  # Single sig (fast, for quick tests):
  python3 train_sgdml_model.py \\
      --training-data outputs/wB97X_surface_20260406_223155/training_data_wB97X_aug.npz \\
      --output-dir    outputs/sgdml_wB97X_test \\
      --sig           50 --n-train 200 --n-valid 30 --no-sym

Output
------
  <output-dir>/
    sgdml_model.pkl         — SGDMLModel (energy + analytic forces)
    sweep_results.json      — sig sweep metrics (if --sig-values used)
    training_summary.txt    — human-readable summary
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from modules.sgdml_pes import train_sgdml, train_sgdml_sweep, SGDMLModel


# =============================================================================
# Helpers
# =============================================================================

def load_training_data(path: str, energy_cutoff: float = 15.0):
    data = np.load(path, allow_pickle=True)
    symbols = data['symbols'].tolist()
    coords  = data['coordinates']   # (n, n_atoms, 3) Ang
    energies = data['energies']     # (n,) Hartree
    forces  = data['forces']        # (n, n_atoms, 3) Ha/Ang

    # Filter frames: keep only near-equilibrium geometries (dE < 50 kcal/mol).
    # sGDML works best on a compact region of the PES; high-energy frames don't improve
    # near-eq frequencies and corrupt the force-field integration consistency check.
    e_min = energies.min()
    e_rel = (energies - e_min) * 627.509474  # kcal/mol relative to minimum
    mask  = e_rel < energy_cutoff
    n_removed = int((~mask).sum())
    if n_removed > 0:
        print(f"  Filtered {n_removed} frames with dE > {energy_cutoff:.0f} kcal/mol "
              f"({int(mask.sum())} kept — near-eq focus for 300K IR)")
        coords   = coords[mask]
        energies = energies[mask]
        forces   = forces[mask]

    # sGDML requires force magnitudes to be non-zero and finite
    fmax = np.abs(forces).max(axis=(1,2))
    bad  = ~np.isfinite(fmax) | (fmax > 500)
    if bad.sum() > 0:
        print(f"  Filtered {bad.sum()} frames with non-finite or extreme forces")
        coords   = coords[~bad]
        energies = energies[~bad]
        forces   = forces[~bad]

    print(f"  Loaded {len(energies)} frames from {path}")
    return symbols, coords, energies, forces


def write_summary(path: Path, symbols, n_frames, best_sig, f_rmse, e_rmse,
                  n_train, n_valid, sweep_results, use_sym):
    lines = [
        "sGDML Training Summary",
        "=" * 50,
        f"Date         : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Molecule     : {''.join(symbols)} ({len(symbols)} atoms)",
        f"Total frames : {n_frames}",
        f"n_train      : {n_train}",
        f"n_valid      : {n_valid}",
        f"use_sym      : {use_sym}",
        f"Best sig     : {best_sig}",
        f"Force RMSE   : {f_rmse:.5f} Ha/Å  ({f_rmse*627.509474:.3f} kcal/mol/Å)",
        f"Energy RMSE  : {e_rmse:.5f} Ha  ({e_rmse*627.509474:.3f} kcal/mol)",
        "",
    ]
    if sweep_results:
        lines += ["Sig sweep:", "-" * 30]
        for sig, fr, er in zip(sweep_results['sig_values'],
                                sweep_results['force_rmse'],
                                sweep_results['energy_rmse']):
            marker = " ←" if sig == sweep_results['best_sig'] else ""
            lines.append(f"  sig={sig!s:>6}: F-RMSE={fr:.5f} Ha/Å  E-RMSE={er:.5f} Ha{marker}")
    path.write_text('\n'.join(lines) + '\n')


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train sGDML ML-PES from npz training data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--training-data', required=True,
                        help='Path to training data .npz (coordinates, energies, forces)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (auto-timestamped if omitted)')
    parser.add_argument('--sig', type=float, default=None,
                        help='Single sig value (skip sweep). Accepts floats (e.g. 0.1, 1, 10).')
    parser.add_argument('--sig-values', default='0.05,0.1,0.2,0.5,1,2,5',
                        help='Comma-separated sig values to sweep (floats accepted). '
                             'Calibrate against descriptor L2 distances via --diagnose-sig.')
    parser.add_argument('--lam', type=float, default=1e-10,
                        help='Regularisation lambda (sGDML default 1e-10)')
    parser.add_argument('--n-train', type=int, default=None,
                        help='Training set size (default: all minus n-valid)')
    parser.add_argument('--n-valid', type=int, default=80,
                        help='Validation set size')
    parser.add_argument('--no-sym', action='store_true',
                        help='Disable symmetry discovery (faster, less data-efficient)')
    parser.add_argument('--use-E-cstr', action='store_true',
                        help='Include energy constraints directly in the kernel (use_E_cstr=True). '
                             'Recommended when training data is not from a single trajectory.')
    parser.add_argument('--energy-cutoff', type=float, default=15.0,
                        help='Keep only frames within this many kcal/mol of the minimum energy '
                             '(default 15 kcal/mol). Use tighter cutoff (e.g. 5) for diverse data.')
    parser.add_argument('--max-processes', type=int, default=4,
                        help='Parallel processes for symmetry discovery')
    parser.add_argument('--name', default='mvko',
                        help='Dataset name tag written into model dict')
    parser.add_argument('--theory', default='wB97X-D/6-31G*',
                        help='Level of theory tag written into model dict')
    args = parser.parse_args()

    ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path(args.output_dir) if args.output_dir else Path('outputs') / f'sgdml_{ts}'
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  sGDML TRAINING")
    print(f"{'='*60}")
    print(f"  Training data : {args.training_data}")
    print(f"  Output dir    : {out}")

    symbols, coords, energies, forces = load_training_data(
        args.training_data, energy_cutoff=args.energy_cutoff)
    use_sym   = not args.no_sym
    use_E_cstr = args.use_E_cstr

    if args.sig is not None:
        # Single sig
        model = train_sgdml(
            symbols, coords, energies, forces,
            sig=args.sig, lam=args.lam, use_sym=use_sym, use_E_cstr=use_E_cstr,
            n_train=args.n_train, n_valid=args.n_valid,
            name=args.name, theory=args.theory,
            max_processes=args.max_processes,
        )
        best_sig      = args.sig
        f_rmse        = (model.train_force_rmse or float('nan')) if model else float('nan')
        e_rmse        = (model.train_energy_rmse or float('nan')) if model else float('nan')
        sweep_results = None
        n_train_used  = (args.n_train or len(energies) - args.n_valid)
        n_valid_used  = min(args.n_valid, len(energies) - n_train_used)
    else:
        # Sig sweep — parse as floats
        sig_values = [float(s) for s in args.sig_values.split(',')]
        model, sweep_results = train_sgdml_sweep(
            symbols, coords, energies, forces,
            sig_values=sig_values, lam=args.lam, use_sym=use_sym, use_E_cstr=use_E_cstr,
            n_train=args.n_train, n_valid=args.n_valid,
            name=args.name, theory=args.theory,
            max_processes=args.max_processes,
        )
        best_sig     = sweep_results['best_sig']
        f_rmse       = (model.train_force_rmse or float('nan')) if model else float('nan')
        e_rmse       = (model.train_energy_rmse or float('nan')) if model else float('nan')
        n_train_used  = sweep_results['n_train']
        n_valid_used  = sweep_results['n_valid']

    # Save model
    model_path = out / 'sgdml_model.pkl'
    model.save(str(model_path))
    print(f"\n  Model saved → {model_path}")

    # Save sweep results
    if sweep_results:
        sr_path = out / 'sweep_results.json'
        with open(sr_path, 'w') as f:
            json.dump(sweep_results, f, indent=2)
        print(f"  Sweep results → {sr_path}")

    # Summary
    write_summary(
        out / 'training_summary.txt',
        symbols, len(energies), best_sig, f_rmse, e_rmse,
        n_train_used, n_valid_used, sweep_results, use_sym,
    )
    print(f"  Summary       → {out / 'training_summary.txt'}")

    print(f"\n  Best sig={best_sig}: force RMSE={f_rmse:.5f} Ha/Å, energy RMSE={e_rmse:.5f} Ha")
    print(f"\n  To validate frequencies and run IR:")
    print(f"  python3 validate_sgdml_frequencies.py --model {model_path} \\")
    print(f"      --training-data {args.training_data}")
    print(f"\n  python3 ir_md_spectrum.py --sgdml-model {model_path} \\")
    print(f"      --training-data {args.training_data} \\")
    print(f"      --steps 30000 --temp 300 --preminimize \\")
    print(f"      --zpe-min-freq 50 --zpe-max-freq 4000 --n-trajectories 5")


if __name__ == '__main__':
    main()
