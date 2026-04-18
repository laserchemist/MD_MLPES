#!/usr/bin/env python3
"""
train_mace_model.py — Train a MACE ML-PES from existing npz training data.

Why MACE instead of Coulomb+KRR or sGDML
-----------------------------------------
Both KRR-based approaches (Coulomb matrix or sGDML) produce unphysical
normal-mode frequencies because the inverse-distance descriptor Jacobian
introduces second-derivative stiffness: C-H modes appear at 10,000–38,000 cm⁻¹
rather than ~3,000 cm⁻¹. This cannot be fixed by tuning hyperparameters — it is
intrinsic to 1/r-based descriptors under RBF/Matérn kernels.

MACE (Multi-Atomic Cluster Expansion) avoids this by:
  • Local atomic energy decomposition — no global descriptor stiffness
  • SO(3) equivariance — correct symmetry by design
  • Force training with forces_weight=100 — smooth PES near equilibrium

Usage
-----
  # Train with default MACE-OFF3 architecture:
  python3 train_mace_model.py \\
      --training-data outputs/wB97X_surface_20260406_223155/training_data_wB97X_aug.npz \\
      --output-dir    outputs/mace_wB97X_20260417 \\
      --n-train 900 --n-valid 80 --epochs 500

  # Quick test run:
  python3 train_mace_model.py \\
      --training-data outputs/.../training_data.npz \\
      --output-dir    outputs/mace_test \\
      --n-train 200 --n-valid 30 --epochs 100

Output
------
  <output-dir>/
    mace_model.pt            — Best MACE checkpoint (loaded by MACEDriver)
    mace_model.symbols.pkl   — Companion atom-order file for MACEDriver
    train.xyz                — Training extxyz (eV / eV/Å)
    valid.xyz                — Validation extxyz
    training_summary.txt     — Human-readable summary

Notes
-----
  • Requires: mace-torch >= 0.3.4, ase, torch with MPS/CUDA backend
  • Training runs mace_run_train as a subprocess (MACE CLI)
  • Default architecture: hidden_irreps='64x0e + 64x1o', r_max=5.0, 2 layers
  • Uses float64 for energy accuracy (matching training data)
  • MPS backend (Apple Silicon) selected automatically; falls back to CPU
  • After training, validates frequencies and suggests ir_md_spectrum.py command
"""

import argparse
import datetime
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from modules.mace_pes import npz_to_extxyz, _pick_device


# =============================================================================
# Helpers
# =============================================================================

def split_extxyz(xyz_path: str, train_path: str, valid_path: str,
                 n_train: int | None, n_valid: int, seed: int = 42) -> tuple[int, int]:
    """
    Split a single extxyz into train/valid sets.

    Reads all frames, shuffles deterministically, writes two extxyz files.
    Returns (n_train_actual, n_valid_actual).
    """
    from ase.io import read, write

    frames = read(xyz_path, index=':')
    n_total = len(frames)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)

    n_v = min(n_valid, n_total // 5)   # cap at 20% of data
    n_t = n_train if n_train is not None else n_total - n_v
    n_t = min(n_t, n_total - n_v)

    train_frames = [frames[i] for i in idx[:n_t]]
    valid_frames = [frames[i] for i in idx[n_t:n_t + n_v]]

    write(train_path, train_frames, format='extxyz')
    write(valid_path, valid_frames, format='extxyz')

    print(f"  Train: {len(train_frames)} frames → {train_path}")
    print(f"  Valid: {len(valid_frames)} frames → {valid_path}")
    return len(train_frames), len(valid_frames)


def find_best_checkpoint(results_dir: Path, model_name: str) -> Path | None:
    """
    Locate the best MACE checkpoint in results_dir.

    mace_run_train writes:
      <results_dir>/<model_name>_run-0_epoch-<N>.pt  (intermediate)
      <results_dir>/<model_name>_run-0_stagetwo.pt   (SWA final, preferred)
      <results_dir>/<model_name>_run-0.pt             (best epoch without SWA)
    """
    candidates = [
        results_dir / f'{model_name}_run-0_stagetwo.pt',
        results_dir / f'{model_name}_run-0.pt',
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fall back: latest epoch checkpoint
    epoch_ckpts = sorted(results_dir.glob(f'{model_name}_run-0_epoch-*.pt'))
    if epoch_ckpts:
        return epoch_ckpts[-1]
    return None


def write_summary(path: Path, symbols, n_frames, n_train, n_valid,
                  model_pt: Path, device: str, dtype: str, epochs: int,
                  hidden_irreps: str, r_max: float, n_layers: int,
                  energy_cutoff: float):
    lines = [
        "MACE Training Summary",
        "=" * 60,
        f"Date              : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Molecule          : {''.join(dict.fromkeys(symbols))} ({len(symbols)} atoms)",
        f"Total frames      : {n_frames}",
        f"n_train           : {n_train}",
        f"n_valid           : {n_valid}",
        f"Energy cutoff     : {energy_cutoff:.0f} kcal/mol",
        "",
        "Architecture",
        "-" * 40,
        f"  hidden_irreps   : {hidden_irreps}",
        f"  r_max           : {r_max} Å",
        f"  n_layers        : {n_layers}",
        f"  device          : {device}",
        f"  dtype           : {dtype}",
        f"  max_epochs      : {epochs}",
        "",
        "Output",
        "-" * 40,
        f"  checkpoint      : {model_pt}",
        f"  symbols file    : {model_pt.with_suffix('.symbols.pkl')}",
        "",
        "Why MACE",
        "-" * 40,
        "  Coulomb+KRR and sGDML share the same pathology: 1/r descriptor",
        "  second derivatives are intrinsically stiff, producing C-H normal",
        "  modes at 10000-38000 cm⁻¹ instead of ~3000 cm⁻¹. MACE avoids",
        "  this via local atomic energies + SO(3) equivariance + force training.",
    ]
    path.write_text('\n'.join(lines) + '\n')


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train MACE ML-PES from npz training data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--training-data', required=True,
                        help='Path to training data .npz (coordinates, energies, forces, symbols)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (auto-timestamped if omitted)')
    parser.add_argument('--n-train', type=int, default=None,
                        help='Training set size (default: all minus n-valid)')
    parser.add_argument('--n-valid', type=int, default=80,
                        help='Validation set size')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Maximum training epochs')
    parser.add_argument('--energy-cutoff', type=float, default=15.0,
                        help='Keep only frames within this many kcal/mol of minimum '
                             '(default 15 kcal/mol for 300K IR)')
    parser.add_argument('--hidden-irreps', default='64x0e + 64x1o',
                        help='MACE hidden irreps (architecture width/symmetry). '
                             'Smaller: "32x0e + 32x1o" for fast tests.')
    parser.add_argument('--r-max', type=float, default=5.0,
                        help='Cutoff radius in Angstrom (neighbourhood list)')
    parser.add_argument('--n-layers', type=int, default=2,
                        help='Number of interaction layers')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--forces-weight', type=float, default=100.0,
                        help='Weight on force RMSE in the loss function')
    parser.add_argument('--swa', action='store_true', default=True,
                        help='Enable Stochastic Weight Averaging (SWA) for final stage. '
                             'Recommended for smoother PES. Use --no-swa to disable.')
    parser.add_argument('--no-swa', dest='swa', action='store_false')
    parser.add_argument('--device', default='cpu',
                        help='Device: cpu (default — faster than MPS for small-molecule '
                             'batches), cuda, mps, auto (MPS→CUDA→CPU). '
                             'Note: MPS requires PyTorch 2.3+ for float64 support.')
    parser.add_argument('--dtype', default='auto',
                        help='Floating point precision: auto (float32 on MPS, float64 on CPU/CUDA), '
                             'float32, or float64. MPS does not support float64.')
    parser.add_argument('--name', default='mace_mvko',
                        help='Model name prefix for output checkpoint files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/valid split')
    args = parser.parse_args()

    ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path(args.output_dir) if args.output_dir else Path('outputs') / f'mace_{ts}'
    out.mkdir(parents=True, exist_ok=True)

    device = _pick_device(args.device)

    # MPS (Apple Silicon) only supports float32
    if args.dtype == 'auto':
        dtype = 'float32' if device == 'mps' else 'float64'
    else:
        dtype = args.dtype
    if device == 'mps' and dtype == 'float64':
        print("  WARNING: MPS does not support float64; switching to float32.")
        dtype = 'float32'

    print(f"\n{'='*60}")
    print("  MACE TRAINING")
    print(f"{'='*60}")
    print(f"  Training data : {args.training_data}")
    print(f"  Output dir    : {out}")
    print(f"  Device        : {device}  dtype={dtype}")

    # ── Convert npz → extxyz ───────────────────────────────────────────
    print("\n  Converting training data to extxyz (eV / eV/Å)…")
    all_xyz = str(out / 'all_frames.xyz')
    n_kept = npz_to_extxyz(args.training_data, all_xyz,
                            energy_cutoff_kcal=args.energy_cutoff)
    if n_kept < 20:
        print(f"  ERROR: Only {n_kept} frames passed energy filter — too few to train.")
        sys.exit(1)

    # Save symbols for MACEDriver
    data    = np.load(args.training_data, allow_pickle=True)
    symbols = data['symbols'].tolist()
    sym_pkl = out / f'{args.name}.symbols.pkl'
    with open(sym_pkl, 'wb') as f:
        pickle.dump(symbols, f)
    print(f"  Symbols saved : {sym_pkl}  ({symbols})")

    # ── Train/valid split ──────────────────────────────────────────────
    print("\n  Splitting into train/valid sets…")
    train_xyz = str((out / 'train.xyz').resolve())
    valid_xyz = str((out / 'valid.xyz').resolve())
    n_train, n_valid = split_extxyz(
        all_xyz, train_xyz, valid_xyz,
        n_train=args.n_train, n_valid=args.n_valid, seed=args.seed,
    )

    # ── Build mace_run_train command ───────────────────────────────────
    swa_start = max(1, int(args.epochs * 0.9))   # SWA starts at 90% of epochs
    out_abs = str(out.resolve())

    cmd = [
        'mace_run_train',
        f'--name={args.name}',
        f'--train_file={train_xyz}',
        f'--valid_file={valid_xyz}',
        '--E0s=average',
        '--model=MACE',
        f'--hidden_irreps={args.hidden_irreps}',
        f'--r_max={args.r_max}',
        f'--num_interactions={args.n_layers}',
        f'--batch_size={args.batch_size}',
        f'--max_num_epochs={args.epochs}',
        f'--forces_weight={args.forces_weight:.0f}',
        '--energy_weight=1',
        f'--lr={args.lr}',
        f'--device={device}',
        '--energy_key=REF_energy',
        '--forces_key=REF_forces',
        f'--default_dtype={dtype}',
        f'--seed={args.seed}',
        '--scheduler_patience=5',
        '--eval_interval=2',
        f'--results_dir={out_abs}',
        f'--checkpoints_dir={out_abs}',
        f'--log_dir={out_abs}',
    ]
    if args.swa:
        cmd += [
            '--swa',
            '--swa_energy_weight=1000',
            f'--swa_forces_weight={args.forces_weight:.0f}',
            f'--start_swa={swa_start}',
        ]

    print(f"\n{'='*60}")
    print("  RUNNING mace_run_train")
    print(f"{'='*60}")
    print('  ' + ' \\\n      '.join(cmd))
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  ERROR: mace_run_train exited with code {result.returncode}")
        print(f"  Check logs in {out}/")
        sys.exit(result.returncode)

    # ── Locate best checkpoint ─────────────────────────────────────────
    best_ckpt = find_best_checkpoint(out, args.name)
    if best_ckpt is None:
        print(f"\n  ERROR: No checkpoint found in {out}/")
        print("  Training may have failed — check logs above.")
        sys.exit(1)

    # Copy to canonical output name and save companion symbols file
    model_pt   = out / 'mace_model.pt'
    model_syms = out / 'mace_model.symbols.pkl'

    import shutil
    shutil.copy2(best_ckpt, model_pt)
    shutil.copy2(sym_pkl, model_syms)

    print(f"\n  Best checkpoint   : {best_ckpt}")
    print(f"  Model saved       : {model_pt}")
    print(f"  Symbols saved     : {model_syms}")

    # ── Write summary ──────────────────────────────────────────────────
    write_summary(
        out / 'training_summary.txt',
        symbols, n_kept, n_train, n_valid,
        model_pt, device, dtype, args.epochs,
        args.hidden_irreps, args.r_max, args.n_layers,
        args.energy_cutoff,
    )
    print(f"  Summary           : {out / 'training_summary.txt'}")

    # ── Suggested next commands ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  NEXT STEPS")
    print(f"{'='*60}")
    print(f"\n  Validate frequencies and MD stability:")
    print(f"  python3 validate_pes_frequencies.py \\")
    print(f"      --mace-model {model_pt} \\")
    print(f"      --training-data {args.training_data}")
    print(f"\n  Run IR spectrum:")
    print(f"  python3 ir_md_spectrum.py \\")
    print(f"      --mace-model {model_pt} \\")
    print(f"      --training-data <data_with_dipoles.npz> \\")
    print(f"      --steps 30000 --temp 300 --preminimize \\")
    print(f"      --zpe-min-freq 50 --zpe-max-freq 4000 --n-trajectories 5")


if __name__ == '__main__':
    main()
