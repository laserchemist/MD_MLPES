#!/usr/bin/env python3
"""
train_conformer_family.py — Train a PESFamily from per-conformer data
======================================================================
Trains one ML-PES model per conformer (from separate training data files),
optionally aligns reference energies to a common zero, assembles a
PESFamily, and saves it together with a conformer_manifest.json that can
be passed to ir_md_spectrum.py --multi-surface --conformer-manifest.

Usage
-----
    python3 train_conformer_family.py \\
        --conformers "s-cis:outputs/mvko_scis/combined_training_data.npz" \\
                     "s-trans:outputs/mvko_strans/combined_training_data.npz" \\
        --gamma 0.001 --alpha 1e-5 \\
        --align-energies \\
        --blend-width 3.0 \\
        --output outputs/conformer_family_YYYYMMDD/

The --conformers argument is one or more "label:path" pairs.
For each conformer, the lowest energy in its training set is used as the
reference energy when --align-energies is set, so all surfaces share a
common energy zero (the global minimum across all conformers).

Output files
------------
  outputs/conformer_family_*/
    <label>_model.pkl          — per-conformer trained ML-PES
    family.pkl                 — assembled PESFamily
    conformer_manifest.json    — manifest for ir_md_spectrum.py
    summary.json               — training metrics per conformer
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

HARTREE_TO_KCAL = 627.509474


def train_one(label, data_path, gamma, alpha, out_dir):
    """Train a single ML-PES for one conformer. Returns (trainer, E_min_ha)."""
    try:
        from modules.ml_pes import MLPESTrainer, MLPESConfig
        from modules.data_formats import TrajectoryData
    except ImportError:
        from ml_pes import MLPESTrainer, MLPESConfig
        from data_formats import TrajectoryData

    data = np.load(data_path, allow_pickle=True)
    symbols    = data['symbols'].tolist()
    coords     = data['coordinates']
    energies   = data['energies']
    forces     = data.get('forces', np.zeros((len(coords), len(symbols), 3)))

    traj = TrajectoryData(symbols=symbols, coordinates=coords,
                          energies=energies, forces=forces)

    cfg = MLPESConfig(gamma=gamma, alpha=alpha,
                      tune_hyperparameters=False, validation_split=0.1)
    trainer = MLPESTrainer(cfg)
    trainer.train(traj)

    pkl_path = out_dir / f'{label}_model.pkl'
    trainer.save(str(pkl_path))

    rmse = trainer.training_history.get('rmse_kcal',
           trainer.training_history.get('best_rmse_kcal', float('nan')))
    e_min = float(energies.min())
    print(f"  [{label}]  {len(coords)} frames  RMSE={rmse:.4f} kcal/mol  "
          f"E_min={e_min:.6f} Ha")
    return trainer, e_min, symbols, str(pkl_path), float(rmse)


def main():
    parser = argparse.ArgumentParser(
        description='Train a PESFamily from per-conformer training data')
    parser.add_argument('--conformers', nargs='+', required=True, metavar='LABEL:PATH',
                        help='One or more "label:path.npz" pairs')
    parser.add_argument('--gamma',          type=float, default=0.001)
    parser.add_argument('--alpha',          type=float, default=1e-5)
    parser.add_argument('--align-energies', action='store_true',
                        help='Shift each surface so that E=0 at the global minimum')
    parser.add_argument('--blend-width',    type=float, default=3.0,
                        help='Softmin blending width in kcal/mol (default 3.0)')
    parser.add_argument('--output',         default=None)
    args = parser.parse_args()

    # Parse label:path pairs
    conformers = {}
    for spec in args.conformers:
        if ':' not in spec:
            raise ValueError(f"--conformers entries must be 'label:path', got: {spec!r}")
        label, path = spec.split(':', 1)
        conformers[label] = path

    # Output directory
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.output) if args.output else \
              Path(f'outputs/conformer_family_{ts}')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train per-conformer models
    print(f"\nTraining {len(conformers)} conformer model(s)  "
          f"(γ={args.gamma}, α={args.alpha})\n")

    trainers = {}
    e_mins   = {}
    pkl_paths = {}
    summary   = {}
    symbols   = None

    for label, data_path in conformers.items():
        trainer, e_min, syms, pkl, rmse = train_one(
            label, data_path, args.gamma, args.alpha, out_dir)
        trainers[label]  = trainer
        e_mins[label]    = e_min
        pkl_paths[label] = pkl
        summary[label]   = {'rmse_kcal': rmse, 'e_min_ha': e_min, 'model': pkl}
        if symbols is None:
            symbols = syms
        elif syms != symbols:
            raise ValueError(
                f"Conformer '{label}' has different atom ordering from first conformer")

    # Reference energy alignment
    reference_energies = {}
    if args.align_energies:
        global_e_min = min(e_mins.values())
        print(f"\nGlobal E_min = {global_e_min:.6f} Ha  "
              f"(= {global_e_min * HARTREE_TO_KCAL:.4f} kcal/mol)")
        for label, e_min in e_mins.items():
            offset = e_min - global_e_min
            reference_energies[label] = e_min  # subtract this from each surface
            print(f"  [{label}]  offset = {offset * HARTREE_TO_KCAL:.4f} kcal/mol")
    else:
        reference_energies = {lbl: 0.0 for lbl in conformers}

    # Build PESFamily
    try:
        from modules.pes_family import PESFamily
    except ImportError:
        from pes_family import PESFamily

    family = PESFamily.from_trainers(symbols, trainers,
                                      blend_width=args.blend_width,
                                      reference_energies=reference_energies
                                      if args.align_energies else None)
    family_pkl = out_dir / 'family.pkl'
    family.save(str(family_pkl))

    # Write conformer manifest (for ir_md_spectrum.py --conformer-manifest)
    manifest = dict(pkl_paths)                     # label → pkl path
    manifest['_blend_width'] = args.blend_width
    if args.align_energies:
        manifest['_reference_energies'] = reference_energies
    manifest_path = out_dir / 'conformer_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Summary
    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutput: {out_dir}")
    print(f"  family.pkl              : {family_pkl}")
    print(f"  conformer_manifest.json : {manifest_path}")
    print(f"\nTo run multi-surface IR spectrum:")
    print(f"  python3 ir_md_spectrum.py \\")
    print(f"    --model {list(pkl_paths.values())[0]} \\")
    print(f"    --training-data <dipole_data.npz> \\")
    print(f"    --multi-surface \\")
    print(f"    --conformer-manifest {manifest_path} \\")
    print(f"    --blend-width {args.blend_width} \\")
    print(f"    --steps 30000 --temp 300 --preminimize \\")
    print(f"    --zpe-min-freq 50 --zpe-max-freq 4000")


if __name__ == '__main__':
    main()
