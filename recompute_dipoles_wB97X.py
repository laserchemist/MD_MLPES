#!/usr/bin/env python3
"""
Recompute dipoles for an existing training-data npz using wB97X-D/6-31G*.

Takes a dataset whose dipoles were computed at a different level (e.g. B3LYP)
and replaces the dipole array with fresh wB97X-D/6-31G* values at the same
geometries.  Energies and forces are preserved as-is (they are not recomputed).

Usage:
    python3 recompute_dipoles_wB97X.py \
        --input  outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \
        --output outputs/mvko_dipoles_wB97X_base \
        [--method wb97x-d] [--basis 6-31G*] \
        [--resume]          # skip frames whose dipole already written to output
        [--dry-run]         # print geometries without PSI4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

AU_TO_DEBYE = 2.541746


def psi4_dipole(symbols, coords_ang, method='wb97x-d', basis='6-31G*'):
    """Return dipole vector in Debye via PSI4, or None if unavailable."""
    try:
        import psi4
        psi4.core.clean_options()
        psi4.core.clean()
        psi4.core.be_quiet()
        psi4.set_memory('4 GB')
        psi4.set_num_threads(4)
        psi4.set_options({'basis': basis, 'scf_type': 'df', 'reference': 'rhf',
                          'maxiter': 200, 'e_convergence': 1e-7, 'd_convergence': 1e-7})
        geom = '0 1\n'
        for s, c in zip(symbols, coords_ang):
            geom += f'{s}  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}\n'
        geom += 'units angstrom\nno_reorient\nno_com\n'
        mol = psi4.geometry(geom)
        _, wfn = psi4.energy(f'{method}/{basis}', molecule=mol,
                             return_wfn=True, properties=['dipole'])
        try:
            dip = np.array(psi4.variable('SCF DIPOLE'))
            if np.linalg.norm(dip) < 1e-10:
                raise ValueError('zero')
            return dip * AU_TO_DEBYE
        except Exception:
            psi4.oeprop(wfn, 'DIPOLE')
            return np.array([wfn.variable(f'DIPOLE {ax}') for ax in ['X', 'Y', 'Z']])
    except ImportError:
        return None


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--input',   required=True,  help='Source npz with coordinates/energies/forces')
    ap.add_argument('--output',  required=True,  help='Output directory')
    ap.add_argument('--method',  default='wb97x-d')
    ap.add_argument('--basis',   default='6-31G*')
    ap.add_argument('--resume',  action='store_true',
                    help='If partial_dipoles.npy exists in output dir, skip completed frames')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Load input dataset
    d = np.load(args.input, allow_pickle=True)
    symbols  = list(d['symbols'])
    coords   = d['coordinates']      # (N, n_atoms, 3)
    energies = d['energies']         # (N,)
    forces   = d['forces']           # (N, n_atoms, 3)
    old_dips = d['dipoles']          # (N, 3) — to be replaced
    n = len(coords)
    print(f'Loaded {n} frames from {args.input}')
    print(f'Old dipole |μ| range: {np.linalg.norm(old_dips,axis=1).min():.3f}–'
          f'{np.linalg.norm(old_dips,axis=1).max():.3f} D  (will be replaced)')

    # Resume: load partial results if they exist
    partial_path = out / 'partial_dipoles.npy'
    new_dips = np.full((n, 3), np.nan)
    start_idx = 0
    if args.resume and partial_path.exists():
        saved = np.load(partial_path)
        done = int(np.sum(~np.isnan(saved[:, 0])))
        new_dips[:len(saved)] = saved
        start_idx = done
        print(f'Resuming from frame {start_idx} ({done} already done)')

    t0_total = time.perf_counter()
    n_failed = 0

    for i in range(start_idx, n):
        t0 = time.perf_counter()
        print(f'  [{i+1:3d}/{n}]  ', end='', flush=True)

        if args.dry_run:
            print(f'SKIP (--dry-run)')
            continue

        dip = psi4_dipole(symbols, coords[i], method=args.method, basis=args.basis)
        dt = time.perf_counter() - t0

        if dip is None:
            print(f'PSI4 unavailable — aborting')
            sys.exit(1)

        new_dips[i] = dip
        mag = np.linalg.norm(dip)
        delta = mag - np.linalg.norm(old_dips[i])
        print(f'|μ|={mag:.3f} D  (Δ vs B3LYP: {delta:+.3f} D)  ({dt:.1f}s)')

        # Checkpoint after every frame
        np.save(partial_path, new_dips)

    if args.dry_run:
        print('\nDry run complete — no PSI4 calls made.')
        return

    done_mask = ~np.isnan(new_dips[:, 0])
    n_done = done_mask.sum()
    print(f'\n{n_done}/{n} frames completed successfully.')
    if n_done < n:
        print(f'  {n - n_done} frames failed — check partial_dipoles.npy')
        n_failed = n - n_done

    # Save final dataset with wB97X-D dipoles
    out_npz = out / 'training_with_dipoles.npz'
    final_dips = new_dips.copy()
    final_dips[~done_mask] = old_dips[~done_mask]   # keep old value for failures (flagged by nan before)

    np.savez(out_npz,
             symbols=np.array(symbols),
             coordinates=coords,
             energies=energies,
             forces=forces,
             dipoles=final_dips,
             metadata=np.array(json.dumps({
                 'source': str(args.input),
                 'method': args.method,
                 'basis':  args.basis,
                 'n_frames': n,
                 'n_completed': int(n_done),
                 'n_failed': int(n_failed),
                 'ts': time.strftime('%Y%m%d_%H%M%S'),
             })))
    print(f'Saved → {out_npz}')
    print(f'New |μ| range: {np.linalg.norm(final_dips,axis=1).min():.3f}–'
          f'{np.linalg.norm(final_dips,axis=1).max():.3f} D')
    print(f'Total time: {(time.perf_counter()-t0_total)/60:.1f} min')


if __name__ == '__main__':
    main()
