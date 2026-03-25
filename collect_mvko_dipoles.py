"""
Collect PSI4 dipoles for MVKO training frames.

Loads existing training data (energies/forces already correct), runs
PSI4 energy + oeprop on a representative subset to get dipoles, then
saves a new training file with dipoles populated.

Usage:
    python3 collect_mvko_dipoles.py \
        --training-data outputs/mvko_20260319_081314/combined_training_data.npz \
        --n-frames 150 \
        --output    outputs/mvko_dipoles_YYYYMMDD/training_with_dipoles.npz
"""
import argparse
import time
from pathlib import Path

import numpy as np

# ── PSI4 setup ────────────────────────────────────────────────────────────────
try:
    import psi4
    PSI4_AVAILABLE = True
    print(f"PSI4 {psi4.__version__} available")
except ImportError:
    PSI4_AVAILABLE = False
    print("PSI4 not available — cannot collect dipoles")

PSI4_METHOD  = 'b3lyp'
PSI4_OPTIONS = {'basis': '6-31G*', 'scf_type': 'df', 'reference': 'rhf',
                'maxiter': 200, 'e_convergence': 1e-7, 'd_convergence': 1e-7}
PSI4_MEM_GB  = 4
PSI4_THREADS = 4
AU_TO_DEBYE  = 2.541746


def _psi4_setup():
    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.be_quiet()
    psi4.set_memory(f'{PSI4_MEM_GB} GB')
    psi4.set_num_threads(PSI4_THREADS)
    psi4.set_options(PSI4_OPTIONS)


def _mol_str(symbols, coords):
    lines = ['0 1']
    for s, c in zip(symbols, coords):
        lines.append(f'{s}  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}')
    lines += ['units angstrom', 'no_reorient', 'no_com']
    return '\n'.join(lines)


def psi4_dipole(symbols, coords):
    """
    Run PSI4 single-point energy + dipole and return dipole in Debye.
    Uses properties=['dipole'] approach per PSI4 1.10 best practice.
    """
    _psi4_setup()
    mol = psi4.geometry(_mol_str(symbols, coords))
    e, wfn = psi4.energy(f'{PSI4_METHOD}/{PSI4_OPTIONS["basis"]}',
                         molecule=mol, return_wfn=True,
                         properties=['dipole'])
    dipole_D = np.zeros(3)
    # 'SCF DIPOLE' is stored as a 3-vector in atomic units (e·bohr)
    try:
        dip_au = np.array(psi4.variable('SCF DIPOLE'))
        dipole_D = dip_au * AU_TO_DEBYE
    except Exception:
        try:
            # Fallback: oeprop path — stores directly in Debye
            psi4.oeprop(wfn, 'DIPOLE')
            dipole_D = np.array([wfn.variable(f'DIPOLE {ax}')
                                 for ax in ['X', 'Y', 'Z']])
        except Exception:
            pass
    return dipole_D


def select_frames(coords, energies, n_frames, seed=42):
    """
    Select n_frames representative frames via a mix of:
      - low-energy (near equilibrium)
      - high-energy (anharmonic regions)
      - random (coverage)
    """
    N = len(coords)
    n_frames = min(n_frames, N)
    rng = np.random.default_rng(seed)

    idx_sort = np.argsort(energies)
    n_low  = n_frames // 3
    n_high = n_frames // 3
    n_rand = n_frames - n_low - n_high

    low_idx  = idx_sort[:n_low * 2][::2][:n_low]        # every other low-E frame
    high_idx = idx_sort[-(n_high * 2)::2][-n_high:]     # every other high-E frame
    remaining = np.setdiff1d(np.arange(N), np.union1d(low_idx, high_idx))
    rand_idx = rng.choice(remaining, size=min(n_rand, len(remaining)), replace=False)

    sel = np.unique(np.concatenate([low_idx, high_idx, rand_idx]))[:n_frames]
    return sel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data', required=True)
    parser.add_argument('--n-frames', type=int, default=150,
                        help='Number of frames to compute dipoles for (default 150)')
    parser.add_argument('--output', default=None,
                        help='Output npz path (default: auto-named in outputs/)')
    args = parser.parse_args()

    if not PSI4_AVAILABLE:
        raise RuntimeError("PSI4 required to collect dipoles")

    # Load training data
    data = np.load(args.training_data, allow_pickle=True)
    symbols    = data['symbols'].tolist()
    coords_all = data['coordinates']
    energies   = data['energies']
    forces     = data['forces']
    n_total    = len(coords_all)
    print(f"Loaded {n_total} training frames from {args.training_data}", flush=True)

    # Select representative frames
    sel_idx = select_frames(coords_all, energies, args.n_frames)
    n_sel   = len(sel_idx)
    print(f"Selected {n_sel} frames for dipole collection", flush=True)

    # Output path
    if args.output is None:
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = Path(f'outputs/mvko_dipoles_{ts}')
    else:
        out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'training_with_dipoles.npz' if args.output is None else Path(args.output)

    # Collect dipoles
    dipoles_out = np.zeros((n_sel, 3))
    coords_out  = coords_all[sel_idx]
    energies_out = energies[sel_idx]
    forces_out   = forces[sel_idx]

    t0 = time.time()
    n_ok = 0
    for i, idx in enumerate(sel_idx):
        c = coords_all[idx]
        try:
            d = psi4_dipole(symbols, c)
            dipoles_out[i] = d
            n_ok += 1
        except Exception as exc:
            print(f"  Frame {idx}: PSI4 failed ({exc}), dipole=0")

        if (i + 1) % 10 == 0 or i == n_sel - 1:
            elapsed = time.time() - t0
            mag = np.linalg.norm(dipoles_out[i])
            print(f"  Frame {i+1}/{n_sel}  idx={idx}  |μ|={mag:.4f} D  "
                  f"[{elapsed:.0f}s elapsed, {elapsed/(i+1):.1f}s/frame]", flush=True)

    print(f"\n  {n_ok}/{n_sel} frames with valid dipoles")
    print(f"  Dipole magnitudes: min={np.linalg.norm(dipoles_out, axis=-1).min():.4f}  "
          f"max={np.linalg.norm(dipoles_out, axis=-1).max():.4f}  "
          f"mean={np.linalg.norm(dipoles_out, axis=-1).mean():.4f} D")

    # Save
    import json as _json
    np.savez(out_path,
             symbols=np.array(symbols),
             coordinates=coords_out,
             energies=energies_out,
             forces=forces_out,
             dipoles=dipoles_out,
             metadata=np.array(_json.dumps({
                 'source': 'collect_mvko_dipoles',
                 'n_frames': int(n_sel),
                 'n_ok': int(n_ok),
                 'method': f'{PSI4_METHOD}/{PSI4_OPTIONS["basis"]}',
             })))
    print(f"\nSaved to {out_path}")
    print(f"\nRun IR spectrum with:")
    print(f"  python3 ir_md_spectrum.py \\")
    print(f"    --model outputs/mvko_20260319_081314/mlpes_initial.pkl \\")
    print(f"    --training-data {out_path} \\")
    print(f"    --steps 30000 --temp 300 --timestep 0.5 --save-every 1 \\")
    print(f"    --preminimize --zpe-min-freq 50 --zpe-max-freq 4000")


if __name__ == '__main__':
    main()
