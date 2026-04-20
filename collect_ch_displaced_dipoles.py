"""
Collect PSI4 B3LYP/6-31G* dipoles for frames with large C-H displacement,
sampled from NM-PES MD trajectories.  Merges with an existing dipole dataset
and saves a combined training file for the dipole surface.

Usage:
    python3 collect_ch_displaced_dipoles.py \
        --traj-dir      outputs/ir_spectrum_NM_PES_delta_300K_v2 \
        --existing      outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \
        --n-new  120 \
        --output outputs/mvko_dipoles_ch_v2_YYYYMMDD/training_with_dipoles.npz
    # Dry-run (no PSI4):
        python3 collect_ch_displaced_dipoles.py --dry-run ...
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── PSI4 ──────────────────────────────────────────────────────────────────────
try:
    import psi4
    PSI4_AVAILABLE = True
    print(f"PSI4 {psi4.__version__} available")
except ImportError:
    PSI4_AVAILABLE = False
    print("PSI4 not found — use --dry-run to preview selection without computing dipoles")

PSI4_METHOD  = 'b3lyp'
PSI4_OPTIONS = {'basis': '6-31G*', 'scf_type': 'df', 'reference': 'rhf',
                'maxiter': 200, 'e_convergence': 1e-7, 'd_convergence': 1e-7}
PSI4_MEM_GB  = 4
PSI4_THREADS = 4
AU_TO_DEBYE  = 2.541746

# MVKO atom ordering: C O O C C C H H H H H H  (indices 0-11)
# C-H bond pairs (0-indexed): vinyl-C2-H, vinyl=CH2 x2, methyl x3
CH_PAIRS = [(3, 6), (4, 7), (4, 8), (5, 9), (5, 10), (5, 11)]
CH_EQ_MEAN = 1.090  # Å — mean equilibrium C-H from PSI4 min

# Sampling fractions by |ΔrCH| tier
FRAC_LARGE  = 0.60   # top quartile of |ΔrCH|
FRAC_MIDDLE = 0.20   # 25–75th percentile
FRAC_NEAR   = 0.20   # bottom quartile (near-eq coverage)

DEDUP_RMSD  = 0.025  # Å — reject new frame if within this RMSD of any existing frame


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
    """Single-point B3LYP/6-31G* dipole (Debye)."""
    _psi4_setup()
    mol = psi4.geometry(_mol_str(symbols, coords))
    _, wfn = psi4.energy(f'{PSI4_METHOD}/{PSI4_OPTIONS["basis"]}',
                         molecule=mol, return_wfn=True,
                         properties=['dipole'])
    try:
        return np.array(psi4.variable('SCF DIPOLE')) * AU_TO_DEBYE
    except Exception:
        try:
            psi4.oeprop(wfn, 'DIPOLE')
            return np.array([wfn.variable(f'DIPOLE {ax}') for ax in ['X', 'Y', 'Z']])
        except Exception:
            return np.zeros(3)


def read_xyz_trajectory(path, stride=1):
    """Read an XYZ trajectory file; return (symbols, coords array)."""
    frames_coords = []
    symbols = None
    with open(path) as fh:
        frame_idx = 0
        while True:
            line = fh.readline()
            if not line:
                break
            try:
                n_atoms = int(line.strip())
            except ValueError:
                break
            fh.readline()  # comment line
            atoms = []
            c = []
            for _ in range(n_atoms):
                parts = fh.readline().split()
                atoms.append(parts[0])
                c.append([float(x) for x in parts[1:4]])
            if frame_idx % stride == 0:
                frames_coords.append(c)
                if symbols is None:
                    symbols = atoms
            frame_idx += 1
    return symbols, np.array(frames_coords, dtype=np.float64)


def max_ch_displacement(coords_array, ch_eq=CH_EQ_MEAN):
    """Return max |ΔrCH| across all C-H pairs for each frame."""
    disps = np.zeros(len(coords_array))
    for i, j in CH_PAIRS:
        r = np.linalg.norm(coords_array[:, i] - coords_array[:, j], axis=-1)
        disps = np.maximum(disps, np.abs(r - ch_eq))
    return disps


def rmsd_to_set(coords, ref_set):
    """Min RMSD between coords (12,3) and any frame in ref_set."""
    if len(ref_set) == 0:
        return np.inf
    diffs = ref_set - coords[None]          # (N, 12, 3)
    return np.sqrt((diffs ** 2).sum(axis=(1, 2)) / diffs.shape[1]).min()


def select_frames(coords_pool, n_new, rng, existing_coords=None):
    """
    Stratified selection biased toward large |ΔrCH|.
    Returns indices into coords_pool.
    """
    disps = max_ch_displacement(coords_pool)
    N = len(coords_pool)

    q75 = np.percentile(disps, 75)
    q25 = np.percentile(disps, 25)
    large_mask  = disps >= q75
    middle_mask = (disps >= q25) & (disps < q75)
    near_mask   = disps < q25

    n_large  = int(n_new * FRAC_LARGE)
    n_middle = int(n_new * FRAC_MIDDLE)
    n_near   = n_new - n_large - n_middle

    def _sample(mask, n):
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return np.array([], dtype=int)
        # Sort by displacement descending (large tier) or random
        order = np.argsort(-disps[idxs]) if mask is large_mask else rng.permutation(len(idxs))
        return idxs[order[:n]]

    # Re-reference large_mask in the closure properly
    def _sample_tier(mask, n, prefer_large):
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return np.array([], dtype=int)
        if prefer_large:
            order = np.argsort(-disps[idxs])
        else:
            order = rng.permutation(len(idxs))
        return idxs[order[:n]]

    sel = np.unique(np.concatenate([
        _sample_tier(large_mask,  n_large,  True),
        _sample_tier(middle_mask, n_middle, False),
        _sample_tier(near_mask,   n_near,   False),
    ]))

    # Deduplicate against existing
    if existing_coords is not None and len(existing_coords) > 0:
        keep = []
        for idx in sel:
            if rmsd_to_set(coords_pool[idx], existing_coords) >= DEDUP_RMSD:
                keep.append(idx)
        sel = np.array(keep, dtype=int)

    print(f"  |ΔrCH| range in pool: {disps.min():.4f}–{disps.max():.4f} Å  "
          f"q25={q25:.4f} q75={q75:.4f}")
    print(f"  Tier counts after dedup — large: {(large_mask[sel]).sum()}  "
          f"middle: {(middle_mask[sel]).sum()}  near: {(near_mask[sel]).sum()}")
    print(f"  Total selected: {len(sel)}")
    return sel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj-dir',  default='outputs/ir_spectrum_NM_PES_delta_300K_v2',
                        help='Directory containing traj_01.xyz … traj_05.xyz')
    parser.add_argument('--existing',  default='outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz',
                        help='Existing dipole training data to merge with')
    parser.add_argument('--n-new',     type=int, default=120,
                        help='Number of new dipole frames to compute (default 120)')
    parser.add_argument('--stride',    type=int, default=30,
                        help='Read every Nth frame from each trajectory (default 30)')
    parser.add_argument('--output',    default=None)
    parser.add_argument('--dry-run',   action='store_true',
                        help='Select frames and print stats without running PSI4')
    args = parser.parse_args()

    if not args.dry_run and not PSI4_AVAILABLE:
        sys.exit("PSI4 required (or use --dry-run)")

    rng = np.random.default_rng(42)

    # ── Load existing dipole data ──────────────────────────────────────────────
    print(f"\nLoading existing dipoles from {args.existing}")
    ex = np.load(args.existing, allow_pickle=True)
    ex_symbols   = ex['symbols'].tolist()
    ex_coords    = ex['coordinates']    # (150, 12, 3)
    ex_energies  = ex['energies']
    ex_forces    = ex['forces']
    ex_dipoles   = ex['dipoles']
    print(f"  {len(ex_coords)} existing frames  "
          f"|μ| {np.linalg.norm(ex_dipoles, axis=-1).mean():.3f} D mean")

    # ── Read NM-PES trajectories ───────────────────────────────────────────────
    traj_dir = Path(args.traj_dir)
    xyz_files = sorted(traj_dir.glob('traj_*.xyz'))
    if not xyz_files:
        sys.exit(f"No traj_*.xyz found in {traj_dir}")
    print(f"\nReading {len(xyz_files)} trajectories (stride={args.stride})…")

    all_symbols = None
    pool_coords = []
    for p in xyz_files:
        syms, crds = read_xyz_trajectory(p, stride=args.stride)
        pool_coords.append(crds)
        if all_symbols is None:
            all_symbols = syms
        print(f"  {p.name}: {len(crds)} frames after stride")
    pool_coords = np.concatenate(pool_coords, axis=0)
    print(f"  Pool total: {len(pool_coords)} frames")

    # ── Select C-H-displacement-biased frames ─────────────────────────────────
    print(f"\nSelecting {args.n_new} new frames (C-H-displacement-biased)…")
    sel_idx = select_frames(pool_coords, args.n_new, rng, existing_coords=ex_coords)
    new_coords = pool_coords[sel_idx]

    # Print C-H bond stats for selected frames
    for i, j in CH_PAIRS:
        r = np.linalg.norm(new_coords[:, i] - new_coords[:, j], axis=-1)
        print(f"  C{i}-H{j} bond: {r.min():.4f}–{r.max():.4f} Å  mean={r.mean():.4f}")

    if args.dry_run:
        print("\n[dry-run] Skipping PSI4. Would compute dipoles for the frames above.")
        print("Re-run without --dry-run to collect dipoles.")
        return

    # ── Compute PSI4 dipoles ───────────────────────────────────────────────────
    n_sel = len(sel_idx)
    print(f"\nComputing PSI4 {PSI4_METHOD}/{PSI4_OPTIONS['basis']} dipoles "
          f"for {n_sel} frames…")
    new_dipoles  = np.zeros((n_sel, 3))
    new_energies = np.zeros(n_sel)   # placeholder (PES energies not available from XYZ)
    new_forces   = np.zeros((n_sel, len(all_symbols), 3))

    t0 = time.time()
    n_ok = 0
    for i, c in enumerate(new_coords):
        try:
            d = psi4_dipole(all_symbols, c)
            new_dipoles[i] = d
            n_ok += 1
        except Exception as exc:
            print(f"  Frame {i}: PSI4 failed ({exc})")
        if (i + 1) % 10 == 0 or i == n_sel - 1:
            elapsed = time.time() - t0
            mag = np.linalg.norm(new_dipoles[i])
            print(f"  {i+1}/{n_sel}  |μ|={mag:.4f} D  "
                  f"[{elapsed:.0f}s  {elapsed/(i+1):.1f}s/frame]", flush=True)

    print(f"\n  {n_ok}/{n_sel} frames with valid dipoles")

    # ── Merge and save ─────────────────────────────────────────────────────────
    merged_coords   = np.concatenate([ex_coords,   new_coords],   axis=0)
    merged_energies = np.concatenate([ex_energies, new_energies], axis=0)
    merged_forces   = np.concatenate([ex_forces,   new_forces],   axis=0)
    merged_dipoles  = np.concatenate([ex_dipoles,  new_dipoles],  axis=0)

    if args.output is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = Path(f'outputs/mvko_dipoles_ch_v2_{ts}/training_with_dipoles.npz')
    else:
        out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(out_path,
             symbols=np.array(ex_symbols),
             coordinates=merged_coords,
             energies=merged_energies,
             forces=merged_forces,
             dipoles=merged_dipoles,
             metadata=np.array(json.dumps({
                 'source': 'collect_ch_displaced_dipoles',
                 'n_existing': int(len(ex_coords)),
                 'n_new': int(n_ok),
                 'n_total': int(len(merged_coords)),
                 'method': f'{PSI4_METHOD}/{PSI4_OPTIONS["basis"]}',
                 'traj_dir': str(args.traj_dir),
                 'stride': args.stride,
                 'ch_eq_mean_ang': CH_EQ_MEAN,
             })))
    print(f"\nSaved {len(merged_coords)} frames → {out_path}")
    mags = np.linalg.norm(merged_dipoles, axis=-1)
    print(f"  |μ| range: {mags.min():.3f}–{mags.max():.3f} D  mean={mags.mean():.3f}")
    print(f"\nRun IR spectrum with the new dipole data:")
    print(f"  python3 ir_md_spectrum.py \\")
    print(f"    --nm-pes-model outputs/wB97X_nm_model_v2/nm_pes_model.pkl \\")
    print(f"    --training-data {out_path} \\")
    print(f"    --steps 30000 --temp 300 --preminimize \\")
    print(f"    --zpe-min-freq 50 --zpe-max-freq 4000 \\")
    print(f"    --n-trajectories 5 \\")
    print(f"    --nm-pes-bond-wall-factor 1.15 --nm-pes-bond-wall-stiffness 15.0 \\")
    print(f"    --max-bond-extension 1.35 \\")
    print(f"    --output-dir outputs/ir_spectrum_NM_PES_ch_dipoles_300K")


if __name__ == '__main__':
    main()
