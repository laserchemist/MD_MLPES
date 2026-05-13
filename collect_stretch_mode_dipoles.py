"""
Collect PSI4 B3LYP/6-31G* dipoles for geometries displaced along specific
NM eigenvectors extracted from the NM-PES model.

Targets the O-O and C-O stretch modes (985–1108 cm⁻¹) that correspond to
experimental MVKO bands B4–B6 (Chung & Lee 2021 Fig. 3).  Unlike MD-frame
selection, displacing along eigenvectors guarantees coverage of the PURE
stretch character needed for ∂μ/∂q sensitivity in the dipole KRR.

Usage:
    python3 collect_stretch_mode_dipoles.py \
        --nm-model  outputs/wB97X_nm_model_v2/mlpes_wB97X_nm.pkl \
        --existing  outputs/mvko_dipoles_ch_v2_20260420/training_with_dipoles.npz \
        --mode-indices "10 11 12 13 14 15 16" \
        --amplitudes "-3 -2 -1 1 2 3" \
        --output    outputs/mvko_dipoles_stretch_YYYYMMDD/training_with_dipoles.npz
    # Dry-run:
        python3 collect_stretch_mode_dipoles.py --dry-run ...
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import psi4
    PSI4_AVAILABLE = True
    print(f"PSI4 {psi4.__version__} available")
except ImportError:
    PSI4_AVAILABLE = False
    print("PSI4 not found — use --dry-run")

PSI4_METHOD  = 'b3lyp'
PSI4_OPTIONS = {'basis': '6-31G*', 'scf_type': 'df', 'reference': 'rhf',
                'maxiter': 200, 'e_convergence': 1e-7, 'd_convergence': 1e-7}
PSI4_MEM_GB  = 4
PSI4_THREADS = 4
AU_TO_DEBYE  = 2.541746
ANG_TO_BOHR  = 1.8897259886


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


def load_nm_model(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def make_displaced_geometry(model, mode_idx, amplitude):
    """
    Return Cartesian coordinates (n_atoms, 3) displaced along NM eigenvector
    mode_idx by `amplitude` × coord_scale[mode_idx] in NM coordinate space.

    Conversion:
        q_k  = amplitude × coord_scale[k]          (physical NM coord, √amu·Bohr)
        dr_mw = U_vib[:,k] × q_k                   (mass-weighted Cartesian, √amu·Bohr)
        dr_ang = dr_mw / sqrt_mass / ANG_TO_BOHR   (Cartesian Angstrom)
        coords = eq_coords + dr_ang.reshape(n_atoms, 3)
    """
    eq   = model['eq_coords_ang']         # (n_atoms, 3)
    U    = model['U_vib']                 # (3N, n_vib)
    sqm  = model['sqrt_mass']             # (3N,)
    cs   = model['coord_scale']           # (n_vib,)  thermal amplitudes in √amu·Bohr

    q_phys = amplitude * cs[mode_idx]    # physical NM amplitude (√amu·Bohr)
    dr_mw  = U[:, mode_idx] * q_phys    # (3N,) mass-weighted Cartesian
    dr_ang = dr_mw / sqm / ANG_TO_BOHR  # (3N,) Angstrom
    return eq + dr_ang.reshape(eq.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nm-model',  required=True,
                        help='Path to NM-PES pkl (e.g. outputs/wB97X_nm_model_v2/mlpes_wB97X_nm.pkl)')
    parser.add_argument('--existing',  required=False, default=None,
                        help='Existing dipole npz to merge with (omit to start fresh)')
    parser.add_argument('--mode-indices', default='10 11 12 13 14 15',
                        help='0-indexed mode numbers to displace (default: modes 11-16, 985-1108 cm-1)')
    parser.add_argument('--amplitudes', default='-3 -2 -1 1 2 3',
                        help='Displacement amplitudes in units of coord_scale (default: ±1,2,3)')
    parser.add_argument('--output',    default=None)
    parser.add_argument('--dry-run',   action='store_true')
    args = parser.parse_args()

    if not args.dry_run and not PSI4_AVAILABLE:
        sys.exit("PSI4 required (or use --dry-run)")

    mode_indices = [int(x) for x in args.mode_indices.split()]
    amplitudes   = [float(x) for x in args.amplitudes.split()]

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading NM-PES model from {args.nm_model}")
    model = load_nm_model(args.nm_model)
    freqs  = model['freqs_vib']
    symbols = model['symbols']
    print(f"  Target modes (0-indexed → 1-indexed):")
    for k in mode_indices:
        print(f"    Mode {k+1:2d}: {freqs[k]:8.1f} cm-1  "
              f"(coord_scale = {model['coord_scale'][k]:.4f} √amu·Bohr)")

    # ── Load existing dipoles ─────────────────────────────────────────────────
    if args.existing is not None:
        print(f"\nLoading existing dipoles from {args.existing}")
        ex = np.load(args.existing, allow_pickle=True)
        ex_coords   = ex['coordinates']
        ex_energies = ex['energies']
        ex_forces   = ex['forces']
        ex_dipoles  = ex['dipoles']
        ex_symbols  = ex['symbols'].tolist()
        print(f"  {len(ex_coords)} existing frames")
    else:
        print("\nStarting fresh — no existing dipole dataset")
        n_atoms     = len(model['symbols'])
        ex_coords   = np.zeros((0, n_atoms, 3))
        ex_energies = np.zeros(0)
        ex_forces   = np.zeros((0, n_atoms, 3))
        ex_dipoles  = np.zeros((0, 3))
        ex_symbols  = model['symbols']

    # ── Generate displaced geometries ─────────────────────────────────────────
    new_coords = []
    labels = []
    for k in mode_indices:
        for amp in amplitudes:
            c = make_displaced_geometry(model, k, amp)
            # Sanity check — no BONDED pair shorter than 0.7 Å or longer than 2.5 Å
            # Use simple distance threshold: pairs closer than 1.8 Å at eq are bonded
            ok = True
            eq_c = model['eq_coords_ang']
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    d_eq = np.linalg.norm(eq_c[i]-eq_c[j])
                    if d_eq < 1.8:   # bonded pair
                        d = np.linalg.norm(c[i]-c[j])
                        if d < 0.7 or d > 2.5:
                            ok = False
                            break
            if ok:
                new_coords.append(c)
                labels.append((k+1, amp))
    new_coords = np.array(new_coords)
    print(f"\nGenerated {len(new_coords)} displaced geometries "
          f"({len(mode_indices)} modes × {len(amplitudes)} amplitudes, "
          f"{len(mode_indices)*len(amplitudes)-len(new_coords)} rejected for bad geometry)")

    # Show bond lengths for key stretch modes
    OO_PAIR, CO_PAIR = (1, 2), (0, 1)
    print("\n  Key bond lengths in displaced frames:")
    for (midx, amp), c in zip(labels, new_coords):
        oo = np.linalg.norm(c[OO_PAIR[0]] - c[OO_PAIR[1]])
        co = np.linalg.norm(c[CO_PAIR[0]] - c[CO_PAIR[1]])
        print(f"    Mode {midx:2d}  amp={amp:+.0f}   O-O={oo:.4f} Å  C-O={co:.4f} Å")

    if args.dry_run:
        print("\n[dry-run] Skipping PSI4.")
        return

    # ── Compute PSI4 dipoles ──────────────────────────────────────────────────
    n_new = len(new_coords)
    print(f"\nComputing PSI4 {PSI4_METHOD}/{PSI4_OPTIONS['basis']} dipoles "
          f"for {n_new} frames…")
    new_dipoles  = np.zeros((n_new, 3))
    new_energies = np.zeros(n_new)
    new_forces   = np.zeros((n_new, len(symbols), 3))

    t0 = time.time()
    n_ok = 0
    for i, c in enumerate(new_coords):
        k, amp = labels[i]
        try:
            d = psi4_dipole(symbols, c)
            new_dipoles[i] = d
            n_ok += 1
        except Exception as exc:
            print(f"  Frame {i} (Mode {k}, amp={amp:+.0f}): PSI4 failed ({exc})")
        if (i+1) % 6 == 0 or i == n_new-1:
            elapsed = time.time()-t0
            mag = np.linalg.norm(new_dipoles[i])
            print(f"  {i+1}/{n_new}  Mode {k} amp={amp:+.0f}  |μ|={mag:.4f} D  "
                  f"[{elapsed:.0f}s  {elapsed/(i+1):.1f}s/frame]", flush=True)

    print(f"\n  {n_ok}/{n_new} frames with valid dipoles")

    # ── Merge and save ────────────────────────────────────────────────────────
    merged_coords   = np.concatenate([ex_coords,   new_coords],   axis=0)
    merged_energies = np.concatenate([ex_energies, new_energies], axis=0)
    merged_forces   = np.concatenate([ex_forces,   new_forces],   axis=0)
    merged_dipoles  = np.concatenate([ex_dipoles,  new_dipoles],  axis=0)

    if args.output is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = Path(f'outputs/mvko_dipoles_stretch_{ts}/training_with_dipoles.npz')
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
                 'source': 'collect_stretch_mode_dipoles',
                 'n_existing': int(len(ex_coords)),
                 'n_new': int(n_ok),
                 'n_total': int(len(merged_coords)),
                 'method': f'{PSI4_METHOD}/{PSI4_OPTIONS["basis"]}',
                 'mode_indices': mode_indices,
                 'amplitudes': amplitudes,
                 'target_freqs_cm1': [float(freqs[k]) for k in mode_indices],
             })))
    print(f"\nSaved {len(merged_coords)} frames → {out_path}")
    mags = np.linalg.norm(merged_dipoles, axis=-1)
    print(f"  |μ| range: {mags.min():.3f}–{mags.max():.3f} D  mean={mags.mean():.3f}")
    print(f"\nRun IR with:")
    print(f"  python3 ir_md_spectrum.py \\")
    print(f"    --nm-pes-model outputs/wB97X_nm_model_v2/mlpes_wB97X_nm.pkl \\")
    print(f"    --training-data {out_path} \\")
    print(f"    --steps 30000 --temp 300 --preminimize \\")
    print(f"    --zpe-min-freq 50 --zpe-max-freq 4000 \\")
    print(f"    --n-trajectories 5 \\")
    print(f"    --nm-pes-bond-wall-factor 1.15 --nm-pes-bond-wall-stiffness 15.0 \\")
    print(f"    --max-bond-extension 1.35 \\")
    print(f"    --output-dir outputs/ir_spectrum_NM_PES_stretch_dipoles_300K")


if __name__ == '__main__':
    main()
