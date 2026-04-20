#!/usr/bin/env python3
"""
generate_ch_adaptive_training.py — Adaptive near-equilibrium sampling for C-H stretch modes.

Why this script exists
----------------------
MACE trained on the wB97X augmented dataset (1126 frames) produces C-H normal-mode
frequencies of ~2400–2605 cm⁻¹ at the tightly-minimized geometry, vs. the PSI4
harmonic values of 3049–3323 cm⁻¹. Root cause: the training data has large-amplitude
C-H displacements from the CASSCF NM grid, but is sparse in the small-to-intermediate
amplitude range (0.05–0.25 Å) that determines the curvature (and hence frequency) of
the C-H potential wells.

Strategy
--------
1. Load PSI4 Hessian → identify C-H stretch modes (freq > CH_FREQ_CUTOFF cm⁻¹).
2. Load wB97X-D equilibrium geometry (min-energy training frame).
3. Generate ±displacements along ONLY the C-H modes at three amplitude scales:
     • fine   scale: T_nm = 300 K,  n_amplitudes = 6  (near-equilibrium curvature)
     • medium scale: T_nm = 1000 K, n_amplitudes = 8  (mid-range anharmonicity)
     • coarse scale: T_nm = 2000 K, n_amplitudes = 4  (fill gap to existing grid)
   Frames already present in the training data (RMSD < DEDUP_RMSD Å) are skipped.
4. Run PSI4 wB97X-D3/6-31G* gradient + dipole on each new geometry.
5. Merge with existing training data → save augmented npz.
6. (Optional) Optionally retrain MACE immediately with --retrain flag.

Usage
-----
    python3 generate_ch_adaptive_training.py \\
        --training-data outputs/wB97X_surface_20260406_223155/training_data_wB97X_aug.npz \\
        --hessian       outputs/mvko_20260319_074852/psi4_hessian.npy \\
        --eq-coords     outputs/mvko_20260319_081314/psi4_eq_coords.npy \\
        --output-dir    outputs/ch_adaptive_20260418

    # After generating data, retrain MACE in one shot:
    python3 generate_ch_adaptive_training.py ... --retrain \\
        --mace-output-dir outputs/mace_wB97X_ch_adaptive_20260418

Output
------
    <output-dir>/
        ch_adaptive_frames.npz           — new frames only (energy/forces/dipoles)
        training_data_ch_aug.npz         — merged: original + new
        run_log.txt                      — per-frame PSI4 results
"""

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'modules'))

logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Physical constants
ANGSTROM_TO_BOHR = 1.88972612456
AU_TO_DEBYE      = 2.541746

# C-H stretch mode selection threshold (cm⁻¹)
CH_FREQ_CUTOFF = 2800.0

# Skip frames within this RMSD of any existing training frame
DEDUP_RMSD = 0.03   # Angstrom


# =============================================================================
# PSI4 single-point (wB97X-D3 / 6-31G*)
# =============================================================================

def psi4_single_point(symbols, coords, method='wB97X-D', basis='6-31G*'):
    """
    Compute energy (Ha), forces (Ha/Å), dipole (Debye) for one geometry.
    Returns (energy, forces, dipole, error_str_or_None).
    """
    mol_str = "0 1\n"
    for s, c in zip(symbols, coords):
        mol_str += f"{s}  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}\n"
    mol_str += "units angstrom\nno_reorient\nno_com"

    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.be_quiet()
    psi4.set_memory('4 GB')
    psi4.set_num_threads(4)
    psi4.set_options({
        'basis':         basis,
        'scf_type':      'df',
        'reference':     'rhf',
        'maxiter':       200,
        'e_convergence': 1e-7,
        'd_convergence': 1e-7,
    })

    try:
        mol       = psi4.geometry(mol_str)
        grad_mat, wfn = psi4.gradient(f'{method}/{basis}', molecule=mol, return_wfn=True)

        energy    = float(wfn.energy())
        n_atoms   = len(symbols)
        grad_bohr = np.array([[grad_mat.get(i, j) for j in range(3)]
                               for i in range(n_atoms)])
        forces    = -grad_bohr / ANGSTROM_TO_BOHR

        dipole = np.zeros(3)
        try:
            psi4.oeprop(wfn, 'DIPOLE')
            dipole = np.array([
                psi4.variable('DIPOLE X') * AU_TO_DEBYE,
                psi4.variable('DIPOLE Y') * AU_TO_DEBYE,
                psi4.variable('DIPOLE Z') * AU_TO_DEBYE,
            ])
        except Exception:
            pass

        return energy, forces, dipole, None

    except Exception as e:
        return None, None, np.zeros(3), str(e)


# =============================================================================
# Displacement generation
# =============================================================================

def generate_ch_displacements(symbols, coords_eq, hessian,
                               ch_freq_cutoff=CH_FREQ_CUTOFF):
    """
    Generate ±NM displacements along C-H stretch modes only, at three scales.

    Returns
    -------
    displacements : list of (coords [Å, (N,3)], mode_idx, freq_cmiv, factor, T_scale)
    ch_mode_indices : list of vibrational mode indices (0-based) used
    ch_freqs : corresponding frequencies in cm⁻¹
    """
    from normal_modes import compute_normal_modes, thermal_amplitude_angstrom
    from normal_modes import KB_HARTREE_PER_K, BOHR_TO_ANGSTROM

    freqs, eigvecs_mw, eigenvalues, mass_vec_3N = compute_normal_modes(symbols, hessian)

    n_atoms = len(symbols)

    # Select C-H stretch modes
    ch_indices = [i for i, f in enumerate(freqs) if f >= ch_freq_cutoff]
    ch_freqs   = [freqs[i] for i in ch_indices]
    print(f"  C-H stretch modes ({len(ch_indices)}): "
          + ", ".join(f"mode {i+1} @ {freqs[i]:.0f} cm⁻¹" for i in ch_indices))

    # Three amplitude scales.
    # dedup=False on fine scale: small-amplitude displacements are the whole
    # point of this script; don't skip them just because an MD frame happened
    # to land nearby in absolute coordinate space.
    scales = [
        dict(T=300,   n=8,  max_f=1.0,  label='fine',   dedup=False),
        dict(T=1000,  n=8,  max_f=1.5,  label='medium',  dedup=True),
        dict(T=2000,  n=4,  max_f=2.0,  label='coarse',  dedup=True),
    ]

    displacements = []
    for mode_idx in ch_indices:
        ev   = eigenvalues[mode_idx]
        if ev <= 0:
            continue
        L_mw = eigvecs_mw[:, mode_idx]          # mass-weighted eigvec (3N,)

        for sc in scales:
            kT    = sc['T'] * KB_HARTREE_PER_K
            Q_cl  = np.sqrt(2.0 * kT / ev)          # Bohr√amu
            dr_bohr = Q_cl * L_mw / np.sqrt(mass_vec_3N)  # (3N,) Bohr
            dr_ang  = (dr_bohr * BOHR_TO_ANGSTROM).reshape(n_atoms, 3)

            factors = np.linspace(sc['max_f'] / sc['n'], sc['max_f'], sc['n'])
            for fac in factors:
                for sign in (+1, -1):
                    disp_coords = coords_eq + sign * fac * dr_ang
                    displacements.append((
                        disp_coords, mode_idx, freqs[mode_idx],
                        sign * fac, sc['label'], sc['dedup']
                    ))

    print(f"  Total displacements generated: {len(displacements)}")
    return displacements, ch_indices, ch_freqs


def _atom_mass(symbol):
    """Return atomic mass in amu."""
    masses = {'H': 1.00794, 'C': 12.011, 'N': 14.007, 'O': 15.999,
              'F': 18.998, 'S': 32.06,  'Cl': 35.45}
    return masses.get(symbol, 12.0)


# =============================================================================
# Deduplication
# =============================================================================

def is_duplicate(coords_new, coords_existing, tol=DEDUP_RMSD):
    """
    Return True if coords_new is within tol Å of any existing frame.

    Uses max per-atom displacement rather than RMSD — this correctly handles
    C-H stretch displacements which move a single H atom by ~0.05 Å while the
    overall RMSD (averaged over all 12 atoms) would be only ~0.01 Å and would
    incorrectly flag distinct geometries as duplicates.
    """
    for ref in coords_existing:
        max_disp = np.max(np.linalg.norm(coords_new - ref, axis=1))  # max over atoms
        if max_disp < tol:
            return True
    return False


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Adaptive C-H stretch sampling for MACE retraining',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--training-data', required=True,
                        help='Existing npz training data (will be augmented)')
    parser.add_argument('--hessian',
                        default='outputs/mvko_20260319_074852/psi4_hessian.npy',
                        help='PSI4 Hessian .npy file (36×36 for 12-atom molecule)')
    parser.add_argument('--eq-coords',
                        default=None,
                        help='Equilibrium coords .npy (default: min-energy frame in training-data)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (auto-timestamped if omitted)')
    parser.add_argument('--method', default='wB97X-D',
                        help='PSI4 method')
    parser.add_argument('--basis', default='6-31G*',
                        help='PSI4 basis set')
    parser.add_argument('--ch-freq-cutoff', type=float, default=CH_FREQ_CUTOFF,
                        help='Frequency threshold (cm⁻¹) to identify C-H stretch modes')
    parser.add_argument('--energy-cutoff', type=float, default=80.0,
                        help='Skip frames with ΔE > this value above training minimum (kcal/mol)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Generate and list displacements without running PSI4')
    parser.add_argument('--retrain', action='store_true',
                        help='Retrain MACE after generating data')
    parser.add_argument('--mace-output-dir', default=None,
                        help='MACE output directory when --retrain is set')
    parser.add_argument('--n-train', type=int, default=None,
                        help='n_train for MACE retraining (default: all - n_valid)')
    parser.add_argument('--n-valid', type=int, default=80,
                        help='n_valid for MACE retraining')
    parser.add_argument('--epochs', type=int, default=500,
                        help='MACE epochs for retraining')
    args = parser.parse_args()

    ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path(args.output_dir) if args.output_dir else Path('outputs') / f'ch_adaptive_{ts}'
    out.mkdir(parents=True, exist_ok=True)

    # ── PSI4 ──────────────────────────────────────────────────────────────────
    if not args.dry_run:
        global psi4
        try:
            import psi4
            print(f"PSI4 {psi4.__version__} available")
        except ImportError:
            print("ERROR: PSI4 not available. Use --dry-run to preview displacements.")
            sys.exit(1)

    # ── Load existing training data ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ADAPTIVE C-H TRAINING DATA GENERATION")
    print(f"{'='*60}")
    print(f"  Training data : {args.training_data}")
    print(f"  Output dir    : {out}")

    data    = np.load(args.training_data, allow_pickle=True)
    symbols = list(data['symbols'])
    coords_all  = data['coordinates']           # (N_frames, n_atoms, 3)
    energies_all = data['energies']             # (N_frames,) Ha
    forces_all   = data['forces']
    has_dipoles  = 'dipoles' in data
    dipoles_all  = data['dipoles'] if has_dipoles else None

    E_min = float(np.min(energies_all))
    print(f"  Frames        : {len(coords_all)}")
    print(f"  E_min         : {E_min:.6f} Ha  ({len(symbols)}-atom molecule)")

    # ── Equilibrium geometry ───────────────────────────────────────────────────
    if args.eq_coords:
        coords_eq = np.load(args.eq_coords)
        print(f"  Eq. coords    : {args.eq_coords}")
    else:
        idx_min   = int(np.argmin(energies_all))
        coords_eq = coords_all[idx_min]
        print(f"  Eq. coords    : min-energy training frame {idx_min}  "
              f"(E={energies_all[idx_min]:.6f} Ha)")

    # ── Load Hessian + identify C-H modes ─────────────────────────────────────
    print(f"\n  Loading Hessian from {args.hessian}")
    hessian = np.load(args.hessian)
    print(f"  Hessian shape : {hessian.shape}")

    print("\n  Identifying C-H stretch modes ...")
    displacements, ch_indices, ch_freqs = generate_ch_displacements(
        symbols, coords_eq, hessian, ch_freq_cutoff=args.ch_freq_cutoff
    )

    # ── Deduplicate against existing frames ───────────────────────────────────
    print(f"\n  Deduplicating against {len(coords_all)} existing frames "
          f"(max per-atom displacement < {DEDUP_RMSD} Å; fine scale exempt) ...")
    new_disps = []
    for item in displacements:
        c, mode_idx, freq, fac, label, do_dedup = item
        if do_dedup and is_duplicate(c, coords_all, tol=DEDUP_RMSD):
            continue
        new_disps.append(item)
    print(f"  After dedup   : {len(new_disps)} / {len(displacements)} frames kept")

    if args.dry_run:
        print("\n  [DRY RUN] Displacement summary:")
        for i, (c, mode_idx, freq, fac, label, _) in enumerate(new_disps[:20]):
            print(f"    {i+1:3d}  mode {mode_idx+1}  {freq:.0f} cm⁻¹  "
                  f"factor={fac:+.3f}  scale={label}")
        if len(new_disps) > 20:
            print(f"    ... ({len(new_disps)-20} more)")
        print(f"\n  Estimated PSI4 time: ~{len(new_disps)*2:.0f}–{len(new_disps)*4:.0f} min "
              f"(2–4 min/frame on 4 threads)")
        return

    # ── Run PSI4 single-points ─────────────────────────────────────────────────
    HARTREE_TO_KCAL = 627.509474
    energy_tol = args.energy_cutoff / HARTREE_TO_KCAL   # Ha

    new_coords  = []
    new_energies = []
    new_forces  = []
    new_dipoles = []
    log_lines   = ["frame, mode, freq_cmiv, factor, scale, energy_Ha, dE_kcal, status"]

    print(f"\n  Running PSI4 {args.method}/{args.basis} single-points ...")
    print(f"  Energy cutoff : {args.energy_cutoff:.0f} kcal/mol above minimum\n")

    t0 = time.time()
    n_ok = n_skip_E = n_fail = 0

    for i, (c, mode_idx, freq, fac, label, _) in enumerate(new_disps):
        elapsed = time.time() - t0
        rate    = elapsed / max(i, 1)
        eta_min = rate * (len(new_disps) - i) / 60
        print(f"  [{i+1:3d}/{len(new_disps)}]  mode {mode_idx+1} ({freq:.0f} cm⁻¹)  "
              f"factor={fac:+.3f}  scale={label}  ETA {eta_min:.0f} min", end='  ')

        energy, forces, dipole, err = psi4_single_point(
            symbols, c, method=args.method, basis=args.basis
        )

        if err is not None:
            print(f"FAIL: {err[:60]}")
            log_lines.append(f"{i+1}, {mode_idx+1}, {freq:.0f}, {fac:+.4f}, {label}, "
                             f"None, None, FAIL: {err[:40]}")
            n_fail += 1
            continue

        dE = (energy - E_min) * HARTREE_TO_KCAL
        if dE > args.energy_cutoff:
            print(f"SKIP (ΔE={dE:.0f} kcal/mol > cutoff)")
            log_lines.append(f"{i+1}, {mode_idx+1}, {freq:.0f}, {fac:+.4f}, {label}, "
                             f"{energy:.8f}, {dE:.1f}, SKIP_ENERGY")
            n_skip_E += 1
            continue

        print(f"OK  ΔE={dE:.1f} kcal/mol")
        log_lines.append(f"{i+1}, {mode_idx+1}, {freq:.0f}, {fac:+.4f}, {label}, "
                         f"{energy:.8f}, {dE:.1f}, OK")
        new_coords.append(c)
        new_energies.append(energy)
        new_forces.append(forces)
        new_dipoles.append(dipole)
        n_ok += 1

    # ── Save new frames ────────────────────────────────────────────────────────
    print(f"\n  PSI4 done: {n_ok} OK, {n_skip_E} energy-skipped, {n_fail} failed")

    (out / 'run_log.txt').write_text('\n'.join(log_lines) + '\n')

    if n_ok == 0:
        print("  ERROR: No frames collected — nothing to save.")
        sys.exit(1)

    new_coords_arr   = np.array(new_coords)
    new_energies_arr = np.array(new_energies)
    new_forces_arr   = np.array(new_forces)
    new_dipoles_arr  = np.array(new_dipoles)

    np.savez(out / 'ch_adaptive_frames.npz',
             symbols=np.array(symbols),
             coordinates=new_coords_arr,
             energies=new_energies_arr,
             forces=new_forces_arr,
             dipoles=new_dipoles_arr)
    print(f"  New frames    : {out / 'ch_adaptive_frames.npz'}  ({n_ok} frames)")

    # ── Merge with existing ────────────────────────────────────────────────────
    merged_coords   = np.concatenate([coords_all, new_coords_arr],  axis=0)
    merged_energies = np.concatenate([energies_all, new_energies_arr])
    merged_forces   = np.concatenate([forces_all, new_forces_arr],  axis=0)

    if has_dipoles and dipoles_all is not None:
        merged_dipoles = np.concatenate([dipoles_all, new_dipoles_arr], axis=0)
    else:
        merged_dipoles = new_dipoles_arr  # only new frames have dipoles

    aug_path = out / 'training_data_ch_aug.npz'
    np.savez(aug_path,
             symbols=np.array(symbols),
             coordinates=merged_coords,
             energies=merged_energies,
             forces=merged_forces,
             dipoles=merged_dipoles)
    print(f"  Merged data   : {aug_path}  ({len(merged_coords)} frames total)")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        'date':          datetime.datetime.now().isoformat(),
        'training_data': args.training_data,
        'hessian':       args.hessian,
        'method':        args.method,
        'basis':         args.basis,
        'ch_modes':      [int(i+1) for i in ch_indices],
        'ch_freqs':      [float(f) for f in ch_freqs],
        'n_generated':   len(new_disps),
        'n_ok':          n_ok,
        'n_skip_energy': n_skip_E,
        'n_failed':      n_fail,
        'n_original':    len(coords_all),
        'n_merged':      len(merged_coords),
        'ch_adaptive_frames': str(out / 'ch_adaptive_frames.npz'),
        'merged_data':   str(aug_path),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f"  Summary       : {out / 'summary.json'}")

    print(f"\n{'='*60}")
    print("  NEXT STEP — retrain MACE:")
    print(f"{'='*60}")
    mace_out = args.mace_output_dir or f"outputs/mace_wB97X_ch_adaptive_{ts}"
    print(f"\n  python3 train_mace_model.py \\")
    print(f"      --training-data {aug_path} \\")
    print(f"      --output-dir    {mace_out} \\")
    print(f"      --n-train {args.n_train or 'all'} --n-valid {args.n_valid} "
          f"--epochs {args.epochs} \\")
    print(f"      --energy-cutoff 50 --device cpu")

    # ── Optional auto-retrain ──────────────────────────────────────────────────
    if args.retrain:
        import subprocess
        cmd = [
            'python3', 'train_mace_model.py',
            f'--training-data={aug_path}',
            f'--output-dir={mace_out}',
            f'--n-valid={args.n_valid}',
            f'--epochs={args.epochs}',
            '--energy-cutoff=50',
            '--device=cpu',
        ]
        if args.n_train:
            cmd.append(f'--n-train={args.n_train}')
        print(f"\n  Auto-retraining MACE ...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n  ERROR: MACE retraining failed (exit {result.returncode})")
            sys.exit(result.returncode)


if __name__ == '__main__':
    main()
