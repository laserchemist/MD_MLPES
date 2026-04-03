#!/usr/bin/env python3
"""
Generate C-H stretch training data by displacing only the high-frequency
C-H stretch normal modes (>= min_freq_cm1) at large amplitudes.

Replaces the T=8000K full-mode run that hung on soft bending modes — those
have a_thermal >> 1 Ang at 8000K and cause PSI4 SCF to not converge.

Strategy
--------
  1. Load existing Hessian from a pre-computed .npy file (avoids recomputing)
     OR compute a new one from PSI4.
  2. Select only modes with frequency >= min_freq_cm1 (default 2500 cm-1,
     capturing the 6 C-H stretch modes at 3049-3323 cm-1).
  3. Displace +-1 to +-n_amplitudes * a_thermal(T_nm) along each selected mode.
  4. Run PSI4 B3LYP/6-31G* single-point at each displaced geometry.
  5. Save as nm_displacements.npz (same format as generate_nm_training.py).

Usage
-----
    python3 generate_ch_stretch_training.py \\
        --training-data outputs/mvko_20260319_081314/combined_training_data.npz \\
        --hessian       outputs/casscf_nm_delta_20260401_110049/hessian_used.npy \\
        --T-nm 8000 --n-amplitudes 8 --max-factor 10 \\
        --min-freq 2500
"""

import sys, argparse, logging, time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'modules'))

logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')

try:
    import psi4
    print(f"PSI4 {psi4.__version__} available")
except ImportError:
    print("PSI4 not available"); sys.exit(1)

from normal_modes import (
    compute_hessian_psi4, compute_normal_modes,
    KB_HARTREE_PER_K, BOHR_TO_ANGSTROM, FREQ_CONV,
)
from data_formats import TrajectoryData, load_trajectory

HARTREE_TO_KCAL  = 627.509474
ANGSTROM_TO_BOHR = 1.88972612456
AU_TO_DEBYE      = 2.541746


def psi4_single_point(symbols, coords, method='B3LYP', basis='6-31G*'):
    mol_str = "0 1\n"
    for s, c in zip(symbols, coords):
        mol_str += f"{s}  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}\n"
    mol_str += "units angstrom\nno_reorient\nno_com"

    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.be_quiet()
    psi4.set_memory('2 GB')
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
        mol = psi4.geometry(mol_str)
        grad_mat, wfn = psi4.gradient(f'{method}/{basis}', molecule=mol, return_wfn=True)
        energy = float(wfn.energy())
        n = len(symbols)
        grad = np.array([[grad_mat.get(i, j) for j in range(3)] for i in range(n)])
        forces = -grad / ANGSTROM_TO_BOHR

        dipole = np.zeros(3)
        try:
            psi4.oeprop(wfn, 'DIPOLE')
            for k, ax in enumerate('XYZ'):
                for key in (f'SCF DIPOLE {ax}', f'DIPOLE {ax}'):
                    try:
                        dipole[k] = psi4.variable(key) * AU_TO_DEBYE; break
                    except Exception:
                        pass
        except Exception:
            pass

        return energy, forces, dipole, None

    except Exception as e:
        return None, None, np.zeros(3), str(e)


def main():
    parser = argparse.ArgumentParser(description='C-H stretch NM displacement training data')
    parser.add_argument('--training-data', required=True)
    parser.add_argument('--hessian',       default=None,
                        help='Pre-computed Hessian .npy (skips PSI4 Hessian if provided)')
    parser.add_argument('--method',  default='B3LYP')
    parser.add_argument('--basis',   default='6-31G*')
    parser.add_argument('--T-nm',    type=float, default=8000.0)
    parser.add_argument('--n-amplitudes', type=int,   default=8)
    parser.add_argument('--max-factor',   type=float, default=10.0)
    parser.add_argument('--min-freq',     type=float, default=2500.0,
                        help='Only displace modes >= this frequency (cm-1)')
    args = parser.parse_args()

    # ── Load equilibrium geometry ─────────────────────────────────────────────
    traj = load_trajectory(args.training_data)
    symbols   = list(traj.symbols)
    coords_eq = traj.coordinates[0]
    print(f"Molecule: {symbols}  ({len(symbols)} atoms)")
    print(f"Equilibrium geometry from frame 0 of {args.training_data}")

    # ── Hessian ───────────────────────────────────────────────────────────────
    if args.hessian:
        print(f"\nLoading Hessian from {args.hessian}")
        hessian = np.load(args.hessian)
    else:
        print(f"\nComputing PSI4 Hessian ({args.method}/{args.basis}) ...")
        t0 = time.time()
        hessian = compute_hessian_psi4(symbols, coords_eq, args.method, args.basis)
        print(f"  Done in {time.time()-t0:.1f} s")

    # ── Normal modes ──────────────────────────────────────────────────────────
    frequencies, eigvecs_mw, eigenvalues, mass_vec = compute_normal_modes(symbols, hessian)
    n_vib = len(frequencies)
    print(f"\nNormal modes ({n_vib} vibrational):")
    for i, f in enumerate(frequencies):
        tag = " <-- C-H" if f >= args.min_freq else ""
        print(f"  Mode {i:2d}: {f:8.1f} cm-1{tag}")

    # ── Select high-frequency modes ───────────────────────────────────────────
    selected = [i for i, f in enumerate(frequencies) if f >= args.min_freq]
    print(f"\nSelected {len(selected)} modes >= {args.min_freq:.0f} cm-1: "
          f"modes {selected}")

    # ── Generate displacements ────────────────────────────────────────────────
    T      = args.T_nm
    kT     = T * KB_HARTREE_PER_K
    factors = np.linspace(args.max_factor / args.n_amplitudes,
                          args.max_factor, args.n_amplitudes)
    print(f"\nAmplitude factors: {[round(f,2) for f in factors]}")

    displacements = []   # (coords, mode_idx, factor, freq)
    for mode_idx in selected:
        ev   = eigenvalues[mode_idx]
        freq = frequencies[mode_idx]
        if ev <= 0:
            continue
        Q_cl   = np.sqrt(2.0 * kT / ev)
        dr_ang = (Q_cl * eigvecs_mw[:, mode_idx] / np.sqrt(mass_vec)
                  * BOHR_TO_ANGSTROM).reshape(len(symbols), 3)
        a_therm = np.linalg.norm(dr_ang)
        print(f"  Mode {mode_idx} ({freq:.1f} cm-1): "
              f"a_thermal = {a_therm:.4f} Ang, "
              f"max = {args.max_factor * a_therm:.3f} Ang")
        for fac in factors:
            displacements.append((coords_eq + fac * dr_ang,  mode_idx,  fac, freq))
            displacements.append((coords_eq - fac * dr_ang,  mode_idx, -fac, freq))

    print(f"\nTotal displacements: {len(displacements)} "
          f"({len(selected)} modes × {args.n_amplitudes} amplitudes × 2 signs)")

    # ── PSI4 single-points ────────────────────────────────────────────────────
    print(f"\nRunning PSI4 {args.method}/{args.basis} single-points ...")
    coords_list, energies_list, forces_list, dipoles_list = [], [], [], []
    n_failed = 0

    for k, (coords_d, mode_idx, fac, freq) in enumerate(displacements):
        sign = '+' if fac > 0 else '-'
        print(f"  [{k+1}/{len(displacements)}] mode {mode_idx} "
              f"({freq:.0f} cm-1)  factor={sign}{abs(fac):.2f} ...",
              end=' ', flush=True)
        t0 = time.time()
        energy, forces, dipole, err = psi4_single_point(
            symbols, coords_d, args.method, args.basis)
        dt = time.time() - t0
        if err:
            n_failed += 1
            print(f"FAILED ({dt:.0f}s): {err[:60]}")
            continue
        e_rel = (energy - traj.energies.min()) * HARTREE_TO_KCAL
        print(f"OK  dE={e_rel:+.1f} kcal/mol  ({dt:.0f}s)")
        coords_list.append(coords_d)
        energies_list.append(energy)
        forces_list.append(forces)
        dipoles_list.append(dipole)

    n_ok = len(coords_list)
    print(f"\n{'='*60}")
    print(f"Completed: {n_ok} succeeded, {n_failed} failed "
          f"out of {len(displacements)} total")

    if n_ok == 0:
        print("No successful points — nothing to save."); return

    # ── Save ──────────────────────────────────────────────────────────────────
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path('outputs') / f'nm_ch_stretch_{ts}'
    out.mkdir(parents=True, exist_ok=True)

    result = TrajectoryData(
        symbols     = symbols,
        coordinates = np.array(coords_list),
        energies    = np.array(energies_list),
        forces      = np.array(forces_list),
        dipoles     = np.array(dipoles_list),
        metadata    = {
            'source':       'ch_stretch_nm',
            'method':       args.method,
            'basis':        args.basis,
            'T_nm':         T,
            'n_amplitudes': args.n_amplitudes,
            'max_factor':   args.max_factor,
            'min_freq_cm1': args.min_freq,
            'selected_modes': selected,
        },
    )

    npz_path = out / 'nm_displacements.npz'
    from data_formats import save_trajectory
    save_trajectory(result, str(npz_path))
    print(f"\nSaved {n_ok} frames to {npz_path}")

    # Summary of C-H coverage
    coords_arr = np.array(coords_list)
    c_idx = [i for i, s in enumerate(symbols) if s == 'C']
    h_idx = [i for i, s in enumerate(symbols) if s == 'H']
    ch_dists = []
    for frame in coords_arr:
        for hi in h_idx:
            nn = min(np.linalg.norm(frame[hi] - frame[ci]) for ci in c_idx)
            ch_dists.append(nn)
    ch = np.array(ch_dists)
    dE = (np.array(energies_list) - traj.energies.min()) * HARTREE_TO_KCAL
    print(f"\nC-H bond coverage:")
    print(f"  normal (< 1.3 Ang): {(ch<1.3).sum()} / {len(ch)}")
    print(f"  stretched (1.3-2.0): {((ch>=1.3)&(ch<2.0)).sum()} / {len(ch)}")
    print(f"  dissociated (> 2.0): {(ch>=2.0).sum()} / {len(ch)}")
    print(f"  max C-H: {ch.max():.2f} Ang")
    print(f"\nEnergy coverage:")
    print(f"  < 10 kcal/mol:  {(dE<10).sum()}")
    print(f"  10-100 kcal/mol: {((dE>=10)&(dE<100)).sum()}")
    print(f"  > 100 kcal/mol: {(dE>100).sum()}")
    print(f"  max dE: {dE.max():.1f} kcal/mol")


if __name__ == '__main__':
    main()
