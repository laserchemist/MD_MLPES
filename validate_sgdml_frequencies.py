#!/usr/bin/env python3
"""
validate_sgdml_frequencies.py — Validate sGDML model quality via:
  1. Normal mode frequencies at PSI4 equilibrium (sGDML Hessian vs PSI4 reference)
  2. Short 1000-step MD stability test at 300 K
  3. Energy/force prediction errors on a held-out validation set

Usage
-----
  python3 validate_sgdml_frequencies.py \\
      --model    outputs/sgdml_wB97X_20260417/sgdml_model.pkl \\
      --training-data outputs/wB97X_surface_20260406_223155/training_data_wB97X_aug.npz \\
      --psi4-hessian  outputs/mvko_20260319_081314/nm_displacements.npz  # optional

Output
------
  Prints frequency table (sGDML vs PSI4 if reference provided).
  Saves validate_sgdml_<ts>/validation_report.txt
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# Physical constants
KB_HARTREE_PER_K = 3.1668114e-6
AMU_TO_AU        = 1822.888486
ANGSTROM_TO_BOHR = 1.88972612456
HARTREE_TO_KCAL  = 627.509474
FREQ_CONV        = 5140.48   # sqrt(Ha/(Bohr²·amu)) → cm⁻¹


def compute_normal_modes(driver, coords_eq: np.ndarray) -> tuple:
    """
    Compute mass-weighted Hessian → normal mode frequencies.

    Returns
    -------
    freqs_cmiv : (3N,) frequencies in cm⁻¹ (negative = imaginary)
    eigvecs    : (3N, 3N) mass-weighted eigenvectors
    """
    symbols = driver.symbols
    masses  = driver.masses   # amu
    n_atoms = driver.n_atoms
    n3      = n_atoms * 3

    print("  Computing Hessian via FD on analytic forces (δ=0.005 Å)…")
    H = driver.analytic_hessian(coords_eq, delta=0.005)   # (3N, 3N) Ha/Å²

    # Convert to atomic units: Ha/Å² → Ha/Bohr²
    H_au = H * (ANGSTROM_TO_BOHR ** 2)

    # Mass-weight
    mass_vec = np.repeat(masses, 3)          # (3N,) amu
    inv_sqrt_m = 1.0 / np.sqrt(mass_vec)
    Hmw = (inv_sqrt_m[:, None] * H_au) * inv_sqrt_m[None, :]

    eigenvals, eigvecs = np.linalg.eigh(Hmw)

    # Convert eigenvalues → cm⁻¹
    signs  = np.sign(eigenvals)
    freqs  = signs * np.sqrt(np.abs(eigenvals)) * FREQ_CONV
    return freqs, eigvecs


def run_short_md(driver, coords0: np.ndarray, n_steps: int = 1000,
                 temperature: float = 300.0) -> dict:
    """Run a short MD to check stability (no ZPE init, plain MB)."""
    from modules.bakken import run_md, maxwell_boltzmann_velocities
    print(f"\n  Running {n_steps}-step MD at {temperature} K (stability check)…")
    result = run_md(
        driver, coords0, n_steps=n_steps, temperature=temperature,
        timestep=0.5, save_every=10, preminimize=True,
        preminimize_steps=200, preminimize_tol=0.01,
        max_bond_extension=3.0,
    )
    return result


def validate_predictions(driver, symbols, coords, energies, forces,
                          n_sample: int = 100) -> dict:
    """Compute energy and force errors on a random sample of frames."""
    idx = np.random.choice(len(energies), min(n_sample, len(energies)), replace=False)
    e_pred  = np.array([driver.energy(coords[i]) for i in idx])
    e_ref   = energies[idx]
    f_pred  = np.array([driver.forces(coords[i]) for i in idx])
    f_ref   = forces[idx]

    e_err  = e_pred - e_ref
    e_rmse = float(np.sqrt(np.mean(e_err**2)))
    e_mae  = float(np.mean(np.abs(e_err)))

    f_err  = (f_pred - f_ref).flatten()
    f_rmse = float(np.sqrt(np.mean(f_err**2)))
    f_mae  = float(np.mean(np.abs(f_err)))

    return {
        'n_sample':     len(idx),
        'e_rmse_Ha':    e_rmse,
        'e_rmse_kcal':  e_rmse * HARTREE_TO_KCAL,
        'e_mae_kcal':   e_mae  * HARTREE_TO_KCAL,
        'f_rmse_HaAng': f_rmse,
        'f_mae_HaAng':  f_mae,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Validate sGDML frequencies, MD stability, and prediction errors',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model',          required=True,
                        help='sGDML model .pkl (from train_sgdml_model.py)')
    parser.add_argument('--training-data',  required=True,
                        help='Training data .npz (coordinates, energies, forces, dipoles)')
    parser.add_argument('--psi4-hessian',   default=None,
                        help='Optional: nm_displacements.npz with PSI4 frequencies for comparison')
    parser.add_argument('--output-dir',     default=None,
                        help='Output directory (auto-timestamped if omitted)')
    parser.add_argument('--md-steps',       type=int, default=1000,
                        help='MD stability test steps (default 1000)')
    parser.add_argument('--skip-md',        action='store_true',
                        help='Skip MD stability test (frequencies only)')
    parser.add_argument('--n-validate',     type=int, default=100,
                        help='Frames to sample for error validation (default 100)')
    args = parser.parse_args()

    ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path(args.output_dir) if args.output_dir else \
          Path('outputs') / f'validate_sgdml_{ts}'
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  sGDML VALIDATION")
    print(f"{'='*60}")

    # ── Load model ─────────────────────────────────────────────────────────
    from modules.sgdml_pes import SGDMLDriver
    driver = SGDMLDriver(args.model)
    print(f"  Model         : {args.model}")
    print(f"  Molecule      : {driver.symbols}  ({driver.n_atoms} atoms)")

    # ── Load training data ─────────────────────────────────────────────────
    data     = np.load(args.training_data, allow_pickle=True)
    symbols  = data['symbols'].tolist()
    coords   = data['coordinates']
    energies = data['energies']
    forces   = data['forces']
    print(f"  Training data : {args.training_data}  ({len(energies)} frames)")

    # Starting geometry: lowest-energy training frame
    i_eq = int(np.argmin(energies))
    coords_eq = coords[i_eq]
    print(f"  Equilibrium frame index: {i_eq}  "
          f"(E={energies[i_eq]:.6f} Ha, {(energies[i_eq]-energies.min())*HARTREE_TO_KCAL:.3f} kcal/mol above min)")

    # ── Prediction errors ──────────────────────────────────────────────────
    print(f"\n  Sampling {args.n_validate} frames for energy/force errors…")
    np.random.seed(42)
    err = validate_predictions(driver, symbols, coords, energies, forces,
                               n_sample=args.n_validate)
    print(f"  Energy RMSE : {err['e_rmse_kcal']:.4f} kcal/mol")
    print(f"  Energy MAE  : {err['e_mae_kcal']:.4f} kcal/mol")
    print(f"  Force  RMSE : {err['f_rmse_HaAng']:.5f} Ha/Å  "
          f"({err['f_rmse_HaAng']*HARTREE_TO_KCAL:.3f} kcal/mol/Å)")
    print(f"  Force  MAE  : {err['f_mae_HaAng']:.5f} Ha/Å")

    # ── Normal mode frequencies ────────────────────────────────────────────
    print(f"\n  Normal mode frequencies at lowest-energy frame:")
    freqs, _ = compute_normal_modes(driver, coords_eq)

    n_vib = driver.n_atoms * 3 - 6
    freqs_sorted = sorted(freqs, key=abs)
    trans_rot    = freqs_sorted[:6]
    vib_modes    = freqs_sorted[6:]

    print(f"\n  Translational/rotational (should be ~0):")
    print(f"    {[f'{f:+.1f}' for f in trans_rot]} cm⁻¹")

    n_imaginary = sum(1 for f in vib_modes if f < 0)
    print(f"\n  Vibrational modes ({n_vib} expected, {n_imaginary} imaginary):")
    print(f"  {'Mode':>4}  {'sGDML':>10}", end='')
    if args.psi4_hessian:
        print(f"  {'PSI4':>10}  {'Δ (cm⁻¹)':>10}", end='')
    print()

    psi4_freqs = None
    if args.psi4_hessian:
        try:
            nm = np.load(args.psi4_hessian, allow_pickle=True)
            psi4_freqs = nm['frequencies'] if 'frequencies' in nm else None
            if psi4_freqs is not None:
                psi4_freqs = np.sort(psi4_freqs)[-n_vib:]
                print(f"  (PSI4 reference loaded: {len(psi4_freqs)} modes)")
        except Exception as e:
            print(f"  Warning: could not load PSI4 hessian: {e}")

    vib_sorted = sorted(vib_modes)
    for i, f in enumerate(vib_sorted):
        tag = ' *** IMAGINARY' if f < 0 else ''
        line = f"  {i+1:>4}  {f:>10.1f}"
        if psi4_freqs is not None and i < len(psi4_freqs):
            delta = f - psi4_freqs[i]
            line += f"  {psi4_freqs[i]:>10.1f}  {delta:>+10.1f}"
        print(line + tag)

    # ── MD stability ──────────────────────────────────────────────────────
    md_result = None
    if not args.skip_md:
        md_result = run_short_md(driver, coords_eq, n_steps=args.md_steps)
        e_traj = md_result['energies_ml']
        c_traj = md_result['coords_traj']
        e_drift = float(np.max(e_traj) - np.min(e_traj)) * HARTREE_TO_KCAL
        dissoc  = md_result.get('dissociation_step')
        print(f"\n  MD stability ({args.md_steps} steps):")
        print(f"    Energy range: {e_drift:.3f} kcal/mol")
        if dissoc:
            print(f"    *** Dissociation detected at step {dissoc} ***")
        else:
            print(f"    No dissociation detected — trajectory stable")

    # ── Report ────────────────────────────────────────────────────────────
    report = [
        "sGDML Validation Report",
        "=" * 60,
        f"Date     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model    : {args.model}",
        f"Data     : {args.training_data}",
        "",
        "Prediction Errors",
        "-" * 40,
        f"  n_sample      : {err['n_sample']}",
        f"  Energy RMSE   : {err['e_rmse_kcal']:.4f} kcal/mol",
        f"  Energy MAE    : {err['e_mae_kcal']:.4f} kcal/mol",
        f"  Force  RMSE   : {err['f_rmse_HaAng']:.5f} Ha/Å",
        f"  Force  MAE    : {err['f_mae_HaAng']:.5f} Ha/Å",
        "",
        "Normal Mode Frequencies (cm⁻¹)",
        "-" * 40,
        f"  Imaginary modes: {n_imaginary}",
    ]
    for i, f in enumerate(vib_sorted):
        tag = ' IMAGINARY' if f < 0 else ''
        line = f"  mode {i+1:2d}: {f:8.1f}"
        if psi4_freqs is not None and i < len(psi4_freqs):
            line += f"  (PSI4: {psi4_freqs[i]:8.1f}  Δ={f-psi4_freqs[i]:+.1f})"
        report.append(line + tag)

    if md_result:
        report += [
            "",
            "MD Stability",
            "-" * 40,
            f"  Steps        : {args.md_steps}",
            f"  Energy range : {e_drift:.3f} kcal/mol",
            f"  Dissociation : {'YES at step ' + str(dissoc) if dissoc else 'no'}",
        ]

    rep_path = out / 'validation_report.txt'
    rep_path.write_text('\n'.join(report) + '\n')
    print(f"\n  Report saved → {rep_path}")

    # Save frequencies as npy for easy plotting
    np.save(out / 'sgdml_frequencies.npy', np.array(vib_sorted))
    print(f"  Frequencies  → {out / 'sgdml_frequencies.npy'}")


if __name__ == '__main__':
    main()
