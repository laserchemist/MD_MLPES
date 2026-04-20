#!/usr/bin/env python3
"""
validate_pes_frequencies.py — Validate any ML-PES driver via:
  1. Normal mode frequencies at the training-data equilibrium geometry
  2. Comparison to PSI4 reference frequencies (if supplied)
  3. Short 1000-step MD stability test at 300 K
  4. Energy/force prediction errors on a random sample

Supports three driver backends — select with exactly one of:
  --model       : Coulomb-matrix KRR (MLPESDriver, legacy)
  --sgdml-model : sGDML (SGDMLDriver)
  --mace-model  : MACE neural network (MACEDriver, preferred)

Usage
-----
  python3 validate_pes_frequencies.py \\
      --mace-model   outputs/mace_wB97X_20260417/mace_model.pt \\
      --training-data outputs/wB97X_surface_20260406_223155/training_data_wB97X_aug.npz

  python3 validate_pes_frequencies.py \\
      --model        outputs/mvko_20260319_081314/mlpes_initial.pkl \\
      --training-data outputs/mvko_20260319_081314/combined_training_data.npz \\
      --psi4-hessian outputs/mvko_20260319_081314/nm_displacements.npz

  python3 validate_pes_frequencies.py \\
      --sgdml-model  outputs/sgdml_wB97X_test/sgdml_model.pkl \\
      --training-data outputs/.../training_data.npz --skip-md

Output
------
  outputs/validate_pes_<ts>/
    validation_report.txt
    frequencies.npy          — vibrational frequencies (cm⁻¹)
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


# =============================================================================
# Driver loading
# =============================================================================

def load_driver(model_path, sgdml_model_path, mace_model_path):
    """Load the appropriate driver based on which path is provided."""
    if mace_model_path:
        from modules.mace_pes import MACEDriver
        driver = MACEDriver(mace_model_path)
        driver_type = 'MACE'
    elif sgdml_model_path:
        from modules.sgdml_pes import SGDMLDriver
        driver = SGDMLDriver(sgdml_model_path)
        driver_type = 'sGDML'
    else:
        from modules.bakken import MLPESDriver
        driver = MLPESDriver(model_path)
        driver_type = 'Coulomb+KRR'
    return driver, driver_type


# =============================================================================
# Normal mode computation
# =============================================================================

def compute_normal_modes(driver, coords_eq: np.ndarray,
                          delta: float = 0.005) -> tuple:
    """
    Compute mass-weighted Hessian → normal mode frequencies.

    Uses driver.analytic_hessian if available, otherwise FD.
    Returns (freqs_cmiv, eigvecs, H_ang2)
      freqs_cmiv : (3N,) frequencies in cm⁻¹ (negative = imaginary)
      eigvecs    : (3N, 3N) mass-weighted eigenvectors (columns)
    """
    n_atoms = driver.n_atoms
    n3      = n_atoms * 3
    masses  = driver.masses

    if getattr(driver, '_has_analytic', False):
        print(f"  Computing Hessian via analytic forces (δ={delta} Å)…")
        H = driver.analytic_hessian(coords_eq, delta=delta)  # Ha/Å²
    else:
        print(f"  Computing Hessian via FD on forces (δ={delta} Å)…")
        H = np.zeros((n3, n3))
        for i in range(n3):
            rp = coords_eq.flatten().copy(); rp[i] += delta
            rm = coords_eq.flatten().copy(); rm[i] -= delta
            fp = driver.forces(rp.reshape(n_atoms, 3)).flatten()
            fm = driver.forces(rm.reshape(n_atoms, 3)).flatten()
            H[i] = -(fp - fm) / (2.0 * delta)
        H = 0.5 * (H + H.T)

    # Ha/Å² → Ha/Bohr²
    H_bohr2 = H * (ANGSTROM_TO_BOHR ** 2)

    # Mass-weight
    mass_vec = np.repeat(masses, 3)
    inv_sqrt_m = 1.0 / np.sqrt(mass_vec)
    Hmw = (inv_sqrt_m[:, None] * H_bohr2) * inv_sqrt_m[None, :]

    eigenvals, eigvecs = np.linalg.eigh(Hmw)
    signs = np.sign(eigenvals)
    freqs = signs * np.sqrt(np.abs(eigenvals)) * FREQ_CONV
    return freqs, eigvecs


# =============================================================================
# MD stability check
# =============================================================================

def run_stability_md(driver, coords0: np.ndarray,
                      n_steps: int = 1000, temperature: float = 300.0) -> dict:
    """Short MD to check trajectory stability."""
    from modules.bakken import run_md
    print(f"\n  Running {n_steps}-step stability MD at {temperature} K…")
    return run_md(
        driver, coords0, n_steps=n_steps, temperature=temperature,
        timestep=0.5, save_every=10, preminimize=True,
        preminimize_steps=200, preminimize_tol=0.01,
        max_bond_extension=3.0,
    )


# =============================================================================
# Prediction error validation
# =============================================================================

def validate_predictions(driver, coords, energies, forces,
                          n_sample: int = 100) -> dict:
    """Energy and force RMSE on a random sample."""
    idx = np.random.choice(len(energies), min(n_sample, len(energies)), replace=False)
    e_pred = np.array([driver.energy(coords[i]) for i in idx])
    f_pred = np.array([driver.forces(coords[i]) for i in idx])

    e_err  = e_pred - energies[idx]
    f_err  = (f_pred - forces[idx]).flatten()

    # Centre energies (absolute offset is irrelevant)
    e_err -= e_err.mean()

    return {
        'n_sample':     len(idx),
        'e_rmse_kcal':  float(np.sqrt(np.mean(e_err**2))) * HARTREE_TO_KCAL,
        'e_mae_kcal':   float(np.mean(np.abs(e_err))) * HARTREE_TO_KCAL,
        'f_rmse_HaAng': float(np.sqrt(np.mean(f_err**2))),
        'f_mae_HaAng':  float(np.mean(np.abs(f_err))),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Validate ML-PES frequencies, MD stability, and prediction errors',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Driver selection — exactly one required
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--model',       default=None,
                     help='Coulomb+KRR model .pkl (MLPESDriver, legacy)')
    grp.add_argument('--sgdml-model', default=None,
                     help='sGDML model .pkl (SGDMLDriver)')
    grp.add_argument('--mace-model',  default=None,
                     help='MACE model .pt (MACEDriver, preferred)')

    parser.add_argument('--training-data', required=True,
                        help='Training data .npz (coordinates, energies, forces, symbols)')
    parser.add_argument('--psi4-hessian', default=None,
                        help='Optional nm_displacements.npz with PSI4 reference frequencies')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (auto-timestamped if omitted)')
    parser.add_argument('--md-steps', type=int, default=1000,
                        help='MD stability test steps')
    parser.add_argument('--skip-md', action='store_true',
                        help='Skip MD stability test')
    parser.add_argument('--n-validate', type=int, default=100,
                        help='Frames to sample for error validation')
    parser.add_argument('--hessian-delta', type=float, default=0.005,
                        help='FD displacement step for Hessian (Å)')
    args = parser.parse_args()

    ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path(args.output_dir) if args.output_dir else \
          Path('outputs') / f'validate_pes_{ts}'
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  ML-PES VALIDATION")
    print(f"{'='*60}")

    # ── Load driver ────────────────────────────────────────────────────
    driver, driver_type = load_driver(args.model, args.sgdml_model, args.mace_model)
    model_path = args.mace_model or args.sgdml_model or args.model
    print(f"  Driver type   : {driver_type}")
    print(f"  Model         : {model_path}")
    print(f"  Molecule      : {driver.symbols}  ({driver.n_atoms} atoms)")

    # ── Load training data ─────────────────────────────────────────────
    data     = np.load(args.training_data, allow_pickle=True)
    symbols  = data['symbols'].tolist()
    coords   = data['coordinates']
    energies = data['energies']
    forces   = data['forces']
    print(f"  Training data : {args.training_data}  ({len(energies)} frames)")

    # Equilibrium: lowest-energy frame
    i_eq = int(np.argmin(energies))
    coords_eq = coords[i_eq]
    print(f"  Eq. frame     : {i_eq}  (E={energies[i_eq]:.6f} Ha)")

    # ── Prediction errors ──────────────────────────────────────────────
    print(f"\n  Sampling {args.n_validate} frames for prediction errors…")
    np.random.seed(42)
    err = validate_predictions(driver, coords, energies, forces,
                               n_sample=args.n_validate)
    print(f"  Energy RMSE   : {err['e_rmse_kcal']:.4f} kcal/mol  "
          f"(MAE {err['e_mae_kcal']:.4f} kcal/mol)")
    print(f"  Force RMSE    : {err['f_rmse_HaAng']:.5f} Ha/Å  "
          f"(MAE {err['f_mae_HaAng']:.5f} Ha/Å)  "
          f"= {err['f_rmse_HaAng']*HARTREE_TO_KCAL:.3f} kcal/mol/Å")

    # ── Normal mode frequencies ────────────────────────────────────────
    print(f"\n  Computing normal mode frequencies…")
    freqs, _ = compute_normal_modes(driver, coords_eq, delta=args.hessian_delta)

    n_vib = driver.n_atoms * 3 - 6
    freqs_sorted = sorted(freqs, key=abs)
    trans_rot = freqs_sorted[:6]
    vib_modes = freqs_sorted[6:]

    print(f"\n  Trans/rot (expect ~0 cm⁻¹):")
    print(f"    {[f'{f:+.1f}' for f in trans_rot]}")

    n_imaginary = sum(1 for f in vib_modes if f < 0)
    n_unphysical = sum(1 for f in vib_modes if abs(f) > 5000)
    print(f"\n  Vibrational modes ({n_vib} expected):")
    print(f"    Imaginary   : {n_imaginary}")
    print(f"    Unphysical  : {n_unphysical}  (|freq| > 5000 cm⁻¹)")
    print(f"    Max freq    : {max(abs(f) for f in vib_modes):.0f} cm⁻¹")

    # Load PSI4 reference if provided
    psi4_freqs = None
    if args.psi4_hessian:
        try:
            nm = np.load(args.psi4_hessian, allow_pickle=True)
            psi4_freqs = nm.get('frequencies', None)
            if psi4_freqs is not None:
                psi4_freqs = np.sort(np.abs(psi4_freqs))[-n_vib:]
                print(f"  PSI4 ref      : {len(psi4_freqs)} modes loaded")
        except Exception as e:
            print(f"  Warning: could not load PSI4 hessian: {e}")

    print(f"\n  {'Mode':>4}  {'This model':>12}", end='')
    if psi4_freqs is not None:
        print(f"  {'PSI4':>10}  {'Δ (cm⁻¹)':>10}", end='')
    print()
    print(f"  {'─'*4}  {'─'*12}", end='')
    if psi4_freqs is not None:
        print(f"  {'─'*10}  {'─'*10}", end='')
    print()

    vib_sorted = sorted(vib_modes)
    for i, f in enumerate(vib_sorted):
        tag = ' *** IMAGINARY' if f < 0 else (
              ' *** UNPHYSICAL' if abs(f) > 5000 else '')
        line = f"  {i+1:>4}  {f:>12.1f}"
        if psi4_freqs is not None and i < len(psi4_freqs):
            delta = f - psi4_freqs[i]
            line += f"  {psi4_freqs[i]:>10.1f}  {delta:>+10.1f}"
        print(line + tag)

    # ── MD stability ──────────────────────────────────────────────────
    md_result = None
    if not args.skip_md:
        md_result = run_stability_md(driver, coords_eq, n_steps=args.md_steps)
        e_traj  = md_result['energies_ml']
        e_drift = float(np.max(e_traj) - np.min(e_traj)) * HARTREE_TO_KCAL
        dissoc  = md_result.get('dissociation_step')
        print(f"\n  MD stability ({args.md_steps} steps):")
        print(f"    Energy range: {e_drift:.3f} kcal/mol")
        if dissoc:
            print(f"    *** Dissociation at step {dissoc} ***")
        else:
            print(f"    Stable — no dissociation detected")

    # ── Report ────────────────────────────────────────────────────────
    report = [
        "ML-PES Validation Report",
        "=" * 60,
        f"Date         : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Driver type  : {driver_type}",
        f"Model        : {model_path}",
        f"Data         : {args.training_data}",
        "",
        "Prediction Errors",
        "-" * 40,
        f"  n_sample      : {err['n_sample']}",
        f"  Energy RMSE   : {err['e_rmse_kcal']:.4f} kcal/mol",
        f"  Energy MAE    : {err['e_mae_kcal']:.4f} kcal/mol",
        f"  Force  RMSE   : {err['f_rmse_HaAng']:.5f} Ha/Å  "
        f"({err['f_rmse_HaAng']*HARTREE_TO_KCAL:.3f} kcal/mol/Å)",
        f"  Force  MAE    : {err['f_mae_HaAng']:.5f} Ha/Å",
        "",
        "Normal Mode Frequencies (cm⁻¹)",
        "-" * 40,
        f"  Imaginary modes  : {n_imaginary}",
        f"  Unphysical modes : {n_unphysical}  (>5000 cm⁻¹)",
        f"  Max frequency    : {max(abs(f) for f in vib_modes):.0f} cm⁻¹",
        f"  (Coulomb+KRR typically has 5/9 unphysical at >5000 cm⁻¹)",
        f"  (MACE expected: 0 imaginary, 0 unphysical, max ~3200 cm⁻¹)",
    ]
    for i, f in enumerate(vib_sorted):
        tag = ' IMAGINARY' if f < 0 else (' UNPHYSICAL' if abs(f) > 5000 else '')
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
    print(f"\n  Report saved  : {rep_path}")

    np.save(out / 'frequencies.npy', np.array(vib_sorted))
    print(f"  Frequencies   : {out / 'frequencies.npy'}")

    # Summary assessment
    print(f"\n  ── Assessment ──")
    if n_imaginary == 0 and n_unphysical == 0:
        print(f"  PASS: No imaginary or unphysical modes — PES curvature is physical.")
    elif n_unphysical > 0:
        if driver_type == 'MACE':
            print(f"  FAIL: {n_unphysical} unphysical modes (>5000 cm⁻¹) — "
                  f"training data coverage issue (MACE C-H curvature requires large-amplitude")
            print(f"        C-H frames; 15 kcal/mol energy cutoff may remove them).")
            print(f"        Try: wider energy cutoff, more C-H stretch data, or NM-KRR backend.")
        else:
            print(f"  FAIL: {n_unphysical} unphysical modes (>5000 cm⁻¹) — "
                  f"descriptor stiffness artifact (Coulomb+KRR or sGDML).")
            print(f"        Switch to MACE: python3 train_mace_model.py \\")
            print(f"            --training-data {args.training_data} --output-dir outputs/mace_{ts}")
    elif n_imaginary > 0:
        print(f"  WARN: {n_imaginary} imaginary modes — ML-PES minimum is a saddle point.")
        print(f"        Check training data coverage near equilibrium.")
    if md_result and dissoc:
        print(f"  FAIL: MD dissociated at step {dissoc} — PES is not stable for MD.")


if __name__ == '__main__':
    main()
