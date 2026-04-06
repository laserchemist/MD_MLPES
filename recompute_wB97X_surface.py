#!/usr/bin/env python3
"""
recompute_wB97X_surface.py — Recompute training data at ωB97X-D/6-31G*

Loads the existing MVKO training geometries (B3LYP/6-31G*) and recomputes
energies, forces, and dipoles at ωB97X-D/6-31G* using PSI4.  Both surfaces
are retained for direct comparison of ML-PES and IR spectra.

Motivation
----------
B3LYP/6-31G* lacks dispersion and has poor charge-transfer description of
the zwitterionic (C⁺)(O-O⁻) Criegee ground state.  ωB97X-D (Chai & Head-Gordon
2008) is a range-separated hybrid with Grimme D2 dispersion correction, trained
at the same level as ANI-2x.  A better base surface reduces the multireference
correction needed from CASSCF/NEVPT2 delta-ML.

Workflow
--------
1. Load existing training data (energies + coordinates + forces + dipoles).
2. Filter: keep only frames with dE_B3LYP < --max-energy kcal/mol.
3. For each frame: run PSI4 gradient at ωB97X-D/6-31G* to get energy,
   forces, and dipole.
4. Save new training data with both B3LYP and ωB97X energies stored.
5. Retrain ML-PES (KRR) on ωB97X surface and compare Hessian frequencies
   against B3LYP model.

Usage
-----
    python3 recompute_wB97X_surface.py \\
        --training-data outputs/mvko_20260319_081314/combined_training_data.npz \\
        --eq-coords     outputs/mvko_20260319_081314/psi4_eq_coords.npy \\
        --b3lyp-model   outputs/mvko_20260319_081314/mlpes_initial.pkl \\
        --max-energy    100.0 \\
        --method        wb97x-d

    # Resume a partial run (skip already-computed frames):
    python3 recompute_wB97X_surface.py \\
        --training-data outputs/mvko_20260319_081314/combined_training_data.npz \\
        --eq-coords     outputs/mvko_20260319_081314/psi4_eq_coords.npy \\
        --b3lyp-model   outputs/mvko_20260319_081314/mlpes_initial.pkl \\
        --resume        outputs/wB97X_surface_<ts>

    # Skip retraining (just compare energies):
    python3 recompute_wB97X_surface.py ... --no-retrain

Outputs (in outputs/wB97X_surface_<ts>/)
-----------------------------------------
training_data_wB97X.npz    — new training data at ωB97X-D level
                             (coords identical to input; energies/forces replaced)
b3lyp_energies.npy         — original B3LYP energies for each kept frame (Ha)
wb97x_energies.npy         — new ωB97X-D energies (Ha)
delta_b3lyp_wb97x.npy      — relative correction δ = E_wB97X − E_B3LYP (kcal/mol)
mlpes_wB97X.pkl            — retrained KRR model on ωB97X surface
comparison.json            — RMSE, Hessian freq comparison, timing
results.json               — per-frame status (energy, converged, error)
"""

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'modules'))

try:
    import psi4
    print(f"PSI4 {psi4.__version__} available")
except ImportError:
    print("PSI4 not available — cannot recompute energies"); sys.exit(1)

from data_formats import TrajectoryData, load_trajectory, save_trajectory
from ml_pes import MLPESTrainer, MLPESConfig, CoulombMatrixDescriptor

HARTREE_TO_KCAL  = 627.509474
ANGSTROM_TO_BOHR = 1.88972612456
AU_TO_DEBYE      = 2.541746

MVKOO_SYMBOLS = ['C', 'O', 'O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']


# ── PSI4 single-point ──────────────────────────────────────────────────────────

def psi4_single_point(symbols, coords_ang, method='wb97x-d', basis='6-31G*'):
    """
    Run PSI4 gradient at the given geometry.  Returns (energy_Ha, forces_Ha_per_ang,
    dipole_debye, error_str_or_None).

    Notes
    -----
    - method='wb97x-d'  → Chai-Gordon ωB97X-D (range-sep + Grimme D2)
    - method='b3lyp'    → B3LYP (DFT-D3 can be added via b3lyp-d3)
    PSI4 1.10 functional name: lowercase without prefix (wb97x-d).
    """
    mol_str = "0 1\n"
    for s, c in zip(symbols, coords_ang):
        mol_str += f"{s}  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}\n"
    mol_str += "units angstrom\nno_reorient\nno_com\nsymmetry c1"

    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.be_quiet()
    psi4.set_memory('3 GB')
    psi4.set_num_threads(4)
    psi4.set_options({
        'basis':         basis,
        'scf_type':      'df',
        'reference':     'rhf',
        'maxiter':       200,
        'e_convergence': 1e-7,
        'd_convergence': 1e-7,
        'dft_spherical_points': 590,
        'dft_radial_points':    99,
    })

    try:
        mol = psi4.geometry(mol_str)
        grad_mat, wfn = psi4.gradient(f'{method}/{basis}', molecule=mol,
                                       return_wfn=True)
        energy = float(wfn.energy())
        n = len(symbols)
        grad = np.array([[grad_mat.get(i, j) for j in range(3)] for i in range(n)])
        forces = -grad / ANGSTROM_TO_BOHR

        dipole = np.zeros(3)
        try:
            psi4.oeprop(wfn, 'DIPOLE')
            for k, ax in enumerate('XYZ'):
                for key in (f'SCF DIPOLE {ax}', f'DIPOLE {ax}', f'DFT DIPOLE {ax}'):
                    try:
                        dipole[k] = psi4.variable(key) * AU_TO_DEBYE
                        break
                    except Exception:
                        pass
        except Exception:
            pass

        return energy, forces, dipole, None

    except Exception as exc:
        return None, None, np.zeros(3), str(exc)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Recompute MVKO training data at ωB97X-D/6-31G*')
    parser.add_argument('--training-data', required=True,
                        help='Path to existing training .npz (B3LYP geometries)')
    parser.add_argument('--eq-coords', default=None,
                        help='PSI4 equilibrium .npy for relative energy alignment '
                             '(default: use lowest-energy frame in training data)')
    parser.add_argument('--b3lyp-model', default=None,
                        help='Existing B3LYP KRR model .pkl for Hessian comparison')
    parser.add_argument('--method', default='wb97x-d',
                        help='PSI4 DFT method name (default: wb97x-d)')
    parser.add_argument('--basis',  default='6-31G*')
    parser.add_argument('--max-energy', type=float, default=100.0,
                        help='Max B3LYP relative energy (kcal/mol) for frame inclusion')
    parser.add_argument('--gamma',  type=float, default=0.001,
                        help='KRR gamma for retrained ωB97X model')
    parser.add_argument('--alpha',  type=float, default=1e-5,
                        help='KRR regularisation alpha')
    parser.add_argument('--resume', default=None,
                        help='Output directory from a partial run to resume')
    parser.add_argument('--no-retrain', action='store_true',
                        help='Skip ML-PES retraining (just compute energies)')
    args = parser.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.resume) if args.resume else \
              Path(f'outputs/wB97X_surface_{ts}')
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / 'results.json'

    # ── Load training data ─────────────────────────────────────────────────────
    print(f"\nLoading training data: {args.training_data}")
    traj = load_trajectory(args.training_data)
    symbols   = list(traj.symbols)
    n_frames  = len(traj.coordinates)
    print(f"  {n_frames} frames, {len(symbols)} atoms ({symbols})")

    energies_b3lyp = np.array(traj.energies)          # (N,) Ha
    e_min_b3lyp    = energies_b3lyp.min()
    dE_b3lyp       = (energies_b3lyp - e_min_b3lyp) * HARTREE_TO_KCAL   # kcal/mol

    # Apply energy filter
    keep_mask = dE_b3lyp < args.max_energy
    keep_idx  = np.where(keep_mask)[0]
    print(f"  Keeping {len(keep_idx)}/{n_frames} frames "
          f"(dE_B3LYP < {args.max_energy:.0f} kcal/mol)")

    # ── Equilibrium reference energy ────────────────────────────────────────────
    if args.eq_coords is not None:
        eq_coords = np.load(args.eq_coords)
        print(f"  Equilibrium geometry: {args.eq_coords}")
    else:
        eq_idx    = int(energies_b3lyp.argmin())
        eq_coords = traj.coordinates[eq_idx]
        print(f"  Equilibrium geometry: frame {eq_idx} (lowest B3LYP energy)")

    # ── Resume: load existing results ──────────────────────────────────────────
    completed = {}  # frame_idx → result dict
    if results_path.exists():
        with open(results_path) as f:
            completed = {int(k): v for k, v in json.load(f).items()}
        print(f"  Resuming: {len(completed)} frames already computed")

    # ── Compute ωB97X-D energies ───────────────────────────────────────────────
    print(f"\nRunning PSI4 {args.method}/{args.basis} single-points "
          f"({len(keep_idx)} frames)...")

    n_ok = n_fail = 0
    t_start = time.time()

    for rank, idx in enumerate(keep_idx):
        idx = int(idx)
        if idx in completed:
            status = completed[idx].get('status', '?')
            if status == 'ok':
                n_ok += 1
                continue

        coords_ang = traj.coordinates[idx]
        t0 = time.time()
        energy, forces, dipole, err = psi4_single_point(
            symbols, coords_ang, method=args.method, basis=args.basis)
        elapsed = time.time() - t0

        if energy is None:
            n_fail += 1
            print(f"  [{rank+1}/{len(keep_idx)}] frame {idx}: FAILED in {elapsed:.1f}s — {err}")
            completed[idx] = {
                'status': 'failed', 'error': err,
                'dE_b3lyp': float(dE_b3lyp[idx]),
            }
        else:
            n_ok += 1
            completed[idx] = {
                'status':     'ok',
                'energy_ha':  float(energy),
                'forces':     forces.tolist(),
                'dipole_D':   dipole.tolist(),
                'dE_b3lyp':   float(dE_b3lyp[idx]),
                'elapsed_s':  float(elapsed),
            }
            elapsed_total = time.time() - t_start
            rate = elapsed_total / n_ok
            remaining = (len(keep_idx) - n_ok) * rate / 3600
            print(f"  [{rank+1}/{len(keep_idx)}] frame {idx}: "
                  f"E={energy:.8f} Ha  dE_B3LYP={dE_b3lyp[idx]:.1f} kcal/mol  "
                  f"{elapsed:.1f}s  [ETA {remaining:.1f}h]")

        # Save checkpoint after every frame
        with open(results_path, 'w') as f:
            json.dump({str(k): v for k, v in completed.items()}, f, indent=2)

    print(f"\nComputed: {n_ok} ok, {n_fail} failed")

    # ── Assemble new training arrays ───────────────────────────────────────────
    ok_idx     = sorted(i for i, r in completed.items() if r['status'] == 'ok')
    wb97x_e    = np.array([completed[i]['energy_ha'] for i in ok_idx])
    b3lyp_e    = energies_b3lyp[ok_idx]
    e_min_wb97 = wb97x_e.min()

    # Align relative energies: set minimum to 0
    wb97x_e_rel = wb97x_e - e_min_wb97                        # Ha
    b3lyp_e_rel = b3lyp_e - e_min_b3lyp                       # Ha

    delta_kcal = (wb97x_e_rel - b3lyp_e_rel) * HARTREE_TO_KCAL

    np.save(out_dir / 'b3lyp_energies.npy',     b3lyp_e)
    np.save(out_dir / 'wb97x_energies.npy',     wb97x_e)
    np.save(out_dir / 'delta_b3lyp_wb97x.npy',  delta_kcal)
    np.save(out_dir / 'ok_frame_indices.npy',   np.array(ok_idx))

    print(f"\nδ(ωB97X-D − B3LYP) statistics (kcal/mol, relative energies):")
    print(f"  mean  = {delta_kcal.mean():.3f}")
    print(f"  std   = {delta_kcal.std():.3f}")
    print(f"  min   = {delta_kcal.min():.3f}")
    print(f"  max   = {delta_kcal.max():.3f}")

    # ── Build ωB97X TrajectoryData ─────────────────────────────────────────────
    new_coords  = traj.coordinates[ok_idx]
    new_forces  = np.array([completed[i]['forces']  for i in ok_idx])
    new_dipoles = np.array([completed[i]['dipole_D'] for i in ok_idx])

    traj_wb97x = TrajectoryData(
        symbols     = symbols,
        coordinates = new_coords,
        energies    = list(wb97x_e),
        forces      = new_forces,
        dipoles     = new_dipoles,
        metadata    = json.dumps({
            'method': args.method, 'basis': args.basis,
            'source': str(args.training_data),
            'n_frames': len(ok_idx),
            'created': ts,
        }),
    )
    save_trajectory(traj_wb97x, out_dir / 'training_data_wB97X.npz')
    print(f"\nSaved {len(ok_idx)} frames → {out_dir}/training_data_wB97X.npz")

    # ── Retrain KRR on ωB97X surface ──────────────────────────────────────────
    if not args.no_retrain:
        print(f"\nRetraining KRR on ωB97X-D surface "
              f"(γ={args.gamma}, α={args.alpha}) ...")

        config = MLPESConfig(
            gamma=args.gamma,
            alpha=args.alpha,
            train_forces=False,
        )
        trainer = MLPESTrainer(config)
        trainer.train(traj_wb97x)
        rmse_train = trainer.training_rmse * HARTREE_TO_KCAL
        print(f"  Train RMSE: {rmse_train:.4f} kcal/mol")

        model_path = out_dir / 'mlpes_wB97X.pkl'
        trainer.save(str(model_path))
        print(f"  Model saved → {model_path}")

        # ── Hessian frequency comparison ───────────────────────────────────────
        try:
            from modules.bakken import MLPESDriver
            from modules.normal_modes import compute_normal_modes

            driver_wb97x = MLPESDriver(trainer, symbols)
            hess_wb97x   = driver_wb97x.analytic_hessian(eq_coords)
            freqs_wb97x, *_ = compute_normal_modes(symbols, hess_wb97x)

            freq_info = {
                'wb97x_freqs_cm1': [float(f) for f in freqs_wb97x],
                'wb97x_n_imaginary': int((freqs_wb97x < 0).sum()),
                'wb97x_ch_cluster_cm1': [float(f) for f in freqs_wb97x if f > 2000],
            }

            if args.b3lyp_model and Path(args.b3lyp_model).exists():
                with open(args.b3lyp_model, 'rb') as f:
                    trainer_b3lyp = pickle.load(f)
                driver_b3lyp = MLPESDriver(trainer_b3lyp, symbols)
                hess_b3lyp   = driver_b3lyp.analytic_hessian(eq_coords)
                freqs_b3lyp, *_ = compute_normal_modes(symbols, hess_b3lyp)
                freq_info['b3lyp_freqs_cm1']  = [float(f) for f in freqs_b3lyp]
                freq_info['b3lyp_n_imaginary'] = int((freqs_b3lyp < 0).sum())
                freq_info['b3lyp_ch_cluster_cm1'] = [float(f) for f in freqs_b3lyp if f > 2000]

                print("\nHessian frequency comparison (analytic KRR):")
                print(f"  B3LYP:   {int((freqs_b3lyp<0).sum())} imaginary, "
                      f"C-H cluster: {[int(f) for f in freqs_b3lyp if f > 2000]} cm-1")
                print(f"  ωB97X-D: {int((freqs_wb97x<0).sum())} imaginary, "
                      f"C-H cluster: {[int(f) for f in freqs_wb97x if f > 2000]} cm-1")

        except Exception as exc:
            freq_info = {'error': str(exc)}
            print(f"  Hessian comparison skipped: {exc}")

        comparison = {
            'n_frames':          len(ok_idx),
            'method':            args.method,
            'basis':             args.basis,
            'gamma':             args.gamma,
            'alpha':             args.alpha,
            'train_rmse_kcal':   float(rmse_train),
            'delta_mean_kcal':   float(delta_kcal.mean()),
            'delta_std_kcal':    float(delta_kcal.std()),
            'delta_min_kcal':    float(delta_kcal.min()),
            'delta_max_kcal':    float(delta_kcal.max()),
            **freq_info,
        }
        with open(out_dir / 'comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)

    print(f"\nAll outputs in: {out_dir}")
    print("Next steps:")
    print("  1. Run IR spectrum with ωB97X model and compare to B3LYP IR")
    print("  2. Feed ωB97X training data to casscf_nm_systematic.py for smaller corrections")


if __name__ == '__main__':
    main()
