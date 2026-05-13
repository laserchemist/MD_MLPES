#!/usr/bin/env python3
"""
Fix imaginary normal modes in wB97X NM-PES model by adding intermediate
single-point PSI4 energies along the sparse torsional mode (mode 2) and
near-origin points for the asymmetric CH3 modes (11 and 23).

Usage:
    # Dry run: just print displaced geometries
    python3 fix_nm_pes_gaps.py --model outputs/wB97X_nm_model_v4/mlpes_wB97X_nm.pkl --dry-run

    # Compute PSI4 single-points (requires PSI4)
    python3 fix_nm_pes_gaps.py --model outputs/wB97X_nm_model_v4/mlpes_wB97X_nm.pkl \
        --output-dir outputs/nm_pes_gap_fix_<ts>

    # After computing, merge into existing training data and retrain:
    python3 fix_nm_pes_gaps.py --model outputs/wB97X_nm_model_v4/mlpes_wB97X_nm.pkl \
        --merge-data outputs/nm_pes_gap_fix_<ts>/new_points.npz \
        --existing-data <training_data.npz> \
        --retrain --output-dir outputs/wB97X_nm_model_v5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'modules'))
from nm_pes import NMKRRPESModel

# NM-coordinate units: sqrt(amu)·Bohr
BOHR_TO_ANG = 0.529177210903
AMU_TO_AU   = 1822.888486
FREQ_CONV   = 5140.48  # cm-1 / sqrt(Ha/(Bohr2·amu))

# New intermediate displacements to add
# Format: (0-indexed mode, q-values to add on the POSITIVE side;
#          mirror is added automatically for symmetric modes)
GAPS_TO_FIX = {
    1: {  # Mode 2, 167.6 cm-1: fill the 1.495-unit gap
        'name': 'COO/vinyl torsion',
        'q_new': [0.30, 0.60, 0.90, 1.20],  # positive side
        'add_negative': True,                 # add -q too (symmetric)
    },
    10: {  # Mode 11, 985.3 cm-1: near-origin to anchor CH3 umbrella curvature
        'name': 'CH3 umbrella',
        'q_new': [0.05, 0.10, 0.15],
        'add_negative': True,
    },
    22: {  # Mode 23, 1534.1 cm-1: near-origin CH3 scissors
        'name': 'CH3 scissors',
        'q_new': [0.05, 0.10, 0.15],
        'add_negative': True,
    },
}


def q_to_cart(model: NMKRRPESModel, mode_idx: int, q_val: float) -> np.ndarray:
    """
    Displace the equilibrium geometry along NM mode `mode_idx` by q_val
    (in sqrt(amu)·Bohr units) and return Cartesian coords in Angstrom.

    q = U_vib^T M^{1/2} (R - R_eq), so inverse is:
    ΔR (Bohr) = M^{-1/2} U_vib[:,i] * q_val
    """
    masses_amu = np.array([
        {'H': 1.00794, 'C': 12.011, 'N': 14.007, 'O': 15.999,
         'S': 32.065, 'F': 18.998, 'Cl': 35.453}[s]
        for s in model.symbols
    ])
    # M^{-1/2} per Cartesian DOF
    inv_sqrt_mass = np.repeat(1.0 / np.sqrt(masses_amu), 3)  # (3N,)

    # eigenvector for this mode: (3N,) in mass-weighted coords
    eigvec = model.U_vib[:, mode_idx]  # (3N,)

    # displacement in Bohr: ΔR_bohr = M^{-1/2} * eigvec * q_val
    delta_bohr = inv_sqrt_mass * eigvec * q_val  # (3N,)
    delta_ang = delta_bohr.reshape(-1, 3) * BOHR_TO_ANG

    return model.eq_coords_ang + delta_ang


def run_psi4_singlepoint(symbols, coords_ang, method='wB97X-D', basis='6-31G*'):
    """Run PSI4 single-point energy (no gradient). Returns energy in Hartree."""
    try:
        import psi4
        psi4.core.be_quiet()
        psi4.set_options({
            'basis': basis,
            'scf_type': 'df',
            'reference': 'rhf',
            'maxiter': 200,
            'e_convergence': 1e-7,
            'd_convergence': 1e-7,
        })
        geom_str = '\n'.join(
            f"{sym}  {x:.10f}  {y:.10f}  {z:.10f}"
            for sym, (x, y, z) in zip(symbols, coords_ang)
        )
        mol = psi4.geometry(f"\n{geom_str}\nunits angstrom\n")
        energy = psi4.energy(f'{method}/{basis}', molecule=mol)
        return float(energy)
    except ImportError:
        print("  [no PSI4] returning mock energy")
        return None


def generate_new_points(model: NMKRRPESModel) -> list[dict]:
    """Generate list of (mode_idx, q_val, coords) records for new single-points."""
    records = []
    for mode_idx, cfg in GAPS_TO_FIX.items():
        q_list = list(cfg['q_new'])
        if cfg['add_negative']:
            q_list += [-q for q in cfg['q_new']]
        for q_val in sorted(q_list):
            coords = q_to_cart(model, mode_idx, q_val)
            records.append({
                'mode_idx': mode_idx,
                'freq_cm1': float(model.freqs_vib[mode_idx]),
                'mode_name': cfg['name'],
                'q_val': q_val,
                'coords_ang': coords,
            })
    return records


def compute_and_save(model, records, output_dir, dry_run=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    e_ref = model.predict_ha(np.zeros(model.U_vib.shape[1]))  # not PSI4 ref — just for display
    results = []

    for i, rec in enumerate(records):
        mode_idx = rec['mode_idx']
        q_val    = rec['q_val']
        coords   = rec['coords_ang']
        name     = rec['mode_name']
        freq     = rec['freq_cm1']

        print(f"[{i+1:2d}/{len(records)}] Mode {mode_idx+1} ({freq:.0f} cm-1, {name}), "
              f"q={q_val:+.3f}  ...", end=' ', flush=True)

        if dry_run:
            print(f"SKIPPED (dry run)")
            print(f"  Coords (Ang): " + "  ".join(
                f"{sym} ({c[0]:.4f},{c[1]:.4f},{c[2]:.4f})"
                for sym, c in zip(model.symbols, coords)
            ))
            continue

        t0 = time.perf_counter()
        energy_ha = run_psi4_singlepoint(model.symbols, coords)
        dt = time.perf_counter() - t0

        if energy_ha is None:
            print("FAILED (PSI4 not available)")
            continue

        e_ref_ha = model.y_train_ha[np.all(np.abs(model.X_train_q) < 1e-8, axis=1)].mean()
        delta_kcal = (energy_ha - e_ref_ha) * 627.509

        print(f"E={energy_ha:.8f} Ha  ΔE={delta_kcal:+.4f} kcal/mol  ({dt:.1f}s)")

        results.append({
            'mode_idx': mode_idx,
            'q_val': q_val,
            'energy_ha': energy_ha,
            'coords_ang': coords.tolist(),
        })

    if results:
        # Save new data as npz for merging
        coords_arr = np.array([r['coords_ang'] for r in results])
        energy_arr = np.array([r['energy_ha'] for r in results])
        q_vals_arr = np.array([[0.0]*model.U_vib.shape[1] for r in results])
        for j, r in enumerate(results):
            q_vals_arr[j, r['mode_idx']] = r['q_val']

        npz_path = output_dir / 'new_points.npz'
        np.savez(npz_path,
                 symbols=model.symbols,
                 coords_ang=coords_arr,
                 energy_ha=energy_arr,
                 q_vals=q_vals_arr,
                 )
        print(f"\nSaved {len(results)} new points to {npz_path}")

        # Save JSON summary
        summary = {
            'n_points': len(results),
            'modes_fixed': list(GAPS_TO_FIX.keys()),
            'results': [
                {'mode_idx': r['mode_idx'], 'q_val': r['q_val'],
                 'energy_ha': r['energy_ha']} for r in results
            ],
        }
        (output_dir / 'gap_fix_summary.json').write_text(json.dumps(summary, indent=2))


def merge_and_retrain(model, new_npz_path, output_dir, alpha=1e-6, gamma=0.2):
    """
    Merge new single-points (from new_npz_path) into the existing model's
    training data and retrain NM-KRR with new hyperparameters.

    new_npz_path: output of compute_and_save() — contains coords_ang, energy_ha, q_vals
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load new points (q_vals are already projected NM coordinates)
    new_data     = np.load(new_npz_path, allow_pickle=True)
    new_q        = new_data['q_vals']       # (N_new, n_vib)
    new_energies = new_data['energy_ha']    # (N_new,)
    print(f"New points loaded: {len(new_energies)} frames from {new_npz_path}")

    # Combine with existing model training data
    all_q      = np.concatenate([model.X_train_q, new_q],      axis=0)
    all_energy = np.concatenate([model.y_train_ha, new_energies], axis=0)
    print(f"Training set: {len(model.X_train_q)} existing + {len(new_energies)} new "
          f"= {len(all_energy)} total frames")

    n_vib = model.U_vib.shape[1]

    # Build new model — NMKRRPESModel.__init__ runs the KRR fit automatically
    new_model = NMKRRPESModel(
        eq_coords_ang = model.eq_coords_ang,
        U_vib         = model.U_vib,
        sqrt_mass     = model.sqrt_mass,
        freqs_vib     = model.freqs_vib,
        symbols       = model.symbols,
        gamma         = gamma,
        alpha_reg     = alpha,
        X_train_q     = all_q,
        y_train_ha    = all_energy,
        wall_factor   = model.wall_factor,
        wall_stiffness= model.wall_stiffness,
        wall_mode     = model.wall_mode,
        coord_scale   = model.coord_scale,
    )

    # Check imaginary modes via 1D curvature (fast proxy)
    q0 = np.zeros(n_vib)
    e0 = new_model.predict_ha(q0)
    imag_modes = []
    print("\nMode curvature check at q=0 (2nd-deriv FD, eps=0.03):")
    eps = 0.03
    for i in range(n_vib):
        qp = q0.copy(); qp[i] = eps
        qm = q0.copy(); qm[i] = -eps
        curv_kcal = (new_model.predict_ha(qp) + new_model.predict_ha(qm) - 2*e0) / eps**2 * 627.509
        if curv_kcal < 0:
            imag_freq = -FREQ_CONV * np.sqrt(-curv_kcal / 627.509)
            imag_modes.append(i)
            print(f"  Mode {i+1:2d} ({new_model.freqs_vib[i]:7.1f} cm-1): "
                  f"IMAGINARY {imag_freq:.1f} cm-1  (curv={curv_kcal:.4f} kcal/mol/q²)")
    if not imag_modes:
        print("  All 30 modes have positive curvature at q=0.")
    else:
        print(f"  {len(imag_modes)} imaginary modes remain (see above).")

    # Save
    save_path = output_dir / 'mlpes_wB97X_nm.pkl'
    new_model.save(str(save_path))

    summary = {
        'n_train': int(len(all_energy)),
        'n_new_points': int(len(new_energies)),
        'gamma': gamma,
        'alpha_reg': alpha,
        'n_imag_modes_diag': len(imag_modes),
        'imag_mode_indices': imag_modes,
    }
    (output_dir / 'nm_pes_summary.json').write_text(json.dumps(summary, indent=2))
    print(f"\nSaved retrained model to {save_path}")
    return new_model


def main():
    ap = argparse.ArgumentParser(description='Fix NM-PES imaginary modes by adding gap-filling PSI4 points')
    ap.add_argument('--model', required=True, help='Path to NMKRRPESModel .pkl')
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--dry-run', action='store_true', help='Print geometries without running PSI4')
    ap.add_argument('--merge-data', default=None,
                    help='new_points.npz from a previous --compute run; merges into model training data')
    ap.add_argument('--retrain', action='store_true',
                    help='Retrain after merging (requires --merge-data)')
    ap.add_argument('--alpha', type=float, default=1e-6,
                    help='KRR regularisation for retrained model (default 1e-6)')
    ap.add_argument('--gamma', type=float, default=0.2,
                    help='RBF gamma for retrained model (default 0.2)')
    args = ap.parse_args()

    model = NMKRRPESModel.load(args.model)
    print(f"Loaded model: {len(model.symbols)}-atom system, {model.U_vib.shape[1]} vib modes, "
          f"{len(model.X_train_q)} training frames")
    print(f"Modes to fix: {list(GAPS_TO_FIX.keys())} (0-indexed)")

    if args.output_dir is None:
        ts = time.strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'outputs/nm_pes_gap_fix_{ts}'

    if args.merge_data and args.retrain:
        merge_and_retrain(model, args.merge_data,
                          args.output_dir, alpha=args.alpha, gamma=args.gamma)
        return

    records = generate_new_points(model)
    print(f"\nWill compute {len(records)} new single-point energies:")
    for r in records:
        print(f"  Mode {r['mode_idx']+1} ({r['freq_cm1']:.0f} cm-1, {r['mode_name']}): "
              f"q={r['q_val']:+.3f}")

    compute_and_save(model, records, args.output_dir, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
