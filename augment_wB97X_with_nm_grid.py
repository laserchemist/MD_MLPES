#!/usr/bin/env python3
"""
augment_wB97X_with_nm_grid.py — Add the 232 CASSCF NM-grid frames (wB97X
single-points already computed) to the wB97X training set and retrain.

This fixes the C-H elongation problem by providing wB97X energies at
stretched C-H geometries (modes 24-29 in the grid), giving the KRR a
repulsive wall without needing separate PSI4 calculations.

Usage
-----
    python3 augment_wB97X_with_nm_grid.py
"""

import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'modules'))

from data_formats import TrajectoryData, load_trajectory, save_trajectory
from ml_pes import MLPESTrainer, MLPESConfig
from modules.normal_modes import compute_normal_modes

KCAL        = 627.509474
ANG2BOHR    = 1.88972612463
BOHR2ANG    = 1.0 / ANG2BOHR
KB_HA_PER_K = 3.1668114e-6

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_TRAJ   = 'outputs/wB97X_surface_20260406_223155/training_data_wB97X.npz'
GRID_JSON   = 'outputs/casscf_wB97X_nm_grid_20260407_184904/results.json'
EQ_COORDS   = 'outputs/mvko_20260319_081314/psi4_eq_coords.npy'
HESSIAN     = 'outputs/casscf_nm_delta_20260401_110049/hessian_used.npy'
OUT_DIR     = Path('outputs/wB97X_surface_20260406_223155')
GAMMA       = 0.001
ALPHA       = 1e-5
T_NM        = 300.0


def reconstruct_nm_coords(eq_coords_ang, eigvecs_mw, sqrt_mass, eigenvalues,
                           mode_idx, sign, factor):
    """Reconstruct Cartesian coords for one NM displacement frame."""
    n_vib    = eigvecs_mw.shape[1]
    kT       = T_NM * KB_HA_PER_K
    lam      = eigenvalues[mode_idx]
    a_therm  = np.sqrt(2 * kT / lam)
    q_vec    = np.zeros(n_vib)
    q_vec[mode_idx] = sign * factor * a_therm
    mw_disp          = eigvecs_mw @ q_vec                        # (3N,) sqrt(amu)·Bohr
    cart_disp_bohr   = mw_disp / sqrt_mass                       # (3N,) Bohr
    cart_disp_ang    = cart_disp_bohr.reshape(-1, 3) * BOHR2ANG  # (N,3) Ang
    return eq_coords_ang + cart_disp_ang


def main():
    # ── load base wB97X training data ──────────────────────────────────────────
    print(f'Loading base training data: {BASE_TRAJ}')
    base = load_trajectory(BASE_TRAJ)
    symbols    = list(base.symbols)
    base_coords = np.array(base.coordinates)
    base_e      = np.array(base.energies)
    base_forces = np.array(base.forces)
    base_dipoles= np.array(base.dipoles)
    print(f'  {len(base_coords)} frames')

    # ── load NM eigenvectors ───────────────────────────────────────────────────
    print(f'Loading equilibrium coords + Hessian for NM reconstruction ...')
    eq_coords = np.load(EQ_COORDS)          # (N, 3) Ang
    H_cart    = np.load(HESSIAN)            # (3N, 3N) Ha/Bohr²

    from modules.normal_modes import ATOMIC_MASSES
    masses_amu = np.array([ATOMIC_MASSES[s] for s in symbols])  # (N,)
    sqrt_mass  = np.repeat(np.sqrt(masses_amu), 3)        # (3N,)

    # Mass-weight Hessian
    H_mw = H_cart / np.outer(sqrt_mass, sqrt_mass)
    eigenvalues_all, eigvecs_all = np.linalg.eigh(H_mw)
    # Keep vibrational modes (skip 6 translational/rotational)
    # Sort by eigenvalue, take last n_vib positive ones
    n_vib = len(symbols) * 3 - 6
    pos_idx = np.where(eigenvalues_all > 0)[0]
    if len(pos_idx) < n_vib:
        # Take highest magnitude
        sort_idx = np.argsort(np.abs(eigenvalues_all))[-n_vib:]
        sort_idx = sort_idx[np.argsort(eigenvalues_all[sort_idx])]
    else:
        sort_idx = pos_idx[-n_vib:]
    eigenvalues = eigenvalues_all[sort_idx]   # (n_vib,) positive
    eigvecs_mw  = eigvecs_all[:, sort_idx]    # (3N, n_vib)
    print(f'  {n_vib} vibrational modes, freq range '
          f'{627.509*np.sqrt(eigenvalues[0]/219474.63**2*627.509**2):.0f}–... cm⁻¹')

    # Recompute frequencies for info
    FREQ_CONV = 5140.48  # cm⁻¹ / sqrt(Ha/(Bohr²·amu))
    freqs = FREQ_CONV * np.sqrt(eigenvalues)
    print(f'  Freq range: {freqs[0]:.1f} – {freqs[-1]:.1f} cm⁻¹')

    # ── load CASSCF grid results ───────────────────────────────────────────────
    print(f'\nLoading CASSCF grid: {GRID_JSON}')
    with open(GRID_JSON) as f:
        grid = json.load(f)
    ok_frames = [r for r in grid.values()
                 if r.get('status') == 'ok' and r.get('e_wb97x_ha') is not None]
    print(f'  {len(ok_frames)} clean frames with wB97X energy')

    # Reconstruct coords for each grid frame
    grid_coords  = []
    grid_e       = []
    grid_forces  = []   # zeros — forces not stored in grid results
    grid_dipoles = []   # zeros — dipoles not computed

    for r in ok_frames:
        c = reconstruct_nm_coords(
            eq_coords, eigvecs_mw, sqrt_mass, eigenvalues,
            r['mode_idx'], r['sign'], r['factor'])
        grid_coords.append(c)
        grid_e.append(r['e_wb97x_ha'])
        grid_forces.append(np.zeros((len(symbols), 3)))
        grid_dipoles.append(np.zeros(3))

    grid_coords  = np.array(grid_coords)
    grid_e       = np.array(grid_e)
    grid_forces  = np.array(grid_forces)
    grid_dipoles = np.array(grid_dipoles)
    dE_grid = (grid_e - grid_e.min()) * KCAL
    print(f'  Grid ΔE range: {float(dE_grid.min()):.1f} – {float(dE_grid.max()):.1f} kcal/mol')

    # Check C-H distances in grid frames to confirm coverage
    h_idx = [i for i, s in enumerate(symbols) if s == 'H']
    c_idx = [i for i, s in enumerate(symbols) if s == 'C']
    max_ch = max(
        np.linalg.norm(grid_coords[f, ci] - grid_coords[f, hi])
        for f in range(len(grid_coords))
        for ci in c_idx for hi in h_idx
        if np.linalg.norm(grid_coords[f, ci] - grid_coords[f, hi]) < 3.0
    )
    print(f'  Max C-H distance in grid frames: {max_ch:.3f} Å  '
          f'(repulsive wall coverage)')

    # ── merge ──────────────────────────────────────────────────────────────────
    all_coords  = np.concatenate([base_coords,  grid_coords],  axis=0)
    all_e       = np.concatenate([base_e,       grid_e],       axis=0)
    all_forces  = np.concatenate([base_forces,  grid_forces],  axis=0)
    all_dipoles = np.concatenate([base_dipoles, grid_dipoles], axis=0)
    print(f'\nMerged: {len(base_coords)} base + {len(grid_coords)} grid '
          f'= {len(all_coords)} total frames')

    traj_aug = TrajectoryData(
        symbols     = symbols,
        coordinates = all_coords,
        energies    = all_e,
        forces      = all_forces,
        dipoles     = all_dipoles,
        metadata    = json.dumps({
            'method': 'wb97x-d', 'basis': '6-31G*',
            'n_base': len(base_coords), 'n_grid': len(grid_coords),
            'note': 'augmented with CASSCF NM-grid wB97X frames for C-H repulsion',
        }),
    )
    aug_path = OUT_DIR / 'training_data_wB97X_aug.npz'
    save_trajectory(traj_aug, aug_path)
    print(f'Saved augmented training data → {aug_path}')

    # ── retrain ────────────────────────────────────────────────────────────────
    print(f'\nTraining KRR (γ={GAMMA}, α={ALPHA}) on {len(all_coords)} frames ...')
    config  = MLPESConfig(gamma=GAMMA, alpha=ALPHA, train_forces=False,
                          tune_hyperparameters=False)
    trainer = MLPESTrainer(config)
    trainer.train(traj_aug)

    model_path = OUT_DIR / 'mlpes_wB97X_aug.pkl'
    trainer.save(str(model_path))
    print(f'Model saved → {model_path}')

    print(f'\nNext:')
    print(f'  python3 ir_md_spectrum.py \\')
    print(f'    --model {model_path} \\')
    print(f'    --training-data outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \\')
    print(f'    --nm-delta-model outputs/casscf_wB97X_nm_grid_20260407_184904/nm_delta_s0_model.pkl \\')
    print(f'    --steps 30000 --temp 300 --preminimize --zpe-min-freq 50 --zpe-max-freq 4000 \\')
    print(f'    --n-trajectories 5 --max-bond-extension 2.0 \\')
    print(f'    --output-dir outputs/ir_spectrum_wB97X_delta_v3_300K')


if __name__ == '__main__':
    main()
