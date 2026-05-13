#!/usr/bin/env python3
"""
build_conformer_nm_pes.py — Build a NM-KRR PES for any MVKO conformer.

Pipeline:
  Step 1  PSI4 wB97X-D/6-31G* geometry optimisation
  Step 2  PSI4 wB97X-D/6-31G* Hessian → normal modes
  Step 3  NM-displaced single-points (±1…n × thermal amplitude per mode)
  Step 4  [optional] Multi-T PSI4 MD for additional coverage
  Step 5  NM-KRR training (LOO-CV γ/α sweep)

Usage
-----
    # Anti-trans conformer (geometry from test_molecules.py):
    python3 build_conformer_nm_pes.py \\
        --molecule mvko_anti_trans \\
        --out-dir  outputs/anti_trans_nm_pes_$(date +%Y%m%d)

    # Resume from saved state:
    python3 build_conformer_nm_pes.py \\
        --restart outputs/anti_trans_nm_pes_20260420/state.json

    # Skip Steps 1-2 if optimised geometry + Hessian already exist:
    python3 build_conformer_nm_pes.py \\
        --molecule mvko_anti_trans \\
        --eq-coords outputs/.../eq_coords.npy \\
        --hessian   outputs/.../hessian.npy \\
        --out-dir   outputs/anti_trans_nm_pes_$(date +%Y%m%d)

    # Train only (all PSI4 data already collected):
    python3 build_conformer_nm_pes.py \\
        --restart outputs/anti_trans_nm_pes_20260420/state.json --steps 5

Notes
-----
• Uses wB97X-D/6-31G* throughout (matches syn-trans training set).
• Atom ordering must match the molecule entry in test_molecules.py
  (C1=Criegee C, O1=proximal, O2=distal, C2=vinyl-CH, C3=vinyl-CH2,
   C4=methyl, H1-H6 as labelled).
• NM-KRR model is saved as mlpes_nm.pkl — compatible with NMPESDriver
  and all ir_md_spectrum.py flags (--nm-pes-model).
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'modules'))

from data_formats import TrajectoryData, save_trajectory, load_trajectory
from modules.normal_modes import ATOMIC_MASSES, FREQ_CONV, BOHR_TO_ANGSTROM
from modules.nm_pes import NMKRRPESModel
from modules.test_molecules import get_molecule

# ── PSI4 settings ─────────────────────────────────────────────────────────────
PSI4_METHOD  = 'wb97x-d'
PSI4_OPTIONS = {
    'basis':         '6-31G*',
    'scf_type':      'df',
    'reference':     'rhf',
    'maxiter':       200,
    'e_convergence': 1e-7,
    'd_convergence': 1e-7,
}
PSI4_MEM_GB  = 6
PSI4_THREADS = 4

ANG_TO_BOHR  = 1.88972612463
BOHR_TO_ANG  = 1.0 / ANG_TO_BOHR
AU_TO_DEBYE  = 2.541746
KCAL         = 627.509474
KB_HA_PER_K  = 3.1668114e-6

# ── NM-KRR hyperparameter grid ────────────────────────────────────────────────
DEFAULT_GAMMA_VALUES = [0.05, 0.1, 0.2, 0.5, 1.0]
DEFAULT_ALPHA_VALUES = [1e-6, 1e-5, 1e-4]


# =============================================================================
# PSI4 helpers
# =============================================================================

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


def psi4_optimize(symbols, coords_init):
    """wB97X-D/6-31G* geometry optimisation. Returns (coords_ang, energy_ha)."""
    _psi4_setup()
    mol = psi4.geometry(_mol_str(symbols, coords_init))
    print(f"  Optimising {len(symbols)}-atom geometry ({PSI4_METHOD}/{PSI4_OPTIONS['basis']}) ...")
    energy_opt = psi4.optimize(f'{PSI4_METHOD}/{PSI4_OPTIONS["basis"]}', molecule=mol)
    geom_bohr  = np.array(mol.geometry())
    coords_opt = geom_bohr * BOHR_TO_ANG
    return coords_opt, float(energy_opt)


def psi4_hessian(symbols, coords_ang):
    """wB97X-D/6-31G* Hessian. Returns (H_ha_bohr2, energy_ha)."""
    _psi4_setup()
    mol = psi4.geometry(_mol_str(symbols, coords_ang))
    print(f"  Computing Hessian ({PSI4_METHOD}/{PSI4_OPTIONS['basis']}) ...")
    hess_obj, wfn = psi4.hessian(f'{PSI4_METHOD}/{PSI4_OPTIONS["basis"]}',
                                  molecule=mol, return_wfn=True)
    H = np.array(hess_obj)   # (3N, 3N) Ha/Bohr²
    return H, float(wfn.energy())


def psi4_singlepoint(symbols, coords):
    """wB97X-D/6-31G* energy + gradient + dipole."""
    _psi4_setup()
    mol = psi4.geometry(_mol_str(symbols, coords))
    grad_obj, wfn = psi4.gradient(f'{PSI4_METHOD}/{PSI4_OPTIONS["basis"]}',
                                   molecule=mol, return_wfn=True,
                                   properties=['dipole'])
    energy_ha = float(wfn.energy())
    grad_bohr = np.array(grad_obj)
    forces    = -grad_bohr / ANG_TO_BOHR
    dipole_D  = np.zeros(3)
    try:
        dip_au   = np.array(psi4.variable('SCF DIPOLE'))
        dipole_D = dip_au * AU_TO_DEBYE
    except Exception:
        try:
            psi4.oeprop(wfn, 'DIPOLE')
            dipole_D = np.array([wfn.variable(f'DIPOLE {ax}') for ax in 'XYZ'])
        except Exception:
            pass
    return energy_ha, forces, dipole_D


# =============================================================================
# Normal-mode helpers
# =============================================================================

def compute_nm_modes(symbols, eq_coords_ang, H_cart_ha_bohr2):
    """
    Returns (freqs_cm, U_vib, sqrt_mass, eigenvalues, coord_scale)
    where coord_scale[k] = thermal amplitude at T=300 K in sqrt(amu)·Bohr.
    """
    masses_amu = np.array([ATOMIC_MASSES[s] for s in symbols])
    sqrt_mass  = np.repeat(np.sqrt(masses_amu), 3)
    H_mw = H_cart_ha_bohr2 / np.outer(sqrt_mass, sqrt_mass)
    evals_all, evecs_all = np.linalg.eigh(H_mw)

    n_vib = len(symbols) * 3 - 6
    pos_idx = np.where(evals_all > 0)[0]
    if len(pos_idx) >= n_vib:
        sort_idx = pos_idx[-n_vib:]
    else:
        sort_idx = np.argsort(np.abs(evals_all))[-n_vib:]
        sort_idx = sort_idx[np.argsort(evals_all[sort_idx])]

    eigenvalues = evals_all[sort_idx]
    U_vib       = evecs_all[:, sort_idx]
    freqs       = FREQ_CONV * np.sqrt(np.abs(eigenvalues)) * np.sign(eigenvalues)

    # Thermal amplitude at 300 K: a_k = sqrt(2 kT / lambda_k) in sqrt(amu)·Bohr
    # (classical 1-sigma amplitude; factor 2 matches train_wB97X_nm_model.py)
    T_ref = 300.0
    coord_scale = np.sqrt(2.0 * KB_HA_PER_K * T_ref / np.maximum(eigenvalues, 1e-12))

    return freqs, U_vib, sqrt_mass, eigenvalues, coord_scale


def make_displaced_geometry(eq_ang, U_vib, sqrt_mass, coord_scale, mode_k, amp):
    """
    Displace equilibrium geometry along normal mode k by amp × coord_scale[k].

    amp is dimensionless; positive/negative for ± displacement.
    Returns (N, 3) Angstrom.
    """
    q_phys = amp * coord_scale[mode_k]        # in sqrt(amu)·Bohr
    dr_mw  = U_vib[:, mode_k] * q_phys       # mass-weighted displacement (sqrt(amu)·Bohr)
    dr_ang = dr_mw / sqrt_mass / ANG_TO_BOHR  # Cartesian Angstrom
    return eq_ang + dr_ang.reshape(eq_ang.shape)


def geometry_ok(coords, min_d=0.70, max_d=3.5):
    """Return True if no bonded atom pair is unphysically close or far."""
    n = len(coords)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < min_d:
                return False
    return True


# =============================================================================
# NM training data generation
# =============================================================================

def collect_nm_data(symbols, eq_coords_ang, U_vib, sqrt_mass, coord_scale,
                    T_nm=800.0, n_amplitudes=5, max_factor=2.5, out_dir=None):
    """
    Generate ±n × thermal_amplitude(T_nm) displacements along all modes.
    Runs PSI4 single-points. Returns TrajectoryData.
    """
    n_vib    = U_vib.shape[1]
    cs_Tnm   = coord_scale * np.sqrt(T_nm / 300.0)   # rescale to T_nm

    amplitudes = [a * max_factor / n_amplitudes
                  for a in range(1, n_amplitudes + 1)]

    all_coords, all_e, all_f, all_d = [], [], [], []
    n_ok = n_skip = 0

    for k in range(n_vib):
        for sign in [+1, -1]:
            for amp in amplitudes:
                coords = make_displaced_geometry(
                    eq_coords_ang, U_vib, sqrt_mass, cs_Tnm, k, sign * amp)
                if not geometry_ok(coords):
                    n_skip += 1
                    continue
                try:
                    e, f, d = psi4_singlepoint(symbols, coords)
                    all_coords.append(coords)
                    all_e.append(e)
                    all_f.append(f)
                    all_d.append(d)
                    n_ok += 1
                    print(f'    mode {k:2d} sign={sign:+d} amp={amp:.2f}  E={e:.6f} Ha  ok={n_ok}')
                except Exception as exc:
                    n_skip += 1
                    print(f'    mode {k:2d} sign={sign:+d} amp={amp:.2f}  FAILED: {exc}')

    print(f'  NM displacements: {n_ok} ok, {n_skip} skipped/failed')
    traj = TrajectoryData(
        symbols=symbols,
        coordinates=np.array(all_coords),
        energies=np.array(all_e),
        forces=np.array(all_f),
        dipoles=np.array(all_d),
    )
    if out_dir is not None:
        save_trajectory(traj, str(out_dir / 'nm_displacements.npz'))
    return traj


# =============================================================================
# NM-KRR training
# =============================================================================

def loo_cv_rmse(X_q, y, gamma, alpha):
    M     = len(y)
    y_mean = float(np.mean(y))
    yc    = y - y_mean
    A2    = np.sum(X_q ** 2, axis=1, keepdims=True)
    K     = np.exp(-gamma * (A2 + A2.T - 2.0 * X_q @ X_q.T))
    K_reg = K.copy()
    K_reg[np.diag_indices_from(K_reg)] += alpha
    beta  = np.linalg.solve(K_reg, yc)
    preds = K @ beta + y_mean
    K_reg_inv = np.linalg.inv(K_reg)
    H_diag = np.einsum('ij,ji->i', K, K_reg_inv)
    residuals = (y - preds) / np.maximum(1.0 - H_diag, 1e-10)
    return float(np.sqrt(np.mean(residuals ** 2))) * KCAL


def train_nm_krr(traj, symbols, eq_coords_ang, U_vib, sqrt_mass, coord_scale,
                 freqs_cm, out_dir, max_de=50.0,
                 gamma_values=None, alpha_values=None,
                 gamma=None, alpha=None,
                 wall_factor=1.15, wall_stiffness=15.0):
    """Train NM-KRR and save mlpes_nm.pkl."""
    if gamma_values is None:
        gamma_values = DEFAULT_GAMMA_VALUES
    if alpha_values is None:
        alpha_values = DEFAULT_ALPHA_VALUES

    coords = np.array(traj.coordinates)
    e_ha   = np.array(traj.energies)
    dE     = (e_ha - e_ha.min()) * KCAL
    mask   = dE < max_de
    coords = coords[mask]
    e_ha   = e_ha[mask]
    dE     = dE[mask]
    print(f'  Training: {len(coords)} / {len(traj.energies)} frames (ΔE < {max_de} kcal/mol)')

    # Project coordinates to NM space (unscaled, in sqrt(amu)·Bohr)
    n_vib   = U_vib.shape[1]
    eq_flat = eq_coords_ang.flatten() * ANG_TO_BOHR
    delta   = coords.reshape(len(coords), -1) * ANG_TO_BOHR - eq_flat[None, :]
    delta_mw = delta * sqrt_mass[None, :]
    X_q     = delta_mw @ U_vib                     # (M, n_vib) unscaled
    X_q_sc  = X_q / coord_scale[np.newaxis, :]    # (M, n_vib) dimensionless for LOO-CV

    if gamma is not None and alpha is not None:
        best_g, best_a = gamma, alpha
        best_rmse = loo_cv_rmse(X_q_sc, e_ha, best_g, best_a)
        print(f'  Fixed γ={best_g}, α={best_a}, LOO-CV RMSE={best_rmse:.3f} kcal/mol')
    else:
        print(f'  LOO-CV grid: {len(gamma_values)}×{len(alpha_values)} combos ...')
        best_rmse, best_g, best_a = 1e9, None, None
        for g in gamma_values:
            for a in alpha_values:
                rmse = loo_cv_rmse(X_q_sc, e_ha, g, a)
                print(f'    γ={g}  α={a}  LOO-CV={rmse:.3f} kcal/mol')
                if rmse < best_rmse:
                    best_rmse, best_g, best_a = rmse, g, a
        print(f'  Best: γ={best_g}, α={best_a}, LOO-CV RMSE={best_rmse:.3f} kcal/mol')

    # NMKRRPESModel takes unscaled X_train_q and coord_scale; handles scaling internally
    model = NMKRRPESModel(
        eq_coords_ang  = eq_coords_ang,
        U_vib          = U_vib,
        sqrt_mass      = sqrt_mass,
        freqs_vib      = freqs_cm,
        symbols        = symbols,
        gamma          = best_g,
        alpha_reg      = best_a,
        X_train_q      = X_q,
        y_train_ha     = e_ha,
        cv_rmse_kcal   = best_rmse,
        wall_factor    = wall_factor,
        wall_stiffness = wall_stiffness,
        wall_mode      = 'thermal',
        coord_scale    = coord_scale,
    )

    model_path = out_dir / 'mlpes_nm.pkl'
    model.save(str(model_path))
    print(f'  Saved NM-KRR model → {model_path}')
    return model, model_path


# =============================================================================
# State / checkpoint
# =============================================================================

def save_state(state, out_dir):
    p = out_dir / 'state.json'
    with open(p, 'w') as f:
        json.dump(state, f, indent=2)


def load_state(path):
    with open(path) as f:
        return json.load(f)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description='Build NM-KRR PES for an MVKO conformer')
    ap.add_argument('--molecule',    default='mvko_anti_trans',
                    help='Key in test_molecules.py (default: mvko_anti_trans)')
    ap.add_argument('--out-dir',     default=None,
                    help='Output directory (default: outputs/<molecule>_nm_pes_<ts>)')
    ap.add_argument('--restart',     default=None,
                    help='Resume from state.json checkpoint')
    ap.add_argument('--steps',       default='1,2,3,5',
                    help='Comma-separated steps to run (1=opt,2=hessian,3=NM-data,'
                         '4=MD-data,5=train). Default: 1,2,3,5')
    # PSI4 overrides
    ap.add_argument('--eq-coords',   default=None,
                    help='Pre-computed equilibrium geometry .npy (skips Step 1)')
    ap.add_argument('--hessian',     default=None,
                    help='Pre-computed Hessian .npy (skips Step 2)')
    ap.add_argument('--nm-data',     default=None,
                    help='Pre-computed NM displacements .npz (skips Step 3)')
    ap.add_argument('--md-data',     default=None,
                    help='Pre-computed MD data .npz (skips Step 4)')
    # NM displacement parameters
    ap.add_argument('--T-nm',        type=float, default=800.0,
                    help='Temperature for thermal amplitude scaling (K). Default 800')
    ap.add_argument('--n-amplitudes', type=int,  default=5,
                    help='Number of amplitude steps per mode. Default 5')
    ap.add_argument('--max-factor',  type=float, default=2.5,
                    help='Max amplitude as multiple of thermal amplitude. Default 2.5')
    # MD parameters
    ap.add_argument('--md-temps',    default='300,600,1000',
                    help='Comma-separated PSI4 MD temperatures (K). Default 300,600,1000')
    ap.add_argument('--md-steps',    type=int,   default=100,
                    help='MD steps per temperature. Default 100')
    # NM-KRR hyperparameters
    ap.add_argument('--gamma',       type=float, default=None,
                    help='Fixed γ (skip LOO-CV). Default: sweep')
    ap.add_argument('--alpha',       type=float, default=None,
                    help='Fixed α (skip LOO-CV). Default: sweep')
    ap.add_argument('--gamma-values', default=None,
                    help='Comma-separated γ sweep values')
    ap.add_argument('--alpha-values', default=None,
                    help='Comma-separated α sweep values')
    ap.add_argument('--max-de',      type=float, default=50.0,
                    help='Max ΔE filter for training (kcal/mol). Default 50')
    ap.add_argument('--wall-factor',    type=float, default=1.15)
    ap.add_argument('--wall-stiffness', type=float, default=15.0)
    args = ap.parse_args()

    steps = set(int(s.strip()) for s in args.steps.split(','))

    # ── Output directory ──────────────────────────────────────────────────────
    if args.restart:
        state   = load_state(args.restart)
        out_dir = Path(state['out_dir'])
        print(f'Resuming from {args.restart}')
        mol_key = state['molecule']
    else:
        ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
        mol_key = args.molecule
        out_dir = Path(args.out_dir or f'outputs/{mol_key}_nm_pes_{ts}')
        state   = {'molecule': mol_key, 'out_dir': str(out_dir), 'steps_done': []}

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load molecule ─────────────────────────────────────────────────────────
    mol     = get_molecule(mol_key)
    symbols = list(mol.symbols)
    coords0 = np.array(mol.coordinates)
    print(f'Molecule: {mol_key}  ({len(symbols)} atoms)')

    # ── PSI4 ──────────────────────────────────────────────────────────────────
    global psi4
    try:
        import psi4 as _psi4
        psi4 = _psi4
        print(f'PSI4 {psi4.__version__} available')
    except ImportError:
        psi4 = None
        if steps & {1, 2, 3, 4}:
            print('WARNING: PSI4 not available — Steps 1-4 require PSI4')

    # ── Step 1: Geometry optimisation ─────────────────────────────────────────
    if args.eq_coords:
        eq_coords = np.load(args.eq_coords)
        print(f'Loaded eq_coords from {args.eq_coords}')
        state['eq_coords'] = args.eq_coords
    elif 1 in steps:
        print('\n=== Step 1: PSI4 geometry optimisation ===')
        eq_coords, E_opt = psi4_optimize(symbols, coords0)
        eq_path = out_dir / 'eq_coords.npy'
        np.save(eq_path, eq_coords)
        state['eq_coords']  = str(eq_path)
        state['eq_energy']  = E_opt
        state['steps_done'].append(1)
        save_state(state, out_dir)
        print(f'  Optimised energy: {E_opt:.8f} Ha')
        print(f'  Saved → {eq_path}')
    elif 'eq_coords' in state:
        eq_coords = np.load(state['eq_coords'])
    else:
        raise RuntimeError('No equilibrium geometry: run Step 1 or pass --eq-coords')

    # ── Step 2: Hessian ───────────────────────────────────────────────────────
    if args.hessian:
        H = np.load(args.hessian)
        print(f'Loaded Hessian from {args.hessian}')
        state['hessian'] = args.hessian
    elif 2 in steps:
        print('\n=== Step 2: PSI4 Hessian ===')
        H, E_hess = psi4_hessian(symbols, eq_coords)
        h_path = out_dir / 'hessian.npy'
        np.save(h_path, H)
        state['hessian']     = str(h_path)
        state['hess_energy'] = E_hess
        state['steps_done'].append(2)
        save_state(state, out_dir)
        print(f'  Saved → {h_path}')
    elif 'hessian' in state:
        H = np.load(state['hessian'])
    else:
        raise RuntimeError('No Hessian: run Step 2 or pass --hessian')

    # ── Compute normal modes ──────────────────────────────────────────────────
    freqs_cm, U_vib, sqrt_mass, eigenvalues, coord_scale = \
        compute_nm_modes(symbols, eq_coords, H)

    print(f'\n  Normal modes ({len(freqs_cm)} vibrational):')
    neg = np.sum(freqs_cm < 0)
    if neg:
        print(f'  WARNING: {neg} imaginary modes')
    for k, f in enumerate(freqs_cm):
        marker = ' ***' if f < 0 else ''
        print(f'    mode {k+1:2d}: {f:8.1f} cm⁻¹{marker}')

    nm_path = out_dir / 'normal_modes.npz'
    np.savez(nm_path, freqs=freqs_cm, U_vib=U_vib,
             sqrt_mass=sqrt_mass, eigenvalues=eigenvalues,
             coord_scale=coord_scale, eq_coords=eq_coords,
             symbols=np.array(symbols))
    state['normal_modes'] = str(nm_path)
    save_state(state, out_dir)

    # ── Step 3: NM-displaced training data ────────────────────────────────────
    if args.nm_data:
        nm_traj = load_trajectory(args.nm_data)
        print(f'Loaded NM data from {args.nm_data}  ({len(nm_traj.energies)} frames)')
        state['nm_data'] = args.nm_data
    elif 3 in steps:
        print(f'\n=== Step 3: NM-displaced single-points (T={args.T_nm} K) ===')
        nm_traj = collect_nm_data(
            symbols, eq_coords, U_vib, sqrt_mass, coord_scale,
            T_nm=args.T_nm, n_amplitudes=args.n_amplitudes,
            max_factor=args.max_factor, out_dir=out_dir,
        )
        state['nm_data']    = str(out_dir / 'nm_displacements.npz')
        state['steps_done'].append(3)
        save_state(state, out_dir)
    elif 'nm_data' in state:
        nm_traj = load_trajectory(state['nm_data'])
    else:
        nm_traj = None

    # ── Step 4: PSI4 MD data (optional) ──────────────────────────────────────
    md_traj = None
    if args.md_data:
        md_traj = load_trajectory(args.md_data)
        print(f'Loaded MD data from {args.md_data}  ({len(md_traj.energies)} frames)')
        state['md_data'] = args.md_data
    elif 4 in steps:
        print('\n=== Step 4: PSI4 MD data ===')
        from modules.direct_md import DirectMDConfig, DirectMDRunner
        from modules.test_molecules import TestMolecule
        md_temps = [float(t) for t in args.md_temps.split(',')]
        md_parts = []
        for T in md_temps:
            print(f'  MD at {T} K, {args.md_steps} steps ...')
            mol_obj = TestMolecule(
                name=mol_key, formula=mol.formula,
                symbols=symbols, coordinates=eq_coords,
            )
            cfg = DirectMDConfig(
                n_steps=args.md_steps, timestep=0.5, temperature=T,
                output_frequency=1, thermostat='berendsen',
                method=PSI4_METHOD, basis=PSI4_OPTIONS['basis'],
                memory=f'{PSI4_MEM_GB}GB', threads=PSI4_THREADS,
                calculate_dipole=False,
            )
            runner = DirectMDRunner(cfg, output_dir=str(out_dir / f'md_T{int(T)}'))
            t = runner.run(mol_obj)
            if t.dipoles is None:
                t = TrajectoryData(symbols=t.symbols, coordinates=t.coordinates,
                                   energies=t.energies, forces=t.forces,
                                   dipoles=np.zeros((len(t.energies), 3)))
            md_parts.append(t)

        md_traj = md_parts[0]
        for p in md_parts[1:]:
            md_traj = TrajectoryData(
                symbols=md_traj.symbols,
                coordinates=np.concatenate([md_traj.coordinates, p.coordinates]),
                energies=np.concatenate([md_traj.energies, p.energies]),
                forces=np.concatenate([md_traj.forces, p.forces]),
                dipoles=np.concatenate([md_traj.dipoles, p.dipoles]),
            )
        md_path = out_dir / 'md_training.npz'
        save_trajectory(md_traj, str(md_path))
        state['md_data']    = str(md_path)
        state['steps_done'].append(4)
        save_state(state, out_dir)
        print(f'  MD data: {len(md_traj.energies)} frames')

    # ── Merge training data ───────────────────────────────────────────────────
    if nm_traj is None:
        raise RuntimeError('No NM training data available — run Step 3')

    if md_traj is not None:
        all_coords = np.concatenate([nm_traj.coordinates, md_traj.coordinates])
        all_e      = np.concatenate([nm_traj.energies,    md_traj.energies])
        all_f_nm   = nm_traj.forces if nm_traj.forces is not None \
                     else np.zeros_like(nm_traj.coordinates)
        all_f_md   = md_traj.forces if md_traj.forces is not None \
                     else np.zeros_like(md_traj.coordinates)
        all_f      = np.concatenate([all_f_nm, all_f_md])
        all_d      = np.concatenate([nm_traj.dipoles, md_traj.dipoles])
    else:
        all_coords = nm_traj.coordinates
        all_e      = nm_traj.energies
        all_f      = nm_traj.forces if nm_traj.forces is not None \
                     else np.zeros_like(nm_traj.coordinates)
        all_d      = nm_traj.dipoles

    combined = TrajectoryData(
        symbols=symbols, coordinates=all_coords,
        energies=all_e, forces=all_f, dipoles=all_d,
    )
    comb_path = out_dir / 'combined_training_data.npz'
    save_trajectory(combined, str(comb_path))
    state['combined_data'] = str(comb_path)
    save_state(state, out_dir)
    print(f'\n  Combined training data: {len(all_e)} frames → {comb_path}')

    # ── Step 5: Train NM-KRR ─────────────────────────────────────────────────
    if 5 in steps:
        print('\n=== Step 5: NM-KRR training ===')
        gamma_vals = ([float(g) for g in args.gamma_values.split(',')]
                      if args.gamma_values else None)
        alpha_vals = ([float(a) for a in args.alpha_values.split(',')]
                      if args.alpha_values else None)

        model, model_path = train_nm_krr(
            combined, symbols, eq_coords, U_vib, sqrt_mass, coord_scale, freqs_cm,
            out_dir=out_dir, max_de=args.max_de,
            gamma_values=gamma_vals, alpha_values=alpha_vals,
            gamma=args.gamma, alpha=args.alpha,
            wall_factor=args.wall_factor, wall_stiffness=args.wall_stiffness,
        )
        state['model'] = str(model_path)
        state['steps_done'].append(5)
        save_state(state, out_dir)

        print(f'\n  NM frequencies:')
        for k, f in enumerate(freqs_cm):
            print(f'    mode {k+1:2d}: {f:8.1f} cm⁻¹')

        print(f'\n=== Done ===')
        print(f'NM-KRR model: {model_path}')
        print(f'\nRun IR spectrum:')
        dipole_data = 'outputs/mvko_dipoles_stretch_20260420/training_with_dipoles.npz'
        print(f'  python3 ir_md_spectrum.py \\')
        print(f'    --nm-pes-model {model_path} \\')
        print(f'    --training-data {dipole_data} \\')
        print(f'    --steps 30000 --temp 300 --preminimize \\')
        print(f'    --zpe-min-freq 50 --zpe-max-freq 4000 \\')
        print(f'    --n-trajectories 5 \\')
        print(f'    --nm-pes-bond-wall-factor {args.wall_factor} '
              f'--nm-pes-bond-wall-stiffness {args.wall_stiffness} \\')
        print(f'    --max-bond-extension 1.35 \\')
        print(f'    --output-dir outputs/{mol_key}_ir_300K')

        print(f'\nCollect anti-trans dipoles (AFTER IR model validated):')
        print(f'  python3 collect_ch_displaced_dipoles.py \\')
        print(f'    --nm-pes-model {model_path} \\')
        print(f'    --out-dir outputs/{mol_key}_dipoles')


if __name__ == '__main__':
    main()
