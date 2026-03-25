#!/usr/bin/env python3
"""
MVKO ML-PES IR Spectrum Workflow
=================================
Methyl vinyl ketone oxide  (CH₂=CH)(CH₃)COO  —  C₄H₆O₂  (12 atoms)

A Criegee intermediate formed from ozonolysis of methyl vinyl ketone (MVK).
Its unimolecular and bimolecular reactions are important in atmospheric
oxidation chemistry.

Pipeline
--------
  Step 1  PSI4 geometry optimisation (B3LYP/6-31G*)
  Step 2  PSI4 Hessian → normal mode frequencies & vectors
  Step 3  NM-displaced geometries + PSI4 single-points → training data
  Step 4  Multi-temperature PSI4 MD → additional training data
  Step 5  Initial ML-PES training (KRR, GridSearch over γ, α)
  Step 6  Adaptive refinement loop:
            a) ML-MD trajectory
            b) PSI4 single-points on high-energy frames
            c) Add to training set and retrain
            d) Repeat until mean error < convergence threshold
  Step 7  IR spectrum (bakken pre-min + ZPE-floor ML-MD → dipole ACF)

Usage
-----
    python3 mvko_workflow.py                        # full run
    python3 mvko_workflow.py --steps 1,2,3          # only optimisation + NM + NM data
    python3 mvko_workflow.py --restart outputs/mvko_<ts>/state.json

Notes for MVKO vs CH₂OO
------------------------
  • 12 atoms (vs 5)  →  n_desc = 78 Coulomb features (vs 15)
  • 30 vibrational modes (vs 9)
  • Larger descriptor space → smaller optimal γ (try 5× lower than CH₂OO)
  • More training frames needed: target ≥ 600 before IR run
  • PSI4 per-point cost ~ 5× higher; use DF-B3LYP for speed
  • Torsional modes (methyl, vinyl) need explicit NM + high-T MD coverage
  • Singlet ground state  (charge=0, mult=1)
"""

import sys
import os
import json
import argparse
import pickle
import datetime
import time
import numpy as np
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'modules'))

# ── Framework imports ─────────────────────────────────────────────────────────
import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')

from data_formats import TrajectoryData, save_trajectory, load_trajectory
from ml_pes import MLPESConfig, MLPESTrainer
from normal_modes import (
    compute_hessian_psi4, compute_normal_modes,
    generate_nm_displacements, ATOMIC_MASSES, FREQ_CONV,
    BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR, KB_HARTREE_PER_K,
)
from test_molecules import get_molecule
from bakken import MLPESDriver, minimize_geometry, run_md

# ── PSI4 ──────────────────────────────────────────────────────────────────────
try:
    import psi4
    PSI4_AVAILABLE = True
    print(f"PSI4 {psi4.__version__} available")
except ImportError:
    PSI4_AVAILABLE = False
    print("PSI4 not available — Steps 1-4 and 6 require PSI4")

# ── Constants ─────────────────────────────────────────────────────────────────
HARTREE_TO_KCAL  = 627.509474
ANG_TO_BOHR      = ANGSTROM_TO_BOHR

# PSI4 settings (consistent throughout — never mix)
PSI4_OPTIONS = {
    'basis':         '6-31G*',
    'scf_type':      'df',          # density-fitted DF-B3LYP for speed
    'reference':     'rhf',
    'maxiter':       200,
    'e_convergence': 1e-7,
    'd_convergence': 1e-7,
}
PSI4_METHOD  = 'b3lyp'
PSI4_MEM_GB  = 4
PSI4_THREADS = 4

MOLECULE_NAME = 'mvko'

# ── Adaptive refinement parameters ───────────────────────────────────────────
ADAPT_N_ROUNDS       = 5          # max adaptive rounds
ADAPT_MD_STEPS       = 300        # ML-MD steps per round (dense)
ADAPT_T_K            = 500.0      # temperature for adaptive sampling (K) — higher = more coverage
ADAPT_TIMESTEP_FS    = 0.5        # fs
ADAPT_N_FRAMES_PSI4  = 20        # PSI4 single-points per round (sample high-error frames)
ADAPT_CONV_KCAL      = 0.5        # convergence: mean error < this (kcal/mol)

# ── Hyperparameter grid for MVKO ──────────────────────────────────────────────
# γ scaled ~5× smaller than CH₂OO (γ=0.01) because n_desc is ~5× larger.
# The grid search will find the optimal pair.
GAMMA_RANGE = [0.0002, 0.0005, 0.001, 0.002, 0.005]
ALPHA_RANGE = [1e-5, 1e-4, 1e-3, 1e-2]


# =============================================================================
# PSI4 helpers
# =============================================================================

def _psi4_setup():
    """Configure PSI4 options (call before every PSI4 computation)."""
    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.be_quiet()
    psi4.set_memory(f'{PSI4_MEM_GB} GB')
    psi4.set_num_threads(PSI4_THREADS)
    psi4.set_options(PSI4_OPTIONS)


def _mol_str(symbols, coords):
    """Build a PSI4 molecule string (Angstrom, no reorient/recentre)."""
    lines = ['0 1']
    for s, c in zip(symbols, coords):
        lines.append(f'{s}  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}')
    lines += ['units angstrom', 'no_reorient', 'no_com']
    return '\n'.join(lines)


def psi4_singlepoint(symbols, coords):
    """
    PSI4 single-point energy, gradient, and dipole at the given geometry.

    Returns:
        energy_ha  : float (Hartree)
        forces_hpa : (N, 3) ndarray (Hartree/Angstrom)
        dipole_D   : (3,) ndarray (Debye)
    """
    _psi4_setup()
    mol = psi4.geometry(_mol_str(symbols, coords))
    grad_obj, wfn = psi4.gradient(f'{PSI4_METHOD}/{PSI4_OPTIONS["basis"]}',
                                  molecule=mol, return_wfn=True,
                                  properties=['dipole'])
    energy_ha = wfn.energy()

    # Forces: -gradient (PSI4 gradient in Hartree/Bohr → Hartree/Ang)
    grad_bohr = np.array(grad_obj)                         # (N, 3) Ha/Bohr
    forces    = -grad_bohr / ANG_TO_BOHR                  # (N, 3) Ha/Ang

    # Dipole — request via properties=['dipole']; 'SCF DIPOLE' vector is in AU
    AU_TO_DEBYE = 2.541746
    dipole_D = np.zeros(3)
    try:
        dip_au = np.array(psi4.variable('SCF DIPOLE'))   # (3,) AU (e·bohr)
        dipole_D = dip_au * AU_TO_DEBYE
    except Exception:
        try:
            # Fallback: oeprop path (stores in Debye directly)
            psi4.oeprop(wfn, 'DIPOLE')
            dipole_D = np.array([wfn.variable(f'DIPOLE {ax}')
                                 for ax in ['X', 'Y', 'Z']])
        except Exception:
            pass

    return energy_ha, forces, dipole_D


def psi4_optimize(symbols, coords_init):
    """
    Full PSI4 geometry optimisation (B3LYP/6-31G*).

    Returns:
        coords_opt : (N, 3) Angstrom — optimised geometry
        energy_opt : float Hartree
    """
    _psi4_setup()
    mol = psi4.geometry(_mol_str(symbols, coords_init))
    print(f"  Optimising {len(symbols)}-atom geometry with PSI4 "
          f"({PSI4_METHOD}/{PSI4_OPTIONS['basis']}) ...")

    # psi4.optimize returns energy; optimised geometry is stored in mol
    energy_opt = psi4.optimize(f'{PSI4_METHOD}/{PSI4_OPTIONS["basis"]}',
                               molecule=mol)

    # Extract optimised Cartesian coordinates (PSI4 returns in Bohr)
    geom_bohr = np.array(mol.geometry())            # (N, 3) Bohr
    coords_opt = geom_bohr * BOHR_TO_ANGSTROM       # → Angstrom
    return coords_opt, float(energy_opt)


# =============================================================================
# Training data helpers
# =============================================================================

def _geometry_ok(coords, min_dist=0.70):
    """
    Return True if no pair of atoms is closer than min_dist Angstrom.

    Displaced NM geometries can push atoms unphysically close for large
    amplitudes on low-frequency (torsional) modes; PSI4 SCF fails in those
    cases.  Filtering before the PSI4 call avoids wasted CPU time.
    """
    n = len(coords)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(coords[i] - coords[j]) < min_dist:
                return False
    return True


def collect_nm_displacements(symbols, coords_eq, hessian_bohr2,
                              T_nm=800.0, n_amplitudes=5, max_factor=2.5):
    """
    Compute PSI4 single-points at NM-displaced geometries.

    Displacement amplitude for mode k at temperature T:
        a_k = sqrt(kT / ω_k²·m_k)  [Å, thermal amplitude]
    Displaced by ±n × a_k for n = 1 .. n_amplitudes (capped at max_factor).

    Geometries with any interatomic distance < 0.70 Å are skipped (prevents
    SCF convergence failures from over-displaced low-frequency modes).
    PSI4 failures on individual frames are caught and skipped gracefully.

    Returns TrajectoryData with energies, forces, and dipoles.
    """
    frequencies, eigvecs_mw, eigenvalues, mass_vec = compute_normal_modes(
        symbols, hessian_bohr2
    )
    print(f"\n  PSI4 NM frequencies (cm⁻¹): "
          + "  ".join(f"{f:.0f}" for f in frequencies))

    displaced_coords = generate_nm_displacements(
        symbols, coords_eq, eigvecs_mw, eigenvalues, mass_vec,
        T=T_nm, n_amplitudes=n_amplitudes, max_factor=max_factor,
    )
    n_disp = len(displaced_coords)
    print(f"  Generated {n_disp} NM-displaced geometries (T_nm={T_nm:.0f} K, "
          f"±{n_amplitudes} amplitudes, max_factor={max_factor})")

    all_coords, all_energies, all_forces, all_dipoles = [], [], [], []
    n_skipped_geom = 0
    n_skipped_scf  = 0
    t0 = time.time()

    for k, (c, mode_idx, factor) in enumerate(displaced_coords):
        # Geometry validity check — skip over-displaced frames
        if not _geometry_ok(c):
            n_skipped_geom += 1
            continue

        try:
            e, f, d = psi4_singlepoint(symbols, c)
        except Exception as exc:
            n_skipped_scf += 1
            print(f"  NM SP {k+1}/{n_disp}  mode={mode_idx} fac={factor:+.2f}  "
                  f"SKIPPED ({type(exc).__name__})")
            continue

        all_coords.append(c)
        all_energies.append(e)
        all_forces.append(f)
        all_dipoles.append(d)

        if len(all_coords) % 10 == 0 or k == n_disp - 1:
            elapsed = time.time() - t0
            print(f"  NM SP {len(all_coords)} collected ({k+1}/{n_disp} attempted)  "
                  f"mode={mode_idx} fac={factor:+.2f}  "
                  f"E={e*HARTREE_TO_KCAL:.2f} kcal/mol  [{elapsed:.0f}s]")

    print(f"\n  NM displacements: {len(all_coords)} collected, "
          f"{n_skipped_geom} skipped (clash), {n_skipped_scf} skipped (SCF fail)")

    return TrajectoryData(
        symbols=symbols,
        coordinates=np.array(all_coords),
        energies=np.array(all_energies),
        forces=np.array(all_forces),
        dipoles=np.array(all_dipoles),
    )


def collect_md_data(symbols, coords_eq, temperature, out_dir, n_steps=200, dt=0.5):
    """
    PSI4 direct-MD (Velocity-Verlet + Berendsen) at a single temperature.

    Uses DirectMDRunner from modules/direct_md.py.
    Returns TrajectoryData.
    """
    from modules.direct_md import DirectMDConfig, DirectMDRunner
    from modules.test_molecules import TestMolecule

    mol_obj = TestMolecule(
        name=MOLECULE_NAME, formula='C4H6O2',
        symbols=symbols, coordinates=coords_eq,
    )
    config = DirectMDConfig(
        n_steps=n_steps,
        timestep=dt,
        temperature=temperature,
        output_frequency=1,
        thermostat='berendsen',
        method=PSI4_METHOD,
        basis=PSI4_OPTIONS['basis'],
        memory=f'{PSI4_MEM_GB}GB',
        threads=PSI4_THREADS,
        calculate_dipole=False,
    )
    # DirectMDRunner(config) — molecule passed to run()
    runner = DirectMDRunner(config, output_dir=str(out_dir / f'direct_md_T{int(temperature)}'))
    traj   = runner.run(mol_obj)   # returns TrajectoryData directly

    # Ensure dipoles array is present (DirectMD may not compute them by default)
    if traj.dipoles is None:
        traj = TrajectoryData(
            symbols=traj.symbols,
            coordinates=traj.coordinates,
            energies=traj.energies,
            forces=traj.forces,
            dipoles=np.zeros((len(traj.energies), 3)),
        )
    return traj


def merge_trajectories(*trajs):
    """Concatenate multiple TrajectoryData objects."""
    symbols = trajs[0].symbols
    coords   = np.concatenate([t.coordinates for t in trajs], axis=0)
    energies = np.concatenate([t.energies    for t in trajs])
    forces   = np.concatenate([t.forces      for t in trajs], axis=0) \
               if all(t.forces is not None for t in trajs) else None
    dipoles  = np.concatenate([t.dipoles     for t in trajs], axis=0) \
               if all(t.dipoles is not None for t in trajs) else None
    return TrajectoryData(
        symbols=symbols,
        coordinates=coords,
        energies=energies,
        forces=forces,
        dipoles=dipoles,
    )


# =============================================================================
# ML-PES training
# =============================================================================

def train_mlpes(traj, out_path, gamma_range=None, alpha_range=None):
    """
    Train ML-PES via KRR with grid-search over γ and α.

    Returns MLPESTrainer with the best model saved to out_path.
    """
    config = MLPESConfig(
        tune_hyperparameters=True,
        gamma_range=gamma_range or GAMMA_RANGE,
        alpha_range=alpha_range or ALPHA_RANGE,
    )
    trainer = MLPESTrainer(config)
    trainer.train(traj)
    trainer.save(out_path)
    h = trainer.training_history
    print(f"  ML-PES trained  γ={h.get('best_gamma')}  α={h.get('best_alpha')}  "
          f"RMSE={h.get('best_rmse_kcal', float('nan')):.4f} kcal/mol  "
          f"({len(traj.coordinates)} frames)")
    return trainer


# =============================================================================
# Adaptive refinement
# =============================================================================

def adaptive_round(driver, symbols, coords_eq, training_traj,
                    out_dir, round_idx, seed=None):
    """
    One adaptive refinement round:
      1. Run ML-MD to explore the ML-PES.
      2. Select ADAPT_N_FRAMES_PSI4 frames (highest ML energy spread).
      3. Run PSI4 single-points on those frames.
      4. Return new TrajectoryData with the PSI4 points.

    Returns:
        new_traj : TrajectoryData  (PSI4 single-points, ready to merge)
        mean_err : float  (mean |E_ML - E_PSI4| in kcal/mol over sampled frames)
    """
    rng_seed = seed if seed is not None else 100 + round_idx
    print(f"\n  [Round {round_idx}] ML-MD exploration "
          f"({ADAPT_MD_STEPS} steps, {ADAPT_T_K:.0f} K) ...")
    md_result = run_md(
        driver, coords_eq,
        n_steps=ADAPT_MD_STEPS,
        temperature=ADAPT_T_K,
        timestep=ADAPT_TIMESTEP_FS,
        save_every=1,
        seed=rng_seed,
        preminimize=True,
        preminimize_steps=200,
        preminimize_tol=0.005,
    )
    traj_coords  = md_result['coords_traj']   # (n_frames, N, 3)
    traj_e_ml    = md_result['energies_ml']   # (n_frames,)
    n_frames     = len(traj_coords)

    # Select frames with largest spread in ML energy (covers PES broadly)
    # + a few random frames (exploration)
    e_range = traj_e_ml.max() - traj_e_ml.min()
    n_top   = max(1, ADAPT_N_FRAMES_PSI4 * 3 // 4)   # top 75% by energy spread
    n_rand  = ADAPT_N_FRAMES_PSI4 - n_top

    idx_top  = np.argsort(np.abs(traj_e_ml - traj_e_ml.mean()))[-n_top:]
    rng      = np.random.default_rng(rng_seed + 1)
    idx_rand = rng.choice(n_frames, size=n_rand, replace=False)
    sel_idx  = np.unique(np.concatenate([idx_top, idx_rand]))[:ADAPT_N_FRAMES_PSI4]

    print(f"  [Round {round_idx}] PSI4 single-points on {len(sel_idx)} frames "
          f"(ML energy range = {e_range*HARTREE_TO_KCAL:.2f} kcal/mol) ...")

    new_coords, new_E, new_F, new_D = [], [], [], []
    errors = []
    for k, idx in enumerate(sel_idx):
        c = traj_coords[idx]
        e_ml = traj_e_ml[idx]
        try:
            e_psi4, f_psi4, d_psi4 = psi4_singlepoint(symbols, c)
        except Exception as exc:
            print(f"    frame {idx}: PSI4 failed ({exc}), skipping")
            continue
        err = abs(e_psi4 - e_ml) * HARTREE_TO_KCAL
        errors.append(err)
        new_coords.append(c)
        new_E.append(e_psi4)
        new_F.append(f_psi4)
        new_D.append(d_psi4)
        print(f"    frame {idx:4d}  E_ML={e_ml*HARTREE_TO_KCAL:.2f}  "
              f"E_PSI4={e_psi4*HARTREE_TO_KCAL:.2f}  err={err:.3f} kcal/mol")

    mean_err = float(np.mean(errors)) if errors else float('nan')
    print(f"  [Round {round_idx}] mean |ΔE| = {mean_err:.3f} kcal/mol  "
          f"({len(new_E)} PSI4 frames)")

    new_traj = TrajectoryData(
        symbols=symbols,
        coordinates=np.array(new_coords),
        energies=np.array(new_E),
        forces=np.array(new_F),
        dipoles=np.array(new_D),
    )
    return new_traj, mean_err


# =============================================================================
# State / checkpoint helpers
# =============================================================================

def _save_state(state_path, state):
    with open(state_path, 'w') as fh:
        json.dump(state, fh, indent=2)


def _load_state(state_path):
    with open(state_path) as fh:
        return json.load(fh)


# =============================================================================
# Main workflow
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MVKO ML-PES → IR spectrum workflow')
    parser.add_argument('--steps', default='1,2,3,4,5,6,7',
                        help='Comma-separated step numbers to run (default: all)')
    parser.add_argument('--restart', default=None,
                        help='Path to state.json from a previous run to resume')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (auto-timestamped if omitted)')
    parser.add_argument('--T-nm', type=float, default=800.0,
                        help='NM displacement temperature in K (default 800; '
                             'higher values increase coverage but risk SCF failures '
                             'on low-frequency torsional modes)')
    parser.add_argument('--n-amplitudes', type=int, default=5,
                        help='NM displacement amplitudes (default 5 → ±5 per mode)')
    parser.add_argument('--md-temps', default='300,600,1000',
                        help='Comma-separated PSI4-MD temperatures in K (default 300,600,1000)')
    parser.add_argument('--md-steps', type=int, default=200,
                        help='PSI4-MD steps per temperature (default 200)')
    parser.add_argument('--adapt-rounds', type=int, default=ADAPT_N_ROUNDS,
                        help=f'Adaptive refinement rounds (default {ADAPT_N_ROUNDS})')
    parser.add_argument('--adapt-conv', type=float, default=ADAPT_CONV_KCAL,
                        help=f'Convergence threshold kcal/mol (default {ADAPT_CONV_KCAL})')
    parser.add_argument('--ir-steps', type=int, default=30000,
                        help='ML-MD steps for IR spectrum (default 30000)')
    parser.add_argument('--ir-temp', type=float, default=300.0,
                        help='Temperature for IR MD run (default 300 K)')
    args = parser.parse_args()

    steps_to_run = {int(s) for s in args.steps.split(',')}
    md_temps     = [float(t) for t in args.md_temps.split(',')]

    # ── Output directory & state ───────────────────────────────────────────
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path('outputs') / f'mvko_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = out_dir / 'state.json'
    if args.restart and Path(args.restart).exists():
        state = _load_state(args.restart)
        print(f"  Resuming from {args.restart}")
    else:
        state = {}

    print(f"\n{'='*70}")
    print(f"  MVKO ML-PES IR SPECTRUM WORKFLOW")
    print(f"{'='*70}")
    print(f"  Steps to run : {sorted(steps_to_run)}")
    print(f"  Output dir   : {out_dir}")

    # ── Molecule from library ──────────────────────────────────────────────
    mol      = get_molecule(MOLECULE_NAME)
    symbols  = mol.symbols
    coords0  = mol.coordinates.copy()

    print(f"  Molecule     : {mol.formula}  ({len(symbols)} atoms)")
    print(f"  Description  : {mol.description}")

    # ======================================================================
    # Step 1 — PSI4 geometry optimisation
    # ======================================================================
    if 1 in steps_to_run:
        if not PSI4_AVAILABLE:
            print("\n[Step 1] SKIPPED — PSI4 not available")
        elif 'opt_coords' in state and 'opt_energy' in state:
            print("\n[Step 1] Already done (found in state), loading ...")
            coords_opt = np.array(state['opt_coords'])
            E_opt      = state['opt_energy']
        else:
            print(f"\n{'─'*70}")
            print(f"[Step 1] PSI4 geometry optimisation  ({PSI4_METHOD}/{PSI4_OPTIONS['basis']})")
            print(f"{'─'*70}")
            t0 = time.time()
            coords_opt, E_opt = psi4_optimize(symbols, coords0)
            elapsed = time.time() - t0
            print(f"  Optimised energy : {E_opt:.8f} Ha  ({E_opt*HARTREE_TO_KCAL:.2f} kcal/mol)")
            print(f"  Elapsed          : {elapsed:.0f} s")
            print(f"\n  Optimised geometry (Angstrom):")
            for s, c in zip(symbols, coords_opt):
                print(f"    {s:2s}  {c[0]:10.6f}  {c[1]:10.6f}  {c[2]:10.6f}")

            # Save
            xyz_path = out_dir / 'mvko_optimised.xyz'
            with open(xyz_path, 'w') as fh:
                fh.write(f'{len(symbols)}\n')
                fh.write(f'MVKO B3LYP/6-31G* optimised  E={E_opt:.8f} Ha\n')
                for s, c in zip(symbols, coords_opt):
                    fh.write(f'{s}  {c[0]:.8f}  {c[1]:.8f}  {c[2]:.8f}\n')
            print(f"  XYZ saved        : {xyz_path}")

            state['opt_coords']  = coords_opt.tolist()
            state['opt_energy']  = float(E_opt)
            state['opt_xyz']     = str(xyz_path)
            _save_state(state_path, state)
    else:
        # Use initial guess if step 1 skipped
        coords_opt = coords0.copy()
        E_opt      = None

    # Subsequent steps use the optimised geometry
    if 'opt_coords' in state:
        coords_opt = np.array(state['opt_coords'])
    coords_eq = coords_opt.copy()

    # ======================================================================
    # Step 2 — PSI4 Hessian & normal mode analysis
    # ======================================================================
    hessian_bohr2 = None
    if 2 in steps_to_run:
        if not PSI4_AVAILABLE:
            print("\n[Step 2] SKIPPED — PSI4 not available")
        elif 'hessian_path' in state:
            print("\n[Step 2] Already done, loading Hessian ...")
            hessian_bohr2 = np.load(state['hessian_path'])
        else:
            print(f"\n{'─'*70}")
            print(f"[Step 2] PSI4 Hessian & normal mode analysis")
            print(f"{'─'*70}")
            t0 = time.time()
            hessian_bohr2 = compute_hessian_psi4(symbols, coords_eq,
                                                  method=PSI4_METHOD,
                                                  basis=PSI4_OPTIONS['basis'])
            elapsed = time.time() - t0
            frequencies, _, _, _ = compute_normal_modes(symbols, hessian_bohr2)
            print(f"  PSI4 Hessian computed in {elapsed:.0f} s")
            print(f"\n  PSI4 normal mode frequencies (cm⁻¹):")
            for k, freq in enumerate(frequencies):
                print(f"    Mode {k+1:2d}: {freq:8.1f} cm⁻¹")

            hess_path = out_dir / 'psi4_hessian.npy'
            np.save(hess_path, hessian_bohr2)
            state['hessian_path'] = str(hess_path)
            _save_state(state_path, state)
    elif 'hessian_path' in state:
        hessian_bohr2 = np.load(state['hessian_path'])

    # ======================================================================
    # Step 3 — NM-displaced geometries + PSI4 single-points
    # ======================================================================
    if 3 in steps_to_run:
        if not PSI4_AVAILABLE:
            print("\n[Step 3] SKIPPED — PSI4 not available")
        elif hessian_bohr2 is None:
            print("\n[Step 3] SKIPPED — no Hessian available (run Step 2 first)")
        elif 'nm_data_path' in state:
            print("\n[Step 3] Already done, loading NM training data ...")
        else:
            print(f"\n{'─'*70}")
            print(f"[Step 3] NM-displaced geometries + PSI4 single-points")
            print(f"  T_nm={args.T_nm:.0f} K,  ±{args.n_amplitudes} amplitudes")
            print(f"{'─'*70}")
            t0 = time.time()
            nm_traj = collect_nm_displacements(
                symbols, coords_eq, hessian_bohr2,
                T_nm=args.T_nm,
                n_amplitudes=args.n_amplitudes,
            )
            elapsed = time.time() - t0
            nm_path = out_dir / 'nm_training_data.npz'
            save_trajectory(nm_traj, str(nm_path))
            print(f"\n  NM data: {len(nm_traj.coordinates)} frames in {elapsed:.0f} s")
            print(f"  Saved: {nm_path}")
            state['nm_data_path'] = str(nm_path)
            _save_state(state_path, state)

    # ======================================================================
    # Step 4 — Multi-temperature PSI4 MD
    # ======================================================================
    if 4 in steps_to_run:
        if not PSI4_AVAILABLE:
            print("\n[Step 4] SKIPPED — PSI4 not available")
        elif 'md_data_paths' in state and len(state['md_data_paths']) >= len(md_temps):
            print("\n[Step 4] Already done, loading MD training data ...")
        else:
            print(f"\n{'─'*70}")
            print(f"[Step 4] Multi-temperature PSI4 MD")
            print(f"  Temperatures: {md_temps} K,  steps/T: {args.md_steps}")
            print(f"{'─'*70}")
            md_paths = state.get('md_data_paths', [])
            done_temps = set(state.get('md_done_temps', []))
            for T in md_temps:
                if T in done_temps:
                    print(f"\n  T={T:.0f} K: already done, skipping")
                    continue
                print(f"\n  Running PSI4-MD at T={T:.0f} K ({args.md_steps} steps) ...")
                t0 = time.time()
                md_traj = collect_md_data(symbols, coords_eq,
                                          temperature=T,
                                          out_dir=out_dir,
                                          n_steps=args.md_steps)
                elapsed = time.time() - t0
                p = out_dir / f'md_T{int(T)}.npz'
                save_trajectory(md_traj, str(p))
                md_paths.append(str(p))
                done_temps.add(T)
                print(f"  T={T:.0f} K: {len(md_traj.coordinates)} frames in {elapsed:.0f} s")
                state['md_data_paths']  = md_paths
                state['md_done_temps']  = list(done_temps)
                _save_state(state_path, state)

    # ======================================================================
    # Step 5 — Initial ML-PES training
    # ======================================================================
    if 5 in steps_to_run:
        print(f"\n{'─'*70}")
        print(f"[Step 5] Initial ML-PES training")
        print(f"{'─'*70}")

        # Gather all available training data
        trajs_to_merge = []

        # Equilibrium point
        if PSI4_AVAILABLE and 'opt_energy' in state:
            e0, f0, d0 = psi4_singlepoint(symbols, coords_eq)
            eq_traj = TrajectoryData(
                symbols=symbols,
                coordinates=np.array([coords_eq]),
                energies=np.array([e0]),
                forces=np.array([f0]),
                dipoles=np.array([d0]),
            )
            trajs_to_merge.append(eq_traj)

        if 'nm_data_path' in state:
            trajs_to_merge.append(load_trajectory(state['nm_data_path']))

        for p in state.get('md_data_paths', []):
            trajs_to_merge.append(load_trajectory(p))

        if not trajs_to_merge:
            print("  No training data found — cannot train. Run Steps 3-4 first.")
            return

        combined_traj = merge_trajectories(*trajs_to_merge)
        combined_path = out_dir / 'combined_training_data.npz'
        save_trajectory(combined_traj, str(combined_path))
        state['combined_data_path'] = str(combined_path)

        model_path = str(out_dir / 'mlpes_initial.pkl')
        trainer    = train_mlpes(combined_traj, model_path)
        state['model_path']         = model_path
        state['n_training_frames']  = len(combined_traj.coordinates)
        _save_state(state_path, state)
        print(f"  Training data    : {len(combined_traj.coordinates)} frames")
        print(f"  Model saved      : {model_path}")

    # ======================================================================
    # Step 6 — Adaptive refinement loop
    # ======================================================================
    if 6 in steps_to_run:
        if not PSI4_AVAILABLE:
            print("\n[Step 6] SKIPPED — PSI4 not available")
        elif 'model_path' not in state:
            print("\n[Step 6] SKIPPED — no model found (run Step 5 first)")
        else:
            print(f"\n{'─'*70}")
            print(f"[Step 6] Adaptive refinement loop  "
                  f"(max {args.adapt_rounds} rounds, "
                  f"conv={args.adapt_conv} kcal/mol)")
            print(f"{'─'*70}")

            adapt_done  = state.get('adapt_rounds_done', 0)
            adapt_errs  = state.get('adapt_errors', [])
            model_path  = state['model_path']
            data_path   = state.get('combined_data_path',
                                    state.get('nm_data_path'))

            for rnd in range(adapt_done + 1, args.adapt_rounds + 1):
                print(f"\n  ── Adaptive round {rnd}/{args.adapt_rounds} ──")
                driver = MLPESDriver(model_path)
                new_traj, mean_err = adaptive_round(
                    driver, symbols, coords_eq,
                    load_trajectory(data_path),
                    out_dir, rnd, seed=200 + rnd * 7,
                )

                # Save new PSI4 frames
                new_path = out_dir / f'adapt_round{rnd}.npz'
                save_trajectory(new_traj, str(new_path))

                # Merge and retrain
                all_data  = merge_trajectories(load_trajectory(data_path), new_traj)
                data_path = str(out_dir / 'combined_training_data.npz')
                save_trajectory(all_data, data_path)
                state['combined_data_path'] = data_path

                model_path = str(out_dir / f'mlpes_adapt_round{rnd}.pkl')
                trainer    = train_mlpes(all_data, model_path)
                state['model_path']        = model_path
                state['n_training_frames'] = len(all_data.coordinates)

                adapt_errs.append(float(mean_err))
                state['adapt_rounds_done'] = rnd
                state['adapt_errors']      = adapt_errs
                _save_state(state_path, state)

                print(f"  Round {rnd} complete  "
                      f"mean|ΔE|={mean_err:.3f} kcal/mol  "
                      f"total frames={len(all_data.coordinates)}")

                if mean_err <= args.adapt_conv:
                    print(f"\n  Converged at round {rnd} "
                          f"(mean|ΔE|={mean_err:.3f} < {args.adapt_conv} kcal/mol)")
                    break
            else:
                print(f"\n  Max rounds ({args.adapt_rounds}) reached without convergence.")
                print(f"  Final errors: {[f'{e:.3f}' for e in adapt_errs]} kcal/mol")

    # ======================================================================
    # Step 7 — IR spectrum
    # ======================================================================
    if 7 in steps_to_run:
        print(f"\n{'─'*70}")
        print(f"[Step 7] IR spectrum via ML-MD dipole ACF")
        print(f"{'─'*70}")

        if 'model_path' not in state:
            print("  No model found — run Steps 5-6 first.")
            return

        data_path   = state.get('combined_data_path',
                                state.get('nm_data_path'))
        model_path  = state['model_path']
        ir_out      = out_dir / 'ir_spectrum'
        n_frames    = state.get('n_training_frames', '?')

        print(f"  Model       : {model_path}")
        print(f"  Data        : {data_path}")
        print(f"  Frames      : {n_frames}")
        print(f"  MD steps    : {args.ir_steps}  ({args.ir_temp:.0f} K)")

        cmd = (
            f"python3 ir_md_spectrum.py"
            f" --model {model_path}"
            f" --training-data {data_path}"
            f" --steps {args.ir_steps}"
            f" --temp {args.ir_temp}"
            f" --timestep 0.5 --save-every 1"
            f" --preminimize"
            f" --zpe-min-freq 50 --zpe-max-freq 4000"
            f" --output-dir {ir_out}"
        )
        print(f"\n  Running: {cmd}\n")
        ret = os.system(cmd)
        if ret == 0:
            state['ir_output_dir'] = str(ir_out)
            _save_state(state_path, state)
        else:
            print(f"  ir_md_spectrum.py exited with code {ret}")

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  MVKO WORKFLOW COMPLETE")
    print(f"{'='*70}")
    print(f"  Output dir       : {out_dir}")
    print(f"  State file       : {state_path}")
    if 'model_path' in state:
        print(f"  Final model      : {state['model_path']}")
    if 'n_training_frames' in state:
        print(f"  Training frames  : {state['n_training_frames']}")
    if 'adapt_errors' in state:
        errs = state['adapt_errors']
        print(f"  Adaptive errors  : {[f'{e:.3f}' for e in errs]} kcal/mol")
    if 'ir_output_dir' in state:
        print(f"  IR output        : {state['ir_output_dir']}")
    print()

    next_cmd = (
        f"python3 ir_md_spectrum.py \\\n"
        f"    --model {state.get('model_path', '<model.pkl>')} \\\n"
        f"    --training-data {state.get('combined_data_path', '<data.npz>')} \\\n"
        f"    --steps 30000 --temp 300 --timestep 0.5 --save-every 1 \\\n"
        f"    --preminimize --zpe-min-freq 50 --zpe-max-freq 4000"
    )
    print(f"  IR spectrum command:\n    {next_cmd}\n")


if __name__ == '__main__':
    main()
