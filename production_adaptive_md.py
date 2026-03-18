#!/usr/bin/env python3
"""
Production Adaptive ML-PES MD with automatic refinement.

Algorithm per cycle:
  1. Run ML-MD for --steps steps (fast, no PSI4)
  2. Validate every snapshot with PSI4
  3. If max_error < --error-threshold: CONVERGED, stop
  4. Add frames with error > --error-threshold to training data
  5. Retrain model (full 4×4 hyperparameter search)
  6. Repeat up to --max-cycles

Usage:
  python3 production_adaptive_md.py \\
      --model outputs/nm_training_20260308_203606/mlpes_model_nm.pkl \\
      --training-data outputs/nm_training_20260308_203606/combined_training_data.npz \\
      --steps 500 --temp 300 --max-cycles 5

Units: coordinates Angstrom, energies Hartree, forces Hartree/Angstrom,
       time femtoseconds, masses amu.  (See CLAUDE.md for full table.)
"""

import sys
import os
import argparse
import json
import pickle
import datetime
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------
# Path setup — must happen before any local imports
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'modules'))

# ------------------------------------------------------------------
# Physical constants (canonical set from direct_md.py)
# ------------------------------------------------------------------
KB_HARTREE_PER_K  = 3.1668114e-6      # Hartree / K
AMU_TO_AU         = 1822.888486       # m_u → m_e
FS_TO_AU          = 41.341374575751   # fs → a.u. time
BOHR_TO_ANG       = 0.529177210903
ANG_TO_BOHR       = 1.0 / BOHR_TO_ANG
HARTREE_TO_KCAL   = 627.509474

ATOMIC_MASSES = {
    'H':  1.00794, 'He': 4.002602, 'C': 12.011,  'N': 14.007,
    'O':  15.999,  'F': 18.9984,   'S': 32.06,   'Cl': 35.453,
    'Br': 79.904,  'I': 126.904,   'P': 30.97376, 'Si': 28.0855,
}

# ------------------------------------------------------------------
# Module imports
# ------------------------------------------------------------------
from data_formats import TrajectoryData, load_trajectory, save_trajectory
from ml_pes import MLPESTrainer, MLPESConfig

# Lazy PSI4
PSI4_AVAILABLE = False
try:
    import psi4
    PSI4_AVAILABLE = True
except ImportError:
    pass

# generate_nm_training helpers (no side effects when imported)
from generate_nm_training import psi4_single_point, retrain_mlpes, combine_trajectories


# ==============================================================================
# ML-PES wrapper (energy + finite-difference forces)
# ==============================================================================

class MLPESDriver:
    """Thin wrapper around MLPESTrainer providing energy + FD forces."""

    def __init__(self, model_path: str):
        self.trainer = MLPESTrainer.load(model_path)
        self.symbols  = self.trainer.symbols
        self.model_path = Path(model_path)

    def energy(self, coords: np.ndarray) -> float:
        return self.trainer.predict(self.symbols, coords)

    def forces(self, coords: np.ndarray, delta: float = 0.005) -> np.ndarray:
        """Cartesian forces via central finite differences (Hartree/Angstrom)."""
        n_atoms = len(self.symbols)
        F = np.zeros_like(coords)
        for i in range(n_atoms):
            for ax in range(3):
                cp = coords.copy(); cp[i, ax] += delta
                cm = coords.copy(); cm[i, ax] -= delta
                F[i, ax] = -(self.energy(cp) - self.energy(cm)) / (2 * delta)
        return F


# ==============================================================================
# Velocity-Verlet MD engine
# ==============================================================================

def _maxwell_boltzmann(masses_amu: np.ndarray, T: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Initial velocities (Angstrom/fs) from Maxwell-Boltzmann distribution."""
    n_atoms = len(masses_amu)
    masses_au = masses_amu * AMU_TO_AU
    velocities_au = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        sigma = np.sqrt(KB_HARTREE_PER_K * T / masses_au[i])
        velocities_au[i] = rng.normal(0.0, sigma, 3)
    # Remove center-of-mass drift
    total_p = (masses_au[:, None] * velocities_au).sum(axis=0)
    velocities_au -= total_p / masses_au.sum()
    # Convert a.u. velocity → Angstrom/fs
    # v [a.u.] = v [Ang/fs] * FS_TO_AU / ANG_TO_BOHR
    # ⟹ v [Ang/fs] = v [a.u.] * ANG_TO_BOHR / FS_TO_AU
    return velocities_au * BOHR_TO_ANG / FS_TO_AU


def _kinetic_temperature(velocities: np.ndarray, masses_amu: np.ndarray) -> float:
    """Instantaneous temperature from kinetic energy (velocities in Ang/fs)."""
    masses_au = masses_amu * AMU_TO_AU
    # KE in Hartree: 0.5 * sum m * v² with v in a.u.
    v_au = velocities * ANG_TO_BOHR * FS_TO_AU  # Ang/fs → a.u.
    ke = 0.5 * (masses_au[:, None] * v_au ** 2).sum()
    n_dof = 3 * len(masses_amu) - 6
    return 2.0 * ke / (n_dof * KB_HARTREE_PER_K)


def run_ml_md(driver: MLPESDriver,
              coords0: np.ndarray,
              n_steps: int,
              temperature: float,
              timestep: float = 0.5,
              snapshot_every: int = 10,
              thermostat_tau: float = 50.0,
              seed: int = 42) -> dict:
    """
    Velocity-Verlet ML-MD with Berendsen thermostat.

    Args:
        driver        : MLPESDriver instance
        coords0       : Starting coordinates (N, 3) Angstrom
        n_steps       : Number of MD steps
        temperature   : Target temperature (K)
        timestep      : fs
        snapshot_every: Save snapshot every this many steps
        thermostat_tau: Berendsen coupling time (fs)
        seed          : RNG seed

    Returns:
        dict with keys: snapshots, steps, energies_ml, symbols
    """
    masses_amu = np.array([ATOMIC_MASSES[s] for s in driver.symbols])
    masses_au  = masses_amu * AMU_TO_AU
    dt_au      = timestep * FS_TO_AU          # fs → a.u.
    dt_ang     = timestep                      # stay in Ang/fs

    rng = np.random.default_rng(seed)
    coords    = coords0.copy()
    velocities = _maxwell_boltzmann(masses_amu, temperature, rng)
    forces     = driver.forces(coords)         # Hartree/Ang

    snapshots   = []
    snap_steps  = []
    snap_energies = []

    from tqdm import tqdm
    pbar = tqdm(range(1, n_steps + 1), desc="ML-MD", unit="step")
    for step in pbar:
        # Half-step velocity update
        # F [Hartree/Ang] / m [amu] in a.u. units:
        # a [Ang/fs²] = F [Hartree/Ang] * (Hartree→kg·m²/s²) / (amu→kg) → messy
        # Simplest: work entirely in a.u., then convert
        # a [a.u.] = F [Hartree/Bohr] / m_au
        # F_bohr = F_ang * ANG_TO_BOHR
        F_bohr  = forces * ANG_TO_BOHR               # Hartree/Bohr
        acc_au  = F_bohr / masses_au[:, None]         # Bohr/a.u.² per atom

        v_au    = velocities * ANG_TO_BOHR * FS_TO_AU
        v_au   += 0.5 * acc_au * dt_au
        # Position update (a.u.)
        r_au    = coords * ANG_TO_BOHR
        r_au   += v_au * dt_au
        coords  = r_au * BOHR_TO_ANG                  # back to Angstrom

        # New forces
        forces  = driver.forces(coords)
        F_bohr  = forces * ANG_TO_BOHR
        acc_au  = F_bohr / masses_au[:, None]
        v_au   += 0.5 * acc_au * dt_au
        velocities = v_au * BOHR_TO_ANG / FS_TO_AU   # back to Ang/fs

        # Berendsen thermostat
        T_curr = _kinetic_temperature(velocities, masses_amu)
        if T_curr > 0:
            lam = np.sqrt(1.0 + (dt_ang / thermostat_tau) * (temperature / T_curr - 1.0))
            velocities *= lam

        # Save snapshot
        if step % snapshot_every == 0:
            e = driver.energy(coords)
            snapshots.append(coords.copy())
            snap_steps.append(step)
            snap_energies.append(e)
            pbar.set_postfix({'E': f'{e*HARTREE_TO_KCAL:.1f} kcal/mol',
                              'T': f'{_kinetic_temperature(velocities, masses_amu):.0f} K'})

    return {
        'snapshots':    np.array(snapshots),
        'steps':        snap_steps,
        'energies_ml':  np.array(snap_energies),
        'symbols':      driver.symbols,
        'parameters':   {'n_steps': n_steps, 'temperature': temperature,
                         'timestep': timestep, 'snapshot_every': snapshot_every},
    }


# ==============================================================================
# PSI4 Validation
# ==============================================================================

def validate_snapshots(snapshot_data: dict,
                        method: str = 'B3LYP',
                        basis:  str = '6-31G*') -> dict:
    """
    Run PSI4 single-point energy+forces+dipole on every snapshot.

    Returns dict with keys:
        coords_list, energies_psi4, forces_psi4, dipoles_psi4,
        energies_ml, errors_energy (kcal/mol), valid_indices
    """
    symbols   = snapshot_data['symbols']
    snapshots = snapshot_data['snapshots']
    ml_energies = snapshot_data['energies_ml']
    n = len(snapshots)

    coords_out, e_psi4_out, f_psi4_out, d_psi4_out = [], [], [], []
    e_ml_out, errors_out, valid_idx = [], [], []

    from tqdm import tqdm
    for i, coords in enumerate(tqdm(snapshots, desc="PSI4 validations")):
        e, f, d, err = psi4_single_point(symbols, coords, method, basis)
        if err is not None:
            print(f"   ⚠️  Frame {i}: {err}")
            continue
        coords_out.append(coords)
        e_psi4_out.append(e)
        f_psi4_out.append(f)
        d_psi4_out.append(d)
        e_ml_out.append(ml_energies[i])
        errors_out.append(abs(ml_energies[i] - e) * HARTREE_TO_KCAL)
        valid_idx.append(i)

    return {
        'coords_list':   np.array(coords_out),
        'energies_psi4': np.array(e_psi4_out),
        'forces_psi4':   np.array(f_psi4_out),
        'dipoles_psi4':  np.array(d_psi4_out),
        'energies_ml':   np.array(e_ml_out),
        'errors_energy': np.array(errors_out),
        'valid_indices': valid_idx,
    }


# ==============================================================================
# Adaptive refinement loop
# ==============================================================================

def adaptive_loop(model_path: str,
                  training_data_path: str,
                  n_steps: int,
                  temperature: float,
                  timestep: float,
                  snapshot_every: int,
                  error_threshold: float,
                  max_cycles: int,
                  method: str,
                  basis: str,
                  output_dir: Path) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    # Track per-cycle stats for summary table
    history = []
    current_model = model_path
    current_data  = training_data_path

    print(f"\n{'=' * 70}")
    print("  PRODUCTION ADAPTIVE ML-PES MD")
    print(f"{'=' * 70}")
    print(f"  Model          : {current_model}")
    print(f"  Training data  : {current_data}")
    print(f"  MD steps/cycle : {n_steps}  ({n_steps * timestep:.0f} fs)")
    print(f"  Temperature    : {temperature} K")
    print(f"  Snapshots/cycle: {n_steps // snapshot_every}")
    print(f"  Error threshold: {error_threshold} kcal/mol")
    print(f"  Max cycles     : {max_cycles}")
    print(f"  Output dir     : {output_dir}")

    for cycle in range(max_cycles):
        cycle_dir = output_dir / f"cycle_{cycle:02d}"
        cycle_dir.mkdir(exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"  CYCLE {cycle} / {max_cycles - 1}")
        print(f"{'=' * 70}")

        # ------------------------------------------------------------------
        # Phase 1: ML-MD
        # ------------------------------------------------------------------
        print(f"\n--- Phase 1: ML-MD ({n_steps} steps) ---")
        driver = MLPESDriver(current_model)
        training_traj = load_trajectory(current_data)
        start_idx = np.argmin(training_traj.energies)
        coords0 = training_traj.coordinates[start_idx].copy()
        print(f"  Starting from frame {start_idx}  "
              f"(E = {training_traj.energies[start_idx] * HARTREE_TO_KCAL:.2f} kcal/mol)")

        snapshot_data = run_ml_md(
            driver, coords0, n_steps, temperature,
            timestep=timestep, snapshot_every=snapshot_every,
        )

        with open(cycle_dir / 'snapshots.pkl', 'wb') as fh:
            pickle.dump(snapshot_data, fh)

        ml_e = snapshot_data['energies_ml']
        print(f"  ML energy range: "
              f"{(ml_e.max() - ml_e.min()) * HARTREE_TO_KCAL:.2f} kcal/mol  "
              f"(min {ml_e.min() * HARTREE_TO_KCAL:.2f}, "
              f"max {ml_e.max() * HARTREE_TO_KCAL:.2f})")

        # ------------------------------------------------------------------
        # Phase 2: PSI4 Validation
        # ------------------------------------------------------------------
        print(f"\n--- Phase 2: PSI4 validation "
              f"({len(snapshot_data['snapshots'])} snapshots) ---")
        val = validate_snapshots(snapshot_data, method, basis)

        n_valid = len(val['valid_indices'])
        if n_valid == 0:
            print("  ❌ All PSI4 calculations failed — stopping")
            break

        with open(cycle_dir / 'validation.pkl', 'wb') as fh:
            pickle.dump(val, fh)

        errs = val['errors_energy']
        mean_e = errs.mean()
        max_e  = errs.max()
        std_e  = errs.std()
        n_above = (errs > error_threshold).sum()

        print(f"\n  Energy errors (kcal/mol):")
        print(f"    Mean  : {mean_e:.3f}")
        print(f"    Std   : {std_e:.3f}")
        print(f"    Max   : {max_e:.3f}")
        print(f"    > {error_threshold:.1f} kcal/mol: {n_above} / {n_valid} frames")

        history.append({
            'cycle': cycle, 'n_valid': n_valid,
            'mean_error': mean_e, 'max_error': max_e, 'std_error': std_e,
            'n_above_threshold': int(n_above),
            'model': current_model, 'data': current_data,
        })

        # ------------------------------------------------------------------
        # Convergence check
        # ------------------------------------------------------------------
        if max_e <= error_threshold:
            print(f"\n  ✅ CONVERGED — max error {max_e:.3f} kcal/mol ≤ "
                  f"threshold {error_threshold:.1f} kcal/mol")
            break

        if cycle == max_cycles - 1:
            print(f"\n  ⚠️  Max cycles reached ({max_cycles}) without convergence")
            break

        # ------------------------------------------------------------------
        # Refinement: add high-error frames
        # ------------------------------------------------------------------
        mask = errs > error_threshold
        n_add = mask.sum()
        print(f"\n--- Refinement: adding {n_add} frames (error > "
              f"{error_threshold:.1f} kcal/mol) ---")

        sel_coords  = val['coords_list'][mask]
        sel_energies = val['energies_psi4'][mask]
        sel_forces   = val['forces_psi4'][mask]
        sel_dipoles  = val['dipoles_psi4'][mask]

        # Combine with current training data
        orig = load_trajectory(current_data)
        orig_dip = (orig.dipoles if orig.dipoles is not None
                    else np.zeros((orig.n_frames, 3)))

        combined = TrajectoryData(
            symbols     = orig.symbols,
            coordinates = np.concatenate([orig.coordinates, sel_coords]),
            energies    = np.concatenate([orig.energies,    sel_energies]),
            forces      = np.concatenate([orig.forces,      sel_forces]),
            dipoles     = np.concatenate([orig_dip,         sel_dipoles]),
            metadata    = {'source': f'adaptive_cycle_{cycle}',
                           'n_orig': orig.n_frames, 'n_added': int(n_add)},
        )
        e_range = (combined.energies.max() - combined.energies.min()) * HARTREE_TO_KCAL
        print(f"  Training set: {orig.n_frames} + {n_add} = {combined.n_frames} frames  "
              f"({e_range:.1f} kcal/mol range)")

        aug_path = cycle_dir / 'augmented_training_data.npz'
        save_trajectory(combined, str(aug_path))

        # Retrain
        _, new_model_path = retrain_mlpes(combined, cycle_dir)
        current_model = str(new_model_path)
        current_data  = str(aug_path)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  ADAPTIVE MD SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Cycle':>6}  {'N valid':>8}  {'Mean (kcal/mol)':>16}  "
          f"{'Std':>8}  {'Max':>8}  {'> thresh':>9}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*16}  {'-'*8}  {'-'*8}  {'-'*9}")
    for h in history:
        print(f"  {h['cycle']:>6}  {h['n_valid']:>8}  {h['mean_error']:>16.3f}  "
              f"{h['std_error']:>8.3f}  {h['max_error']:>8.3f}  "
              f"{h['n_above_threshold']:>9}")

    # Save summary JSON
    summary = {
        'history': history,
        'final_model': current_model,
        'final_training_data': current_data,
        'converged': bool(len(history) > 0 and history[-1]['max_error'] <= error_threshold),
        'parameters': {
            'n_steps': n_steps, 'temperature': temperature,
            'timestep': timestep, 'snapshot_every': snapshot_every,
            'error_threshold': error_threshold, 'max_cycles': max_cycles,
            'method': method, 'basis': basis,
        },
        'date': datetime.datetime.now().isoformat(),
    }
    with open(output_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n  Final model         : {current_model}")
    print(f"  Final training data : {current_data}")
    print(f"  Summary saved       : {output_dir / 'summary.json'}")

    # Generate figure
    try:
        plot_production_run(output_dir, summary)
    except Exception as exc:
        print(f"\n  ⚠️  Figure generation failed: {exc}")

    if summary['converged']:
        print(f"\n  ✅ Converged in {len(history)} cycle(s)!")
    else:
        final = history[-1]
        print(f"\n  Max error after {len(history)} cycle(s): "
              f"{final['max_error']:.3f} kcal/mol  "
              f"(threshold: {error_threshold:.1f})")
        print(f"  → Run again with the final model to continue refinement:")
        print(f"    python3 production_adaptive_md.py \\")
        print(f"        --model {current_model} \\")
        print(f"        --training-data {current_data} \\")
        print(f"        --steps {n_steps} --temp {temperature} "
              f"--max-cycles {max_cycles}")


# ==============================================================================
# Figure generation
# ==============================================================================

def plot_production_run(output_dir: Path, summary: dict) -> Path:
    """
    Create a comprehensive diagnostic figure for the adaptive production run.

    Panels:
      Row 1 (trajectory):  ML energy trace | ML vs PSI4 parity | error vs time
      Row 2 (statistics):  error histogram  | force error dist  | cycle summary

    Works for any number of cycles (1 = converged immediately, N = with refinement).
    Saves as <output_dir>/production_run_figure.png and returns the path.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    history = summary['history']
    n_cycles = len(history)
    params   = summary['parameters']
    timestep = params['timestep']
    snap_every = params['snapshot_every']

    # Load per-cycle data
    cycle_snaps = []
    cycle_vals  = []
    for h in history:
        c = int(h['cycle'])
        cdir = output_dir / f"cycle_{c:02d}"
        with open(cdir / 'snapshots.pkl', 'rb') as fh:
            cycle_snaps.append(pickle.load(fh))
        with open(cdir / 'validation.pkl', 'rb') as fh:
            cycle_vals.append(pickle.load(fh))

    # Use the LAST cycle for trajectory panels
    snaps = cycle_snaps[-1]
    val   = cycle_vals[-1]

    steps     = np.array(snaps['steps'])
    time_fs   = steps * timestep
    ml_e_Ha   = snaps['energies_ml']
    psi4_e_Ha = val['energies_psi4']
    ml_e2_Ha  = val['energies_ml']
    errors    = val['errors_energy']                    # kcal/mol

    # Relative energies (kcal/mol, relative to PSI4 min)
    e_ref   = psi4_e_Ha.min()
    ml_rel  = (ml_e_Ha  - e_ref) * HARTREE_TO_KCAL
    psi4_rel = (psi4_e_Ha - e_ref) * HARTREE_TO_KCAL
    ml2_rel  = (ml_e2_Ha  - e_ref) * HARTREE_TO_KCAL

    # Force magnitudes (kcal/mol/Å)
    f_psi4 = val['forces_psi4']                        # (N, n_atoms, 3) Hartree/Å
    f_mag  = np.linalg.norm(
        f_psi4.reshape(len(f_psi4), -1), axis=1
    ) * HARTREE_TO_KCAL

    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor('#f8f9fa')
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.42, wspace=0.35,
                           left=0.07, right=0.97, top=0.91, bottom=0.08)

    cycle_label = f"Cycle {history[-1]['cycle']}"
    title_suffix = f"  |  {params['n_steps']} steps · {params['temperature']:.0f} K · " \
                   f"{params['method']}/{params['basis']}"
    fig.suptitle(f"Adaptive ML-PES MD — Production Run{title_suffix}",
                 fontsize=13, fontweight='bold', y=0.97)

    # ── Panel 1: ML energy trace ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_fs, ml_rel, color='steelblue', lw=1.2, label='ML-PES')
    ax1.scatter(time_fs, ml2_rel, s=14, color='steelblue', zorder=3,
                label='ML @ validated frames')
    ax1.scatter(time_fs, psi4_rel, s=14, marker='D', color='firebrick',
                zorder=4, label='PSI4')
    ax1.set_xlabel('Time (fs)')
    ax1.set_ylabel('Energy (kcal/mol, relative)')
    ax1.set_title(f'Trajectory — {cycle_label}', fontsize=10)
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Parity plot ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    lo = min(psi4_rel.min(), ml2_rel.min()) - 0.5
    hi = max(psi4_rel.max(), ml2_rel.max()) + 0.5
    ax2.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5, label='Perfect')
    sc = ax2.scatter(psi4_rel, ml2_rel, c=errors, cmap='RdYlGn_r',
                     vmin=0, vmax=max(2.0, errors.max()), s=30, zorder=3)
    cb = fig.colorbar(sc, ax=ax2, shrink=0.85)
    cb.set_label('Error (kcal/mol)', fontsize=8)
    ax2.set_xlabel('PSI4 energy (kcal/mol, relative)')
    ax2.set_ylabel('ML-PES energy (kcal/mol, relative)')
    ax2.set_title(f'ML vs PSI4 Parity — {cycle_label}', fontsize=10)
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Error vs time ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.fill_between(time_fs, 0, errors, alpha=0.3, color='tomato')
    ax3.plot(time_fs, errors, color='tomato', lw=1.2)
    ax3.axhline(params['error_threshold'], color='navy', lw=1.2,
                ls='--', label=f"Threshold ({params['error_threshold']} kcal/mol)")
    ax3.set_xlabel('Time (fs)')
    ax3.set_ylabel('|ML − PSI4| (kcal/mol)')
    ax3.set_title(f'Energy Error vs Time — {cycle_label}', fontsize=10)
    ax3.set_ylim(bottom=0)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Error histogram ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    all_errors = np.concatenate([cv['errors_energy'] for cv in cycle_vals])
    bins = np.linspace(0, max(all_errors.max(), params['error_threshold'] * 1.5), 30)
    colors = plt.cm.tab10(np.linspace(0, 1, n_cycles))
    for ci, (cv, col) in enumerate(zip(cycle_vals, colors)):
        ax4.hist(cv['errors_energy'], bins=bins, alpha=0.55, color=col,
                 label=f"Cycle {ci}", edgecolor='white', lw=0.5)
    ax4.axvline(params['error_threshold'], color='navy', lw=1.5,
                ls='--', label='Threshold')
    ax4.set_xlabel('|ML − PSI4| (kcal/mol)')
    ax4.set_ylabel('Count')
    ax4.set_title('Error Distribution (all cycles)', fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: Force magnitude distribution ────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(f_mag, bins=20, color='mediumseagreen', edgecolor='white',
             alpha=0.8, lw=0.5)
    ax5.set_xlabel('PSI4 force magnitude (kcal/mol/Å)')
    ax5.set_ylabel('Count')
    ax5.set_title(f'PSI4 Force Magnitudes — {cycle_label}', fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.axvline(np.median(f_mag), color='darkgreen', lw=1.5,
                ls='--', label=f'Median {np.median(f_mag):.1f}')
    ax5.legend(fontsize=8)

    # ── Panel 6: Cycle summary ────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    if n_cycles > 1:
        cycles  = [h['cycle'] for h in history]
        means   = [h['mean_error'] for h in history]
        maxes   = [h['max_error'] for h in history]
        n_train = []
        for ci, h in enumerate(history):
            cdir = output_dir / f"cycle_{ci:02d}"
            aug  = cdir / 'augmented_training_data.npz'
            if aug.exists():
                d = np.load(str(aug), allow_pickle=True)
                n_train.append(len(d['energies']))
            else:
                n_train.append(None)

        ax6.plot(cycles, means, 'o-', color='steelblue', label='Mean error', lw=1.5)
        ax6.plot(cycles, maxes, 's--', color='tomato', label='Max error', lw=1.5)
        ax6.axhline(params['error_threshold'], color='navy', lw=1,
                    ls=':', label='Threshold')
        ax6.set_xlabel('Cycle')
        ax6.set_ylabel('Error (kcal/mol)')
        ax6.set_title('Convergence Across Cycles', fontsize=10)
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        ax6.set_xticks(cycles)
    else:
        # Single cycle: show text summary box
        h = history[0]
        converged = summary['converged']
        status = "CONVERGED" if converged else "NOT YET CONVERGED"
        txt = (
            f"Run Summary\n"
            f"{'─' * 26}\n"
            f"Cycles completed : {n_cycles}\n"
            f"Status           : {status}\n\n"
            f"MD trajectory\n"
            f"  Steps    : {params['n_steps']}\n"
            f"  Time     : {params['n_steps']*timestep:.0f} fs\n"
            f"  Temp     : {params['temperature']:.0f} K\n"
            f"  Snapshots: {h['n_valid']}\n\n"
            f"Final errors\n"
            f"  Mean  : {h['mean_error']:.3f} kcal/mol\n"
            f"  Std   : {h['std_error']:.3f} kcal/mol\n"
            f"  Max   : {h['max_error']:.3f} kcal/mol\n"
            f"  > {params['error_threshold']:.1f} : {h['n_above_threshold']} frames\n\n"
            f"QM level : {params['method']}/{params['basis']}"
        )
        ax6.text(0.05, 0.95, txt, transform=ax6.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                           edgecolor='goldenrod', alpha=0.9))
        ax6.axis('off')
        ax6.set_title('Run Summary', fontsize=10)

    # Save
    fig_path = output_dir / 'production_run_figure.png'
    fig.savefig(str(fig_path), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved: {fig_path}")
    return fig_path


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Production adaptive ML-PES MD with automatic refinement',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model',         required=True, help='ML-PES model (.pkl)')
    parser.add_argument('--training-data', required=True, help='Training data (.npz)')
    parser.add_argument('--steps',         type=int,   default=500,
                        help='ML-MD steps per cycle')
    parser.add_argument('--temp',          type=float, default=300.0,
                        help='MD temperature (K)')
    parser.add_argument('--timestep',      type=float, default=0.5,
                        help='Timestep (fs)')
    parser.add_argument('--snapshot-every', type=int,  default=10,
                        help='Save snapshot every N steps')
    parser.add_argument('--error-threshold', type=float, default=2.0,
                        help='Add frames with error > this (kcal/mol) to training')
    parser.add_argument('--max-cycles',    type=int,   default=5,
                        help='Maximum adaptive cycles')
    parser.add_argument('--method',        default='B3LYP', help='QM method')
    parser.add_argument('--basis',         default='6-31G*', help='Basis set')
    parser.add_argument('--output-dir',    default=None,
                        help='Output directory (auto-timestamped if omitted)')
    args = parser.parse_args()

    if not PSI4_AVAILABLE:
        print("❌ PSI4 is required for Phase 2 validation. Aborting.")
        sys.exit(1)

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path(args.output_dir) if args.output_dir else \
          Path('outputs') / f'adaptive_production_{ts}'

    adaptive_loop(
        model_path         = args.model,
        training_data_path = args.training_data,
        n_steps            = args.steps,
        temperature        = args.temp,
        timestep           = args.timestep,
        snapshot_every     = args.snapshot_every,
        error_threshold    = args.error_threshold,
        max_cycles         = args.max_cycles,
        method             = args.method,
        basis              = args.basis,
        output_dir         = out,
    )


if __name__ == '__main__':
    main()
