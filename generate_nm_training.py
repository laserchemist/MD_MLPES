#!/usr/bin/env python3
"""
Generate ML-PES Training Data via Normal Mode Distortions + High-T PSI4 MD.

Addresses the core failure mode of thermal ML-MD: the ML-PES predicts a
flat energy landscape outside the training distribution, causing the molecule
to drift into high-energy regions with large PSI4 errors.

Strategy
--------
1. Normal mode distortions  (--no-nm to skip)
   - Compute Hessian at equilibrium via PSI4
   - Diagonalise → normal mode vectors and frequencies
   - Displace ±n × a_thermal(T_nm) along each mode
   - PSI4 single-point at each displaced geometry
   → Covers the PES systematically near equilibrium, including modes
     that thermal MD at 300 K rarely excites

2. Multi-temperature PSI4 MD  (--no-md to skip)
   - Run direct PSI4-MD (Velocity-Verlet + Berendsen thermostat) at
     several temperatures (default 300, 600, 1000 K)
   - Captures anharmonic regions and large-amplitude motion

3. Combine with existing training data, retrain ML-PES

Usage
-----
    python3 generate_nm_training.py \\
        --training-data outputs/.../augmented_training_data.npz \\
        --model         outputs/.../retrained_mlpes_model.pkl

    python3 generate_nm_training.py \\
        --training-data outputs/.../augmented_training_data.npz \\
        --T-nm 2000 --n-amplitudes 6 --no-md

All units follow the project convention (Angstrom, Hartree, Debye, fs, amu).
"""

import sys
import argparse
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / 'modules'))

logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# ── framework imports ─────────────────────────────────────────────────────────
try:
    import psi4
    PSI4_AVAILABLE = True
    print(f"✅ PSI4 {psi4.__version__} available")
except ImportError:
    PSI4_AVAILABLE = False
    print("❌ PSI4 not available — cannot generate ab initio training data")
    sys.exit(1)

from normal_modes import (
    compute_hessian_psi4, compute_normal_modes,
    generate_nm_displacements, ATOMIC_MASSES, FREQ_CONV,
)
from data_formats import TrajectoryData, save_trajectory, load_trajectory
from direct_md import DirectMDConfig, DirectMDRunner, initialize_velocities
from test_molecules import TestMolecule
from ml_pes import MLPESConfig, MLPESTrainer

HARTREE_TO_KCAL  = 627.509474
ANGSTROM_TO_BOHR = 1.88972612456
AU_TO_DEBYE      = 2.541746


# ==============================================================================
# PSI4 single-point
# ==============================================================================

def psi4_single_point(symbols, coords, method, basis):
    """
    Compute energy, forces, and dipole for one geometry.

    Returns
    -------
    energy  : float or None   Hartree
    forces  : ndarray or None  Hartree / Angstrom, shape (N, 3)
    dipole  : ndarray          Debye, shape (3,)  (zeros if unavailable)
    error   : str or None
    """
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
        grad_mat, wfn = psi4.gradient(
            f'{method}/{basis}', molecule=mol, return_wfn=True
        )

        energy    = float(wfn.energy())
        n_atoms   = len(symbols)
        grad_bohr = np.array([[grad_mat.get(i, j) for j in range(3)]
                               for i in range(n_atoms)])
        forces    = -grad_bohr / ANGSTROM_TO_BOHR          # Hartree / Ang

        # Dipole (Debye)
        dipole = np.zeros(3)
        try:
            psi4.oeprop(wfn, 'DIPOLE')
            dipole = np.array([
                psi4.variable('DIPOLE X') * AU_TO_DEBYE,
                psi4.variable('DIPOLE Y') * AU_TO_DEBYE,
                psi4.variable('DIPOLE Z') * AU_TO_DEBYE,
            ])
        except Exception:
            pass                                            # zeros is fine

        return energy, forces, dipole, None

    except Exception as e:
        return None, None, np.zeros(3), str(e)


# ==============================================================================
# Normal mode displacement sampling
# ==============================================================================

def run_nm_sampling(symbols, coords_eq, method, basis,
                    T_nm=1000.0, n_amplitudes=4, max_factor=3.0):
    """
    Compute Hessian, get normal modes, run PSI4 at each displaced geometry.

    Returns TrajectoryData (or None if everything failed).
    """
    print("\n" + "=" * 70)
    print("  NORMAL MODE DISPLACEMENT SAMPLING")
    print("=" * 70)

    # ── Step 1: Hessian ───────────────────────────────────────────────────────
    print(f"\n📐 Computing PSI4 Hessian at {method}/{basis} …")
    t0 = time.time()
    hessian = compute_hessian_psi4(symbols, coords_eq, method, basis)
    print(f"   ✅ Done in {time.time()-t0:.1f} s")

    # ── Step 2: Normal modes ──────────────────────────────────────────────────
    frequencies, eigvecs_mw, eigenvalues, mass_vec = \
        compute_normal_modes(symbols, hessian)

    print(f"\n🎵 Vibrational frequencies ({len(frequencies)} modes):")
    for i, (f, ev) in enumerate(zip(frequencies, eigenvalues)):
        # thermal amplitude at T_nm in Angstrom
        L_mw   = eigvecs_mw[:, i]
        from normal_modes import KB_HARTREE_PER_K, BOHR_TO_ANGSTROM
        Q_cl   = np.sqrt(2.0 * T_nm * KB_HARTREE_PER_K / ev) if ev > 0 else 0.0
        dr_ang = (Q_cl * L_mw / np.sqrt(mass_vec)) * BOHR_TO_ANGSTROM
        a_therm = np.linalg.norm(dr_ang)
        print(f"   Mode {i+1:2d}: {f:7.1f} cm⁻¹  "
              f"a_thermal({T_nm:.0f}K) = {a_therm:.4f} Å")

    # ── Step 3: Generate displacements ────────────────────────────────────────
    print(f"\n⚙️  Generating displacements "
          f"(T_nm={T_nm}K, {n_amplitudes} steps, max={max_factor}×a_thermal) …")
    displacements = generate_nm_displacements(
        symbols, coords_eq, eigvecs_mw, eigenvalues, mass_vec,
        T=T_nm, n_amplitudes=n_amplitudes, max_factor=max_factor,
    )
    print(f"   Displaced geometries: {len(displacements)}")

    # ── Step 4: PSI4 single-points ────────────────────────────────────────────
    print(f"\n🔬 Running PSI4 single-points …")
    coords_list, energies_list, forces_list, dipoles_list = [], [], [], []
    n_failed = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(displacements, desc="NM single-points")
    except ImportError:
        iterator = displacements

    for coords_d, mode_idx, factor in iterator:
        energy, forces, dipole, err = psi4_single_point(
            symbols, coords_d, method, basis
        )
        if err:
            n_failed += 1
            logger.debug(f"Mode {mode_idx} factor={factor:+.2f}: {err}")
            continue
        coords_list.append(coords_d)
        energies_list.append(energy)
        forces_list.append(forces)
        dipoles_list.append(dipole)

    n_ok = len(coords_list)
    print(f"\n✅ NM sampling: {n_ok}/{len(displacements)} succeeded  "
          f"({n_failed} failed)")

    if n_ok == 0:
        print("⚠️  All NM single-points failed — skipping NM data")
        return None

    return TrajectoryData(
        symbols     = symbols,
        coordinates = np.array(coords_list),
        energies    = np.array(energies_list),
        forces      = np.array(forces_list),
        dipoles     = np.array(dipoles_list),
        metadata    = {
            'source':       'nm_displacements',
            'method':       method,
            'basis':        basis,
            'T_nm':         T_nm,
            'n_amplitudes': n_amplitudes,
            'max_factor':   max_factor,
        },
    )


# ==============================================================================
# Multi-temperature PSI4 MD
# ==============================================================================

def run_multi_temp_md(symbols, coords_eq, method, basis,
                      temperatures, n_steps_per_T, timestep=0.5):
    """
    Run PSI4-MD (Velocity-Verlet) at each temperature and combine frames.

    Returns TrajectoryData (or None if everything failed).
    """
    print("\n" + "=" * 70)
    print("  MULTI-TEMPERATURE PSI4 MD")
    print("=" * 70)

    # Build a TestMolecule for DirectMDRunner
    formula = ''.join(
        f"{s}{symbols.count(s)}" if symbols.count(s) > 1 else s
        for s in dict.fromkeys(symbols)
    )
    mol = TestMolecule(
        name         = 'custom',
        formula      = formula,
        symbols      = symbols,
        coordinates  = coords_eq.copy(),
        charge       = 0,
        multiplicity = 1,
    )

    all_coords, all_energies, all_forces, all_dipoles = [], [], [], []

    for T in temperatures:
        print(f"\n🌡️  PSI4 MD at {T} K  ({n_steps_per_T} steps × {timestep} fs) …")
        config = DirectMDConfig(
            method            = method,
            basis             = basis,
            temperature       = T,
            timestep          = timestep,
            n_steps           = n_steps_per_T,
            output_frequency  = 1,          # save every step → max coverage
            thermostat        = 'berendsen',
            thermostat_coupling = 20.0,     # tight coupling → stays near T
            calculate_dipole  = True,
            save_dipole       = True,
        )
        runner = DirectMDRunner(config)

        # Use a fresh molecule with randomised velocities each temperature
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            traj = runner.run(mol)          # DirectMDRunner.run returns TrajectoryData

        if traj is None or traj.n_frames == 0:
            print(f"   ⚠️  No frames from {T} K run")
            continue

        all_coords.append(traj.coordinates)
        all_energies.append(traj.energies)
        all_forces.append(traj.forces)
        dip = traj.dipoles if traj.dipoles is not None \
              else np.zeros((traj.n_frames, 3))
        all_dipoles.append(dip)
        print(f"   ✅ {traj.n_frames} frames")

    if not all_coords:
        print("⚠️  All MD runs failed — skipping MD data")
        return None

    traj_combined = TrajectoryData(
        symbols     = symbols,
        coordinates = np.concatenate(all_coords,   axis=0),
        energies    = np.concatenate(all_energies),
        forces      = np.concatenate(all_forces,   axis=0),
        dipoles     = np.concatenate(all_dipoles,  axis=0),
        metadata    = {
            'source':        'multi_T_psi4_md',
            'method':        method,
            'basis':         basis,
            'temperatures':  temperatures,
            'n_steps_per_T': n_steps_per_T,
        },
    )
    print(f"\n✅ Multi-T MD total: {traj_combined.n_frames} frames")
    return traj_combined


# ==============================================================================
# Combine trajectories
# ==============================================================================

def combine_trajectories(trajs):
    """Stack a list of TrajectoryData objects into one."""
    symbols = trajs[0].symbols
    has_dipoles = all(t.dipoles is not None for t in trajs)

    combined = TrajectoryData(
        symbols     = symbols,
        coordinates = np.concatenate([t.coordinates for t in trajs], axis=0),
        energies    = np.concatenate([t.energies    for t in trajs]),
        forces      = np.concatenate([t.forces      for t in trajs], axis=0),
        dipoles     = np.concatenate(
            [t.dipoles if t.dipoles is not None else np.zeros((t.n_frames, 3))
             for t in trajs], axis=0
        ) if has_dipoles else None,
        metadata    = {'source': 'combined', 'n_components': len(trajs)},
    )
    return combined


# ==============================================================================
# Retraining
# ==============================================================================

def retrain_mlpes(training_traj, output_dir):
    """Train a fresh ML-PES on combined trajectory data."""
    print("\n" + "=" * 70)
    print("  TRAINING ML-PES")
    print("=" * 70)
    print(f"   Frames : {training_traj.n_frames}")
    e = training_traj.energies
    print(f"   Energy range : {(e.max()-e.min())*HARTREE_TO_KCAL:.1f} kcal/mol")

    config = MLPESConfig(
        model_type          = 'kernel_ridge',
        descriptor_type     = 'coulomb_matrix',
        tune_hyperparameters= True,
        gamma_range         = [0.001, 0.01, 0.1, 1.0],
        alpha_range         = [0.001, 0.01, 0.1, 1.0],
        train_forces        = False,     # keep disabled — see CLAUDE.md
        validation_split    = 0.15,
    )
    trainer = MLPESTrainer(config)
    trainer.train(training_traj)

    model_path = output_dir / 'mlpes_model_nm.pkl'
    trainer.save(str(model_path))
    print(f"\n💾 Model saved: {model_path}")
    return trainer, model_path


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate ML-PES training data via NM distortions + high-T PSI4 MD',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--training-data', required=True,
                        help='Existing training data (.npz)')
    parser.add_argument('--model', default=None,
                        help='Existing ML-PES model (.pkl) — used for metadata only')
    parser.add_argument('--method', default=None,
                        help='QM method (default: from training-data metadata)')
    parser.add_argument('--basis',  default=None,
                        help='Basis set (default: from training-data metadata)')

    # NM options
    nm_grp = parser.add_argument_group('Normal mode options')
    nm_grp.add_argument('--no-nm', action='store_true',
                        help='Skip normal mode displacement sampling')
    nm_grp.add_argument('--T-nm', type=float, default=1000.0,
                        help='Temperature (K) for NM amplitude scale')
    nm_grp.add_argument('--n-amplitudes', type=int, default=4,
                        help='Amplitude steps per mode (±)')
    nm_grp.add_argument('--max-factor', type=float, default=3.0,
                        help='Max amplitude as multiple of a_thermal')

    # MD options
    md_grp = parser.add_argument_group('Multi-temperature PSI4 MD options')
    md_grp.add_argument('--no-md', action='store_true',
                        help='Skip multi-temperature PSI4 MD')
    md_grp.add_argument('--md-temps', default='300,600,1000',
                        help='MD temperatures in K, comma-separated')
    md_grp.add_argument('--md-steps', type=int, default=50,
                        help='PSI4 MD steps per temperature')
    md_grp.add_argument('--md-timestep', type=float, default=0.5,
                        help='MD timestep in fs')

    parser.add_argument('--no-retrain', action='store_true',
                        help='Skip ML-PES retraining (save data only)')
    args = parser.parse_args()

    # ── Load existing training data ───────────────────────────────────────────
    print(f"\n📂 Loading training data: {args.training_data}")
    existing_traj = load_trajectory(args.training_data)
    symbols = list(existing_traj.symbols)
    print(f"   Frames : {existing_traj.n_frames}   Atoms : {len(symbols)}")

    # Equilibrium geometry = lowest-energy frame
    eq_idx    = int(np.argmin(existing_traj.energies))
    coords_eq = existing_traj.coordinates[eq_idx].copy()
    E_eq      = existing_traj.energies[eq_idx]
    print(f"   Equilibrium frame : {eq_idx}  "
          f"E = {E_eq * HARTREE_TO_KCAL:.2f} kcal/mol")

    # Method / basis from metadata or CLI
    meta   = existing_traj.metadata or {}
    method = args.method or meta.get('method', 'B3LYP')
    basis  = args.basis  or meta.get('basis',  '6-31G*')
    print(f"   QM level : {method}/{basis}")

    # If a model .pkl is given, cross-check method/basis from metadata
    if args.model:
        with open(args.model, 'rb') as f:
            model_meta = pickle.load(f).get('metadata', {})
        theory = model_meta.get('theory', {})
        if isinstance(theory, dict):
            method = args.method or theory.get('method', method)
            basis  = args.basis  or theory.get('basis',  basis)
        print(f"   (from model) QM level : {method}/{basis}")

    # ── Output directory ──────────────────────────────────────────────────────
    ts         = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'outputs/nm_training_{ts}')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")

    # ── Collect new data ──────────────────────────────────────────────────────
    new_trajs = [existing_traj]
    print(f"\n   Starting data: {existing_traj.n_frames} frames  "
          f"({(existing_traj.energies.max()-existing_traj.energies.min())*HARTREE_TO_KCAL:.1f} kcal/mol range)")

    # 1. Normal mode displacements
    if not args.no_nm:
        nm_traj = run_nm_sampling(
            symbols, coords_eq, method, basis,
            T_nm        = args.T_nm,
            n_amplitudes= args.n_amplitudes,
            max_factor  = args.max_factor,
        )
        if nm_traj is not None:
            nm_path = output_dir / 'nm_displacements.npz'
            save_trajectory(nm_traj, str(nm_path))
            new_trajs.append(nm_traj)
            e_rng = (nm_traj.energies.max()-nm_traj.energies.min()) * HARTREE_TO_KCAL
            print(f"\n   NM data: {nm_traj.n_frames} frames  ({e_rng:.1f} kcal/mol range)")
            print(f"   Saved : {nm_path}")

    # 2. Multi-temperature PSI4 MD
    if not args.no_md:
        temps  = [float(t) for t in args.md_temps.split(',')]
        md_traj = run_multi_temp_md(
            symbols, coords_eq, method, basis,
            temperatures  = temps,
            n_steps_per_T = args.md_steps,
            timestep      = args.md_timestep,
        )
        if md_traj is not None:
            md_path = output_dir / 'multi_T_md.npz'
            save_trajectory(md_traj, str(md_path))
            new_trajs.append(md_traj)
            e_rng = (md_traj.energies.max()-md_traj.energies.min()) * HARTREE_TO_KCAL
            print(f"\n   MD data: {md_traj.n_frames} frames  ({e_rng:.1f} kcal/mol range)")
            print(f"   Saved : {md_path}")

    # ── Combine ───────────────────────────────────────────────────────────────
    print("\n🔗 Combining datasets …")
    combined = combine_trajectories(new_trajs)
    e_rng    = (combined.energies.max()-combined.energies.min()) * HARTREE_TO_KCAL
    print(f"   Total frames : {combined.n_frames}")
    print(f"   Energy range : {e_rng:.1f} kcal/mol  "
          f"(was {(existing_traj.energies.max()-existing_traj.energies.min())*HARTREE_TO_KCAL:.1f})")

    combined_path = output_dir / 'combined_training_data.npz'
    save_trajectory(combined, str(combined_path))
    print(f"💾 Combined data saved: {combined_path}")

    # ── Retrain ───────────────────────────────────────────────────────────────
    if not args.no_retrain:
        trainer, model_path = retrain_mlpes(combined, output_dir)
        print(f"\n✅ Done!")
        print(f"   New model       : {model_path}")
        print(f"   Training data   : {combined_path}")
        print(f"\n   Validate with:")
        print(f"   python3 two_phase_workflow.py \\")
        print(f"     --model {model_path} \\")
        print(f"     --training-data {combined_path} \\")
        print(f"     --steps 200 --temp 300")
    else:
        print(f"\n✅ Data generation done (retraining skipped).")
        print(f"   Training data: {combined_path}")


if __name__ == '__main__':
    main()
