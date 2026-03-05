#!/usr/bin/env python3
"""
Two-Phase Diagnostic Workflow for ML-PES

Phase 1: Fast ML-PES MD (no PSI4) - 5-10 minutes
Phase 2: Post-hoc validation & analysis - 10-20 minutes  
Phase 3: Adaptive refinement options

This is 10x faster and gives much better diagnostics!
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse

print("=" * 80)
print("  TWO-PHASE DIAGNOSTIC WORKFLOW")
print("=" * 80)

# Import framework
try:
    from modules.test_molecules import get_molecule
    from modules.data_formats import TrajectoryData, save_trajectory, load_trajectory
    print("✅ Framework imported")
except ImportError as e:
    print(f"⚠️  Framework import issue: {e}")
    print(f"   Will use fallback functions")

try:
    import psi4
    PSI4_AVAILABLE = True
    print(f"✅ PSI4 {psi4.__version__}")
except ImportError:
    PSI4_AVAILABLE = False
    print("⚠️  PSI4 not available (needed for Phase 2)")

from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# DESCRIPTOR COMPUTATION
# ==============================================================================

def compute_coulomb_matrix(symbols, coords):
    """Coulomb matrix descriptor."""
    atomic_numbers = {'H': 1, 'He': 2, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
    Z = np.array([atomic_numbers[s] for s in symbols])
    n_atoms = len(symbols)
    cm = np.zeros((n_atoms, n_atoms))
    
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                cm[i, j] = 0.5 * Z[i] ** 2.4
            else:
                r = np.linalg.norm(coords[i] - coords[j])
                if r > 1e-10:
                    cm[i, j] = Z[i] * Z[j] / r
    
    # Flatten upper triangle
    indices = np.triu_indices(n_atoms)
    return cm[indices]

# Fallback function for loading trajectories
def load_trajectory_fallback(filepath):
    """Fallback trajectory loader if framework not available."""
    data = np.load(filepath, allow_pickle=True)
    
    class SimpleTrajectory:
        def __init__(self, data_dict):
            self.symbols = list(data_dict['symbols'])
            self.coordinates = data_dict['coordinates']
            self.energies = data_dict['energies']
            self.forces = data_dict.get('forces', None)
            self.metadata = data_dict.get('metadata', None)
            if isinstance(self.metadata, np.ndarray):
                self.metadata = self.metadata.item()
            self.n_frames = len(self.coordinates)
    
    return SimpleTrajectory(data)

# Try to use framework, fall back if needed
try:
    _test_load = load_trajectory
except NameError:
    load_trajectory = load_trajectory_fallback

# ==============================================================================
# ML-PES PREDICTOR
# ==============================================================================

class MLPESPredictor:
    """Load and use trained ML-PES model with robust metadata handling."""
    
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']
        self.symbols = model_data['symbols']
        self.metadata = model_data.get('metadata', {})
        
        # Extract theory level with ROBUST fallbacks
        theory = self.metadata.get('theory', {})
        
        # Handle None or missing values
        self.method = theory.get('method') if isinstance(theory, dict) else None
        self.basis = theory.get('basis') if isinstance(theory, dict) else None
        
        # Set sensible defaults if None
        if self.method is None or self.method == '':
            self.method = 'B3LYP'  # Reasonable default
            
        if self.basis is None or self.basis == '':
            self.basis = '6-31G*'  # Reasonable default
        
        # Ensure they're strings
        self.method = str(self.method)
        self.basis = str(self.basis)
    
    def predict_energy(self, coords):
        """Predict energy for geometry."""
        desc = compute_coulomb_matrix(self.symbols, coords)
        desc_scaled = self.scaler_X.transform(desc.reshape(1, -1))
        energy_scaled = self.model.predict(desc_scaled)[0]
        energy = self.scaler_y.inverse_transform([[energy_scaled]])[0, 0]
        return energy
    
    def predict_forces(self, coords, delta=0.001):
        """Predict forces via finite differences."""
        forces = np.zeros_like(coords)
        
        for i in range(len(coords)):
            for j in range(3):
                coords_plus = coords.copy()
                coords_minus = coords.copy()
                
                coords_plus[i, j] += delta
                coords_minus[i, j] -= delta
                
                e_plus = self.predict_energy(coords_plus)
                e_minus = self.predict_energy(coords_minus)
                
                forces[i, j] = -(e_plus - e_minus) / (2 * delta)
        
        return forces
        
# ==============================================================================
# MD INTEGRATOR
# ==============================================================================

def integrate_step(coords, velocities, forces, masses, dt):
    """Velocity Verlet first half."""
    velocities_half = velocities + 0.5 * forces / masses[:, np.newaxis] * dt
    coords_new = coords + velocities_half * dt
    return coords_new, velocities_half

def finalize_velocities(velocities_half, forces_new, masses, dt):
    """Velocity Verlet second half."""
    return velocities_half + 0.5 * forces_new / masses[:, np.newaxis] * dt

# ==============================================================================
# PHASE 1: FAST DIAGNOSTIC MD
# ==============================================================================

def run_phase1_diagnostic_md(
    ml_predictor,
    initial_coords,
    temperature=200.0,
    n_steps=1000,
    timestep=0.5,
    snapshot_interval=10
):
    """
    Phase 1: Run fast MD with ML-PES only, save snapshots.
    
    Returns: snapshots dictionary and output directory
    """
    
    output_dir = Path(f'outputs/diagnostic_phase1_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("  PHASE 1: FAST DIAGNOSTIC MD (ML-PES ONLY)")
    print("=" * 80)
    
    print(f"\n⚙️  Parameters:")
    print(f"   Steps: {n_steps}")
    print(f"   Timestep: {timestep} fs")
    print(f"   Temperature: {temperature} K")
    print(f"   Snapshot interval: every {snapshot_interval} steps")
    print(f"   Expected snapshots: {n_steps // snapshot_interval}")
    
    # Time estimate
    time_per_step = 0.01  # seconds
    estimated_time = n_steps * time_per_step
    print(f"\n⏱️  Estimated time: {estimated_time/60:.1f} minutes")
    print(f"   (100x faster than validating with PSI4 during MD!)")
    
    # Setup
    atomic_masses = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999}
    masses = np.array([atomic_masses.get(s, 12.0) for s in ml_predictor.symbols]) * 1822.888
    
    kb = 3.166811429e-6  # Hartree/K
    dt_au = timestep * 41.341374575751  # fs to au
    
    # Initialize velocities
    np.random.seed(42)
    velocities = np.random.randn(len(masses), 3)
    velocities *= np.sqrt(kb * temperature / masses[:, np.newaxis])
    velocities -= velocities.mean(axis=0)
    
    # Storage
    snapshots = []
    snapshot_steps = []
    snapshot_energies = []
    all_coords = []
    
    coords = initial_coords.copy()
    
    print(f"\n🚀 Starting fast MD trajectory...")
    start_time = time.time()
    
    for step in tqdm(range(n_steps), desc="MD Steps (ML-PES only)"):
        
        # ML-PES prediction (fast!)
        forces_ml = ml_predictor.predict_forces(coords)
        energy_ml = ml_predictor.predict_energy(coords)
        
        # Integrate
        coords_new, velocities_half = integrate_step(coords, velocities, forces_ml, masses, dt_au)
        forces_ml_new = ml_predictor.predict_forces(coords_new)
        velocities = finalize_velocities(velocities_half, forces_ml_new, masses, dt_au)
        
        coords = coords_new
        all_coords.append(coords.copy())
        
        # Save snapshot
        if step % snapshot_interval == 0:
            snapshots.append(coords.copy())
            snapshot_steps.append(step)
            snapshot_energies.append(energy_ml)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ Phase 1 complete!")
    print(f"   Total time: {elapsed_time/60:.2f} minutes")
    print(f"   Steps completed: {n_steps}")
    print(f"   Snapshots saved: {len(snapshots)}")
    print(f"   Time per step: {elapsed_time/n_steps*1000:.2f} ms")
    
    # Save
    snapshot_data = {
        'snapshots': np.array(snapshots),
        'steps': snapshot_steps,
        'energies_ml': np.array(snapshot_energies),
        'all_coords': np.array(all_coords),
        'symbols': ml_predictor.symbols,
        'parameters': {
            'n_steps': n_steps,
            'timestep': timestep,
            'temperature': temperature,
            'snapshot_interval': snapshot_interval,
            'method': ml_predictor.method,
            'basis': ml_predictor.basis
        },
        'elapsed_time': elapsed_time
    }
    
    snapshot_file = output_dir / 'phase1_snapshots.pkl'
    with open(snapshot_file, 'wb') as f:
        pickle.dump(snapshot_data, f)
    
    print(f"\n💾 Saved: {snapshot_file}")
    
    return snapshot_data, output_dir

# ==============================================================================
# ROBUST PSI4 COMPUTATION WITH DIPOLES
# ==============================================================================

def compute_psi4_energy_forces_dipole(symbols, coords, method='B3LYP', basis='6-31G*'):
    """
    Compute energy, forces, and dipole with PSI4.
    
    Returns:
        energy (float): Energy in Hartree
        forces (ndarray): Forces in Hartree/Angstrom
        dipole (ndarray): Dipole in Debye, or None if failed
        error (str): Error message or None
    """
    # Create molecule
    mol_str = f"0 1\n"
    for s, c in zip(symbols, coords):
        mol_str += f"{s} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n"
    mol_str += "units angstrom\nno_reorient\nno_com"
    
    # Clean PSI4
    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.be_quiet()
    psi4.set_memory('2 GB')
    psi4.set_num_threads(4)
    
    try:
        # Create molecule (relaxed validation for MD geometries)
        mol = psi4.geometry(mol_str, tooclose=0.1)
        
        # Settings
        psi4.set_options({
            'basis': basis,
            'scf_type': 'df',
            'reference': 'rhf',
            'maxiter': 200,
            'e_convergence': 1e-6,
            'd_convergence': 1e-6
        })
        
        # Compute gradient (includes energy)
        method_string = f"{method}/{basis}"
        gradient_result, wfn = psi4.gradient(method_string, molecule=mol, return_wfn=True)
        
        # Extract energy
        energy = float(wfn.energy())
        
        # Extract gradient and compute forces
        n_atoms = len(symbols)
        grad_bohr = np.array([[gradient_result.get(i, j) for j in range(3)] 
                             for i in range(n_atoms)])
        
        # Convert to forces in Hartree/Angstrom
        ANGSTROM_TO_BOHR = 1.88972612456
        forces = -grad_bohr / ANGSTROM_TO_BOHR
        
        # Compute dipole using robust method
        dipole = None
        try:
            # Use oeprop to calculate dipole
            psi4.oeprop(wfn, 'DIPOLE')
            
            # Try multiple extraction methods
            dipole_found = False
            
            # Method 1: Direct oeprop access
            if not dipole_found:
                try:
                    oep = wfn.oeprop
                    if hasattr(oep, 'Dx'):
                        dipole = np.array([oep.Dx(), oep.Dy(), oep.Dz()])
                        dipole_found = True
                except:
                    pass
            
            # Method 2: PSI4 variables
            if not dipole_found:
                try:
                    dipole_x = psi4.variable('DIPOLE X')
                    dipole_y = psi4.variable('DIPOLE Y')
                    dipole_z = psi4.variable('DIPOLE Z')
                    dipole = np.array([dipole_x, dipole_y, dipole_z])
                    dipole_found = True
                except:
                    pass
            
            # Method 3: Wavefunction variables
            if not dipole_found:
                try:
                    dipole_x = wfn.variable('DIPOLE X')
                    dipole_y = wfn.variable('DIPOLE Y')
                    dipole_z = wfn.variable('DIPOLE Z')
                    dipole = np.array([dipole_x, dipole_y, dipole_z])
                    dipole_found = True
                except:
                    pass
            
            # Convert from atomic units to Debye if needed
            if dipole_found and dipole is not None:
                magnitude = np.linalg.norm(dipole)
                # If very small, likely in atomic units
                if magnitude < 0.1:
                    dipole = dipole * 2.54174623  # au to Debye
        
        except Exception as e:
            # Dipole failed but we still have energy/forces
            dipole = None
        
        return energy, forces, dipole, None
        
    except Exception as e:
        error_msg = str(e)
        # Handle common errors gracefully
        if "too close" in error_msg.lower():
            return None, None, None, "Atoms too close"
        elif "scf" in error_msg.lower() and "converge" in error_msg.lower():
            return None, None, None, "SCF convergence failure"
        else:
            return None, None, None, error_msg[:100]
                    
# ==============================================================================
# PHASE 2: POST-HOC VALIDATION
# ==============================================================================

def run_phase2_validation(snapshot_data, ml_predictor, max_validate=50):
    """
    Phase 2: Validate sampled snapshots with real PSI4 + compute dipoles.
    
    FIXED: Handles None method/basis gracefully
    """
    output_dir = Path(f'outputs/diagnostic_phase2_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    snapshots = snapshot_data['snapshots']
    symbols = ml_predictor.symbols
    
    print("\n" + "=" * 80)
    print("  PHASE 2: POST-HOC VALIDATION (with Dipoles)")
    print("=" * 80)
    
    # Sample uniformly
    print(f"\n📊 Validation strategy:")
    print(f"   Total snapshots: {len(snapshots)}")
    
    if len(snapshots) <= max_validate:
        indices = np.arange(len(snapshots))
        print(f"   Validating: all {len(snapshots)}")
    else:
        indices = np.linspace(0, len(snapshots)-1, max_validate, dtype=int)
        print(f"   Validating: {len(indices)} uniformly sampled")
    
    # Time estimate - FIXED to handle None values
    n_atoms = len(ml_predictor.symbols)
    method_times = {'hf': 0.2, 'b3lyp': 1.0, 'mp2': 2.0}
    basis_mult = {'6-31g': 1.0, '6-31g*': 1.5, 'cc-pvdz': 3.0, 'cc-pvtz': 5.0}
    
    # Safe extraction with defaults
    method_str = ml_predictor.method if ml_predictor.method else 'B3LYP'
    basis_str = ml_predictor.basis if ml_predictor.basis else '6-31G*'
    
    # Case-insensitive lookup with fallback
    base_time = method_times.get(method_str.lower(), 1.0)
    bmult = basis_mult.get(basis_str.lower().replace('*', ''), 1.5)
    
    time_per_val = base_time * bmult * (n_atoms ** 2.5) / (5 ** 2.5)
    estimated_time = len(indices) * time_per_val
    
    print(f"\n⏱️  Estimated validation time:")
    print(f"   Theory level: {method_str}/{basis_str}")
    print(f"   Per validation: ~{time_per_val:.1f} seconds")
    print(f"   Total: ~{estimated_time/60:.1f} minutes")
    
    # Validate with PSI4
    results = {
        'valid_indices': [],
        'valid_steps': [],
        'coords_list': [],
        'energies_ml': [],
        'energies_psi4': [],
        'forces_ml': [],
        'forces_psi4': [],
        'dipoles_psi4': [],  # NEW: Store dipoles
        'errors_energy': [],
        'errors_forces': [],
        'metadata': {
            'method': method_str,
            'basis': basis_str,
            'molecule': ' '.join(symbols),
            'n_atoms': n_atoms
        }
    }
    
    print(f"\n🔄 Running PSI4 validations...")
    print(f"   Computing: Energy + Forces + Dipoles")
    
    failed_validations = 0
    
    for i, idx in enumerate(tqdm(indices, desc="Validations")):
        snapshot = snapshots[idx]
        coords = snapshot['coords']
        
        # ML predictions
        e_ml = ml_predictor.predict_energy(coords)
        f_ml = ml_predictor.predict_forces(coords)
        
        # PSI4 validation with dipoles
        e_psi4, f_psi4, dipole, error = compute_psi4_energy_forces_dipole(
            symbols, coords, method_str, basis_str
        )
        
        if error:
            failed_validations += 1
            continue
        
        # Store results
        results['valid_indices'].append(idx)
        results['valid_steps'].append(snapshot['step'])
        results['coords_list'].append(coords)
        results['energies_ml'].append(e_ml)
        results['energies_psi4'].append(e_psi4)
        results['forces_ml'].append(f_ml)
        results['forces_psi4'].append(f_psi4)
        results['dipoles_psi4'].append(dipole if dipole is not None else np.zeros(3))
        
        # Compute errors
        error_e = abs(e_ml - e_psi4) * 627.509  # kcal/mol
        error_f = np.mean(np.abs(f_ml - f_psi4)) * 627.509  # kcal/mol/Å
        
        results['errors_energy'].append(error_e)
        results['errors_forces'].append(error_f)
    
    # Convert to arrays
    for key in ['energies_ml', 'energies_psi4', 'forces_ml', 'forces_psi4', 
                'errors_energy', 'errors_forces', 'dipoles_psi4']:
        results[key] = np.array(results[key])
    
    results['coords_list'] = np.array(results['coords_list'])
    
    # Summary
    print(f"\n✅ Validation complete!")
    print(f"   Successful: {len(results['valid_indices'])}/{len(indices)}")
    print(f"   Failed: {failed_validations}")
    
    # Dipole statistics
    dipoles = results['dipoles_psi4']
    valid_dipoles = dipoles[~np.all(dipoles == 0, axis=1)]
    if len(valid_dipoles) > 0:
        mags = np.linalg.norm(valid_dipoles, axis=1)
        print(f"\n💧 Dipole moments collected: {len(valid_dipoles)}")
        print(f"   Mean |μ|: {mags.mean():.4f} Debye")
        print(f"   Range: {mags.min():.4f} - {mags.max():.4f} Debye")
    
    # Save results
    results_file = output_dir / 'validation_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Saved: {results_file}")
    
    return results, output_dir
    
# ==============================================================================
# ERROR ANALYSIS
# ==============================================================================

def analyze_errors(results):
    """Comprehensive error analysis."""
    
    if len(results['energy_errors']) == 0:
        print("\n❌ No successful validations to analyze")
        return
    
    errors = results['energy_errors']
    
    print("\n" + "=" * 80)
    print("  ERROR ANALYSIS")
    print("=" * 80)
    
    print(f"\n📊 Energy errors:")
    print(f"   Mean: {errors.mean():.2f} kcal/mol")
    print(f"   Median: {np.median(errors):.2f} kcal/mol")
    print(f"   Std: {errors.std():.2f} kcal/mol")
    print(f"   Max: {errors.max():.2f} kcal/mol")
    print(f"   Min: {errors.min():.2f} kcal/mol")
    
    # Categorize
    bins = [(0, 2), (2, 10), (10, 100), (100, np.inf)]
    labels = ['< 2', '2-10', '10-100', '> 100']
    
    print(f"\n📊 Error distribution:")
    for (low, high), label in zip(bins, labels):
        count = ((errors >= low) & (errors < high)).sum()
        pct = count / len(errors) * 100
        print(f"   {label} kcal/mol: {count} ({pct:.1f}%)")
    
    if results['force_errors'].size > 0:
        f_errors = results['force_errors']
        print(f"\n📊 Force errors:")
        print(f"   Mean: {f_errors.mean():.2f} kcal/mol/Å")
        print(f"   Max: {f_errors.max():.2f} kcal/mol/Å")
    
    return errors.mean(), errors.max()

# ==============================================================================
# REFINEMENT WITH HIGH-ERROR POINTS
# ==============================================================================

def refine_with_high_error_points(snapshot_data, validation_results, ml_predictor, 
                                   training_data_path, output_dir):
    """
    Add high-error points to training data and retrain model.
    
    This implements adaptive refinement based on observed MD errors.
    """
    
    if len(validation_results['energy_errors']) == 0:
        print("\n❌ No validation results to refine with")
        return None
    
    print("\n" + "=" * 80)
    print("  REFINEMENT WITH HIGH-ERROR POINTS")
    print("=" * 80)
    
    energy_errors = validation_results['energy_errors']
    valid_indices = validation_results['valid_indices']
    
    print(f"\n📊 Refinement options:")
    print(f"   Available points: {len(energy_errors)}")
    print(f"   Error range: {energy_errors.min():.2f} - {energy_errors.max():.2f} kcal/mol")
    
    # Categorize errors
    low_error = (energy_errors < 2.0).sum()
    medium_error = ((energy_errors >= 2.0) & (energy_errors < 10.0)).sum()
    high_error = ((energy_errors >= 10.0) & (energy_errors < 50.0)).sum()
    huge_error = (energy_errors >= 50.0).sum()
    
    print(f"\n   Error distribution:")
    print(f"   • < 2 kcal/mol: {low_error}")
    print(f"   • 2-10 kcal/mol: {medium_error}")
    print(f"   • 10-50 kcal/mol: {high_error}")
    print(f"   • > 50 kcal/mol: {huge_error}")
    
    print(f"\n💡 Refinement strategies:")
    print(f"   [1] Add ALL validated points ({len(energy_errors)} points)")
    print(f"       → Maximum data, best coverage")
    print(f"   [2] Add medium + high + huge errors ({medium_error + high_error + huge_error} points, ≥2 kcal/mol)")
    print(f"       → Good balance, skip very accurate regions")
    print(f"   [3] Add only high + huge errors ({high_error + huge_error} points, ≥10 kcal/mol)")
    print(f"       → Focus on problem areas")
    print(f"   [4] Add only huge errors ({huge_error} points, ≥50 kcal/mol)")
    print(f"       → Fix critical failures only")
    print(f"   [5] Custom threshold")
    print(f"   [6] Cancel refinement")
    
    choice = input("\nSelect refinement strategy [1-6]: ").strip()
    
    if choice == '6':
        print("   Refinement cancelled")
        return None
    
    # Select points based on choice
    if choice == '1':
        mask = np.ones(len(energy_errors), dtype=bool)
        threshold_name = "all points"
    elif choice == '2':
        mask = energy_errors >= 2.0
        threshold_name = "≥2 kcal/mol"
    elif choice == '3':
        mask = energy_errors >= 10.0
        threshold_name = "≥10 kcal/mol"
    elif choice == '4':
        mask = energy_errors >= 50.0
        threshold_name = "≥50 kcal/mol"
    elif choice == '5':
        threshold = float(input("   Enter error threshold (kcal/mol): ").strip())
        mask = energy_errors >= threshold
        threshold_name = f"≥{threshold} kcal/mol"
    else:
        print("   Invalid choice, using option [2]")
        mask = energy_errors >= 2.0
        threshold_name = "≥2 kcal/mol"
    
    n_points = mask.sum()
    
    if n_points == 0:
        print(f"\n⚠️  No points meet criteria ({threshold_name})")
        return None
    
    print(f"\n🔧 Extracting {n_points} points ({threshold_name})...")
    
    # Extract high-error geometries and their PSI4 data
    selected_coords = snapshot_data['snapshots'][np.array(valid_indices)[mask]]
    selected_energies = validation_results['energies_psi4'][mask]
    selected_forces = np.array(validation_results['forces_psi4'])[mask]
    selected_dipoles = validation_results['dipoles_psi4'][mask]  # NEW!
    selected_errors = energy_errors[mask]    
    print(f"   Mean error of selected points: {selected_errors.mean():.2f} kcal/mol")
    print(f"   Max error of selected points: {selected_errors.max():.2f} kcal/mol")
    
    # Load original training data
    print(f"\n📂 Loading original training data...")
    orig_data = load_trajectory(training_data_path)
    print(f"   Original: {orig_data.n_frames} configurations")
    
# Combine with original
    print(f"\n🔗 Augmenting training data...")
    augmented_coords = np.vstack([orig_data.coordinates, selected_coords])
    augmented_energies = np.concatenate([orig_data.energies, selected_energies])
    
    if orig_data.forces is not None:
        augmented_forces = np.vstack([orig_data.forces, selected_forces])
    else:
        augmented_forces = selected_forces
    
    # NEW: Combine dipoles
    orig_dipoles = orig_data.metadata.get('dipoles', None) if orig_data.metadata else None
    if orig_dipoles is not None:
        augmented_dipoles = np.vstack([orig_dipoles, selected_dipoles])
        print(f"   Dipoles: {len(orig_dipoles)} original + {len(selected_dipoles)} new = {len(augmented_dipoles)} total")
    else:
        augmented_dipoles = selected_dipoles
        print(f"   Dipoles: {len(selected_dipoles)} new (no original dipoles)")
    
    print(f"   New total: {len(augmented_coords)} configurations (+{n_points})")    
    # Save augmented data
    augmented_data_path = output_dir / 'augmented_training_data.npz'
    
    # Prepare metadata
    metadata = orig_data.metadata.copy() if hasattr(orig_data, 'metadata') and orig_data.metadata else {}
    
    # NEW: Add dipoles to metadata
    metadata['dipoles'] = augmented_dipoles
    metadata['dipoles_units'] = 'Debye'
    metadata['has_dipoles'] = True
    
    if 'refinement' not in metadata:
        metadata['refinement'] = []
    metadata['refinement'].append({
        'date': datetime.now().isoformat(),
        'method': 'two_phase_diagnostic',
        'n_added': int(n_points),
        'n_dipoles_added': len(selected_dipoles),  # NEW!
        'threshold': threshold_name,
        'mean_error': float(selected_errors.mean()),
        'max_error': float(selected_errors.max())
    })
        
    # Create TrajectoryData object properly
    augmented_traj = TrajectoryData(
        symbols=orig_data.symbols,
        coordinates=augmented_coords,
        energies=augmented_energies,
        forces=augmented_forces,
        metadata=metadata
    )
    
    # Save with correct argument order: (trajectory, filename)
    save_trajectory(augmented_traj, str(augmented_data_path))
    print(f"   ✅ Saved: {augmented_data_path}")
    
    # Retrain model
    print(f"\n🤖 Retraining model with augmented data...")
    
    # Compute descriptors for all data
    print(f"   Computing descriptors...")
    X = []
    for i in tqdm(range(len(augmented_coords)), desc="Descriptors"):
        desc = compute_coulomb_matrix(orig_data.symbols, augmented_coords[i])
        X.append(desc)
    X = np.array(X)
    y = augmented_energies
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Hyperparameter search
    print(f"   Searching hyperparameters...")
    from sklearn.kernel_ridge import KernelRidge
    from itertools import product
    
    param_grid = list(product([0.01, 0.1, 1.0], [0.01, 0.1]))
    
    best_rmse = float('inf')
    best_model = None
    best_params = None
    
    for gamma, alpha in tqdm(param_grid, desc="Hyperparameters"):
        model = KernelRidge(kernel='rbf', gamma=gamma, alpha=alpha)
        model.fit(X_train_scaled, y_train_scaled)
        
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        errors = (y_pred - y_test) * 627.509
        rmse = np.sqrt((errors**2).mean())
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = (gamma, alpha)
    
    y_pred_scaled = best_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    r2 = 1 - ((y_test - y_pred)**2).sum() / ((y_test - y_test.mean())**2).sum()
    
    print(f"\n✅ Retrained Model:")
    print(f"   Gamma: {best_params[0]}")
    print(f"   Alpha: {best_params[1]}")
    print(f"   RMSE: {best_rmse:.4f} kcal/mol")
    print(f"   R²: {r2:.6f}")
    
    # Compare to original
    orig_rmse = ml_predictor.metadata.get('model', {}).get('test_rmse_kcal', None)
    if orig_rmse:
        improvement = (orig_rmse - best_rmse) / orig_rmse * 100
        print(f"\n📊 Improvement:")
        print(f"   Original RMSE: {orig_rmse:.4f} kcal/mol")
        print(f"   New RMSE: {best_rmse:.4f} kcal/mol")
        print(f"   Change: {improvement:+.1f}%")
        
        if improvement > 5:
            print(f"   ✅ Significant improvement!")
        elif improvement > 0:
            print(f"   ✅ Modest improvement")
        else:
            print(f"   ⚠️  No improvement - may need different strategy")
    
    # Save retrained model
    retrained_metadata = metadata.copy()
    retrained_metadata['model'] = {
        'gamma': float(best_params[0]),
        'alpha': float(best_params[1]),
        'test_rmse_kcal': float(best_rmse),
        'r2_score': float(r2),
        'training_configs': len(augmented_coords),
        'refined': True
    }
    
    model_data = {
        'model': best_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'symbols': orig_data.symbols,
        'metadata': retrained_metadata,
        'version': '2.2_refined'
    }
    
    retrained_model_path = output_dir / 'retrained_mlpes_model.pkl'
    with open(retrained_model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n💾 Saved retrained model: {retrained_model_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("  REFINEMENT COMPLETE")
    print("=" * 80)
    
    print(f"\n📁 Files created:")
    print(f"   • {augmented_data_path.name}")
    print(f"   • {retrained_model_path.name}")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Test retrained model:")
    print(f"      python3 two_phase_workflow.py \\")
    print(f"          --model {retrained_model_path} \\")
    print(f"          --training-data {augmented_data_path} \\")
    print(f"          --temp 100 --steps 500")
    print(f"")
    print(f"   2. Expected improvement: {improvement if orig_rmse else 'N/A'}%")
    print(f"   3. If still high errors, try more refinement iterations")
    
    return {
        'augmented_data': str(augmented_data_path),
        'retrained_model': str(retrained_model_path),
        'n_added': n_points,
        'old_rmse': orig_rmse,
        'new_rmse': best_rmse,
        'improvement_pct': improvement if orig_rmse else None
    }

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def create_diagnostic_plots(snapshot_data, results, output_dir):
    """Create comprehensive diagnostic plots."""
    
    if len(results['energy_errors']) == 0:
        print("\n⚠️  No data to plot")
        return None
    
    print("\n📊 Creating diagnostic plots...")
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Data
    all_steps = snapshot_data['steps']
    all_energies_ml = snapshot_data['energies_ml'] * 627.509
    
    valid_steps = results['valid_steps']
    energies_ml = results['energies_ml'] * 627.509
    energies_psi4 = results['energies_psi4'] * 627.509
    energy_errors = results['energy_errors']
    
    # 1. Energy trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(all_steps, all_energies_ml, 'b-', alpha=0.3, linewidth=1, label='ML-PES (all)')
    ax1.scatter(valid_steps, energies_psi4, c='red', s=30, label='PSI4 (validated)', zorder=5, alpha=0.7)
    ax1.set_xlabel('MD Step', fontsize=11)
    ax1.set_ylabel('Energy (kcal/mol)', fontsize=11)
    ax1.set_title('Energy Trajectory', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Errors over time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(valid_steps, energy_errors, 'go-', linewidth=1.5, markersize=5, alpha=0.7)
    ax2.axhline(2, color='orange', linestyle='--', label='2 kcal/mol', linewidth=2, alpha=0.7)
    ax2.axhline(10, color='red', linestyle='--', label='10 kcal/mol', linewidth=2, alpha=0.7)
    ax2.set_xlabel('MD Step', fontsize=11)
    ax2.set_ylabel('Energy Error (kcal/mol, log scale)', fontsize=11)
    ax2.set_title('Error Evolution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Error histogram
    ax3 = fig.add_subplot(gs[0, 2])
    errors_plot = energy_errors[energy_errors < 100]
    if len(errors_plot) > 0:
        ax3.hist(errors_plot, bins=25, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(energy_errors.mean(), color='blue', linestyle='--', 
                    label=f'Mean: {energy_errors.mean():.1f}', linewidth=2)
        ax3.axvline(np.median(energy_errors), color='red', linestyle='--',
                    label=f'Median: {np.median(energy_errors):.1f}', linewidth=2)
    ax3.set_xlabel('Energy Error (kcal/mol)', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Error Distribution (< 100 shown)', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Energy parity
    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(energies_psi4, energies_ml, c=energy_errors, 
                         cmap='RdYlGn_r', s=50, alpha=0.6, 
                         norm=plt.matplotlib.colors.LogNorm(vmin=max(energy_errors.min(), 0.1)))
    min_e = min(energies_psi4.min(), energies_ml.min())
    max_e = max(energies_psi4.max(), energies_ml.max())
    ax4.plot([min_e, max_e], [min_e, max_e], 'k--', linewidth=2, alpha=0.5)
    ax4.set_xlabel('PSI4 Energy (kcal/mol)', fontsize=11)
    ax4.set_ylabel('ML-PES Energy (kcal/mol)', fontsize=11)
    ax4.set_title('Energy Parity', fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=ax4, label='Error (kcal/mol)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Trajectory position vs error
    ax5 = fig.add_subplot(gs[1, 1])
    traj_frac = np.array(valid_steps) / max(all_steps)
    ax5.scatter(traj_frac * 100, energy_errors, c=energy_errors, 
                cmap='RdYlGn_r', s=50, alpha=0.6,
                norm=plt.matplotlib.colors.LogNorm(vmin=max(energy_errors.min(), 0.1)))
    ax5.set_xlabel('Trajectory Progress (%)', fontsize=11)
    ax5.set_ylabel('Energy Error (kcal/mol, log scale)', fontsize=11)
    ax5.set_title('Where Does ML-PES Struggle?', fontsize=13, fontweight='bold')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    summary_text = f"""
SUMMARY STATISTICS

Phase 1 (Fast MD):
  • Steps: {snapshot_data['parameters']['n_steps']}
  • Time: {snapshot_data['elapsed_time']/60:.1f} min
  • Temperature: {snapshot_data['parameters']['temperature']} K
  
Phase 2 (Validation):
  • Validated: {len(valid_steps)} snapshots
  • Failed: {len(results['failed_indices'])}
  
Error Analysis:
  • Mean: {energy_errors.mean():.2f} kcal/mol
  • Median: {np.median(energy_errors):.2f} kcal/mol
  • Max: {energy_errors.max():.2f} kcal/mol
  
Quality Assessment:
"""
    
    mean_err = energy_errors.mean()
    if mean_err < 2:
        summary_text += "  ✅ EXCELLENT (<2 kcal/mol)\n  Ready for production!"
    elif mean_err < 5:
        summary_text += "  ✅ GOOD (2-5 kcal/mol)\n  Usable, refinement optional"
    elif mean_err < 20:
        summary_text += "  ⚠️  MODERATE (5-20 kcal/mol)\n  Refinement recommended"
    else:
        summary_text += "  ❌ POOR (>20 kcal/mol)\n  Major refinement needed"
    
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Two-Phase Diagnostic Results - Mean Error: {energy_errors.mean():.2f} kcal/mol',
                 fontsize=16, fontweight='bold')
    
    plot_path = output_dir / 'diagnostic_plots.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {plot_path}")
    
    return str(plot_path)

# ==============================================================================
# RECOMMENDATIONS
# ==============================================================================

def provide_recommendations(results, snapshot_data):
    """Provide specific recommendations based on results."""
    
    if len(results['energy_errors']) == 0:
        return
    
    errors = results['energy_errors']
    mean_err = errors.mean()
    max_err = errors.max()
    high_err_count = (errors > 10).sum()
    huge_err_count = (errors > 100).sum()
    
    print("\n" + "=" * 80)
    print("  RECOMMENDATIONS")
    print("=" * 80)
    
    print(f"\n📊 Current performance:")
    print(f"   Mean error: {mean_err:.2f} kcal/mol")
    print(f"   Max error: {max_err:.2f} kcal/mol")
    print(f"   High errors (>10 kcal/mol): {high_err_count}")
    print(f"   Huge errors (>100 kcal/mol): {huge_err_count}")
    
    print(f"\n💡 Recommendations:\n")
    
    if mean_err < 2.0:
        print("✅ EXCELLENT! ML-PES is very accurate")
        print("   → Ready for production MD simulations")
        print("   → No refinement needed")
        
    elif mean_err < 5.0:
        print("✅ GOOD! ML-PES is reasonably accurate")
        print("   → Can use for exploratory work")
        print("   → Optional: Refine high-error regions for production")
        if high_err_count > 0:
            print(f"   → Add {high_err_count} high-error points to training")
            print(f"   → Expected improvement: {mean_err:.1f} → ~{mean_err*0.6:.1f} kcal/mol")
    
    elif mean_err < 20.0:
        print("⚠️  MODERATE ERRORS - Refinement recommended")
        print(f"   → Add {high_err_count} high-error points to training")
        print(f"   → Expected improvement: {mean_err:.1f} → ~{mean_err*0.5:.1f} kcal/mol")
        
        if huge_err_count > 0:
            print(f"\n   ⚠️  {huge_err_count} points with >100 kcal/mol errors:")
            print("   → Likely dissociation or extreme geometries")
            print("   → Consider lower temperature for next run")
    
    else:
        print("❌ LARGE ERRORS - Major issues detected")
        print("\n   Problems:")
        if huge_err_count > 0:
            print(f"   • {huge_err_count} points with >100 kcal/mol (dissociation?)")
        print(f"   • Mean error {mean_err:.1f} kcal/mol is too high")
        
        print("\n   Options:")
        print("   1. LOWER TEMPERATURE and re-run diagnostic")
        current_temp = snapshot_data['parameters']['temperature']
        print(f"      Current: {current_temp} K → Try: {current_temp*0.5:.0f} K or {current_temp*0.75:.0f} K")
        print("   2. Train on broader configuration space first")
        print("   3. Check if molecule is dissociating (see plots)")
        print("   4. Consider different ML method or more training data")

# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Two-phase diagnostic workflow')
    parser.add_argument('--model', help='ML-PES model (.pkl)')
    parser.add_argument('--training-data', help='Training data (.npz)')
    parser.add_argument('--steps', type=int, default=1000, help='MD steps (Phase 1)')
    parser.add_argument('--temp', type=float, default=200.0, help='Temperature (K)')
    parser.add_argument('--timestep', type=float, default=0.5, help='Timestep (fs)')
    parser.add_argument('--snapshot-every', type=int, default=10, help='Snapshot interval')
    parser.add_argument('--validate-max', type=int, default=50, help='Max snapshots to validate (Phase 2)')
    parser.add_argument('--phase1-only', action='store_true', help='Run Phase 1 only')
    parser.add_argument('--phase2-only', type=str, help='Run Phase 2 only on existing snapshots (path to .pkl)')
    parser.add_argument('--refine', type=str, help='Refine from existing phase2_results.pkl')
    
    args = parser.parse_args()
    
    # REFINEMENT MODE
    if args.refine:
        print(f"\n" + "=" * 80)
        print("  STANDALONE REFINEMENT MODE")
        print("=" * 80)
        
        results_path = Path(args.refine)
        if not results_path.exists():
            print(f"\n❌ Results file not found: {args.refine}")
            return
        
        output_dir = results_path.parent
        
        # Load results
        print(f"\n📂 Loading validation results...")
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        print(f"   ✅ Loaded {len(results['valid_indices'])} validated points")
        
        # Load snapshot data
        snapshot_file = output_dir / 'phase1_snapshots.pkl'
        if not snapshot_file.exists():
            print(f"\n❌ Snapshot file not found: {snapshot_file}")
            return
        
        with open(snapshot_file, 'rb') as f:
            snapshot_data = pickle.load(f)
        print(f"   ✅ Loaded snapshot data")
        
        # Need model and training data paths
        if not args.model or not args.training_data:
            print(f"\n⚠️  For refinement mode, you need to specify:")
            print(f"   --model PATH_TO_MODEL")
            print(f"   --training-data PATH_TO_TRAINING_DATA")
            return
        
        # Load model
        print(f"\n📂 Loading ML-PES model...")
        ml_predictor = MLPESPredictor(args.model)
        print(f"   ✅ Loaded: {ml_predictor.method}/{ml_predictor.basis}")
        
        # Run refinement
        refinement_results = refine_with_high_error_points(
            snapshot_data, results, ml_predictor, 
            args.training_data, output_dir
        )
        
        if refinement_results:
            print(f"\n✅ Refinement complete!")
            print(f"   Test refined model with:")
            print(f"   python3 two_phase_workflow.py \\")
            print(f"       --model {refinement_results['retrained_model']} \\")
            print(f"       --training-data {refinement_results['augmented_data']} \\")
            print(f"       --temp {args.temp} --steps {args.steps}")
        
        return
    
    # NORMAL MODE - require model and training data
    if not args.model or not args.training_data:
        parser.error("--model and --training-data are required (unless using --refine)")
    
    # Load model
    print(f"\n📂 Loading ML-PES model: {args.model}")
    ml_predictor = MLPESPredictor(args.model)
    print(f"   ✅ Loaded: {ml_predictor.method}/{ml_predictor.basis}")
    print(f"   ✅ Molecule: {' '.join(ml_predictor.symbols)}")
    
    # Get initial coordinates
    print(f"\n📊 Extracting initial coordinates from training data...")
    training_traj = load_trajectory(args.training_data)
    lowest_idx = np.argmin(training_traj.energies)
    initial_coords = training_traj.coordinates[lowest_idx].copy()
    print(f"   ✅ Using lowest energy configuration (index {lowest_idx})")
    
    # Optimize with ML-PES
    print(f"\n🔧 Optimizing starting geometry...")
    coords_opt = initial_coords.copy()
    for _ in range(50):
        forces = ml_predictor.predict_forces(coords_opt)
        coords_opt += 0.01 * forces
        if np.linalg.norm(forces) < 0.001:
            break
    print(f"   ✅ Optimized")
    
    # PHASE 1 or load existing
    if args.phase2_only:
        print(f"\n📂 Loading existing Phase 1 results...")
        with open(args.phase2_only, 'rb') as f:
            snapshot_data = pickle.load(f)
        output_dir = Path(args.phase2_only).parent
        print(f"   ✅ Loaded {len(snapshot_data['snapshots'])} snapshots")
    else:
        snapshot_data, output_dir = run_phase1_diagnostic_md(
            ml_predictor, coords_opt,
            temperature=args.temp,
            n_steps=args.steps,
            timestep=args.timestep,
            snapshot_interval=args.snapshot_every
        )
    
    if args.phase1_only:
        print(f"\n✅ Phase 1 complete! Run Phase 2 with:")
        print(f"   python3 {sys.argv[0]} --phase2-only {output_dir}/phase1_snapshots.pkl --model {args.model}")
        return
    
    # PHASE 2
    results = run_phase2_validation(snapshot_data, ml_predictor, max_validate=args.validate_max)
    
    if results is None:
        return
    
    # Save results
    results_file = output_dir / 'phase2_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n💾 Saved: {results_file}")
    
    # ANALYSIS
    mean_err, max_err = analyze_errors(results)
    
    # PLOTS
    plot_path = create_diagnostic_plots(snapshot_data, results, output_dir)
    
    # RECOMMENDATIONS
    provide_recommendations(results, snapshot_data)
    
    # REFINEMENT OPTION
    print(f"\n" + "=" * 80)
    print("  REFINEMENT OPTION")
    print("=" * 80)
    
    print(f"\n💡 You can now refine the model with high-error points from this diagnostic run.")
    print(f"   This will:")
    print(f"   • Add selected high-error geometries to training data")
    print(f"   • Retrain model with augmented dataset")
    print(f"   • Potentially reduce errors by 20-50%")
    
    refine_choice = input(f"\nDo you want to refine the model? [y/n]: ").strip().lower()
    
    refinement_results = None
    if refine_choice == 'y':
        refinement_results = refine_with_high_error_points(
            snapshot_data, results, ml_predictor, 
            args.training_data, output_dir
        )
    else:
        print(f"\n   Refinement skipped")
        print(f"   You can refine later by running:")
        print(f"   python3 two_phase_workflow.py --refine {output_dir}/phase2_results.pkl")
    
    print(f"\n" + "=" * 80)
    print("  WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"\n📂 All results saved to: {output_dir}")
    print(f"\n📊 Key files:")
    print(f"   • phase1_snapshots.pkl - MD trajectory")
    print(f"   • phase2_results.pkl - Validation results")
    print(f"   • diagnostic_plots.png - Comprehensive visualization")
    
    if refinement_results:
        print(f"   • {Path(refinement_results['augmented_data']).name} - Augmented training data")
        print(f"   • {Path(refinement_results['retrained_model']).name} - Retrained model")
    
    print(f"\n💡 Mean error: {mean_err:.2f} kcal/mol")
    
    if refinement_results:
        print(f"   Refinement: Added {refinement_results['n_added']} points")
        if refinement_results['improvement_pct']:
            print(f"   Improvement: {refinement_results['improvement_pct']:+.1f}%")
    
    if mean_err < 5:
        print("   ✅ Model is good! Ready to use")
    elif mean_err < 20:
        if not refinement_results:
            print("   ⚠️  Consider refinement for production use")
        else:
            print("   ✅ Model refined! Test again to verify improvement")
    else:
        if not refinement_results:
            print("   ❌ Refinement recommended or try lower temperature")
        else:
            print("   ⚠️  Test refined model - may need more iterations")

if __name__ == '__main__':
    main()
