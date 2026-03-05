#!/usr/bin/env python3
"""
On-the-Fly ML-PES Validation and Training Augmentation

Run MD trajectory with ML-PES and periodically validate against PSI4.
High-error points are automatically added to training data and model can be retrained.

Features:
- Run MD with ML-PES (fast)
- Validate against PSI4 every N steps
- Track energy and force errors
- Add high-error points to training data
- Optional on-the-fly retraining
- Comprehensive error analysis and visualization

Author: PSI4-MD Framework
Version: 2.2
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

print("=" * 80)
print("  ON-THE-FLY ML-PES VALIDATION & TRAINING AUGMENTATION")
print("=" * 80)

# Import framework
try:
    from modules.test_molecules import get_molecule
    from modules.data_formats import TrajectoryData, save_trajectory, load_trajectory
    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    print(f"❌ Framework import failed: {e}")
    sys.exit(1)

try:
    import psi4
    PSI4_AVAILABLE = True
    print(f"✅ PSI4 {psi4.__version__}")
except ImportError:
    print("❌ PSI4 required for validation")
    sys.exit(1)

from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from itertools import product

# ==============================================================================
# TIMING AND DIAGNOSTICS
# ==============================================================================

def estimate_time_per_step(method, basis, n_atoms):
    """Estimate time per PSI4 calculation."""
    method_times = {'hf': 0.2, 'b3lyp': 1.0, 'wb97x-d': 1.2, 'pbe0': 0.8, 'mp2': 5.0}
    basis_multipliers = {'sto-3g': 0.5, '6-31g': 1.0, '6-31g*': 1.5, '6-31+g**': 2.5, 'cc-pvdz': 3.0, 'cc-pvtz': 8.0}
    
    base_time = method_times.get(method.lower(), 1.0)
    basis_mult = basis_multipliers.get(basis.lower(), 1.5)
    
    return base_time * basis_mult * (n_atoms ** 2.5) / (5 ** 2.5)

def check_geometry_sanity(symbols, coords):
    """Check if geometry is reasonable (not dissociated/collapsed)."""
    n_atoms = len(symbols)
    atomic_numbers = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
    
    # Compute nuclear repulsion
    nuc_rep = 0.0
    min_dist = 100.0
    
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            min_dist = min(min_dist, dist)
            
            Z_i = atomic_numbers.get(symbols[i], 1)
            Z_j = atomic_numbers.get(symbols[j], 1)
            nuc_rep += Z_i * Z_j / dist
            
            # Check for collapsed atoms
            if dist < 0.8:  # < 0.42 Å
                return False, f"Atoms collapsed (dist={dist:.2f} Bohr)"
    
    # Check for dissociation
    if min_dist > 10.0:  # > 5.3 Å
        return False, f"Molecule dissociated (min_dist={min_dist:.2f} Bohr)"
    
    # Check nuclear repulsion
    if n_atoms >= 3:
        if nuc_rep < 5.0:
            return False, f"Dissociated (nuc_rep={nuc_rep:.1f} Ha)"
        if nuc_rep > 2000.0:
            return False, f"Collapsed (nuc_rep={nuc_rep:.1f} Ha)"
    
    return True, f"OK (nuc_rep={nuc_rep:.1f} Ha)"

# ==============================================================================
# COULOMB MATRIX
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
                r_ij = np.linalg.norm(coords[i] - coords[j])
                if r_ij > 1e-10:
                    cm[i, j] = Z[i] * Z[j] / r_ij
    
    return cm[np.triu_indices(n_atoms)].flatten()

# ==============================================================================
# ML-PES PREDICTOR
# ==============================================================================

class MLPESPredictor:
    """Wrapper for ML-PES predictions."""
    
    def __init__(self, model_path: str):
        """Load ML-PES model."""
        
        print(f"\n📂 Loading ML-PES model: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model_data = pickle.load(f)
        
        self.model = self.model_data['model']
        self.scaler_X = self.model_data['scaler_X']
        self.scaler_y = self.model_data['scaler_y']
        self.symbols = self.model_data['symbols']
        self.metadata = self.model_data.get('metadata', {})
        
        # Extract theory level
        theory = self.metadata.get('theory', {})
        self.method = theory.get('method', 'unknown')
        self.basis = theory.get('basis', 'unknown')
        
        print(f"✅ Loaded model:")
        print(f"   Theory: {self.method}/{self.basis}")
        print(f"   Molecule: {' '.join(self.symbols)}")
        
        if 'model' in self.metadata:
            rmse = self.metadata['model'].get('test_rmse_kcal', 0)
            print(f"   Training RMSE: {rmse:.4f} kcal/mol")
    
    def predict_energy(self, coords):
        """Predict energy for geometry."""
        desc = compute_coulomb_matrix(self.symbols, coords)
        desc_scaled = self.scaler_X.transform([desc])
        e_scaled = self.model.predict(desc_scaled)
        energy = self.scaler_y.inverse_transform([[e_scaled[0]]])[0, 0]
        return energy
    
    def predict_forces(self, coords, delta=0.001):
        """Predict forces via finite differences."""
        n_atoms = len(self.symbols)
        forces = np.zeros((n_atoms, 3))
        
        for i in range(n_atoms):
            for j in range(3):
                # Forward
                coords_plus = coords.copy()
                coords_plus[i, j] += delta
                e_plus = self.predict_energy(coords_plus)
                
                # Backward
                coords_minus = coords.copy()
                coords_minus[i, j] -= delta
                e_minus = self.predict_energy(coords_minus)
                
                # Central difference
                forces[i, j] = -(e_plus - e_minus) / (2 * delta)
        
        return forces

# ==============================================================================
# PSI4 CALCULATOR
# ==============================================================================

def compute_psi4_energy_forces(symbols, coords, method, basis):
    """Compute PSI4 energy and forces with robust convergence settings."""
    
    # Create molecule string
    mol_str = "\n0 1\n"
    for s, c in zip(symbols, coords):
        mol_str += f"{s} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n"
    mol_str += "units bohr\nno_reorient\nno_com"
    
    # IMPORTANT: Clear all options to avoid conflicts
    psi4.core.clean_options()
    psi4.core.clean()
    
    # Set up PSI4 with robust convergence settings
    psi4.core.be_quiet()
    psi4.set_memory('2 GB')
    psi4.set_num_threads(4)
    
    mol = psi4.geometry(mol_str)
    
    # Use more robust SCF settings
    psi4.set_options({
        'basis': basis,
        'scf_type': 'df',
        'reference': 'rhf',
        'maxiter': 200,
        'damping_percentage': 20,
        'level_shift': 0.1,
        'e_convergence': 1e-8,
        'd_convergence': 1e-8
    })
    
    try:
        # Use properties() instead of energy() to get dipole
        energy, wfn = psi4.properties(
            method, 
            properties=['dipole'],  # Request dipole
            return_wfn=True, 
            molecule=mol
        )
        
        # Get dipole moment (in Debye)
        dipole = np.array(wfn.variable('SCF DIPOLE'))
        
        # Compute forces
        gradient = psi4.gradient(method, molecule=mol)
        forces = -np.array(gradient)
        
        return energy, forces, dipole  # Return dipole instead of None
            
    except psi4.driver.p4util.exceptions.SCFConvergenceError as e:
        # SCF didn't converge - try with more aggressive settings
        print(f"\n⚠️  SCF convergence issue, retrying with aggressive settings...")
        
        # IMPORTANT: Clear and reset
        psi4.core.clean_options()
        psi4.core.clean()
        psi4.core.be_quiet()
        psi4.set_memory('2 GB')
        psi4.set_num_threads(4)
        
        mol = psi4.geometry(mol_str)
        
        psi4.set_options({
            'basis': basis,
            'scf_type': 'df',
            'reference': 'rhf',
            'maxiter': 500,
            'damping_percentage': 50,
            'level_shift': 0.5,
            'soscf': 'true',
            'soscf_start_convergence': 1e-1,
            'e_convergence': 1e-6,  # Relaxed
            'd_convergence': 1e-6
        })
        
        try:
            energy, wfn = psi4.energy(method, molecule=mol, return_wfn=True)
            
            # Clear SOSCF for next call
            psi4.core.clean_options()
            psi4.core.clean()
            
            gradient = psi4.gradient(method, molecule=mol)
            forces = -np.array(gradient)
            
            print(f"   ✅ Converged with aggressive settings")
            return energy, forces
        
        except Exception as e2:
            # Still didn't converge
            print(f"   ❌ Could not converge - skipping")
            raise RuntimeError(f"SCF convergence failed: {str(e2)[:100]}")
    
    except Exception as e:
        print(f"\n❌ PSI4 error: {str(e)[:100]}")
        raise

# ==============================================================================
# VELOCITY VERLET INTEGRATOR
# ==============================================================================

def integrate_step(coords, velocities, forces, masses, dt):
    """Single velocity Verlet step."""
    
    # Update positions
    coords_new = coords + velocities * dt + 0.5 * forces / masses[:, np.newaxis] * dt**2
    
    # Half velocity update
    velocities_half = velocities + 0.5 * forces / masses[:, np.newaxis] * dt
    
    return coords_new, velocities_half

def finalize_velocities(velocities_half, forces, masses, dt):
    """Finalize velocity update."""
    velocities_new = velocities_half + 0.5 * forces / masses[:, np.newaxis] * dt
    return velocities_new

# ==============================================================================
# ON-THE-FLY MD
# ==============================================================================

def run_on_the_fly_md(
    ml_predictor: MLPESPredictor,
    initial_coords: np.ndarray,
    masses: np.ndarray,
    temperature: float = 300.0,
    n_steps: int = 1000,
    timestep: float = 0.5,  # fs
    validation_interval: int = 10,
    error_threshold_energy: float = 5.0,  # kcal/mol
    error_threshold_force: float = 10.0,  # kcal/mol/Å
    retrain_interval: int = 50,  # steps
    output_dir: Path = None
):
    """
    Run MD with ML-PES and validate against PSI4.
    
    Parameters:
    -----------
    ml_predictor : MLPESPredictor
        Loaded ML-PES model
    initial_coords : np.ndarray
        Starting coordinates (Bohr)
    masses : np.ndarray
        Atomic masses (amu)
    temperature : float
        Target temperature (K)
    n_steps : int
        Number of MD steps
    timestep : float
        Time step (fs)
    validation_interval : int
        Validate every N steps
    error_threshold_energy : float
        Add point to training if error > threshold (kcal/mol)
    error_threshold_force : float
        Add point to training if force error > threshold (kcal/mol/Å)
    retrain_interval : int
        Retrain model every N high-error points (0 = no retraining)
    output_dir : Path
        Output directory
    """
    
    if output_dir is None:
        output_dir = Path(f'outputs/on_the_fly_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("  STARTING ON-THE-FLY MD WITH VALIDATION")
    print("=" * 80)
    
    print(f"\n⚙️  Parameters:")
    print(f"   Steps: {n_steps}")
    print(f"   Timestep: {timestep} fs")
    print(f"   Temperature: {temperature} K")
    print(f"   Validation: every {validation_interval} steps")
    print(f"   Error threshold: {error_threshold_energy} kcal/mol (energy)")
    print(f"                   {error_threshold_force} kcal/mol/Å (force)")
    if retrain_interval > 0:
        print(f"   Retraining: every {retrain_interval} high-error points")
    else:
        print(f"   Retraining: at end only")
    
    # Convert units
    kb = 3.166811429e-6  # Hartree/K
    dt_au = timestep * 41.341374575751  # fs to au
    
    # Initialize velocities (Maxwell-Boltzmann)
    np.random.seed(42)
    velocities = np.random.randn(len(masses), 3)
    velocities *= np.sqrt(kb * temperature / masses[:, np.newaxis])
    
    # Remove COM motion
    velocities -= velocities.mean(axis=0)
    
    # Storage
    trajectory_coords = [initial_coords.copy()]
    trajectory_energies_ml = []
    trajectory_energies_psi4 = []
    trajectory_forces_ml = []
    trajectory_forces_psi4 = []
    
    validation_steps = []
    energy_errors = []
    force_errors = []
    
    high_error_coords = []
    high_error_energies = []
    high_error_forces = []
    
    coords = initial_coords.copy()
    
    print(f"\n🚀 Starting MD trajectory...")
    
    # Print time estimate
    time_per_validation = estimate_time_per_step(ml_predictor.method, ml_predictor.basis, len(ml_predictor.symbols))
    n_validations = n_steps // validation_interval
    estimated_time = (n_steps * 0.01) + (n_validations * time_per_validation)
    
    print(f"\n⏱️  Estimated timing:")
    print(f"   Per validation: ~{time_per_validation:.1f} seconds")
    print(f"   Total: ~{estimated_time/60:.1f} minutes")
    
    start_time = time.time()
    validation_times = []
    
    for step in tqdm(range(n_steps), desc="MD Steps"):
        
        # Check geometry sanity every 10 steps
        if step % 10 == 0 and step > 0:
            is_sane, reason = check_geometry_sanity(ml_predictor.symbols, coords)
            if not is_sane:
                tqdm.write(f"\n❌ Step {step}: Geometry problem - {reason}")
                tqdm.write(f"   Stopping MD - molecule may be dissociating or collapsing")
                tqdm.write(f"   Try: Lower temperature, shorter timestep, or better starting geometry")
                break
        
        # Predict forces with ML-PES
        forces_ml = ml_predictor.predict_forces(coords)
        energy_ml = ml_predictor.predict_energy(coords)
        
        # Integrate
        coords_new, velocities_half = integrate_step(coords, velocities, forces_ml, masses, dt_au)
        forces_ml_new = ml_predictor.predict_forces(coords_new)
        velocities = finalize_velocities(velocities_half, forces_ml_new, masses, dt_au)
        
        coords = coords_new
        trajectory_coords.append(coords.copy())
        
        # Validate against PSI4
        if step % validation_interval == 0:
            
            val_start = time.time()
            
            try:
                # Compute PSI4
                energy_psi4, forces_psi4 = compute_psi4_energy_forces(
                    ml_predictor.symbols, coords, 
                    ml_predictor.method, ml_predictor.basis
                )
                
                val_time = time.time() - val_start
                validation_times.append(val_time)
                
                # Store
                trajectory_energies_ml.append(energy_ml)
                trajectory_energies_psi4.append(energy_psi4)
                trajectory_forces_ml.append(forces_ml)
                trajectory_forces_psi4.append(forces_psi4)
                
                # Calculate errors
                e_error = abs(energy_ml - energy_psi4) * 627.509  # kcal/mol
                f_error = np.sqrt(((forces_ml - forces_psi4)**2).mean()) * 627.509 / 0.529177  # kcal/mol/Å
                
                validation_steps.append(step)
                energy_errors.append(e_error)
                force_errors.append(f_error)
                
                # Warn if errors are HUGE (suggests major problem)
                if e_error > 100.0:
                    tqdm.write(f"\n⚠️  Step {step}: Very large error ({e_error:.1f} kcal/mol)!")
                    tqdm.write(f"      This suggests ML-PES is far from training data")
                    tqdm.write(f"      Or molecule is in unusual configuration")
                
                # Check if high error
                if e_error > error_threshold_energy or f_error > error_threshold_force:
                    high_error_coords.append(coords.copy())
                    high_error_energies.append(energy_psi4)
                    high_error_forces.append(forces_psi4.copy())
                    
                    # Print warning (only if not huge)
                    if e_error <= 100.0:
                        tqdm.write(f"\n   ⚠️  Step {step}: High error detected!")
                        tqdm.write(f"      Energy error: {e_error:.3f} kcal/mol")
                        tqdm.write(f"      Force error: {f_error:.3f} kcal/mol/Å")
                        tqdm.write(f"      → Added to training data")
                    
                    # Retrain if interval reached
                    if retrain_interval > 0 and len(high_error_coords) % retrain_interval == 0:
                        tqdm.write(f"\n   🔄 Retraining with {len(high_error_coords)} new points...")
                        tqdm.write(f"      (Retraining will occur at end)")
            
            except RuntimeError as e:
                # PSI4 failed to converge
                tqdm.write(f"\n   ⚠️  Step {step}: PSI4 convergence failed - skipping validation")
                tqdm.write(f"      Continuing MD with ML-PES...")
            
            except Exception as e:
                # Unexpected error
                tqdm.write(f"\n   ❌ Step {step}: Unexpected error - {str(e)[:100]}")
                tqdm.write(f"      Continuing MD with ML-PES...")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ MD trajectory complete!")
    print(f"   Total steps: {n_steps}")
    print(f"   Validation attempts: {n_steps // validation_interval}")
    print(f"   Successful validations: {len(validation_steps)}")
    if len(validation_steps) < n_steps // validation_interval:
        failed = (n_steps // validation_interval) - len(validation_steps)
        print(f"   ⚠️  Failed validations (SCF issues): {failed}")
        
        # Helpful suggestions if many failures
        failure_rate = failed / (n_steps // validation_interval)
        if failure_rate > 0.3:  # More than 30% failed
            print(f"\n   💡 High failure rate ({failure_rate*100:.0f}%). Consider:")
            print(f"      - Lower temperature (current: {temperature} K)")
            print(f"      - Start from optimized geometry")
            print(f"      - Use different SCF initial guess")
            print(f"      - Try HF instead of DFT for problematic steps")
    print(f"   High-error points: {len(high_error_coords)}")
    
    print(f"\n⏱️  Timing statistics:")
    print(f"   Total time: {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)")
    print(f"   MD steps: {n_steps * (elapsed_time/n_steps):.1f} seconds total")
    if validation_times:
        mean_val_time = np.mean(validation_times)
        median_val_time = np.median(validation_times)
        max_val_time = np.max(validation_times)
        print(f"   Validations: {len(validation_times)} successful")
        print(f"      Mean: {mean_val_time:.1f} seconds/validation")
        print(f"      Median: {median_val_time:.1f} seconds/validation")
        print(f"      Max: {max_val_time:.1f} seconds (slowest)")
        print(f"      Total: {sum(validation_times)/60:.1f} minutes")
        
        # Prediction vs actual
        predicted_time = estimate_time_per_step(ml_predictor.method, ml_predictor.basis, len(ml_predictor.symbols))
        print(f"   Prediction: {predicted_time:.1f} sec/validation (actual: {mean_val_time:.1f} sec)")
    
    # Create results dictionary
    results = {
        'trajectory_coords': np.array(trajectory_coords),
        'validation_steps': validation_steps,
        'energies_ml': trajectory_energies_ml,
        'energies_psi4': trajectory_energies_psi4,
        'forces_ml': trajectory_forces_ml,
        'forces_psi4': trajectory_forces_psi4,
        'energy_errors': energy_errors,
        'force_errors': force_errors,
        'high_error_coords': np.array(high_error_coords) if high_error_coords else np.array([]),
        'high_error_energies': np.array(high_error_energies) if high_error_energies else np.array([]),
        'high_error_forces': np.array(high_error_forces) if high_error_forces else np.array([]),
        'parameters': {
            'n_steps': n_steps,
            'timestep': timestep,
            'temperature': temperature,
            'validation_interval': validation_interval,
            'error_threshold_energy': error_threshold_energy,
            'error_threshold_force': error_threshold_force,
            'symbols': ml_predictor.symbols,
            'method': ml_predictor.method,
            'basis': ml_predictor.basis
        }
    }
    
    return results, output_dir

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_validation_results(results, output_dir):
    """Plot comprehensive validation results."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    steps = results['validation_steps']
    time_fs = np.array(steps) * results['parameters']['timestep']
    
    # 1. Energy comparison
    ax1 = fig.add_subplot(gs[0, 0])
    e_ml = np.array(results['energies_ml']) * 627.509
    e_psi4 = np.array(results['energies_psi4']) * 627.509
    
    ax1.plot(time_fs, e_ml - e_ml[0], 'b-', label='ML-PES', linewidth=2)
    ax1.plot(time_fs, e_psi4 - e_psi4[0], 'r--', label='PSI4', linewidth=2)
    ax1.set_xlabel('Time (fs)', fontsize=12)
    ax1.set_ylabel('Relative Energy (kcal/mol)', fontsize=12)
    ax1.set_title('Energy Comparison', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Energy error over time
    ax2 = fig.add_subplot(gs[0, 1])
    e_errors = np.array(results['energy_errors'])
    
    ax2.plot(time_fs, e_errors, 'g-', linewidth=2)
    ax2.axhline(results['parameters']['error_threshold_energy'], 
                color='red', linestyle='--', label='Threshold', linewidth=2)
    ax2.set_xlabel('Time (fs)', fontsize=12)
    ax2.set_ylabel('Energy Error (kcal/mol)', fontsize=12)
    ax2.set_title(f'Energy Error (Mean: {e_errors.mean():.3f} kcal/mol)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Force error over time
    ax3 = fig.add_subplot(gs[1, 0])
    f_errors = np.array(results['force_errors'])
    
    ax3.plot(time_fs, f_errors, 'purple', linewidth=2)
    ax3.axhline(results['parameters']['error_threshold_force'],
                color='red', linestyle='--', label='Threshold', linewidth=2)
    ax3.set_xlabel('Time (fs)', fontsize=12)
    ax3.set_ylabel('Force RMSE (kcal/mol/Å)', fontsize=12)
    ax3.set_title(f'Force Error (Mean: {f_errors.mean():.3f} kcal/mol/Å)', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Error distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(e_errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(e_errors.mean(), color='blue', linestyle='--', 
                label=f'Mean: {e_errors.mean():.3f}', linewidth=2)
    ax4.axvline(results['parameters']['error_threshold_energy'], 
                color='red', linestyle='--', label='Threshold', linewidth=2)
    ax4.set_xlabel('Energy Error (kcal/mol)', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Error Distribution', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Energy parity plot
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.scatter(e_psi4, e_ml, alpha=0.6, s=50)
    
    min_e = min(e_psi4.min(), e_ml.min())
    max_e = max(e_psi4.max(), e_ml.max())
    ax5.plot([min_e, max_e], [min_e, max_e], 'r--', linewidth=2, label='Perfect')
    
    ax5.set_xlabel('PSI4 Energy (kcal/mol)', fontsize=12)
    ax5.set_ylabel('ML-PES Energy (kcal/mol)', fontsize=12)
    ax5.set_title('Energy Parity Plot', fontsize=14)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Force parity plot
    ax6 = fig.add_subplot(gs[2, 1])
    forces_ml_flat = np.concatenate(results['forces_ml']).flatten() * 627.509 / 0.529177
    forces_psi4_flat = np.concatenate(results['forces_psi4']).flatten() * 627.509 / 0.529177
    
    ax6.scatter(forces_psi4_flat, forces_ml_flat, alpha=0.3, s=10)
    
    min_f = min(forces_psi4_flat.min(), forces_ml_flat.min())
    max_f = max(forces_psi4_flat.max(), forces_ml_flat.max())
    ax6.plot([min_f, max_f], [min_f, max_f], 'r--', linewidth=2, label='Perfect')
    
    ax6.set_xlabel('PSI4 Force (kcal/mol/Å)', fontsize=12)
    ax6.set_ylabel('ML-PES Force (kcal/mol/Å)', fontsize=12)
    ax6.set_title('Force Parity Plot', fontsize=14)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'On-the-Fly Validation Results\n'
                 f'{results["parameters"]["method"]}/{results["parameters"]["basis"]} | '
                 f'{len(results["high_error_coords"])} high-error points',
                 fontsize=16, fontweight='bold')
    
    plot_path = output_dir / 'validation_results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Saved validation plot: {plot_path}")
    
    return str(plot_path)

# ==============================================================================
# RETRAIN WITH AUGMENTED DATA
# ==============================================================================

def retrain_with_augmented_data(
    original_training_data_path: str,
    high_error_coords: np.ndarray,
    high_error_energies: np.ndarray,
    high_error_forces: np.ndarray,
    symbols: list,
    output_dir: Path
):
    """Retrain ML-PES with augmented training data."""
    
    print("\n" + "=" * 80)
    print("  RETRAINING WITH AUGMENTED DATA")
    print("=" * 80)
    
    # Load original data
    print(f"\n📂 Loading original training data...")
    original_data = load_trajectory(original_training_data_path)
    
    print(f"   Original: {original_data.n_frames} configurations")
    print(f"   New: {len(high_error_coords)} high-error points")
    
    # Combine data
    augmented_coords = np.vstack([original_data.coordinates, high_error_coords])
    augmented_energies = np.concatenate([original_data.energies, high_error_energies])
    
    if original_data.forces is not None:
        augmented_forces = np.vstack([original_data.forces, high_error_forces])
    else:
        augmented_forces = None
    
    print(f"   Augmented: {len(augmented_coords)} configurations")
    
    # Train new model
    print(f"\n🤖 Training new model...")
    
    # Compute descriptors
    X = []
    for coords in tqdm(augmented_coords, desc="Descriptors"):
        desc = compute_coulomb_matrix(symbols, coords)
        X.append(desc)
    X = np.array(X)
    y = augmented_energies
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Hyperparameter search
    param_grid = list(product([0.001, 0.01, 0.1, 1.0], [0.01, 0.1, 1.0]))
    
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
    
    print(f"\n✅ Retrained model:")
    print(f"   Gamma: {best_params[0]}")
    print(f"   Alpha: {best_params[1]}")
    print(f"   RMSE: {best_rmse:.4f} kcal/mol")
    print(f"   R²: {r2:.6f}")
    
    # Save augmented data
    augmented_traj = TrajectoryData(
        symbols=symbols,
        coordinates=augmented_coords,
        energies=augmented_energies,
        forces=augmented_forces,
        metadata={'n_original': original_data.n_frames, 
                  'n_augmented': len(high_error_coords)}
    )
    
    augmented_data_path = output_dir / 'augmented_training_data.npz'
    save_trajectory(augmented_traj, str(augmented_data_path))
    print(f"\n💾 Saved augmented data: {augmented_data_path}")
    
    # Save retrained model
    model_data = {
        'model': best_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'symbols': symbols,
        'metadata': {
            'training': {
                'n_original': original_data.n_frames,
                'n_augmented': len(high_error_coords),
                'n_total': len(augmented_coords)
            },
            'model': {
                'gamma': float(best_params[0]),
                'alpha': float(best_params[1]),
                'test_rmse_kcal': float(best_rmse),
                'r2_score': float(r2)
            }
        },
        'version': '2.2'
    }
    
    retrained_model_path = output_dir / 'retrained_mlpes_model.pkl'
    with open(retrained_model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"💾 Saved retrained model: {retrained_model_path}")
    
    return best_rmse, r2

# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

def main():
    """Main on-the-fly validation workflow."""
    
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='On-the-fly ML-PES validation and training')
    parser.add_argument('--model', required=True, help='Path to ML-PES model (.pkl)')
    parser.add_argument('--training-data', help='Path to original training data (.npz)')
    parser.add_argument('--steps', type=int, default=500, help='Number of MD steps')
    parser.add_argument('--temp', type=float, default=300.0, help='Temperature (K)')
    parser.add_argument('--timestep', type=float, default=0.5, help='Timestep (fs)')
    parser.add_argument('--validate-every', type=int, default=10, 
                       help='Validate every N steps')
    parser.add_argument('--error-threshold-e', type=float, default=5.0,
                       help='Energy error threshold (kcal/mol)')
    parser.add_argument('--error-threshold-f', type=float, default=10.0,
                       help='Force error threshold (kcal/mol/Å)')
    parser.add_argument('--retrain', action='store_true',
                       help='Retrain model with augmented data at end')
    
    args = parser.parse_args()
    
    # Load model
    predictor = MLPESPredictor(args.model)
    
    # Get initial coordinates
    initial_coords = None
    
    # Strategy 1: Try to load from molecule library
    mol_name = predictor.metadata.get('molecule', {}).get('name', 'unknown')
    
    if mol_name != 'unknown':
        print(f"\n📊 Trying to load reference geometry for: {mol_name}")
        try:
            molecule = get_molecule(mol_name)
            initial_coords = molecule.coordinates.copy()
            print(f"   ✅ Loaded from molecule library")
        except:
            print(f"   ⚠️  Not found in library")
    
    # Strategy 2: Extract from training data
    if initial_coords is None:
        if args.training_data is None:
            print(f"\n❌ Cannot determine initial coordinates!")
            print(f"   Molecule '{mol_name}' not in library")
            print(f"   Need --training-data to extract coordinates")
            sys.exit(1)
        
        print(f"\n📊 Extracting initial coordinates from training data...")
        try:
            training_traj = load_trajectory(args.training_data)
            
            # Use the configuration with lowest energy (most stable)
            energies = training_traj.energies
            lowest_idx = np.argmin(energies)
            
            initial_coords = training_traj.coordinates[lowest_idx].copy()
            
            print(f"   ✅ Using configuration {lowest_idx} (lowest energy)")
            print(f"      Energy: {energies[lowest_idx]*627.509:.2f} kcal/mol (most stable)")
        except Exception as e:
            print(f"   ❌ Error loading training data: {e}")
            sys.exit(1)
    
    # Strategy 3: Quick geometry optimization with ML-PES
    print(f"\n🔧 Optimizing starting geometry with ML-PES...")
    print(f"   (Ensures SCF will converge at step 0)")
    
    # Simple steepest descent optimization
    coords_opt = initial_coords.copy()
    learning_rate = 0.01  # Bohr
    
    for opt_step in range(50):
        forces = predictor.predict_forces(coords_opt)
        
        # Update coordinates
        coords_opt += learning_rate * forces
        
        # Check convergence
        force_norm = np.linalg.norm(forces)
        if force_norm < 0.001:  # Ha/Bohr
            print(f"   ✅ Converged in {opt_step+1} steps (force norm: {force_norm:.6f})")
            break
        
        if opt_step % 10 == 0:
            energy = predictor.predict_energy(coords_opt)
            print(f"      Step {opt_step}: E = {energy*627.509:.2f} kcal/mol, |F| = {force_norm:.6f}")
    
    initial_coords = coords_opt
    
    # Verify with PSI4 that this geometry converges
    print(f"\n🔍 Verifying starting geometry with PSI4...")
    try:
        test_energy, test_forces = compute_psi4_energy_forces(
            predictor.symbols, initial_coords,
            predictor.method, predictor.basis
        )
        print(f"   ✅ SCF converged for starting geometry")
        print(f"      E_PSI4 = {test_energy*627.509:.2f} kcal/mol")
    except Exception as e:
        print(f"   ❌ Starting geometry still problematic: {e}")
        print(f"   💡 Try using lower temperature or different starting point")
        sys.exit(1)
    
    # Get masses
    mass_map = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998}
    masses = np.array([mass_map[s] for s in predictor.symbols])
    
    # Run on-the-fly MD
    results, output_dir = run_on_the_fly_md(
        ml_predictor=predictor,
        initial_coords=initial_coords,
        masses=masses,
        temperature=args.temp,
        n_steps=args.steps,
        timestep=args.timestep,
        validation_interval=args.validate_every,
        error_threshold_energy=args.error_threshold_e,
        error_threshold_force=args.error_threshold_f,
        retrain_interval=0  # Retrain at end only
    )
    
    # Save results
    results_path = output_dir / 'validation_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n💾 Saved results: {results_path}")
    
    # Visualize
    plot_validation_results(results, output_dir)
    
    # Statistics
    print("\n" + "=" * 80)
    print("  VALIDATION STATISTICS")
    print("=" * 80)
    
    e_errors = np.array(results['energy_errors'])
    f_errors = np.array(results['force_errors'])
    
    print(f"\n📊 Energy Errors:")
    print(f"   Mean: {e_errors.mean():.4f} kcal/mol")
    print(f"   Std: {e_errors.std():.4f} kcal/mol")
    print(f"   Max: {e_errors.max():.4f} kcal/mol")
    print(f"   > Threshold: {(e_errors > args.error_threshold_e).sum()} points")
    
    print(f"\n📊 Force Errors:")
    print(f"   Mean: {f_errors.mean():.4f} kcal/mol/Å")
    print(f"   Std: {f_errors.std():.4f} kcal/mol/Å")
    print(f"   Max: {f_errors.max():.4f} kcal/mol/Å")
    print(f"   > Threshold: {(f_errors > args.error_threshold_f).sum()} points")
    
    print(f"\n📊 High-Error Points:")
    print(f"   Total: {len(results['high_error_coords'])} configurations")
    print(f"   Percentage: {len(results['high_error_coords'])/len(e_errors)*100:.1f}%")
    
    # Retrain if requested
    if args.retrain and len(results['high_error_coords']) > 0:
        
        if args.training_data is None:
            print("\n⚠️  --training-data required for retraining")
        else:
            rmse_new, r2_new = retrain_with_augmented_data(
                original_training_data_path=args.training_data,
                high_error_coords=results['high_error_coords'],
                high_error_energies=results['high_error_energies'],
                high_error_forces=results['high_error_forces'],
                symbols=predictor.symbols,
                output_dir=output_dir
            )
            
            # Compare
            rmse_old = predictor.metadata.get('model', {}).get('test_rmse_kcal', 0)
            
            print(f"\n📊 Model Comparison:")
            print(f"   Original RMSE: {rmse_old:.4f} kcal/mol")
            print(f"   Retrained RMSE: {rmse_new:.4f} kcal/mol")
            
            if rmse_new < rmse_old:
                improvement = (rmse_old - rmse_new) / rmse_old * 100
                print(f"   ✅ Improvement: {improvement:.1f}%")
            else:
                print(f"   ⚠️  No improvement")
    
    elif args.retrain:
        print(f"\n✅ No high-error points found - model is accurate!")
    
    print("\n" + "=" * 80)
    print("✅ ON-THE-FLY VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\n📂 Output directory: {output_dir}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
