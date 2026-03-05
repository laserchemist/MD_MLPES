#!/usr/bin/env python3
"""
Advanced ML-PES: Adaptive Sampling & Multi-Level Training

Features:
1. Proper metadata storage (method, basis, theory level)
2. Adaptive sampling based on high gradients
3. Multi-level training (critical points with higher theory)
4. Active learning workflow

Author: PSI4-MD Framework
Date: 2025
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
from typing import Dict, List, Tuple, Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("=" * 80)
print("  ADVANCED ML-PES: ADAPTIVE SAMPLING & MULTI-LEVEL")
print("=" * 80)

# Import framework
try:
    from modules.test_molecules import get_molecule, add_random_displacement
    from modules.direct_md import DirectMDConfig, run_direct_md
    from modules.data_formats import TrajectoryData, save_trajectory, load_trajectory
    print("✅ Framework imported\n")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

try:
    import psi4
    print(f"✅ PSI4 {psi4.__version__}\n")
    PSI4_AVAILABLE = True
except ImportError:
    print("❌ PSI4 required for this workflow\n")
    PSI4_AVAILABLE = False

# ==============================================================================
# METADATA MANAGEMENT
# ==============================================================================

class MLPESMetadata:
    """Store complete metadata about ML-PES training."""
    
    def __init__(self):
        self.theory_level = {
            'method': None,
            'basis': None,
            'reference': None  # RHF, UHF, ROHF
        }
        self.training_info = {
            'n_configs': 0,
            'n_atoms': 0,
            'molecule': None,
            'energy_range_kcal': 0.0,
            'training_type': 'energy_only',  # or 'energy_forces'
            'date_trained': None
        }
        self.performance = {
            'train_rmse_kcal': 0.0,
            'test_rmse_kcal': 0.0,
            'r2_score': 0.0
        }
        self.adaptive_sampling = {
            'used': False,
            'high_gradient_points': 0,
            'multi_level_points': {}  # {method: count}
        }
        self.model_params = {
            'gamma': 0.0,
            'alpha': 0.0,
            'descriptor': 'coulomb_matrix'
        }
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'theory_level': self.theory_level,
            'training_info': self.training_info,
            'performance': self.performance,
            'adaptive_sampling': self.adaptive_sampling,
            'model_params': self.model_params
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary."""
        metadata = cls()
        metadata.theory_level = data.get('theory_level', {})
        metadata.training_info = data.get('training_info', {})
        metadata.performance = data.get('performance', {})
        metadata.adaptive_sampling = data.get('adaptive_sampling', {})
        metadata.model_params = data.get('model_params', {})
        return metadata

def save_model_with_metadata(model_data, metadata: MLPESMetadata, filepath: str):
    """Save model with complete metadata."""
    complete_data = {
        'model': model_data['model'],
        'scaler_X': model_data['scaler_X'],
        'scaler_y': model_data['scaler_y'],
        'symbols': model_data['symbols'],
        'metadata': metadata.to_dict(),
        'version': '2.0'  # Version with metadata
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(complete_data, f)
    
    print(f"\n💾 Model saved with metadata:")
    print(f"   Theory: {metadata.theory_level['method']}/{metadata.theory_level['basis']}")
    print(f"   RMSE: {metadata.performance['test_rmse_kcal']:.4f} kcal/mol")
    print(f"   Path: {filepath}")

def load_model_with_metadata(filepath: str):
    """Load model and check metadata."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Check version
    if 'version' not in data:
        print("⚠️  Old model format (no metadata)")
        metadata = None
    else:
        metadata = MLPESMetadata.from_dict(data['metadata'])
    
    return data, metadata

# ==============================================================================
# COULOMB MATRIX DESCRIPTOR
# ==============================================================================

def compute_coulomb_matrix(symbols, coords):
    """Coulomb matrix descriptor."""
    atomic_numbers = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
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
# ADAPTIVE SAMPLING: IDENTIFY HIGH-GRADIENT REGIONS
# ==============================================================================

def identify_high_gradient_regions(trajectory: TrajectoryData, 
                                   percentile: float = 90.0) -> np.ndarray:
    """
    Identify configurations with high gradients (steep energy changes).
    
    Args:
        trajectory: Trajectory with forces
        percentile: Keep configurations above this percentile of force magnitude
    
    Returns:
        Indices of high-gradient configurations
    """
    if trajectory.forces is None:
        raise ValueError("Trajectory must contain forces for gradient analysis")
    
    # Compute force magnitudes for each configuration
    force_magnitudes = []
    for i in range(trajectory.n_frames):
        # Total force magnitude = sqrt(sum of all force components squared)
        forces_flat = trajectory.forces[i].flatten()
        magnitude = np.sqrt((forces_flat**2).sum())
        force_magnitudes.append(magnitude)
    
    force_magnitudes = np.array(force_magnitudes)
    
    # Find threshold
    threshold = np.percentile(force_magnitudes, percentile)
    
    # Identify high-gradient points
    high_gradient_idx = np.where(force_magnitudes >= threshold)[0]
    
    print(f"\n📊 High-Gradient Analysis:")
    print(f"   Force magnitude range: {force_magnitudes.min():.4f} to {force_magnitudes.max():.4f} Ha/Bohr")
    print(f"   Threshold ({percentile}th percentile): {threshold:.4f} Ha/Bohr")
    print(f"   High-gradient points: {len(high_gradient_idx)} / {trajectory.n_frames}")
    print(f"   Percentage: {100*len(high_gradient_idx)/trajectory.n_frames:.1f}%")
    
    return high_gradient_idx

def sample_additional_points_near_high_gradients(trajectory: TrajectoryData,
                                                 high_gradient_idx: np.ndarray,
                                                 n_samples_per_point: int = 3,
                                                 displacement_std: float = 0.05) -> TrajectoryData:
    """
    Generate additional sampling points near high-gradient regions.
    
    Args:
        trajectory: Original trajectory
        high_gradient_idx: Indices of high-gradient points
        n_samples_per_point: Number of displaced samples per high-gradient point
        displacement_std: Standard deviation of displacement (Angstrom)
    
    Returns:
        New trajectory with additional sampling points (NO energies/forces yet!)
    """
    print(f"\n🎯 Generating {n_samples_per_point} samples near each high-gradient point...")
    
    new_coords = []
    
    for idx in tqdm(high_gradient_idx, desc="Sampling"):
        base_coords = trajectory.coordinates[idx]
        
        for _ in range(n_samples_per_point):
            # Add small random displacement
            displacement = np.random.normal(0, displacement_std, size=base_coords.shape)
            new_coord = base_coords + displacement
            new_coords.append(new_coord)
    
    new_coords = np.array(new_coords)
    
    # Create trajectory structure (energies/forces will be computed separately)
    new_trajectory = TrajectoryData(
        symbols=trajectory.symbols,
        coordinates=new_coords,
        energies=np.zeros(len(new_coords)),  # Placeholder
        forces=np.zeros((len(new_coords), len(trajectory.symbols), 3))  # Placeholder
    )
    
    print(f"   ✅ Generated {len(new_coords)} additional sampling points")
    
    return new_trajectory

# ==============================================================================
# MULTI-LEVEL CALCULATIONS
# ==============================================================================

def compute_single_point_energy_force(molecule_obj, method: str, basis: str):
    """
    Compute single-point energy and gradient with PSI4.
    
    Returns:
        (energy_ha, forces_ha_per_bohr)
    """
    if not PSI4_AVAILABLE:
        raise RuntimeError("PSI4 required for calculations")
    
    # Set up calculation
    psi4.set_memory('2GB')
    psi4.core.be_quiet()
    psi4.set_options({
        'reference': 'rhf',
        'scf_type': 'df',
        'd_convergence': 1e-8,
        'e_convergence': 1e-8
    })
    
    # Energy
    energy = psi4.energy(f'{method}/{basis}', molecule=molecule_obj)
    
    # Gradient
    gradient = psi4.gradient(f'{method}/{basis}', molecule=molecule_obj)
    forces = -np.array(gradient)  # Force = -gradient
    
    return energy, forces

def compute_multi_level_points(coordinates: np.ndarray,
                               symbols: List[str],
                               methods: List[Tuple[str, str]],
                               description: str = "Multi-level") -> Dict:
    """
    Compute energies/forces at multiple theory levels for given coordinates.
    
    Args:
        coordinates: (N_points, N_atoms, 3) array
        symbols: Atomic symbols
        methods: List of (method, basis) tuples, e.g., [('B3LYP', '6-31G*'), ('CCSD(T)', 'cc-pVTZ')]
        description: Description for progress bar
    
    Returns:
        Dictionary with results for each method
    """
    print(f"\n🔬 Computing {len(coordinates)} points at {len(methods)} theory levels...")
    
    results = {f"{method}/{basis}": {'energies': [], 'forces': []} 
               for method, basis in methods}
    
    # Create PSI4 molecule template
    mol_string = f"{0} 1\n"  # Charge, multiplicity
    for symbol in symbols:
        mol_string += f"{symbol} 0.0 0.0 0.0\n"
    mol_string += "units angstrom\nno_reorient\nno_com"
    
    for i in tqdm(range(len(coordinates)), desc=description):
        coords = coordinates[i]
        
        # Update geometry
        geometry_block = ""
        for j, symbol in enumerate(symbols):
            x, y, z = coords[j]
            geometry_block += f"{symbol} {x:.10f} {y:.10f} {z:.10f}\n"
        
        full_mol_string = f"0 1\n{geometry_block}units angstrom\nno_reorient\nno_com"
        
        # Compute at each theory level
        for method, basis in methods:
            key = f"{method}/{basis}"
            
            try:
                psi4_mol = psi4.geometry(full_mol_string)
                energy, forces = compute_single_point_energy_force(psi4_mol, method, basis)
                
                results[key]['energies'].append(energy)
                results[key]['forces'].append(forces)
                
            except Exception as e:
                print(f"\n⚠️  Failed for {key} at point {i}: {e}")
                # Use NaN for failed calculations
                results[key]['energies'].append(np.nan)
                results[key]['forces'].append(np.full((len(symbols), 3), np.nan))
    
    # Convert to arrays
    for key in results:
        results[key]['energies'] = np.array(results[key]['energies'])
        results[key]['forces'] = np.array(results[key]['forces'])
    
    return results

# ==============================================================================
# ADAPTIVE WORKFLOW
# ==============================================================================

def adaptive_sampling_workflow(base_trajectory: TrajectoryData,
                               base_method: str,
                               base_basis: str,
                               high_level_method: str = None,
                               high_level_basis: str = None,
                               gradient_percentile: float = 90.0,
                               n_adaptive_samples: int = 3):
    """
    Complete adaptive sampling workflow.
    
    Steps:
    1. Identify high-gradient regions from base trajectory
    2. Sample additional points near these regions
    3. Optionally compute these points at higher theory level
    4. Combine with base data
    
    Args:
        base_trajectory: Initial trajectory with energies/forces
        base_method: Method used for base trajectory (e.g., 'B3LYP')
        base_basis: Basis used for base trajectory (e.g., '6-31G*')
        high_level_method: Optional higher-level method (e.g., 'CCSD(T)')
        high_level_basis: Optional higher-level basis (e.g., 'cc-pVTZ')
        gradient_percentile: Percentile for high-gradient cutoff
        n_adaptive_samples: Samples per high-gradient point
    
    Returns:
        Enhanced trajectory, metadata
    """
    print("=" * 80)
    print("  ADAPTIVE SAMPLING WORKFLOW")
    print("=" * 80)
    
    # Step 1: Identify high-gradient regions
    high_grad_idx = identify_high_gradient_regions(base_trajectory, gradient_percentile)
    
    # Step 2: Generate additional sampling points
    new_points = sample_additional_points_near_high_gradients(
        base_trajectory, high_grad_idx, n_adaptive_samples
    )
    
    # Step 3: Compute energies/forces for new points
    if high_level_method and high_level_basis and PSI4_AVAILABLE:
        print(f"\n🎯 Computing new points at HIGH level: {high_level_method}/{high_level_basis}")
        
        methods = [(high_level_method, high_level_basis)]
        results = compute_multi_level_points(
            new_points.coordinates,
            new_points.symbols,
            methods,
            description="High-level calculations"
        )
        
        key = f"{high_level_method}/{high_level_basis}"
        new_points.energies = results[key]['energies']
        new_points.forces = results[key]['forces']
        
        # Remove any failed calculations (NaN)
        valid_mask = ~np.isnan(new_points.energies)
        new_points.coordinates = new_points.coordinates[valid_mask]
        new_points.energies = new_points.energies[valid_mask]
        new_points.forces = new_points.forces[valid_mask]
        
        print(f"   ✅ Computed {len(new_points.energies)} high-level points")
        
    elif PSI4_AVAILABLE:
        print(f"\n🔬 Computing new points at BASE level: {base_method}/{base_basis}")
        
        methods = [(base_method, base_basis)]
        results = compute_multi_level_points(
            new_points.coordinates,
            new_points.symbols,
            methods,
            description="Base-level calculations"
        )
        
        key = f"{base_method}/{base_basis}"
        new_points.energies = results[key]['energies']
        new_points.forces = results[key]['forces']
        
        valid_mask = ~np.isnan(new_points.energies)
        new_points.coordinates = new_points.coordinates[valid_mask]
        new_points.energies = new_points.energies[valid_mask]
        new_points.forces = new_points.forces[valid_mask]
        
        print(f"   ✅ Computed {len(new_points.energies)} base-level points")
    
    else:
        print("\n⚠️  PSI4 not available - cannot compute new points")
        return base_trajectory, None
    
    # Step 4: Combine trajectories
    print(f"\n📦 Combining trajectories...")
    
    combined_coords = np.vstack([base_trajectory.coordinates, new_points.coordinates])
    combined_energies = np.concatenate([base_trajectory.energies, new_points.energies])
    combined_forces = np.vstack([base_trajectory.forces, new_points.forces])
    
    enhanced_trajectory = TrajectoryData(
        symbols=base_trajectory.symbols,
        coordinates=combined_coords,
        energies=combined_energies,
        forces=combined_forces,
        metadata={
            'base_method': base_method,
            'base_basis': base_basis,
            'high_level_method': high_level_method,
            'high_level_basis': high_level_basis,
            'base_points': len(base_trajectory.energies),
            'adaptive_points': len(new_points.energies),
            'total_points': len(combined_energies)
        }
    )
    
    print(f"   Base trajectory: {len(base_trajectory.energies)} points")
    print(f"   Adaptive points: {len(new_points.energies)} points")
    print(f"   Total: {len(combined_energies)} points")
    
    # Create metadata
    metadata = MLPESMetadata()
    metadata.theory_level['method'] = base_method
    metadata.theory_level['basis'] = base_basis
    metadata.training_info['n_configs'] = len(combined_energies)
    metadata.adaptive_sampling['used'] = True
    metadata.adaptive_sampling['high_gradient_points'] = len(high_grad_idx)
    if high_level_method:
        metadata.adaptive_sampling['multi_level_points'] = {
            f"{high_level_method}/{high_level_basis}": len(new_points.energies)
        }
    
    return enhanced_trajectory, metadata

# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

def main():
    """Interactive adaptive sampling workflow."""
    
    print("\n" + "=" * 80)
    print("  STEP 1: SELECT BASE TRAINING DATA")
    print("=" * 80)
    
    # Find existing data
    import glob
    found = glob.glob('outputs/*/training_data/*.npz')
    
    if not found:
        print("\n❌ No training data found!")
        print("   Generate base training data first")
        sys.exit(1)
    
    found = sorted(found, key=lambda x: Path(x).stat().st_mtime, reverse=True)
    
    print(f"\n📂 Found {len(found)} dataset(s):\n")
    
    for i, path in enumerate(found[:5], 1):
        p = Path(path)
        traj = load_trajectory(path)
        e_range = (traj.energies.max() - traj.energies.min()) * 627.509
        print(f"  [{i}] {path}")
        print(f"      Configs: {traj.n_frames}, Energy range: {e_range:.2f} kcal/mol")
        print()
    
    choice = int(input(f"Select dataset [1-{min(5, len(found))}]: "))
    base_data_path = found[choice - 1]
    
    print(f"\n✅ Loading: {base_data_path}")
    base_trajectory = load_trajectory(base_data_path)
    
    if base_trajectory.forces is None:
        print("\n❌ Trajectory must contain forces for adaptive sampling!")
        sys.exit(1)
    
    # Get theory level
    print(f"\n" + "=" * 80)
    print("  STEP 2: THEORY LEVEL CONFIGURATION")
    print("=" * 80)
    
    print(f"\nBase trajectory theory level:")
    base_method = input("  Method (e.g., B3LYP, HF): ").strip() or 'B3LYP'
    base_basis = input("  Basis (e.g., 6-31G*, cc-pVDZ): ").strip() or '6-31G*'
    
    print(f"\nAdaptive sampling options:")
    print("  [1] Same theory level (faster)")
    print("  [2] Higher theory level for critical points (more accurate)")
    
    mode = input("\nSelect [1-2]: ").strip()
    
    if mode == '2':
        print(f"\nHigher theory level for adaptive points:")
        high_method = input("  Method (e.g., CCSD(T), MP2): ").strip()
        high_basis = input("  Basis (e.g., cc-pVTZ, aug-cc-pVDZ): ").strip()
    else:
        high_method = None
        high_basis = None
    
    # Adaptive sampling parameters
    print(f"\n" + "=" * 80)
    print("  STEP 3: ADAPTIVE SAMPLING PARAMETERS")
    print("=" * 80)
    
    percentile = float(input("\nGradient percentile threshold [90]: ").strip() or "90")
    n_samples = int(input("Samples per high-gradient point [3]: ").strip() or "3")
    
    # Run adaptive workflow
    enhanced_traj, metadata = adaptive_sampling_workflow(
        base_trajectory,
        base_method,
        base_basis,
        high_method,
        high_basis,
        percentile,
        n_samples
    )
    
    # Save enhanced training data
    output_dir = Path(f'outputs/adaptive_mlpes_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    enhanced_data_path = output_dir / 'enhanced_training_data.npz'
    save_trajectory(enhanced_traj, str(enhanced_data_path))
    
    print(f"\n✅ Enhanced training data saved: {enhanced_data_path}")
    
    # Train ML-PES
    print(f"\n" + "=" * 80)
    print("  STEP 4: TRAIN ML-PES ON ENHANCED DATA")
    print("=" * 80)
    
    train_choice = input("\nTrain ML-PES now? [y/n]: ").strip().lower()
    
    if train_choice == 'y':
        # Simple training (can be replaced with full workflow)
        from itertools import product
        
        X = []
        for i in tqdm(range(enhanced_traj.n_frames), desc="Descriptors"):
            desc = compute_coulomb_matrix(enhanced_traj.symbols, enhanced_traj.coordinates[i])
            X.append(desc)
        X = np.array(X)
        y = enhanced_traj.energies
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        print("\n🤖 Training models...")
        
        best_rmse = float('inf')
        best_model = None
        best_params = None
        
        for gamma in tqdm([0.01, 0.1, 1.0], desc="Hyperparameters"):
            for alpha in [0.01, 0.1]:
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
        
        print(f"\n✅ Best model: gamma={best_params[0]}, alpha={best_params[1]}")
        print(f"   RMSE: {best_rmse:.4f} kcal/mol")
        
        # Update metadata
        metadata.performance['test_rmse_kcal'] = best_rmse
        metadata.model_params['gamma'] = best_params[0]
        metadata.model_params['alpha'] = best_params[1]
        metadata.training_info['date_trained'] = datetime.now().isoformat()
        
        # Save model with metadata
        model_data = {
            'model': best_model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'symbols': enhanced_traj.symbols
        }
        
        model_path = output_dir / 'adaptive_mlpes_model.pkl'
        save_model_with_metadata(model_data, metadata, str(model_path))
    
    print(f"\n🎉 Adaptive sampling workflow complete!")
    print(f"   Output directory: {output_dir}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
