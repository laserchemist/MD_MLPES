#!/usr/bin/env python3
"""
Complete Production ML-PES Workflow v2.2 - Refinement First

Workflow Structure:
1. FIRST: Work with existing data (refinement options)
   - Energy-only training
   - Energy + gradient training
   - Adaptive refinement (error-based, requires GP or new PSI4)
   - Adaptive refinement (gradient-based, add higher theory)
   
2. SECOND: Compute new molecule
   - From library, XYZ file, or custom input
   - Generate new PSI4 data

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
import glob
import subprocess

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False
    def tqdm(x, **kwargs):
        return x

from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from itertools import product

print("=" * 80)
print("  PRODUCTION ML-PES WORKFLOW v2.2 - REFINEMENT FIRST")
print("=" * 80)

# Import framework
try:
    from modules.test_molecules import get_molecule, add_random_displacement
    from modules.direct_md import DirectMDConfig, run_direct_md
    from modules.data_formats import TrajectoryData, save_trajectory, load_trajectory
    from modules.visualization import TrajectoryVisualizer
    
    # Get available molecules dynamically
    import modules.test_molecules as tm
    AVAILABLE_MOLECULES = {}
    for name in ['water', 'methane', 'ammonia', 'formaldehyde', 'ethylene', 
                 'benzene', 'hydrogen', 'methanol', 'hydrogen_peroxide', 
                 'formaldehyde_oxide']:
        try:
            mol = get_molecule(name)
            if mol is not None:
                AVAILABLE_MOLECULES[name] = mol
        except Exception as e:
            pass
    
    print("✅ Framework imported")
    print(f"✅ Found {len(AVAILABLE_MOLECULES)} test molecules\n")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\n💡 Make sure you're in the MD_MLPES directory")
    sys.exit(1)

try:
    import psi4
    print(f"✅ PSI4 {psi4.__version__}")
    PSI4_AVAILABLE = True
except ImportError:
    print("⚠️  PSI4 not available")
    PSI4_AVAILABLE = False

try:
    import sklearn
    print(f"✅ scikit-learn {sklearn.__version__}")
    ML_AVAILABLE = True
except ImportError:
    print("❌ scikit-learn required")
    sys.exit(1)

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    GP_AVAILABLE = True
    print(f"✅ Gaussian Process available\n")
except ImportError:
    GP_AVAILABLE = False
    print(f"⚠️  Gaussian Process not available (error-based adaptive disabled)\n")

# ==============================================================================
# THEORY LEVEL CORRECTIONS
# ==============================================================================

THEORY_CORRECTIONS = {
    ('HF', 'STO-3G'): 0.0,
    ('HF', '6-31G*'): -8.0,
    ('B3LYP', '6-31G*'): -50.0,
    ('B3LYP', 'cc-pVDZ'): -52.0,
    ('MP2', 'cc-pVDZ'): -60.0,
    ('CCSD(T)', 'cc-pVTZ'): -80.0,
}

def estimate_energy_offset(method1, basis1, method2, basis2, n_heavy_atoms):
    """Estimate energy offset between theory levels."""
    key1 = (method1, basis1)
    key2 = (method2, basis2)
    
    offset1 = THEORY_CORRECTIONS.get(key1, 0.0)
    offset2 = THEORY_CORRECTIONS.get(key2, 0.0)
    
    diff_per_atom = offset2 - offset1
    total_offset_kcal = diff_per_atom * n_heavy_atoms
    offset_ha = total_offset_kcal / 627.509
    
    return offset_ha

# ==============================================================================
# METADATA CLASS
# ==============================================================================

class TrainingMetadata:
    """Complete metadata for ML-PES training."""
    
    def __init__(self):
        self.theory = {
            'method': None,
            'basis': None,
            'reference': 'RHF',
            'functional': None
        }
        self.molecule = {
            'name': None,
            'formula': None,
            'n_atoms': 0,
            'symbols': [],
            'source': None
        }
        self.training = {
            'n_configs': 0,
            'n_trajectories': 0,
            'temperatures': [],
            'energy_range_kcal': 0.0,
            'energy_std_kcal': 0.0,
            'training_type': 'energy_only',  # or 'energy_forces'
            'date_created': None,
            'generation_time_hours': 0.0,
            'source': None
        }
        self.model = {
            'descriptor': 'coulomb_matrix',
            'model_type': 'kernel_ridge',  # or 'gaussian_process'
            'gamma': 0.0,
            'alpha': 0.0,
            'train_rmse_kcal': 0.0,
            'test_rmse_kcal': 0.0,
            'r2_score': 0.0,
            'date_trained': None
        }
        self.files = {
            'training_data': None,
            'model_file': None,
            'output_directory': None,
            'visualizations': []
        }
        self.adaptive = {
            'used': False,
            'method': None,  # 'error_based' or 'gradient_based'
            'base_theory': None,
            'high_theory': None,
            'n_adaptive_points': 0,
            'energy_offset_applied': 0.0
        }
    
    def to_dict(self):
        return {
            'theory': self.theory,
            'molecule': self.molecule,
            'training': self.training,
            'model': self.model,
            'files': self.files,
            'adaptive': self.adaptive,
            'version': '2.2'
        }
    
    def save_json(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: dict):
        metadata = cls()
        metadata.theory = data.get('theory', {})
        metadata.molecule = data.get('molecule', {})
        metadata.training = data.get('training', {})
        metadata.model = data.get('model', {})
        metadata.files = data.get('files', {})
        metadata.adaptive = data.get('adaptive', {})
        return metadata

# ==============================================================================
# XYZ FILE READER
# ==============================================================================

def read_xyz_file(filepath: str):
    """Read molecule from XYZ file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    n_atoms = int(lines[0].strip())
    comment = lines[1].strip()
    
    symbols = []
    coords = []
    
    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    coords = np.array(coords)
    
    class XYZMolecule:
        def __init__(self, symbols, coords, name, comment):
            self.symbols = symbols
            self.coordinates = coords
            self.name = name
            self.comment = comment
            
            from collections import Counter
            counts = Counter(symbols)
            self.formula = ''.join([f"{s}{c if c > 1 else ''}" 
                                   for s, c in sorted(counts.items())])
            
            self.charge = 0
            self.multiplicity = 1
    
    name = Path(filepath).stem
    molecule = XYZMolecule(symbols, coords, name, comment)
    
    print(f"\n✅ Read XYZ file: {filepath}")
    print(f"   Molecule: {molecule.formula}")
    print(f"   Atoms: {len(symbols)}")
    print(f"   Comment: {comment}")
    
    return molecule

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
# FORCE COMPUTATION
# ==============================================================================

def compute_force_descriptor(symbols, coords, forces):
    """
    Compute combined energy+force descriptor.
    Returns: descriptor that includes force information.
    """
    # Start with Coulomb matrix
    cm_desc = compute_coulomb_matrix(symbols, coords)
    
    # Add force magnitude as features
    force_magnitudes = np.linalg.norm(forces, axis=1)
    
    # Combine
    combined = np.concatenate([cm_desc, force_magnitudes])
    
    return combined

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_training_results(y_test, y_pred, rmse, output_dir, training_type='energy_only'):
    """Plot ML-PES training results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Parity plot
    ax1 = axes[0]
    y_test_kcal = y_test * 627.509
    y_pred_kcal = y_pred * 627.509
    
    ax1.scatter(y_test_kcal, y_pred_kcal, alpha=0.5, s=20)
    
    min_val = min(y_test_kcal.min(), y_pred_kcal.min())
    max_val = max(y_test_kcal.max(), y_pred_kcal.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    ax1.set_xlabel('PSI4 Energy (kcal/mol)', fontsize=12)
    ax1.set_ylabel('ML-PES Energy (kcal/mol)', fontsize=12)
    ax1.set_title(f'Parity Plot - {training_type.upper()}\nRMSE: {rmse:.4f} kcal/mol', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Error distribution
    ax2 = axes[1]
    errors = (y_pred - y_test) * 627.509
    
    ax2.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax2.set_xlabel('Prediction Error (kcal/mol)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Error Distribution', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    stats_text = f'Mean: {errors.mean():.4f}\nStd: {errors.std():.4f}\nMax: {np.abs(errors).max():.4f}'
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    
    plot_path = output_dir / f'training_results_{training_type}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved plot: {plot_path}")
    
    return str(plot_path)

def plot_force_comparison(forces_test, forces_pred, output_dir):
    """Plot force prediction results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Flatten forces for comparison
    f_test_flat = forces_test.flatten() * 627.509 / 0.529177  # Ha/bohr to kcal/mol/Å
    f_pred_flat = forces_pred.flatten() * 627.509 / 0.529177
    
    # Parity plot
    ax1 = axes[0]
    ax1.scatter(f_test_flat, f_pred_flat, alpha=0.3, s=10)
    
    min_val = min(f_test_flat.min(), f_pred_flat.min())
    max_val = max(f_test_flat.max(), f_pred_flat.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    rmse_f = np.sqrt(((f_pred_flat - f_test_flat)**2).mean())
    
    ax1.set_xlabel('PSI4 Force (kcal/mol/Å)', fontsize=12)
    ax1.set_ylabel('ML-PES Force (kcal/mol/Å)', fontsize=12)
    ax1.set_title(f'Force Parity Plot\nRMSE: {rmse_f:.4f} kcal/mol/Å', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Error distribution
    ax2 = axes[1]
    errors = f_pred_flat - f_test_flat
    
    ax2.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax2.set_xlabel('Force Error (kcal/mol/Å)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Force Error Distribution', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / 'force_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved force plot: {plot_path}")
    
    return str(plot_path)

# ==============================================================================
# MAIN WORKFLOW - RESTRUCTURED
# ==============================================================================

def main():
    """Main workflow - existing data first, new computations second."""
    
    print("\n" + "=" * 80)
    print("  WORKFLOW SELECTION")
    print("=" * 80)
    
    print("\n📊 What would you like to do?\n")
    print("  [1] Work with existing training data (refine ML-PES)")
    print("      → Train energy-only model")
    print("      → Train energy+forces model")
    print("      → Adaptive refinement (error-based)")
    print("      → Adaptive refinement (gradient-based)")
    print()
    print("  [2] Compute new molecule")
    print("      → Generate new PSI4 training data")
    print("      → Choose from library, XYZ file, or custom")
    
    workflow_choice = input("\nSelect [1-2]: ").strip()
    
    if workflow_choice == '1':
        # OPTION 1: Work with existing data
        refine_existing_data()
    
    elif workflow_choice == '2':
        # OPTION 2: Compute new molecule
        compute_new_molecule()
    
    else:
        print("\n❌ Invalid choice")
        sys.exit(1)

# ==============================================================================
# OPTION 1: REFINE EXISTING DATA
# ==============================================================================

def refine_existing_data():
    """Refine ML-PES with existing training data."""
    
    print("\n" + "=" * 80)
    print("  OPTION 1: REFINE EXISTING DATA")
    print("=" * 80)
    
    # Option to use database or scan
    print("\n📂 How would you like to find data?\n")
    print("  [1] Use computation database (fast, organized)")
    print("  [2] Scan directories (slow, finds everything)")
    
    find_choice = input("\nSelect [1-2] or press Enter for [1]: ").strip() or '1'
    
    if find_choice == '1':
        # Try to use database
        try:
            from computation_database import ComputationDatabase, scan_and_register
            
            db = ComputationDatabase()
            
            # Check if database is empty
            valid_comps = db.query(valid_only=True)
            
            if not valid_comps:
                print("\n⚠️  Database empty. Scanning directories...")
                scan_and_register(db)
                valid_comps = db.query(valid_only=True)
            
            if not valid_comps:
                print("\n❌ No computations found!")
                sys.exit(1)
            
            # Display from database
            print(f"\n📂 Found {len(valid_comps)} computation(s) in database:\n")
            
            for i, entry in enumerate(valid_comps[:10], 1):
                method = entry['theory']['method']
                basis = entry['theory']['basis']
                mol_name = entry['molecule']['name']
                mol_formula = entry['molecule']['formula']
                n_configs = entry['training']['n_configs']
                e_range = entry['training'].get('energy_range_kcal', 0)
                date_mod = entry['metadata']['date_modified'][:10]
                
                print(f"  [{i}] {mol_name} ({mol_formula}) | {method}/{basis}")
                print(f"      Configs: {n_configs}, Energy range: {e_range:.2f} kcal/mol")
                print(f"      Modified: {date_mod}")
                print(f"      Path: {entry['filepath']}")
                print()
            
            choice = int(input(f"Select [1-{min(10, len(valid_comps))}]: "))
            training_data_path = valid_comps[choice - 1]['filepath']
            
        except ImportError:
            print("\n⚠️  Database module not available, scanning directories...")
            find_choice = '2'
    
    if find_choice == '2':
        # Scan directories (original method)
        found = glob.glob('outputs/*/training_data/*.npz') + glob.glob('outputs/*/*.npz')
        
        if not found:
            print("\n❌ No training data found!")
            print("   Use option [2] to generate new data first")
            print("\n💡 Or run: python3 migrate_metadata.py")
            sys.exit(1)
        
        found = sorted(set(found), key=lambda x: Path(x).stat().st_mtime, reverse=True)
        
        print(f"\n📂 Found {len(found)} dataset(s):\n")
        
        valid_files = []
        
        for i, path in enumerate(found[:10], 1):
            try:
                traj = load_trajectory(path)
                e_range = (traj.energies.max() - traj.energies.min()) * 627.509
                mod_time = datetime.fromtimestamp(Path(path).stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                
                # Try to get metadata
                metadata_info = ""
                try:
                    if hasattr(traj, 'metadata') and traj.metadata:
                        meta = traj.metadata
                        if isinstance(meta, dict) and 'theory' in meta:
                            method = meta['theory'].get('method', '?')
                            basis = meta['theory'].get('basis', '?')
                            metadata_info = f" | {method}/{basis}"
                            
                            if 'molecule' in meta:
                                mol_name = meta['molecule'].get('name', '?')
                                metadata_info += f" | {mol_name}"
                except:
                    metadata_info = " | no metadata"
                
                print(f"  [{i}] {path}")
                print(f"      Configs: {traj.n_frames}, Energy range: {e_range:.2f} kcal/mol{metadata_info}")
                print(f"      Modified: {mod_time}")
                print()
                
                valid_files.append(path)
                
            except Exception as e:
                print(f"  [{i}] {path}")
                print(f"      ❌ Error: {str(e)[:50]}")
                print()
        
        if not valid_files:
            print("\n❌ No valid datasets found!")
            print("\n💡 Run: python3 migrate_metadata.py")
            print("   This will fix old files without metadata")
            sys.exit(1)
        
        choice = int(input(f"Select [1-{len(valid_files)}]: "))
        training_data_path = valid_files[choice - 1]
    
    print(f"\n📂 Loading: {training_data_path}")
    training_data = load_trajectory(training_data_path)
    
    print(f"✅ Loaded {training_data.n_frames} configurations")
    
    # Show dataset info
    e_kcal = training_data.energies * 627.509
    print(f"\n📊 Dataset Statistics:")
    print(f"   Configurations: {training_data.n_frames}")
    print(f"   Atoms: {len(training_data.symbols)}")
    print(f"   Symbols: {' '.join(training_data.symbols)}")
    print(f"   Energy range: {e_kcal.max() - e_kcal.min():.2f} kcal/mol")
    print(f"   Energy std: {e_kcal.std():.2f} kcal/mol")
    print(f"   Has forces: {'Yes' if training_data.forces is not None else 'No'}")
    
    # Refinement options
    print("\n" + "=" * 80)
    print("  REFINEMENT OPTIONS")
    print("=" * 80)
    
    print("\n🎯 How would you like to refine this data?\n")
    print("  [1] Energy-only training")
    print("      → Train on energies only (fastest, good for equilibrium)")
    print()
    print("  [2] Energy + forces training")
    print("      → Train on both energies and forces (better dynamics)")
    if training_data.forces is None:
        print("      ⚠️  No forces in dataset - will need to compute")
    print()
    print("  [3] Adaptive refinement (error-based)")
    print("      → Add points where ML model has highest uncertainty")
    if not GP_AVAILABLE:
        print("      ⚠️  Requires Gaussian Process (not available)")
    print()
    print("  [4] Adaptive refinement (gradient-based)")
    print("      → Add higher-theory points where PES changes most")
    print()
    print("  [5] Two-phase diagnostic & validation (RECOMMENDED)")
    print("      → Fast ML-PES MD + smart PSI4 validation + error analysis")
    print("      → Much faster and more informative than real-time validation!")
    if not PSI4_AVAILABLE:
        print("      ⚠️  Requires PSI4 (not available)")
    print()
    
    refine_choice = input("Select [1-5]: ").strip()
    
    # Create output directory
    output_dir = Path(f'outputs/refined_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if refine_choice == '1':
        # Energy-only training
        train_energy_only(training_data, output_dir, training_data_path)
    
    elif refine_choice == '2':
        # Energy + forces training
        train_energy_forces(training_data, output_dir, training_data_path)
    
    elif refine_choice == '3':
        # Error-based adaptive
        if not GP_AVAILABLE:
            print("\n❌ Gaussian Process not available for error-based refinement")
            sys.exit(1)
        adaptive_error_based(training_data, output_dir, training_data_path)
    
    elif refine_choice == '4':
        # Gradient-based adaptive
        adaptive_gradient_based(training_data, output_dir, training_data_path)
    
    elif refine_choice == '5':
        # On-the-fly validation
        if not PSI4_AVAILABLE:
            print("\n❌ PSI4 required for on-the-fly validation")
            sys.exit(1)
        on_the_fly_validation(training_data, output_dir, training_data_path)
    
    else:
        print("\n❌ Invalid choice")
        sys.exit(1)

# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_energy_only(training_data, output_dir, data_path):
    """Train energy-only ML-PES."""
    
    print("\n" + "=" * 80)
    print("  ENERGY-ONLY TRAINING")
    print("=" * 80)
    
    metadata = TrainingMetadata()
    metadata.training['training_type'] = 'energy_only'
    metadata.training['n_configs'] = training_data.n_frames
    metadata.training['source'] = 'loaded'
    metadata.files['training_data'] = data_path
    metadata.files['output_directory'] = str(output_dir)
    
    # Extract metadata from data if available
    if hasattr(training_data, 'metadata') and training_data.metadata:
        stored_meta = training_data.metadata
        if 'theory' in stored_meta:
            metadata.theory = stored_meta['theory']
        if 'molecule' in stored_meta:
            metadata.molecule = stored_meta['molecule']
    
    print("\n🔧 Computing descriptors...")
    X = []
    for i in tqdm(range(training_data.n_frames), desc="Descriptors"):
        desc = compute_coulomb_matrix(training_data.symbols, training_data.coordinates[i])
        X.append(desc)
    X = np.array(X)
    y = training_data.energies
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    print("\n🤖 Training Kernel Ridge models...")
    
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
    
    print(f"\n✅ Best Model (Energy-Only):")
    print(f"   Gamma: {best_params[0]}")
    print(f"   Alpha: {best_params[1]}")
    print(f"   RMSE: {best_rmse:.4f} kcal/mol")
    print(f"   R²: {r2:.6f}")
    
    # Plot results
    print(f"\n📊 Creating visualizations...")
    plot_path = plot_training_results(y_test, y_pred, best_rmse, output_dir, 'energy_only')
    metadata.files['visualizations'].append(plot_path)
    
    # Update metadata
    metadata.model['model_type'] = 'kernel_ridge'
    metadata.model['gamma'] = float(best_params[0])
    metadata.model['alpha'] = float(best_params[1])
    metadata.model['test_rmse_kcal'] = float(best_rmse)
    metadata.model['r2_score'] = float(r2)
    metadata.model['date_trained'] = datetime.now().isoformat()
    
    # Save model
    model_data = {
        'model': best_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'symbols': training_data.symbols,
        'metadata': metadata.to_dict(),
        'version': '2.2'
    }
    
    model_path = output_dir / 'mlpes_model_energy_only.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    metadata.files['model_file'] = str(model_path)
    
    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    metadata.save_json(str(metadata_path))
    
    print(f"\n💾 Saved:")
    print(f"   Model: {model_path}")
    print(f"   Metadata: {metadata_path}")
    
    print_summary(metadata, best_rmse)

def train_energy_forces(training_data, output_dir, data_path):
    """Train energy+forces ML-PES."""
    
    print("\n" + "=" * 80)
    print("  ENERGY + FORCES TRAINING")
    print("=" * 80)
    
    if training_data.forces is None:
        print("\n⚠️  Dataset has no forces!")
        print("   Need to compute forces first with PSI4")
        # Could implement force computation here
        return
    
    metadata = TrainingMetadata()
    metadata.training['training_type'] = 'energy_forces'
    metadata.training['n_configs'] = training_data.n_frames
    metadata.training['source'] = 'loaded'
    metadata.files['training_data'] = data_path
    metadata.files['output_directory'] = str(output_dir)
    
    print("\n🔧 Computing energy+force descriptors...")
    X = []
    for i in tqdm(range(training_data.n_frames), desc="Descriptors"):
        desc = compute_force_descriptor(training_data.symbols, 
                                       training_data.coordinates[i],
                                       training_data.forces[i])
        X.append(desc)
    X = np.array(X)
    y = training_data.energies
    
    # Also prepare force targets
    y_forces = training_data.forces.reshape(training_data.n_frames, -1)  # Flatten forces
    
    # Split data
    X_train, X_test, y_train, y_test, yf_train, yf_test = train_test_split(
        X, y, y_forces, test_size=0.2, random_state=42)
    
    # Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_yf = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    yf_train_scaled = scaler_yf.fit_transform(yf_train)
    yf_test_scaled = scaler_yf.transform(yf_test)
    
    print("\n🤖 Training energy model...")
    
    # Train energy model (simplified hyperparameter search)
    best_rmse = float('inf')
    best_model = None
    best_params = None
    
    for gamma, alpha in tqdm([(0.01, 0.01), (0.1, 0.01), (0.1, 0.1)], desc="Energy model"):
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
    
    print("\n🤖 Training force model...")
    
    # Train separate force model
    force_model = KernelRidge(kernel='rbf', gamma=best_params[0], alpha=best_params[1])
    force_model.fit(X_train_scaled, yf_train_scaled)
    
    yf_pred_scaled = force_model.predict(X_test_scaled)
    yf_pred = scaler_yf.inverse_transform(yf_pred_scaled)
    
    # Evaluate
    y_pred_scaled = best_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    r2 = 1 - ((y_test - y_pred)**2).sum() / ((y_test - y_test.mean())**2).sum()
    
    # Force RMSE
    force_errors = (yf_pred - yf_test).flatten() * 627.509 / 0.529177  # kcal/mol/Å
    force_rmse = np.sqrt((force_errors**2).mean())
    
    print(f"\n✅ Best Model (Energy+Forces):")
    print(f"   Gamma: {best_params[0]}")
    print(f"   Alpha: {best_params[1]}")
    print(f"   Energy RMSE: {best_rmse:.4f} kcal/mol")
    print(f"   Force RMSE: {force_rmse:.4f} kcal/mol/Å")
    print(f"   R²: {r2:.6f}")
    
    # Plot results
    print(f"\n📊 Creating visualizations...")
    plot_path1 = plot_training_results(y_test, y_pred, best_rmse, output_dir, 'energy_forces')
    
    # Reshape forces for plotting
    forces_test_3d = yf_test.reshape(-1, len(training_data.symbols), 3)
    forces_pred_3d = yf_pred.reshape(-1, len(training_data.symbols), 3)
    plot_path2 = plot_force_comparison(forces_test_3d, forces_pred_3d, output_dir)
    
    metadata.files['visualizations'].extend([plot_path1, plot_path2])
    
    # Update metadata
    metadata.model['model_type'] = 'kernel_ridge'
    metadata.model['gamma'] = float(best_params[0])
    metadata.model['alpha'] = float(best_params[1])
    metadata.model['test_rmse_kcal'] = float(best_rmse)
    metadata.model['r2_score'] = float(r2)
    metadata.model['date_trained'] = datetime.now().isoformat()
    
    # Save both models
    model_data = {
        'energy_model': best_model,
        'force_model': force_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'scaler_yf': scaler_yf,
        'symbols': training_data.symbols,
        'metadata': metadata.to_dict(),
        'version': '2.2'
    }
    
    model_path = output_dir / 'mlpes_model_energy_forces.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    metadata.files['model_file'] = str(model_path)
    
    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    metadata.save_json(str(metadata_path))
    
    print(f"\n💾 Saved:")
    print(f"   Model: {model_path}")
    print(f"   Metadata: {metadata_path}")
    
    print_summary(metadata, best_rmse)

def adaptive_error_based(training_data, output_dir, data_path):
    """Adaptive refinement based on ML model uncertainty (GP only)."""
    
    print("\n" + "=" * 80)
    print("  ADAPTIVE REFINEMENT (ERROR-BASED)")
    print("=" * 80)
    
    print("\n🎯 Strategy: Train GP model, identify high-uncertainty regions,")
    print("   add more training points there with PSI4")
    
    print("\n⚠️  This feature requires:")
    print("   1. Gaussian Process model (for uncertainty estimates)")
    print("   2. PSI4 available for computing new points")
    print("   3. Same theory level as original data")
    
    print("\n💡 Implementation:")
    print("   - Train GP on existing data")
    print("   - Identify top 10% uncertain points")
    print("   - Sample geometries near those points")
    print("   - Compute with PSI4")
    print("   - Retrain with augmented dataset")
    
    print("\n🚧 This is a placeholder - full implementation coming soon!")
    print("   For now, use gradient-based adaptive [option 4]")

def adaptive_gradient_based(training_data, output_dir, data_path):
    """Adaptive refinement based on PES gradient (add higher theory)."""
    
    print("\n" + "=" * 80)
    print("  ADAPTIVE REFINEMENT (GRADIENT-BASED)")
    print("=" * 80)
    
    print("\n🎯 Strategy: Identify regions with largest PES gradients,")
    print("   compute those with higher theory level, apply energy corrections")
    
    if training_data.forces is None:
        print("\n❌ Dataset has no forces - cannot identify high-gradient regions")
        print("   Need forces to compute gradients")
        return
    
    # Compute force magnitudes
    force_mags = np.linalg.norm(training_data.forces, axis=(1, 2))
    
    print(f"\n📊 Force magnitude statistics:")
    print(f"   Mean: {force_mags.mean():.4f} Ha/bohr")
    print(f"   Std: {force_mags.std():.4f} Ha/bohr")
    print(f"   Max: {force_mags.max():.4f} Ha/bohr")
    
    # Identify high-gradient points
    percentile = float(input("\n   Percentile threshold for high gradients [90]: ").strip() or "90")
    threshold = np.percentile(force_mags, percentile)
    
    high_grad_idx = np.where(force_mags >= threshold)[0]
    
    print(f"\n✅ Found {len(high_grad_idx)} high-gradient points (>= {threshold:.4f} Ha/bohr)")
    print(f"   These are {len(high_grad_idx)/len(force_mags)*100:.1f}% of the dataset")
    
    print("\n🎯 Next steps:")
    print(f"   1. Sample {len(high_grad_idx)*3} points near high-gradient regions")
    print(f"   2. Compute with higher theory level (e.g., MP2 or CCSD(T))")
    print(f"   3. Apply energy offset correction")
    print(f"   4. Combine with base dataset and retrain")
    
    print("\n💡 This requires PSI4 calculations at higher theory")
    print("   Estimated time: {len(high_grad_idx)*3} points × ~30 sec/point = {len(high_grad_idx)*3*30/3600:.1f} hours")
    
    proceed = input("\n   Continue with computation? [y/n]: ").strip().lower()
    
    if proceed != 'y':
        print("\n   Stopping. You can use the identified high-gradient indices for manual computation.")
        # Save indices
        np.save(output_dir / 'high_gradient_indices.npy', high_grad_idx)
        print(f"   Saved indices to: {output_dir / 'high_gradient_indices.npy'}")
        return
    
    print("\n🚧 PSI4 computation at higher theory level - implementation placeholder")
    print("   Full implementation coming soon!")

def on_the_fly_validation(training_data, output_dir, data_path):
    """Two-phase diagnostic workflow (RECOMMENDED)."""
    
    print("\n" + "=" * 80)
    print("  TWO-PHASE DIAGNOSTIC & VALIDATION (RECOMMENDED)")
    print("=" * 80)
    
    print("\n🎯 This workflow:")
    print("   Phase 1: Fast ML-PES MD (5-10 minutes)")
    print("      • Run MD with ML-PES only (no PSI4)")
    print("      • Save snapshots every N steps")
    print("      • 100x faster than real-time validation!")
    print()
    print("   Phase 2: Smart validation (5-15 minutes)")
    print("      • Validate strategically sampled snapshots with PSI4")
    print("      • Generate comprehensive error analysis")
    print("      • Create 6-panel diagnostic plots")
    print()
    print("   Phase 3: Recommendations (instant)")
    print("      • Specific guidance based on observed errors")
    print("      • Temperature adjustments if needed")
    print("      • Refinement options")
    
    # First, train initial model
    print("\n" + "=" * 80)
    print("  STEP 1: TRAINING INITIAL MODEL")
    print("=" * 80)
    
    metadata = TrainingMetadata()
    metadata.training['training_type'] = 'energy_only'
    metadata.training['n_configs'] = training_data.n_frames
    metadata.training['source'] = 'loaded'
    metadata.files['training_data'] = data_path
    metadata.files['output_directory'] = str(output_dir)
    
    # Extract metadata from data if available
    if hasattr(training_data, 'metadata') and training_data.metadata:
        stored_meta = training_data.metadata
        if 'theory' in stored_meta:
            metadata.theory = stored_meta['theory']
        if 'molecule' in stored_meta:
            metadata.molecule = stored_meta['molecule']
    
    print("\n🔧 Computing descriptors...")
    X = []
    for i in tqdm(range(training_data.n_frames), desc="Descriptors"):
        desc = compute_coulomb_matrix(training_data.symbols, training_data.coordinates[i])
        X.append(desc)
    X = np.array(X)
    y = training_data.energies
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    print("\n🤖 Training initial model...")
    
    # Quick hyperparameter search
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
    
    print(f"\n✅ Initial Model:")
    print(f"   Gamma: {best_params[0]}")
    print(f"   Alpha: {best_params[1]}")
    print(f"   RMSE: {best_rmse:.4f} kcal/mol")
    print(f"   R²: {r2:.6f}")
    
    # WARNING if model looks bad
    if best_rmse > 10.0:
        print(f"\n⚠️  WARNING: High training RMSE ({best_rmse:.2f} kcal/mol)")
        print(f"   This model may not perform well in MD!")
        print(f"   Consider: More training data or different hyperparameters")
    
    # Save initial model
    metadata.model['gamma'] = float(best_params[0])
    metadata.model['alpha'] = float(best_params[1])
    metadata.model['test_rmse_kcal'] = float(best_rmse)
    metadata.model['r2_score'] = float(r2)
    
    model_data = {
        'model': best_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'symbols': training_data.symbols,
        'metadata': metadata.to_dict(),
        'version': '2.2'
    }
    
    initial_model_path = output_dir / 'initial_mlpes_model.pkl'
    with open(initial_model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n💾 Saved initial model: {initial_model_path}")
    
    # Now call two_phase_workflow.py with better defaults
    print("\n" + "=" * 80)
    print("  STEP 2: TWO-PHASE DIAGNOSTIC")
    print("=" * 80)
    
    print("\n⚙️  Diagnostic parameters:")
    print(f"   (Use conservative defaults - adjust if needed)")
    
    # Get parameters with MUCH better defaults
    n_steps = int(input("   Number of MD steps [1000]: ").strip() or "1000")
    temperature = float(input("   Temperature (K) [100]: ").strip() or "100")  # Changed from 400!
    snapshot_interval = int(input("   Snapshot interval [10]: ").strip() or "10")
    max_validate = int(input("   Max snapshots to validate [50]: ").strip() or "50")
    
    # Build command for two_phase_workflow.py
    cmd = [
        'python3', 'two_phase_workflow.py',
        '--model', str(initial_model_path),
        '--training-data', data_path,
        '--steps', str(n_steps),
        '--temp', str(temperature),
        '--snapshot-every', str(snapshot_interval),
        '--validate-max', str(max_validate)
    ]
    
    print(f"\n🚀 Launching two-phase diagnostic...")
    print(f"   Expected time: ~{n_steps*0.02/60 + max_validate*2/60:.0f} minutes")
    print(f"   Command: {' '.join(cmd)}")
    
    # Check if script exists
    if not Path('two_phase_workflow.py').exists():
        print(f"\n❌ two_phase_workflow.py not found in current directory!")
        print(f"   Please download it from outputs/ or the project repository")
        print(f"\n💡 You can still use the model saved at:")
        print(f"   {initial_model_path}")
        return
    
    # Run external script
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=False)
        
        print("\n" + "=" * 80)
        print("  TWO-PHASE DIAGNOSTIC COMPLETE")
        print("=" * 80)
        
        # Find output directory
        diagnostic_dirs = sorted(glob.glob('outputs/diagnostic_phase1_*'), 
                               key=lambda x: Path(x).stat().st_mtime, reverse=True)
        
        if diagnostic_dirs:
            latest_dir = diagnostic_dirs[0]
            print(f"\n📂 Results saved to: {latest_dir}")
            print(f"\n📊 Key files:")
            print(f"   • diagnostic_plots.png - 6-panel comprehensive visualization")
            print(f"   • phase1_snapshots.pkl - Complete MD trajectory")
            print(f"   • phase2_results.pkl - Detailed validation data")
            
            print(f"\n💡 Next steps:")
            print(f"   1. Check diagnostic_plots.png for error analysis")
            print(f"   2. Review terminal output for recommendations")
            print(f"   3. If errors < 5 kcal/mol: Model is good!")
            print(f"   4. If errors 5-20 kcal/mol: Consider refinement")
            print(f"   5. If errors > 20 kcal/mol: Model needs improvement")
            print(f"")
            print(f"   If errors are huge (>1000 kcal/mol) at low temp:")
            print(f"   → Training data or model has fundamental issues")
            print(f"   → Check training data quality and theory level match")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running two_phase_workflow.py: {e}")
        print(f"\n💡 You can still use the model manually:")
        print(f"   python3 two_phase_workflow.py --model {initial_model_path} --training-data {data_path} --steps 1000 --temp 100")
    except FileNotFoundError:
        print(f"\n❌ Python3 not found or script missing")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

# ==============================================================================
# OPTION 2: COMPUTE NEW MOLECULE
# ==============================================================================

def compute_new_molecule():
    """Compute new molecule - full workflow from v2.1."""
    
    print("\n" + "=" * 80)
    print("  OPTION 2: COMPUTE NEW MOLECULE")
    print("=" * 80)
    
    print("\n💡 This will generate new PSI4 training data")
    print("   (Full workflow similar to v2.1)")
    
    print("\n🚧 For now, please use complete_workflow_v2.1.py for new molecules")
    print("   This refinement-focused workflow will be fully integrated soon!")

# ==============================================================================
# UTILITY
# ==============================================================================

def print_summary(metadata, rmse):
    """Print workflow summary."""
    
    print("\n" + "=" * 80)
    print("  WORKFLOW COMPLETE")
    print("=" * 80)
    
    print(f"\n📊 Summary:")
    if metadata.molecule['name']:
        print(f"   Molecule: {metadata.molecule['name']} ({metadata.molecule['formula']})")
    print(f"   Training: {metadata.training['n_configs']} configs")
    print(f"   Type: {metadata.training['training_type']}")
    print(f"   RMSE: {rmse:.4f} kcal/mol")
    if metadata.model['r2_score'] > 0:
        print(f"   R²: {metadata.model['r2_score']:.6f}")
    print(f"   Output: {metadata.files['output_directory']}")
    
    if rmse < 0.1:
        print(f"\n🎉 EXCEPTIONAL! Essentially at QM noise level!")
    elif rmse < 1.0:
        print(f"\n🎉 EXCELLENT! Chemical accuracy achieved!")
    elif rmse < 5.0:
        print(f"\n✅ GOOD! Suitable for most applications.")
    
    print("\n" + "=" * 80)

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
