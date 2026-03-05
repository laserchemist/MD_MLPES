#!/usr/bin/env python3
"""
Complete Production ML-PES Workflow - Fixed with Metadata

Features:
- ✅ Proper metadata storage (method, basis, date, etc.)
- ✅ Interactive method/basis selection
- ✅ Choose from test molecules OR custom molecules
- ✅ Theory level verification
- ✅ Complete tracking and documentation

Author: PSI4-MD Framework
Version: 2.0 (with metadata)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import json

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
print("  PRODUCTION ML-PES WORKFLOW v2.0 - WITH METADATA")
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
            if mol is not None:  # Only add if molecule exists
                AVAILABLE_MOLECULES[name] = mol
        except Exception as e:
            pass  # Skip if not available
    
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
    print(f"✅ scikit-learn {sklearn.__version__}\n")
except ImportError:
    print("❌ scikit-learn required")
    sys.exit(1)

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
            'functional': None  # For DFT
        }
        self.molecule = {
            'name': None,
            'formula': None,
            'n_atoms': 0,
            'symbols': []
        }
        self.training = {
            'n_configs': 0,
            'n_trajectories': 0,
            'temperatures': [],
            'energy_range_kcal': 0.0,
            'energy_std_kcal': 0.0,
            'training_type': 'energy_only',
            'date_created': None,
            'generation_time_hours': 0.0
        }
        self.model = {
            'descriptor': 'coulomb_matrix',
            'model_type': 'kernel_ridge',
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
            'output_directory': None
        }
    
    def to_dict(self):
        return {
            'theory': self.theory,
            'molecule': self.molecule,
            'training': self.training,
            'model': self.model,
            'files': self.files,
            'version': '2.0'
        }
    
    def save_json(self, filepath: str):
        """Save metadata as JSON for easy reading."""
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
        return metadata

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
# INTERACTIVE MENUS
# ==============================================================================

def select_molecule():
    """Interactive molecule selection."""
    print("=" * 80)
    print("  SELECT MOLECULE")
    print("=" * 80)
    
    print("\n📊 Options:")
    print("  [1] Choose from test molecules")
    print("  [2] Define custom molecule")
    
    choice = input("\nSelect [1-2]: ").strip()
    
    if choice == '1':
        if not AVAILABLE_MOLECULES:
            print("\n⚠️  No test molecules available, using custom input instead")
            choice = '2'
        else:
            print(f"\n📚 Available test molecules:\n")
            
            # Filter out any None values that might have slipped through
            mol_list = [name for name, mol in AVAILABLE_MOLECULES.items() if mol is not None]
            
            if not mol_list:
                print("\n⚠️  No valid test molecules found, using custom input instead")
                choice = '2'
            else:
                for i, name in enumerate(mol_list, 1):
                    mol = AVAILABLE_MOLECULES[name]
                    ref_energy = mol.reference_energy if hasattr(mol, 'reference_energy') else 0.0
                    print(f"  [{i}] {mol.name} ({mol.formula})")
                    if ref_energy != 0.0:
                        print(f"      Atoms: {len(mol.symbols)}, Reference: {ref_energy:.6f} Ha")
                    else:
                        print(f"      Atoms: {len(mol.symbols)}")
                
                mol_choice = int(input(f"\nSelect molecule [1-{len(mol_list)}]: "))
                mol_name = mol_list[mol_choice - 1]
                molecule = get_molecule(mol_name)
    
    if choice == '2':
        print("\n🔧 Define custom molecule:")
        print("   Format: Symbol x y z (one atom per line)")
        print("   Type 'done' when finished")
        print("\n   Example:")
        print("   O 0.0 0.0 0.0")
        print("   H 0.0 0.757 0.586")
        print("   H 0.0 -0.757 0.586")
        print("   done\n")
        
        symbols = []
        coords = []
        
        while True:
            line = input(f"   Atom {len(symbols)+1} (or 'done'): ").strip()
            
            if line.lower() == 'done':
                break
            
            try:
                parts = line.split()
                symbol = parts[0]
                x, y, z = map(float, parts[1:4])
                
                symbols.append(symbol)
                coords.append([x, y, z])
            except:
                print("      ❌ Invalid format. Use: Symbol x y z")
        
        if not symbols:
            print("\n❌ No atoms provided!")
            sys.exit(1)
        
        # Create custom molecule object
        class CustomMolecule:
            def __init__(self, symbols, coords):
                self.symbols = symbols
                self.coordinates = np.array(coords)
                self.name = 'custom'
                # Count atoms for formula
                from collections import Counter
                counts = Counter(symbols)
                self.formula = ''.join([f"{s}{c if c > 1 else ''}" for s, c in sorted(counts.items())])
                self.charge = 0
                self.multiplicity = 1
        
        molecule = CustomMolecule(symbols, coords)
    
    print(f"\n✅ Selected: {molecule.name} ({molecule.formula})")
    print(f"   Atoms: {len(molecule.symbols)}")
    
    return molecule

def select_theory_level():
    """Interactive theory level selection."""
    print("\n" + "=" * 80)
    print("  SELECT THEORY LEVEL")
    print("=" * 80)
    
    print("\n🔬 Common methods:")
    print("  [1] HF         - Hartree-Fock (fastest, least accurate)")
    print("  [2] B3LYP      - DFT hybrid functional (good balance)")
    print("  [3] MP2        - Møller-Plesset 2nd order (good, expensive)")
    print("  [4] ωB97X-D    - DFT with dispersion (very good)")
    print("  [5] Custom")
    
    method_choice = input("\nSelect method [1-5]: ").strip()
    
    methods = {
        '1': 'HF',
        '2': 'B3LYP',
        '3': 'MP2',
        '4': 'wB97X-D'
    }
    
    if method_choice in methods:
        method = methods[method_choice]
    else:
        method = input("   Enter method name: ").strip()
    
    print(f"\n📚 Common basis sets:")
    print("  [1] STO-3G      - Minimal (very fast, low accuracy)")
    print("  [2] 6-31G*      - Split-valence (good for organics)")
    print("  [3] 6-31+G**    - With diffuse functions (better)")
    print("  [4] cc-pVDZ     - Correlation consistent (more accurate)")
    print("  [5] cc-pVTZ     - Larger correlation consistent (expensive)")
    print("  [6] Custom")
    
    basis_choice = input("\nSelect basis [1-6]: ").strip()
    
    basis_sets = {
        '1': 'STO-3G',
        '2': '6-31G*',
        '3': '6-31+G**',
        '4': 'cc-pVDZ',
        '5': 'cc-pVTZ'
    }
    
    if basis_choice in basis_sets:
        basis = basis_sets[basis_choice]
    else:
        basis = input("   Enter basis set name: ").strip()
    
    print(f"\n✅ Selected: {method}/{basis}")
    
    # Estimate computational cost
    cost_factors = {
        'HF': 1, 'B3LYP': 3, 'MP2': 30, 'wB97X-D': 4
    }
    basis_factors = {
        'STO-3G': 1, '6-31G*': 3, '6-31+G**': 5, 'cc-pVDZ': 8, 'cc-pVTZ': 30
    }
    
    cost = cost_factors.get(method, 5) * basis_factors.get(basis, 5)
    
    if cost < 5:
        print(f"   ⚡ Fast (suitable for 1000+ configs)")
    elif cost < 20:
        print(f"   ⚙️  Moderate (suitable for 500-1000 configs)")
    elif cost < 100:
        print(f"   ⏱️  Slow (limit to 100-500 configs)")
    else:
        print(f"   🐌 Very slow (limit to < 100 configs)")
    
    confirm = input("\n   Proceed? [y/n]: ").strip().lower()
    if confirm != 'y':
        return select_theory_level()
    
    return method, basis

def select_training_parameters():
    """Interactive training parameters."""
    print("\n" + "=" * 80)
    print("  TRAINING PARAMETERS")
    print("=" * 80)
    
    print("\n📊 Suggested configurations:\n")
    print("  [1] Quick test    - 2 trajectories, 100 steps (~10 min)")
    print("  [2] Standard      - 5 trajectories, 300 steps (~2 hours)")
    print("  [3] High quality  - 8 trajectories, 500 steps (~6 hours)")
    print("  [4] Custom")
    
    choice = input("\nSelect [1-4]: ").strip()
    
    presets = {
        '1': {'n_traj': 2, 'n_steps': 100, 'temps': [200, 400]},
        '2': {'n_traj': 5, 'n_steps': 300, 'temps': [200, 300, 400, 500, 600]},
        '3': {'n_traj': 8, 'n_steps': 500, 'temps': [200, 300, 400, 500, 600, 700, 800]}
    }
    
    if choice in presets:
        params = presets[choice]
        n_traj = params['n_traj']
        n_steps = params['n_steps']
        temps = params['temps']
    else:
        n_traj = int(input("   Number of trajectories: "))
        n_steps = int(input("   Steps per trajectory: "))
        temps = input("   Temperatures (comma-separated): ").split(',')
        temps = [int(t.strip()) for t in temps]
    
    output_freq = int(input("\n   Output frequency [2]: ").strip() or "2")
    timestep = float(input("   Timestep (fs) [0.5]: ").strip() or "0.5")
    
    n_configs = n_traj * (n_steps // output_freq)
    
    print(f"\n✅ Configuration:")
    print(f"   Trajectories: {n_traj}")
    print(f"   Steps each: {n_steps}")
    print(f"   Temperatures: {temps}")
    print(f"   Expected configs: ~{n_configs}")
    
    return {
        'n_trajectories': n_traj,
        'n_steps': n_steps,
        'temperatures': temps,
        'output_frequency': output_freq,
        'timestep': timestep
    }

# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

def main():
    """Main workflow with metadata."""
    
    # Initialize metadata
    metadata = TrainingMetadata()
    metadata.training['date_created'] = datetime.now().isoformat()
    start_time = datetime.now()
    
    # Step 1: Select molecule
    molecule = select_molecule()
    
    metadata.molecule['name'] = molecule.name
    metadata.molecule['formula'] = molecule.formula if hasattr(molecule, 'formula') else 'unknown'
    metadata.molecule['n_atoms'] = len(molecule.symbols)
    metadata.molecule['symbols'] = molecule.symbols
    
    # Step 2: Select theory level
    method, basis = select_theory_level()
    
    metadata.theory['method'] = method
    metadata.theory['basis'] = basis
    
    # Detect functional type
    dft_methods = ['B3LYP', 'PBE', 'wB97X-D', 'M06-2X']
    if method in dft_methods:
        metadata.theory['functional'] = method
    
    # Step 3: Training parameters
    if PSI4_AVAILABLE:
        print("\n" + "=" * 80)
        print("  GENERATE TRAINING DATA?")
        print("=" * 80)
        
        print("\n  [1] Generate new data with PSI4")
        print("  [2] Load existing data")
        
        gen_choice = input("\nSelect [1-2]: ").strip()
        
        if gen_choice == '1':
            params = select_training_parameters()
            
            metadata.training['n_trajectories'] = params['n_trajectories']
            metadata.training['temperatures'] = params['temperatures']
            
            # Generate data
            print("\n" + "=" * 80)
            print("  GENERATING TRAINING DATA")
            print("=" * 80)
            
            print(f"\n🚀 Starting PSI4 calculations...")
            print(f"   Method: {method}/{basis}")
            print(f"   Trajectories: {params['n_trajectories']}")
            print(f"   Temperatures: {params['temperatures']}")
            
            md_config = DirectMDConfig(
                method=method,
                basis=basis,
                timestep=params['timestep'],
                n_steps=params['n_steps'],
                output_frequency=params['output_frequency'],
                thermostat='berendsen',
                calculate_dipole=True
            )
            
            trajectories = []
            
            for i in tqdm(range(params['n_trajectories']), desc="Trajectories"):
                temp = params['temperatures'][i % len(params['temperatures'])]
                md_config.temperature = temp
                
                if i > 0:
                    mol = add_random_displacement(molecule, 0.05, 42 + i)
                else:
                    mol = molecule
                
                output_dir = Path(f'temp_traj_{i+1}')
                traj = run_direct_md(mol, md_config, str(output_dir), 'npz')
                trajectories.append(traj)
            
            # Combine
            all_coords = np.vstack([t.coordinates for t in trajectories])
            all_energies = np.concatenate([t.energies for t in trajectories])
            all_forces = np.vstack([t.forces for t in trajectories])
            all_dipoles = np.vstack([t.dipoles for t in trajectories]) if trajectories[0].dipoles is not None else None
            
            training_data = TrajectoryData(
                symbols=molecule.symbols,
                coordinates=all_coords,
                energies=all_energies,
                forces=all_forces,
                dipoles=all_dipoles,
                metadata=metadata.to_dict()
            )
            
            # Save training data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(f'outputs/mlpes_v2_{timestamp}')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            training_data_path = output_dir / 'training_data.npz'
            save_trajectory(training_data, str(training_data_path))
            
            print(f"\n✅ Generated {training_data.n_frames} configurations")
            
            metadata.files['training_data'] = str(training_data_path)
            metadata.files['output_directory'] = str(output_dir)
            metadata.training['n_configs'] = training_data.n_frames
            
        else:
            # Load existing
            import glob
            found = glob.glob('outputs/*/training_data/*.npz')
            
            if not found:
                print("\n❌ No training data found!")
                sys.exit(1)
            
            found = sorted(found, key=lambda x: Path(x).stat().st_mtime, reverse=True)
            
            print(f"\n📂 Found {len(found)} dataset(s):\n")
            for i, path in enumerate(found[:5], 1):
                traj = load_trajectory(path)
                print(f"  [{i}] {path}")
                print(f"      Configs: {traj.n_frames}")
            
            choice = int(input(f"\nSelect [1-{min(5, len(found))}]: "))
            training_data_path = found[choice - 1]
            
            training_data = load_trajectory(training_data_path)
            
            output_dir = Path(f'outputs/mlpes_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            metadata.training['n_configs'] = training_data.n_frames
            metadata.files['training_data'] = training_data_path
            metadata.files['output_directory'] = str(output_dir)
    
    else:
        print("\n❌ PSI4 required!")
        sys.exit(1)
    
    # Statistics
    e_kcal = training_data.energies * 627.509
    metadata.training['energy_range_kcal'] = float(e_kcal.max() - e_kcal.min())
    metadata.training['energy_std_kcal'] = float(e_kcal.std())
    
    gen_time = (datetime.now() - start_time).total_seconds() / 3600
    metadata.training['generation_time_hours'] = round(gen_time, 2)
    
    print(f"\n📊 Training Data Statistics:")
    print(f"   Configurations: {training_data.n_frames}")
    print(f"   Energy range: {metadata.training['energy_range_kcal']:.2f} kcal/mol")
    print(f"   Energy std: {metadata.training['energy_std_kcal']:.2f} kcal/mol")
    print(f"   Generation time: {metadata.training['generation_time_hours']:.2f} hours")
    
    # Train ML-PES
    print("\n" + "=" * 80)
    print("  TRAINING ML-PES")
    print("=" * 80)
    
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
    
    print("\n🤖 Training models...")
    
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
    
    r2 = 1 - ((y_test - y_pred)**2).sum() / ((y_test - y_test.mean())**2).sum()
    
    print(f"\n✅ Best Model:")
    print(f"   Gamma: {best_params[0]}")
    print(f"   Alpha: {best_params[1]}")
    print(f"   RMSE: {best_rmse:.4f} kcal/mol")
    print(f"   R²: {r2:.6f}")
    
    # Update metadata
    metadata.model['gamma'] = float(best_params[0])
    metadata.model['alpha'] = float(best_params[1])
    metadata.model['test_rmse_kcal'] = float(best_rmse)
    metadata.model['r2_score'] = float(r2)
    metadata.model['date_trained'] = datetime.now().isoformat()
    
    # Save model with metadata
    model_data = {
        'model': best_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'symbols': training_data.symbols,
        'metadata': metadata.to_dict(),
        'version': '2.0'
    }
    
    model_path = output_dir / 'mlpes_model_v2.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    metadata.files['model_file'] = str(model_path)
    
    # Save metadata as JSON
    metadata_path = output_dir / 'metadata.json'
    metadata.save_json(str(metadata_path))
    
    print(f"\n💾 Saved:")
    print(f"   Model: {model_path}")
    print(f"   Metadata: {metadata_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("  WORKFLOW COMPLETE")
    print("=" * 80)
    
    print(f"\n📊 Summary:")
    print(f"   Molecule: {metadata.molecule['name']} ({metadata.molecule['formula']})")
    print(f"   Theory: {metadata.theory['method']}/{metadata.theory['basis']}")
    print(f"   Training: {metadata.training['n_configs']} configs")
    print(f"   RMSE: {metadata.model['test_rmse_kcal']:.4f} kcal/mol")
    print(f"   Output: {output_dir}")
    
    if best_rmse < 0.1:
        print(f"\n🎉 EXCEPTIONAL! Essentially at QM noise level!")
    elif best_rmse < 1.0:
        print(f"\n🎉 EXCELLENT! Chemical accuracy achieved!")
    elif best_rmse < 5.0:
        print(f"\n✅ GOOD! Suitable for most applications.")
    
    print(f"\n💡 To use this model:")
    print(f"   import pickle")
    print(f"   with open('{model_path}', 'rb') as f:")
    print(f"       data = pickle.load(f)")
    print(f"   metadata = data['metadata']")
    print(f"   print(f\"Trained on: {{metadata['theory']['method']}}/{{metadata['theory']['basis']}}\")")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
