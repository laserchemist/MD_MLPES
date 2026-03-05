#!/usr/bin/env python3
"""
Complete Production ML-PES Workflow v2.1 - Enhanced

New in v2.1:
- ✅ Read molecules from .xyz files
- ✅ Load existing training data option
- ✅ Enhanced PSI4 calculation visualization
- ✅ Energy offset correction for mixed theory levels
- ✅ Progress plots during training generation

Author: PSI4-MD Framework
Version: 2.1
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import json
import matplotlib.pyplot as plt

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
print("  PRODUCTION ML-PES WORKFLOW v2.1 - ENHANCED")
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
    print(f"✅ scikit-learn {sklearn.__version__}\n")
except ImportError:
    print("❌ scikit-learn required")
    sys.exit(1)

# ==============================================================================
# THEORY LEVEL CORRECTIONS
# ==============================================================================

# Approximate energy offsets between theory levels (kcal/mol per heavy atom)
# These are rough estimates - should be refined for your specific system
THEORY_CORRECTIONS = {
    ('HF', 'STO-3G'): 0.0,  # Reference
    ('HF', '6-31G*'): -8.0,
    ('B3LYP', '6-31G*'): -50.0,
    ('B3LYP', 'cc-pVDZ'): -52.0,
    ('MP2', 'cc-pVDZ'): -60.0,
    ('CCSD(T)', 'cc-pVTZ'): -80.0,
}

def estimate_energy_offset(method1, basis1, method2, basis2, n_heavy_atoms):
    """
    Estimate energy offset between two theory levels.
    
    Returns offset in Hartree (add to method1 energies to match method2 scale)
    """
    key1 = (method1, basis1)
    key2 = (method2, basis2)
    
    offset1 = THEORY_CORRECTIONS.get(key1, 0.0)
    offset2 = THEORY_CORRECTIONS.get(key2, 0.0)
    
    # Difference in kcal/mol per heavy atom
    diff_per_atom = offset2 - offset1
    
    # Total offset in kcal/mol
    total_offset_kcal = diff_per_atom * n_heavy_atoms
    
    # Convert to Hartree
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
            'source': None  # 'library', 'xyz_file', 'custom'
        }
        self.training = {
            'n_configs': 0,
            'n_trajectories': 0,
            'temperatures': [],
            'energy_range_kcal': 0.0,
            'energy_std_kcal': 0.0,
            'training_type': 'energy_only',
            'date_created': None,
            'generation_time_hours': 0.0,
            'source': None  # 'generated', 'loaded'
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
            'output_directory': None,
            'visualizations': []
        }
        self.adaptive = {
            'used': False,
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
            'version': '2.1'
        }
    
    def save_json(self, filepath: str):
        """Save metadata as JSON."""
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
    """
    Read molecule from XYZ file.
    
    XYZ format:
    <number of atoms>
    <comment line>
    <element> <x> <y> <z>
    ...
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse
    n_atoms = int(lines[0].strip())
    comment = lines[1].strip()
    
    symbols = []
    coords = []
    
    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    coords = np.array(coords)
    
    # Create molecule object
    class XYZMolecule:
        def __init__(self, symbols, coords, name, comment):
            self.symbols = symbols
            self.coordinates = coords
            self.name = name
            self.comment = comment
            
            # Generate formula
            from collections import Counter
            counts = Counter(symbols)
            self.formula = ''.join([f"{s}{c if c > 1 else ''}" 
                                   for s, c in sorted(counts.items())])
            
            self.charge = 0
            self.multiplicity = 1
    
    # Extract name from filename
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
# VISUALIZATION
# ==============================================================================

def plot_trajectory_energies(trajectories, output_dir, method, basis):
    """Plot energies from generated trajectories."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Combined plot
    ax1 = axes[0]
    all_energies = []
    
    for i, traj in enumerate(trajectories):
        times = np.arange(traj.n_frames) * 0.5  # Assume 0.5 fs timestep
        energies_kcal = (traj.energies - traj.energies[0]) * 627.509
        
        ax1.plot(times, energies_kcal, 'o-', label=f'Traj {i+1}', alpha=0.7, markersize=3)
        all_energies.extend(energies_kcal)
    
    ax1.set_xlabel('Time (fs)', fontsize=12)
    ax1.set_ylabel('Relative Energy (kcal/mol)', fontsize=12)
    ax1.set_title(f'MD Trajectories - {method}/{basis}', fontsize=14)
    ax1.legend(fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Energy distribution
    ax2 = axes[1]
    ax2.hist(all_energies, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Relative Energy (kcal/mol)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Energy Distribution', fontsize=14)
    ax2.axvline(np.mean(all_energies), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_energies):.2f} kcal/mol', linewidth=2)
    ax2.axvline(np.median(all_energies), color='green', linestyle='--',
                label=f'Median: {np.median(all_energies):.2f} kcal/mol', linewidth=2)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / 'trajectory_energies.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved plot: {plot_path}")
    
    return str(plot_path)

def plot_training_results(y_test, y_pred, rmse, output_dir):
    """Plot ML-PES training results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Parity plot
    ax1 = axes[0]
    y_test_kcal = y_test * 627.509
    y_pred_kcal = y_pred * 627.509
    
    ax1.scatter(y_test_kcal, y_pred_kcal, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_test_kcal.min(), y_pred_kcal.min())
    max_val = max(y_test_kcal.max(), y_pred_kcal.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    ax1.set_xlabel('PSI4 Energy (kcal/mol)', fontsize=12)
    ax1.set_ylabel('ML-PES Energy (kcal/mol)', fontsize=12)
    ax1.set_title(f'Parity Plot (RMSE: {rmse:.4f} kcal/mol)', fontsize=14)
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
    
    # Add statistics
    stats_text = f'Mean: {errors.mean():.4f}\nStd: {errors.std():.4f}\nMax: {np.abs(errors).max():.4f}'
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    
    plot_path = output_dir / 'training_results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved plot: {plot_path}")
    
    return str(plot_path)

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
    print("  [2] Read from XYZ file")
    print("  [3] Define custom coordinates")
    
    choice = input("\nSelect [1-3]: ").strip()
    
    molecule = None
    source = None
    
    if choice == '1':
        # Test molecules
        if not AVAILABLE_MOLECULES:
            print("\n⚠️  No test molecules available")
            return select_molecule()
        
        print(f"\n📚 Available test molecules:\n")
        
        mol_list = [name for name, mol in AVAILABLE_MOLECULES.items() if mol is not None]
        
        if not mol_list:
            print("\n⚠️  No valid test molecules found")
            return select_molecule()
        
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
        source = 'library'
    
    elif choice == '2':
        # Read XYZ file
        print("\n📁 Enter path to XYZ file:")
        print("   (or drag file here)")
        
        xyz_path = input("   Path: ").strip().strip("'\"")
        
        if not Path(xyz_path).exists():
            print(f"\n❌ File not found: {xyz_path}")
            return select_molecule()
        
        try:
            molecule = read_xyz_file(xyz_path)
            source = 'xyz_file'
        except Exception as e:
            print(f"\n❌ Error reading XYZ file: {e}")
            return select_molecule()
    
    elif choice == '3':
        # Custom input
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
        
        # Create custom molecule
        class CustomMolecule:
            def __init__(self, symbols, coords):
                self.symbols = symbols
                self.coordinates = np.array(coords)
                self.name = 'custom'
                from collections import Counter
                counts = Counter(symbols)
                self.formula = ''.join([f"{s}{c if c > 1 else ''}" 
                                       for s, c in sorted(counts.items())])
                self.charge = 0
                self.multiplicity = 1
        
        molecule = CustomMolecule(symbols, coords)
        source = 'custom'
    
    else:
        print("\n❌ Invalid choice")
        return select_molecule()
    
    if molecule:
        print(f"\n✅ Selected: {molecule.name} ({molecule.formula})")
        print(f"   Atoms: {len(molecule.symbols)}")
        print(f"   Source: {source}")
    
    return molecule, source

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
    
    # Estimate cost
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
    """Main workflow with all enhancements."""
    
    # Initialize metadata
    metadata = TrainingMetadata()
    metadata.training['date_created'] = datetime.now().isoformat()
    start_time = datetime.now()
    
    # Step 1: Select molecule
    molecule, source = select_molecule()
    
    metadata.molecule['name'] = molecule.name
    metadata.molecule['formula'] = molecule.formula if hasattr(molecule, 'formula') else 'unknown'
    metadata.molecule['n_atoms'] = len(molecule.symbols)
    metadata.molecule['symbols'] = molecule.symbols
    metadata.molecule['source'] = source
    
    # Step 2: Select theory level
    method, basis = select_theory_level()
    
    metadata.theory['method'] = method
    metadata.theory['basis'] = basis
    
    # Detect functional
    dft_methods = ['B3LYP', 'PBE', 'wB97X-D', 'M06-2X']
    if method in dft_methods:
        metadata.theory['functional'] = method
    
    # Step 3: Generate or load data
    print("\n" + "=" * 80)
    print("  TRAINING DATA")
    print("=" * 80)
    
    print("\n📊 Options:")
    print("  [1] Generate new data with PSI4")
    print("  [2] Load existing training data")
    
    data_choice = input("\nSelect [1-2]: ").strip()
    
    training_data = None
    output_dir = Path(f'outputs/mlpes_v2.1_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if data_choice == '1':
        # Generate new data
        if not PSI4_AVAILABLE:
            print("\n❌ PSI4 required to generate data!")
            sys.exit(1)
        
        params = select_training_parameters()
        
        metadata.training['n_trajectories'] = params['n_trajectories']
        metadata.training['temperatures'] = params['temperatures']
        metadata.training['source'] = 'generated'
        
        print("\n" + "=" * 80)
        print("  GENERATING TRAINING DATA")
        print("=" * 80)
        
        print(f"\n🚀 Starting PSI4 calculations...")
        print(f"   Method: {method}/{basis}")
        print(f"   Molecule: {molecule.name}")
        print(f"   Trajectories: {params['n_trajectories']}")
        
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
            
            traj_dir = output_dir / f'traj_{i+1}_T{temp}K'
            traj = run_direct_md(mol, md_config, str(traj_dir), 'npz')
            trajectories.append(traj)
        
        # Visualize trajectories
        print(f"\n📊 Creating visualizations...")
        viz_path = plot_trajectory_energies(trajectories, output_dir, method, basis)
        metadata.files['visualizations'].append(viz_path)
        
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
        
        # Save
        training_data_path = output_dir / 'training_data.npz'
        save_trajectory(training_data, str(training_data_path))
        
        print(f"\n✅ Generated {training_data.n_frames} configurations")
        
        metadata.files['training_data'] = str(training_data_path)
        
    elif data_choice == '2':
        # Load existing data
        import glob
        found = glob.glob('outputs/*/training_data/*.npz') + glob.glob('outputs/*/*.npz')
        
        if not found:
            print("\n❌ No training data found!")
            print("   Generate training data first with option [1]")
            sys.exit(1)
        
        found = sorted(set(found), key=lambda x: Path(x).stat().st_mtime, reverse=True)
        
        print(f"\n📂 Found {len(found)} dataset(s):\n")
        
        for i, path in enumerate(found[:10], 1):
            try:
                traj = load_trajectory(path)
                e_range = (traj.energies.max() - traj.energies.min()) * 627.509
                mod_time = datetime.fromtimestamp(Path(path).stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                
                print(f"  [{i}] {path}")
                print(f"      Configs: {traj.n_frames}, Energy range: {e_range:.2f} kcal/mol")
                print(f"      Modified: {mod_time}")
                print()
            except:
                print(f"  [{i}] {path} (unable to read)")
        
        choice = int(input(f"Select dataset [1-{min(10, len(found))}]: "))
        training_data_path = found[choice - 1]
        
        print(f"\n📂 Loading: {training_data_path}")
        training_data = load_trajectory(training_data_path)
        
        metadata.training['source'] = 'loaded'
        metadata.files['training_data'] = training_data_path
        
        print(f"✅ Loaded {training_data.n_frames} configurations")
    
    else:
        print("\n❌ Invalid choice")
        sys.exit(1)
    
    # Update metadata
    metadata.files['output_directory'] = str(output_dir)
    metadata.training['n_configs'] = training_data.n_frames
    
    # Statistics
    e_kcal = training_data.energies * 627.509
    metadata.training['energy_range_kcal'] = float(e_kcal.max() - e_kcal.min())
    metadata.training['energy_std_kcal'] = float(e_kcal.std())
    
    if data_choice == '1':
        gen_time = (datetime.now() - start_time).total_seconds() / 3600
        metadata.training['generation_time_hours'] = round(gen_time, 2)
    
    print(f"\n📊 Training Data Statistics:")
    print(f"   Configurations: {training_data.n_frames}")
    print(f"   Energy range: {metadata.training['energy_range_kcal']:.2f} kcal/mol")
    print(f"   Energy std: {metadata.training['energy_std_kcal']:.2f} kcal/mol")
    
    # Train ML-PES
    print("\n" + "=" * 80)
    print("  TRAINING ML-PES")
    print("=" * 80)
    
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
    
    # Final prediction
    y_pred_scaled = best_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    r2 = 1 - ((y_test - y_pred)**2).sum() / ((y_test - y_test.mean())**2).sum()
    
    print(f"\n✅ Best Model:")
    print(f"   Gamma: {best_params[0]}")
    print(f"   Alpha: {best_params[1]}")
    print(f"   RMSE: {best_rmse:.4f} kcal/mol")
    print(f"   R²: {r2:.6f}")
    
    # Plot results
    print(f"\n📊 Creating training visualizations...")
    plot_path = plot_training_results(y_test, y_pred, best_rmse, output_dir)
    metadata.files['visualizations'].append(plot_path)
    
    # Update metadata
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
        'version': '2.1'
    }
    
    model_path = output_dir / 'mlpes_model_v2.1.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    metadata.files['model_file'] = str(model_path)
    
    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    metadata.save_json(str(metadata_path))
    
    print(f"\n💾 Saved:")
    print(f"   Model: {model_path}")
    print(f"   Metadata: {metadata_path}")
    print(f"   Visualizations: {len(metadata.files['visualizations'])} plots")
    
    # Summary
    print("\n" + "=" * 80)
    print("  WORKFLOW COMPLETE")
    print("=" * 80)
    
    print(f"\n📊 Summary:")
    print(f"   Molecule: {metadata.molecule['name']} ({metadata.molecule['formula']})")
    print(f"   Source: {metadata.molecule['source']}")
    print(f"   Theory: {metadata.theory['method']}/{metadata.theory['basis']}")
    print(f"   Training: {metadata.training['n_configs']} configs ({metadata.training['source']})")
    print(f"   RMSE: {metadata.model['test_rmse_kcal']:.4f} kcal/mol")
    print(f"   R²: {metadata.model['r2_score']:.6f}")
    print(f"   Output: {output_dir}")
    
    if best_rmse < 0.1:
        print(f"\n🎉 EXCEPTIONAL! Essentially at QM noise level!")
    elif best_rmse < 1.0:
        print(f"\n🎉 EXCELLENT! Chemical accuracy achieved!")
    elif best_rmse < 5.0:
        print(f"\n✅ GOOD! Suitable for most applications.")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
