#!/usr/bin/env python3
"""
Interactive ML-PES Training and Prediction Workflow

Features:
- Interactive menu to select training data
- Train on energies (working) or energies+forces (experimental)
- Prediction workflow after training
- Progress tracking with tqdm

Author: PSI4-MD Framework
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import glob
import pickle

# Progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("Installing tqdm for progress bars...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
    from tqdm import tqdm
    TQDM_AVAILABLE = True

from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from itertools import product

print("=" * 80)
print("  INTERACTIVE ML-PES TRAINING & PREDICTION")
print("=" * 80)

# Import framework
try:
    from modules.test_molecules import get_molecule
    from modules.direct_md import DirectMDConfig, run_direct_md
    from modules.data_formats import load_trajectory, TrajectoryData
    print("✅ Framework imported\n")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# ==============================================================================
# DESCRIPTOR FUNCTIONS
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
# INTERACTIVE MENU FOR DATA SELECTION
# ==============================================================================

def select_training_data():
    """Interactive menu to select training data."""
    print("=" * 80)
    print("  SELECT TRAINING DATA")
    print("=" * 80)
    
    # Search for available training data
    patterns = [
        'outputs/*/training_data/*.npz',
        'outputs/*/training_data/filtered_training_data.npz',
        'outputs/*/training_data/combined_training_data.npz'
    ]
    
    found_files = []
    for pattern in patterns:
        found_files.extend(glob.glob(pattern))
    
    # Remove duplicates and sort by modification time (newest first)
    found_files = list(set(found_files))
    found_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    
    if not found_files:
        print("\n❌ No training data found in outputs/ directories!")
        print("\n💡 Generate training data first using:")
        print("   python3 simple_production_workflow.py")
        sys.exit(1)
    
    # Display options
    print(f"\n📂 Found {len(found_files)} training dataset(s):\n")
    
    for i, filepath in enumerate(found_files, 1):
        path = Path(filepath)
        
        # Get file info
        size_mb = path.stat().st_size / 1024 / 1024
        mod_time = datetime.fromtimestamp(path.stat().st_mtime)
        
        # Try to load to get info
        try:
            traj = load_trajectory(str(path))
            n_configs = traj.n_frames
            n_atoms = len(traj.symbols)
            energy_range = (traj.energies.max() - traj.energies.min()) * 627.509
            
            print(f"  [{i}] {path}")
            print(f"      Configurations: {n_configs}")
            print(f"      Atoms: {n_atoms}")
            print(f"      Energy range: {energy_range:.2f} kcal/mol")
            print(f"      Modified: {mod_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"      Size: {size_mb:.1f} MB")
            print()
        except Exception as e:
            print(f"  [{i}] {path}")
            print(f"      Size: {size_mb:.1f} MB")
            print(f"      Modified: {mod_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"      ⚠️  Could not read: {e}")
            print()
    
    # Get user selection
    while True:
        try:
            choice = input(f"Select dataset [1-{len(found_files)}] (or 'q' to quit): ").strip()
            
            if choice.lower() == 'q':
                print("Exiting...")
                sys.exit(0)
            
            choice = int(choice)
            if 1 <= choice <= len(found_files):
                selected_file = found_files[choice - 1]
                print(f"\n✅ Selected: {selected_file}\n")
                return selected_file
            else:
                print(f"❌ Please enter a number between 1 and {len(found_files)}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================

def get_training_config():
    """Interactive menu for training configuration."""
    print("=" * 80)
    print("  TRAINING CONFIGURATION")
    print("=" * 80)
    
    print("\n📊 What should the model train on?\n")
    print("  [1] Energy only (RECOMMENDED)")
    print("      ✅ Fast, reliable, proven to work (0.037 kcal/mol RMSE)")
    print("      ✅ Uses energies from PSI4 calculations")
    print()
    print("  [2] Energy + Forces (EXPERIMENTAL)")
    print("      ⚠️  More complex, may not converge well")
    print("      ⚠️  Can improve accuracy if data quality is very high")
    print("      Uses both energies AND gradients from PSI4")
    print()
    
    while True:
        try:
            choice = input("Select training mode [1-2]: ").strip()
            
            if choice == '1':
                train_forces = False
                print("\n✅ Training on energies only")
                break
            elif choice == '2':
                train_forces = True
                print("\n⚠️  Training on energies + forces (experimental)")
                print("   Note: This may not work as well as energy-only training")
                break
            else:
                print("❌ Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)
    
    # Hyperparameter tuning
    print("\n🔧 Hyperparameter tuning:\n")
    print("  [1] Auto-tune (recommended) - tries multiple gamma/alpha values")
    print("  [2] Fast (use defaults) - gamma=0.1, alpha=0.01")
    print()
    
    while True:
        try:
            choice = input("Select option [1-2]: ").strip()
            
            if choice == '1':
                tune = True
                gamma_range = [0.001, 0.01, 0.1, 1.0]
                alpha_range = [0.01, 0.1, 1.0]
                print(f"\n✅ Will test {len(gamma_range)*len(alpha_range)} combinations")
                break
            elif choice == '2':
                tune = False
                gamma_range = [0.1]
                alpha_range = [0.01]
                print("\n✅ Using defaults: gamma=0.1, alpha=0.01")
                break
            else:
                print("❌ Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)
    
    return {
        'train_forces': train_forces,
        'tune': tune,
        'gamma_range': gamma_range,
        'alpha_range': alpha_range
    }

# ==============================================================================
# TRAINING FUNCTION
# ==============================================================================

def train_mlpes(training_data, config):
    """Train ML-PES model."""
    print("\n" + "=" * 80)
    print("  TRAINING ML-PES")
    print("=" * 80)
    
    # Compute descriptors
    print(f"\n🔧 Computing descriptors for {training_data.n_frames} configurations...")
    
    X = []
    for i in tqdm(range(training_data.n_frames), desc="Descriptors"):
        desc = compute_coulomb_matrix(training_data.symbols, training_data.coordinates[i])
        X.append(desc)
    X = np.array(X)
    
    y_energy = training_data.energies
    
    # Handle forces if requested
    if config['train_forces'] and training_data.forces is not None:
        print("\n🔧 Preparing force data...")
        
        # Flatten forces: (n_frames, n_atoms, 3) -> (n_frames, n_atoms*3)
        y_forces = training_data.forces.reshape(training_data.n_frames, -1)
        
        # Combine energies and forces
        # Scale forces to same magnitude as energies
        force_scale = np.abs(y_energy).mean() / np.abs(y_forces).mean()
        y_forces_scaled = y_forces * force_scale
        
        # Concatenate
        y = np.hstack([y_energy.reshape(-1, 1), y_forces_scaled])
        
        print(f"   Training on {y.shape[1]} outputs (1 energy + {y_forces.shape[1]} force components)")
    else:
        y = y_energy
        print(f"   Training on energies only")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Data split:")
    print(f"   Training: {len(X_train)} configs")
    print(f"   Test: {len(X_test)} configs")
    
    # Scale
    print(f"\n🔧 Scaling data...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    if config['train_forces']:
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
    else:
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Train
    print(f"\n🤖 Training model...")
    
    param_grid = list(product(config['gamma_range'], config['alpha_range']))
    
    best_rmse = float('inf')
    best_model = None
    best_params = None
    results = []
    
    for gamma, alpha in tqdm(param_grid, desc="Hyperparameters"):
        model = KernelRidge(kernel='rbf', gamma=gamma, alpha=alpha)
        model.fit(X_train_scaled, y_train_scaled)
        
        y_pred_scaled = model.predict(X_test_scaled)
        
        if config['train_forces']:
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_test_unscaled = scaler_y.inverse_transform(y_test_scaled)
            
            # Extract energy predictions
            y_pred_energy = y_pred[:, 0]
            y_test_energy = y_test_unscaled[:, 0]
        else:
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_pred_energy = y_pred
            y_test_energy = y_test
        
        # Calculate RMSE on energies
        errors = (y_pred_energy - y_test_energy) * 627.509
        rmse = np.sqrt((errors**2).mean())
        mae = np.abs(errors).mean()
        
        results.append({
            'gamma': gamma,
            'alpha': alpha,
            'rmse': rmse,
            'mae': mae
        })
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = (gamma, alpha)
    
    # Display results
    print(f"\n📊 Results:")
    if len(results) > 1:
        print(f"   {'Gamma':<10} {'Alpha':<10} {'RMSE (kcal/mol)':<20}")
        print(f"   {'-' * 50}")
        for r in sorted(results, key=lambda x: x['rmse'])[:min(5, len(results))]:
            marker = '⭐' if r['rmse'] == best_rmse else '  '
            print(f"   {marker} {r['gamma']:<8.4f} {r['alpha']:<8.2f}   {r['rmse']:>8.4f}")
    
    print(f"\n✅ Best Model:")
    print(f"   Gamma: {best_params[0]}")
    print(f"   Alpha: {best_params[1]}")
    print(f"   RMSE: {best_rmse:.4f} kcal/mol")
    
    # Return model data
    return {
        'model': best_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'symbols': training_data.symbols,
        'gamma': best_params[0],
        'alpha': best_params[1],
        'rmse': best_rmse,
        'train_forces': config['train_forces'],
        'n_train': len(X_train),
        'n_test': len(X_test),
        'all_results': results,
        'metadata': training_data.metadata if hasattr(training_data, 'metadata') else {}
    }

# ==============================================================================
# PREDICTION WORKFLOW
# ==============================================================================

def prediction_workflow(model_data):
    """Interactive prediction workflow."""
    print("\n" + "=" * 80)
    print("  PREDICTION WORKFLOW")
    print("=" * 80)
    
    print(f"\n📊 Model Info:")
    print(f"   RMSE: {model_data['rmse']:.4f} kcal/mol")
    print(f"   Gamma: {model_data['gamma']}")
    print(f"   Alpha: {model_data['alpha']}")
    print(f"   Training: {model_data['train_forces'] and 'Energy + Forces' or 'Energy only'}")
    
    while True:
        print(f"\n" + "=" * 80)
        print("  PREDICTION OPTIONS")
        print("=" * 80)
        print("\n  [1] Predict single geometry (manual input)")
        print("  [2] Predict from trajectory file")
        print("  [3] Generate test trajectory with PSI4 and compare")
        print("  [4] Save model and exit")
        print("  [q] Quit")
        
        try:
            choice = input("\nSelect option: ").strip().lower()
            
            if choice == 'q' or choice == '4':
                break
            
            elif choice == '1':
                predict_single_geometry(model_data)
            
            elif choice == '2':
                predict_from_trajectory(model_data)
            
            elif choice == '3':
                predict_with_psi4_comparison(model_data)
            
            else:
                print("❌ Invalid choice")
        
        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def predict_single_geometry(model_data):
    """Predict energy for manually entered geometry."""
    print("\n📍 Enter coordinates (Angstrom):")
    print(f"   Molecule: {' '.join(model_data['symbols'])}")
    print(f"   Format: x y z (one atom per line)")
    print(f"   Type 'example' for water equilibrium geometry")
    print()
    
    coords = []
    for i, symbol in enumerate(model_data['symbols']):
        while True:
            try:
                line = input(f"   {symbol} [{i+1}]: ").strip()
                
                if line.lower() == 'example' and i == 0:
                    # Water equilibrium
                    coords = np.array([
                        [0.0, 0.0, 0.0],
                        [0.0, 0.757, 0.586],
                        [0.0, -0.757, 0.586]
                    ])
                    print("   Using example water geometry")
                    break
                
                x, y, z = map(float, line.split())
                coords.append([x, y, z])
                break
            except ValueError:
                print("      ❌ Invalid format. Enter: x y z")
            except KeyboardInterrupt:
                print("\n")
                return
    
    if len(coords) == len(model_data['symbols']):
        coords = np.array(coords)
    
    # Predict
    desc = compute_coulomb_matrix(model_data['symbols'], coords)
    desc_scaled = model_data['scaler_X'].transform([desc])
    
    if model_data['train_forces']:
        pred_scaled = model_data['model'].predict(desc_scaled)
        pred = model_data['scaler_y'].inverse_transform(pred_scaled)
        energy = pred[0, 0]
    else:
        e_scaled = model_data['model'].predict(desc_scaled)
        energy = model_data['scaler_y'].inverse_transform([[e_scaled[0]]])[0, 0]
    
    print(f"\n✅ Predicted Energy:")
    print(f"   {energy:.8f} Hartree")
    print(f"   {energy * 627.509:.4f} kcal/mol")
    print(f"   {energy * 27.2114:.4f} eV")

def predict_from_trajectory(model_data):
    """Predict energies from trajectory file."""
    print("\n📂 Enter trajectory file path:")
    print("   (or drag file here)")
    
    try:
        filepath = input("   Path: ").strip().strip("'\"")
        
        if not Path(filepath).exists():
            print(f"   ❌ File not found: {filepath}")
            return
        
        print(f"\n   Loading {filepath}...")
        traj = load_trajectory(filepath)
        
        print(f"   Loaded {traj.n_frames} frames")
        
        # Predict
        print(f"\n🤖 Predicting energies...")
        
        X = []
        for i in tqdm(range(traj.n_frames), desc="Computing descriptors"):
            desc = compute_coulomb_matrix(traj.symbols, traj.coordinates[i])
            X.append(desc)
        X = np.array(X)
        
        X_scaled = model_data['scaler_X'].transform(X)
        
        if model_data['train_forces']:
            pred_scaled = model_data['model'].predict(X_scaled)
            pred = model_data['scaler_y'].inverse_transform(pred_scaled)
            ml_energies = pred[:, 0]
        else:
            e_scaled = model_data['model'].predict(X_scaled)
            ml_energies = model_data['scaler_y'].inverse_transform(e_scaled.reshape(-1, 1)).flatten()
        
        print(f"\n✅ Predictions complete!")
        print(f"   Mean energy: {ml_energies.mean():.6f} Ha ({ml_energies.mean()*627.509:.2f} kcal/mol)")
        print(f"   Energy range: {ml_energies.min():.6f} to {ml_energies.max():.6f} Ha")
        print(f"   Span: {(ml_energies.max()-ml_energies.min())*627.509:.2f} kcal/mol")
        
        # Compare if reference energies available
        if traj.energies is not None:
            errors = (ml_energies - traj.energies) * 627.509
            rmse = np.sqrt((errors**2).mean())
            mae = np.abs(errors).mean()
            
            print(f"\n📊 Comparison with reference:")
            print(f"   RMSE: {rmse:.4f} kcal/mol")
            print(f"   MAE: {mae:.4f} kcal/mol")
            print(f"   Max error: {np.abs(errors).max():.4f} kcal/mol")
        
    except KeyboardInterrupt:
        print("\n")
        return
    except Exception as e:
        print(f"\n❌ Error: {e}")

def predict_with_psi4_comparison(model_data):
    """Generate test trajectory and compare."""
    try:
        import psi4
    except ImportError:
        print("\n❌ PSI4 not available")
        return
    
    print("\n🔬 Generating test trajectory with PSI4...")
    
    # Detect theory level from model metadata
    if 'metadata' in model_data:
        method = model_data['metadata'].get('method', 'HF')
        basis = model_data['metadata'].get('basis', '6-31G*')
    else:
        # Ask user
        print("\n⚠️  CRITICAL: Must use same theory level as training data!")
        print("\n   Common training data theory levels:")
        print("   - outputs/improved_mlpes_20251230_183101: B3LYP/6-31G*")
        print("   - outputs/improved_mlpes_20251230_180012: HF/6-31G*")
        print()
        method = input("   Enter method [HF/B3LYP/MP2]: ").strip() or 'B3LYP'
        basis = input("   Enter basis [6-31G*/STO-3G]: ").strip() or '6-31G*'
    
    print(f"\n   Using: {method}/{basis}")
    print("   ⚠️  This MUST match your training data theory level!")
    
    print("\n   Enter number of MD steps [default: 20]: ", end='')
    
    try:
        n_steps_input = input().strip()
        n_steps = int(n_steps_input) if n_steps_input else 20
    except:
        n_steps = 20
    
    # Create test molecule
    molecule = get_molecule('water')  # Assume water for now
    
    config = DirectMDConfig(
        method=method,
        basis=basis,
        temperature=300,
        timestep=0.5,
        n_steps=n_steps,
        output_frequency=1
    )
    
    output_dir = Path('outputs/temp_test_traj')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"   Running {n_steps} steps...")
    test_traj = run_direct_md(molecule, config, str(output_dir), 'npz')
    
    # Predict with ML
    print(f"\n🤖 Predicting with ML-PES...")
    
    X = []
    for i in range(test_traj.n_frames):
        desc = compute_coulomb_matrix(test_traj.symbols, test_traj.coordinates[i])
        X.append(desc)
    X = np.array(X)
    
    X_scaled = model_data['scaler_X'].transform(X)
    
    if model_data['train_forces']:
        pred_scaled = model_data['model'].predict(X_scaled)
        pred = model_data['scaler_y'].inverse_transform(pred_scaled)
        ml_energies = pred[:, 0]
    else:
        e_scaled = model_data['model'].predict(X_scaled)
        ml_energies = model_data['scaler_y'].inverse_transform(e_scaled.reshape(-1, 1)).flatten()
    
    psi4_energies = test_traj.energies
    
    # Compare
    errors = (ml_energies - psi4_energies) * 627.509
    rmse = np.sqrt((errors**2).mean())
    mae = np.abs(errors).mean()
    
    print(f"\n📊 Comparison:")
    print(f"   RMSE: {rmse:.4f} kcal/mol")
    print(f"   MAE: {mae:.4f} kcal/mol")
    print(f"   Max error: {np.abs(errors).max():.4f} kcal/mol")
    
    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        times = np.arange(len(psi4_energies)) * config.timestep
        
        ax1.plot(times, psi4_energies*627.509, 'o-', label='PSI4', lw=2, ms=6)
        ax1.plot(times, ml_energies*627.509, 's--', label='ML-PES', lw=2, ms=5, alpha=0.7)
        ax1.set_xlabel('Time (fs)')
        ax1.set_ylabel('Energy (kcal/mol)')
        ax1.set_title(f'PSI4 vs ML-PES (RMSE: {rmse:.4f} kcal/mol)')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2.plot(times, errors, 'o-', color='red', lw=2, ms=6)
        ax2.axhline(0, color='black', ls='--', alpha=0.5)
        ax2.fill_between(times, errors, alpha=0.3, color='red')
        ax2.set_xlabel('Time (fs)')
        ax2.set_ylabel('Error (kcal/mol)')
        ax2.set_title('Prediction Error')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plot_path = 'outputs/test_comparison.png'
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        print(f"\n✅ Plot saved: {plot_path}")
        
    except Exception as e:
        print(f"\n⚠️  Could not create plot: {e}")

# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

def main():
    """Main interactive workflow."""
    
    # Step 1: Select training data
    data_path = select_training_data()
    
    # Load data
    print(f"📂 Loading training data...")
    training_data = load_trajectory(data_path)
    
    e_kcal = training_data.energies * 627.509
    print(f"\n📊 Dataset Statistics:")
    print(f"   Configurations: {training_data.n_frames}")
    print(f"   Atoms: {len(training_data.symbols)} ({' '.join(training_data.symbols)})")
    print(f"   Energy range: {e_kcal.min():.2f} to {e_kcal.max():.2f} kcal/mol")
    print(f"   Energy span: {e_kcal.max() - e_kcal.min():.2f} kcal/mol")
    print(f"   Has forces: {'Yes' if training_data.forces is not None else 'No'}")
    
    # Step 2: Get training configuration
    config = get_training_config()
    
    # Step 3: Train model
    model_data = train_mlpes(training_data, config)
    
    # Step 4: Prediction workflow
    prediction_workflow(model_data)
    
    # Step 5: Save model
    print("\n" + "=" * 80)
    print("  SAVE MODEL")
    print("=" * 80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_path = f'outputs/mlpes_model_{timestamp}.pkl'
    
    print(f"\n💾 Save model to: [{default_path}]")
    save_path = input("   (press Enter for default, or enter custom path): ").strip()
    
    if not save_path:
        save_path = default_path
    
    # Create directory if needed
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ Model saved: {save_path}")
    
    print(f"\n💡 To use this model later:")
    print(f"   import pickle")
    print(f"   with open('{save_path}', 'rb') as f:")
    print(f"       model = pickle.load(f)")
    
    print("\n🎉 Workflow complete!")
    print("=" * 80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
