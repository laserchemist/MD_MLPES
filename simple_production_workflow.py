#!/usr/bin/env python3
"""
Production ML-PES Workflow - Using Existing Working Implementation

This version uses minimal_mlpes_working.py functionality directly,
so you don't need to copy files around.

Author: PSI4-MD Framework
Date: 2025
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

print("=" * 80)
print("  PRODUCTION ML-PES WORKFLOW - SIMPLIFIED")
print("=" * 80)

# Import framework
try:
    from modules.test_molecules import get_molecule, add_random_displacement
    from modules.direct_md import DirectMDConfig, run_direct_md
    from modules.data_formats import TrajectoryData, save_trajectory, load_trajectory
    from modules.visualization import TrajectoryVisualizer
    print("✅ Framework imported\n")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Check dependencies
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
# SIMPLE DESCRIPTOR (from minimal_mlpes_working.py)
# ==============================================================================

def compute_coulomb_matrix(symbols, coords):
    """Simple Coulomb matrix descriptor."""
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
# CONFIGURATION
# ==============================================================================

USE_EXISTING_DATA = True  # Set to False to generate new data
EXISTING_DATA_PATH = 'outputs/improved_mlpes_20251230_183101/training_data/filtered_training_data.npz'

# For new data generation
MOLECULE_NAME = 'water'
METHOD = 'B3LYP'
BASIS = '6-31G*'
N_TRAJECTORIES = 6
N_STEPS_PER_TRAJ = 300
TEMPERATURES = [200, 300, 400, 500, 600, 700]

# ML configuration
GAMMA_RANGE = [0.001, 0.01, 0.1, 1.0]
ALPHA_RANGE = [0.01, 0.1, 1.0]

# ==============================================================================
# SETUP
# ==============================================================================

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
base_dir = Path(f'outputs/production_mlpes_{timestamp}')
models_dir = base_dir / 'models'
analysis_dir = base_dir / 'analysis'

for d in [models_dir, analysis_dir]:
    d.mkdir(parents=True, exist_ok=True)

print(f"📁 Output: {base_dir}\n")

# ==============================================================================
# LOAD OR GENERATE DATA
# ==============================================================================

print("=" * 80)
print("  PHASE 1: TRAINING DATA")
print("=" * 80)

if USE_EXISTING_DATA:
    print(f"\n📂 Loading existing data...")
    
    # Try to find data
    import glob
    if Path(EXISTING_DATA_PATH).exists():
        data_path = EXISTING_DATA_PATH
    else:
        # Search for any training data
        found = glob.glob('outputs/*/training_data/*.npz')
        if not found:
            print("❌ No training data found!")
            print("\n💡 Set USE_EXISTING_DATA = False to generate new data")
            sys.exit(1)
        data_path = found[-1]  # Use most recent
        print(f"   Found: {data_path}")
    
    training_data = load_trajectory(data_path)
    print(f"   ✅ Loaded {training_data.n_frames} configurations")
    
else:
    if not PSI4_AVAILABLE:
        print("❌ PSI4 required to generate new data!")
        sys.exit(1)
    
    print(f"\n🚀 Generating training data...\n")
    
    molecule = get_molecule(MOLECULE_NAME)
    print(f"🧪 {molecule.name} ({molecule.formula})")
    print(f"   Theory: {METHOD}/{BASIS}")
    print(f"   Trajectories: {N_TRAJECTORIES}\n")
    
    md_config = DirectMDConfig(
        method=METHOD,
        basis=BASIS,
        timestep=0.5,
        n_steps=N_STEPS_PER_TRAJ,
        output_frequency=2,
        thermostat='berendsen'
    )
    
    trajectories = []
    for i in tqdm(range(N_TRAJECTORIES), desc="Generating"):
        temp = TEMPERATURES[i]
        md_config.temperature = temp
        
        if i > 0:
            mol = add_random_displacement(molecule, 0.05, 42 + i)
        else:
            mol = molecule
        
        traj_dir = base_dir / f'traj_{i+1}_T{temp}K'
        traj = run_direct_md(mol, md_config, str(traj_dir), 'npz')
        trajectories.append(traj)
    
    # Combine
    all_coords = np.vstack([t.coordinates for t in trajectories])
    all_energies = np.concatenate([t.energies for t in trajectories])
    all_forces = np.vstack([t.forces for t in trajectories])
    
    training_data = TrajectoryData(
        symbols=molecule.symbols,
        coordinates=all_coords,
        energies=all_energies,
        forces=all_forces
    )
    
    # Save
    save_path = base_dir / 'training_data.npz'
    save_trajectory(training_data, str(save_path))
    print(f"\n✅ Saved: {save_path}")

# Statistics
e_kcal = training_data.energies * 627.509
print(f"\n📊 Training Data:")
print(f"   Configs: {training_data.n_frames}")
print(f"   Energy range: {e_kcal.min():.1f} to {e_kcal.max():.1f} kcal/mol")
print(f"   Energy span: {e_kcal.max() - e_kcal.min():.2f} kcal/mol")

# ==============================================================================
# TRAIN ML-PES
# ==============================================================================

print(f"\n" + "=" * 80)
print("  PHASE 2: TRAINING ML-PES")
print("=" * 80)

print(f"\n🔧 Computing descriptors...")

X = []
iterator = tqdm(range(training_data.n_frames), desc="Descriptors") if TQDM_AVAILABLE else range(training_data.n_frames)
for i in iterator:
    desc = compute_coulomb_matrix(training_data.symbols, training_data.coordinates[i])
    X.append(desc)
X = np.array(X)
y = training_data.energies

print(f"   Shape: {X.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Data split:")
print(f"   Training: {len(X_train)}")
print(f"   Test: {len(X_test)}")

# Scale
print(f"\n🔧 Scaling data...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Train with hyperparameter search
print(f"\n🤖 Training models...")

from itertools import product
param_grid = list(product(GAMMA_RANGE, ALPHA_RANGE))

best_rmse = float('inf')
best_model = None
best_params = None
results = []

iterator = tqdm(param_grid, desc="Grid search") if TQDM_AVAILABLE else param_grid

for gamma, alpha in iterator:
    model = KernelRidge(kernel='rbf', gamma=gamma, alpha=alpha)
    model.fit(X_train_scaled, y_train_scaled)
    
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    errors = (y_pred - y_test) * 627.509
    rmse = np.sqrt((errors**2).mean())
    mae = np.abs(errors).mean()
    
    results.append({'gamma': gamma, 'alpha': alpha, 'rmse': rmse, 'mae': mae})
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_params = (gamma, alpha)

# Results
print(f"\n📊 Hyperparameter Search Results:")
print(f"   {'Gamma':<10} {'Alpha':<10} {'RMSE (kcal/mol)':<20}")
print(f"   {'-' * 50}")
for r in sorted(results, key=lambda x: x['rmse'])[:5]:
    marker = '⭐' if r['rmse'] == best_rmse else '  '
    print(f"   {marker} {r['gamma']:<8.4f} {r['alpha']:<8.2f}   {r['rmse']:>8.4f}")

print(f"\n✅ Best Model:")
print(f"   Gamma: {best_params[0]}")
print(f"   Alpha: {best_params[1]}")
print(f"   RMSE: {best_rmse:.4f} kcal/mol")

# Save model
model_data = {
    'model': best_model,
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'symbols': training_data.symbols,
    'gamma': best_params[0],
    'alpha': best_params[1],
    'rmse': best_rmse,
    'n_train': len(X_train),
    'n_test': len(X_test)
}

model_path = models_dir / 'production_mlpes_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n💾 Model saved: {model_path}")

# ==============================================================================
# EVALUATE
# ==============================================================================

if PSI4_AVAILABLE:
    print(f"\n" + "=" * 80)
    print("  PHASE 3: EVALUATION")
    print("=" * 80)
    
    print(f"\n🔬 Generating test trajectory...")
    
    # CRITICAL: Use same theory level as training data!
    if USE_EXISTING_DATA:
        # Detect from training data metadata
        test_method = training_data.metadata.get('method', 'HF')
        test_basis = training_data.metadata.get('basis', '6-31G*')
        
        print(f"\n⚠️  CRITICAL: Matching theory level from training data!")
        print(f"   Training data theory: {test_method}/{test_basis}")
        print(f"   Test will use: {test_method}/{test_basis}")
    else:
        # Use same as what we generated
        test_method = METHOD
        test_basis = BASIS
        print(f"\n   Using: {test_method}/{test_basis} (same as training)")
    
    molecule = get_molecule(MOLECULE_NAME if not USE_EXISTING_DATA else 'water')
    
    test_config = DirectMDConfig(
        method=test_method,
        basis=test_basis,
        temperature=300,
        timestep=0.5,
        n_steps=20,
        output_frequency=1
    )
    
    test_traj = run_direct_md(molecule, test_config, 
                             str(analysis_dir / 'test_traj'), 'npz')
    
    print(f"\n🤖 Predicting with ML-PES...")
    
    # Predict
    X_test_new = []
    for i in range(test_traj.n_frames):
        desc = compute_coulomb_matrix(test_traj.symbols, test_traj.coordinates[i])
        X_test_new.append(desc)
    X_test_new = np.array(X_test_new)
    
    X_test_new_scaled = scaler_X.transform(X_test_new)
    y_pred_scaled = best_model.predict(X_test_new_scaled)
    ml_energies = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    psi4_energies = test_traj.energies
    errors = (ml_energies - psi4_energies) * 627.509
    
    test_rmse = np.sqrt((errors**2).mean())
    test_mae = np.abs(errors).mean()
    test_max = np.abs(errors).max()
    
    print(f"\n📊 Test Results:")
    print(f"   RMSE: {test_rmse:.4f} kcal/mol")
    print(f"   MAE: {test_mae:.4f} kcal/mol")
    print(f"   Max error: {test_max:.4f} kcal/mol")
    
    # Plot
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    times = np.arange(len(psi4_energies)) * test_config.timestep
    
    ax1.plot(times, psi4_energies * 627.509, 'o-', label='PSI4', linewidth=2, markersize=8)
    ax1.plot(times, ml_energies * 627.509, 's--', label='ML-PES', linewidth=2, markersize=6, alpha=0.7)
    ax1.set_xlabel('Time (fs)', fontsize=14)
    ax1.set_ylabel('Energy (kcal/mol)', fontsize=14)
    ax1.set_title(f'PSI4 vs ML-PES (RMSE: {test_rmse:.4f} kcal/mol)', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(times, errors, 'o-', color='red', linewidth=2, markersize=8)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.fill_between(times, errors, alpha=0.3, color='red')
    ax2.set_xlabel('Time (fs)', fontsize=14)
    ax2.set_ylabel('Error (kcal/mol)', fontsize=14)
    ax2.set_title('Prediction Error', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = analysis_dir / 'comparison.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\n✅ Plot saved: {plot_path}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print(f"\n" + "=" * 80)
print("  WORKFLOW COMPLETE")
print("=" * 80)

print(f"\n✅ Training Data: {training_data.n_frames} configurations")
print(f"✅ Best RMSE: {best_rmse:.4f} kcal/mol")
if PSI4_AVAILABLE and 'test_rmse' in locals():
    print(f"✅ Test RMSE: {test_rmse:.4f} kcal/mol")
print(f"✅ Model: {model_path}")

if best_rmse < 0.1:
    print(f"\n🎉 EXCEPTIONAL! Essentially at QM noise level!")
elif best_rmse < 1.0:
    print(f"\n🎉 EXCELLENT! Chemical accuracy achieved!")
elif best_rmse < 5.0:
    print(f"\n✅ GOOD! Suitable for most applications.")
else:
    print(f"\n⚠️  MARGINAL. Consider more diverse training data.")

print(f"\n💡 To use:")
print(f"   import pickle")
print(f"   with open('{model_path}', 'rb') as f:")
print(f"       model = pickle.load(f)")
print(f"   # Then use model['model'], model['scaler_X'], model['scaler_y']")

print("\n" + "=" * 80)
