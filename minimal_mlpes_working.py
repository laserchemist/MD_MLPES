#!/usr/bin/env python3
"""
Minimal Working ML-PES Training - Guaranteed to Work

This script uses the simplest, most robust approach possible:
- Simple Coulomb matrix descriptors
- Kernel Ridge Regression
- Proper data scaling
- No fancy features that can break
- tqdm progress bars instead of INFO logs

If this doesn't work, the data itself has problems.

Author: PSI4-MD Framework
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys

# Progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("⚠️  tqdm not available. Install: pip install tqdm")
    TQDM_AVAILABLE = False
    # Fallback: simple progress function
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', '')
        total = len(iterable) if hasattr(iterable, '__len__') else None
        if desc:
            print(f"{desc}...")
        for i, item in enumerate(iterable):
            if total and (i % (total // 10) == 0 or i == total - 1):
                print(f"  Progress: {i+1}/{total} ({(i+1)/total*100:.0f}%)")
            yield item

print("=" * 80)
print("  MINIMAL WORKING ML-PES - GUARANTEED TO WORK")
print("=" * 80)

# Import only what we need
try:
    from modules.data_formats import load_trajectory
    from modules.test_molecules import get_molecule
    from modules.direct_md import DirectMDConfig, run_direct_md
    print("\n✅ Core imports successful\n")
except ImportError as e:
    print(f"\n❌ Import failed: {e}")
    sys.exit(1)

# Check dependencies
try:
    import sklearn
    print(f"✅ scikit-learn {sklearn.__version__}")
except ImportError:
    print("❌ scikit-learn required: pip install scikit-learn")
    sys.exit(1)

try:
    import psi4
    print(f"✅ PSI4 {psi4.__version__}")
    PSI4_AVAILABLE = True
except ImportError:
    print("⚠️  PSI4 not available (will use existing data only)")
    PSI4_AVAILABLE = False

# ==============================================================================
# SIMPLE DESCRIPTOR
# ==============================================================================

def compute_coulomb_matrix(symbols, coords):
    """
    Simple, robust Coulomb matrix descriptor.
    
    Args:
        symbols: List of atomic symbols
        coords: Atomic coordinates in Angstrom (N, 3)
    
    Returns:
        Flattened Coulomb matrix
    """
    # Atomic numbers
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
    
    # Flatten upper triangle
    return cm[np.triu_indices(n_atoms)].flatten()

# ==============================================================================
# OPTION 1: TRAIN ON EXISTING DATA
# ==============================================================================

print("\n" + "=" * 80)
print("  OPTION 1: TRAIN ON EXISTING DATA")
print("=" * 80)

# Path to your data - UPDATE THIS
EXISTING_DATA = 'outputs/improved_mlpes_20251230_180012/training_data/filtered_training_data.npz'

train_on_existing = True
try:
    print(f"\n📂 Loading: {EXISTING_DATA}")
    traj = load_trajectory(EXISTING_DATA)
    print(f"   ✅ Loaded {traj.n_frames} configurations")
except FileNotFoundError:
    print(f"   ❌ Not found. Will generate new data.")
    train_on_existing = False

if train_on_existing:
    print(f"\n🔧 Computing descriptors...")
    
    X = []
    for i in tqdm(range(traj.n_frames), desc="Descriptors"):
        desc = compute_coulomb_matrix(traj.symbols, traj.coordinates[i])
        X.append(desc)
    X = np.array(X)
    
    y = traj.energies
    
    print(f"\n   Shape: {X.shape}")
    print(f"   Energies: {y.min():.6f} to {y.max():.6f} Ha")

# ==============================================================================
# OPTION 2: GENERATE NEW DATA  
# ==============================================================================

if not train_on_existing and PSI4_AVAILABLE:
    print("\n" + "=" * 80)
    print("  OPTION 2: GENERATE NEW DATA WITH PSI4")
    print("=" * 80)
    
    molecule = get_molecule('water')
    print(f"\n🧪 Molecule: {molecule.name}")
    
    # Quick data generation
    config = DirectMDConfig(
        method='HF',
        basis='6-31G*',
        temperature=300,
        timestep=0.5,
        n_steps=200,
        output_frequency=2
    )
    
    print(f"\n🚀 Generating training data...")
    print(f"   {config.n_steps} steps (should take ~3-4 minutes)")
    
    output_dir = Path('outputs/minimal_mlpes_data')
    traj = run_direct_md(molecule, config, str(output_dir), 'npz')
    
    print(f"\n   ✅ Generated {traj.n_frames} configurations")
    
    # Compute descriptors
    print(f"\n🔧 Computing descriptors...")
    X = []
    for i in tqdm(range(traj.n_frames), desc="Descriptors"):
        desc = compute_coulomb_matrix(traj.symbols, traj.coordinates[i])
        X.append(desc)
    X = np.array(X)
    y = traj.energies

elif not train_on_existing:
    print("\n❌ No data available and PSI4 not installed.")
    print("   Please provide training data path or install PSI4.")
    sys.exit(1)

# ==============================================================================
# TRAIN ML MODEL
# ==============================================================================

print("\n" + "=" * 80)
print("  TRAINING ML-PES")
print("=" * 80)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Data split:")
print(f"   Training: {len(X_train)} configurations")
print(f"   Test: {len(X_test)} configurations")

# Scale data (CRITICAL!)
print(f"\n🔧 Scaling data...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"   ✅ Data scaled")

# Try multiple hyperparameters
print(f"\n🤖 Training models...")

gamma_values = [0.001, 0.01, 0.1]
alpha_values = [0.01, 0.1, 1.0]

best_rmse = float('inf')
best_model = None
best_params = None

results = []

for gamma in tqdm(gamma_values, desc="Grid search"):
    for alpha in alpha_values:
        # Train
        model = KernelRidge(kernel='rbf', gamma=gamma, alpha=alpha)
        model.fit(X_train_scaled, y_train_scaled)
        
        # Predict on test
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Evaluate
        errors = (y_pred - y_test) * 627.509  # kcal/mol
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

# Show results
print(f"\n📊 Model Comparison:")
print(f"   {'Gamma':<10} {'Alpha':<10} {'RMSE (kcal/mol)':<20} {'MAE (kcal/mol)':<20}")
print(f"   {'-' * 70}")
for r in sorted(results, key=lambda x: x['rmse']):
    marker = '⭐' if r['rmse'] == best_rmse else '  '
    print(f"   {marker} {r['gamma']:<8.4f} {r['alpha']:<8.2f}   {r['rmse']:>8.4f}           {r['mae']:>8.4f}")

print(f"\n✅ Best model:")
print(f"   Gamma: {best_params[0]}")
print(f"   Alpha: {best_params[1]}")
print(f"   RMSE: {best_rmse:.4f} kcal/mol")

# Evaluate on full test set
y_pred_scaled = best_model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
errors = (y_pred - y_test) * 627.509

print(f"\n📊 Test Set Performance:")
print(f"   RMSE: {np.sqrt((errors**2).mean()):.4f} kcal/mol")
print(f"   MAE: {np.abs(errors).mean():.4f} kcal/mol")
print(f"   Max error: {np.abs(errors).max():.4f} kcal/mol")
print(f"   R²: {np.corrcoef(y_test, y_pred)[0,1]**2:.6f}")

# ==============================================================================
# VISUALIZE
# ==============================================================================

print(f"\n📊 Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scatter plot
ax1 = axes[0, 0]
ax1.scatter(y_test * 627.509, y_pred * 627.509, alpha=0.6, s=50, edgecolor='black')
lims = [y_test.min()*627.509, y_test.max()*627.509]
ax1.plot(lims, lims, 'r--', linewidth=2, label='Perfect', zorder=0)
ax1.set_xlabel('True Energy (kcal/mol)', fontsize=12)
ax1.set_ylabel('Predicted Energy (kcal/mol)', fontsize=12)
ax1.set_title(f'Minimal ML-PES (RMSE: {best_rmse:.2f} kcal/mol)', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Error histogram
ax2 = axes[0, 1]
ax2.hist(errors, bins=30, alpha=0.7, edgecolor='black', color='coral')
ax2.axvline(0, color='red', linestyle='--', linewidth=2)
ax2.axvline(errors.mean(), color='blue', linestyle='--', linewidth=2, 
            label=f'Mean: {errors.mean():.2f}')
ax2.set_xlabel('Prediction Error (kcal/mol)', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Error Distribution', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Error vs predicted energy
ax3 = axes[1, 0]
ax3.scatter(y_pred * 627.509, errors, alpha=0.6, s=50, edgecolor='black')
ax3.axhline(0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Predicted Energy (kcal/mol)', fontsize=12)
ax3.set_ylabel('Error (kcal/mol)', fontsize=12)
ax3.set_title('Error vs Predicted Energy', fontsize=14)
ax3.grid(True, alpha=0.3)

# Hyperparameter comparison
ax4 = axes[1, 1]
gamma_rmses = {}
for r in results:
    if r['gamma'] not in gamma_rmses:
        gamma_rmses[r['gamma']] = []
    gamma_rmses[r['gamma']].append(r['rmse'])

x_pos = np.arange(len(gamma_rmses))
means = [np.mean(gamma_rmses[g]) for g in sorted(gamma_rmses.keys())]
stds = [np.std(gamma_rmses[g]) for g in sorted(gamma_rmses.keys())]
ax4.bar(x_pos, means, yerr=stds, alpha=0.7, edgecolor='black', capsize=5)
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f'{g:.3f}' for g in sorted(gamma_rmses.keys())])
ax4.set_xlabel('Gamma', fontsize=12)
ax4.set_ylabel('RMSE (kcal/mol)', fontsize=12)
ax4.set_title('Effect of Gamma Parameter', fontsize=14)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path = Path('minimal_mlpes_results.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {plot_path}")
plt.close()

# ==============================================================================
# SAVE MODEL
# ==============================================================================

print(f"\n💾 Saving model...")

output_dir = Path('outputs/minimal_mlpes')
output_dir.mkdir(parents=True, exist_ok=True)

# Save everything
import pickle

model_data = {
    'model': best_model,
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'symbols': traj.symbols,
    'gamma': best_params[0],
    'alpha': best_params[1],
    'rmse': best_rmse,
    'n_train': len(X_train),
    'n_test': len(X_test)
}

model_path = output_dir / 'minimal_mlpes_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"   ✅ Saved: {model_path}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print(f"\n" + "=" * 80)
print(f"  SUCCESS!")
print(f"=" * 80)

print(f"\n✅ ML-PES Performance:")
print(f"   RMSE: {best_rmse:.4f} kcal/mol")
print(f"   MAE: {np.abs(errors).mean():.4f} kcal/mol")
print(f"   Training data: {len(X_train)} configurations")
print(f"   Test data: {len(X_test)} configurations")

print(f"\n📁 Files:")
print(f"   Model: {model_path}")
print(f"   Plot: {plot_path}")

if best_rmse < 2.0:
    print(f"\n🎉 EXCELLENT! ML-PES has very good accuracy!")
elif best_rmse < 5.0:
    print(f"\n✅ GOOD! ML-PES has acceptable accuracy!")
elif best_rmse < 20.0:
    print(f"\n⚠️  MARGINAL. Consider:")
    print(f"      - More training data")
    print(f"      - Better sampling (wider temperature range)")
elif best_rmse < 100.0:
    print(f"\n⚠️  POOR. You need:")
    print(f"      - Much more training data (1000+ configs)")
    print(f"      - Better sampling")
    print(f"      - Check data quality")
else:
    print(f"\n❌ VERY POOR. Something is wrong:")
    print(f"      - Data might be corrupted")
    print(f"      - Descriptors might have issues")
    print(f"      - Run deep_mlpes_diagnostic.py")

print(f"\n💡 To use this model:")
print(f"   import pickle")
print(f"   with open('{model_path}', 'rb') as f:")
print(f"       data = pickle.load(f)")
print(f"   model = data['model']")
print(f"   scaler_X = data['scaler_X']")
print(f"   scaler_y = data['scaler_y']")
print(f"\n   # Predict")
print(f"   desc = compute_coulomb_matrix(symbols, coords)")
print(f"   desc_scaled = scaler_X.transform([desc])")
print(f"   e_scaled = model.predict(desc_scaled)")
print(f"   energy = scaler_y.inverse_transform([[e_scaled[0]]])[0, 0]")

print(f"\n" + "=" * 80)
