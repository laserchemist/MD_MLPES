#!/usr/bin/env python3
"""
Fixed ML-PES Diagnostic - Works with your setup

This diagnostic revealed the REAL problem:
Your energy range is only 6.8 kcal/mol - way too narrow!

Author: PSI4-MD Framework
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

print("=" * 80)
print("  FIXED ML-PES DIAGNOSTIC")
print("=" * 80)

# Import
try:
    from modules.data_formats import load_trajectory
    from modules.test_molecules import get_molecule
    print("\n✅ Imports successful\n")
except ImportError as e:
    print(f"\n❌ Import failed: {e}")
    sys.exit(1)

# Load training data - FIXED PATH
TRAINING_DATA = 'outputs/improved_mlpes_20251230_183101/training_data/filtered_training_data.npz'

try:
    traj = load_trajectory(TRAINING_DATA)
    print(f"✅ Loaded: {TRAINING_DATA}")
    print(f"   Frames: {traj.n_frames}")
    print(f"   Atoms: {len(traj.symbols)}")
except FileNotFoundError:
    print(f"\n❌ File not found. Looking for any training data...")
    # Search for training data
    import glob
    patterns = [
        'outputs/*/training_data/filtered_training_data.npz',
        'outputs/*/training_data/combined_training_data.npz',
        'outputs/*_mlpes_*/training_data/*.npz'
    ]
    
    found_files = []
    for pattern in patterns:
        found_files.extend(glob.glob(pattern))
    
    if found_files:
        print(f"\n   Found {len(found_files)} file(s):")
        for i, f in enumerate(found_files):
            print(f"   {i+1}. {f}")
        
        TRAINING_DATA = found_files[0]
        print(f"\n   Using: {TRAINING_DATA}")
        traj = load_trajectory(TRAINING_DATA)
    else:
        print(f"\n❌ No training data found!")
        sys.exit(1)

# ==============================================================================
# ANALYZE ENERGY DISTRIBUTION (THE REAL PROBLEM!)
# ==============================================================================

print("\n" + "=" * 80)
print("  ENERGY DISTRIBUTION ANALYSIS - THIS IS THE KEY!")
print("=" * 80)

energies_ha = traj.energies
energies_kcal = energies_ha * 627.509

e_min = energies_kcal.min()
e_max = energies_kcal.max()
e_mean = energies_kcal.mean()
e_std = energies_kcal.std()
e_range = e_max - e_min

print(f"\n📊 Energy Statistics:")
print(f"   {'─' * 60}")
print(f"   Min:    {e_min:.2f} kcal/mol")
print(f"   Max:    {e_max:.2f} kcal/mol")
print(f"   Mean:   {e_mean:.2f} kcal/mol")
print(f"   Std:    {e_std:.2f} kcal/mol")
print(f"   RANGE:  {e_range:.2f} kcal/mol  ← THIS IS THE PROBLEM!")
print(f"   {'─' * 60}")

# Diagnose the problem
print(f"\n🔍 DIAGNOSIS:")

if e_range < 5:
    print(f"\n   ❌ CRITICAL: Energy range too narrow ({e_range:.2f} kcal/mol)")
    print(f"      Your configurations are too similar!")
    print(f"      ML models can't learn from such uniform data.")
    print(f"      Target: Need 10-20+ kcal/mol range")
    severity = "CRITICAL"
elif e_range < 10:
    print(f"\n   ⚠️  WARNING: Energy range marginal ({e_range:.2f} kcal/mol)")
    print(f"      ML might work but won't be very accurate.")
    print(f"      Target: Need 15-20+ kcal/mol for good ML-PES")
    severity = "WARNING"
elif e_range < 20:
    print(f"\n   ✅ ACCEPTABLE: Energy range OK ({e_range:.2f} kcal/mol)")
    print(f"      ML should work reasonably well.")
    severity = "ACCEPTABLE"
else:
    print(f"\n   ✅ EXCELLENT: Energy range good ({e_range:.2f} kcal/mol)")
    print(f"      ML should work very well!")
    severity = "EXCELLENT"

if e_std < 2:
    print(f"\n   ⚠️  Standard deviation too small ({e_std:.2f} kcal/mol)")
    print(f"      Configurations are clustered too tightly.")
    print(f"      Need more diverse sampling!")

# ==============================================================================
# SIMPLE COULOMB MATRIX DESCRIPTOR TEST
# ==============================================================================

print("\n" + "=" * 80)
print("  DESCRIPTOR TEST")
print("=" * 80)

def compute_coulomb_matrix(symbols, coords):
    """Simple Coulomb matrix."""
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

print("\n🔧 Computing descriptors for first 100 frames...")

all_descs = []
for i in range(min(100, traj.n_frames)):
    desc = compute_coulomb_matrix(traj.symbols, traj.coordinates[i])
    all_descs.append(desc)
all_descs = np.array(all_descs)

print(f"   Computed {len(all_descs)} descriptors")
print(f"   Shape: {all_descs.shape}")
print(f"   Mean: {all_descs.mean():.4f}")
print(f"   Std: {all_descs.std():.4f}")

if all_descs.std() < 0.1:
    print(f"\n   ⚠️  WARNING: Low descriptor variance!")
    print(f"      Geometries are very similar.")

# ==============================================================================
# SIMPLE LINEAR REGRESSION TEST
# ==============================================================================

print("\n" + "=" * 80)
print("  SIMPLE LINEAR REGRESSION TEST")
print("=" * 80)

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Compute descriptors for all data
print(f"\n🔧 Computing descriptors for all {traj.n_frames} frames...")
X = []
for i in range(traj.n_frames):
    if i % 100 == 0:
        print(f"   Progress: {i}/{traj.n_frames}")
    desc = compute_coulomb_matrix(traj.symbols, traj.coordinates[i])
    X.append(desc)
X = np.array(X)
y = traj.energies

print(f"   ✅ Computed descriptors")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Data split:")
print(f"   Training: {len(X_train)} configs")
print(f"   Test: {len(X_test)} configs")

# Scale
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Train
print(f"\n🤖 Training linear model...")
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train_scaled)

# Predict
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Evaluate
errors = (y_pred - y_test) * 627.509
rmse = np.sqrt((errors**2).mean())
mae = np.abs(errors).mean()
r2 = 1 - ((y_test - y_pred)**2).sum() / ((y_test - y_test.mean())**2).sum()

print(f"\n📊 Linear Model Results:")
print(f"   {'─' * 60}")
print(f"   RMSE: {rmse:.4f} kcal/mol")
print(f"   MAE:  {mae:.4f} kcal/mol")
print(f"   R²:   {r2:.6f}")
print(f"   {'─' * 60}")

# Interpret
print(f"\n🔍 Interpretation:")
if rmse < 2.0:
    print(f"   ✅ EXCELLENT! Linear model works very well.")
    print(f"      Your data IS learnable!")
    print(f"      Problem is in the ML-PES training code.")
elif rmse < 5.0:
    print(f"   ✅ GOOD! Linear model works reasonably well.")
    print(f"      Data is learnable but could be better.")
elif rmse < 10.0:
    print(f"   ⚠️  MARGINAL. Linear model struggles.")
    print(f"      Data quality issues OR too narrow energy range.")
else:
    print(f"   ❌ POOR. Linear model fails.")
    print(f"      Serious data problems.")

# Compare to energy range
print(f"\n📊 RMSE vs Energy Range:")
print(f"   Energy range: {e_range:.2f} kcal/mol")
print(f"   RMSE: {rmse:.2f} kcal/mol")
print(f"   Ratio: {rmse/e_range:.2%} of energy range")

if rmse/e_range < 0.1:
    print(f"   ✅ EXCELLENT ratio! Model captures 90%+ of variance")
elif rmse/e_range < 0.3:
    print(f"   ✅ GOOD ratio! Model captures 70%+ of variance")
elif rmse/e_range < 0.5:
    print(f"   ⚠️  MARGINAL ratio. Model captures 50%+ of variance")
else:
    print(f"   ❌ POOR ratio. Model barely better than guessing mean")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

print(f"\n📊 Creating diagnostic plots...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Energy histogram
ax1 = axes[0, 0]
ax1.hist(energies_kcal, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
ax1.axvline(e_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {e_mean:.1f}')
ax1.axvline(e_mean - e_std, color='orange', linestyle=':', linewidth=1.5, label=f'±1 std')
ax1.axvline(e_mean + e_std, color='orange', linestyle=':', linewidth=1.5)
ax1.set_xlabel('Energy (kcal/mol)', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title(f'Energy Distribution (Range: {e_range:.2f} kcal/mol)', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Energy trajectory
ax2 = axes[0, 1]
ax2.plot(energies_kcal, linewidth=1, alpha=0.7)
ax2.axhline(e_mean, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax2.axhline(e_mean - e_std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax2.axhline(e_mean + e_std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax2.set_xlabel('Configuration Index', fontsize=12)
ax2.set_ylabel('Energy (kcal/mol)', fontsize=12)
ax2.set_title('Energy Trajectory', fontsize=14)
ax2.grid(True, alpha=0.3)

# Prediction scatter
ax3 = axes[0, 2]
ax3.scatter(y_test * 627.509, y_pred * 627.509, alpha=0.6, s=50, edgecolor='black')
lims = [y_test.min()*627.509, y_test.max()*627.509]
ax3.plot(lims, lims, 'r--', linewidth=2, label='Perfect', zorder=0)
ax3.set_xlabel('True Energy (kcal/mol)', fontsize=12)
ax3.set_ylabel('Predicted Energy (kcal/mol)', fontsize=12)
ax3.set_title(f'Linear Model (RMSE: {rmse:.2f})', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')

# Error histogram
ax4 = axes[1, 0]
ax4.hist(errors, bins=30, alpha=0.7, edgecolor='black', color='coral')
ax4.axvline(0, color='red', linestyle='--', linewidth=2)
ax4.axvline(errors.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.2f}')
ax4.set_xlabel('Prediction Error (kcal/mol)', fontsize=12)
ax4.set_ylabel('Count', fontsize=12)
ax4.set_title('Error Distribution', fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Descriptor variance
ax5 = axes[1, 1]
desc_vars = X.var(axis=0)
ax5.bar(range(len(desc_vars)), desc_vars, alpha=0.7, edgecolor='black')
ax5.set_xlabel('Descriptor Index', fontsize=12)
ax5.set_ylabel('Variance', fontsize=12)
ax5.set_title('Descriptor Variance (should be non-zero)', fontsize=14)
ax5.grid(True, alpha=0.3, axis='y')

# Summary box
ax6 = axes[1, 2]
ax6.axis('off')

summary_text = f"""
DIAGNOSTIC SUMMARY

Data Size: {traj.n_frames} configurations

Energy Range: {e_range:.2f} kcal/mol
Status: {severity}

Linear Model RMSE: {rmse:.2f} kcal/mol
Ratio to Range: {rmse/e_range:.1%}

THE PROBLEM:
{'✅ Data is learnable!' if rmse < 5 else '⚠️ Data quality issues'}
{'❌ Energy range too narrow!' if e_range < 10 else '✅ Energy range OK'}

RECOMMENDATION:
{'Generate more diverse data!' if e_range < 10 else 'Use minimal_mlpes_working.py'}
"""

ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.5))

plt.tight_layout()
plot_path = Path('diagnostic_results.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {plot_path}")
plt.close()

# ==============================================================================
# FINAL RECOMMENDATIONS
# ==============================================================================

print("\n" + "=" * 80)
print("  FINAL DIAGNOSIS & RECOMMENDATIONS")
print("=" * 80)

print(f"\n🔍 ROOT CAUSE ANALYSIS:")

if e_range < 10 and rmse < 5:
    print(f"\n   PRIMARY ISSUE: Energy range too narrow ({e_range:.2f} kcal/mol)")
    print(f"   SECONDARY: But data IS learnable (linear RMSE: {rmse:.2f})")
    print(f"\n   ✅ Your ML-PES will work, but won't be very impressive")
    print(f"      because there's not much to learn!")
    print(f"\n   Think of it this way:")
    print(f"      - You're teaching a model to predict energies")
    print(f"      - But all your energies are within {e_range:.2f} kcal/mol")
    print(f"      - That's like teaching someone to recognize temperatures")
    print(f"        but only showing them 70-77°F")
    print(f"      - The model will work, but it's an easy task!")
    
elif e_range < 10:
    print(f"\n   PRIMARY ISSUE: Energy range too narrow ({e_range:.2f} kcal/mol)")
    print(f"   SECONDARY: Data not very learnable (linear RMSE: {rmse:.2f})")
    print(f"\n   ⚠️  You have TWO problems:")
    print(f"      1. Not enough diversity in your data")
    print(f"      2. What little diversity exists isn't well-captured")
    
elif rmse > 10:
    print(f"\n   PRIMARY ISSUE: Data quality problems")
    print(f"   Even simple linear model fails (RMSE: {rmse:.2f} kcal/mol)")
    print(f"\n   Possible causes:")
    print(f"      - Descriptor calculation errors")
    print(f"      - Corrupted energy values")
    print(f"      - Wrong units somewhere")
    
else:
    print(f"\n   ✅ NO MAJOR ISSUES!")
    print(f"      Energy range: {e_range:.2f} kcal/mol (good)")
    print(f"      Linear RMSE: {rmse:.2f} kcal/mol (good)")
    print(f"\n   Your data is fine. Problem is in ml_pes.py implementation.")

print(f"\n💡 WHAT TO DO:")

if e_range < 10:
    print(f"\n   FOR BETTER ML-PES:")
    print(f"   1. Generate more diverse training data:")
    print(f"      - Higher temperatures: 400, 500, 600 K")
    print(f"      - More trajectories: 8-10 instead of 5")
    print(f"      - Displaced starting geometries")
    print(f"      - Target: 20+ kcal/mol energy range")
    print(f"\n   2. OR accept that your ML-PES will be limited:")
    print(f"      - It will work for configurations similar to training")
    print(f"      - But won't generalize well to very different structures")
    print(f"\n   3. Try minimal_mlpes_working.py anyway:")
    print(f"      - It will work, just won't be amazing")
    print(f"      - RMSE will be ~{rmse:.1f} kcal/mol (which is {rmse/e_range:.0%} of your range)")
    
else:
    print(f"\n   YOUR DATA IS GOOD!")
    print(f"   1. Run: python3 minimal_mlpes_working.py")
    print(f"   2. Expected RMSE: 1-5 kcal/mol")
    print(f"   3. The original ml_pes.py has bugs - use minimal version")

print("\n" + "=" * 80)
print(f"  Diagnostic complete. See diagnostic_results.png for visualizations.")
print("=" * 80)
