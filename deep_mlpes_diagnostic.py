#!/usr/bin/env python3
"""
Deep ML-PES Diagnostic - Find the Bug

RMSE of 246 kcal/mol with 1005 configurations is TERRIBLE.
Something is fundamentally broken. Let's find it.

This script will check:
1. Descriptor calculation sanity
2. Energy referencing issues
3. Data scaling problems
4. Model training bugs
5. Prediction pipeline issues

Author: PSI4-MD Framework
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

print("=" * 80)
print("  DEEP ML-PES DIAGNOSTIC - FINDING THE BUG")
print("=" * 80)

# Import
try:
    from modules.data_formats import load_trajectory
    from modules.ml_pes import MLPESConfig, MLPESTrainer
    from modules.test_molecules import get_molecule
    print("\n✅ Imports successful\n")
except ImportError as e:
    print(f"\n❌ Import failed: {e}")
    sys.exit(1)

# Load your training data - UPDATE THIS PATH
TRAINING_DATA = 'outputs/improved_mlpes_20251230_180012/training_data/filtered_training_data.npz'

try:
    traj = load_trajectory(TRAINING_DATA)
    print(f"✅ Loaded: {TRAINING_DATA}")
    print(f"   Frames: {traj.n_frames}")
    print(f"   Atoms: {len(traj.symbols)}")
except FileNotFoundError:
    print(f"\n❌ File not found: {TRAINING_DATA}")
    print("\n   Please update TRAINING_DATA path in this script")
    sys.exit(1)

# ==============================================================================
# TEST 1: Check raw energy values
# ==============================================================================

print("\n" + "=" * 80)
print("  TEST 1: RAW ENERGY VALUES")
print("=" * 80)

print(f"\nEnergies (Hartree):")
print(f"   Min:  {traj.energies.min():.8f}")
print(f"   Max:  {traj.energies.max():.8f}")
print(f"   Mean: {traj.energies.mean():.8f}")
print(f"   Std:  {traj.energies.std():.8f}")

print(f"\nEnergies (kcal/mol):")
e_kcal = traj.energies * 627.509
print(f"   Min:  {e_kcal.min():.2f}")
print(f"   Max:  {e_kcal.max():.2f}")
print(f"   Mean: {e_kcal.mean():.2f}")
print(f"   Std:  {e_kcal.std():.2f}")

# Check if energies are reasonable for water
water = get_molecule('water')
expected_energy = water.reference_energy
print(f"\nExpected energy for water (HF/STO-3G): {expected_energy:.6f} Ha")
print(f"Your data mean: {traj.energies.mean():.6f} Ha")
print(f"Difference: {abs(traj.energies.mean() - expected_energy)*627.509:.1f} kcal/mol")

if abs(traj.energies.mean() - expected_energy) > 0.1:
    print("⚠️  Warning: Energies seem off. Different basis set?")

# ==============================================================================
# TEST 2: Check descriptor calculation
# ==============================================================================

print("\n" + "=" * 80)
print("  TEST 2: DESCRIPTOR CALCULATION")
print("=" * 80)

from modules.ml_pes import CoulombMatrixDescriptor, InternalsDescriptor

# Test Coulomb matrix
print("\nTesting Coulomb Matrix descriptor...")
cm_desc = CoulombMatrixDescriptor()

# Compute for first frame
desc1 = cm_desc.compute(traj.symbols, traj.coordinates[0])
print(f"   Shape: {desc1.shape}")
print(f"   Values: min={desc1.min():.4f}, max={desc1.max():.4f}, mean={desc1.mean():.4f}")

# Check for NaN or Inf
if np.any(np.isnan(desc1)) or np.any(np.isinf(desc1)):
    print("   ❌ ERROR: NaN or Inf in descriptors!")
else:
    print("   ✅ No NaN or Inf")

# Compute for all frames
print("\n   Computing descriptors for all frames...")
all_descs = []
for i in range(min(100, traj.n_frames)):  # Test first 100
    desc = cm_desc.compute(traj.symbols, traj.coordinates[i])
    all_descs.append(desc)
all_descs = np.array(all_descs)

print(f"   Computed {len(all_descs)} descriptors")
print(f"   Shape: {all_descs.shape}")
print(f"   Mean: {all_descs.mean():.4f}")
print(f"   Std: {all_descs.std():.4f}")

# Check variability
if all_descs.std() < 0.01:
    print("   ⚠️  Warning: Very low descriptor variance - geometries too similar?")
else:
    print("   ✅ Good descriptor variance")

# ==============================================================================
# TEST 3: Simple linear regression test
# ==============================================================================

print("\n" + "=" * 80)
print("  TEST 3: SIMPLE LINEAR REGRESSION")
print("=" * 80)

print("\nTrying simplest possible model (linear regression)...")

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Use subset for speed
n_train = min(500, traj.n_frames - 50)
n_test = 50

# Compute descriptors
print(f"   Computing descriptors for {n_train} training + {n_test} test...")
X_train = []
for i in range(n_train):
    desc = cm_desc.compute(traj.symbols, traj.coordinates[i])
    X_train.append(desc)
X_train = np.array(X_train)

X_test = []
for i in range(n_train, n_train + n_test):
    desc = cm_desc.compute(traj.symbols, traj.coordinates[i])
    X_test.append(desc)
X_test = np.array(X_test)

y_train = traj.energies[:n_train]
y_test = traj.energies[n_train:n_train + n_test]

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Train
print("   Training linear model...")
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train_scaled)

# Predict
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Evaluate
errors = (y_pred - y_test) * 627.509
rmse = np.sqrt((errors**2).mean())
mae = np.abs(errors).mean()

print(f"\n   Linear Model Results:")
print(f"   RMSE: {rmse:.4f} kcal/mol")
print(f"   MAE:  {mae:.4f} kcal/mol")

if rmse < 10:
    print("   ✅ GOOD! Linear model works - data is learnable")
elif rmse < 50:
    print("   ⚠️  MARGINAL - linear model struggles but data might be OK")
else:
    print("   ❌ BAD! Even linear model fails - something wrong with data or descriptors")

# ==============================================================================
# TEST 4: Check scaling/referencing
# ==============================================================================

print("\n" + "=" * 80)
print("  TEST 4: ENERGY SCALING/REFERENCING")
print("=" * 80)

# Check if energies need referencing
print("\nEnergy statistics:")
print(f"   Range: {(traj.energies.max() - traj.energies.min())*627.509:.2f} kcal/mol")
print(f"   Relative to mean: {((traj.energies - traj.energies.mean())*627.509).std():.2f} kcal/mol std")

# Try with referenced energies
y_train_ref = y_train - y_train.mean()
y_test_ref = y_test - y_train.mean()

scaler_y_ref = StandardScaler()
y_train_ref_scaled = scaler_y_ref.fit_transform(y_train_ref.reshape(-1, 1)).flatten()
y_test_ref_scaled = scaler_y_ref.transform(y_test_ref.reshape(-1, 1)).flatten()

model_ref = Ridge(alpha=1.0)
model_ref.fit(X_train_scaled, y_train_ref_scaled)

y_pred_ref_scaled = model_ref.predict(X_test_scaled)
y_pred_ref = scaler_y_ref.inverse_transform(y_pred_ref_scaled.reshape(-1, 1)).flatten()
y_pred_ref += y_train.mean()  # Add back reference

errors_ref = (y_pred_ref - y_test) * 627.509
rmse_ref = np.sqrt((errors_ref**2).mean())

print(f"\n   With energy referencing:")
print(f"   RMSE: {rmse_ref:.4f} kcal/mol")

if abs(rmse_ref - rmse) > 1.0:
    print("   ⚠️  Referencing makes a difference!")

# ==============================================================================
# TEST 5: Visualize predictions
# ==============================================================================

print("\n" + "=" * 80)
print("  TEST 5: VISUALIZATION")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scatter plot
ax1 = axes[0, 0]
ax1.scatter(y_test * 627.509, y_pred * 627.509, alpha=0.6, s=50)
ax1.plot([y_test.min()*627.509, y_test.max()*627.509], 
         [y_test.min()*627.509, y_test.max()*627.509], 
         'r--', linewidth=2, label='Perfect')
ax1.set_xlabel('True Energy (kcal/mol)', fontsize=12)
ax1.set_ylabel('Predicted Energy (kcal/mol)', fontsize=12)
ax1.set_title(f'Simple Linear Model (RMSE: {rmse:.2f})', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Error distribution
ax2 = axes[0, 1]
ax2.hist(errors, bins=20, alpha=0.7, edgecolor='black')
ax2.axvline(0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Prediction Error (kcal/mol)', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Error Distribution', fontsize=14)
ax2.grid(True, alpha=0.3)

# Descriptor variance
ax3 = axes[1, 0]
desc_vars = X_train.var(axis=0)
ax3.bar(range(len(desc_vars)), desc_vars)
ax3.set_xlabel('Descriptor Index', fontsize=12)
ax3.set_ylabel('Variance', fontsize=12)
ax3.set_title('Descriptor Variance (should be non-zero)', fontsize=14)
ax3.grid(True, alpha=0.3)

# Energy correlation
ax4 = axes[1, 1]
ax4.plot(y_train[:100] * 627.509, linewidth=1, label='Training')
ax4.set_xlabel('Configuration Index', fontsize=12)
ax4.set_ylabel('Energy (kcal/mol)', fontsize=12)
ax4.set_title('Training Energy Trajectory', fontsize=14)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
diag_path = Path('ml_pes_deep_diagnostic.png')
plt.savefig(diag_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Diagnostic plot saved: {diag_path}")
plt.close()

# ==============================================================================
# SUMMARY AND RECOMMENDATIONS
# ==============================================================================

print("\n" + "=" * 80)
print("  DIAGNOSTIC SUMMARY")
print("=" * 80)

print(f"\n📊 Results:")
print(f"   Data size: {traj.n_frames} configurations")
print(f"   Energy range: {(traj.energies.max() - traj.energies.min())*627.509:.2f} kcal/mol")
print(f"   Linear model RMSE: {rmse:.2f} kcal/mol")

print(f"\n🔍 Diagnosis:")

if rmse < 5:
    print("   ✅ DATA IS GOOD! Linear model works well.")
    print("   ✅ Descriptors are working correctly.")
    print("   ✅ The problem is in the ML-PES trainer implementation.")
    print("\n   🐛 BUG LOCATION: modules/ml_pes.py training code")
    print("      Likely issues:")
    print("      - Kernel parameters are way off")
    print("      - Scaling is broken in MLPESTrainer")
    print("      - Force training is interfering")
    print("      - Cross-validation split is wrong")
    
elif rmse < 50:
    print("   ⚠️  DATA IS MARGINAL.")
    print("   ⚠️  Even simple model struggles.")
    print("\n   Recommendations:")
    print("      - Need more diverse data")
    print("      - Try different descriptor (internals)")
    print("      - Check for outliers")
    
else:
    print("   ❌ DATA HAS PROBLEMS!")
    print("   ❌ Even linear model fails completely.")
    print("\n   Critical issues:")
    print("      - Descriptors might be broken")
    print("      - Energy values are corrupted")
    print("      - Data scaling is wrong")

print(f"\n💡 Next Steps:")
if rmse < 5:
    print("   1. Use the simple linear model (it works!)")
    print("   2. Debug the MLPESTrainer class")
    print("   3. Try kernel_params={'gamma': 0.001, 'alpha': 0.01}")
    print("   4. Disable force training temporarily")
elif rmse < 50:
    print("   1. Generate more diverse training data")
    print("   2. Try internals descriptor")
    print("   3. Check for and remove outliers")
else:
    print("   1. Check descriptor implementation")
    print("   2. Verify energy units are correct")
    print("   3. Look for NaN/Inf in data")

print("\n" + "=" * 80)
