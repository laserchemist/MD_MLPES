#!/usr/bin/env python3
"""
Quick diagnostic for huge ML-PES errors.

This script tests if the model works on its own training data,
which helps identify if the problem is:
1. In the model itself
2. In the validation setup
3. In data units/theory mismatch
"""

import pickle
import numpy as np
import sys
from pathlib import Path

try:
    from modules.data_formats import load_trajectory
    print("✅ Framework loaded")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure you're in the MD_MLPES directory")
    sys.exit(1)

# Coulomb matrix (same as in two_phase_workflow.py)
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
                r = np.linalg.norm(coords[i] - coords[j])
                if r > 1e-10:
                    cm[i, j] = Z[i] * Z[j] / r
    
    indices = np.triu_indices(n_atoms)
    return cm[indices]

print("=" * 70)
print("  DIAGNOSTIC FOR HUGE ML-PES ERRORS")
print("=" * 70)

# Get file paths
import argparse
parser = argparse.ArgumentParser(description='Diagnose huge ML-PES errors')
parser.add_argument('--model', default='outputs/refined_20251231_121834/initial_mlpes_model.pkl',
                   help='Path to ML-PES model')
parser.add_argument('--training-data', default='outputs/mlpes_v2_20251231_040244/training_data.npz',
                   help='Path to training data')
args = parser.parse_args()

# Load everything
print("\n📂 Loading files...")
print(f"   Model: {args.model}")
print(f"   Training data: {args.training_data}")

if not Path(args.model).exists():
    print(f"\n❌ Model not found: {args.model}")
    sys.exit(1)

if not Path(args.training_data).exists():
    print(f"\n❌ Training data not found: {args.training_data}")
    sys.exit(1)

data = load_trajectory(args.training_data)
print(f"   ✅ Loaded training data: {data.n_frames} frames")

with open(args.model, 'rb') as f:
    model_data = pickle.load(f)
print(f"   ✅ Loaded model")

# Test on training data point
idx = np.argmin(data.energies)
coords = data.coordinates[idx]
true_e = data.energies[idx]

print(f"\n🔧 Testing model on training data point {idx}...")
print(f"   (Using lowest energy configuration)")

# Predict
desc = compute_coulomb_matrix(data.symbols, coords)
desc_scaled = model_data['scaler_X'].transform(desc.reshape(1, -1))
e_scaled = model_data['model'].predict(desc_scaled)[0]
pred_e = model_data['scaler_y'].inverse_transform([[e_scaled]])[0, 0]

error_ha = abs(pred_e - true_e)
error_kcal = error_ha * 627.509

# Display results
print("\n" + "=" * 70)
print("  RESULTS")
print("=" * 70)

print(f"\n📊 Training data energy range:")
e_min = data.energies.min()
e_max = data.energies.max()
e_range = e_max - e_min
print(f"   {e_min:.6f} to {e_max:.6f} Ha")
print(f"   {e_min*627.509:.1f} to {e_max*627.509:.1f} kcal/mol")
print(f"   Range: {e_range*627.509:.1f} kcal/mol")

if abs(e_min) < 10:
    print(f"   ⚠️  Energies look RELATIVE (min ~ 0)")
elif abs(e_min) > 100:
    print(f"   ✅ Energies look ABSOLUTE (min ~ -{abs(e_min):.0f} Ha)")
else:
    print(f"   ❓ Unclear if relative or absolute")

print(f"\n🎯 Test on training data point:")
print(f"   True energy: {true_e:.6f} Ha ({true_e*627.509:.1f} kcal/mol)")
print(f"   Predicted:   {pred_e:.6f} Ha ({pred_e*627.509:.1f} kcal/mol)")
print(f"   Error:       {error_ha:.6f} Ha ({error_kcal:.2f} kcal/mol)")

print(f"\n📋 Metadata:")
if hasattr(data, 'metadata') and data.metadata:
    training_theory = data.metadata.get('theory', None)
    if training_theory:
        print(f"   Training data theory:")
        print(f"      Method: {training_theory.get('method', 'UNKNOWN')}")
        print(f"      Basis: {training_theory.get('basis', 'UNKNOWN')}")
    else:
        print(f"   Training data: NO THEORY INFO!")
else:
    print(f"   Training data: NO METADATA!")

model_theory = model_data['metadata'].get('theory', None)
if model_theory:
    print(f"   Model theory:")
    print(f"      Method: {model_theory.get('method', 'UNKNOWN')}")
    print(f"      Basis: {model_theory.get('basis', 'UNKNOWN')}")
else:
    print(f"   Model: NO THEORY INFO!")

# Diagnosis
print("\n" + "=" * 70)
print("  DIAGNOSIS")
print("=" * 70)

if error_kcal < 1.0:
    print("\n✅ MODEL WORKS ON TRAINING DATA!")
    print(f"   Error: {error_kcal:.3f} kcal/mol (excellent!)")
    print()
    print("   The model itself is fine.")
    print("   Problem is in the VALIDATION setup:")
    print()
    print("   Likely causes:")
    print("   1. PSI4 validation using different theory level")
    print("      → Check that validation uses same method/basis")
    print("   2. Coordinate units mismatch (Angstrom vs Bohr)")
    print("      → Check that both use same units")
    print("   3. Energy reference offset")
    print("      → Training used relative, validation uses absolute")
    print()
    print("   Next steps:")
    print("   • Check PSI4 validation setup in two_phase_workflow.py")
    print("   • Verify theory level matches training data")
    print("   • Check coordinate and energy units")

elif error_kcal < 10.0:
    print("\n⚠️  MODEL HAS SMALL ERROR ON TRAINING DATA")
    print(f"   Error: {error_kcal:.3f} kcal/mol")
    print()
    print("   Model is working but not perfectly.")
    print()
    print("   Likely causes:")
    print("   1. Hyperparameters not optimal")
    print("   2. Descriptor may not be ideal for this molecule")
    print()
    print("   But this doesn't explain 3397 kcal/mol errors!")
    print("   Something else is wrong in validation setup.")

elif error_kcal < 100.0:
    print("\n❌ MODEL HAS LARGE ERROR ON TRAINING DATA")
    print(f"   Error: {error_kcal:.1f} kcal/mol")
    print()
    print("   Model isn't learning properly from training data.")
    print()
    print("   Likely causes:")
    print("   1. Descriptor computation is wrong")
    print("   2. Data loading issue")
    print("   3. Need different ML approach")
    print()
    print("   Next steps:")
    print("   • Check descriptor calculation")
    print("   • Try different hyperparameters")
    print("   • Consider different ML method")

else:
    print("\n❌ MODEL IS COMPLETELY BROKEN!")
    print(f"   Error: {error_kcal:.1f} kcal/mol (HUGE!)")
    print()
    print("   Model doesn't work even on its own training data!")
    print()
    print("   Likely causes:")
    print("   1. Data loading is corrupted")
    print("   2. Descriptor computation is fundamentally wrong")
    print("   3. Scaling is wrong")
    print("   4. Model file is corrupted")
    print()
    print("   Next steps:")
    print("   • Retrain model from scratch")
    print("   • Verify training data loads correctly")
    print("   • Check descriptor implementation")

# Additional checks
print("\n" + "=" * 70)
print("  ADDITIONAL CHECKS")
print("=" * 70)

print(f"\n🔍 Coordinate check:")
# Check typical bond length
if len(data.symbols) >= 2:
    c_pos = coords[0]
    o_pos = coords[1]
    distance = np.linalg.norm(c_pos - o_pos)
    print(f"   Distance between atoms 0-1: {distance:.3f}")
    
    if 1.0 < distance < 2.0:
        print(f"   ✅ Looks like Angstroms (typical: 1.2 Å for C-O)")
    elif 2.0 < distance < 4.0:
        print(f"   ✅ Looks like Bohr (typical: 2.3 Bohr for C-O)")
    elif distance < 1.0:
        print(f"   ❌ Too short! Units might be wrong")
    else:
        print(f"   ❌ Too long! Units might be wrong")

print(f"\n🔍 Model test RMSE:")
try:
    test_rmse = model_data['metadata']['model'].get('test_rmse_kcal', 'N/A')
    test_r2 = model_data['metadata']['model'].get('r2_score', 'N/A')
    print(f"   Test RMSE: {test_rmse} kcal/mol")
    print(f"   Test R²: {test_r2}")
    
    if isinstance(test_rmse, (int, float)) and test_rmse < 5.0:
        print(f"   ✅ Model performed well on test set")
    elif isinstance(test_rmse, (int, float)) and test_rmse > 50.0:
        print(f"   ❌ Model performed poorly even in training")
except:
    print(f"   ❓ Could not extract test metrics")

# Summary
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)

print(f"\n📊 Key findings:")
print(f"   • Model error on training data: {error_kcal:.2f} kcal/mol")
print(f"   • Your MD validation error: ~3397 kcal/mol")
print(f"   • Ratio: {3397/max(error_kcal, 0.001):.0f}x worse in MD")

if error_kcal < 1.0:
    print(f"\n💡 Conclusion:")
    print(f"   Model is GOOD, validation setup is WRONG")
    print(f"   → Focus on fixing validation (theory level, units, reference)")
elif error_kcal < 100:
    print(f"\n💡 Conclusion:")
    print(f"   Model has issues but not enough to explain MD errors")
    print(f"   → Both model AND validation need work")
else:
    print(f"\n💡 Conclusion:")
    print(f"   Model is fundamentally broken")
    print(f"   → Retrain from scratch with verified data")

print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)
print()
