#!/usr/bin/env python3
"""
ML-PES Training Diagnostics and Fixes

This script helps diagnose why ML-PES training failed and provides
solutions to improve model performance.

Common Issues:
1. Not enough training data
2. Poor sampling (not covering configuration space)
3. Training/test distribution mismatch
4. Wrong hyperparameters
5. Bad descriptors

Author: PSI4-MD Framework
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 80)
print("  ML-PES TRAINING DIAGNOSTICS")
print("=" * 80)

# Import framework
try:
    from modules.data_formats import load_trajectory
    from modules.ml_pes import MLPESConfig, MLPESTrainer
    from modules.test_molecules import get_molecule
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Run: python3 fix_imports.py")
    exit(1)

# Configuration - USER: UPDATE THESE PATHS
TRAINING_DATA_PATH = 'outputs/improved_mlpes_20251230_180012/training_data/combined_training_data.npz'
MODEL_PATH = 'outputs/improved_mlpes_20251230_180012/models/water_pes_model.pkl'
OUTPUT_DIR = Path('outputs/ml_pes_diagnostics')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n📊 PHASE 1: ANALYZING TRAINING DATA")
print("=" * 80)

# Load training data
try:
    train_traj = load_trajectory(TRAINING_DATA_PATH)
    print(f"\n✅ Loaded training data: {TRAINING_DATA_PATH}")
except FileNotFoundError:
    print(f"\n❌ Training data not found: {TRAINING_DATA_PATH}")
    print("\n💡 UPDATE the TRAINING_DATA_PATH variable at the top of this script")
    print("   with the actual path to your combined_training_data.npz file")
    exit(1)

print(f"\n   Training Data Statistics:")
print(f"   {'─' * 60}")
print(f"   Configurations: {train_traj.n_frames}")
print(f"   Atoms: {len(train_traj.symbols)}")
print(f"   Energy range: {train_traj.energies.min():.6f} to {train_traj.energies.max():.6f} Ha")
print(f"   Energy range: {train_traj.energies.min()*627.509:.1f} to {train_traj.energies.max()*627.509:.1f} kcal/mol")
print(f"   Energy span: {(train_traj.energies.max() - train_traj.energies.min())*627.509:.2f} kcal/mol")

# Check if we have enough data
n_configs = train_traj.n_frames
if n_configs < 200:
    print(f"\n   ⚠️  WARNING: Only {n_configs} configurations!")
    print(f"      Recommendation: Need 500-1000+ for good ML-PES")
    print(f"      Status: HIGH RISK OF POOR PERFORMANCE")
elif n_configs < 500:
    print(f"\n   ⚠️  WARNING: Only {n_configs} configurations")
    print(f"      Recommendation: Aim for 1000+ for production ML-PES")
    print(f"      Status: MARGINAL - may work for simple cases")
else:
    print(f"\n   ✅ Good! {n_configs} configurations should be sufficient")

# Analyze energy distribution
energies_kcal = train_traj.energies * 627.509
e_mean = energies_kcal.mean()
e_std = energies_kcal.std()

print(f"\n   Energy Distribution:")
print(f"   {'─' * 60}")
print(f"   Mean: {e_mean:.2f} kcal/mol")
print(f"   Std:  {e_std:.2f} kcal/mol")
print(f"   Min:  {energies_kcal.min():.2f} kcal/mol")
print(f"   Max:  {energies_kcal.max():.2f} kcal/mol")

if e_std < 1.0:
    print(f"\n   ⚠️  WARNING: Very narrow energy range (std={e_std:.2f})")
    print(f"      Problem: Training data may not cover enough of PES")
    print(f"      Solution: Run at higher temperatures or add displaced structures")

# Analyze force distribution
if train_traj.forces is not None:
    force_mags = np.linalg.norm(train_traj.forces, axis=2)  # Shape: (frames, atoms)
    force_mean = force_mags.mean()
    force_max = force_mags.max()
    
    print(f"\n   Force Distribution:")
    print(f"   {'─' * 60}")
    print(f"   Mean force: {force_mean:.2f} kcal/mol/Å")
    print(f"   Max force:  {force_max:.2f} kcal/mol/Å")
    
    if force_max > 100:
        print(f"\n   ⚠️  WARNING: Very large forces detected!")
        print(f"      Problem: Some geometries may be highly distorted")
        print(f"      Solution: Filter out high-force configurations")

# Visualize energy distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Energy histogram
ax1 = axes[0, 0]
ax1.hist(energies_kcal, bins=50, alpha=0.7, edgecolor='black')
ax1.axvline(e_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {e_mean:.1f}')
ax1.set_xlabel('Energy (kcal/mol)', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Training Data: Energy Distribution', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Energy trajectory
ax2 = axes[0, 1]
ax2.plot(energies_kcal, linewidth=1)
ax2.axhline(e_mean, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax2.set_xlabel('Configuration Index', fontsize=12)
ax2.set_ylabel('Energy (kcal/mol)', fontsize=12)
ax2.set_title('Training Data: Energy Trajectory', fontsize=14)
ax2.grid(True, alpha=0.3)

# Force distribution
if train_traj.forces is not None:
    ax3 = axes[1, 0]
    ax3.hist(force_mags.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax3.axvline(force_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {force_mean:.1f}')
    ax3.set_xlabel('Force Magnitude (kcal/mol/Å)', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Training Data: Force Distribution', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Force trajectory
    ax4 = axes[1, 1]
    ax4.plot(force_mags.max(axis=1), linewidth=1, label='Max Force')
    ax4.plot(force_mags.mean(axis=1), linewidth=1, label='Mean Force', alpha=0.7)
    ax4.set_xlabel('Configuration Index', fontsize=12)
    ax4.set_ylabel('Force Magnitude (kcal/mol/Å)', fontsize=12)
    ax4.set_title('Training Data: Force Trajectory', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
diag_plot_path = OUTPUT_DIR / 'training_data_diagnostics.png'
plt.savefig(diag_plot_path, dpi=300, bbox_inches='tight')
print(f"\n   ✅ Saved diagnostic plot: {diag_plot_path}")
plt.close()

# Check for outliers
print("\n   Checking for outliers...")
energy_threshold = e_mean + 3 * e_std
outliers = np.where(energies_kcal > energy_threshold)[0]
if len(outliers) > 0:
    print(f"   ⚠️  Found {len(outliers)} energy outliers (> 3σ)")
    print(f"      Frames: {outliers[:10]}..." if len(outliers) > 10 else f"      Frames: {outliers}")
else:
    print(f"   ✅ No energy outliers detected")

if train_traj.forces is not None:
    force_threshold = force_mean + 5 * force_mags.std()
    force_outliers = np.where(force_mags.max(axis=1) > force_threshold)[0]
    if len(force_outliers) > 0:
        print(f"   ⚠️  Found {len(force_outliers)} force outliers (> 5σ)")
        print(f"      Consider removing these frames")
    else:
        print(f"   ✅ No force outliers detected")

# Recommendations
print("\n" + "=" * 80)
print("  RECOMMENDATIONS")
print("=" * 80)

issues = []
if n_configs < 500:
    issues.append("NOT_ENOUGH_DATA")
if e_std < 1.0:
    issues.append("NARROW_ENERGY_RANGE")
if train_traj.forces is not None and force_max > 100:
    issues.append("HIGH_FORCES")

if not issues:
    print("\n✅ Training data looks good! If ML-PES still performs poorly, try:")
    print("   1. Different descriptors (internals instead of coulomb_matrix)")
    print("   2. Tune hyperparameters (gamma, alpha)")
    print("   3. Try neural network instead of kernel ridge")
else:
    print("\n🔧 ISSUES DETECTED - HERE'S HOW TO FIX THEM:\n")
    
    if "NOT_ENOUGH_DATA" in issues:
        print("   ❌ NOT ENOUGH TRAINING DATA")
        print("      Current: {} configurations".format(n_configs))
        print("      Target: 500-1000+ configurations")
        print("      Solution:")
        print("         • Increase N_STEPS_PER_TRAJ to 400-500")
        print("         • Increase N_TRAJECTORIES to 4-6")
        print("         • Set output_frequency=2 to save more frames")
    
    if "NARROW_ENERGY_RANGE" in issues:
        print("\n   ❌ NARROW ENERGY SAMPLING")
        print("      Current std: {:.2f} kcal/mol".format(e_std))
        print("      Target: 3-10 kcal/mol")
        print("      Solution:")
        print("         • Use wider temperature range: [200, 300, 400, 500] K")
        print("         • Add displaced starting geometries")
        print("         • Run longer trajectories to explore more")
    
    if "HIGH_FORCES" in issues:
        print("\n   ❌ HIGH FORCE CONFIGURATIONS")
        print("      Max force: {:.1f} kcal/mol/Å".format(force_max))
        print("      Solution:")
        print("         • Filter out frames with forces > 50 kcal/mol/Å")
        print("         • Lower temperature (< 400 K)")
        print("         • Use smaller timestep (0.25 fs)")

print("\n" + "=" * 80)
print("  IMPROVED TRAINING CONFIGURATION")
print("=" * 80)

print("""
Edit your train_mlpes_workflow.py with these settings:

# For better ML-PES performance:
N_TRAJECTORIES = 5                    # More trajectories
N_STEPS_PER_TRAJ = 400                # More steps per trajectory
TEMPERATURES = [200, 250, 300, 350, 400]  # Wider temperature range

md_config = DirectMDConfig(
    method='B3LYP',     # Better than HF (or keep HF if speed matters)
    basis='6-31G*',     # Better than STO-3G (or keep STO-3G for speed)
    timestep=0.5,
    n_steps=400,
    output_frequency=2,  # Save every 2 steps for more data
    thermostat='berendsen',
    calculate_dipole=True
)

ml_config = MLPESConfig(
    model_type='kernel_ridge',
    descriptor_type='coulomb_matrix',  # Try 'internals' if this fails
    train_forces=True,
    force_weight=10.0,                 # Increased force weight
    kernel='rbf',
    kernel_params={'gamma': 0.01, 'alpha': 0.1},  # Tuned parameters
    validation_split=0.2,
    random_seed=42
)

Expected result: ~1000 configurations → RMSE 1-3 kcal/mol
Estimated time: 2-3 hours with PSI4
""")

print("\n💡 Quick fixes to try FIRST:")
print("   1. Increase N_STEPS_PER_TRAJ from 200 → 400")
print("   2. Add more trajectories: N_TRAJECTORIES = 5")
print("   3. Set force_weight=10.0 in ml_config")
print("   4. Try gamma=0.01 instead of 0.1")

print("\n" + "=" * 80)
