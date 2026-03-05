#!/usr/bin/env python3
"""
IMPROVED ML-PES Training Workflow

This version includes:
- Better default parameters
- More training data generation
- Data quality checks
- Outlier filtering
- Hyperparameter tuning
- Multiple model comparison

Based on lessons learned from initial poor performance.

Author: PSI4-MD Framework
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("  IMPROVED ML-PES TRAINING WORKFLOW")
print("=" * 80)

# Import framework
try:
    from modules.test_molecules import get_molecule, add_random_displacement
    from modules.direct_md import DirectMDConfig, run_direct_md
    from modules.data_formats import TrajectoryData, save_trajectory, load_trajectory
    from modules.ml_pes import MLPESConfig, MLPESTrainer, evaluate_model
    from modules.visualization import TrajectoryVisualizer, plot_training_curves
    from modules.dashboard_integration import create_live_dashboard
    print("✅ Framework imported\n")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# IMPROVED CONFIGURATION - ADJUST THESE
MOLECULE_NAME = 'water'

N_TRAJECTORIES = 8
N_STEPS_PER_TRAJ = 700
TEMPERATURES = [200, 300, 400, 500, 600, 700]  # ← Much wider!
OUTPUT_FREQUENCY = 2

# Quality control
FILTER_HIGH_FORCES = True
MAX_FORCE_THRESHOLD = 80.0   # kcal/mol/Å
MIN_ENERGY_THRESHOLD = -100.0  # Relative to mean (kcal/mol)
MAX_ENERGY_THRESHOLD = 100.0   # Relative to mean (kcal/mol)

# Create output structure
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
base_dir = Path(f'outputs/improved_mlpes_{timestamp}')
training_dir = base_dir / 'training_data'
models_dir = base_dir / 'models'
analysis_dir = base_dir / 'analysis'

for d in [training_dir, models_dir, analysis_dir]:
    d.mkdir(parents=True, exist_ok=True)

print(f"📁 Output: {base_dir}\n")

# Check PSI4
try:
    import psi4
    print(f"✅ PSI4 version {psi4.__version__}")
except ImportError:
    print("❌ PSI4 required!")
    sys.exit(1)

# ==============================================================================
# PHASE 1: GENERATE TRAINING DATA
# ==============================================================================

print("\n" + "=" * 80)
print("  PHASE 1: GENERATING TRAINING DATA")
print("=" * 80)

molecule = get_molecule(MOLECULE_NAME)
print(f"\n🧪 Molecule: {molecule.name} ({molecule.formula})")

# IMPROVED MD configuration
md_config = DirectMDConfig(
    method='B3LYP',        # Use 'B3LYP' for better accuracy (3x slower)
    basis='6-31G*',     # Better than STO-3G
    timestep=0.5,
    n_steps=N_STEPS_PER_TRAJ,
    output_frequency=OUTPUT_FREQUENCY,
    thermostat='berendsen',
    calculate_dipole=True,
    memory='2GB'
)

print(f"\n⚙️  Configuration:")
print(f"   Method: {md_config.method}/{md_config.basis}")
print(f"   Trajectories: {N_TRAJECTORIES}")
print(f"   Steps per trajectory: {N_STEPS_PER_TRAJ}")
print(f"   Output frequency: every {OUTPUT_FREQUENCY} steps")
print(f"   Temperatures: {TEMPERATURES} K")

expected_configs = N_TRAJECTORIES * (N_STEPS_PER_TRAJ // OUTPUT_FREQUENCY)
print(f"\n📊 Expected configurations: ~{expected_configs}")
print(f"   Estimated time: ~{expected_configs * 2 / 60:.1f} minutes")

trajectories = []
total_frames = 0

print(f"\n🚀 Generating trajectories...")

for i, temp in enumerate(TEMPERATURES[:N_TRAJECTORIES]):
    print(f"\n{'─' * 80}")
    print(f"  Trajectory {i+1}/{N_TRAJECTORIES}: T = {temp} K")
    print(f"{'─' * 80}")
    
    # Add variety with random displacements
    if i > 0:
        displaced_mol = add_random_displacement(
            molecule, 
            displacement_magnitude=0.05,
            random_seed=42 + i
        )
    else:
        displaced_mol = molecule
    
    md_config.temperature = temp
    traj_output = training_dir / f'traj_{i+1}_T{temp}K'
    
    print(f"   Running MD...")
    trajectory = run_direct_md(
        displaced_mol,
        md_config,
        output_dir=str(traj_output),
        save_format='npz'
    )
    
    trajectories.append(trajectory)
    total_frames += trajectory.n_frames
    
    print(f"   ✅ Complete: {trajectory.n_frames} frames")
    print(f"      Energy: {trajectory.energies.min()*627.509:.1f} to {trajectory.energies.max()*627.509:.1f} kcal/mol")

print(f"\n✅ Data generation complete: {total_frames} configurations")

# ==============================================================================
# PHASE 2: DATA QUALITY CHECKS AND FILTERING
# ==============================================================================

print("\n" + "=" * 80)
print("  PHASE 2: DATA QUALITY CHECKS")
print("=" * 80)

# Combine trajectories
print(f"\n📦 Combining trajectories...")

all_coords = []
all_energies = []
all_forces = []
all_dipoles = []
all_temps = []

for i, traj in enumerate(trajectories):
    all_coords.append(traj.coordinates)
    all_energies.append(traj.energies)
    all_forces.append(traj.forces)
    if traj.dipoles is not None:
        all_dipoles.append(traj.dipoles)
    # Track which temperature each frame came from
    all_temps.extend([TEMPERATURES[i]] * traj.n_frames)

combined_coords = np.vstack(all_coords)
combined_energies = np.concatenate(all_energies)
combined_forces = np.vstack(all_forces)
combined_dipoles = np.vstack(all_dipoles) if all_dipoles else None
combined_temps = np.array(all_temps)

print(f"   Combined: {len(combined_energies)} configurations")

# Calculate statistics
energies_kcal = combined_energies * 627.509
e_mean = energies_kcal.mean()
e_std = energies_kcal.std()
force_mags = np.linalg.norm(combined_forces, axis=2)
f_mean = force_mags.mean()
f_max = force_mags.max(axis=1)

print(f"\n📊 Data Statistics:")
print(f"   Energy: {e_mean:.2f} ± {e_std:.2f} kcal/mol")
print(f"   Range: {energies_kcal.min():.2f} to {energies_kcal.max():.2f} kcal/mol")
print(f"   Forces: {f_mean:.2f} ± {force_mags.std():.2f} kcal/mol/Å")
print(f"   Max force: {f_max.max():.2f} kcal/mol/Å")

# Quality filtering
if FILTER_HIGH_FORCES or True:  # Always check
    print(f"\n🔍 Checking data quality...")
    
    # Energy outliers
    e_min_thresh = e_mean + MIN_ENERGY_THRESHOLD
    e_max_thresh = e_mean + MAX_ENERGY_THRESHOLD
    energy_mask = (energies_kcal >= e_min_thresh) & (energies_kcal <= e_max_thresh)
    
    # Force outliers  
    force_mask = f_max <= MAX_FORCE_THRESHOLD
    
    # Combined mask
    good_mask = energy_mask & force_mask
    n_bad = (~good_mask).sum()
    
    if n_bad > 0:
        print(f"   ⚠️  Found {n_bad} outliers ({n_bad/len(good_mask)*100:.1f}%)")
        print(f"      Energy outliers: {(~energy_mask).sum()}")
        print(f"      Force outliers: {(~force_mask).sum()}")
        print(f"   🧹 Filtering...")
        
        combined_coords = combined_coords[good_mask]
        combined_energies = combined_energies[good_mask]
        combined_forces = combined_forces[good_mask]
        if combined_dipoles is not None:
            combined_dipoles = combined_dipoles[good_mask]
        combined_temps = combined_temps[good_mask]
        
        print(f"   ✅ Kept {len(combined_energies)} configurations")
    else:
        print(f"   ✅ No outliers detected")

# Create filtered trajectory
filtered_trajectory = TrajectoryData(
    symbols=molecule.symbols,
    coordinates=combined_coords,
    energies=combined_energies,
    forces=combined_forces,
    dipoles=combined_dipoles,
    metadata={
        'molecule': molecule.name,
        'method': md_config.method,
        'basis': md_config.basis,
        'n_trajectories': N_TRAJECTORIES,
        'temperatures': TEMPERATURES[:N_TRAJECTORIES],
        'total_frames': len(combined_energies),
        'filtered': FILTER_HIGH_FORCES
    }
)

# Save
combined_path = training_dir / 'filtered_training_data.npz'
save_trajectory(filtered_trajectory, str(combined_path))
print(f"\n   💾 Saved: {combined_path}")

# ==============================================================================
# PHASE 3: TRAIN MULTIPLE ML-PES MODELS
# ==============================================================================

print("\n" + "=" * 80)
print("  PHASE 3: TRAINING ML-PES MODELS")
print("=" * 80)

# Try multiple configurations
configs_to_try = [
    {
        'name': 'KRR_Coulomb_gamma0.01',
        'config': MLPESConfig(
            model_type='kernel_ridge',
            descriptor_type='coulomb_matrix',
            train_forces=True,
            force_weight=10.0,  # Increased
            kernel='rbf',
            kernel_params={'gamma': 0.01, 'alpha': 0.1},  # Tuned
            validation_split=0.2,
            random_seed=42
        )
    },
    {
        'name': 'KRR_Coulomb_gamma0.1',
        'config': MLPESConfig(
            model_type='kernel_ridge',
            descriptor_type='coulomb_matrix',
            train_forces=True,
            force_weight=10.0,
            kernel='rbf',
            kernel_params={'gamma': 0.1, 'alpha': 1.0},
            validation_split=0.2,
            random_seed=42
        )
    },
    {
        'name': 'KRR_Internals_gamma0.01',
        'config': MLPESConfig(
            model_type='kernel_ridge',
            descriptor_type='internals',  # Different descriptor
            train_forces=True,
            force_weight=10.0,
            kernel='rbf',
            kernel_params={'gamma': 0.01, 'alpha': 0.1},
            validation_split=0.2,
            random_seed=42
        )
    }
]

print(f"\n🤖 Training {len(configs_to_try)} different models...")

best_model = None
best_rmse = float('inf')
results = []

for i, config_dict in enumerate(configs_to_try):
    name = config_dict['name']
    config = config_dict['config']
    
    print(f"\n{'─' * 80}")
    print(f"  Model {i+1}/{len(configs_to_try)}: {name}")
    print(f"{'─' * 80}")
    print(f"   Model: {config.model_type}")
    print(f"   Descriptor: {config.descriptor_type}")
    print(f"   Kernel params: {config.kernel_params}")
    
    try:
        print(f"   Training...")
        trainer = MLPESTrainer(config)
        trainer.train(filtered_trajectory)
        
        # Evaluate
        metrics = evaluate_model(trainer, filtered_trajectory)
        rmse = metrics['rmse_kcal']
        
        print(f"   ✅ RMSE: {rmse:.4f} kcal/mol")
        print(f"      MAE:  {metrics['mae_kcal']:.4f} kcal/mol")
        print(f"      R²:   {metrics['r2_score']:.6f}")
        
        # Save model
        model_path = models_dir / f'{name}.pkl'
        trainer.save(str(model_path))
        
        results.append({
            'name': name,
            'rmse': rmse,
            'mae': metrics['mae_kcal'],
            'r2': metrics['r2_score'],
            'model_path': model_path,
            'trainer': trainer
        })
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = trainer
            best_name = name
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")

# Summary
print(f"\n{'=' * 80}")
print(f"  MODEL COMPARISON")
print(f"{'=' * 80}\n")

print(f"   {'Model':<30} {'RMSE (kcal/mol)':<20} {'R²':<10}")
print(f"   {'-' * 60}")
for r in sorted(results, key=lambda x: x['rmse']):
    marker = '⭐' if r['name'] == best_name else '  '
    print(f"   {marker} {r['name']:<28} {r['rmse']:>8.4f}           {r['r2']:>8.6f}")

print(f"\n✅ Best model: {best_name}")
print(f"   RMSE: {best_rmse:.4f} kcal/mol")

# ==============================================================================
# PHASE 4: TEST BEST MODEL
# ==============================================================================

print("\n" + "=" * 80)
print("  PHASE 4: TESTING BEST MODEL")
print("=" * 80)

print(f"\n🔬 Generating test trajectory...")
test_config = DirectMDConfig(
    method=md_config.method,
    basis=md_config.basis,
    temperature=300,
    timestep=0.5,
    n_steps=20,
    output_frequency=1
)

test_traj = run_direct_md(
    molecule,
    test_config,
    output_dir=str(analysis_dir / 'test_trajectory'),
    save_format='npz'
)

# Predict with best model
print(f"🤖 Predicting with best ML-PES...")
ml_energies = []
for i in range(test_traj.n_frames):
    e = best_model.predict(test_traj.symbols, test_traj.coordinates[i])
    ml_energies.append(e)
ml_energies = np.array(ml_energies)

# Compare
psi4_energies = test_traj.energies
errors = (ml_energies - psi4_energies) * 627.509

print(f"\n📊 Test Results:")
print(f"   Mean error: {errors.mean():.4f} ± {errors.std():.4f} kcal/mol")
print(f"   Max error:  {errors.max():.4f} kcal/mol")
print(f"   RMSE:       {np.sqrt((errors**2).mean()):.4f} kcal/mol")

# Create comparison plot
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

times = np.arange(len(psi4_energies)) * test_config.timestep

# Energy comparison
ax1 = axes[0]
ax1.plot(times, psi4_energies * 627.509, 'o-', label='PSI4', linewidth=2, markersize=8)
ax1.plot(times, ml_energies * 627.509, 's--', label=f'ML-PES ({best_name})', linewidth=2, markersize=6, alpha=0.7)
ax1.set_xlabel('Time (fs)', fontsize=14)
ax1.set_ylabel('Energy (kcal/mol)', fontsize=14)
ax1.set_title(f'PSI4 vs ML-PES: {best_name}', fontsize=16)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# Error
ax2 = axes[1]
ax2.plot(times, errors, 'o-', color='red', linewidth=2, markersize=8)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.fill_between(times, errors, alpha=0.3, color='red')
ax2.set_xlabel('Time (fs)', fontsize=14)
ax2.set_ylabel('Error (kcal/mol)', fontsize=14)
ax2.set_title(f'Prediction Error (RMSE: {np.sqrt((errors**2).mean()):.3f} kcal/mol)', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
comparison_path = analysis_dir / 'best_model_comparison.png'
plt.savefig(comparison_path, dpi=300)
plt.close()

print(f"\n✅ Comparison plot: {comparison_path}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 80)
print("  WORKFLOW COMPLETE")
print("=" * 80)

print(f"\n✅ Training Data:")
print(f"   • {len(filtered_trajectory.energies)} configurations")
print(f"   • {N_TRAJECTORIES} trajectories at {TEMPERATURES[:N_TRAJECTORIES]} K")
print(f"   • Method: {md_config.method}/{md_config.basis}")

print(f"\n✅ Best ML-PES: {best_name}")
print(f"   • RMSE: {best_rmse:.4f} kcal/mol")
print(f"   • Test RMSE: {np.sqrt((errors**2).mean()):.4f} kcal/mol")

if best_rmse < 2.0:
    print(f"\n🎉 EXCELLENT! ML-PES has good accuracy (< 2 kcal/mol)")
elif best_rmse < 5.0:
    print(f"\n✅ GOOD! ML-PES has acceptable accuracy (< 5 kcal/mol)")
else:
    print(f"\n⚠️  MARGINAL. For better accuracy:")
    print(f"   • Increase N_STEPS_PER_TRAJ to 500-1000")
    print(f"   • Add more trajectories (N_TRAJECTORIES = 8-10)")
    print(f"   • Try better theory level (B3LYP/6-31G*)")

print(f"\n📁 Files:")
print(f"   • Training data: {combined_path}")
print(f"   • Best model: {models_dir / f'{best_name}.pkl'}")
print(f"   • Comparison: {comparison_path}")

print("\n" + "=" * 80)
