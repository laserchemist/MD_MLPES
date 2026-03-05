#!/usr/bin/env python3
"""
ML-PES Training Workflow with Real PSI4 Data

This script demonstrates the complete workflow for training a machine learning
potential energy surface from ab initio molecular dynamics:

1. Generate training data with multiple MD trajectories
2. Sample different regions of configuration space
3. Train ML-PES model with cross-validation
4. Evaluate model performance
5. Use ML-PES for fast dynamics

For production ML-PES, you want:
- 500-2000+ configurations
- Multiple temperatures for better sampling
- Different starting geometries
- Both equilibrium and non-equilibrium structures

Author: PSI4-MD Framework
Date: 2025
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("  ML-PES TRAINING WORKFLOW - PRODUCTION DATA GENERATION")
print("=" * 80)

# Import framework
try:
    from modules.test_molecules import get_molecule, add_random_displacement
    from modules.direct_md import DirectMDConfig, run_direct_md
    from modules.data_formats import (
        TrajectoryData, save_trajectory, load_trajectory, convert_format
    )
    from modules.ml_pes import MLPESConfig, MLPESTrainer, evaluate_model
    from modules.visualization import TrajectoryVisualizer, plot_training_curves
    from modules.dashboard_integration import create_live_dashboard
    print("✅ Framework imported successfully\n")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Run: python3 fix_imports.py")
    sys.exit(1)

# Check dependencies
try:
    import psi4
    PSI4_VERSION = psi4.__version__
    print(f"✅ PSI4 available (version {PSI4_VERSION})")
except ImportError:
    print("❌ PSI4 not available!")
    print("   This script requires real PSI4 calculations")
    print("   Install: conda install -c psi4 psi4")
    sys.exit(1)

try:
    import sklearn
    print(f"✅ scikit-learn available (version {sklearn.__version__})")
    ML_AVAILABLE = True
except ImportError:
    print("❌ scikit-learn not available!")
    print("   Install: pip install scikit-learn")
    sys.exit(1)

# Configuration
MOLECULE_NAME = 'water'  # Change to any test molecule
N_TRAJECTORIES = 3       # Number of different trajectories
N_STEPS_PER_TRAJ = 200   # Steps per trajectory (adjust based on time)
TEMPERATURES = [200, 300, 400]  # Different temperatures for better sampling

N_TRAJECTORIES = 5
N_STEPS_PER_TRAJ = 400
TEMPERATURES = [200, 250, 300, 350, 400]

# Create output structure
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
base_dir = Path(f'outputs/ml_pes_workflow_{timestamp}')
training_dir = base_dir / 'training_data'
models_dir = base_dir / 'models'
analysis_dir = base_dir / 'analysis'

for d in [training_dir, models_dir, analysis_dir]:
    d.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Output directory: {base_dir}")
print(f"   Training data: {training_dir}")
print(f"   Models: {models_dir}")
print(f"   Analysis: {analysis_dir}")

# ==============================================================================
# PHASE 1: GENERATE TRAINING DATA
# ==============================================================================

print("\n" + "=" * 80)
print("  PHASE 1: GENERATING TRAINING DATA WITH PSI4")
print("=" * 80)

molecule = get_molecule(MOLECULE_NAME)
print(f"\n🧪 Molecule: {molecule.name} ({molecule.formula})")
print(f"   Atoms: {len(molecule.symbols)}")
print(f"   Reference energy: {molecule.reference_energy:.6f} Ha")

# MD configuration
md_config = DirectMDConfig(
    method = 'B3LYP', 
    basis = '6-31+G**',
    timestep=0.5,
    n_steps=N_STEPS_PER_TRAJ,
    output_frequency=2,  # Save every 2 steps
    thermostat='berendsen',
    calculate_dipole=True,
    memory='2GB',
    threads=1
)

print(f"\n⚙️  MD Configuration:")
print(f"   Method: {md_config.method}/{md_config.basis}")
print(f"   Timestep: {md_config.timestep} fs")
print(f"   Steps per trajectory: {md_config.n_steps}")
print(f"   Output frequency: every {md_config.output_frequency} steps")

trajectories = []
total_frames = 0

print(f"\n🚀 Generating {N_TRAJECTORIES} trajectories at different conditions...")
print(f"   (This will take some time with real PSI4 calculations)")

for i, temp in enumerate(TEMPERATURES[:N_TRAJECTORIES]):
    print(f"\n{'─' * 80}")
    print(f"  Trajectory {i+1}/{N_TRAJECTORIES}: T = {temp} K")
    print(f"{'─' * 80}")
    
    # Add small random displacement for variety
    if i > 0:
        displaced_mol = add_random_displacement(
            molecule, 
            displacement_magnitude=0.05,
            random_seed=42 + i
        )
    else:
        displaced_mol = molecule
    
    # Update config for this trajectory
    md_config.temperature = temp
    
    # Run MD
    traj_output = training_dir / f'trajectory_{i+1}_T{temp}K'
    
    print(f"\n   Starting MD simulation...")
    print(f"   Estimated time: ~{N_STEPS_PER_TRAJ * 2} seconds")
    
    trajectory = run_direct_md(
        displaced_mol,
        md_config,
        output_dir=str(traj_output),
        save_format='npz'
    )
    
    trajectories.append(trajectory)
    total_frames += trajectory.n_frames
    
    print(f"\n   ✅ Trajectory {i+1} complete!")
    print(f"      Frames: {trajectory.n_frames}")
    print(f"      Energy range: {trajectory.energies.min():.6f} to {trajectory.energies.max():.6f} Ha")
    print(f"      Energy range: {trajectory.energies.min()*627.509:.1f} to {trajectory.energies.max()*627.509:.1f} kcal/mol")

print(f"\n{'=' * 80}")
print(f"  DATA GENERATION COMPLETE")
print(f"{'=' * 80}")
print(f"\n📊 Total configurations generated: {total_frames}")
print(f"   From {N_TRAJECTORIES} trajectories")
print(f"   Temperatures: {TEMPERATURES[:N_TRAJECTORIES]} K")

# Combine trajectories
print(f"\n📦 Combining trajectories into single training set...")

all_coords = []
all_energies = []
all_forces = []
all_dipoles = []

for traj in trajectories:
    all_coords.append(traj.coordinates)
    all_energies.append(traj.energies)
    all_forces.append(traj.forces)
    if traj.dipoles is not None:
        all_dipoles.append(traj.dipoles)

combined_trajectory = TrajectoryData(
    symbols=molecule.symbols,
    coordinates=np.vstack(all_coords),
    energies=np.concatenate(all_energies),
    forces=np.vstack(all_forces),
    dipoles=np.vstack(all_dipoles) if all_dipoles else None,
    metadata={
        'molecule': molecule.name,
        'method': md_config.method,
        'basis': md_config.basis,
        'n_trajectories': N_TRAJECTORIES,
        'temperatures': TEMPERATURES[:N_TRAJECTORIES],
        'total_frames': total_frames
    }
)

# Save combined trajectory
combined_path = training_dir / 'combined_training_data.npz'
save_trajectory(combined_trajectory, str(combined_path))

print(f"   ✅ Combined trajectory saved: {combined_path}")
print(f"   Total frames: {combined_trajectory.n_frames}")

# ==============================================================================
# PHASE 2: TRAIN ML-PES MODEL
# ==============================================================================

print("\n" + "=" * 80)
print("  PHASE 2: TRAINING ML POTENTIAL ENERGY SURFACE")
print("=" * 80)

print(f"\n🤖 Training ML-PES on {combined_trajectory.n_frames} configurations...")

# ML-PES configuration
ml_config = MLPESConfig(
    model_type='kernel_ridge',
    descriptor_type='coulomb_matrix',
    train_forces=True,
    force_weight=1.0,
    kernel='rbf',
    kernel_params={'gamma': 0.1, 'alpha': 1.0},
    validation_split=0.2,
    random_seed=42
)

print(f"\n⚙️  ML-PES Configuration:")
print(f"   Model: {ml_config.model_type}")
print(f"   Descriptor: {ml_config.descriptor_type}")
print(f"   Training on forces: {ml_config.train_forces}")
print(f"   Validation split: {ml_config.validation_split}")
print(f"   Kernel: {ml_config.kernel}")

# Train model
print(f"\n🧠 Training model...")
trainer = MLPESTrainer(ml_config)
trainer.train(combined_trajectory)

# Save model
model_path = models_dir / f'{molecule.name}_pes_model.pkl'
trainer.save(str(model_path))

print(f"\n✅ ML-PES training complete!")
print(f"   Model saved: {model_path}")

# ==============================================================================
# PHASE 3: EVALUATE MODEL PERFORMANCE
# ==============================================================================

print("\n" + "=" * 80)
print("  PHASE 3: EVALUATING ML-PES PERFORMANCE")
print("=" * 80)

# Evaluate on training data
print(f"\n📊 Evaluating model on training data...")
metrics = evaluate_model(trainer, combined_trajectory)

print(f"\n   Model Performance Metrics:")
print(f"   {'─' * 60}")
print(f"   RMSE (energy):     {metrics['rmse_kcal']:.4f} kcal/mol")
print(f"   MAE (energy):      {metrics['mae_kcal']:.4f} kcal/mol")
print(f"   R² score:          {metrics['r2_score']:.6f}")
print(f"   Max error:         {metrics['max_error_kcal']:.4f} kcal/mol")
print(f"   {'─' * 60}")

# Test prediction on reference structure
print(f"\n🧪 Testing prediction on reference structure...")
ref_energy = trainer.predict(molecule.symbols, molecule.coordinates)
print(f"   ML-PES prediction: {ref_energy:.6f} Ha ({ref_energy*627.509:.1f} kcal/mol)")
print(f"   Reference energy:  {molecule.reference_energy:.6f} Ha ({molecule.reference_energy*627.509:.1f} kcal/mol)")
print(f"   Difference:        {abs(ref_energy - molecule.reference_energy)*627.509:.2f} kcal/mol")

# Generate test trajectory to compare
print(f"\n🔬 Generating test trajectory with PSI4 for comparison...")
test_config = DirectMDConfig(
    method=md_config.method,
    basis=md_config.basis,
    temperature=300,
    timestep=0.5,
    n_steps=20,  # Short test
    output_frequency=1
)

test_traj = run_direct_md(
    molecule,
    test_config,
    output_dir=str(analysis_dir / 'test_trajectory'),
    save_format='npz'
)

# Predict on test trajectory with ML-PES
print(f"\n🤖 Predicting energies with ML-PES on test trajectory...")
ml_energies = []
for i in range(test_traj.n_frames):
    e = trainer.predict(test_traj.symbols, test_traj.coordinates[i])
    ml_energies.append(e)

ml_energies = np.array(ml_energies)

# Compare
psi4_energies = test_traj.energies
errors = (ml_energies - psi4_energies) * 627.509  # kcal/mol

print(f"\n   Comparison on test trajectory:")
print(f"   {'─' * 60}")
print(f"   PSI4 energy range:  {psi4_energies.min():.6f} to {psi4_energies.max():.6f} Ha")
print(f"   ML-PES range:       {ml_energies.min():.6f} to {ml_energies.max():.6f} Ha")
print(f"   Mean error:         {errors.mean():.4f} kcal/mol")
print(f"   Std error:          {errors.std():.4f} kcal/mol")
print(f"   Max error:          {errors.max():.4f} kcal/mol")
print(f"   {'─' * 60}")

# ==============================================================================
# PHASE 4: VISUALIZE RESULTS
# ==============================================================================

print("\n" + "=" * 80)
print("  PHASE 4: GENERATING VISUALIZATIONS")
print("=" * 80)

viz_dir = analysis_dir / 'visualizations'
viz_dir.mkdir(exist_ok=True)

# Visualize training data
print(f"\n📊 Creating training data visualizations...")
viz_training = TrajectoryVisualizer(combined_trajectory, output_dir=str(viz_dir / 'training'))
viz_training.plot_energy_trajectory()
viz_training.plot_force_magnitudes()
viz_training.plot_summary_dashboard()

# Visualize test trajectory
print(f"📊 Creating test trajectory visualizations...")
viz_test = TrajectoryVisualizer(test_traj, output_dir=str(viz_dir / 'test'))
viz_test.plot_energy_trajectory()
viz_test.plot_summary_dashboard()

# Create comparison plot
print(f"📊 Creating PSI4 vs ML-PES comparison plot...")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Energy comparison
ax1 = axes[0]
times = np.arange(len(psi4_energies)) * test_config.timestep
ax1.plot(times, psi4_energies * 627.509, 'o-', label='PSI4 (HF/STO-3G)', linewidth=2, markersize=8)
ax1.plot(times, ml_energies * 627.509, 's--', label='ML-PES', linewidth=2, markersize=6, alpha=0.7)
ax1.set_xlabel('Time (fs)', fontsize=14)
ax1.set_ylabel('Energy (kcal/mol)', fontsize=14)
ax1.set_title(f'PSI4 vs ML-PES: Energy Comparison ({molecule.name})', fontsize=16)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# Error plot
ax2 = axes[1]
ax2.plot(times, errors, 'o-', color='red', linewidth=2, markersize=8)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.fill_between(times, errors, alpha=0.3, color='red')
ax2.set_xlabel('Time (fs)', fontsize=14)
ax2.set_ylabel('Error (kcal/mol)', fontsize=14)
ax2.set_title(f'ML-PES Prediction Error (Mean: {errors.mean():.3f} ± {errors.std():.3f} kcal/mol)', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
comparison_plot = viz_dir / 'psi4_vs_mlpes_comparison.png'
plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Saved: {comparison_plot}")

# Plot training curves if available
if trainer.training_history:
    print(f"📊 Creating training curves...")
    try:
        plot_training_curves(
            trainer.training_history,
            str(viz_dir / 'training_curves.png')
        )
    except Exception as e:
        print(f"   ⚠️  Could not create training curves: {e}")

print(f"\n   ✅ Visualizations saved to: {viz_dir}")

# ==============================================================================
# PHASE 5: GENERATE DASHBOARD
# ==============================================================================

print("\n" + "=" * 80)
print("  PHASE 5: GENERATING WEB DASHBOARD")
print("=" * 80)

try:
    dashboard_path = create_live_dashboard(str(base_dir))
    if dashboard_path:
        print(f"\n✅ Dashboard generated!")
        print(f"   {dashboard_path}")
    else:
        print(f"\n⚠️  Dashboard generation skipped")
except Exception as e:
    print(f"\n⚠️  Dashboard generation failed: {e}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 80)
print("  WORKFLOW COMPLETE - SUMMARY")
print("=" * 80)

print(f"\n✅ Training Data Generation:")
print(f"   • {N_TRAJECTORIES} trajectories computed with PSI4")
print(f"   • {total_frames} total configurations")
print(f"   • Temperatures: {TEMPERATURES[:N_TRAJECTORIES]} K")
print(f"   • Method: {md_config.method}/{md_config.basis}")

print(f"\n✅ ML-PES Model:")
print(f"   • Model type: {ml_config.model_type}")
print(f"   • Descriptor: {ml_config.descriptor_type}")
print(f"   • RMSE: {metrics['rmse_kcal']:.4f} kcal/mol")
print(f"   • R² score: {metrics['r2_score']:.6f}")

print(f"\n✅ Test Performance:")
print(f"   • Mean error: {errors.mean():.4f} ± {errors.std():.4f} kcal/mol")
print(f"   • Max error: {errors.max():.4f} kcal/mol")

print(f"\n📁 Output Files:")
print(f"   • Training data: {combined_path}")
print(f"   • ML-PES model: {model_path}")
print(f"   • Visualizations: {viz_dir}")
print(f"   • Comparison plot: {comparison_plot}")

print(f"\n💡 Next Steps:")
print(f"   1. View comparison: open {comparison_plot}")
print(f"   2. Inspect training data: open {viz_dir / 'training'}")
print(f"   3. Load model: trainer = MLPESTrainer.load('{model_path}')")
print(f"   4. For production: Increase n_steps to 500-1000 per trajectory")
print(f"   5. For better accuracy: Use larger basis set (6-31G*, 6-31+G**)")
print(f"   6. For more sampling: Add more trajectories at different temps")

print(f"\n🎉 ML-PES training workflow complete!")
print(f"\n⚠️  Note: For production ML-PES, you should:")
print(f"   - Use 1000-2000+ configurations")
print(f"   - Sample multiple temperatures (200-500 K)")
print(f"   - Use better theory level (B3LYP/6-31G* or higher)")
print(f"   - Include equilibrium + non-equilibrium structures")

print("\n" + "=" * 80)
