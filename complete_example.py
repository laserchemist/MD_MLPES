#!/usr/bin/env python3
"""
Complete PSI4-MD Framework Example with Dashboard - CORRECTED

This example demonstrates the full workflow with REAL PSI4 calculations:
1. Loading test molecules
2. Running MD simulations with PSI4
3. Training ML-PES models
4. Creating visualizations
5. Generating interactive web dashboard

Requires: PSI4 installed (conda install -c psi4 psi4)
Optional: scikit-learn for ML-PES

Author: PSI4-MD Framework
"""

import sys
from pathlib import Path

print("=" * 70)
print("  PSI4-MD FRAMEWORK - COMPLETE EXAMPLE WITH DASHBOARD")
print("=" * 70)

# Step 0: Import framework components
print("\n📦 Step 1: Importing framework...")
try:
    from modules.test_molecules import get_molecule, get_all_molecules
    from modules.direct_md import DirectMDConfig, run_direct_md
    from modules.data_formats import save_trajectory, load_trajectory
    from modules.visualization import TrajectoryVisualizer
    from modules.dashboard_integration import create_live_dashboard
    print("   ✅ Framework imported successfully")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    print("\n   FIX: Run fix_imports.py first!")
    print("   python3 fix_imports.py")
    sys.exit(1)

# Check PSI4
try:
    import psi4
    print(f"   ✅ PSI4 available (version {psi4.__version__})")
    PSI4_AVAILABLE = True
except ImportError:
    print("   ⚠️  PSI4 not available - will use mock calculations")
    PSI4_AVAILABLE = False

# Try importing ML-PES (optional)
try:
    from modules.ml_pes import MLPESConfig, MLPESTrainer
    ML_AVAILABLE = True
    print("   ✅ ML-PES module available")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"   ⚠️  ML-PES not available (install scikit-learn)")

# Step 1: Display available molecules
print("\n" + "=" * 70)
print("📚 Step 2: Available Test Molecules")
print("=" * 70)

molecules = get_all_molecules()
print(f"\nFound {len(molecules)} test molecules:")
for name, mol in molecules.items():
    print(f"   • {name:20s} {mol.formula:8s} - {mol.description[:40]}")

# Step 2: Load a molecule
print("\n" + "=" * 70)
print("🧪 Step 3: Loading Water Molecule")
print("=" * 70)

water = get_molecule('water')
print(f"\nLoaded: {water.name}")
print(f"Formula: {water.formula}")
print(f"Atoms: {len(water.symbols)}")
print(f"Mass: {water.mass:.2f} amu")
print(f"Reference Energy: {water.reference_energy:.4f} Ha ({water.reference_method})")

# Step 3: Configure and run MD simulation
print("\n" + "=" * 70)
print("🚀 Step 4: Running Molecular Dynamics")
print("=" * 70)

# Create output directory
output_dir = Path('outputs/example_workflow')
output_dir.mkdir(parents=True, exist_ok=True)

# Configure MD
config = DirectMDConfig(
    method='HF',
    basis='STO-3G',
    temperature=300.0,
    timestep=0.5,  # fs
    n_steps=50,    # Short trajectory for demo
    output_frequency=5,
    thermostat='berendsen',
    calculate_dipole=True
)

print(f"\nMD Configuration:")
print(f"   Method: {config.method}/{config.basis}")
print(f"   Temperature: {config.temperature} K")
print(f"   Steps: {config.n_steps} ({config.timestep} fs timestep)")
print(f"   Thermostat: {config.thermostat}")

if PSI4_AVAILABLE:
    print(f"\n🔬 Running REAL PSI4 simulation...")
else:
    print(f"\n⚠️  Running mock simulation (PSI4 not available)...")

trajectory = run_direct_md(
    water,
    config,
    output_dir=str(output_dir / 'md_simulation'),
    save_format='npz'
)

print(f"\n✅ MD Simulation Complete!")
print(f"   Frames generated: {trajectory.n_frames}")
print(f"   Energy range: {trajectory.energies.min():.6f} to {trajectory.energies.max():.6f} Ha")
if PSI4_AVAILABLE:
    print(f"   Energy range: {trajectory.energies.min()*627.509:.1f} to {trajectory.energies.max()*627.509:.1f} kcal/mol")
print(f"   Trajectory saved: {output_dir / 'md_simulation' / 'water_trajectory.npz'}")

# Step 4: Create visualizations
print("\n" + "=" * 70)
print("📊 Step 5: Creating Visualizations")
print("=" * 70)

viz_dir = output_dir / 'visualizations'
viz_dir.mkdir(exist_ok=True)

viz = TrajectoryVisualizer(trajectory, output_dir=str(viz_dir))

print("\nGenerating plots...")

try:
    print("   1. Energy trajectory...")
    viz.plot_energy_trajectory()
except Exception as e:
    print(f"      ⚠️  Failed: {e}")

try:
    print("   2. Force magnitudes...")
    viz.plot_force_magnitudes()  # Corrected method name!
except Exception as e:
    print(f"      ⚠️  Failed: {e}")

if trajectory.dipoles is not None:
    try:
        print("   3. Dipole trajectory...")
        viz.plot_dipole_trajectory()
    except Exception as e:
        print(f"      ⚠️  Failed: {e}")

try:
    print("   4. 3D molecular structure...")
    viz.plot_3d_structure(frame_index=0)
except Exception as e:
    print(f"      ⚠️  Failed: {e}")

try:
    print("   5. Summary dashboard...")
    viz.plot_summary_dashboard()
except Exception as e:
    print(f"      ⚠️  Failed: {e}")

plot_count = len(list(viz_dir.glob('*.png')))
print(f"\n✅ Visualizations saved to: {viz_dir}")
print(f"   Generated {plot_count} plots")

# Step 5: Train ML-PES (if available)
if ML_AVAILABLE:
    print("\n" + "=" * 70)
    print("🤖 Step 6: Training ML Potential Energy Surface")
    print("=" * 70)
    
    try:
        ml_dir = output_dir / 'ml_models'
        ml_dir.mkdir(exist_ok=True)
        
        # Configure ML-PES
        ml_config = MLPESConfig(
            model_type='kernel_ridge',
            descriptor_type='coulomb_matrix',
            train_forces=True,
            kernel='rbf',
            kernel_params={'gamma': 0.1, 'alpha': 1.0},
            validation_split=0.2,
            random_seed=42
        )
        
        print(f"\nML-PES Configuration:")
        print(f"   Model: {ml_config.model_type}")
        print(f"   Descriptor: {ml_config.descriptor_type}")
        print(f"   Train forces: {ml_config.train_forces}")
        
        print(f"\n🧠 Training model...")
        
        trainer = MLPESTrainer(ml_config)
        trainer.train(trajectory)
        
        # Save model
        model_path = ml_dir / 'water_pes.pkl'
        trainer.save(str(model_path))
        
        print(f"\n✅ ML-PES Training Complete!")
        print(f"   Model saved: {model_path}")
        
        # Test prediction
        test_energy = trainer.predict(water.symbols, water.coordinates)
        print(f"   Test prediction: {test_energy:.6f} Ha")
        print(f"   Reference: {water.reference_energy:.6f} Ha")
        print(f"   Difference: {abs(test_energy - water.reference_energy)*627.509:.1f} kcal/mol")
        
    except Exception as e:
        print(f"\n⚠️  ML training failed: {e}")
        print("   This is OK - continuing with dashboard generation")
else:
    print("\n" + "=" * 70)
    print("⚠️  Step 6: ML-PES Training Skipped")
    print("=" * 70)
    print("\n   ML-PES module not available")
    print("   Install scikit-learn to enable: pip install scikit-learn")

# Step 6: Generate web dashboard
print("\n" + "=" * 70)
print("🌐 Step 7: Generating Web Dashboard")
print("=" * 70)

print(f"\n📊 Scanning outputs directory...")
try:
    dashboard_path = create_live_dashboard(str(output_dir))
    
    if dashboard_path:
        print(f"\n✅ Dashboard Generated Successfully!")
        print(f"\n📂 Dashboard files:")
        print(f"   • HTML: {dashboard_path}")
        print(f"   • Data: {dashboard_path.parent / 'dashboard_data.json'}")
        
        print(f"\n🌐 TO VIEW THE DASHBOARD:")
        print(f"\n   Method 1 (Mac - Easiest):")
        print(f"      open {dashboard_path}")
        
        print(f"\n   Method 2 (Any Browser):")
        print(f"      Drag this file into your browser: {dashboard_path}")
        
        print(f"\n   Method 3 (Local Web Server):")
        print(f"      cd {dashboard_path.parent}")
        print(f"      python3 -m http.server 8000")
        print(f"      Then open: http://localhost:8000/dashboard.html")
    else:
        print("\n⚠️  Dashboard generation failed (template not found)")
except Exception as e:
    print(f"\n⚠️  Dashboard generation failed: {e}")
    dashboard_path = None

# Step 7: Summary
print("\n" + "=" * 70)
print("📋 WORKFLOW COMPLETE - SUMMARY")
print("=" * 70)

print(f"\n✅ Generated Files:")
print(f"   • MD trajectory: {trajectory.n_frames} frames")
if PSI4_AVAILABLE:
    print(f"     (REAL PSI4 calculations: HF/STO-3G)")
else:
    print(f"     (Mock calculations - install PSI4 for real QM)")
print(f"   • Visualizations: {plot_count} plots")
if ML_AVAILABLE:
    print(f"   • ML-PES model: Trained and saved")
if dashboard_path:
    print(f"   • Web dashboard: Generated")

print(f"\n📁 Output Directory: {output_dir.absolute()}")
print(f"   └── md_simulation/")
print(f"       └── water_trajectory.npz")
print(f"   └── visualizations/")
print(f"       ├── energy_trajectory.png")
print(f"       ├── force_magnitudes.png")
if trajectory.dipoles is not None:
    print(f"       ├── dipole_trajectory.png")
print(f"       ├── 3d_structure_frame_0.png")
print(f"       └── summary_dashboard.png")
if ML_AVAILABLE:
    print(f"   └── ml_models/")
    print(f"       └── water_pes.pkl")
if dashboard_path:
    print(f"   └── dashboard.html              ← OPEN THIS!")
    print(f"   └── dashboard_data.json")

print(f"\n🎉 Example Complete!")

if PSI4_AVAILABLE:
    print(f"\n🎊 CONGRATULATIONS! You ran real PSI4 quantum chemistry calculations!")
    print(f"   Your trajectory contains real ab initio energies and forces.")

print(f"\n💡 Next Steps:")
if dashboard_path:
    print(f"   1. 🌐 View dashboard: open {dashboard_path}")
else:
    print(f"   1. 🌐 View dashboard: open {output_dir}/dashboard.html")
print(f"   2. 📊 Check plots: ls {viz_dir}")
print(f"   3. 📈 Load trajectory: python3 -c 'from modules.data_formats import load_trajectory; t = load_trajectory(\"{output_dir / 'md_simulation' / 'water_trajectory.npz'}\")'")
print(f"   4. 🧪 Try other molecules: python3 -c 'from modules.test_molecules import get_molecule; m = get_molecule(\"formaldehyde\")'")

print("\n" + "=" * 70)
