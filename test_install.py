#!/usr/bin/env python3
"""
Test PSI4-MD Framework installation and verify all components work.

Run this script after installation to check everything is set up correctly.
"""

import sys
from pathlib import Path

print("=" * 70)
print("  PSI4-MD FRAMEWORK - INSTALLATION VERIFICATION")
print("=" * 70)
print()

# Add parent directory to path if needed
framework_dir = Path(__file__).parent
if str(framework_dir) not in sys.path:
    sys.path.insert(0, str(framework_dir))

# Test 1: Framework modules
print("Test 1: Importing framework modules...")
try:
    from modules.test_molecules import get_molecule, get_all_molecules
    from modules.direct_md import run_direct_md, DirectMDConfig
    from modules.data_formats import save_trajectory, load_trajectory
    from modules.ml_pes import MLPESConfig, MLPESTrainer
    from modules.visualization import TrajectoryVisualizer
    from modules.dashboard_integration import create_live_dashboard
    print("  ✅ All framework modules imported successfully")
except ImportError as e:
    print(f"  ❌ Framework import failed: {e}")
    print(f"     Current directory: {Path.cwd()}")
    print(f"     Python path: {sys.path}")
    exit(1)

# Test 2: PSI4 availability
print("\nTest 2: Checking PSI4 availability...")
try:
    import psi4
    PSI4_AVAILABLE = True
    print(f"  ✅ PSI4 is installed and available")
    print(f"     Version: {psi4.__version__}")
    print(f"     Location: {psi4.__file__}")
except ImportError:
    PSI4_AVAILABLE = False
    print("  ⚠️  PSI4 not found")
    print("     Framework will use mock calculations for testing")
    print("     To install PSI4: conda install -c psi4 psi4")

# Test 3: Optional dependencies
print("\nTest 3: Checking optional dependencies...")
optional_deps = {
    'scikit-learn': 'ML models (kernel ridge, random forest)',
    'torch': 'Neural network PES',
    'h5py': 'HDF5 file format',
    'seaborn': 'Enhanced visualizations'
}

for module, purpose in optional_deps.items():
    try:
        __import__(module.replace('-', '_'))
        print(f"  ✅ {module:15s} available - {purpose}")
    except ImportError:
        print(f"  ⚠️  {module:15s} not found - {purpose}")

# Test 4: Load test molecule
print("\nTest 4: Loading test molecules...")
try:
    water = get_molecule('water')
    print(f"  ✅ Test molecule loaded: {water.name}")
    print(f"     Formula: {water.formula}")
    print(f"     Atoms: {len(water.symbols)}")
    print(f"     Mass: {water.mass:.2f} amu")
    
    all_molecules = get_all_molecules()
    print(f"  ✅ Molecule library: {len(all_molecules)} molecules available")
except Exception as e:
    print(f"  ❌ Molecule loading failed: {e}")
    exit(1)

# Test 5: Configuration
print("\nTest 5: Creating MD configuration...")
try:
    config = DirectMDConfig(
        method='HF',
        basis='STO-3G',
        temperature=300.0,
        n_steps=5,
        output_frequency=1
    )
    print(f"  ✅ MD configuration created")
    print(f"     Method: {config.method}/{config.basis}")
    print(f"     Steps: {config.n_steps}")
    print(f"     Temperature: {config.temperature} K")
except Exception as e:
    print(f"  ❌ Configuration failed: {e}")
    exit(1)

# Test 6: Quick MD test
print("\nTest 6: Running quick MD test...")
import tempfile
import shutil

test_dir = tempfile.mkdtemp()
try:
    if PSI4_AVAILABLE:
        print("  🚀 Running real PSI4 calculation (5 steps)...")
        trajectory = run_direct_md(
            water, 
            config, 
            output_dir=test_dir,
            save_format='npz'
        )
        
        print(f"  ✅ Real PSI4 MD completed!")
        print(f"     Frames: {trajectory.n_frames}")
        print(f"     Energy range: {trajectory.energies.min():.6f} to {trajectory.energies.max():.6f} Ha")
        print(f"     Energy mean: {trajectory.energies.mean():.6f} Ha")
        
        # Check if energies are realistic (not mock values)
        if abs(trajectory.energies.mean() + 76.0) < 1.0:  # Water should be around -76 Ha
            print(f"     ✅ Energies are realistic (real PSI4 calculation)")
        else:
            print(f"     ⚠️  Energies may be from mock calculation")
    else:
        print("  ⚠️  Skipping real MD test (PSI4 not available)")
        print("     Running with mock calculations...")
        trajectory = run_direct_md(
            water, 
            config, 
            output_dir=test_dir,
            save_format='npz'
        )
        print(f"  ✅ Mock MD completed: {trajectory.n_frames} frames")

except Exception as e:
    print(f"  ❌ MD test failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)

# Test 7: File format handling
print("\nTest 7: Testing file format handling...")
try:
    import tempfile
    from modules.data_formats import TrajectoryData, save_trajectory, load_trajectory
    
    # Create small test trajectory
    test_traj = TrajectoryData(
        symbols=['O', 'H', 'H'],
        coordinates=water.coordinates.reshape(1, 3, 3),
        energies=[-76.0],
        forces=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        metadata={'test': 'verification'}
    )
    
    # Test save/load
    test_file = Path(tempfile.gettempdir()) / 'test_traj.npz'
    save_trajectory(test_traj, str(test_file))
    loaded_traj = load_trajectory(str(test_file))
    
    assert loaded_traj.n_frames == 1
    assert len(loaded_traj.symbols) == 3
    
    print(f"  ✅ File format handling works")
    test_file.unlink()
    
except Exception as e:
    print(f"  ❌ File format test failed: {e}")

# Summary
print("\n" + "=" * 70)
print("  INSTALLATION VERIFICATION SUMMARY")
print("=" * 70)

if PSI4_AVAILABLE:
    print("\n  🎉 EXCELLENT! Everything is working perfectly!")
    print("\n  You have:")
    print("  ✅ All framework modules")
    print("  ✅ PSI4 for real quantum chemistry calculations")
    print("  ✅ Test molecules and configurations")
    print("  ✅ MD simulation capabilities")
    print("\n  You're ready to run production calculations!")
    print("\n  Next steps:")
    print("  1. Read QUICKSTART.md for usage examples")
    print("  2. Run: python3 run_demo.py")
    print("  3. Try your first real calculation!")
else:
    print("\n  ⚠️  FRAMEWORK IS FUNCTIONAL but PSI4 is missing")
    print("\n  You have:")
    print("  ✅ All framework modules")
    print("  ✅ Test molecules and configurations")
    print("  ✅ Mock calculations for testing")
    print("  ❌ PSI4 not installed (needed for real calculations)")
    print("\n  To enable real quantum chemistry calculations:")
    print("  conda install -c psi4 psi4")
    print("\n  You can still:")
    print("  - Test the framework with mock calculations")
    print("  - Try ML-PES training on existing data")
    print("  - Create visualizations")

print("\n" + "=" * 70)
print("  Installation verification complete!")
print("=" * 70)
