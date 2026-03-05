#!/usr/bin/env python3
"""
IR Spectroscopy Setup Diagnostic

Checks your installation and finds available files for IR spectrum calculation.

Usage:
    python3 check_ir_setup.py
"""

import sys
from pathlib import Path

print("=" * 80)
print("  IR SPECTROSCOPY SETUP DIAGNOSTIC")
print("=" * 80)

# Check Python modules
print("\n📦 Checking Python dependencies...")

required_modules = {
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'sklearn': 'scikit-learn',
    'matplotlib': 'Matplotlib',
    'pickle': 'pickle (built-in)',
}

missing = []
for module, name in required_modules.items():
    try:
        __import__(module)
        print(f"   ✅ {name}")
    except ImportError:
        print(f"   ❌ {name} - NOT FOUND")
        missing.append(name)

if missing:
    print(f"\n❌ Missing dependencies: {', '.join(missing)}")
    print(f"\n💡 Install with:")
    print(f"   pip install numpy scipy scikit-learn matplotlib")
    sys.exit(1)

# Check PSI4 (optional)
print("\n🔬 Checking PSI4 (optional for dipole calculation)...")
try:
    import psi4
    print(f"   ✅ PSI4 {psi4.__version__}")
    psi4_available = True
except ImportError:
    print(f"   ⚠️  PSI4 not available")
    print(f"   Note: PSI4 is optional. You can train ML dipole surface instead.")
    psi4_available = False

# Check framework modules
print("\n📚 Checking framework modules...")

framework_modules = [
    'modules/data_formats.py',
    'ir_spectroscopy.py',
    'compute_ir_spectrum.py',
    'visualize_ir_spectrum.py'
]

for module in framework_modules:
    if Path(module).exists():
        print(f"   ✅ {module}")
    else:
        print(f"   ❌ {module} - NOT FOUND")
        print(f"      Make sure you're in the MD_MLPES directory")

# Find ML-PES models
print("\n🔍 Searching for ML-PES models...")

outputs_dir = Path('outputs')
if not outputs_dir.exists():
    print(f"   ⚠️  No outputs/ directory found")
    print(f"   Run complete_workflow_v2.2.py to generate training data and models")
else:
    model_files = list(outputs_dir.rglob("*mlpes*.pkl")) + list(outputs_dir.rglob("*model*.pkl"))
    model_files = [f for f in model_files if '__pycache__' not in str(f)]
    
    if model_files:
        print(f"   ✅ Found {len(model_files)} model file(s):")
        for i, f in enumerate(model_files[:5], 1):
            print(f"      [{i}] {f}")
        if len(model_files) > 5:
            print(f"      ... and {len(model_files) - 5} more")
    else:
        print(f"   ❌ No model files found")
        print(f"   Run: python3 complete_workflow_v2.2.py")

# Find training data
print("\n📊 Searching for training data...")

if outputs_dir.exists():
    data_files = list(outputs_dir.rglob("*training*.npz")) + \
                 list(outputs_dir.rglob("*augmented*.npz"))
    
    if data_files:
        print(f"   ✅ Found {len(data_files)} data file(s):")
        for i, f in enumerate(data_files[:5], 1):
            # Get file size
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"      [{i}] {f} ({size_mb:.1f} MB)")
        if len(data_files) > 5:
            print(f"      ... and {len(data_files) - 5} more")
    else:
        print(f"   ❌ No training data files found")
        print(f"   Run: python3 complete_workflow_v2.2.py")

# Check for dipoles in training data
print("\n💧 Checking for dipole moments in training data...")

if data_files:
    try:
        from modules.data_formats import load_trajectory
        
        # Check most recent file
        latest_data = sorted(data_files)[-1]
        print(f"   Checking: {latest_data.name}")
        
        data = load_trajectory(str(latest_data))
        
        if 'dipoles' in data.metadata:
            dipoles = data.metadata['dipoles']
            print(f"   ✅ Found {len(dipoles)} dipole moments in metadata")
            print(f"   You can use --train-dipole option")
        else:
            print(f"   ⚠️  No dipoles in training data")
            print(f"   Options:")
            if psi4_available:
                print(f"      1. Use PSI4 to compute dipoles (slow)")
            else:
                print(f"      1. Install PSI4 to compute dipoles")
            print(f"      2. Generate new training data with dipoles")
    
    except Exception as e:
        print(f"   ⚠️  Error checking data: {e}")

# Summary and recommendations
print("\n" + "=" * 80)
print("  SUMMARY & RECOMMENDATIONS")
print("=" * 80)

print("\n📋 Your setup:")

all_good = True

if missing:
    print(f"   ❌ Dependencies: Missing {len(missing)}")
    all_good = False
else:
    print(f"   ✅ Dependencies: All installed")

if not outputs_dir.exists() or not model_files:
    print(f"   ❌ ML-PES models: None found")
    all_good = False
else:
    print(f"   ✅ ML-PES models: {len(model_files)} available")

if not outputs_dir.exists() or not data_files:
    print(f"   ❌ Training data: None found")
    all_good = False
else:
    print(f"   ✅ Training data: {len(data_files)} available")

print("")

if all_good and model_files and data_files:
    print("🎉 Setup looks good! You're ready to compute IR spectra.")
    print("")
    print("💡 Quick start:")
    print("   bash IR_workflow_fixed.sh")
    print("")
    print("   or manually:")
    print(f"   python3 compute_ir_spectrum.py \\")
    print(f"       --model {model_files[-1]} \\")
    print(f"       --training-data {data_files[-1]} \\")
    print(f"       --train-dipole \\")
    print(f"       --temp 300 \\")
    print(f"       --output outputs/ir_test")
else:
    print("⚠️  Setup incomplete. Please address the issues above.")
    print("")
    print("💡 To get started:")
    print("   1. Install missing dependencies")
    print("   2. Generate training data:")
    print("      python3 complete_workflow_v2.2.py")
    print("   3. Train ML-PES model (if not already done)")
    print("   4. Then run IR spectroscopy workflow")

print("")
