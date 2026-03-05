#!/usr/bin/env python3
"""
Quick diagnostic and fix for MD_MLPES imports
"""

import sys
from pathlib import Path

print("🔍 Diagnosing import issues...")
print("=" * 60)

# Check current directory
cwd = Path.cwd()
print(f"Current directory: {cwd}")

# Check if modules directory exists
modules_dir = cwd / 'modules'
print(f"\n✅ modules/ directory exists: {modules_dir.exists()}")

if modules_dir.exists():
    # Check __init__.py
    init_file = modules_dir / '__init__.py'
    print(f"✅ modules/__init__.py exists: {init_file.exists()}")
    
    if init_file.exists():
        print(f"   Size: {init_file.stat().st_size} bytes")
        if init_file.stat().st_size == 0:
            print("   ⚠️  WARNING: __init__.py is empty!")
    
    # List module files
    py_files = list(modules_dir.glob('*.py'))
    py_files = [f.name for f in py_files if not f.name.startswith('__')]
    print(f"\n📁 Found {len(py_files)} module files:")
    for f in sorted(py_files):
        print(f"   - {f}")

# Try importing
print("\n" + "=" * 60)
print("🧪 Testing imports...")

# Add current directory to path
if str(cwd) not in sys.path:
    sys.path.insert(0, str(cwd))
    print(f"✅ Added {cwd} to Python path")

# Test 1: Can we import the modules package?
try:
    import modules
    print("✅ import modules - SUCCESS")
except ImportError as e:
    print(f"❌ import modules - FAILED: {e}")
    print("\n💡 FIX: Create/fix modules/__init__.py")
    print("   Run: python3 create_init.py")

# Test 2: Can we import test_molecules?
try:
    from modules import test_molecules
    print("✅ from modules import test_molecules - SUCCESS")
except ImportError as e:
    print(f"❌ from modules import test_molecules - FAILED: {e}")

# Test 3: Can we import get_molecule function?
try:
    from modules.test_molecules import get_molecule
    print("✅ from modules.test_molecules import get_molecule - SUCCESS")
    
    # Try to actually use it
    water = get_molecule('water')
    if water:
        print(f"✅ get_molecule('water') - SUCCESS")
        print(f"   Loaded: {water.name} ({water.formula})")
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Function call failed: {e}")

print("\n" + "=" * 60)
print("📋 SUMMARY")
print("=" * 60)

# Check what needs to be fixed
needs_init = not (modules_dir / '__init__.py').exists()
init_empty = False
if (modules_dir / '__init__.py').exists():
    init_empty = (modules_dir / '__init__.py').stat().st_size == 0

if needs_init or init_empty:
    print("\n❌ PROBLEM: modules/__init__.py is missing or empty")
    print("\n✅ SOLUTION: Run this to create it:")
    print("\ncat > modules/__init__.py << 'EOF'")
    print('"""')
    print("PSI4-MD Framework - Core Modules")
    print('"""')
    print()
    print("__version__ = '1.0.0'")
    print()
    print("# Import main components for easy access")
    print("from .test_molecules import TestMolecule, get_molecule, get_all_molecules")
    print("from .data_formats import TrajectoryData, save_trajectory, load_trajectory")
    print("from .direct_md import DirectMDConfig, run_direct_md")
    print("try:")
    print("    from .ml_pes import MLPESConfig, MLPESTrainer, train_pes")
    print("except ImportError:")
    print("    pass")
    print("try:")
    print("    from .visualization import TrajectoryVisualizer")
    print("except ImportError:")
    print("    pass")
    print("EOF")
else:
    print("\n✅ Everything looks good!")
    print("\nIf imports still fail, try:")
    print("  python3 -c 'from modules.test_molecules import get_molecule; print(\"OK\")'")
