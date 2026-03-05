#!/usr/bin/env python3
"""
One-command fix for ml_pes.py import error.

Fixes line 54:
  from data_formats import TrajectoryData, load_trajectory
to:
  from .data_formats import TrajectoryData, load_trajectory
"""

from pathlib import Path

print("🔧 Fixing ml_pes.py imports...")

filepath = Path('modules/ml_pes.py')

if not filepath.exists():
    print(f"❌ File not found: {filepath}")
    print(f"   Current directory: {Path.cwd()}")
    exit(1)

# Read file
with open(filepath, 'r') as f:
    content = f.read()

# Backup
backup = filepath.parent / 'ml_pes.py.backup'
with open(backup, 'w') as f:
    f.write(content)

# Fix the import
old_line = 'from data_formats import TrajectoryData, load_trajectory'
new_line = 'from .data_formats import TrajectoryData, load_trajectory'

if old_line in content:
    content = content.replace(old_line, new_line)
    
    # Write fixed version
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed ml_pes.py")
    print(f"   Changed: {old_line}")
    print(f"   To:      {new_line}")
    print(f"   Backup:  {backup}")
    
    print("\n✅ Now try:")
    print("   python3 example.py")
    print("\nOr run the complete example:")
    print("   python3 complete_example.py")
else:
    print("✅ Import already fixed!")
    print("   If you still get errors, try:")
    print("   python3 complete_example.py")
