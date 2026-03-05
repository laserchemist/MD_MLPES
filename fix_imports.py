#!/usr/bin/env python3
"""
Fix imports in all module files to use proper relative imports.

This script will automatically fix import statements in:
- direct_md.py
- ml_pes.py
- visualization.py
- dashboard_integration.py
- data_formats.py

To make them work properly as a package.
"""

from pathlib import Path
import re

def fix_file_imports(filepath):
    """Fix imports in a single file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    filename = filepath.name
    
    print(f"\n📝 Checking {filename}...")
    
    # Patterns to fix
    fixes = {
        'test_molecules': [
            (r'^from test_molecules import', 'from .test_molecules import'),
            (r'^import test_molecules', 'from . import test_molecules'),
        ],
        'data_formats': [
            (r'^from data_formats import', 'from .data_formats import'),
            (r'^import data_formats', 'from . import data_formats'),
        ],
        'visualization': [
            (r'^from visualization import', 'from .visualization import'),
        ],
        'direct_md': [
            (r'^from direct_md import', 'from .direct_md import'),
        ],
    }
    
    changes_made = 0
    
    # Apply fixes line by line
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        new_line = line
        
        # Check each pattern
        for module, patterns in fixes.items():
            for pattern, replacement in patterns:
                if re.match(pattern, line.strip()):
                    # Don't add dot if already has relative import
                    if not line.strip().startswith('from .'):
                        new_line = re.sub(pattern, replacement, line)
                        if new_line != line:
                            print(f"   Fixed: {line.strip()[:60]}")
                            print(f"      →  {new_line.strip()[:60]}")
                            changes_made += 1
                        break
        
        new_lines.append(new_line)
    
    new_content = '\n'.join(new_lines)
    
    if changes_made > 0:
        # Backup original
        backup_path = filepath.parent / f"{filepath.stem}.backup{filepath.suffix}"
        with open(backup_path, 'w') as f:
            f.write(original_content)
        
        # Write fixed version
        with open(filepath, 'w') as f:
            f.write(new_content)
        
        print(f"   ✅ Made {changes_made} changes")
        print(f"   💾 Backup saved: {backup_path.name}")
        return True
    else:
        print(f"   ✅ No changes needed")
        return False

def main():
    print("=" * 70)
    print("  FIXING MODULE IMPORTS")
    print("=" * 70)
    
    modules_dir = Path('modules')
    
    if not modules_dir.exists():
        print("\n❌ Error: modules/ directory not found!")
        print("   Current directory:", Path.cwd())
        return
    
    # Files to fix
    files_to_fix = [
        'direct_md.py',
        'ml_pes.py',
        'visualization.py',
        'dashboard_integration.py',
    ]
    
    total_fixed = 0
    
    for filename in files_to_fix:
        filepath = modules_dir / filename
        if filepath.exists():
            if fix_file_imports(filepath):
                total_fixed += 1
        else:
            print(f"\n⚠️  {filename} not found (skipping)")
    
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    if total_fixed > 0:
        print(f"\n✅ Fixed imports in {total_fixed} files")
        print("\nBackup files created with .backup extension")
        print("\nNow try:")
        print("  python3 example.py")
    else:
        print("\n✅ All imports already correct!")
        print("\nIf you still get errors, check:")
        print("  1. modules/__init__.py exists")
        print("  2. You're in the correct directory")
        print("  3. Run: python3 -c 'from modules import get_molecule; print(\"OK\")'")

if __name__ == '__main__':
    main()
