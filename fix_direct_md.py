#!/usr/bin/env python3
"""
Quick fix for direct_md.py import issue.
This is the specific file causing the error in example.py
"""

from pathlib import Path

def fix_direct_md():
    filepath = Path('modules/direct_md.py')
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        print(f"   Current directory: {Path.cwd()}")
        return False
    
    print(f"📝 Fixing {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Backup
    backup = filepath.parent / 'direct_md.py.backup'
    with open(backup, 'w') as f:
        f.write(content)
    print(f"   💾 Backup saved: {backup}")
    
    # Fix the specific problematic imports
    replacements = [
        ('from test_molecules import', 'from .test_molecules import'),
        ('from data_formats import', 'from .data_formats import'),
        ('import test_molecules\n', 'from . import test_molecules\n'),
        ('import data_formats\n', 'from . import data_formats\n'),
    ]
    
    changes = 0
    for old, new in replacements:
        if old in content and new not in content:
            content = content.replace(old, new)
            changes += 1
            print(f"   ✅ Fixed: {old[:40]}")
    
    # Write fixed version
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"\n✅ Fixed {changes} imports in direct_md.py")
    return True

if __name__ == '__main__':
    print("=" * 60)
    print("  QUICK FIX: direct_md.py imports")
    print("=" * 60)
    print()
    
    if fix_direct_md():
        print("\n" + "=" * 60)
        print("  TEST IT")
        print("=" * 60)
        print("\nNow try:")
        print("  python3 example.py")
        print("\nOr test import directly:")
        print("  python3 -c 'from modules.ml_pes import MLPESConfig; print(\"✅ Success!\")'")
    else:
        print("\n❌ Fix failed. Are you in the MD_MLPES directory?")
