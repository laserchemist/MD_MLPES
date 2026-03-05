#!/usr/bin/env python3
"""
Rebuild Database with Improved Metadata Extraction

This fixes the "unknown/unknown" issue by better extracting metadata
and inferring molecule names from formulas.
"""

import sys
from pathlib import Path

print("=" * 80)
print("  REBUILDING DATABASE")
print("=" * 80)

# Check if database exists
if Path('computations_db.json').exists():
    backup = input("\n📂 Database exists. Create backup? [y/n]: ").strip().lower()
    if backup == 'y':
        import shutil
        shutil.copy('computations_db.json', 'computations_db.backup.json')
        print("   ✅ Backup created: computations_db.backup.json")

# Import database module
try:
    from computation_database import ComputationDatabase, scan_and_register
except ImportError:
    print("\n❌ Cannot import computation_database.py")
    print("   Make sure it's in the current directory")
    sys.exit(1)

# Create new database
db = ComputationDatabase()

# Scan with verbose output
print("\n" + "=" * 80)
print("  SCANNING WITH IMPROVED METADATA EXTRACTION")
print("=" * 80)

scan_and_register(db, verbose=True)

# Show results
print("\n" + "=" * 80)
print("  DATABASE REBUILT")
print("=" * 80)

db.list_all()

print("\n✅ Database rebuilt successfully!")
print("   Run: python3 complete_workflow_v2.2.py")
