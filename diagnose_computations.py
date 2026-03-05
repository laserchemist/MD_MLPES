#!/usr/bin/env python3
"""
Diagnostic Tool for ML-PES Computations

Checks all training data files and reports:
- What data exists
- What metadata is present
- What's in the database
- What needs to be fixed

Usage:
    python3 diagnose_computations.py
"""

import sys
import glob
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

print("=" * 80)
print("  COMPUTATION DIAGNOSTIC TOOL")
print("=" * 80)

# ==============================================================================
# SCAN FILES
# ==============================================================================

print("\n🔍 STEP 1: Scanning for training data files...")

patterns = [
    'outputs/*/training_data/*.npz',
    'outputs/*/*.npz',
    'outputs/*/training_data.npz'
]

all_files = set()
for pattern in patterns:
    all_files.update(glob.glob(pattern))

all_files = sorted(all_files, key=lambda x: Path(x).stat().st_mtime, reverse=True)

print(f"   Found {len(all_files)} file(s)\n")

if not all_files:
    print("❌ No training data files found!")
    print("   Make sure you're in the MD_MLPES directory")
    sys.exit(1)

# ==============================================================================
# ANALYZE FILES
# ==============================================================================

print("=" * 80)
print("📊 STEP 2: Analyzing each file...")
print("=" * 80)

file_analysis = []

for i, filepath in enumerate(all_files, 1):
    print(f"\n[{i}/{len(all_files)}] {filepath}")
    print("-" * 60)
    
    analysis = {
        'filepath': filepath,
        'exists': True,
        'readable': False,
        'has_data': False,
        'has_metadata': False,
        'metadata_valid': False,
        'details': {}
    }
    
    try:
        # Load file
        data = np.load(filepath, allow_pickle=True)
        analysis['readable'] = True
        print("   ✅ File readable")
        
        # Check basic data
        symbols = data['symbols'].tolist() if 'symbols' in data else []
        energies = data['energies'] if 'energies' in data else None
        forces = data['forces'] if 'forces' in data else None
        coords = data['coordinates'] if 'coordinates' in data else None
        
        if energies is not None and len(symbols) > 0:
            analysis['has_data'] = True
            
            # Calculate stats
            n_configs = len(energies)
            n_atoms = len(symbols)
            e_range_kcal = (energies.max() - energies.min()) * 627.509
            e_mean_kcal = energies.mean() * 627.509
            e_std_kcal = energies.std() * 627.509
            
            # Generate formula
            counts = Counter(symbols)
            formula = ''.join([f"{s}{c if c > 1 else ''}" for s, c in sorted(counts.items())])
            
            analysis['details'] = {
                'n_configs': n_configs,
                'n_atoms': n_atoms,
                'symbols': ' '.join(symbols),
                'formula': formula,
                'has_forces': forces is not None,
                'has_coords': coords is not None,
                'energy_range_kcal': e_range_kcal,
                'energy_mean_kcal': e_mean_kcal,
                'energy_std_kcal': e_std_kcal
            }
            
            print(f"   ✅ Has data:")
            print(f"      Configs: {n_configs}")
            print(f"      Atoms: {n_atoms} ({formula})")
            print(f"      Symbols: {' '.join(symbols)}")
            print(f"      Energy range: {e_range_kcal:.2f} kcal/mol")
            print(f"      Energy mean: {e_mean_kcal:.2f} kcal/mol")
            print(f"      Energy std: {e_std_kcal:.2f} kcal/mol")
            print(f"      Has forces: {forces is not None}")
            print(f"      Has coords: {coords is not None}")
        else:
            print("   ❌ Missing data (no energies or symbols)")
        
        # Check metadata
        if 'metadata' in data:
            analysis['has_metadata'] = True
            meta = data['metadata']
            
            # Check type
            print(f"   📋 Metadata exists (type: {type(meta).__name__})")
            
            # Try to parse
            try:
                if isinstance(meta, np.ndarray):
                    meta = meta.item()
                
                if isinstance(meta, dict):
                    analysis['metadata_valid'] = True
                    
                    # Extract key info
                    method = 'unknown'
                    basis = 'unknown'
                    mol_name = 'unknown'
                    
                    if 'theory' in meta:
                        method = meta['theory'].get('method', 'unknown')
                        basis = meta['theory'].get('basis', 'unknown')
                    
                    if 'molecule' in meta:
                        mol_name = meta['molecule'].get('name', 'unknown')
                    
                    analysis['details'].update({
                        'method': method,
                        'basis': basis,
                        'molecule_name': mol_name
                    })
                    
                    print(f"   ✅ Valid metadata:")
                    print(f"      Method: {method}")
                    print(f"      Basis: {basis}")
                    print(f"      Molecule: {mol_name}")
                else:
                    print(f"   ⚠️  Metadata wrong type: {type(meta)}")
                    print(f"      Expected: dict, Got: {type(meta).__name__}")
                    
            except Exception as e:
                print(f"   ❌ Error parsing metadata: {e}")
        else:
            print("   ❌ No metadata field")
        
        # File info
        file_size = Path(filepath).stat().st_size / 1024 / 1024  # MB
        mod_time = datetime.fromtimestamp(Path(filepath).stat().st_mtime)
        
        print(f"\n   📁 File info:")
        print(f"      Size: {file_size:.2f} MB")
        print(f"      Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        analysis['error'] = str(e)
    
    file_analysis.append(analysis)

# ==============================================================================
# CHECK DATABASE
# ==============================================================================

print("\n" + "=" * 80)
print("🗄️  STEP 3: Checking database...")
print("=" * 80)

db_exists = Path('computations_db.json').exists()

if db_exists:
    print("\n✅ Database file exists: computations_db.json")
    
    try:
        with open('computations_db.json', 'r') as f:
            db_data = json.load(f)
        
        computations = db_data.get('computations', {})
        last_updated = db_data.get('last_updated', 'unknown')
        
        print(f"   Entries: {len(computations)}")
        print(f"   Last updated: {last_updated}")
        
        if computations:
            print(f"\n   📊 Registered computations:")
            for comp_id, entry in computations.items():
                mol_name = entry.get('molecule', {}).get('name', 'unknown')
                method = entry.get('theory', {}).get('method', 'unknown')
                basis = entry.get('theory', {}).get('basis', 'unknown')
                n_configs = entry.get('training', {}).get('n_configs', 0)
                filepath = entry.get('filepath', 'unknown')
                valid = entry.get('metadata', {}).get('valid', True)
                
                status = "✅" if valid else "❌"
                print(f"      {status} {comp_id}")
                print(f"         {mol_name} | {method}/{basis} | {n_configs} configs")
                print(f"         {filepath}")
        else:
            print(f"\n   ⚠️  Database is empty!")
    
    except Exception as e:
        print(f"   ❌ Error reading database: {e}")
else:
    print("\n❌ No database file found!")
    print("   Expected: computations_db.json")

# ==============================================================================
# SUMMARY & RECOMMENDATIONS
# ==============================================================================

print("\n" + "=" * 80)
print("📋 SUMMARY & RECOMMENDATIONS")
print("=" * 80)

# Count issues
files_with_data = sum(1 for a in file_analysis if a['has_data'])
files_with_metadata = sum(1 for a in file_analysis if a['has_metadata'])
files_with_valid_metadata = sum(1 for a in file_analysis if a['metadata_valid'])
files_without_metadata = sum(1 for a in file_analysis if a['has_data'] and not a['metadata_valid'])

print(f"\n📊 File Status:")
print(f"   Total files: {len(file_analysis)}")
print(f"   ✅ With data: {files_with_data}")
print(f"   📋 With metadata field: {files_with_metadata}")
print(f"   ✅ With valid metadata: {files_with_valid_metadata}")
print(f"   ⚠️  Need metadata: {files_without_metadata}")

print(f"\n🗄️  Database Status:")
if db_exists:
    print(f"   ✅ Database exists")
    print(f"   📊 Entries: {len(computations)}")
    
    if len(computations) < files_with_data:
        print(f"   ⚠️  Missing entries: {files_with_data - len(computations)}")
else:
    print(f"   ❌ No database")

print(f"\n💡 Recommendations:")

# Recommendation 1: Migrate metadata
if files_without_metadata > 0:
    print(f"\n1. FIX METADATA ({files_without_metadata} file(s) need metadata):")
    print(f"   python3 migrate_metadata.py --auto --method B3LYP --basis 6-31G*")
    print(f"   ")
    print(f"   This will add metadata to:")
    for a in file_analysis:
        if a['has_data'] and not a['metadata_valid']:
            print(f"   - {a['filepath']}")

# Recommendation 2: Build/update database
if not db_exists or len(computations) < files_with_data:
    print(f"\n2. BUILD/UPDATE DATABASE:")
    print(f"   python3 computation_database.py scan")
    print(f"   ")
    print(f"   This will register all {files_with_data} file(s)")

# Recommendation 3: Use workflow
if files_with_valid_metadata > 0:
    print(f"\n3. USE WORKFLOW:")
    print(f"   python3 complete_workflow_v2.2.py")
    print(f"   ")
    print(f"   Select [1] Use database")
    print(f"   You'll see {files_with_valid_metadata} valid computation(s)")

# Export detailed report
report_file = 'diagnostic_report.json'
report_data = {
    'scan_date': datetime.now().isoformat(),
    'files': file_analysis,
    'database_exists': db_exists,
    'database_entries': len(computations) if db_exists else 0,
    'summary': {
        'total_files': len(file_analysis),
        'files_with_data': files_with_data,
        'files_with_valid_metadata': files_with_valid_metadata,
        'files_need_migration': files_without_metadata
    }
}

with open(report_file, 'w') as f:
    json.dump(report_data, f, indent=2)

print(f"\n📄 Detailed report saved: {report_file}")

print("\n" + "=" * 80)
print("✅ DIAGNOSTIC COMPLETE")
print("=" * 80)
