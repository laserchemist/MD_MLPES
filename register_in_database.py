#!/usr/bin/env python3
"""
Register Refined Data in Database

Simple script to add your refined training data to the computation database.

Usage:
    python3 register_in_database.py \
        outputs/diagnostic_phase1_XXX/augmented_training_data.npz
"""

import sys
import argparse
from pathlib import Path

try:
    from computation_database import ComputationDatabase
    from modules.data_formats import load_trajectory
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure you're in the MD_MLPES directory")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Register data in computation database')
    parser.add_argument('filepath', help='Path to training data (.npz)')
    parser.add_argument('--db', default='computations_db.json', help='Database file')
    parser.add_argument('--name', help='Custom name for molecule')
    parser.add_argument('--notes', default='', help='Additional notes')
    
    args = parser.parse_args()
    
    # Check file exists
    filepath = Path(args.filepath)
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        sys.exit(1)
    
    print("=" * 80)
    print("  REGISTER DATA IN DATABASE")
    print("=" * 80)
    
    # Load data
    print(f"\n📂 Loading: {filepath}")
    try:
        data = load_trajectory(str(filepath))
        print(f"   ✅ Loaded {data.n_frames} configurations")
    except Exception as e:
        print(f"   ❌ Error loading: {e}")
        sys.exit(1)
    
    # Extract metadata
    metadata = data.metadata if hasattr(data, 'metadata') and data.metadata else {}
    theory = metadata.get('theory', {})
    
    method = theory.get('method', 'UNKNOWN')
    basis = theory.get('basis', 'UNKNOWN')
    
    # Determine molecule name
    if args.name:
        mol_name = args.name
    else:
        # Try to guess from metadata or filename
        mol_name = metadata.get('molecule', {}).get('name', filepath.stem)
    
    # Formula from symbols
    from collections import Counter
    symbol_counts = Counter(data.symbols)
    formula = ''.join(f"{s}{c if c > 1 else ''}" for s, c in sorted(symbol_counts.items()))
    
    # Energy statistics
    energies_kcal = data.energies * 627.509
    energy_range = energies_kcal.max() - energies_kcal.min()
    
    # Check if refined
    is_refined = 'refinement' in metadata
    if is_refined:
        refinements = metadata.get('refinement', [])
        n_refinements = len(refinements)
        training_type = f'refined_{n_refinements}x'
        
        # Add refinement info to notes
        if not args.notes:
            last_ref = refinements[-1]
            args.notes = f"Refined {n_refinements} time(s), last: {last_ref.get('threshold', 'N/A')}, " \
                        f"{last_ref.get('n_added', 0)} points added"
    else:
        training_type = 'direct_md' if metadata.get('production_md') else 'energy_only'
    
    # Display summary
    print(f"\n📊 Data Summary:")
    print(f"   File: {filepath.name}")
    print(f"   Molecule: {mol_name} ({formula})")
    print(f"   Theory: {method}/{basis}")
    print(f"   Configurations: {data.n_frames}")
    print(f"   Atoms: {len(data.symbols)} ({' '.join(data.symbols)})")
    print(f"   Has forces: {'Yes' if data.forces is not None else 'No'}")
    print(f"   Training type: {training_type}")
    print(f"   Energy range: {energy_range:.2f} kcal/mol")
    if is_refined:
        print(f"   Refinements: {n_refinements}")
    if args.notes:
        print(f"   Notes: {args.notes}")
    
    # Confirm
    print("\n" + "=" * 80)
    confirm = input("Register in database? [y/n]: ").strip().lower()
    
    if confirm != 'y':
        print("   Cancelled")
        sys.exit(0)
    
    # Load/create database
    print(f"\n📂 Loading database: {args.db}")
    db = ComputationDatabase(args.db)
    
    # Register
    print(f"\n📝 Registering...")
    try:
        comp_id = db.register(
            filepath=str(filepath),
            method=method,
            basis=basis,
            molecule_name=mol_name,
            molecule_formula=formula,
            n_configs=data.n_frames,
            n_atoms=len(data.symbols),
            symbols=data.symbols,
            has_forces=data.forces is not None,
            training_type=training_type,
            energy_range_kcal=float(energy_range),
            notes=args.notes
        )
        
        print(f"   ✅ Registered as: {comp_id}")
        
        # Save database
        db.save()
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 80)
    print("  REGISTRATION COMPLETE")
    print("=" * 80)
    
    print(f"\n✅ {mol_name} registered in database")
    print(f"   ID: {comp_id}")
    print(f"   Database: {args.db}")
    
    print(f"\n💡 You can now:")
    print(f"   • Query database: python3 query_database.py")
    print(f"   • Use in workflows: Select from computation database")
    print(f"   • Update metadata: db.update('{comp_id}', updates)")

if __name__ == '__main__':
    main()
