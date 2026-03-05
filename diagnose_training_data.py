#!/usr/bin/env python3
"""
Training Data Diagnostic Tool
==============================
Check what's in your training data files and fix common issues.

Usage:
    python diagnose_training_data.py path/to/data.npz
    python diagnose_training_data.py path/to/trajectory_dir/

Author: Jonathan
Date: 2026-01-16
"""

import sys
import numpy as np
from pathlib import Path
import json


def diagnose_file(filepath):
    """Diagnose a single npz file."""
    print("\n" + "="*70)
    print(f"DIAGNOSING: {filepath.name}")
    print("="*70)
    
    try:
        data = np.load(str(filepath), allow_pickle=True)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False
    
    print("\n📦 File Contents:")
    print("-"*70)
    for key in data.files:
        item = data[key]
        if hasattr(item, 'shape'):
            print(f"  {key:20s}: {item.dtype:12s} {item.shape}")
        else:
            print(f"  {key:20s}: {type(item).__name__}")
    
    # Check required fields
    print("\n✅ Required Fields:")
    print("-"*70)
    
    required = ['symbols', 'coordinates', 'energies', 'forces']
    all_present = True
    
    for field in required:
        if field in data:
            print(f"  ✓ {field}")
        else:
            print(f"  ✗ {field} MISSING!")
            all_present = False
    
    if not all_present:
        print("\n❌ File is missing required fields!")
        return False
    
    # Check optional fields
    print("\n📋 Optional Fields:")
    print("-"*70)
    
    has_dipoles = False
    dipoles_location = None
    
    # Check for dipoles directly in file
    if 'dipoles' in data:
        dipoles = data['dipoles']
        print(f"  ✓ dipoles: {dipoles.shape}")
        has_dipoles = True
        dipoles_location = "direct"
        
        # Check if dipoles are valid (not all zeros)
        n_nonzero = np.count_nonzero(np.any(dipoles != 0, axis=1))
        n_total = len(dipoles)
        print(f"    Valid (non-zero): {n_nonzero}/{n_total} ({100*n_nonzero/n_total:.1f}%)")
    
    # Check for dipoles in metadata
    if 'metadata' in data:
        try:
            metadata = data['metadata'].item()
            print(f"  ✓ metadata: {type(metadata).__name__}")
            
            if 'dipoles' in metadata:
                dipoles_meta = np.array(metadata['dipoles'])
                print(f"    - dipoles in metadata: {dipoles_meta.shape}")
                if not has_dipoles:
                    has_dipoles = True
                    dipoles_location = "metadata"
                    
                    # Check validity
                    n_nonzero = np.count_nonzero(np.any(dipoles_meta != 0, axis=1))
                    n_total = len(dipoles_meta)
                    print(f"      Valid (non-zero): {n_nonzero}/{n_total} ({100*n_nonzero/n_total:.1f}%)")
            
            # Print other useful metadata
            if 'molecule' in metadata:
                mol_info = metadata['molecule']
                print(f"    - molecule: {mol_info.get('name', 'unknown')}")
                print(f"    - formula: {mol_info.get('formula', 'unknown')}")
            
            if 'theory' in metadata:
                theory = metadata['theory']
                print(f"    - method: {theory.get('method', 'unknown')}")
                print(f"    - basis: {theory.get('basis', 'unknown')}")
            
            if 'has_dipoles' in metadata:
                print(f"    - has_dipoles flag: {metadata['has_dipoles']}")
            
            if 'n_dipoles_valid' in metadata:
                print(f"    - n_dipoles_valid: {metadata['n_dipoles_valid']}")
                print(f"    - n_dipoles_failed: {metadata.get('n_dipoles_failed', 0)}")
                
        except Exception as e:
            print(f"  ⚠️  metadata (error reading): {e}")
    else:
        print(f"  ✗ metadata: Not present")
    
    # Data statistics
    print("\n📊 Data Statistics:")
    print("-"*70)
    
    symbols = data['symbols'].tolist() if hasattr(data['symbols'], 'tolist') else data['symbols']
    coords = data['coordinates']
    energies = data['energies']
    forces = data['forces']
    
    print(f"  Molecule: {' '.join(symbols)}")
    print(f"  Atoms: {len(symbols)}")
    print(f"  Frames: {len(coords)}")
    print(f"  Energy range: {energies.min():.6f} to {energies.max():.6f} Hartree")
    print(f"  Energy span: {(energies.max()-energies.min())*627.509:.2f} kcal/mol")
    
    # Dipole summary
    print("\n🔍 Dipole Status:")
    print("-"*70)
    
    if has_dipoles:
        print(f"  ✓ HAS DIPOLES (location: {dipoles_location})")
        print(f"  ✓ Ready for ML-dipole training!")
    else:
        print(f"  ✗ NO DIPOLES FOUND")
        print(f"  ⚠️  Cannot train ML-dipole model")
        print(f"\n  To fix:")
        print(f"    python data_loading_utils.py {filepath} --add-dipoles")
    
    return has_dipoles


def diagnose_directory(dirpath):
    """Diagnose a directory of trajectory files."""
    print("\n" + "="*70)
    print(f"DIAGNOSING DIRECTORY: {dirpath}")
    print("="*70)
    
    # Look for common patterns
    patterns = [
        'trajectory_*K.npz',
        'trajectory_*.npz',
        '*.npz'
    ]
    
    traj_files = []
    for pattern in patterns:
        traj_files = list(dirpath.glob(pattern))
        if traj_files:
            break
    
    if not traj_files:
        print("\n❌ No .npz files found in directory!")
        return False
    
    print(f"\nFound {len(traj_files)} trajectory files:")
    for f in sorted(traj_files):
        print(f"  - {f.name}")
    
    # Check if there's a combined file
    combined_file = dirpath / 'training_data.npz'
    if combined_file.exists():
        print(f"\n✓ Found combined training_data.npz")
        return diagnose_file(combined_file)
    else:
        print(f"\n⚠️  No combined training_data.npz found")
        print(f"   Individual trajectories need to be combined")
        
        # Check first file as example
        print(f"\n📋 Checking first file as example:")
        has_dipoles = diagnose_file(traj_files[0])
        
        print(f"\n💡 To combine trajectories:")
        print(f"   from data_loading_utils import combine_trajectory_directory")
        print(f"   data = combine_trajectory_directory('{dirpath}')")
        
        return has_dipoles


def main():
    if len(sys.argv) < 2:
        print("\nTraining Data Diagnostic Tool")
        print("="*70)
        print("\nUsage:")
        print("  python diagnose_training_data.py <path>")
        print("\nExamples:")
        print("  # Check a single file")
        print("  python diagnose_training_data.py outputs/training_with_dipoles_water_*/training_data.npz")
        print()
        print("  # Check a directory of trajectories")
        print("  python diagnose_training_data.py outputs/training_with_dipoles_water_*/")
        print()
        sys.exit(1)
    
    filepath = Path(sys.argv[1])
    
    if not filepath.exists():
        print(f"\n❌ Path not found: {filepath}")
        sys.exit(1)
    
    # Determine if it's a file or directory
    if filepath.is_dir():
        has_dipoles = diagnose_directory(filepath)
    else:
        has_dipoles = diagnose_file(filepath)
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if has_dipoles:
        print("\n✅ Data is ready for ML-dipole training!")
        print("\nNext steps:")
        print("  1. Train ML-dipole model:")
        print(f"     python compute_ir_workflow.py --train-dipole --training-data {filepath}")
        print()
        print("  2. After training ML-PES, compute IR spectrum:")
        print("     python compute_ir_workflow.py \\")
        print("         --ml-pes model.pkl \\")
        print("         --dipole-model dipole_surface.pkl \\")
        print("         --temp 300 --steps 50000")
    else:
        print("\n❌ Data needs dipoles before ML-dipole training!")
        print("\nFix options:")
        print("  1. Re-generate with generate_training_data_COMPLETE_WORKING.py")
        print("     (Make sure to let it finish dipole computation)")
        print()
        print("  2. Add dipoles to existing data:")
        print(f"     python data_loading_utils.py {filepath} --add-dipoles")
    
    print()


if __name__ == "__main__":
    main()
