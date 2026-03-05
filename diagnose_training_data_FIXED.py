#!/usr/bin/env python3
"""
Training Data Diagnostic Tool - FIXED
======================================
Check what's in your training data files.

FIXED: NumPy 2.x dtype formatting issue.

Usage:
    python diagnose_training_data_FIXED.py path/to/data.npz

Author: Jonathan
Date: 2026-01-17
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
            # FIXED: Convert dtype to string for NumPy 2.x compatibility
            dtype_str = str(item.dtype)
            print(f"  {key:20s}: {dtype_str:20s} {item.shape}")
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
    
    # Check for dipoles directly
    if 'dipoles' in data:
        dipoles = data['dipoles']
        print(f"  ✓ dipoles: {dipoles.shape}")
        has_dipoles = True
        dipoles_location = "direct"
        
        # Check validity
        n_nonzero = np.count_nonzero(np.any(dipoles != 0, axis=1))
        n_total = len(dipoles)
        print(f"    Valid (non-zero): {n_nonzero}/{n_total} ({100*n_nonzero/n_total:.1f}%)")
    
    # Check for dipoles in metadata
    if 'metadata' in data:
        try:
            metadata = data['metadata'].item()
            print(f"  ✓ metadata: {type(metadata).__name__}")
            
            if isinstance(metadata, dict):
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
                
                # Print other metadata
                if 'molecule' in metadata:
                    mol_info = metadata['molecule']
                    print(f"    - molecule: {mol_info.get('name', 'unknown')}")
                
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
        print(f"    python combine_and_compute_dipoles_WORKING.py {filepath.parent}")
    
    return has_dipoles


def diagnose_directory(dirpath):
    """Diagnose a directory of trajectory files."""
    print("\n" + "="*70)
    print(f"DIAGNOSING DIRECTORY: {dirpath}")
    print("="*70)
    
    # Look for trajectories
    patterns = ['trajectory_*K.npz', 'trajectory_*.npz', '*.npz']
    
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
    
    # Check for combined file
    combined_file = dirpath / 'training_data.npz'
    if combined_file.exists():
        print(f"\n✓ Found combined training_data.npz")
        return diagnose_file(combined_file)
    else:
        print(f"\n⚠️  No combined training_data.npz found")
        print(f"   Run: python combine_and_compute_dipoles_WORKING.py {dirpath}")
        
        # Check first file as example
        print(f"\n📋 Checking first file as example:")
        return diagnose_file(traj_files[0])


def main():
    if len(sys.argv) < 2:
        print("\nTraining Data Diagnostic Tool (FIXED)")
        print("="*70)
        print("\nUsage:")
        print("  python diagnose_training_data_FIXED.py <path>")
        print("\nExamples:")
        print("  # Check a file")
        print("  python diagnose_training_data_FIXED.py outputs/.../training_data.npz")
        print()
        print("  # Check a directory")
        print("  python diagnose_training_data_FIXED.py outputs/training_with_dipoles_ammonia_*/")
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
        print("  python compute_ir_workflow.py --train-dipole --training-data <path>")
    else:
        print("\n❌ Data needs dipoles before ML-dipole training!")
        print("\nFix:")
        print(f"  python combine_and_compute_dipoles_WORKING.py {filepath.parent if filepath.is_file() else filepath}")
    
    print()


if __name__ == "__main__":
    main()
