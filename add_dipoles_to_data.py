#!/usr/bin/env python3
"""
Add Dipole Moments to Existing Training Data

This script computes dipole moments for all configurations in your
training data files and adds them to the metadata.

Usage:
    python3 add_dipoles_to_data.py <training_data.npz>
    
    Or for all files:
    python3 add_dipoles_to_data.py --all

Author: PSI4-MD ML-PES Framework
Date: January 2026
"""

import sys
import numpy as np
from pathlib import Path
import argparse
import shutil

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False
    def tqdm(x, **kwargs):
        return x

print("=" * 80)
print("  ADD DIPOLES TO TRAINING DATA")
print("=" * 80)

# Import modules
try:
    from modules.data_formats import load_trajectory, save_trajectory
    print("✅ Framework modules loaded")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure you're in the MD_MLPES directory")
    sys.exit(1)

try:
    import psi4
    print(f"✅ PSI4 {psi4.__version__}")
except ImportError:
    print("❌ PSI4 required for dipole calculation")
    sys.exit(1)


def compute_psi4_dipole_only(symbols, coords, method='B3LYP', basis='6-31G*'):
    """
    Compute dipole moment using PSI4.
    
    Args:
        symbols: Atomic symbols
        coords: Coordinates in Angstroms
        method: QM method
        basis: Basis set
        
    Returns:
        dipole: Dipole moment in Debye (3,)
        error: Error message or None
    """
    # Create molecule
    mol_str = "\n0 1\n"
    for s, c in zip(symbols, coords):
        mol_str += f"{s} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n"
    mol_str += "units angstrom\nno_reorient\nno_com"
    
    # Clean PSI4
    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.be_quiet()
    psi4.set_memory('2 GB')
    psi4.set_num_threads(4)
    
    mol = psi4.geometry(mol_str)
    
    # Settings
    psi4.set_options({
        'basis': basis,
        'scf_type': 'df',
        'reference': 'rhf',
        'maxiter': 200,
        'e_convergence': 1e-6,
        'd_convergence': 1e-6
    })
    
    try:
        # Compute properties (just dipole, no gradient needed)
        energy, wfn = psi4.properties(
            method, 
            properties=['dipole'], 
            return_wfn=True, 
            molecule=mol
        )
        
        # Get dipole moment (in Debye)
        dipole = np.array(wfn.variable('SCF DIPOLE'))
        
        return dipole, None
        
    except Exception as e:
        return None, str(e)[:200]


def add_dipoles_to_trajectory(filepath, method='B3LYP', basis='6-31G*', 
                              backup=True, verbose=True):
    """
    Add dipole moments to existing trajectory file.
    
    Args:
        filepath: Path to trajectory .npz file
        method: QM method (default: B3LYP)
        basis: Basis set (default: 6-31G*)
        backup: Create backup before modifying
        verbose: Print progress
        
    Returns:
        success: True if successful
    """
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"  Processing: {Path(filepath).name}")
        print(f"{'=' * 80}")
    
    # Load trajectory
    try:
        traj = load_trajectory(str(filepath))
        if verbose:
            print(f"\n✅ Loaded trajectory:")
            print(f"   Frames: {traj.n_frames}")
            print(f"   Molecule: {' '.join(traj.symbols)}")
    except Exception as e:
        print(f"❌ Error loading: {e}")
        return False
    
    # Check if already has dipoles
    if traj.metadata and 'dipoles' in traj.metadata:
        if verbose:
            dipoles = traj.metadata['dipoles']
            print(f"\n⚠️  Already has {len(dipoles)} dipole moments")
            
            response = input("\n   Recompute anyway? [y/N]: ").strip().lower()
            if response != 'y':
                print(f"   Skipping...")
                return True
    
    # Get theory level from metadata
    if traj.metadata and 'theory' in traj.metadata:
        theory = traj.metadata['theory']
        method = theory.get('method', method)
        basis = theory.get('basis', basis)
    
    if verbose:
        print(f"\n🔬 Computing dipoles:")
        print(f"   Method: {method}")
        print(f"   Basis: {basis}")
        print(f"   Configurations: {traj.n_frames}")
        
        # Time estimate
        est_time = traj.n_frames * 1.0  # ~1 min per config
        if est_time < 60:
            print(f"   Estimated time: {est_time:.0f} seconds")
        else:
            print(f"   Estimated time: {est_time/60:.1f} minutes")
    
    # Compute dipoles
    dipoles = []
    failed = []
    
    iterator = tqdm(enumerate(traj.coordinates), total=traj.n_frames, desc="PSI4 dipole") if TQDM else enumerate(traj.coordinates)
    
    for i, coords in iterator:
        dipole, error = compute_psi4_dipole_only(
            traj.symbols, coords, method, basis
        )
        
        if error:
            dipoles.append(np.zeros(3))  # Placeholder
            failed.append((i, error))
        else:
            dipoles.append(dipole)
    
    dipoles_array = np.array(dipoles)
    
    if verbose:
        print(f"\n📊 Results:")
        print(f"   Computed: {len(dipoles)} dipoles")
        if len(failed) > 0:
            print(f"   Failed: {len(failed)} configurations")
            print(f"   Success rate: {100*(1-len(failed)/len(dipoles)):.1f}%")
        else:
            print(f"   All successful! ✅")
    
    # Statistics
    valid_dipoles = dipoles_array[~np.all(dipoles_array == 0, axis=1)]
    if len(valid_dipoles) > 0 and verbose:
        magnitudes = np.linalg.norm(valid_dipoles, axis=1)
        print(f"\n   Dipole statistics:")
        print(f"      Mean |μ|: {magnitudes.mean():.4f} Debye")
        print(f"      Std  |μ|: {magnitudes.std():.4f} Debye")
        print(f"      Range:    {magnitudes.min():.4f} - {magnitudes.max():.4f} Debye")
    
    # Backup original
    if backup:
        backup_path = str(filepath).replace('.npz', '_backup.npz')
        shutil.copy(filepath, backup_path)
        if verbose:
            print(f"\n💾 Backup created: {Path(backup_path).name}")
    
    # Update metadata
    if traj.metadata is None:
        traj.metadata = {}
    
    traj.metadata['dipoles'] = dipoles_array
    traj.metadata['dipoles_units'] = 'Debye'
    traj.metadata['dipoles_method'] = method
    traj.metadata['dipoles_basis'] = basis
    traj.metadata['has_dipoles'] = True
    traj.metadata['n_dipoles_failed'] = len(failed)
    
    # Save updated trajectory
    try:
        save_trajectory(traj, str(filepath))
        if verbose:
            print(f"\n✅ Updated trajectory saved: {Path(filepath).name}")
            print(f"   Added metadata['dipoles'] with shape {dipoles_array.shape}")
    except Exception as e:
        print(f"\n❌ Error saving: {e}")
        return False
    
    return True


def find_training_data_files(directory='.'):
    """Find all training data files in directory."""
    patterns = [
        '**/training_data.npz',
        '**/augmented*.npz',
        '**/*training*.npz'
    ]
    
    files = []
    for pattern in patterns:
        files.extend(Path(directory).glob(pattern))
    
    # Remove duplicates and sort
    files = sorted(set(files))
    
    return files


def main():
    parser = argparse.ArgumentParser(
        description='Add dipole moments to training data files'
    )
    
    parser.add_argument('files', nargs='*', 
                       help='Trajectory files to process')
    parser.add_argument('--all', action='store_true',
                       help='Process all training data files in outputs/')
    parser.add_argument('--method', default='B3LYP',
                       help='QM method (default: B3LYP)')
    parser.add_argument('--basis', default='6-31G*',
                       help='Basis set (default: 6-31G*)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup files')
    
    args = parser.parse_args()
    
    # Determine files to process
    if args.all:
        print("\n🔍 Searching for training data files in outputs/...")
        files = find_training_data_files('outputs')
        
        if not files:
            print("   ❌ No training data files found")
            return
        
        print(f"   ✅ Found {len(files)} file(s):")
        for i, f in enumerate(files, 1):
            print(f"      [{i}] {f}")
        
        response = input(f"\n   Process all {len(files)} files? [y/N]: ").strip().lower()
        if response != 'y':
            print("   Cancelled")
            return
        
    elif args.files:
        files = [Path(f) for f in args.files]
        
        # Check all exist
        missing = [f for f in files if not f.exists()]
        if missing:
            print(f"❌ Files not found:")
            for f in missing:
                print(f"   {f}")
            return
    
    else:
        print("❌ No files specified")
        print("\nUsage:")
        print("   python3 add_dipoles_to_data.py file1.npz file2.npz")
        print("   python3 add_dipoles_to_data.py --all")
        return
    
    # Process files
    success_count = 0
    
    for filepath in files:
        success = add_dipoles_to_trajectory(
            filepath,
            method=args.method,
            basis=args.basis,
            backup=not args.no_backup,
            verbose=True
        )
        
        if success:
            success_count += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"\n   Processed: {len(files)} file(s)")
    print(f"   Successful: {success_count}")
    
    if success_count > 0:
        print(f"\n✅ Dipoles added successfully!")
        print(f"\n💡 Next steps:")
        print(f"   1. Verify dipoles: python3 check_ir_setup.py")
        print(f"   2. Train dipole surface: python3 compute_ir_spectrum.py --train-dipole")
        print(f"   3. Compute IR spectrum")
    
    print("")

if __name__ == '__main__':
    main()
