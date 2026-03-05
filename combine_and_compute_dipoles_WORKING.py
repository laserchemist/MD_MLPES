#!/usr/bin/env python3
"""
Combine Trajectory Data and Compute Dipoles - WORKING VERSION
==============================================================
Uses the PROVEN ChatGPT method that actually works with PSI4 1.10.

Usage:
    python combine_and_compute_dipoles_WORKING.py <directory>

Author: Jonathan  
Date: 2026-01-17
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import argparse


def find_trajectory_files(directory):
    """Find all trajectory files."""
    directory = Path(directory)
    patterns = ['trajectory_*K.npz', 'trajectory_*.npz']
    
    for pattern in patterns:
        files = sorted(directory.glob(pattern))
        if files:
            return files
    return []


def load_and_combine_trajectories(directory):
    """Load and combine all trajectory files."""
    print("\n" + "="*80)
    print("STEP 1: Loading and Combining Trajectories")
    print("="*80)
    
    traj_files = find_trajectory_files(directory)
    
    if not traj_files:
        raise ValueError(f"No trajectory files found in {directory}")
    
    print(f"\nFound {len(traj_files)} trajectory files:")
    for f in traj_files:
        print(f"  - {f.name}")
    
    # Load all trajectories
    all_coords = []
    all_energies = []
    all_forces = []
    symbols = None
    metadata = None
    
    for traj_file in traj_files:
        print(f"\nLoading: {traj_file.name}")
        data = np.load(str(traj_file), allow_pickle=True)
        
        # Get symbols
        if symbols is None:
            symbols = data['symbols'].tolist() if hasattr(data['symbols'], 'tolist') else list(data['symbols'])
            print(f"  Molecule: {' '.join(symbols)}")
        
        # Get metadata
        if metadata is None and 'metadata' in data:
            metadata_item = data['metadata']
            if isinstance(metadata_item, np.ndarray):
                try:
                    metadata = metadata_item.item()
                except:
                    metadata = {}
            elif isinstance(metadata_item, dict):
                metadata = metadata_item
            else:
                metadata = {}
        
        # Accumulate data
        all_coords.append(data['coordinates'])
        all_energies.append(data['energies'])
        all_forces.append(data['forces'] if 'forces' in data else np.zeros_like(data['coordinates']))
        
        print(f"  Frames: {len(data['coordinates'])}")
    
    # Combine
    combined_coords = np.vstack(all_coords)
    combined_energies = np.concatenate(all_energies)
    combined_forces = np.vstack(all_forces)
    
    print(f"\n✅ Combined trajectories:")
    print(f"   Total frames: {len(combined_coords)}")
    print(f"   Coordinates shape: {combined_coords.shape}")
    
    return {
        'symbols': symbols,
        'coordinates': combined_coords,
        'energies': combined_energies,
        'forces': combined_forces,
        'metadata': metadata if isinstance(metadata, dict) else {}
    }


def compute_dipole_working(symbols, coords, method='scf', basis='6-31G*'):
    """
    Compute dipole using PROVEN WORKING method (ChatGPT example).
    
    This is the ONLY method that works with PSI4 1.10.
    
    Returns
    -------
    dipole : np.ndarray or None
        Dipole in Debye [x, y, z]
    error : str or None
        Error message if failed
    """
    import psi4
    
    AU_TO_DEBYE = 2.541746
    
    # Create molecule string
    mol_str = f"0 1\n"
    for s, c in zip(symbols, coords):
        mol_str += f"{s} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n"
    mol_str += "units angstrom\nno_reorient\nno_com"
    
    # Clean PSI4
    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.be_quiet()
    psi4.set_memory('2 GB')
    psi4.set_num_threads(4)
    
    try:
        # Create molecule (PSI4 1.10 removed tooclose parameter)
        mol = psi4.geometry(mol_str)
        
        # Settings (EXACTLY as in ChatGPT example)
        psi4.set_options({
            'basis': basis,
            'scf_type': 'pk',
            'reference': 'rhf'
        })
        
        # THIS IS THE WORKING METHOD
        energy, wfn = psi4.energy(
            method,
            molecule=mol,
            return_wfn=True,
            properties=["dipole"]
        )
        
        # Get dipole - "SCF DIPOLE" works for both SCF and DFT
        dipole_vec = psi4.variable("SCF DIPOLE")
        
        # Convert to numpy array
        dipole_au = np.array([
            dipole_vec[0],
            dipole_vec[1],
            dipole_vec[2]
        ])
        
        # Convert to Debye
        dipole_debye = dipole_au * AU_TO_DEBYE
        
        return dipole_debye, None
        
    except Exception as e:
        error_msg = str(e)
        if "too close" in error_msg.lower():
            return None, "atoms_too_close"
        elif "scf" in error_msg.lower() and "converge" in error_msg.lower():
            return None, "scf_convergence"
        else:
            return None, f"error: {error_msg[:100]}"


def compute_dipoles_for_all_frames(symbols, coordinates, method='scf', basis='6-31G*'):
    """Compute dipoles for all frames using WORKING method."""
    print("\n" + "="*80)
    print("STEP 2: Computing Dipole Moments (WORKING METHOD)")
    print("="*80)
    
    print(f"\n💧 Computing dipoles for {len(coordinates)} configurations...")
    print(f"   Method: {method}/{basis}")
    print(f"   API: PROVEN WORKING (ChatGPT example)")
    print(f"   This will take ~{len(coordinates)/60:.1f} minutes")
    
    dipoles = []
    failed = 0
    failure_types = {}
    first_errors = []
    
    for i, coords in enumerate(tqdm(coordinates, desc="Computing dipoles")):
        dipole, error = compute_dipole_working(symbols, coords, method, basis)
        
        if error:
            dipoles.append(np.zeros(3))
            failed += 1
            
            # Track error type
            error_key = error.split(':')[0] if ':' in error else error
            failure_types[error_key] = failure_types.get(error_key, 0) + 1
            
            # Save first 3 errors
            if len(first_errors) < 3:
                first_errors.append((i, error))
        else:
            dipoles.append(dipole)
    
    dipoles_array = np.array(dipoles)
    
    # Statistics
    valid_dipoles = dipoles_array[~np.all(dipoles_array == 0, axis=1)]
    
    stats = {
        'n_total': len(dipoles),
        'n_failed': failed,
        'n_valid': len(valid_dipoles),
        'failure_types': failure_types,
        'first_errors': first_errors
    }
    
    if len(valid_dipoles) > 0:
        magnitudes = np.linalg.norm(valid_dipoles, axis=1)
        stats['mean_magnitude'] = magnitudes.mean()
        stats['std_magnitude'] = magnitudes.std()
        stats['min_magnitude'] = magnitudes.min()
        stats['max_magnitude'] = magnitudes.max()
    
    print(f"\n✅ Dipoles computed:")
    print(f"   Successful: {stats['n_valid']}/{stats['n_total']} ({100*stats['n_valid']/stats['n_total']:.1f}%)")
    print(f"   Failed: {stats['n_failed']}")
    
    if failed > 0:
        print(f"\n   Failure breakdown:")
        for ftype, count in sorted(failure_types.items(), key=lambda x: -x[1]):
            print(f"      {ftype}: {count}")
        
        if first_errors:
            print(f"\n   First few errors (for debugging):")
            for frame_idx, error_msg in first_errors:
                print(f"      Frame {frame_idx}: {error_msg}")
    
    if len(valid_dipoles) > 0:
        print(f"\n📊 Dipole statistics (valid frames only):")
        print(f"   Mean |μ|: {stats['mean_magnitude']:.4f} Debye")
        print(f"   Std  |μ|: {stats['std_magnitude']:.4f} Debye")
        print(f"   Range: {stats['min_magnitude']:.4f} - {stats['max_magnitude']:.4f} Debye")
    
    return dipoles_array, stats


def save_combined_training_data(directory, combined_data, dipoles, dipole_stats, method, basis):
    """Save combined training data with dipoles."""
    print("\n" + "="*80)
    print("STEP 3: Saving Combined Training Data")
    print("="*80)
    
    # Get metadata - ensure it's a dict
    metadata = combined_data.get('metadata', {})
    if not isinstance(metadata, dict):
        metadata = {}
    
    # Add dipole information
    metadata['has_dipoles'] = True
    metadata['dipoles'] = dipoles.tolist()
    metadata['dipoles_units'] = 'Debye'
    metadata['dipoles_method'] = method
    metadata['dipoles_basis'] = basis
    metadata['dipoles_api'] = 'WORKING (ChatGPT example)'
    metadata['n_dipoles_total'] = int(dipole_stats['n_total'])
    metadata['n_dipoles_valid'] = int(dipole_stats['n_valid'])
    metadata['n_dipoles_failed'] = int(dipole_stats['n_failed'])
    metadata['dipole_failure_types'] = dipole_stats['failure_types']
    metadata['combined_date'] = datetime.now().isoformat()
    
    # Save NPZ file
    output_file = directory / 'training_data.npz'
    
    np.savez(
        output_file,
        symbols=np.array(combined_data['symbols']),
        coordinates=combined_data['coordinates'],
        energies=combined_data['energies'],
        forces=combined_data['forces'],
        metadata=metadata
    )
    
    print(f"\n✅ Saved: {output_file}")
    print(f"\n📦 Contents:")
    print(f"   Frames: {len(combined_data['coordinates'])}")
    print(f"   Dipoles: {dipoles.shape} ({dipole_stats['n_valid']} valid)")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Combine trajectories and compute dipoles (WORKING VERSION)'
    )
    
    parser.add_argument('directory', type=str,
                       help='Directory containing trajectory_*K.npz files')
    parser.add_argument('--method', type=str, default='scf',
                       help='QM method: scf or b3lyp (default: scf)')
    parser.add_argument('--basis', type=str, default='6-31G*',
                       help='Basis set (default: 6-31G*)')
    
    args = parser.parse_args()
    
    directory = Path(args.directory)
    
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        sys.exit(1)
    
    # Check PSI4
    try:
        import psi4
        print(f"✅ PSI4 {psi4.__version__}")
    except ImportError:
        print("❌ PSI4 not available")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("COMBINE TRAJECTORIES AND COMPUTE DIPOLES (WORKING VERSION)")
    print("="*80)
    print(f"\nDirectory: {directory}")
    print(f"Theory: {args.method}/{args.basis}")
    print(f"API: PROVEN WORKING (ChatGPT example)")
    
    # Steps
    combined_data = load_and_combine_trajectories(directory)
    dipoles, dipole_stats = compute_dipoles_for_all_frames(
        combined_data['symbols'],
        combined_data['coordinates'],
        args.method,
        args.basis
    )
    output_file = save_combined_training_data(
        directory, combined_data, dipoles, dipole_stats, args.method, args.basis
    )
    
    # Summary
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    
    print(f"\n✅ Combined training data:")
    print(f"   File: {output_file}")
    print(f"   Frames: {len(combined_data['coordinates'])}")
    print(f"   Valid dipoles: {dipole_stats['n_valid']}/{dipole_stats['n_total']}")
    print(f"   Success rate: {100*dipole_stats['n_valid']/dipole_stats['n_total']:.1f}%")
    
    if dipole_stats['n_valid'] > 0:
        print(f"\n💡 Next steps:")
        print(f"   1. Train ML-dipole:")
        print(f"      python compute_ir_workflow.py --train-dipole --training-data {output_file}")
    else:
        print(f"\n⚠️  NO VALID DIPOLES!")
        print(f"   Check error messages above")
    print()


if __name__ == '__main__':
    main()
