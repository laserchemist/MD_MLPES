#!/usr/bin/env python3
"""
Combine Trajectory Data and Compute Dipoles
============================================
Fix for training data that wasn't combined or doesn't have dipoles.

This script:
1. Combines individual trajectory_*K.npz files
2. Computes dipoles for all frames
3. Saves combined training_data.npz with dipoles

Usage:
    python combine_and_compute_dipoles.py <directory> [options]

Example:
    python combine_and_compute_dipoles.py outputs/training_with_dipoles_ammonia_20260116_123456/

Author: Jonathan  
Date: 2026-01-16
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import argparse


def find_trajectory_files(directory):
    """Find all trajectory files in directory."""
    directory = Path(directory)
    
    # Try different patterns
    patterns = [
        'trajectory_*K.npz',
        'trajectory_*.npz',
    ]
    
    for pattern in patterns:
        files = sorted(directory.glob(pattern))
        if files:
            return files
    
    return []


def load_and_combine_trajectories(directory):
    """
    Load and combine all trajectory files from a directory.
    
    Returns
    -------
    combined_data : dict
        Dictionary with combined arrays
    """
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
        
        # Get symbols from first file
        if symbols is None:
            symbols = data['symbols'].tolist() if hasattr(data['symbols'], 'tolist') else list(data['symbols'])
            print(f"  Molecule: {' '.join(symbols)}")
        
        # Get metadata if available
        if metadata is None and 'metadata' in data:
            metadata = data['metadata'].item() if isinstance(data['metadata'], np.ndarray) else data['metadata']
        
        # Accumulate data
        coords = data['coordinates']
        energies = data['energies']
        forces = data['forces'] if 'forces' in data else np.zeros_like(coords)
        
        all_coords.append(coords)
        all_energies.append(energies)
        all_forces.append(forces)
        
        print(f"  Frames: {len(coords)}")
    
    # Combine
    combined_coords = np.vstack(all_coords)
    combined_energies = np.concatenate(all_energies)
    combined_forces = np.vstack(all_forces)
    
    print(f"\n✅ Combined trajectories:")
    print(f"   Total frames: {len(combined_coords)}")
    print(f"   Coordinates shape: {combined_coords.shape}")
    print(f"   Energies shape: {combined_energies.shape}")
    print(f"   Forces shape: {combined_forces.shape}")
    
    return {
        'symbols': symbols,
        'coordinates': combined_coords,
        'energies': combined_energies,
        'forces': combined_forces,
        'metadata': metadata or {}
    }


def compute_dipole_for_frame(symbols, coords, method='B3LYP', basis='6-31G*'):
    """
    Compute dipole using density matrix method (most robust).
    
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
        # Create molecule (relaxed validation)
        mol = psi4.geometry(mol_str, tooclose=0.1)
        
        # Settings
        psi4.set_options({
            'basis': basis,
            'scf_type': 'df',
            'reference': 'rhf',
            'maxiter': 200,
            'e_convergence': 1e-6,
            'd_convergence': 1e-6
        })
        
        # Compute gradient (includes SCF)
        gradient_result, wfn = psi4.gradient(f"{method}/{basis}", molecule=mol, return_wfn=True)
        
        # Compute dipole from density matrix (ROBUST METHOD)
        Da = wfn.Da()
        mints = psi4.core.MintsHelper(wfn.basisset())
        dipole_ints = mints.ao_dipole()
        
        # Electronic dipole
        dipole_e = np.array([
            Da.vector_dot(dipole_ints[0]),
            Da.vector_dot(dipole_ints[1]),
            Da.vector_dot(dipole_ints[2])
        ])
        
        # Nuclear dipole
        mol_geom = wfn.molecule()
        dipole_n = np.array([
            mol_geom.nuclear_dipole()[0],
            mol_geom.nuclear_dipole()[1],
            mol_geom.nuclear_dipole()[2]
        ])
        
        # Total dipole
        dipole_au = dipole_e + dipole_n
        dipole_debye = dipole_au * AU_TO_DEBYE
        
        return dipole_debye, None
        
    except Exception as e:
        error_msg = str(e)
        if "too close" in error_msg.lower():
            return None, "atoms_too_close"
        elif "scf" in error_msg.lower() and "converge" in error_msg.lower():
            return None, "scf_convergence"
        else:
            return None, "other"


def compute_dipoles_for_all_frames(symbols, coordinates, method='B3LYP', basis='6-31G*'):
    """
    Compute dipoles for all frames.
    
    Returns
    -------
    dipoles : np.ndarray
        Shape (n_frames, 3) in Debye
    stats : dict
        Statistics about computation
    """
    print("\n" + "="*80)
    print("STEP 2: Computing Dipole Moments")
    print("="*80)
    
    print(f"\n💧 Computing dipoles for {len(coordinates)} configurations...")
    print(f"   Method: {method}/{basis}")
    print(f"   This will take ~{len(coordinates)/60:.1f} minutes")
    
    dipoles = []
    failed = 0
    failure_types = {'atoms_too_close': 0, 'scf_convergence': 0, 'other': 0}
    
    for coords in tqdm(coordinates, desc="Computing dipoles"):
        dipole, error = compute_dipole_for_frame(symbols, coords, method, basis)
        
        if error:
            dipoles.append(np.zeros(3))
            failed += 1
            failure_types[error] = failure_types.get(error, 0) + 1
        else:
            dipoles.append(dipole)
    
    dipoles_array = np.array(dipoles)
    
    # Statistics
    valid_dipoles = dipoles_array[~np.all(dipoles_array == 0, axis=1)]
    
    stats = {
        'n_total': len(dipoles),
        'n_failed': failed,
        'n_valid': len(valid_dipoles),
        'failure_types': failure_types
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
        for ftype, count in failure_types.items():
            if count > 0:
                print(f"      {ftype}: {count}")
    
    if len(valid_dipoles) > 0:
        print(f"\n📊 Dipole statistics (valid frames only):")
        print(f"   Mean |μ|: {stats['mean_magnitude']:.4f} Debye")
        print(f"   Std  |μ|: {stats['std_magnitude']:.4f} Debye")
        print(f"   Range: {stats['min_magnitude']:.4f} - {stats['max_magnitude']:.4f} Debye")
    
    return dipoles_array, stats


def save_combined_training_data(directory, combined_data, dipoles, dipole_stats, method, basis):
    """
    Save combined training data with dipoles.
    
    Parameters
    ----------
    directory : Path
        Output directory
    combined_data : dict
        Combined trajectory data
    dipoles : np.ndarray
        Computed dipoles
    dipole_stats : dict
        Dipole statistics
    method, basis : str
        Theory level used
    """
    print("\n" + "="*80)
    print("STEP 3: Saving Combined Training Data")
    print("="*80)
    
    # Update metadata
    metadata = combined_data.get('metadata', {})
    
    # Add dipole information
    metadata['has_dipoles'] = True
    metadata['dipoles'] = dipoles.tolist()  # For JSON compatibility
    metadata['dipoles_units'] = 'Debye'
    metadata['dipoles_method'] = method
    metadata['dipoles_basis'] = basis
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
    print(f"   Symbols: {combined_data['symbols']}")
    print(f"   Frames: {len(combined_data['coordinates'])}")
    print(f"   Coordinates: {combined_data['coordinates'].shape}")
    print(f"   Energies: {combined_data['energies'].shape}")
    print(f"   Forces: {combined_data['forces'].shape}")
    print(f"   Dipoles: {dipoles.shape} ({dipole_stats['n_valid']} valid)")
    
    # Save metadata as JSON for easy inspection
    metadata_file = directory / 'metadata.json'
    with open(metadata_file, 'w') as f:
        # Remove numpy arrays for JSON
        json_metadata = {k: v for k, v in metadata.items() 
                        if not isinstance(v, (np.ndarray, list)) or k == 'dipole_failure_types'}
        json.dump(json_metadata, f, indent=2)
    
    print(f"   Metadata: {metadata_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Combine trajectories and compute dipoles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python combine_and_compute_dipoles.py \\
        outputs/training_with_dipoles_ammonia_20260116_123456/ \\
        --method B3LYP --basis 6-31G*
        """
    )
    
    parser.add_argument('directory', type=str,
                       help='Directory containing trajectory_*K.npz files')
    parser.add_argument('--method', type=str, default='B3LYP',
                       help='QM method (default: B3LYP)')
    parser.add_argument('--basis', type=str, default='6-31G*',
                       help='Basis set (default: 6-31G*)')
    
    args = parser.parse_args()
    
    directory = Path(args.directory)
    
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        sys.exit(1)
    
    # Check if PSI4 is available
    try:
        import psi4
        print(f"✅ PSI4 {psi4.__version__}")
    except ImportError:
        print("❌ PSI4 not available - required for dipole computation")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("COMBINE TRAJECTORIES AND COMPUTE DIPOLES")
    print("="*80)
    print(f"\nDirectory: {directory}")
    print(f"Theory: {args.method}/{args.basis}")
    
    # Step 1: Load and combine
    combined_data = load_and_combine_trajectories(directory)
    
    # Step 2: Compute dipoles
    dipoles, dipole_stats = compute_dipoles_for_all_frames(
        combined_data['symbols'],
        combined_data['coordinates'],
        args.method,
        args.basis
    )
    
    # Step 3: Save
    output_file = save_combined_training_data(
        directory,
        combined_data,
        dipoles,
        dipole_stats,
        args.method,
        args.basis
    )
    
    # Summary
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    
    print(f"\n✅ Combined training data with dipoles:")
    print(f"   File: {output_file}")
    print(f"   Frames: {len(combined_data['coordinates'])}")
    print(f"   Valid dipoles: {dipole_stats['n_valid']}/{dipole_stats['n_total']}")
    print(f"   Success rate: {100*dipole_stats['n_valid']/dipole_stats['n_total']:.1f}%")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Train ML-PES (if not done yet):")
    print(f"      python complete_workflow_v2.2.py")
    print(f"")
    print(f"   2. Train ML-dipole surface:")
    print(f"      python compute_ir_workflow.py \\")
    print(f"          --train-dipole \\")
    print(f"          --training-data {output_file}")
    print(f"")
    print(f"   3. Compute IR spectrum:")
    print(f"      python compute_ir_workflow.py \\")
    print(f"          --ml-pes model.pkl \\")
    print(f"          --dipole-model dipole_surface.pkl \\")
    print(f"          --temp 300 --steps 50000")
    print()


if __name__ == '__main__':
    main()
