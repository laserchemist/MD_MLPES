#!/usr/bin/env python3
"""
Updated PSI4 Computation Functions with Dipole Moments

Add these functions to your workflows to automatically compute and store
dipole moments alongside energies and forces.

Author: PSI4-MD ML-PES Framework
Date: January 2026
"""

import numpy as np

def compute_psi4_energy_forces_dipole(symbols, coords, method='B3LYP', basis='6-31G*'):
    """
    Compute energy, forces, AND dipole moment with PSI4.
    
    This is the KEY UPDATE - adds dipole computation to existing calculations.
    
    Args:
        symbols: Atomic symbols list
        coords: Coordinates in Angstroms (n_atoms, 3)
        method: QM method (default: B3LYP)
        basis: Basis set (default: 6-31G*)
        
    Returns:
        energy: Energy in Hartree
        forces: Forces in Hartree/Angstrom (n_atoms, 3)
        dipole: Dipole moment in Debye (3,)
        error: Error message or None
    """
    import psi4
    
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
        # Compute properties (energy + dipole in one call)
        energy, wfn = psi4.properties(
            method, 
            properties=['dipole'], 
            return_wfn=True, 
            molecule=mol
        )
        
        # Get dipole moment (in Debye)
        dipole = np.array(wfn.variable('SCF DIPOLE'))
        
        # Compute forces
        gradient = psi4.gradient(method, molecule=mol)
        forces = -np.array(gradient)
        
        return energy, forces, dipole, None
        
    except Exception as e:
        return None, None, None, str(e)[:100]


def compute_psi4_snapshot_with_dipole(symbols, coords, method, basis):
    """
    Wrapper for single snapshot computation with dipole.
    
    Returns tuple of (energy, forces, dipole) or (None, None, None) on error.
    """
    energy, forces, dipole, error = compute_psi4_energy_forces_dipole(
        symbols, coords, method, basis
    )
    
    if error:
        return None, None, None
    
    return energy, forces, dipole


# ==============================================================================
# HELPER FUNCTIONS FOR WORKFLOW INTEGRATION
# ==============================================================================

def validate_snapshots_with_dipoles(snapshots, symbols, method, basis, max_validate=50):
    """
    Validate snapshots and collect dipoles.
    
    This replaces the standard validation loop to include dipole collection.
    
    Args:
        snapshots: List of coordinate arrays
        symbols: Atomic symbols
        method: QM method
        basis: Basis set
        max_validate: Maximum number to validate
        
    Returns:
        Dictionary with:
            - energies_psi4: List of energies
            - forces_psi4: List of force arrays
            - dipoles_psi4: List of dipole vectors (NEW!)
            - errors: List of error messages
    """
    import psi4
    from tqdm import tqdm
    
    # Sample if too many
    if len(snapshots) > max_validate:
        indices = np.linspace(0, len(snapshots)-1, max_validate, dtype=int)
        selected_snapshots = [snapshots[i] for i in indices]
    else:
        selected_snapshots = snapshots
    
    energies_psi4 = []
    forces_psi4 = []
    dipoles_psi4 = []  # NEW!
    errors = []
    
    print(f"\n🔄 Validating {len(selected_snapshots)} snapshots with PSI4...")
    print(f"   Computing: Energy + Forces + Dipole")
    
    for i, coords in enumerate(tqdm(selected_snapshots, desc="PSI4 validation")):
        energy, forces, dipole, error = compute_psi4_energy_forces_dipole(
            symbols, coords, method, basis
        )
        
        if error:
            energies_psi4.append(None)
            forces_psi4.append(None)
            dipoles_psi4.append(None)
            errors.append(error)
        else:
            energies_psi4.append(energy)
            forces_psi4.append(forces)
            dipoles_psi4.append(dipole)  # NEW!
            errors.append(None)
    
    return {
        'energies_psi4': energies_psi4,
        'forces_psi4': forces_psi4,
        'dipoles_psi4': dipoles_psi4,  # NEW!
        'errors': errors,
        'n_success': sum(1 for e in errors if e is None),
        'n_failed': sum(1 for e in errors if e is not None)
    }


def save_trajectory_with_dipoles(trajectory_data, dipoles, output_path):
    """
    Save trajectory with dipole moments in metadata.
    
    Args:
        trajectory_data: TrajectoryData object
        dipoles: List or array of dipole moments (n_frames, 3)
        output_path: Output file path
    """
    from modules.data_formats import save_trajectory
    
    # Add dipoles to metadata
    if trajectory_data.metadata is None:
        trajectory_data.metadata = {}
    
    trajectory_data.metadata['dipoles'] = np.array(dipoles)
    trajectory_data.metadata['dipoles_units'] = 'Debye'
    trajectory_data.metadata['has_dipoles'] = True
    
    # Save
    save_trajectory(trajectory_data, output_path)
    
    print(f"   ✅ Saved {len(dipoles)} dipole moments in metadata")


def extract_dipoles_from_trajectory(trajectory_path):
    """
    Extract dipole moments from saved trajectory.
    
    Args:
        trajectory_path: Path to .npz file
        
    Returns:
        dipoles: Array of dipole moments (n_frames, 3) or None
    """
    from modules.data_formats import load_trajectory
    
    traj = load_trajectory(trajectory_path)
    
    if traj.metadata and 'dipoles' in traj.metadata:
        return traj.metadata['dipoles']
    else:
        return None


def check_trajectory_has_dipoles(trajectory_path):
    """
    Check if trajectory file contains dipole data.
    
    Returns:
        bool: True if dipoles present
    """
    dipoles = extract_dipoles_from_trajectory(trajectory_path)
    
    if dipoles is not None:
        print(f"   ✅ Found {len(dipoles)} dipole moments")
        return True
    else:
        print(f"   ❌ No dipoles in trajectory")
        return False


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == '__main__':
    """
    Example: How to use these functions in your workflow
    """
    
    print("=" * 80)
    print("  EXAMPLE: PSI4 Computation with Dipoles")
    print("=" * 80)
    
    # Example molecule (water)
    symbols = ['O', 'H', 'H']
    coords = np.array([
        [0.000000, 0.000000, 0.117181],
        [0.000000, 0.755453, -0.468724],
        [0.000000, -0.755453, -0.468724]
    ])
    
    print("\n📊 Computing properties for H2O...")
    
    energy, forces, dipole, error = compute_psi4_energy_forces_dipole(
        symbols, coords, method='B3LYP', basis='6-31G*'
    )
    
    if error:
        print(f"   ❌ Error: {error}")
    else:
        print(f"   ✅ Energy: {energy:.8f} Hartree")
        print(f"   ✅ Forces: {forces.shape}")
        print(f"   ✅ Dipole: [{dipole[0]:.4f}, {dipole[1]:.4f}, {dipole[2]:.4f}] Debye")
        print(f"   ✅ |μ| = {np.linalg.norm(dipole):.4f} Debye")
    
    print("\n" + "=" * 80)
    print("  To integrate into your workflow:")
    print("=" * 80)
    print("""
    1. Replace compute_psi4_energy_forces_simple with compute_psi4_energy_forces_dipole
    2. Store dipoles in trajectory metadata using save_trajectory_with_dipoles
    3. Extract dipoles later with extract_dipoles_from_trajectory
    4. Use accumulated dipoles to train ML dipole surface
    """)
