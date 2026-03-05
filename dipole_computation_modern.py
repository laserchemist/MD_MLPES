#!/usr/bin/env python3
"""
Modern Dipole Computation for Training Data Generation
=======================================================
Updated version using modern PSI4 API (properties=["dipole"]).

This is cleaner and more reliable than the density matrix approach.
Use this to replace the dipole computation in your training data scripts.

Author: Jonathan
Date: 2026-01-15
"""

import numpy as np
import psi4


def compute_dipole_modern(symbols, coords, method='B3LYP', basis='6-31G*'):
    """
    Compute dipole moment using modern PSI4 API.
    
    This uses the cleaner properties=["dipole"] approach that works
    with all modern PSI4 versions.
    
    Parameters
    ----------
    symbols : list of str
        Atomic symbols
    coords : ndarray, shape (n_atoms, 3)
        Atomic coordinates in Angstroms
    method : str
        QM method (default: B3LYP)
    basis : str
        Basis set (default: 6-31G*)
        
    Returns
    -------
    dipole : ndarray, shape (3,)
        Dipole moment in Debye [μx, μy, μz]
    error : str or None
        Error message if computation failed
    """
    # Conversion factor
    AU_TO_DEBYE = 2.541746
    
    # Create molecule string
    mol_str = f"0 1\n"
    for s, c in zip(symbols, coords):
        mol_str += f"{s} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n"
    mol_str += "units angstrom\nno_reorient\nno_com"
    
    # Clean PSI4 state
    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.be_quiet()
    psi4.set_memory('2 GB')
    psi4.set_num_threads(4)
    
    try:
        # Create molecule (relaxed validation for MD geometries)
        mol = psi4.geometry(mol_str, tooclose=0.1)
        
        # Set options
        psi4.set_options({
            'basis': basis,
            'scf_type': 'pk',  # or 'df' for density-fitted
            'reference': 'rhf',
            'maxiter': 200,
            'e_convergence': 1e-6,
            'd_convergence': 1e-6
        })
        
        # MODERN API: Request dipole in energy call
        energy, wfn = psi4.energy(
            method,
            molecule=mol,
            return_wfn=True,
            properties=["dipole"]  # This is the key!
        )
        
        # Get dipole vector (in atomic units)
        dipole_vec = psi4.variable(f"{method.upper()} DIPOLE")
        
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
        
        # Categorize common errors
        if "too close" in error_msg.lower():
            return None, "Atoms too close (collapsed geometry)"
        elif "scf" in error_msg.lower() and "converge" in error_msg.lower():
            return None, "SCF convergence failure"
        else:
            return None, error_msg[:200]


def compute_dipole_with_gradient(symbols, coords, method='B3LYP', basis='6-31G*'):
    """
    Compute dipole AND gradient using modern PSI4 API.
    
    This is more efficient if you need both dipole and forces,
    as it only runs the SCF once.
    
    Parameters
    ----------
    symbols : list of str
        Atomic symbols
    coords : ndarray, shape (n_atoms, 3)
        Atomic coordinates in Angstroms
    method : str
        QM method
    basis : str
        Basis set
        
    Returns
    -------
    dipole : ndarray, shape (3,) or None
        Dipole moment in Debye
    gradient : ndarray, shape (n_atoms, 3) or None
        Energy gradient (forces)
    error : str or None
        Error message if failed
    """
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
        # Create molecule
        mol = psi4.geometry(mol_str, tooclose=0.1)
        
        # Set options
        psi4.set_options({
            'basis': basis,
            'scf_type': 'pk',
            'reference': 'rhf',
            'maxiter': 200,
            'e_convergence': 1e-6,
            'd_convergence': 1e-6
        })
        
        # Compute gradient (includes energy) with dipole
        gradient_result, wfn = psi4.gradient(
            f"{method}/{basis}",
            molecule=mol,
            return_wfn=True
        )
        
        # After gradient, request dipole from wavefunction
        # Method 1: Use oeprop if available
        try:
            psi4.oeprop(wfn, 'DIPOLE')
            dipole_vec = psi4.variable('DIPOLE')
        except:
            # Method 2: Recompute with properties
            _, wfn2 = psi4.energy(
                method,
                molecule=mol,
                return_wfn=True,
                properties=["dipole"]
            )
            dipole_vec = psi4.variable(f"{method.upper()} DIPOLE")
        
        # Extract dipole
        dipole_au = np.array([
            dipole_vec[0],
            dipole_vec[1],
            dipole_vec[2]
        ])
        dipole_debye = dipole_au * AU_TO_DEBYE
        
        # Convert gradient to forces (negative of gradient)
        forces = -np.array(gradient_result)
        
        return dipole_debye, forces, None
        
    except Exception as e:
        error_msg = str(e)
        if "too close" in error_msg.lower():
            return None, None, "Atoms too close"
        elif "scf" in error_msg.lower() and "converge" in error_msg.lower():
            return None, None, "SCF convergence failure"
        else:
            return None, None, error_msg[:200]


# Example usage
if __name__ == "__main__":
    print("\nTesting Modern Dipole Computation")
    print("="*60)
    
    # Test water
    symbols = ['O', 'H', 'H']
    coords = np.array([
        [0.000000, 0.000000, 0.117176],
        [0.000000, 0.756950, -0.468706],
        [0.000000, -0.756950, -0.468706]
    ])
    
    print("\nMethod 1: Dipole only")
    dipole, error = compute_dipole_modern(symbols, coords)
    
    if error:
        print(f"Error: {error}")
    else:
        print(f"Dipole: [{dipole[0]:.4f}, {dipole[1]:.4f}, {dipole[2]:.4f}] Debye")
        print(f"Magnitude: {np.linalg.norm(dipole):.4f} Debye")
    
    print("\nMethod 2: Dipole + Gradient")
    dipole, forces, error = compute_dipole_with_gradient(symbols, coords)
    
    if error:
        print(f"Error: {error}")
    else:
        print(f"Dipole: [{dipole[0]:.4f}, {dipole[1]:.4f}, {dipole[2]:.4f}] Debye")
        print(f"Magnitude: {np.linalg.norm(dipole):.4f} Debye")
        print(f"Forces shape: {forces.shape}")
    
    print("\n" + "="*60)
    print("To use in your training script:")
    print("  from dipole_computation_modern import compute_dipole_modern")
    print("  dipole, error = compute_dipole_modern(symbols, coords, method, basis)")
