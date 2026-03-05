#!/usr/bin/env python3
"""
WORKING Dipole Calculation - Based on ChatGPT Example
======================================================
Uses the PROVEN method that actually works with PSI4 1.10.

Author: Jonathan
Date: 2026-01-17
"""

import psi4
import numpy as np


def compute_dipole_working(symbols, coords, method='scf', basis='6-31G*'):
    """
    Compute dipole using the PROVEN WORKING method.
    
    Based on ChatGPT example that actually works with PSI4 1.10.
    
    Parameters
    ----------
    symbols : list of str
        Atomic symbols
    coords : ndarray, shape (n_atoms, 3)
        Coordinates in Angstroms
    method : str
        QM method (default: 'scf', can also use 'b3lyp')
    basis : str
        Basis set
        
    Returns
    -------
    dipole : ndarray, shape (3,)
        Dipole in Debye
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
        # Create molecule (PSI4 1.10 removed tooclose parameter)
        mol = psi4.geometry(mol_str)
        
        # Settings
        psi4.set_options({
            'basis': basis,
            'scf_type': 'pk',
            'reference': 'rhf'
        })
        
        # THIS IS THE WORKING METHOD (from ChatGPT)
        energy, wfn = psi4.energy(
            method,
            molecule=mol,
            return_wfn=True,
            properties=["dipole"]
        )
        
        # Get dipole - use "SCF DIPOLE" (this is what works!)
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


# Test it
if __name__ == "__main__":
    print("\nTesting WORKING Dipole Calculation")
    print("="*60)
    
    # Water molecule
    symbols = ['O', 'H', 'H']
    coords = np.array([
        [0.000000, 0.000000, 0.117176],
        [0.000000, 0.756950, -0.468706],
        [0.000000, -0.756950, -0.468706]
    ])
    
    print("\nWater molecule (equilibrium geometry)")
    print("Method: scf/6-31G*")
    print("API: PROVEN WORKING (ChatGPT example)")
    
    dipole, error = compute_dipole_working(symbols, coords, method='scf', basis='6-31G*')
    
    if error:
        print(f"\n✗ Error: {error}")
    else:
        magnitude = np.linalg.norm(dipole)
        print(f"\n✓ SUCCESS!")
        print(f"  Dipole: [{dipole[0]:.4f}, {dipole[1]:.4f}, {dipole[2]:.4f}] Debye")
        print(f"  Magnitude: {magnitude:.4f} Debye")
        print(f"  Expected: ~1.85 Debye")
        
        if abs(magnitude - 1.85) < 0.2:
            print(f"  ✓ Value reasonable!")
        else:
            print(f"  ⚠️  Value differs from expected")
    
    # Test with DFT (B3LYP)
    print("\n" + "-"*60)
    print("Testing with B3LYP...")
    
    dipole, error = compute_dipole_working(symbols, coords, method='b3lyp', basis='6-31G*')
    
    if error:
        print(f"✗ Error: {error}")
    else:
        magnitude = np.linalg.norm(dipole)
        print(f"✓ SUCCESS!")
        print(f"  Dipole: [{dipole[0]:.4f}, {dipole[1]:.4f}, {dipole[2]:.4f}] Debye")
        print(f"  Magnitude: {magnitude:.4f} Debye")
    
    print("\n" + "="*60)
    print("Use this function in combine_and_compute_dipoles.py!")
    print("="*60)
