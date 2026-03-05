#!/usr/bin/env python3
"""
Simple Test: Dipole Moment Calculation with PSI4 (Modern API)
==============================================================
Uses the modern PSI4 API with properties=["dipole"] in energy() call.
This approach is cleaner and more straightforward than using oeprop().

Based on working example, updated for production use.

Author: Jonathan
Date: 2026-01-15
"""

import numpy as np

try:
    import psi4
    PSI4_AVAILABLE = True
except ImportError:
    print("\n" + "!"*60)
    print("PSI4 not installed - Running in DEMO mode")
    print("!"*60)
    PSI4_AVAILABLE = False

# More precise conversion factor
AU_TO_DEBYE = 2.541746


def calculate_dipole_moment(molecule_name='H2O', geometry_string=None, 
                           basis='cc-pVDZ', method='scf'):
    """
    Calculate dipole moment for a molecule using modern PSI4 API.
    
    Parameters
    ----------
    molecule_name : str
        Name of the molecule (for display)
    geometry_string : str
        Z-matrix or Cartesian geometry specification
    basis : str
        Basis set (default: cc-pVDZ)
    method : str
        QM method (default: scf)
    
    Returns
    -------
    dict
        Results including energy, dipole vector, and magnitude
    """
    
    if not PSI4_AVAILABLE:
        print(f"\n{'='*60}")
        print("DEMO MODE: Expected workflow")
        print('='*60)
        print(f"\nMolecule: {molecule_name}")
        print(f"Basis: {basis}")
        print(f"Method: {method}")
        print("\nExpected code:")
        print("  energy, wfn = psi4.energy('scf', return_wfn=True,")
        print("                            properties=['dipole'])")
        print("  dipole_vec = psi4.variable('SCF DIPOLE')")
        print("\nExpected output for H2O:")
        print("  Dipole: [0.000, 0.000, 0.728] a.u.")
        print("  Magnitude: 1.85 Debye")
        return None
    
    # Set up PSI4
    psi4.set_memory('500 MB')
    psi4.core.set_output_file(f'{molecule_name.lower()}_dipole.dat', False)
    
    # Define molecule
    if geometry_string is None:
        # Default: water molecule in Z-matrix format
        geometry_string = """
        0 1
        O
        H 1 0.96
        H 1 0.96 2 104.5
        """
    
    mol = psi4.geometry(geometry_string)
    
    # Set computational parameters
    psi4.set_options({
        'basis': basis,
        'scf_type': 'pk'  # Use conventional integrals
    })
    
    # Run calculation with dipole property request
    print(f"\n{'='*60}")
    print(f"Running {method.upper()} calculation for {molecule_name}")
    print(f"Basis: {basis}")
    print('='*60)
    
    energy, wfn = psi4.energy(
        method,
        molecule=mol,
        return_wfn=True,
        properties=["dipole"]  # Modern way to request properties!
    )
    
    print(f"\n{method.upper()} Energy: {energy:.10f} Hartree")
    
    # Retrieve dipole vector (in atomic units)
    # Can use either psi4.variable() or wfn.variable()
    dipole_vec = psi4.variable(f"{method.upper()} DIPOLE")
    # Or equivalently: dipole_vec = wfn.variable(f"{method.upper()} DIPOLE")
    
    # Convert to numpy array and Debye
    dipole_au = np.array(dipole_vec)
    dipole_debye = dipole_au * AU_TO_DEBYE
    magnitude_debye = np.linalg.norm(dipole_debye)
    
    # Print results
    print("\nDipole Moment Results:")
    print("-" * 40)
    print(f"  Vector (a.u.):   [{dipole_au[0]:8.5f}, {dipole_au[1]:8.5f}, {dipole_au[2]:8.5f}]")
    print(f"  Vector (Debye):  [{dipole_debye[0]:8.5f}, {dipole_debye[1]:8.5f}, {dipole_debye[2]:8.5f}]")
    print(f"  Magnitude:       {magnitude_debye:10.6f} Debye")
    
    # Store results
    results = {
        'molecule': molecule_name,
        'method': method,
        'basis': basis,
        'energy': energy,
        'dipole_au': dipole_au,
        'dipole_debye': dipole_debye,
        'magnitude_debye': magnitude_debye,
        'dipole_x': dipole_au[0],
        'dipole_y': dipole_au[1],
        'dipole_z': dipole_au[2]
    }
    
    # Validate for water
    if molecule_name == 'H2O':
        expected = 1.85  # Debye
        tolerance = 0.3
        
        print("\n" + "="*60)
        print("Validation:")
        print("="*60)
        if abs(magnitude_debye - expected) < tolerance:
            print(f"✓ PASSED: Dipole = {magnitude_debye:.3f} Debye")
            print(f"  (Expected ~{expected} D, tolerance ±{tolerance} D)")
        else:
            print(f"✗ WARNING: Dipole = {magnitude_debye:.3f} Debye")
            print(f"  (Expected ~{expected} D, tolerance ±{tolerance} D)")
    
    return results


def test_multiple_molecules():
    """Test dipole calculations for multiple molecules."""
    
    if not PSI4_AVAILABLE:
        print("\nPSI4 not available - skipping multiple molecule tests")
        return
    
    print("\n" + "="*60)
    print("Testing Multiple Molecules")
    print("="*60)
    
    # Water
    h2o_geom = """
    0 1
    O
    H 1 0.96
    H 1 0.96 2 104.5
    """
    h2o_results = calculate_dipole_moment('H2O', h2o_geom)
    
    # Ammonia
    nh3_geom = """
    0 1
    N
    H 1 1.01
    H 1 1.01 2 106.7
    H 1 1.01 2 106.7 3 120.0
    """
    nh3_results = calculate_dipole_moment('NH3', nh3_geom)
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"H2O:  {h2o_results['magnitude_debye']:.3f} Debye")
    print(f"NH3:  {nh3_results['magnitude_debye']:.3f} Debye")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PSI4 Dipole Moment Calculation (Modern API)")
    print("="*60)
    
    # Test water molecule
    results = calculate_dipole_moment()
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60 + "\n")
    
    # Optionally test more molecules
    if PSI4_AVAILABLE and input("\nTest more molecules? (y/n): ").lower() == 'y':
        test_multiple_molecules()
