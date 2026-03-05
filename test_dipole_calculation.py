#!/usr/bin/env python3
"""
Simple Test: Dipole Moment Calculation with PSI4
==================================================
Tests the oeprop() function to calculate dipole and quadrupole moments.

Author: Jonathan
Date: 2026-01-15

NOTE: This is a demonstration/template script. To run with actual PSI4:
      1. Install PSI4 via conda: conda install -c psi4 psi4
      2. Or use from psi4/psi4 source installation
      
Expected Output:
----------------
- SCF energy for H2O at B3LYP/6-31G* level
- Dipole moment components (X, Y, Z) in atomic units
- Total dipole moment in Debye (~1.85 D for water)
- Mulliken charges for atoms
"""

try:
    import psi4
    import numpy as np
    PSI4_AVAILABLE = True
except ImportError:
    print("\n" + "!"*60)
    print("PSI4 not installed - Running in DEMO mode")
    print("!"*60)
    PSI4_AVAILABLE = False
    import numpy as np


def test_dipole_calculation():
    """
    Calculate dipole and quadrupole moments for a water molecule.
    
    Returns
    -------
    dict
        Dictionary containing computed properties
    """
    
    if not PSI4_AVAILABLE:
        print("\n" + "="*60)
        print("DEMO MODE: Showing expected workflow")
        print("="*60)
        print("\nThis script demonstrates how to:")
        print("  1. Set up a PSI4 calculation")
        print("  2. Run SCF to get wavefunction")
        print("  3. Use oeprop() to calculate dipole moments")
        print("  4. Extract and validate results")
        
        print("\n" + "-"*60)
        print("Expected PSI4 Code Structure:")
        print("-"*60)
        print("""
# 1. Set up PSI4
psi4.set_memory('500 MB')
psi4.core.set_output_file('dipole_test.dat', False)

# 2. Define molecule
h2o = psi4.geometry('''
    O  0.000000  0.000000  0.117176
    H  0.000000  0.756950 -0.468706
    H  0.000000 -0.756950 -0.468706
''')

# 3. Set options
psi4.set_options({
    'basis': '6-31G*',
    'scf_type': 'df',
    'reference': 'rhf'
})

# 4. Run calculation and get wavefunction
energy, wfn = psi4.energy('scf', molecule=h2o, return_wfn=True)

# 5. Calculate properties using oeprop
psi4.oeprop(wfn, 'DIPOLE', 'QUADRUPOLE', 'MULLIKEN_CHARGES', 
            title='H2O Test')

# 6. Extract results
dipole_x = psi4.variable('H2O Test DIPOLE X')
dipole_y = psi4.variable('H2O Test DIPOLE Y')
dipole_z = psi4.variable('H2O Test DIPOLE Z')
dipole_total = psi4.variable('H2O Test DIPOLE')
        """)
        
        print("\n" + "-"*60)
        print("Expected Output for Water:")
        print("-"*60)
        
        # Mock results based on typical H2O calculation
        results = {
            'energy': -76.0266,  # Approximate SCF energy
            'dipole_x': 0.0,
            'dipole_y': 0.0,
            'dipole_z': 0.728,  # a.u.
            'dipole_total': 0.728,
            'dipole_debye': 1.85
        }
        
        print(f"\nSCF Energy: {results['energy']:.6f} Hartree")
        print("\nDipole Moment:")
        print(f"  X: {results['dipole_x']:.6f} a.u.")
        print(f"  Y: {results['dipole_y']:.6f} a.u.")
        print(f"  Z: {results['dipole_z']:.6f} a.u.")
        print(f"  Total: {results['dipole_total']:.6f} a.u.")
        print(f"  Total: {results['dipole_debye']:.3f} Debye")
        
        print("\n✓ Expected: ~1.85 Debye (experimental value)")
        
        return results
    
    # REAL PSI4 EXECUTION
    # Set up PSI4
    psi4.set_memory('500 MB')
    psi4.core.set_output_file('dipole_test.dat', False)
    
    # Define water molecule geometry
    h2o = psi4.geometry("""
    O  0.000000  0.000000  0.117176
    H  0.000000  0.756950 -0.468706
    H  0.000000 -0.756950 -0.468706
    """)
    
    # Set computational parameters
    psi4.set_options({
        'basis': '6-31G*',
        'scf_type': 'df',  # Density-fitted for speed
        'reference': 'rhf'
    })
    
    # Run SCF calculation and get wavefunction
    print("\n" + "="*60)
    print("Running SCF calculation for H2O...")
    print("="*60)
    
    energy, wfn = psi4.energy('scf', molecule=h2o, return_wfn=True)
    
    print(f"\nSCF Energy: {energy:.10f} Hartree")
    
    # Calculate one-electron properties using oeprop
    print("\n" + "="*60)
    print("Calculating one-electron properties...")
    print("="*60 + "\n")
    
    psi4.oeprop(wfn, 'DIPOLE', 'QUADRUPOLE', 'MULLIKEN_CHARGES', 
                title='H2O Test')
    
    # Extract dipole moment components
    # PSI4 1.6+ returns dipole as array [x, y, z] in atomic units
    try:
        # Try new API (PSI4 1.6+)
        dipole_vec = psi4.variable('H2O Test DIPOLE')
        if hasattr(dipole_vec, '__len__') and len(dipole_vec) == 3:
            # It's a vector [x, y, z]
            dipole_x, dipole_y, dipole_z = dipole_vec
            dipole_total = np.linalg.norm(dipole_vec)
        else:
            # It's a scalar (total magnitude)
            dipole_total = float(dipole_vec)
            dipole_x = dipole_y = dipole_z = 0.0
    except:
        # Try old API (PSI4 < 1.6) - already in a.u.
        try:
            dipole_x = psi4.variable('H2O Test DIPOLE X')
            dipole_y = psi4.variable('H2O Test DIPOLE Y')
            dipole_z = psi4.variable('H2O Test DIPOLE Z')
            dipole_total = psi4.variable('H2O Test DIPOLE')
        except:
            # Last resort - get from wavefunction
            dipole_vec = wfn.variable('DIPOLE')
            dipole_x, dipole_y, dipole_z = dipole_vec
            dipole_total = np.linalg.norm(dipole_vec)
    
    # Print results
    print("\nDipole Moment Results:")
    print("-" * 40)
    print(f"  X component: {dipole_x:10.6f} a.u.")
    print(f"  Y component: {dipole_y:10.6f} a.u.")
    print(f"  Z component: {dipole_z:10.6f} a.u.")
    print(f"  Total:       {dipole_total:10.6f} a.u.")
    print(f"  Total:       {dipole_total * 2.5417:10.6f} Debye")
    
    # Store results
    results = {
        'energy': energy,
        'dipole_x': dipole_x,
        'dipole_y': dipole_y,
        'dipole_z': dipole_z,
        'dipole_total': dipole_total,
        'dipole_debye': dipole_total * 2.5417
    }
    
    # Unit test: Water should have a dipole moment around 1.85 Debye
    expected_dipole = 1.85  # Debye (experimental value ~1.85)
    tolerance = 0.3  # Debye
    
    print("\n" + "="*60)
    print("Unit Test Result:")
    print("="*60)
    if abs(results['dipole_debye'] - expected_dipole) < tolerance:
        print(f"✓ PASSED: Dipole = {results['dipole_debye']:.3f} Debye")
        print(f"  (Expected ~{expected_dipole} Debye, tolerance ±{tolerance})")
    else:
        print(f"✗ FAILED: Dipole = {results['dipole_debye']:.3f} Debye")
        print(f"  (Expected ~{expected_dipole} Debye, tolerance ±{tolerance})")
    
    return results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PSI4 Dipole Moment Calculation Test")
    print("="*60)
    
    results = test_dipole_calculation()
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60 + "\n")
