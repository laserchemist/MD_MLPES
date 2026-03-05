#!/usr/bin/env python3
"""
Quick Test: Verify PSI4 1.6+ Compatibility Fix
==============================================

This script tests that the updated code works with PSI4 1.6+.
Run this after updating the code to verify the fix.

Expected output:
- PSI4 version detection
- Successful dipole calculation
- Water dipole ~1.85 Debye
"""

import sys

try:
    import psi4
    import numpy as np
    print(f"✓ PSI4 version: {psi4.__version__}")
    
    # Quick test
    psi4.set_memory('500 MB')
    psi4.core.set_output_file('quick_test.dat', False)
    
    h2o = psi4.geometry("""
    O  0.0  0.0  0.117176
    H  0.0  0.756950 -0.468706
    H  0.0 -0.756950 -0.468706
    """)
    
    psi4.set_options({'basis': '6-31G*', 'scf_type': 'df'})
    
    print("\nRunning quick test calculation...")
    energy, wfn = psi4.energy('scf', molecule=h2o, return_wfn=True)
    psi4.oeprop(wfn, 'DIPOLE', title='Test')
    
    # Try to get dipole (compatible with 1.6+)
    try:
        dipole_vec = psi4.variable('Test DIPOLE')
        if hasattr(dipole_vec, '__len__') and len(dipole_vec) == 3:
            dipole_total = np.linalg.norm(dipole_vec)
            print(f"✓ Using PSI4 1.6+ API (vector return)")
        else:
            dipole_total = float(dipole_vec)
            print(f"✓ Using older PSI4 API")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    
    dipole_debye = dipole_total * 2.5417
    
    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Energy: {energy:.6f} Hartree")
    print(f"  Dipole: {dipole_total:.6f} a.u.")
    print(f"  Dipole: {dipole_debye:.3f} Debye")
    print(f"{'='*50}")
    
    if 1.5 < dipole_debye < 2.2:
        print("\n✓ TEST PASSED - Dipole in expected range!")
        print("✓ PSI4 1.6+ compatibility fix is working!")
    else:
        print(f"\n✗ TEST FAILED - Dipole {dipole_debye:.3f} D outside expected range (1.5-2.2 D)")
        sys.exit(1)
        
except ImportError:
    print("✗ PSI4 not installed")
    print("\nInstall with: conda install -c psi4 psi4")
    sys.exit(1)
