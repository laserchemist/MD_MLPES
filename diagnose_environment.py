#!/usr/bin/env python3
"""
Environment Diagnostic Script
==============================
Check Python, PSI4, and dipole calculation before running workflows.

Usage:
    python diagnose_environment.py

Author: Jonathan
Date: 2026-01-17
"""

import sys
import numpy as np


def check_python():
    """Check Python version."""
    print("\n" + "="*70)
    print("PYTHON ENVIRONMENT")
    print("="*70)
    
    print(f"\nPython version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version info: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    if sys.version_info < (3, 8):
        print("⚠️  WARNING: Python < 3.8 detected. Some features may not work.")
    else:
        print("✓ Python version OK")


def check_numpy():
    """Check NumPy version."""
    print("\n" + "="*70)
    print("NUMPY")
    print("="*70)
    
    print(f"\nNumPy version: {np.__version__}")
    print(f"NumPy location: {np.__file__}")
    
    # Test dtype handling (the issue you encountered)
    try:
        test_array = np.array(['test'])
        dtype_str = str(test_array.dtype)
        print(f"✓ NumPy dtype handling: {dtype_str}")
    except Exception as e:
        print(f"⚠️  NumPy dtype issue: {e}")


def check_psi4():
    """Check PSI4 availability and version."""
    print("\n" + "="*70)
    print("PSI4 QUANTUM CHEMISTRY")
    print("="*70)
    
    try:
        import psi4
        print(f"\n✓ PSI4 found")
        print(f"  Version: {psi4.__version__}")
        print(f"  Location: {psi4.__file__}")
        
        # Parse version
        version_parts = psi4.__version__.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        
        print(f"\n  Version info:")
        print(f"    Major: {major}")
        print(f"    Minor: {minor}")
        
        if major > 1 or (major == 1 and minor >= 6):
            print(f"  ✓ Modern PSI4 (≥1.6)")
            print(f"    Should use: properties=['dipole'] API")
        else:
            print(f"  ⚠️  Older PSI4 (<1.6)")
            print(f"    Should use: oeprop() API")
        
        return True, (major, minor)
        
    except ImportError:
        print("\n✗ PSI4 NOT FOUND")
        print("  Install: conda install -c conda-forge psi4")
        return False, None


def test_simple_dipole_modern(psi4_version):
    """Test dipole calculation with modern API."""
    print("\n" + "="*70)
    print("DIPOLE CALCULATION TEST - MODERN API")
    print("="*70)
    
    try:
        import psi4
        
        # Water molecule
        mol_str = """
        0 1
        O  0.000000  0.000000  0.117176
        H  0.000000  0.756950 -0.468706
        H  0.000000 -0.756950 -0.468706
        units angstrom
        no_reorient
        no_com
        """
        
        print("\nTesting with water molecule...")
        print("Method: B3LYP/6-31G*")
        print("API: Modern (properties=['dipole'])")
        
        # Clean PSI4
        psi4.core.clean_options()
        psi4.core.clean()
        psi4.core.be_quiet()
        psi4.set_memory('2 GB')
        
        # Create molecule
        mol = psi4.geometry(mol_str)
        
        # Set options
        psi4.set_options({
            'basis': '6-31G*',
            'scf_type': 'df',
            'reference': 'rhf'
        })
        
        # Modern API - request dipole
        print("\nComputing energy with dipole...")
        energy, wfn = psi4.energy('B3LYP', molecule=mol, return_wfn=True, 
                                  properties=['dipole'])
        
        print(f"✓ Energy computed: {energy:.6f} Hartree")
        
        # Get dipole
        try:
            dipole_vec = psi4.variable('B3LYP DIPOLE')
            dipole_au = np.array([dipole_vec[0], dipole_vec[1], dipole_vec[2]])
            
            # Convert to Debye
            AU_TO_DEBYE = 2.541746
            dipole_debye = dipole_au * AU_TO_DEBYE
            
            magnitude = np.linalg.norm(dipole_debye)
            
            print(f"✓ Dipole computed:")
            print(f"  Components: [{dipole_debye[0]:.4f}, {dipole_debye[1]:.4f}, {dipole_debye[2]:.4f}] Debye")
            print(f"  Magnitude: {magnitude:.4f} Debye")
            print(f"  Expected: ~1.85 Debye for water")
            
            if abs(magnitude - 1.85) < 0.2:
                print(f"  ✓ Magnitude reasonable!")
                return True
            else:
                print(f"  ⚠️  Magnitude differs from expected")
                return False
                
        except Exception as e:
            print(f"✗ Failed to get dipole: {e}")
            return False
            
    except Exception as e:
        print(f"\n✗ Dipole test failed: {e}")
        import traceback
        print("\nFull error:")
        traceback.print_exc()
        return False


def test_simple_dipole_density_matrix():
    """Test dipole calculation with density matrix method."""
    print("\n" + "="*70)
    print("DIPOLE CALCULATION TEST - DENSITY MATRIX METHOD")
    print("="*70)
    
    try:
        import psi4
        
        # Water molecule
        mol_str = """
        0 1
        O  0.000000  0.000000  0.117176
        H  0.000000  0.756950 -0.468706
        H  0.000000 -0.756950 -0.468706
        units angstrom
        no_reorient
        no_com
        """
        
        print("\nTesting with water molecule...")
        print("Method: B3LYP/6-31G*")
        print("API: Density matrix (robust)")
        
        # Clean PSI4
        psi4.core.clean_options()
        psi4.core.clean()
        psi4.core.be_quiet()
        psi4.set_memory('2 GB')
        
        # Create molecule
        mol = psi4.geometry(mol_str)
        
        # Set options
        psi4.set_options({
            'basis': '6-31G*',
            'scf_type': 'df',
            'reference': 'rhf'
        })
        
        # Compute gradient (includes SCF)
        print("\nComputing gradient...")
        gradient_result, wfn = psi4.gradient('B3LYP/6-31G*', molecule=mol, return_wfn=True)
        
        print(f"✓ Gradient computed")
        
        # Compute dipole from density matrix
        print("\nComputing dipole from density matrix...")
        
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
        
        # Convert to Debye
        AU_TO_DEBYE = 2.541746
        dipole_debye = dipole_au * AU_TO_DEBYE
        
        magnitude = np.linalg.norm(dipole_debye)
        
        print(f"✓ Dipole computed:")
        print(f"  Components: [{dipole_debye[0]:.4f}, {dipole_debye[1]:.4f}, {dipole_debye[2]:.4f}] Debye")
        print(f"  Magnitude: {magnitude:.4f} Debye")
        print(f"  Expected: ~1.85 Debye for water")
        
        if abs(magnitude - 1.85) < 0.2:
            print(f"  ✓ Magnitude reasonable!")
            return True
        else:
            print(f"  ⚠️  Magnitude differs from expected")
            return False
            
    except Exception as e:
        print(f"\n✗ Dipole test failed: {e}")
        import traceback
        print("\nFull error:")
        traceback.print_exc()
        return False


def check_sklearn():
    """Check scikit-learn."""
    print("\n" + "="*70)
    print("SCIKIT-LEARN")
    print("="*70)
    
    try:
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.preprocessing import StandardScaler
        import sklearn
        
        print(f"\n✓ scikit-learn found")
        print(f"  Version: {sklearn.__version__}")
        
        return True
    except ImportError as e:
        print(f"\n✗ scikit-learn NOT FOUND: {e}")
        print("  Install: pip install scikit-learn")
        return False


def main():
    print("="*70)
    print("ENVIRONMENT DIAGNOSTIC TOOL")
    print("="*70)
    print("\nThis will check your Python environment for:")
    print("  - Python version")
    print("  - NumPy")
    print("  - PSI4 quantum chemistry")
    print("  - Dipole calculation capability")
    print("  - scikit-learn")
    
    # Run checks
    check_python()
    check_numpy()
    check_sklearn()
    
    psi4_available, psi4_version = check_psi4()
    
    if psi4_available:
        print("\n" + "="*70)
        print("TESTING DIPOLE CALCULATIONS")
        print("="*70)
        
        # Test modern API
        modern_works = test_simple_dipole_modern(psi4_version)
        
        # Test density matrix method
        density_works = test_simple_dipole_density_matrix()
        
        # Recommendations
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        
        if modern_works:
            print("\n✓ Modern API works!")
            print("  Use: properties=['dipole'] in psi4.energy() calls")
        elif density_works:
            print("\n✓ Density matrix method works!")
            print("  Use: Density matrix computation")
        else:
            print("\n✗ BOTH methods failed!")
            print("  Check PSI4 installation")
            print("  Try: conda install -c conda-forge psi4")
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nEnvironment status:")
    print(f"  Python: {sys.version_info.major}.{sys.version_info.minor}")
    print(f"  NumPy: {np.__version__}")
    
    if psi4_available:
        import psi4
        print(f"  PSI4: {psi4.__version__}")
        
        if modern_works or density_works:
            print("\n✅ READY for ML-dipole workflow!")
        else:
            print("\n⚠️  PSI4 found but dipole calculations failing")
            print("   Troubleshooting needed")
    else:
        print("  PSI4: NOT INSTALLED")
        print("\n❌ NOT READY - Install PSI4 first")
    
    print()


if __name__ == "__main__":
    main()
