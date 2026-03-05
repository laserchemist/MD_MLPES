#!/usr/bin/env python3
"""
Unit Tests for PSI4 Dipole Calculations (Modern API)
====================================================
Tests using the cleaner properties=["dipole"] approach.

Run with: pytest test_dipole_modern.py -v

Author: Jonathan
Date: 2026-01-15
"""

import pytest
import numpy as np

# Check if PSI4 is available
try:
    import psi4
    PSI4_AVAILABLE = True
except ImportError:
    PSI4_AVAILABLE = False

AU_TO_DEBYE = 2.541746


class TestModernDipoleAPI:
    """Test dipole calculations using modern PSI4 API."""
    
    @pytest.mark.skipif(not PSI4_AVAILABLE, reason="PSI4 not installed")
    def test_water_dipole_modern_api(self):
        """Test dipole calculation using properties=['dipole'] in energy call."""
        # Setup
        psi4.set_memory('500 MB')
        psi4.core.set_output_file('test_modern_h2o.dat', False)
        
        # Water geometry (Z-matrix)
        h2o = psi4.geometry("""
        0 1
        O
        H 1 0.96
        H 1 0.96 2 104.5
        """)
        
        psi4.set_options({'basis': 'cc-pVDZ', 'scf_type': 'pk'})
        
        # Modern API: request dipole in energy call
        energy, wfn = psi4.energy(
            'scf',
            molecule=h2o,
            return_wfn=True,
            properties=["dipole"]
        )
        
        # Get dipole vector
        dipole_vec = psi4.variable('SCF DIPOLE')
        dipole_au = np.array(dipole_vec)
        magnitude_debye = np.linalg.norm(dipole_au) * AU_TO_DEBYE
        
        # Assertions
        assert energy < 0, "SCF energy should be negative"
        assert 1.5 < magnitude_debye < 2.2, \
            f"Water dipole {magnitude_debye:.3f} D outside expected range"
    
    @pytest.mark.skipif(not PSI4_AVAILABLE, reason="PSI4 not installed")
    def test_dipole_from_wavefunction(self):
        """Test that dipole can be accessed from wavefunction object."""
        psi4.set_memory('500 MB')
        psi4.core.set_output_file('test_wfn_access.dat', False)
        
        h2o = psi4.geometry("""
        0 1
        O
        H 1 0.96
        H 1 0.96 2 104.5
        """)
        
        psi4.set_options({'basis': 'cc-pVDZ', 'scf_type': 'pk'})
        
        energy, wfn = psi4.energy('scf', molecule=h2o, return_wfn=True,
                                  properties=["dipole"])
        
        # Access via psi4.variable()
        dipole1 = psi4.variable('SCF DIPOLE')
        
        # Access via wfn.variable()
        dipole2 = wfn.variable('SCF DIPOLE')
        
        # Both should give same result
        np.testing.assert_array_almost_equal(dipole1, dipole2, decimal=10)
    
    @pytest.mark.skipif(not PSI4_AVAILABLE, reason="PSI4 not installed")
    def test_conversion_factor(self):
        """Test atomic units to Debye conversion."""
        # Known test case
        dipole_au = 0.728  # Typical water Z-component
        dipole_debye = dipole_au * AU_TO_DEBYE
        
        # Should be around 1.85 Debye
        assert 1.8 < dipole_debye < 1.9, \
            f"Conversion gives {dipole_debye:.3f} D, expected ~1.85 D"
    
    @pytest.mark.skipif(not PSI4_AVAILABLE, reason="PSI4 not installed")
    def test_symmetry_components(self):
        """Test that water has only Z-component due to symmetry."""
        psi4.set_memory('500 MB')
        psi4.core.set_output_file('test_symmetry_modern.dat', False)
        
        # Standard water orientation with C2v symmetry
        h2o = psi4.geometry("""
        0 1
        O
        H 1 0.96
        H 1 0.96 2 104.5
        """)
        
        psi4.set_options({'basis': 'cc-pVDZ', 'scf_type': 'pk'})
        
        energy, wfn = psi4.energy('scf', molecule=h2o, return_wfn=True,
                                  properties=["dipole"])
        
        dipole_vec = np.array(psi4.variable('SCF DIPOLE'))
        
        # X and Y components should be very small
        assert abs(dipole_vec[0]) < 1e-6, "X component should be ~0 by symmetry"
        assert abs(dipole_vec[1]) < 1e-6, "Y component should be ~0 by symmetry"
        assert abs(dipole_vec[2]) > 0.5, "Z component should be significant"
    
    @pytest.mark.skipif(not PSI4_AVAILABLE, reason="PSI4 not installed")
    def test_ammonia_dipole(self):
        """Test dipole calculation for ammonia."""
        psi4.set_memory('500 MB')
        psi4.core.set_output_file('test_nh3.dat', False)
        
        nh3 = psi4.geometry("""
        0 1
        N
        H 1 1.01
        H 1 1.01 2 106.7
        H 1 1.01 2 106.7 3 120.0
        """)
        
        psi4.set_options({'basis': 'cc-pVDZ', 'scf_type': 'pk'})
        
        energy, wfn = psi4.energy('scf', molecule=nh3, return_wfn=True,
                                  properties=["dipole"])
        
        dipole_vec = np.array(psi4.variable('SCF DIPOLE'))
        magnitude_debye = np.linalg.norm(dipole_vec) * AU_TO_DEBYE
        
        # Ammonia has dipole around 1.47 Debye
        assert 1.2 < magnitude_debye < 1.8, \
            f"NH3 dipole {magnitude_debye:.3f} D outside expected range"


class TestBasisSetEffects:
    """Test basis set effects on dipole moments."""
    
    @pytest.mark.skipif(not PSI4_AVAILABLE, reason="PSI4 not installed")
    def test_basis_set_comparison(self):
        """Compare dipole moments with different basis sets."""
        psi4.set_memory('500 MB')
        
        h2o = psi4.geometry("""
        0 1
        O
        H 1 0.96
        H 1 0.96 2 104.5
        """)
        
        basis_sets = ['6-31G', 'cc-pVDZ', 'cc-pVTZ']
        dipoles = []
        
        for basis in basis_sets:
            psi4.core.set_output_file(f'test_{basis.lower()}.dat', False)
            psi4.set_options({'basis': basis, 'scf_type': 'pk'})
            
            energy, wfn = psi4.energy('scf', molecule=h2o, return_wfn=True,
                                      properties=["dipole"])
            
            dipole_vec = np.array(psi4.variable('SCF DIPOLE'))
            magnitude = np.linalg.norm(dipole_vec) * AU_TO_DEBYE
            dipoles.append(magnitude)
        
        # All should be in reasonable range
        for dipole in dipoles:
            assert 1.5 < dipole < 2.2, \
                f"Dipole {dipole:.3f} D outside reasonable range"
        
        # Larger basis sets should converge
        diff = abs(dipoles[-1] - dipoles[-2])
        assert diff < 0.1, "cc-pVDZ and cc-pVTZ should be close"


class TestCartesianGeometry:
    """Test using Cartesian coordinates instead of Z-matrix."""
    
    @pytest.mark.skipif(not PSI4_AVAILABLE, reason="PSI4 not installed")
    def test_cartesian_input(self):
        """Test dipole calculation with Cartesian coordinates."""
        psi4.set_memory('500 MB')
        psi4.core.set_output_file('test_cartesian.dat', False)
        
        # Water in Cartesian coordinates
        h2o = psi4.geometry("""
        0 1
        O  0.000000  0.000000  0.117176
        H  0.000000  0.756950 -0.468706
        H  0.000000 -0.756950 -0.468706
        """)
        
        psi4.set_options({'basis': 'cc-pVDZ', 'scf_type': 'pk'})
        
        energy, wfn = psi4.energy('scf', molecule=h2o, return_wfn=True,
                                  properties=["dipole"])
        
        dipole_vec = np.array(psi4.variable('SCF DIPOLE'))
        magnitude_debye = np.linalg.norm(dipole_vec) * AU_TO_DEBYE
        
        # Should give similar result to Z-matrix
        assert 1.5 < magnitude_debye < 2.2, \
            f"Cartesian dipole {magnitude_debye:.3f} D outside expected range"


def test_conversion_constants():
    """Test that conversion factor is correct."""
    # CODATA 2018 value: 1 a.u. = 2.541746 Debye
    assert abs(AU_TO_DEBYE - 2.541746) < 1e-6, \
        "Conversion factor incorrect"


if __name__ == "__main__":
    print("Running dipole moment calculation tests (modern API)...")
    print("\nTo run with pytest:")
    print("  pytest test_dipole_modern.py -v")
    print("\nTo see which tests will run:")
    print("  pytest test_dipole_modern.py --collect-only")
