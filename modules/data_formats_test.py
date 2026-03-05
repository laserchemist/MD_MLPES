#!/usr/bin/env python3
"""
Unit Tests for Data Formats Module

Tests trajectory data handling and format conversion.

Author: PSI4-MD Framework
Date: 2025
"""

import unittest
import numpy as np
import sys
import tempfile
import shutil
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'modules'))

from data_formats import (
    TrajectoryData, save_trajectory, load_trajectory, convert_format,
    XYZHandler, ExtendedXYZHandler, NPZHandler
)


class TestTrajectoryData(unittest.TestCase):
    """Test TrajectoryData class."""
    
    def setUp(self):
        """Create test trajectory data."""
        self.n_frames = 10
        self.n_atoms = 3
        
        self.test_traj = TrajectoryData(
            symbols=['O', 'H', 'H'],
            coordinates=np.random.randn(self.n_frames, self.n_atoms, 3),
            energies=np.random.randn(self.n_frames) * 0.01 - 76.0,
            forces=np.random.randn(self.n_frames, self.n_atoms, 3) * 0.01,
            times=np.arange(self.n_frames) * 0.5,
            metadata={'molecule': 'water', 'method': 'B3LYP'}
        )
    
    def test_trajectory_creation(self):
        """Test creating a trajectory."""
        self.assertEqual(self.test_traj.n_frames, self.n_frames)
        self.assertEqual(self.test_traj.n_atoms, self.n_atoms)
        self.assertEqual(len(self.test_traj.symbols), self.n_atoms)
    
    def test_shape_validation(self):
        """Test that shape validation works."""
        # This should raise an assertion error (mismatched shapes)
        with self.assertRaises(AssertionError):
            TrajectoryData(
                symbols=['O', 'H', 'H'],
                coordinates=np.random.randn(10, 3, 3),
                energies=np.random.randn(5),  # Wrong number!
                forces=np.random.randn(10, 3, 3)
            )
    
    def test_get_frame(self):
        """Test getting a single frame."""
        frame = self.test_traj.get_frame(0)
        
        self.assertIn('coordinates', frame)
        self.assertIn('energy', frame)
        self.assertIn('forces', frame)
        self.assertEqual(frame['coordinates'].shape, (self.n_atoms, 3))
    
    def test_slice_trajectory(self):
        """Test slicing a trajectory."""
        sliced = self.test_traj.slice(0, 5, 1)
        
        self.assertEqual(sliced.n_frames, 5)
        self.assertEqual(sliced.n_atoms, self.n_atoms)
        
        np.testing.assert_array_equal(
            sliced.coordinates[0],
            self.test_traj.coordinates[0]
        )


class TestFormatHandlers(unittest.TestCase):
    """Test format handlers."""
    
    def setUp(self):
        """Create test data and temporary directory."""
        self.test_dir = tempfile.mkdtemp()
        
        self.test_traj = TrajectoryData(
            symbols=['O', 'H', 'H'],
            coordinates=np.random.randn(5, 3, 3) * 0.1 + np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
            energies=np.random.randn(5) * 0.01 - 76.0,
            forces=np.random.randn(5, 3, 3) * 0.01,
            dipoles=np.random.randn(5, 3) * 0.1,
            times=np.arange(5) * 0.5,
            metadata={'molecule': 'water', 'method': 'test'}
        )
    
    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.test_dir)
    
    def test_xyz_save_load(self):
        """Test XYZ format save/load."""
        handler = XYZHandler()
        filepath = Path(self.test_dir) / 'test.xyz'
        
        # Save
        handler.save(self.test_traj, str(filepath))
        self.assertTrue(filepath.exists())
        
        # Load
        loaded = handler.load(str(filepath))
        
        self.assertEqual(loaded.n_frames, self.test_traj.n_frames)
        self.assertEqual(loaded.n_atoms, self.test_traj.n_atoms)
        
        # Coordinates should match
        np.testing.assert_array_almost_equal(
            loaded.coordinates, self.test_traj.coordinates, decimal=6
        )
    
    def test_extxyz_save_load(self):
        """Test extended XYZ format save/load."""
        handler = ExtendedXYZHandler()
        filepath = Path(self.test_dir) / 'test.extxyz'
        
        # Save
        handler.save(self.test_traj, str(filepath))
        self.assertTrue(filepath.exists())
        
        # Load
        loaded = handler.load(str(filepath))
        
        self.assertEqual(loaded.n_frames, self.test_traj.n_frames)
        
        # Coordinates and forces should match
        np.testing.assert_array_almost_equal(
            loaded.coordinates, self.test_traj.coordinates, decimal=6
        )
        np.testing.assert_array_almost_equal(
            loaded.forces, self.test_traj.forces, decimal=6
        )
    
    def test_npz_save_load(self):
        """Test NPZ format save/load."""
        handler = NPZHandler()
        filepath = Path(self.test_dir) / 'test.npz'
        
        # Save
        handler.save(self.test_traj, str(filepath))
        self.assertTrue(filepath.exists())
        
        # Load
        loaded = handler.load(str(filepath))
        
        self.assertEqual(loaded.n_frames, self.test_traj.n_frames)
        
        # All arrays should match exactly
        np.testing.assert_array_equal(loaded.coordinates, self.test_traj.coordinates)
        np.testing.assert_array_equal(loaded.energies, self.test_traj.energies)
        np.testing.assert_array_equal(loaded.forces, self.test_traj.forces)
        
        # Check optional data
        self.assertIsNotNone(loaded.dipoles)
        np.testing.assert_array_equal(loaded.dipoles, self.test_traj.dipoles)


class TestFormatConversion(unittest.TestCase):
    """Test format conversion functions."""
    
    def setUp(self):
        """Create test directory and data."""
        self.test_dir = tempfile.mkdtemp()
        
        self.test_traj = TrajectoryData(
            symbols=['C', 'O'],
            coordinates=np.random.randn(3, 2, 3),
            energies=np.random.randn(3),
            forces=np.random.randn(3, 2, 3),
            metadata={'test': 'data'}
        )
    
    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.test_dir)
    
    def test_save_trajectory_auto_format(self):
        """Test saving with auto-detected format."""
        filepath = Path(self.test_dir) / 'test.npz'
        
        save_trajectory(self.test_traj, str(filepath))
        self.assertTrue(filepath.exists())
    
    def test_load_trajectory_auto_format(self):
        """Test loading with auto-detected format."""
        filepath = Path(self.test_dir) / 'test.npz'
        
        save_trajectory(self.test_traj, str(filepath))
        loaded = load_trajectory(str(filepath))
        
        self.assertEqual(loaded.n_frames, self.test_traj.n_frames)
    
    def test_convert_format(self):
        """Test format conversion."""
        input_file = Path(self.test_dir) / 'test.xyz'
        output_file = Path(self.test_dir) / 'test.npz'
        
        # Save in XYZ format
        save_trajectory(self.test_traj, str(input_file), format='xyz')
        
        # Convert to NPZ
        convert_format(str(input_file), str(output_file))
        
        self.assertTrue(output_file.exists())
        
        # Load converted file
        loaded = load_trajectory(str(output_file))
        self.assertEqual(loaded.n_frames, self.test_traj.n_frames)


def run_tests(verbosity=2):
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestTrajectoryData))
    suite.addTests(loader.loadTestsFromTestCase(TestFormatHandlers))
    suite.addTests(loader.loadTestsFromTestCase(TestFormatConversion))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("💾 Running Data Formats Unit Tests")
    print("=" * 60)
    
    result = run_tests(verbosity=2)
    
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
