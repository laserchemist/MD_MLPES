#!/usr/bin/env python3
"""
Unit Tests for ML-PES Module

Tests machine learning potential energy surface training and prediction.

Author: PSI4-MD Framework
Date: 2025
"""

import unittest
import numpy as np
import sys
from pathlib import Path
import tempfile
import shutil

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'modules'))

from ml_pes import (
    MLPESConfig, DescriptorGenerator, MLPESTrainer,
    train_pes, evaluate_model, SKLEARN_AVAILABLE, PYTORCH_AVAILABLE
)
from test_molecules import get_molecule
from data_formats import TrajectoryData


class TestDescriptorGenerator(unittest.TestCase):
    """Test descriptor generation."""
    
    def setUp(self):
        """Create test molecule."""
        self.water = get_molecule('water')
    
    def test_coulomb_matrix(self):
        """Test Coulomb matrix generation."""
        gen = DescriptorGenerator('coulomb_matrix')
        desc = gen.generate(self.water.symbols, self.water.coordinates)
        
        # Should be flattened upper triangle
        n_atoms = len(self.water.symbols)
        expected_size = n_atoms * (n_atoms + 1) // 2
        
        self.assertEqual(len(desc), expected_size)
        self.assertTrue(np.all(np.isfinite(desc)))
    
    def test_coordinates_descriptor(self):
        """Test raw coordinates descriptor."""
        gen = DescriptorGenerator('coordinates')
        desc = gen.generate(self.water.symbols, self.water.coordinates)
        
        # Should be flattened coordinates
        expected_size = len(self.water.symbols) * 3
        
        self.assertEqual(len(desc), expected_size)
        np.testing.assert_array_almost_equal(
            desc, self.water.coordinates.flatten()
        )
    
    def test_internals_descriptor(self):
        """Test internal coordinates descriptor."""
        gen = DescriptorGenerator('internals')
        desc = gen.generate(self.water.symbols, self.water.coordinates)
        
        # Should be pairwise distances
        n_atoms = len(self.water.symbols)
        expected_size = n_atoms * (n_atoms - 1) // 2
        
        self.assertEqual(len(desc), expected_size)
        self.assertTrue(np.all(desc > 0))  # All distances should be positive


class TestMLPESConfig(unittest.TestCase):
    """Test ML-PES configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = MLPESConfig()
        
        self.assertEqual(config.model_type, 'kernel_ridge')
        self.assertEqual(config.descriptor_type, 'coulomb_matrix')
        self.assertTrue(config.train_forces)
        self.assertEqual(config.validation_split, 0.2)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MLPESConfig(
            model_type='neural_network',
            descriptor_type='coordinates',
            hidden_layers=[128, 64],
            n_epochs=500
        )
        
        self.assertEqual(config.model_type, 'neural_network')
        self.assertEqual(config.n_epochs, 500)
        self.assertEqual(len(config.hidden_layers), 2)


class TestMLPESTrainer(unittest.TestCase):
    """Test ML-PES trainer."""
    
    def setUp(self):
        """Create test data."""
        self.water = get_molecule('water')
        
        # Generate mock trajectory
        n_frames = 50
        coords_list = []
        energies_list = []
        forces_list = []
        
        for i in range(n_frames):
            displacement = np.random.randn(3, 3) * 0.05
            coords = self.water.coordinates + displacement
            energy = -76.0 + 0.001 * np.sum(displacement**2)
            forces = -0.002 * displacement
            
            coords_list.append(coords)
            energies_list.append(energy)
            forces_list.append(forces)
        
        self.trajectory = TrajectoryData(
            symbols=self.water.symbols,
            coordinates=np.array(coords_list),
            energies=np.array(energies_list),
            forces=np.array(forces_list),
            metadata={'molecule': 'water', 'method': 'test'}
        )
    
    def test_data_preparation(self):
        """Test data preparation."""
        config = MLPESConfig(descriptor_type='coulomb_matrix')
        trainer = MLPESTrainer(config)
        
        descriptors, energies, forces = trainer.prepare_data(self.trajectory)
        
        self.assertEqual(len(descriptors), self.trajectory.n_frames)
        self.assertEqual(len(energies), self.trajectory.n_frames)
        self.assertTrue(descriptors.ndim == 2)
    
    @unittest.skipIf(not SKLEARN_AVAILABLE, "scikit-learn not available")
    def test_kernel_ridge_training(self):
        """Test kernel ridge regression training."""
        config = MLPESConfig(
            model_type='kernel_ridge',
            descriptor_type='coulomb_matrix'
        )
        trainer = MLPESTrainer(config)
        
        # Train
        trainer.train(self.trajectory)
        
        # Check model exists
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.scaler)
        
        # Check training history
        self.assertGreater(len(trainer.training_history['train_energy_rmse']), 0)
        self.assertGreater(len(trainer.training_history['val_energy_rmse']), 0)
    
    @unittest.skipIf(not SKLEARN_AVAILABLE, "scikit-learn not available")
    def test_prediction(self):
        """Test energy prediction."""
        config = MLPESConfig(model_type='kernel_ridge')
        trainer = MLPESTrainer(config)
        trainer.train(self.trajectory)
        
        # Predict on original structure
        energy = trainer.predict(self.water.symbols, self.water.coordinates)
        
        self.assertIsInstance(energy, float)
        self.assertTrue(np.isfinite(energy))
    
    @unittest.skipIf(not PYTORCH_AVAILABLE, "PyTorch not available")
    def test_neural_network_training(self):
        """Test neural network training."""
        config = MLPESConfig(
            model_type='neural_network',
            hidden_layers=[32, 16],
            n_epochs=50,
            early_stopping=False
        )
        trainer = MLPESTrainer(config)
        
        # Train
        trainer.train(self.trajectory)
        
        # Check model exists
        self.assertIsNotNone(trainer.model)
        
        # Check training happened
        self.assertEqual(
            len(trainer.training_history['train_energy_rmse']), 
            config.n_epochs
        )


class TestModelPersistence(unittest.TestCase):
    """Test model saving and loading."""
    
    def setUp(self):
        """Create test directory and data."""
        self.test_dir = tempfile.mkdtemp()
        
        water = get_molecule('water')
        
        # Generate simple trajectory
        coords_list = []
        energies_list = []
        forces_list = []
        
        for i in range(30):
            coords = water.coordinates + np.random.randn(3, 3) * 0.03
            energy = -76.0 + np.random.randn() * 0.001
            forces = np.random.randn(3, 3) * 0.01
            
            coords_list.append(coords)
            energies_list.append(energy)
            forces_list.append(forces)
        
        self.trajectory = TrajectoryData(
            symbols=water.symbols,
            coordinates=np.array(coords_list),
            energies=np.array(energies_list),
            forces=np.array(forces_list)
        )
    
    def tearDown(self):
        """Remove test directory."""
        shutil.rmtree(self.test_dir)
    
    @unittest.skipIf(not SKLEARN_AVAILABLE, "scikit-learn not available")
    def test_save_load(self):
        """Test saving and loading model."""
        # Train model
        config = MLPESConfig(model_type='kernel_ridge')
        trainer = MLPESTrainer(config)
        trainer.train(self.trajectory)
        
        # Save
        model_path = Path(self.test_dir) / 'test_model.pkl'
        trainer.save(str(model_path))
        self.assertTrue(model_path.exists())
        
        # Load
        loaded_trainer = MLPESTrainer.load(str(model_path))
        
        # Test prediction consistency
        water = get_molecule('water')
        pred1 = trainer.predict(water.symbols, water.coordinates)
        pred2 = loaded_trainer.predict(water.symbols, water.coordinates)
        
        np.testing.assert_almost_equal(pred1, pred2, decimal=6)


class TestEvaluation(unittest.TestCase):
    """Test model evaluation."""
    
    @unittest.skipIf(not SKLEARN_AVAILABLE, "scikit-learn not available")
    def test_evaluate_model(self):
        """Test model evaluation metrics."""
        water = get_molecule('water')
        
        # Generate trajectory
        coords_list = []
        energies_list = []
        forces_list = []
        
        for i in range(40):
            coords = water.coordinates + np.random.randn(3, 3) * 0.05
            energy = -76.0 + 0.001 * i
            forces = np.random.randn(3, 3) * 0.01
            
            coords_list.append(coords)
            energies_list.append(energy)
            forces_list.append(forces)
        
        trajectory = TrajectoryData(
            symbols=water.symbols,
            coordinates=np.array(coords_list),
            energies=np.array(energies_list),
            forces=np.array(forces_list)
        )
        
        # Train
        config = MLPESConfig(model_type='kernel_ridge')
        trainer = MLPESTrainer(config)
        trainer.train(trajectory)
        
        # Evaluate
        metrics = evaluate_model(trainer, trajectory)
        
        # Check metrics exist
        self.assertIn('rmse_kcal', metrics)
        self.assertIn('mae_kcal', metrics)
        self.assertIn('r2_score', metrics)
        
        # Check metrics are reasonable (can be negative for poor fits)
        self.assertIsInstance(metrics['r2_score'], float)
        self.assertGreater(metrics['rmse_kcal'], 0)
        self.assertGreater(metrics['mae_kcal'], 0)


def run_tests(verbosity=2):
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestDescriptorGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestMLPESConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestMLPESTrainer))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPersistence))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluation))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("🤖 Running ML-PES Unit Tests")
    print("=" * 60)
    
    if not SKLEARN_AVAILABLE:
        print("⚠️  scikit-learn not available - some tests will be skipped")
    
    if not PYTORCH_AVAILABLE:
        print("⚠️  PyTorch not available - neural network tests will be skipped")
    
    print()
    
    result = run_tests(verbosity=2)
    
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
