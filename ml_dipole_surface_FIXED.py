#!/usr/bin/env python3
"""
ML Dipole Surface Model - FIXED VERSION
=========================================
Train and use ML models to predict dipole moments from molecular geometries.

This version has robust data loading that works with your actual workflow.

Author: Jonathan
Date: 2026-01-16
"""

import numpy as np
from pathlib import Path
import pickle
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class DipoleSurfaceModel:
    """
    Machine learning model for dipole moment surface.
    
    Predicts dipole moment vector [μx, μy, μz] from molecular geometry.
    """
    
    def __init__(self, name='dipole_surface', descriptor='coulomb'):
        """Initialize dipole surface model."""
        self.name = name
        self.descriptor_type = descriptor
        self.models = {}  # Separate model for each component
        self.scalers = {}
        self.is_trained = False
        self.training_stats = {}
        
    def compute_coulomb_matrix(self, symbols, coords):
        """Compute Coulomb matrix descriptor."""
        n_atoms = len(symbols)
        
        # Atomic numbers
        atomic_numbers = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16}
        Z = np.array([atomic_numbers.get(s, 0) for s in symbols])
        
        # Coulomb matrix
        CM = np.zeros((n_atoms, n_atoms))
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    CM[i, j] = 0.5 * Z[i]**2.4
                else:
                    r_ij = np.linalg.norm(coords[i] - coords[j])
                    CM[i, j] = Z[i] * Z[j] / (r_ij + 1e-10)
        
        # Flatten upper triangular part
        descriptor = CM[np.triu_indices(n_atoms)]
        
        return descriptor
    
    def compute_descriptor(self, symbols, coords):
        """Compute molecular descriptor."""
        if self.descriptor_type == 'coulomb':
            return self.compute_coulomb_matrix(symbols, coords)
        else:
            raise NotImplementedError(f"Descriptor '{self.descriptor_type}' not implemented")
    
    def prepare_training_data(self, symbols, coordinates, dipoles, test_size=0.2):
        """Prepare training and test sets."""
        print("\n" + "="*60)
        print("Preparing Training Data")
        print("="*60)
        
        # Compute descriptors
        print(f"\nComputing {self.descriptor_type} descriptors...")
        X = []
        valid_dipoles = []
        
        for i, coords in enumerate(coordinates):
            # Skip frames with failed dipole calculations (zeros)
            if np.allclose(dipoles[i], 0.0):
                continue
                
            descriptor = self.compute_descriptor(symbols, coords)
            X.append(descriptor)
            valid_dipoles.append(dipoles[i])
        
        X = np.array(X)
        dipoles_array = np.array(valid_dipoles)
        
        print(f"✓ Valid frames: {len(X)} / {len(coordinates)}")
        print(f"  Descriptor shape: {X.shape}")
        print(f"  Dipole shape: {dipoles_array.shape}")
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, dipoles_array, test_size=test_size, random_state=42
        )
        
        print(f"\n✓ Data split:")
        print(f"  Training: {len(X_train)} frames")
        print(f"  Testing: {len(X_test)} frames")
        
        # Store statistics
        self.training_stats['n_total'] = len(coordinates)
        self.training_stats['n_valid'] = len(X)
        self.training_stats['n_train'] = len(X_train)
        self.training_stats['n_test'] = len(X_test)
        self.training_stats['dipole_mean'] = dipoles_array.mean(axis=0)
        self.training_stats['dipole_std'] = dipoles_array.std(axis=0)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def train(self, symbols, coordinates, dipoles, 
              alpha=0.01, gamma=0.001, test_size=0.2):
        """Train ML model for dipole surface."""
        print("\n" + "="*60)
        print(f"Training ML Dipole Surface: {self.name}")
        print("="*60)
        
        # Prepare data
        data = self.prepare_training_data(symbols, coordinates, dipoles, test_size)
        
        # Train separate model for each component
        print("\n" + "="*60)
        print("Training Component Models")
        print("="*60)
        
        components = ['X', 'Y', 'Z']
        
        for i, comp in enumerate(components):
            print(f"\n[{comp} Component]")
            
            # Create scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(data['X_train'])
            X_test_scaled = scaler.transform(data['X_test'])
            
            # Train KRR model
            model = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)
            model.fit(X_train_scaled, data['y_train'][:, i])
            
            # Predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Metrics
            train_mae = mean_absolute_error(data['y_train'][:, i], y_train_pred)
            test_mae = mean_absolute_error(data['y_test'][:, i], y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(data['y_train'][:, i], y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(data['y_test'][:, i], y_test_pred))
            
            print(f"  Train MAE: {train_mae:.4f} Debye")
            print(f"  Test MAE:  {test_mae:.4f} Debye")
            print(f"  Train RMSE: {train_rmse:.4f} Debye")
            print(f"  Test RMSE:  {test_rmse:.4f} Debye")
            
            # Store
            self.models[comp] = model
            self.scalers[comp] = scaler
            self.training_stats[f'{comp}_train_mae'] = train_mae
            self.training_stats[f'{comp}_test_mae'] = test_mae
            self.training_stats[f'{comp}_train_rmse'] = train_rmse
            self.training_stats[f'{comp}_test_rmse'] = test_rmse
        
        self.is_trained = True
        self.symbols = symbols  # Store for later use
        
        # Overall statistics
        print("\n" + "="*60)
        print("Overall Performance")
        print("="*60)
        
        # Predict full dipole vectors
        y_train_pred_full = self.predict_batch(data['X_train'])
        y_test_pred_full = self.predict_batch(data['X_test'])
        
        # Magnitude errors
        train_mag_true = np.linalg.norm(data['y_train'], axis=1)
        train_mag_pred = np.linalg.norm(y_train_pred_full, axis=1)
        test_mag_true = np.linalg.norm(data['y_test'], axis=1)
        test_mag_pred = np.linalg.norm(y_test_pred_full, axis=1)
        
        train_mag_mae = mean_absolute_error(train_mag_true, train_mag_pred)
        test_mag_mae = mean_absolute_error(test_mag_true, test_mag_pred)
        
        print(f"\nMagnitude Errors:")
        print(f"  Train MAE: {train_mag_mae:.4f} Debye")
        print(f"  Test MAE:  {test_mag_mae:.4f} Debye")
        
        self.training_stats['magnitude_train_mae'] = train_mag_mae
        self.training_stats['magnitude_test_mae'] = test_mag_mae
        
        print(f"\n✅ Model trained successfully!")
        print(f"   Test accuracy: {test_mag_mae:.4f} Debye MAE")
    
    def predict_batch(self, X):
        """Predict dipoles for batch of descriptors."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        dipoles = np.zeros((len(X), 3))
        
        for i, comp in enumerate(['X', 'Y', 'Z']):
            X_scaled = self.scalers[comp].transform(X)
            dipoles[:, i] = self.models[comp].predict(X_scaled)
        
        return dipoles
    
    def predict(self, coords):
        """Predict dipole for single geometry."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        descriptor = self.compute_descriptor(self.symbols, coords)
        
        dipole = np.zeros(3)
        for i, comp in enumerate(['X', 'Y', 'Z']):
            X_scaled = self.scalers[comp].transform(descriptor.reshape(1, -1))
            dipole[i] = self.models[comp].predict(X_scaled)[0]
        
        return dipole
    
    def save(self, filepath):
        """Save trained model."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        model_data = {
            'name': self.name,
            'descriptor_type': self.descriptor_type,
            'models': self.models,
            'scalers': self.scalers,
            'symbols': self.symbols,
            'training_stats': self.training_stats,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✅ Model saved: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(name=model_data['name'], 
                   descriptor=model_data['descriptor_type'])
        model.models = model_data['models']
        model.scalers = model_data['scalers']
        model.symbols = model_data['symbols']
        model.training_stats = model_data['training_stats']
        model.is_trained = model_data['is_trained']
        
        print(f"✅ Model loaded: {filepath}")
        print(f"   Test MAE: {model.training_stats['magnitude_test_mae']:.4f} Debye")
        
        return model


def train_dipole_surface_from_file(training_data_path, output_path=None):
    """
    Train dipole surface model from training data file.
    
    FIXED VERSION - Handles real data structure robustly.
    """
    print("\n" + "="*60)
    print("Train ML Dipole Surface from File")
    print("="*60)
    
    print(f"\nLoading: {training_data_path}")
    
    # Load data - handle multiple formats
    data_file = np.load(str(training_data_path), allow_pickle=True)
    
    # Extract basic data
    symbols = data_file['symbols']
    if hasattr(symbols, 'tolist'):
        symbols = symbols.tolist()
    else:
        symbols = list(symbols)
    
    coordinates = data_file['coordinates']
    
    # Extract metadata
    metadata = {}
    if 'metadata' in data_file:
        metadata_item = data_file['metadata']
        if isinstance(metadata_item, np.ndarray):
            metadata = metadata_item.item()
        else:
            metadata = metadata_item
    
    print(f"✓ Basic data loaded:")
    print(f"  Frames: {len(coordinates)}")
    print(f"  Atoms: {len(symbols)} ({' '.join(symbols)})")
    
    # Extract dipoles - try multiple locations
    dipoles = None
    
    # Method 1: Direct in npz file
    if 'dipoles' in data_file:
        dipoles = data_file['dipoles']
        print(f"  Dipoles source: direct in npz")
    
    # Method 2: In metadata
    elif 'dipoles' in metadata:
        dipoles = np.array(metadata['dipoles'])
        print(f"  Dipoles source: metadata")
    
    # Method 3: Not found - give helpful error
    else:
        error_msg = (
            "\n" + "="*70 + "\n"
            "❌ ERROR: No dipoles found in training data!\n"
            "="*70 + "\n\n"
            "This file does not contain dipole moment data.\n\n"
            "SOLUTION:\n"
            "  Run the combine_and_compute_dipoles.py script:\n\n"
            f"  python combine_and_compute_dipoles.py {Path(training_data_path).parent}\n\n"
            "This will:\n"
            "  1. Combine all trajectory_*K.npz files\n"
            "  2. Compute dipoles for all frames\n"
            "  3. Create training_data.npz with dipoles\n"
            "="*70
        )
        raise ValueError(error_msg)
    
    print(f"  Dipoles: {dipoles.shape}")
    
    # Check validity
    n_valid = np.count_nonzero(np.any(dipoles != 0, axis=1))
    n_total = len(dipoles)
    print(f"  Valid dipoles: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
    
    if n_valid == 0:
        raise ValueError(
            "All dipoles are zero!\n"
            f"Run: python combine_and_compute_dipoles.py {Path(training_data_path).parent}"
        )
    
    if n_valid < 0.5 * n_total:
        print(f"\n⚠️  WARNING: Only {100*n_valid/n_total:.1f}% dipoles valid!")
        print(f"   Consider re-generating data for better accuracy.")
    
    # Get molecule name
    mol_name = 'molecule'
    if 'molecule' in metadata:
        mol_name = metadata['molecule'].get('name', 'molecule')
    
    # Create and train model
    model = DipoleSurfaceModel(
        name=f"dipole_surface_{mol_name}",
        descriptor='coulomb'
    )
    
    model.train(
        symbols=symbols,
        coordinates=coordinates,
        dipoles=dipoles,
        alpha=0.01,
        gamma=0.001,
        test_size=0.2
    )
    
    # Save if requested
    if output_path:
        model.save(output_path)
    
    return model


if __name__ == "__main__":
    print("ML Dipole Surface Model - Fixed Version")
    print("Import this module to train dipole surface models")
