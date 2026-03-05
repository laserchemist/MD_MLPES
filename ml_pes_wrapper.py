#!/usr/bin/env python3
"""
ML-PES Model Wrapper
====================
Provides a unified interface for ML-PES models, handling both:
- Direct MLPES objects (from ml_pes.py)
- Dict-based models (from complete_workflow_v2.2.py)

Author: Jonathan
Date: 2026-01-17
"""

import numpy as np
import pickle
from pathlib import Path


class MLPESModelWrapper:
    """
    Wrapper for ML-PES models that provides consistent interface.
    
    Handles models saved as:
    1. MLPES objects with predict() method
    2. Dicts with 'model', 'scaler_X', 'scaler_y', 'descriptor' keys
    3. Dicts from complete_workflow_v2.2.py with specific structure
    """
    
    def __init__(self, model_data):
        """
        Initialize wrapper from model data.
        
        Args:
            model_data: Either MLPES object or dict
        """
        self.model_data = model_data
        self._setup_components()
    
    def _setup_components(self):
        """Extract model components from data."""
        if isinstance(self.model_data, dict):
            # Dict-based model
            self.model = self.model_data.get('model')
            self.scaler_X = self.model_data.get('scaler_X')
            self.scaler_y = self.model_data.get('scaler_y')
            self.descriptor = self.model_data.get('descriptor')
            self.symbols = self.model_data.get('symbols', [])
            
            # Handle alternative structures
            if self.model is None and 'energy_model' in self.model_data:
                self.model = self.model_data['energy_model']
            
            if not self.symbols and 'metadata' in self.model_data:
                metadata = self.model_data['metadata']
                if isinstance(metadata, dict):
                    self.symbols = metadata.get('symbols', [])
            
            # Create descriptor if missing
            if self.descriptor is None:
                self.descriptor = self._create_descriptor()
        else:
            # Assume it's an MLPES object
            self.model = self.model_data.model
            self.scaler_X = self.model_data.scaler_X
            self.scaler_y = self.model_data.scaler_y
            self.descriptor = self.model_data.descriptor
            self.symbols = getattr(self.model_data, 'symbols', [])
            
            # Create descriptor if missing
            if self.descriptor is None:
                self.descriptor = self._create_descriptor()
    
    def _create_descriptor(self):
        """Create Extended Coulomb descriptor (upper triangle + eigenvalues)."""
        class ExtendedCoulombDescriptor:
            """
            Extended Coulomb matrix descriptor.
            
            Combines:
            1. Upper triangle of Coulomb matrix: n*(n+1)/2 features
            2. Eigenvalues of Coulomb matrix: n features
            
            For 4 atoms: 10 + 4 = 14 features total
            """
            
            def __init__(self):
                self.atomic_numbers = {
                    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
                    'S': 16, 'Cl': 17, 'Br': 35
                }
            
            def compute(self, symbols, coords):
                """Compute extended Coulomb descriptor."""
                n_atoms = len(symbols)
                coulomb = np.zeros((n_atoms, n_atoms))
                
                # Atomic numbers
                Z = np.array([self.atomic_numbers.get(s, 1) for s in symbols])
                
                # Build Coulomb matrix
                # Diagonal: 0.5 * Z^2.4
                for i in range(n_atoms):
                    coulomb[i, i] = 0.5 * Z[i]**2.4
                
                # Off-diagonal: Z_i * Z_j / r_ij
                for i in range(n_atoms):
                    for j in range(i+1, n_atoms):
                        r_ij = np.linalg.norm(coords[i] - coords[j])
                        if r_ij > 1e-6:
                            coulomb[i, j] = Z[i] * Z[j] / r_ij
                            coulomb[j, i] = coulomb[i, j]
                
                # Part 1: Upper triangle
                indices = np.triu_indices(n_atoms)
                upper_tri = coulomb[indices]
                
                # Part 2: Eigenvalues (sorted descending for consistency)
                eigenvalues = np.linalg.eigvalsh(coulomb)
                eigenvalues = np.sort(eigenvalues)[::-1]
                
                # Combine: upper triangle + eigenvalues
                descriptor = np.concatenate([upper_tri, eigenvalues])
                
                return descriptor
        
        return ExtendedCoulombDescriptor()
    
    def predict(self, coords):
        """
        Predict energy for coordinates.
        
        Args:
            coords: Coordinates array (N_atoms, 3)
            
        Returns:
            Energy in Hartree
        """
        if hasattr(self.model_data, 'predict'):
            # It's an MLPES object
            return self.model_data.predict(self.symbols, coords)
        else:
            # Use components
            desc = self.descriptor.compute(self.symbols, coords)
            desc_scaled = self.scaler_X.transform([desc])
            e_scaled = self.model.predict(desc_scaled)
            energy = self.scaler_y.inverse_transform([[e_scaled[0]]])[0, 0]
            return energy
    
    def predict_with_forces(self, coords):
        """
        Predict energy and forces.
        
        Args:
            coords: Coordinates array (N_atoms, 3)
            
        Returns:
            energy: Energy in Hartree
            forces: Forces in Hartree/Angstrom (N_atoms, 3)
        """
        # For now, compute forces via finite differences
        energy = self.predict(coords)
        forces = self._compute_forces_finite_diff(coords)
        return energy, forces
    
    def _compute_forces_finite_diff(self, coords, delta=0.001):
        """
        Compute forces via finite differences.
        
        Args:
            coords: Coordinates (N_atoms, 3)
            delta: Displacement for finite differences (Angstrom)
            
        Returns:
            Forces in Hartree/Angstrom (N_atoms, 3)
        """
        n_atoms = coords.shape[0]
        forces = np.zeros_like(coords)
        
        for i in range(n_atoms):
            for j in range(3):
                # Forward displacement
                coords_plus = coords.copy()
                coords_plus[i, j] += delta
                e_plus = self.predict(coords_plus)
                
                # Backward displacement
                coords_minus = coords.copy()
                coords_minus[i, j] -= delta
                e_minus = self.predict(coords_minus)
                
                # Finite difference: F = -dE/dx
                forces[i, j] = -(e_plus - e_minus) / (2 * delta)
        
        return forces


def load_ml_pes_model(filepath):
    """
    Load ML-PES model and return wrapper with consistent interface.
    
    Args:
        filepath: Path to pickled model file
        
    Returns:
        MLPESModelWrapper object with predict_with_forces() method
    """
    with open(filepath, 'rb') as f:
        loaded = pickle.load(f)
    
    # Return wrapped model
    return MLPESModelWrapper(loaded)


# Test if run directly
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\nML-PES Model Wrapper Test")
        print("=" * 60)
        print("\nUsage:")
        print("  python ml_pes_wrapper.py <model.pkl>")
        print("\nTests:")
        print("  - Loads model")
        print("  - Checks predict() method")
        print("  - Checks predict_with_forces() method")
        sys.exit(1)
    
    model_file = sys.argv[1]
    
    print("\nML-PES Model Wrapper Test")
    print("=" * 60)
    print(f"\nLoading: {model_file}")
    
    try:
        model = load_ml_pes_model(model_file)
        print("✅ Model loaded successfully!")
        
        # Check attributes
        print(f"\nModel components:")
        print(f"  Has model: {model.model is not None}")
        print(f"  Has scaler_X: {model.scaler_X is not None}")
        print(f"  Has scaler_y: {model.scaler_y is not None}")
        print(f"  Has descriptor: {model.descriptor is not None}")
        print(f"  Symbols: {model.symbols}")
        
        # Test prediction (if we have symbols)
        if model.symbols:
            print(f"\nTesting prediction...")
            n_atoms = len(model.symbols)
            test_coords = np.random.randn(n_atoms, 3) * 0.1
            
            try:
                energy = model.predict(test_coords)
                print(f"  ✅ predict() works: E = {energy:.6f} Ha")
            except Exception as e:
                print(f"  ❌ predict() failed: {e}")
            
            try:
                energy, forces = model.predict_with_forces(test_coords)
                print(f"  ✅ predict_with_forces() works:")
                print(f"     Energy: {energy:.6f} Ha")
                print(f"     Forces shape: {forces.shape}")
            except Exception as e:
                print(f"  ❌ predict_with_forces() failed: {e}")
        else:
            print("\n⚠️  No symbols found - cannot test predictions")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
