#!/usr/bin/env python3
"""
Determine Correct Descriptor for Model
=======================================
Analyzes the scaler to figure out what descriptor was used during training.

Author: Jonathan
Date: 2026-01-17
"""

import pickle
import numpy as np
from pathlib import Path
import sys


def analyze_model_descriptor(model_path):
    """Analyze what descriptor the model expects."""
    print("\n" + "="*70)
    print("DESCRIPTOR ANALYSIS")
    print("="*70)
    
    # Load model
    print(f"\nLoading: {model_path}")
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get info
    symbols = data.get('symbols', [])
    scaler_X = data.get('scaler_X')
    
    if not symbols:
        print("❌ No symbols found!")
        return None
    
    if scaler_X is None:
        print("❌ No scaler_X found!")
        return None
    
    n_atoms = len(symbols)
    n_features = scaler_X.n_features_in_
    
    print(f"\n📊 Model Information:")
    print(f"   Molecule: {' '.join(symbols)}")
    print(f"   Atoms: {n_atoms}")
    print(f"   Expected features: {n_features}")
    
    # Analyze possible descriptors
    print(f"\n🔍 Possible Descriptors:")
    
    # Coulomb matrix (upper triangle)
    coulomb_features = n_atoms * (n_atoms + 1) // 2
    print(f"\n   Coulomb Matrix (upper triangle):")
    print(f"      Formula: n*(n+1)/2 = {n_atoms}*{n_atoms+1}/2 = {coulomb_features}")
    if coulomb_features == n_features:
        print(f"      ✅ MATCH! This is likely the descriptor")
        return "coulomb_matrix"
    else:
        print(f"      ❌ Mismatch ({coulomb_features} != {n_features})")
    
    # Full Coulomb matrix
    full_coulomb = n_atoms * n_atoms
    print(f"\n   Coulomb Matrix (full):")
    print(f"      Formula: n*n = {n_atoms}*{n_atoms} = {full_coulomb}")
    if full_coulomb == n_features:
        print(f"      ✅ MATCH! This is likely the descriptor")
        return "coulomb_matrix_full"
    else:
        print(f"      ❌ Mismatch ({full_coulomb} != {n_features})")
    
    # Sorted Coulomb eigenvalues
    sorted_coulomb = n_atoms
    print(f"\n   Sorted Coulomb Eigenvalues:")
    print(f"      Formula: n = {n_atoms}")
    if sorted_coulomb == n_features:
        print(f"      ✅ MATCH! This is likely the descriptor")
        return "sorted_coulomb"
    else:
        print(f"      ❌ Mismatch ({sorted_coulomb} != {n_features})")
    
    # Extended Coulomb (upper triangle + eigenvalues)
    extended_coulomb = coulomb_features + n_atoms
    print(f"\n   Extended Coulomb (upper + eigenvalues):")
    print(f"      Formula: {coulomb_features} + {n_atoms} = {extended_coulomb}")
    if extended_coulomb == n_features:
        print(f"      ✅ MATCH! This is likely the descriptor")
        return "extended_coulomb"
    else:
        print(f"      ❌ Mismatch ({extended_coulomb} != {n_features})")
    
    # Coordinates (flattened)
    coords_features = n_atoms * 3
    print(f"\n   Coordinates (flattened):")
    print(f"      Formula: n*3 = {n_atoms}*3 = {coords_features}")
    if coords_features == n_features:
        print(f"      ✅ MATCH! This is likely the descriptor")
        return "coordinates"
    else:
        print(f"      ❌ Mismatch ({coords_features} != {n_features})")
    
    # Interatomic distances
    distances = n_atoms * (n_atoms - 1) // 2
    print(f"\n   Interatomic Distances:")
    print(f"      Formula: n*(n-1)/2 = {n_atoms}*{n_atoms-1}/2 = {distances}")
    if distances == n_features:
        print(f"      ✅ MATCH! This is likely the descriptor")
        return "distances"
    else:
        print(f"      ❌ Mismatch ({distances} != {n_features})")
    
    # Coulomb + distances
    coulomb_plus_dist = coulomb_features + distances
    print(f"\n   Coulomb + Distances:")
    print(f"      Formula: {coulomb_features} + {distances} = {coulomb_plus_dist}")
    if coulomb_plus_dist == n_features:
        print(f"      ✅ MATCH! This is likely the descriptor")
        return "coulomb_plus_distances"
    else:
        print(f"      ❌ Mismatch ({coulomb_plus_dist} != {n_features})")
    
    print(f"\n❓ Unknown descriptor type")
    print(f"   The model expects {n_features} features")
    print(f"   None of the standard descriptors match")
    
    return None


def create_descriptor_from_type(descriptor_type, n_atoms=4):
    """Create descriptor code for the given type."""
    print("\n" + "="*70)
    print("DESCRIPTOR CODE")
    print("="*70)
    
    if descriptor_type == "extended_coulomb":
        print("\n✅ Creating Extended Coulomb Descriptor")
        print("\nThis combines:")
        print("  1. Upper triangle of Coulomb matrix")
        print("  2. Eigenvalues of Coulomb matrix")
        
        code = '''
class ExtendedCoulombDescriptor:
    """Extended Coulomb matrix with eigenvalues."""
    
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
        for i in range(n_atoms):
            coulomb[i, i] = 0.5 * Z[i]**2.4
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                r_ij = np.linalg.norm(coords[i] - coords[j])
                if r_ij > 1e-6:
                    coulomb[i, j] = Z[i] * Z[j] / r_ij
                    coulomb[j, i] = coulomb[i, j]
        
        # Upper triangle
        indices = np.triu_indices(n_atoms)
        upper_tri = coulomb[indices]
        
        # Eigenvalues (sorted descending)
        eigenvalues = np.linalg.eigvalsh(coulomb)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Combine
        descriptor = np.concatenate([upper_tri, eigenvalues])
        
        return descriptor
'''
        print(code)
        return "extended_coulomb"
    
    elif descriptor_type == "coulomb_matrix":
        print("\n✅ Using Standard Coulomb Matrix")
        return "coulomb_matrix"
    
    else:
        print(f"\n❓ Unknown type: {descriptor_type}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python3 analyze_descriptor.py <model.pkl>")
        print("\nExample:")
        print("  python3 analyze_descriptor.py outputs/refined_*/mlpes_model*.pkl")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if not Path(model_path).exists():
        print(f"\n❌ File not found: {model_path}")
        sys.exit(1)
    
    # Analyze
    descriptor_type = analyze_model_descriptor(model_path)
    
    if descriptor_type:
        # Generate code
        create_descriptor_from_type(descriptor_type)
        
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print(f"\n✅ Identified descriptor: {descriptor_type}")
        print(f"\n   Update ml_pes_wrapper.py to use this descriptor")
        print(f"   Copy the code above into _create_descriptor() method")
    
    print()
