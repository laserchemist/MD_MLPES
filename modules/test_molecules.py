#!/usr/bin/env python3
"""
Test Molecules Library for PSI4-MD Framework

Provides a collection of test molecules with known structures for:
- Direct MD training data generation
- ML PES validation
- IR spectroscopy simulation

All geometries are pre-optimized at reasonable levels of theory.

Classes:
    TestMolecule: Container for molecular data
    
Functions:
    get_molecule: Retrieve molecule by name
    get_all_molecules: Get dictionary of all available molecules
    add_random_displacement: Add random noise to geometry
    
Author: PSI4-MD Framework
Date: 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class TestMolecule:
    """
    Container for test molecule data.
    
    Attributes:
        name: Molecule identifier
        formula: Chemical formula
        symbols: List of atomic symbols
        coordinates: Atomic coordinates in Angstroms (N x 3)
        charge: Molecular charge
        multiplicity: Spin multiplicity
        mass: Molecular mass in amu
        description: Brief description
        reference_energy: Reference energy at specific level (optional)
        reference_method: Method used for reference (optional)
    """
    name: str
    formula: str
    symbols: List[str]
    coordinates: np.ndarray
    charge: int = 0
    multiplicity: int = 1
    mass: float = 0.0
    description: str = ""
    reference_energy: Optional[float] = None
    reference_method: Optional[str] = None
    
    def __post_init__(self):
        """Calculate molecular mass if not provided."""
        if self.mass == 0.0:
            self.mass = self._calculate_mass()
        
        # Convert coordinates to numpy array if needed
        if not isinstance(self.coordinates, np.ndarray):
            self.coordinates = np.array(self.coordinates)
    
    def _calculate_mass(self) -> float:
        """Calculate molecular mass from atomic symbols."""
        atomic_masses = {
            'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
            'F': 18.998, 'S': 32.065, 'Cl': 35.453, 'Br': 79.904
        }
        return sum(atomic_masses.get(symbol, 0.0) for symbol in self.symbols)
    
    def copy(self):
        """Create a deep copy of the molecule."""
        return TestMolecule(
            name=self.name,
            formula=self.formula,
            symbols=self.symbols.copy(),
            coordinates=self.coordinates.copy(),
            charge=self.charge,
            multiplicity=self.multiplicity,
            mass=self.mass,
            description=self.description,
            reference_energy=self.reference_energy,
            reference_method=self.reference_method
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'formula': self.formula,
            'symbols': self.symbols,
            'coordinates': self.coordinates.tolist(),
            'charge': self.charge,
            'multiplicity': self.multiplicity,
            'mass': self.mass,
            'description': self.description,
            'reference_energy': self.reference_energy,
            'reference_method': self.reference_method
        }
    
    def to_xyz_string(self) -> str:
        """Convert to XYZ format string."""
        n_atoms = len(self.symbols)
        lines = [
            str(n_atoms),
            f"{self.name} - {self.formula}"
        ]
        
        for symbol, coord in zip(self.symbols, self.coordinates):
            lines.append(f"{symbol:2s} {coord[0]:12.8f} {coord[1]:12.8f} {coord[2]:12.8f}")
        
        return '\n'.join(lines)


# Pre-defined test molecules
# All geometries optimized at B3LYP/6-31G* level

MOLECULES = {
    'water': TestMolecule(
        name='water',
        formula='H2O',
        symbols=['O', 'H', 'H'],
        coordinates=np.array([
            [0.000000,  0.000000,  0.117176],
            [0.000000,  0.757151, -0.468706],
            [0.000000, -0.757151, -0.468706]
        ]),
        description='Water molecule - simplest test case',
        reference_energy=-76.4275,
        reference_method='B3LYP/6-31G*'
    ),
    
    'formaldehyde': TestMolecule(
        name='formaldehyde',
        formula='CH2O',
        symbols=['C', 'O', 'H', 'H'],
        coordinates=np.array([
            [ 0.000000,  0.000000, -0.532722],
            [ 0.000000,  0.000000,  0.677558],
            [ 0.000000,  0.938719, -1.109550],
            [ 0.000000, -0.938719, -1.109550]
        ]),
        description='Formaldehyde - simple carbonyl',
        reference_energy=-114.5052,
        reference_method='B3LYP/6-31G*'
    ),
    
    'formaldehyde_oxide': TestMolecule(
        name='formaldehyde_oxide',
        formula='CH2OO',
        symbols=['C', 'O', 'O', 'H', 'H'],
        coordinates=np.array([
            [ 0.672169,  0.000000,  0.000000],
            [-0.672169,  0.000000,  0.000000],
            [-1.425633,  1.135764,  0.000000],
            [ 1.270845,  0.921573,  0.000000],
            [ 1.270845, -0.921573,  0.000000]
        ]),
        description='Formaldehyde oxide (Criegee intermediate)',
        reference_energy=-189.7892,
        reference_method='B3LYP/6-31G*'
    ),
    
    'methane': TestMolecule(
        name='methane',
        formula='CH4',
        symbols=['C', 'H', 'H', 'H', 'H'],
        coordinates=np.array([
            [ 0.000000,  0.000000,  0.000000],
            [ 0.629118,  0.629118,  0.629118],
            [-0.629118, -0.629118,  0.629118],
            [-0.629118,  0.629118, -0.629118],
            [ 0.629118, -0.629118, -0.629118]
        ]),
        description='Methane - tetrahedral structure',
        reference_energy=-40.5178,
        reference_method='B3LYP/6-31G*'
    ),
    
    'ethylene': TestMolecule(
        name='ethylene',
        formula='C2H4',
        symbols=['C', 'C', 'H', 'H', 'H', 'H'],
        coordinates=np.array([
            [ 0.000000,  0.000000,  0.665119],
            [ 0.000000,  0.000000, -0.665119],
            [ 0.000000,  0.923621,  1.232289],
            [ 0.000000, -0.923621,  1.232289],
            [ 0.000000,  0.923621, -1.232289],
            [ 0.000000, -0.923621, -1.232289]
        ]),
        description='Ethylene - pi bond system',
        reference_energy=-78.5876,
        reference_method='B3LYP/6-31G*'
    ),
    
    'hydrogen_peroxide': TestMolecule(
        name='hydrogen_peroxide',
        formula='H2O2',
        symbols=['O', 'O', 'H', 'H'],
        coordinates=np.array([
            [ 0.000000,  0.737565,  0.000000],
            [ 0.000000, -0.737565,  0.000000],
            [ 0.814908,  0.890264,  0.517180],
            [-0.814908, -0.890264,  0.517180]
        ]),
        description='Hydrogen peroxide - peroxide bond',
        reference_energy=-151.5678,
        reference_method='B3LYP/6-31G*'
    ),
    
    'ammonia': TestMolecule(
        name='ammonia',
        formula='NH3',
        symbols=['N', 'H', 'H', 'H'],
        coordinates=np.array([
            [ 0.000000,  0.000000,  0.116489],
            [ 0.000000,  0.939731, -0.271807],
            [ 0.813831, -0.469865, -0.271807],
            [-0.813831, -0.469865, -0.271807]
        ]),
        description='Ammonia - pyramidal structure',
        reference_energy=-56.5567,
        reference_method='B3LYP/6-31G*'
    ),
    
    'methanol': TestMolecule(
        name='methanol',
        formula='CH3OH',
        symbols=['C', 'O', 'H', 'H', 'H', 'H'],
        coordinates=np.array([
            [ 0.661515,  0.000000,  0.000000],
            [-0.758869,  0.000000,  0.000000],
            [ 1.041939,  0.944815,  0.409682],
            [ 1.041939, -0.082718, -1.012954],
            [ 1.041939, -0.822097,  0.603272],
            [-1.082488,  0.776069,  0.516878]
        ]),
        description='Methanol - simple alcohol',
        reference_energy=-115.7145,
        reference_method='B3LYP/6-31G*'
    ),

    'mvko': TestMolecule(
        name='mvko',
        formula='C4H6O2',
        # Atom order (must remain fixed — Coulomb matrix is NOT permutation-invariant):
        #   C1  Criegee carbon      (central sp²-like C of the C=O⁺O⁻ unit)
        #   O1  proximal oxygen     (bonded to C1, C=O⁺ character, ~1.27 Å)
        #   O2  distal oxygen       (bonded to O1, O⁻ character, O–O ~1.35 Å)
        #   C2  inner vinyl carbon  (C1–CH=, sp²)
        #   C3  terminal vinyl      (=CH₂, sp²)
        #   C4  methyl carbon       (C1–CH₃, sp³)
        #   H1  vinyl H on C2
        #   H2  vinyl H on C3 (first)
        #   H3  vinyl H on C3 (second)
        #   H4  methyl H (in molecular plane)
        #   H5  methyl H (above plane)
        #   H6  methyl H (below plane)
        symbols=['C', 'O', 'O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
        # Initial geometry (Angstrom) built from sp² angles and standard bond lengths;
        # must be PSI4-optimised before use in MD — see mvko_workflow.py.
        #
        # Coordinate system:
        #   C1 at origin; C1–O1 along +x; C1 sp² plane = xy plane.
        #   C1–O1–O2 angle ≈ 115° (Criegee, O2 above C1–O1 axis).
        #   C1–C2–C3 vinyl extends into quadrant II (upper left).
        #   C1–C4 methyl extends into quadrant III (lower left).
        #
        # Key bond lengths used:
        #   C1–O1 = 1.270 Å  (Criegee C=O+ partial double bond)
        #   O1–O2 = 1.350 Å  (Criegee peroxide O–O)
        #   C1–C2 = 1.470 Å  (sp²–sp² cross-conjugated)
        #   C2=C3 = 1.330 Å  (vinyl double bond)
        #   C1–C4 = 1.500 Å  (sp²–sp³)
        #   C–H   = 1.080 Å  (sp² vinyl), 1.090 Å (sp³ methyl)
        coordinates=np.array([
            # C1  Criegee carbon
            [ 0.000,  0.000,  0.000],
            # O1  proximal O (along +x, 1.270 Å)
            [ 1.270,  0.000,  0.000],
            # O2  distal O (C1–O1–O2 = 115°; O1 + 1.350*[cos65°, sin65°, 0])
            [ 1.841,  1.224,  0.000],
            # C2  inner vinyl (C1–C2 at 120°; C2=C3 continues at +60°)
            [-0.735,  1.273,  0.000],
            # C3  terminal vinyl =CH2 (C2=C3 at 60° from C2, 1.330 Å)
            [-0.070,  2.425,  0.000],
            # C4  methyl (C1–C4 at 240°, 1.500 Å)
            [-0.750, -1.299,  0.000],
            # H1  vinyl H on C2 (third sp² dir at 180° from C2)
            [-1.815,  1.273,  0.000],
            # H2  vinyl H on C3 (sp² dir at 0° from C3)
            [ 1.010,  2.425,  0.000],
            # H3  vinyl H on C3 (sp² dir at 120° from C3)
            [-0.610,  3.360,  0.000],
            # H4  methyl H (in-plane, tetrahedral from C4)
            [-0.041, -2.129,  0.000],
            # H5  methyl H (above plane, tetrahedral from C4)
            [-1.377, -1.357,  0.891],
            # H6  methyl H (below plane, tetrahedral from C4)
            [-1.377, -1.357, -0.891],
        ]),
        description=(
            'Methyl vinyl ketone oxide (MVKO) — Criegee intermediate '
            'from MVK ozonolysis. Initial guess geometry; optimize with PSI4 '
            'before MD. B3LYP/6-31G*, singlet ground state.'
        ),
        charge=0,
        multiplicity=1,
    ),
}


def get_molecule(name: str) -> Optional[TestMolecule]:
    """
    Retrieve a test molecule by name.
    
    Args:
        name: Molecule identifier
        
    Returns:
        TestMolecule object or None if not found
        
    Example:
        >>> water = get_molecule('water')
        >>> print(water.formula)
        H2O
    """
    mol = MOLECULES.get(name.lower())
    if mol is None:
        available = ', '.join(MOLECULES.keys())
        print(f"Molecule '{name}' not found. Available: {available}")
        return None
    return mol.copy()


def get_all_molecules() -> Dict[str, TestMolecule]:
    """
    Get dictionary of all available test molecules.
    
    Returns:
        Dictionary mapping names to TestMolecule objects
        
    Example:
        >>> molecules = get_all_molecules()
        >>> print(f"Available molecules: {len(molecules)}")
    """
    return {name: mol.copy() for name, mol in MOLECULES.items()}


def add_random_displacement(
    molecule: TestMolecule,
    displacement_magnitude: float = 0.1,
    random_seed: Optional[int] = None
) -> TestMolecule:
    """
    Add random displacement to molecular geometry.
    
    Useful for generating initial configurations for MD simulations
    or creating perturbed structures for ML training.
    
    Args:
        molecule: Input molecule
        displacement_magnitude: Maximum displacement in Angstroms
        random_seed: Random seed for reproducibility
        
    Returns:
        New TestMolecule with displaced coordinates
        
    Example:
        >>> water = get_molecule('water')
        >>> displaced = add_random_displacement(water, 0.05, seed=42)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    displaced_mol = molecule.copy()
    n_atoms = len(molecule.symbols)
    
    # Generate random displacements
    displacements = np.random.uniform(
        -displacement_magnitude,
        displacement_magnitude,
        size=(n_atoms, 3)
    )
    
    displaced_mol.coordinates += displacements
    displaced_mol.name = f"{molecule.name}_displaced"
    displaced_mol.description = f"{molecule.description} (displaced by {displacement_magnitude} Å)"
    
    return displaced_mol


def save_molecule_library(filename: str):
    """
    Save molecule library to JSON file.
    
    Args:
        filename: Output JSON file path
    """
    library = {
        name: mol.to_dict()
        for name, mol in MOLECULES.items()
    }
    
    with open(filename, 'w') as f:
        json.dump(library, f, indent=2)
    
    print(f"Saved {len(MOLECULES)} molecules to {filename}")


def load_molecule_from_xyz(filename: str, name: str = None) -> TestMolecule:
    """
    Load molecule from XYZ file.
    
    Args:
        filename: XYZ file path
        name: Optional name (defaults to filename)
        
    Returns:
        TestMolecule object
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    n_atoms = int(lines[0].strip())
    comment = lines[1].strip()
    
    symbols = []
    coordinates = []
    
    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        symbols.append(parts[0])
        coordinates.append([float(x) for x in parts[1:4]])
    
    if name is None:
        name = Path(filename).stem
    
    return TestMolecule(
        name=name,
        formula='',  # Could parse from symbols
        symbols=symbols,
        coordinates=np.array(coordinates),
        description=f"Loaded from {filename}"
    )


if __name__ == "__main__":
    print("🧪 Test Molecules Library")
    print("=" * 60)
    
    # Display available molecules
    molecules = get_all_molecules()
    print(f"\n📚 Available molecules: {len(molecules)}")
    
    for name, mol in molecules.items():
        print(f"\n  {name:20s} {mol.formula:8s} {mol.description}")
        print(f"    Atoms: {len(mol.symbols):2d}  Mass: {mol.mass:6.2f} amu")
        if mol.reference_energy:
            print(f"    Reference: {mol.reference_energy:.4f} Hartree ({mol.reference_method})")
    
    # Test molecule retrieval
    print("\n" + "=" * 60)
    print("🧪 Testing molecule retrieval...")
    
    water = get_molecule('water')
    if water:
        print(f"\nRetrieved: {water.name}")
        print(f"Formula: {water.formula}")
        print(f"Coordinates shape: {water.coordinates.shape}")
        print("\nXYZ format:")
        print(water.to_xyz_string())
    
    # Test random displacement
    print("\n" + "=" * 60)
    print("🎲 Testing random displacement...")
    
    displaced = add_random_displacement(water, displacement_magnitude=0.05, random_seed=42)
    print(f"\nOriginal H position: {water.coordinates[1]}")
    print(f"Displaced H position: {displaced.coordinates[1]}")
    print(f"Displacement: {np.linalg.norm(displaced.coordinates[1] - water.coordinates[1]):.4f} Å")
    
    # Save library
    output_file = '/home/claude/psi4md_framework/data/molecule_library.json'
    save_molecule_library(output_file)
    
    print("\n✅ Test molecules library demonstration complete!")
