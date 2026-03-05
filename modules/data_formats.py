#!/usr/bin/env python3
"""
Data Format Handlers for PSI4-MD Framework

Handles reading and writing trajectory data in multiple standard formats
for interchangeability between different MD and ML software packages.

Supported Formats:
    - XYZ: Simple coordinate format
    - Extended XYZ: XYZ with energy/forces in comment line
    - NPZ: NumPy compressed format (fast, compact)
    - HDF5: Hierarchical data format (best for large datasets)
    - JSON: Human-readable format (good for metadata)
    - ASE DB: ASE database format
    
Classes:
    TrajectoryData: Container for MD trajectory data
    FormatHandler: Base class for format handlers
    
Functions:
    save_trajectory: Save trajectory in specified format
    load_trajectory: Load trajectory from file
    convert_format: Convert between formats
    
Author: PSI4-MD Framework
Date: 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings


@dataclass
class TrajectoryData:
    """
    Container for molecular dynamics trajectory data.
    
    Attributes:
        symbols: Atomic symbols (N_atoms,)
        coordinates: Atomic coordinates in Angstroms (N_frames, N_atoms, 3)
        energies: Energies in Hartree (N_frames,)
        forces: Forces in Hartree/Angstrom (N_frames, N_atoms, 3)
        dipoles: Dipole moments in Debye (N_frames, 3) - optional
        charges: Atomic charges (N_frames, N_atoms) - optional
        velocities: Velocities in Angstrom/fs (N_frames, N_atoms, 3) - optional
        times: Time points in fs (N_frames,) - optional
        metadata: Additional information
    """
    symbols: List[str]
    coordinates: np.ndarray  # (N_frames, N_atoms, 3)
    energies: np.ndarray     # (N_frames,)
    forces: np.ndarray       # (N_frames, N_atoms, 3)
    dipoles: Optional[np.ndarray] = None      # (N_frames, 3)
    charges: Optional[np.ndarray] = None      # (N_frames, N_atoms)
    velocities: Optional[np.ndarray] = None   # (N_frames, N_atoms, 3)
    times: Optional[np.ndarray] = None        # (N_frames,)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data shapes."""
        n_frames, n_atoms = self.coordinates.shape[0], self.coordinates.shape[1]
        
        # Validate required arrays
        assert self.coordinates.shape == (n_frames, n_atoms, 3), \
            f"Coordinates shape {self.coordinates.shape} != ({n_frames}, {n_atoms}, 3)"
        assert self.energies.shape == (n_frames,), \
            f"Energies shape {self.energies.shape} != ({n_frames},)"
        assert self.forces.shape == (n_frames, n_atoms, 3), \
            f"Forces shape {self.forces.shape} != ({n_frames}, {n_atoms}, 3)"
        assert len(self.symbols) == n_atoms, \
            f"Number of symbols {len(self.symbols)} != {n_atoms}"
        
        # Validate optional arrays
        if self.dipoles is not None:
            assert self.dipoles.shape == (n_frames, 3), \
                f"Dipoles shape {self.dipoles.shape} != ({n_frames}, 3)"
        
        if self.charges is not None:
            assert self.charges.shape == (n_frames, n_atoms), \
                f"Charges shape {self.charges.shape} != ({n_frames}, {n_atoms})"
        
        if self.velocities is not None:
            assert self.velocities.shape == (n_frames, n_atoms, 3), \
                f"Velocities shape {self.velocities.shape} != ({n_frames}, {n_atoms}, 3)"
        
        if self.times is not None:
            assert self.times.shape == (n_frames,), \
                f"Times shape {self.times.shape} != ({n_frames},)"
    
    @property
    def n_frames(self) -> int:
        """Number of trajectory frames."""
        return self.coordinates.shape[0]
    
    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return self.coordinates.shape[1]
    
    def get_frame(self, index: int) -> Dict[str, Any]:
        """
        Get single frame data.
        
        Args:
            index: Frame index
            
        Returns:
            Dictionary with frame data
        """
        frame_data = {
            'symbols': self.symbols,
            'coordinates': self.coordinates[index],
            'energy': self.energies[index],
            'forces': self.forces[index]
        }
        
        if self.dipoles is not None:
            frame_data['dipole'] = self.dipoles[index]
        if self.charges is not None:
            frame_data['charges'] = self.charges[index]
        if self.velocities is not None:
            frame_data['velocities'] = self.velocities[index]
        if self.times is not None:
            frame_data['time'] = self.times[index]
        
        return frame_data
    
    def slice(self, start: int, end: int, step: int = 1) -> 'TrajectoryData':
        """
        Get a slice of the trajectory.
        
        Args:
            start: Start frame index
            end: End frame index
            step: Step size
            
        Returns:
            New TrajectoryData with sliced data
        """
        sliced_data = TrajectoryData(
            symbols=self.symbols.copy(),
            coordinates=self.coordinates[start:end:step],
            energies=self.energies[start:end:step],
            forces=self.forces[start:end:step],
            metadata=self.metadata.copy()
        )
        
        if self.dipoles is not None:
            sliced_data.dipoles = self.dipoles[start:end:step]
        if self.charges is not None:
            sliced_data.charges = self.charges[start:end:step]
        if self.velocities is not None:
            sliced_data.velocities = self.velocities[start:end:step]
        if self.times is not None:
            sliced_data.times = self.times[start:end:step]
        
        return sliced_data


class FormatHandler(ABC):
    """Base class for trajectory format handlers."""
    
    @abstractmethod
    def save(self, trajectory: TrajectoryData, filename: str):
        """Save trajectory to file."""
        pass
    
    @abstractmethod
    def load(self, filename: str) -> TrajectoryData:
        """Load trajectory from file."""
        pass


class XYZHandler(FormatHandler):
    """Handler for XYZ format trajectories."""
    
    def save(self, trajectory: TrajectoryData, filename: str):
        """Save trajectory in XYZ format."""
        with open(filename, 'w') as f:
            for i in range(trajectory.n_frames):
                frame = trajectory.get_frame(i)
                
                # Write header
                f.write(f"{trajectory.n_atoms}\n")
                
                # Comment line with energy
                comment_parts = [f"Energy={frame['energy']:.8f}"]
                if 'time' in frame:
                    comment_parts.append(f"Time={frame['time']:.4f}")
                f.write(" ".join(comment_parts) + "\n")
                
                # Write coordinates
                for j, symbol in enumerate(trajectory.symbols):
                    coord = frame['coordinates'][j]
                    f.write(f"{symbol:2s} {coord[0]:15.8f} {coord[1]:15.8f} {coord[2]:15.8f}\n")
    
    def load(self, filename: str) -> TrajectoryData:
        """Load trajectory from XYZ format."""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        frames = []
        i = 0
        
        while i < len(lines):
            n_atoms = int(lines[i].strip())
            comment = lines[i + 1].strip()
            
            # Parse energy from comment
            energy = None
            time = None
            for part in comment.split():
                if part.startswith('Energy='):
                    energy = float(part.split('=')[1])
                elif part.startswith('Time='):
                    time = float(part.split('=')[1])
            
            # Parse coordinates
            symbols = []
            coordinates = []
            for j in range(i + 2, i + 2 + n_atoms):
                parts = lines[j].split()
                symbols.append(parts[0])
                coordinates.append([float(x) for x in parts[1:4]])
            
            frames.append({
                'symbols': symbols,
                'coordinates': np.array(coordinates),
                'energy': energy
            })
            
            i += 2 + n_atoms
        
        # Compile trajectory
        symbols = frames[0]['symbols']
        coordinates = np.array([f['coordinates'] for f in frames])
        energies = np.array([f['energy'] for f in frames])
        forces = np.zeros_like(coordinates)  # XYZ doesn't store forces
        
        return TrajectoryData(
            symbols=symbols,
            coordinates=coordinates,
            energies=energies,
            forces=forces,
            metadata={'format': 'xyz', 'filename': filename}
        )


class ExtendedXYZHandler(FormatHandler):
    """Handler for extended XYZ format with forces and properties."""
    
    def save(self, trajectory: TrajectoryData, filename: str):
        """Save trajectory in extended XYZ format."""
        with open(filename, 'w') as f:
            for i in range(trajectory.n_frames):
                frame = trajectory.get_frame(i)
                
                # Write header
                f.write(f"{trajectory.n_atoms}\n")
                
                # Extended comment line
                comment_parts = [
                    f"Energy={frame['energy']:.8f}",
                    'Properties=species:S:1:pos:R:3:forces:R:3'
                ]
                if 'time' in frame:
                    comment_parts.append(f"Time={frame['time']:.4f}")
                if 'dipole' in frame:
                    dipole = frame['dipole']
                    comment_parts.append(f"Dipole={dipole[0]:.6f},{dipole[1]:.6f},{dipole[2]:.6f}")
                
                f.write(" ".join(comment_parts) + "\n")
                
                # Write coordinates and forces
                for j, symbol in enumerate(trajectory.symbols):
                    coord = frame['coordinates'][j]
                    force = frame['forces'][j]
                    f.write(f"{symbol:2s} "
                           f"{coord[0]:15.8f} {coord[1]:15.8f} {coord[2]:15.8f} "
                           f"{force[0]:15.8f} {force[1]:15.8f} {force[2]:15.8f}\n")
    
    def load(self, filename: str) -> TrajectoryData:
        """Load trajectory from extended XYZ format."""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        frames = []
        i = 0
        
        while i < len(lines):
            n_atoms = int(lines[i].strip())
            comment = lines[i + 1].strip()
            
            # Parse properties from comment
            energy = None
            time = None
            dipole = None
            
            for part in comment.split():
                if part.startswith('Energy='):
                    energy = float(part.split('=')[1])
                elif part.startswith('Time='):
                    time = float(part.split('=')[1])
                elif part.startswith('Dipole='):
                    dipole_str = part.split('=')[1]
                    dipole = [float(x) for x in dipole_str.split(',')]
            
            # Parse coordinates and forces
            symbols = []
            coordinates = []
            forces = []
            
            for j in range(i + 2, i + 2 + n_atoms):
                parts = lines[j].split()
                symbols.append(parts[0])
                coordinates.append([float(x) for x in parts[1:4]])
                forces.append([float(x) for x in parts[4:7]])
            
            frame_data = {
                'symbols': symbols,
                'coordinates': np.array(coordinates),
                'forces': np.array(forces),
                'energy': energy
            }
            
            if time is not None:
                frame_data['time'] = time
            if dipole is not None:
                frame_data['dipole'] = np.array(dipole)
            
            frames.append(frame_data)
            i += 2 + n_atoms
        
        # Compile trajectory
        symbols = frames[0]['symbols']
        coordinates = np.array([f['coordinates'] for f in frames])
        energies = np.array([f['energy'] for f in frames])
        forces = np.array([f['forces'] for f in frames])
        
        trajectory_data = TrajectoryData(
            symbols=symbols,
            coordinates=coordinates,
            energies=energies,
            forces=forces,
            metadata={'format': 'extxyz', 'filename': filename}
        )
        
        # Add optional data
        if 'time' in frames[0]:
            trajectory_data.times = np.array([f['time'] for f in frames])
        if 'dipole' in frames[0]:
            trajectory_data.dipoles = np.array([f['dipole'] for f in frames])
        
        return trajectory_data


class NPZHandler(FormatHandler):
    """Handler for NumPy NPZ format (fast and compact)."""
    
    def save(self, trajectory: TrajectoryData, filename: str):
        """Save trajectory in NPZ format."""
        save_dict = {
            'symbols': np.array(trajectory.symbols, dtype='U2'),
            'coordinates': trajectory.coordinates,
            'energies': trajectory.energies,
            'forces': trajectory.forces,
        }
        
        # Add optional arrays
        if trajectory.dipoles is not None:
            save_dict['dipoles'] = trajectory.dipoles
        if trajectory.charges is not None:
            save_dict['charges'] = trajectory.charges
        if trajectory.velocities is not None:
            save_dict['velocities'] = trajectory.velocities
        if trajectory.times is not None:
            save_dict['times'] = trajectory.times
        
        # Add metadata as JSON string
        save_dict['metadata'] = json.dumps(trajectory.metadata)
        
        np.savez_compressed(filename, **save_dict)
    
    def load(self, filename: str) -> TrajectoryData:
        """Load trajectory from NPZ format."""
        data = np.load(filename, allow_pickle=True)
        
        # Load metadata
        metadata = json.loads(str(data['metadata']))
        
        trajectory = TrajectoryData(
            symbols=data['symbols'].tolist(),
            coordinates=data['coordinates'],
            energies=data['energies'],
            forces=data['forces'],
            metadata=metadata
        )
        
        # Load optional arrays
        if 'dipoles' in data:
            trajectory.dipoles = data['dipoles']
        if 'charges' in data:
            trajectory.charges = data['charges']
        if 'velocities' in data:
            trajectory.velocities = data['velocities']
        if 'times' in data:
            trajectory.times = data['times']
        
        return trajectory


class HDF5Handler(FormatHandler):
    """Handler for HDF5 format (best for large datasets)."""
    
    def __init__(self):
        """Initialize HDF5 handler."""
        try:
            import h5py
            self.h5py = h5py
            self.available = True
        except ImportError:
            self.available = False
            warnings.warn("h5py not available. HDF5 format will not work.")
    
    def save(self, trajectory: TrajectoryData, filename: str):
        """Save trajectory in HDF5 format."""
        if not self.available:
            raise ImportError("h5py is required for HDF5 format")
        
        with self.h5py.File(filename, 'w') as f:
            # Create groups
            structure = f.create_group('structure')
            dynamics = f.create_group('dynamics')
            optional = f.create_group('optional')
            
            # Save structure data
            structure.create_dataset('symbols', data=np.array(trajectory.symbols, dtype='S2'))
            
            # Save dynamics data
            dynamics.create_dataset('coordinates', data=trajectory.coordinates,
                                  compression='gzip', compression_opts=9)
            dynamics.create_dataset('energies', data=trajectory.energies)
            dynamics.create_dataset('forces', data=trajectory.forces,
                                  compression='gzip', compression_opts=9)
            
            # Save optional data
            if trajectory.dipoles is not None:
                optional.create_dataset('dipoles', data=trajectory.dipoles)
            if trajectory.charges is not None:
                optional.create_dataset('charges', data=trajectory.charges)
            if trajectory.velocities is not None:
                optional.create_dataset('velocities', data=trajectory.velocities,
                                      compression='gzip', compression_opts=9)
            if trajectory.times is not None:
                optional.create_dataset('times', data=trajectory.times)
            
            # Save metadata
            meta_group = f.create_group('metadata')
            for key, value in trajectory.metadata.items():
                if isinstance(value, (str, int, float)):
                    meta_group.attrs[key] = value
    
    def load(self, filename: str) -> TrajectoryData:
        """Load trajectory from HDF5 format."""
        if not self.available:
            raise ImportError("h5py is required for HDF5 format")
        
        with self.h5py.File(filename, 'r') as f:
            # Load structure
            symbols = [s.decode() for s in f['structure/symbols'][:]]
            
            # Load dynamics
            coordinates = f['dynamics/coordinates'][:]
            energies = f['dynamics/energies'][:]
            forces = f['dynamics/forces'][:]
            
            # Load metadata
            metadata = dict(f['metadata'].attrs)
            
            trajectory = TrajectoryData(
                symbols=symbols,
                coordinates=coordinates,
                energies=energies,
                forces=forces,
                metadata=metadata
            )
            
            # Load optional data
            if 'dipoles' in f['optional']:
                trajectory.dipoles = f['optional/dipoles'][:]
            if 'charges' in f['optional']:
                trajectory.charges = f['optional/charges'][:]
            if 'velocities' in f['optional']:
                trajectory.velocities = f['optional/velocities'][:]
            if 'times' in f['optional']:
                trajectory.times = f['optional/times'][:]
            
            return trajectory


# Format registry
FORMAT_HANDLERS = {
    'xyz': XYZHandler(),
    'extxyz': ExtendedXYZHandler(),
    'npz': NPZHandler(),
    'hdf5': HDF5Handler(),
    'h5': HDF5Handler(),
}


def save_trajectory(trajectory: TrajectoryData, filename: str, format: str = None):
    """
    Save trajectory in specified format.
    
    Args:
        trajectory: Trajectory data to save
        filename: Output file path
        format: Format name (auto-detected from extension if None)
        
    Example:
        >>> save_trajectory(traj, 'trajectory.extxyz')
        >>> save_trajectory(traj, 'trajectory.npz', format='npz')
    """
    if format is None:
        # Detect format from extension
        ext = Path(filename).suffix[1:].lower()
        format = ext
    
    handler = FORMAT_HANDLERS.get(format)
    if handler is None:
        raise ValueError(f"Unknown format: {format}. Available: {list(FORMAT_HANDLERS.keys())}")
    
    handler.save(trajectory, filename)
    print(f"Saved trajectory to {filename} ({format} format)")


def load_trajectory(filename: str, format: str = None) -> TrajectoryData:
    """
    Load trajectory from file.
    
    Args:
        filename: Input file path
        format: Format name (auto-detected from extension if None)
        
    Returns:
        TrajectoryData object
        
    Example:
        >>> traj = load_trajectory('trajectory.extxyz')
        >>> print(f"Loaded {traj.n_frames} frames")
    """
    if format is None:
        # Detect format from extension
        ext = Path(filename).suffix[1:].lower()
        format = ext
    
    handler = FORMAT_HANDLERS.get(format)
    if handler is None:
        raise ValueError(f"Unknown format: {format}. Available: {list(FORMAT_HANDLERS.keys())}")
    
    trajectory = handler.load(filename)
    print(f"Loaded trajectory from {filename} ({format} format)")
    print(f"  Frames: {trajectory.n_frames}, Atoms: {trajectory.n_atoms}")
    
    return trajectory


def convert_format(input_file: str, output_file: str,
                  input_format: str = None, output_format: str = None):
    """
    Convert trajectory between formats.
    
    Args:
        input_file: Input file path
        output_file: Output file path
        input_format: Input format (auto-detected if None)
        output_format: Output format (auto-detected if None)
        
    Example:
        >>> convert_format('traj.xyz', 'traj.npz')
    """
    trajectory = load_trajectory(input_file, format=input_format)
    save_trajectory(trajectory, output_file, format=output_format)
    print(f"Converted {input_file} → {output_file}")


if __name__ == "__main__":
    print("💾 Data Format Handlers Demonstration")
    print("=" * 60)
    
    # Create test trajectory data
    n_frames = 10
    n_atoms = 3
    
    test_trajectory = TrajectoryData(
        symbols=['O', 'H', 'H'],
        coordinates=np.random.randn(n_frames, n_atoms, 3) * 0.1 + np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
        energies=np.random.randn(n_frames) * 0.01 - 76.0,
        forces=np.random.randn(n_frames, n_atoms, 3) * 0.01,
        dipoles=np.random.randn(n_frames, 3) * 0.1 + np.array([0.5, 0.5, 0]),
        times=np.arange(n_frames) * 0.5,
        metadata={'molecule': 'water', 'method': 'B3LYP', 'basis': '6-31G*'}
    )
    
    print(f"\n📊 Created test trajectory:")
    print(f"  Frames: {test_trajectory.n_frames}")
    print(f"  Atoms: {test_trajectory.n_atoms}")
    print(f"  Energy range: {test_trajectory.energies.min():.4f} to {test_trajectory.energies.max():.4f} Hartree")
    
    # Test each format
    output_dir = Path('/home/claude/psi4md_framework/data')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    formats_to_test = ['xyz', 'extxyz', 'npz']
    
    for fmt in formats_to_test:
        print(f"\n{'=' * 60}")
        print(f"Testing {fmt.upper()} format...")
        
        output_file = output_dir / f"test_trajectory.{fmt}"
        
        # Save
        save_trajectory(test_trajectory, str(output_file), format=fmt)
        
        # Load
        loaded_traj = load_trajectory(str(output_file), format=fmt)
        
        # Verify
        assert loaded_traj.n_frames == test_trajectory.n_frames
        assert loaded_traj.n_atoms == test_trajectory.n_atoms
        assert np.allclose(loaded_traj.coordinates, test_trajectory.coordinates)
        
        print(f"✅ {fmt.upper()} format verified!")
        
        # Check file size
        file_size = output_file.stat().st_size / 1024  # KB
        print(f"  File size: {file_size:.2f} KB")
    
    # Test format conversion
    print(f"\n{'=' * 60}")
    print("Testing format conversion...")
    convert_format(
        str(output_dir / 'test_trajectory.xyz'),
        str(output_dir / 'test_trajectory_converted.npz')
    )
    
    print("\n✅ Data format handlers demonstration complete!")
