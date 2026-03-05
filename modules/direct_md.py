#!/usr/bin/env python3
"""
Direct Molecular Dynamics with PSI4

Runs ab initio molecular dynamics using PSI4 to generate training data
for machine learning potential energy surfaces. Records energy, gradients,
and dipole moments at each MD step.

This module is the core data generator for ML-PES training.

Classes:
    DirectMDConfig: Configuration for direct MD
    DirectMDRunner: Main MD runner with PSI4
    VelocityVerletIntegrator: Integration scheme
    
Functions:
    run_direct_md: Convenience function to run direct MD
    initialize_velocities: Maxwell-Boltzmann velocity initialization
    
Author: PSI4-MD Framework
Date: 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import logging
from dataclasses import dataclass, field
import time
import json

try:
    import psi4
    PSI4_AVAILABLE = True
except ImportError:
    PSI4_AVAILABLE = False
    print("Warning: PSI4 not available. Using mock calculations.")

from .test_molecules import TestMolecule
from .data_formats import TrajectoryData, save_trajectory

# Physical constants
KB_HARTREE_PER_K = 3.1668114e-6  # Boltzmann constant in Hartree/K
AMU_TO_AU = 1822.888486  # Conversion from amu to atomic units of mass
FS_TO_AU = 41.341374575751  # Conversion from fs to atomic units of time
ANGSTROM_TO_BOHR = 1.88972612456  # Conversion from Angstrom to Bohr
HARTREE_TO_KCAL = 627.509474  # Conversion from Hartree to kcal/mol
DEBYE_TO_AU = 0.393430307  # Conversion from Debye to atomic units

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class DirectMDConfig:
    """
    Configuration for direct MD simulations with PSI4.
    
    Attributes:
        method: Quantum chemistry method (e.g., 'B3LYP', 'HF', 'MP2')
        basis: Basis set (e.g., '6-31G*', 'cc-pVDZ')
        temperature: Temperature in Kelvin
        timestep: Timestep in femtoseconds
        n_steps: Number of MD steps
        output_frequency: Save trajectory every N steps
        thermostat: Thermostat type ('berendsen', 'velocity_rescale', None)
        thermostat_coupling: Coupling time for thermostat (fs)
        memory: PSI4 memory allocation
        threads: Number of threads for PSI4
        calculate_dipole: Whether to calculate dipole moments
        save_dipole: Whether to save dipole data
    """
    method: str = 'B3LYP'
    basis: str = '6-31G*'
    temperature: float = 300.0
    timestep: float = 0.5  # fs
    n_steps: int = 1000
    output_frequency: int = 10
    thermostat: Optional[str] = 'berendsen'
    thermostat_coupling: float = 100.0  # fs
    memory: str = '2GB'
    threads: int = 2
    calculate_dipole: bool = False  # NEW - compute post-hoc instead
    save_dipole: bool = False        # NEW - compute post-hoc instead
    random_seed: int = 42


class VelocityVerletIntegrator:
    """
    Velocity Verlet integrator for molecular dynamics.
    
    This is a symplectic integrator that conserves energy well.
    """
    
    def __init__(self, timestep: float):
        """
        Initialize integrator.
        
        Args:
            timestep: Integration timestep in femtoseconds
        """
        self.timestep = timestep
        self.dt_au = timestep * FS_TO_AU  # Convert to atomic units
    
    def step(self, coordinates: np.ndarray, velocities: np.ndarray,
             forces: np.ndarray, masses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one integration step.
        
        Args:
            coordinates: Positions in Angstrom (N_atoms, 3)
            velocities: Velocities in Angstrom/fs (N_atoms, 3)
            forces: Forces in Hartree/Angstrom (N_atoms, 3)
            masses: Atomic masses in amu (N_atoms,)
            
        Returns:
            Tuple of (new_coordinates, new_velocities)
        """
        # Convert to atomic units
        coords_bohr = coordinates * ANGSTROM_TO_BOHR
        forces_au = forces / ANGSTROM_TO_BOHR  # Forces in Hartree/Bohr
        masses_au = masses * AMU_TO_AU
        vels_au = velocities * ANGSTROM_TO_BOHR / FS_TO_AU  # Bohr/au_time
        
        # Velocity Verlet algorithm
        # v(t + dt/2) = v(t) + (F(t)/m) * dt/2
        half_step_vels = vels_au + 0.5 * forces_au / masses_au[:, np.newaxis] * self.dt_au
        
        # x(t + dt) = x(t) + v(t + dt/2) * dt
        new_coords_bohr = coords_bohr + half_step_vels * self.dt_au
        
        # Need forces at new position for final velocity update
        # This will be provided externally
        new_coords_angstrom = new_coords_bohr / ANGSTROM_TO_BOHR
        
        # Return half-step velocities in Angstrom/fs for later completion
        half_step_vels_angstrom = half_step_vels * FS_TO_AU / ANGSTROM_TO_BOHR
        
        return new_coords_angstrom, half_step_vels_angstrom
    
    def complete_step(self, half_step_velocities: np.ndarray,
                     forces_new: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """
        Complete the velocity update with forces at new position.
        
        Args:
            half_step_velocities: Velocities at t+dt/2 in Angstrom/fs
            forces_new: Forces at new position in Hartree/Angstrom
            masses: Atomic masses in amu
            
        Returns:
            New velocities in Angstrom/fs
        """
        # Convert to atomic units
        vels_au = half_step_velocities * ANGSTROM_TO_BOHR / FS_TO_AU
        forces_au = forces_new / ANGSTROM_TO_BOHR
        masses_au = masses * AMU_TO_AU
        
        # v(t + dt) = v(t + dt/2) + (F(t + dt)/m) * dt/2
        new_vels_au = vels_au + 0.5 * forces_au / masses_au[:, np.newaxis] * self.dt_au
        
        # Convert back to Angstrom/fs
        new_vels_angstrom = new_vels_au * FS_TO_AU / ANGSTROM_TO_BOHR
        
        return new_vels_angstrom


def initialize_velocities(masses: np.ndarray, temperature: float,
                         remove_com_motion: bool = True,
                         random_seed: Optional[int] = None) -> np.ndarray:
    """
    Initialize velocities from Maxwell-Boltzmann distribution.
    
    Args:
        masses: Atomic masses in amu (N_atoms,)
        temperature: Temperature in Kelvin
        remove_com_motion: Whether to remove center of mass motion
        random_seed: Random seed for reproducibility
        
    Returns:
        Velocities in Angstrom/fs (N_atoms, 3)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_atoms = len(masses)
    
    # Generate random velocities from Gaussian distribution
    # Standard deviation from Maxwell-Boltzmann distribution
    velocities_au = np.zeros((n_atoms, 3))
    
    for i in range(n_atoms):
        mass_au = masses[i] * AMU_TO_AU
        sigma = np.sqrt(KB_HARTREE_PER_K * temperature / mass_au)
        velocities_au[i] = np.random.normal(0, sigma, 3)
    
    # Remove center of mass motion if requested
    if remove_com_motion:
        total_momentum = np.sum(masses[:, np.newaxis] * velocities_au, axis=0)
        total_mass = np.sum(masses)
        com_velocity = total_momentum / total_mass
        velocities_au -= com_velocity
    
    # Convert to Angstrom/fs
    velocities_angstrom = velocities_au * FS_TO_AU / ANGSTROM_TO_BOHR
    
    return velocities_angstrom


class DirectMDRunner:
    """
    Main class for running direct MD with PSI4.
    """
    
    def __init__(self, config: DirectMDConfig, output_dir: str = None):
        """
        Initialize direct MD runner.
        
        Args:
            config: MD configuration
            output_dir: Directory for output files
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.integrator = VelocityVerletIntegrator(config.timestep)
        
        # Check PSI4 availability
        if not PSI4_AVAILABLE:
            logger.warning("PSI4 not available! Using mock calculations.")
            self.psi4_available = False
        else:
            self.psi4_available = True
            self._setup_psi4()
        
        # Atomic masses (amu)
        self.atomic_masses = {
            'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
            'F': 18.998, 'S': 32.065, 'Cl': 35.453, 'Br': 79.904
        }
    
    def _setup_psi4(self):
        """Setup PSI4 with configuration."""
        try:
            psi4.set_memory(self.config.memory)
            psi4.set_num_threads(self.config.threads)
            
            # Set PSI4 options
            psi4.set_options({
                'reference': 'rhf',
                'scf_type': 'df',
                'e_convergence': 1e-7,
                'd_convergence': 1e-7,
            })
            
            # Set output file
            output_file = self.output_dir / 'psi4_md.out'
            psi4.core.set_output_file(str(output_file), False)
            
            logger.info(f"PSI4 setup: {self.config.method}/{self.config.basis}")
            
        except Exception as e:
            logger.error(f"PSI4 setup failed: {e}")
            self.psi4_available = False
    
    def _create_psi4_molecule(self, symbols: List[str], coordinates: np.ndarray,
                             charge: int = 0, multiplicity: int = 1):
        """Create PSI4 molecule object."""
        # Build geometry string
        geom_lines = [f"{charge} {multiplicity}"]
        
        for symbol, coord in zip(symbols, coordinates):
            geom_lines.append(f"{symbol} {coord[0]:.10f} {coord[1]:.10f} {coord[2]:.10f}")
        
        geom_lines.extend([
            "units angstrom",
            "symmetry c1",
            "no_reorient",
            "no_com"
        ])
        
        geometry_string = '\n'.join(geom_lines)
        return psi4.geometry(geometry_string)
    
    def _calculate_energy_gradient(self, symbols: List[str], coordinates: np.ndarray,
                               charge: int = 0, multiplicity: int = 1) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
        """
        Calculate energy, gradient, and optionally dipole with PSI4.
        
        FIXED: Only computes dipole if calculate_dipole=True
        FIXED: Robust dipole extraction with multiple fallback methods
        
        Returns:
            Tuple of (energy_hartree, forces_hartree_per_angstrom, dipole_debye)
        """
        if not self.psi4_available:
            return self._mock_calculation(symbols, coordinates)
        
        try:
            # Create PSI4 molecule
            mol = self._create_psi4_molecule(symbols, coordinates, charge, multiplicity)
            
            # Calculate energy and gradient
            method_string = f"{self.config.method}/{self.config.basis}"
            
            # Compute energy and gradient (always needed for MD)
            energy, wfn = psi4.energy(method_string, molecule=mol, return_wfn=True)
            gradient = psi4.gradient(method_string, molecule=mol, ref_wfn=wfn)
            
            # Convert gradient to numpy array (in Hartree/Bohr)
            n_atoms = len(symbols)
            grad_bohr = np.array([[gradient.get(i, j) for j in range(3)] for i in range(n_atoms)])
            
            # Convert to forces in Hartree/Angstrom (force = -gradient)
            forces = -grad_bohr / ANGSTROM_TO_BOHR
            
            # Calculate dipole moment ONLY if requested
            dipole = None
            if self.config.calculate_dipole:
                try:
                    # Use oeprop to explicitly calculate dipole
                    psi4.oeprop(wfn, 'DIPOLE')
                    
                    # Multiple fallback methods for robustness
                    dipole_found = False
                    
                    # Method 1: Direct oeprop access (PSI4 >= 1.4)
                    if not dipole_found:
                        try:
                            oep = wfn.oeprop
                            if hasattr(oep, 'Dx'):
                                dipole_au = np.array([oep.Dx(), oep.Dy(), oep.Dz()])
                                # Convert to Debye
                                dipole = dipole_au / DEBYE_TO_AU
                                dipole_found = True
                        except:
                            pass
                    
                    # Method 2: PSI4 module variables (most common)
                    if not dipole_found:
                        try:
                            dipole_x = psi4.variable('DIPOLE X')
                            dipole_y = psi4.variable('DIPOLE Y')
                            dipole_z = psi4.variable('DIPOLE Z')
                            dipole = np.array([dipole_x, dipole_y, dipole_z])
                            dipole_found = True
                        except:
                            pass
                    
                    # Method 3: Wavefunction variables
                    if not dipole_found:
                        try:
                            dipole_x = wfn.variable('DIPOLE X')
                            dipole_y = wfn.variable('DIPOLE Y')
                            dipole_z = wfn.variable('DIPOLE Z')
                            dipole = np.array([dipole_x, dipole_y, dipole_z])
                            dipole_found = True
                        except:
                            pass
                    
                    # Method 4: Legacy "CURRENT DIPOLE" (PSI4 < 1.4)
                    if not dipole_found:
                        try:
                            dipole_au = wfn.variable('CURRENT DIPOLE')
                            dipole = np.array([dipole_au[i] for i in range(3)]) / DEBYE_TO_AU
                            dipole_found = True
                        except:
                            pass
                    
                    # Method 5: Parse from output (most robust but slower)
                    if not dipole_found:
                        try:
                            import io
                            import sys
                            
                            # Capture output
                            old_stdout = sys.stdout
                            sys.stdout = captured_output = io.StringIO()
                            
                            # Re-run oeprop
                            psi4.oeprop(wfn, 'DIPOLE')
                            
                            # Restore stdout
                            sys.stdout = old_stdout
                            output_text = captured_output.getvalue()
                            
                            # Parse dipole components
                            lines = output_text.split('\n')
                            dipole_x = dipole_y = dipole_z = None
                            
                            for line in lines:
                                if 'Dipole X' in line and 'Total' in line:
                                    dipole_x = float(line.split()[-1])
                                elif 'Dipole Y' in line and 'Total' in line:
                                    dipole_y = float(line.split()[-1])
                                elif 'Dipole Z' in line and 'Total' in line:
                                    dipole_z = float(line.split()[-1])
                            
                            if all(x is not None for x in [dipole_x, dipole_y, dipole_z]):
                                dipole = np.array([dipole_x, dipole_y, dipole_z])
                                dipole_found = True
                        except:
                            pass
                    
                    # If still not found, silent failure (don't spam logs)
                    if not dipole_found:
                        dipole = None
                    
                except Exception as e:
                    # Silent failure for dipole - don't break MD
                    dipole = None
            
            return float(energy), forces, dipole
            
        except Exception as e:
            logger.error(f"PSI4 calculation failed: {e}")
            return self._mock_calculation(symbols, coordinates)
                
            
    def _mock_calculation(self, symbols: List[str], coordinates: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Mock calculation for testing without PSI4."""
        n_atoms = len(symbols)
        
        # Harmonic potential around reference
        ref_coords = coordinates[0]  # Use first atom as reference
        displacements = coordinates - ref_coords
        
        # Mock energy (harmonic)
        energy = -100.0 + 0.5 * np.sum(displacements**2)
        
        # Mock forces (harmonic)
        forces = -displacements * 0.1
        
        # Mock dipole
        dipole = np.array([0.0, 0.0, 1.0])
        
        return energy, forces, dipole
    
    def apply_thermostat(self, velocities: np.ndarray, masses: np.ndarray,
                        current_temp: float) -> np.ndarray:
        """
        Apply thermostat to control temperature.
        
        Args:
            velocities: Current velocities in Angstrom/fs
            masses: Atomic masses in amu
            current_temp: Current temperature in K
            
        Returns:
            Modified velocities
        """
        if self.config.thermostat is None:
            return velocities
        
        target_temp = self.config.temperature
        
        if self.config.thermostat == 'velocity_rescale':
            # Simple velocity rescaling
            if current_temp > 0:
                scale = np.sqrt(target_temp / current_temp)
                return velocities * scale
        
        elif self.config.thermostat == 'berendsen':
            # Berendsen thermostat
            tau = self.config.thermostat_coupling  # fs
            dt = self.config.timestep  # fs
            
            if current_temp > 0:
                lambda_val = np.sqrt(1.0 + (dt / tau) * (target_temp / current_temp - 1.0))
                return velocities * lambda_val
        
        return velocities
    
    def calculate_temperature(self, velocities: np.ndarray, masses: np.ndarray) -> float:
        """
        Calculate instantaneous temperature.
        
        Args:
            velocities: Velocities in Angstrom/fs
            masses: Atomic masses in amu
            
        Returns:
            Temperature in Kelvin
        """
        # Convert to atomic units
        vels_au = velocities * ANGSTROM_TO_BOHR / FS_TO_AU
        masses_au = masses * AMU_TO_AU
        
        # Kinetic energy
        ke_au = 0.5 * np.sum(masses_au[:, np.newaxis] * vels_au**2)
        
        # Temperature from equipartition theorem (3N degrees of freedom)
        n_dof = 3 * len(masses)
        temp = 2 * ke_au / (n_dof * KB_HARTREE_PER_K)
        
        return temp
    
    def run(self, molecule: TestMolecule, initial_velocities: Optional[np.ndarray] = None) -> TrajectoryData:
        """
        Run direct MD simulation.
        
        Args:
            molecule: Initial molecular structure
            initial_velocities: Initial velocities (generated if None)
            
        Returns:
            TrajectoryData with MD trajectory
        """
        logger.info(f"Starting direct MD: {molecule.name}")
        logger.info(f"  Method: {self.config.method}/{self.config.basis}")
        logger.info(f"  Steps: {self.config.n_steps}, dt: {self.config.timestep} fs")
        logger.info(f"  Temperature: {self.config.temperature} K")
        
        # Get atomic masses
        masses = np.array([self.atomic_masses[symbol] for symbol in molecule.symbols])
        
        # Initialize velocities if not provided
        if initial_velocities is None:
            velocities = initialize_velocities(
                masses, self.config.temperature,
                random_seed=self.config.random_seed
            )
        else:
            velocities = initial_velocities.copy()
        
        # Storage for trajectory
        coordinates_list = []
        energies_list = []
        forces_list = []
        dipoles_list = [] if self.config.save_dipole else None
        times_list = []
        
        # Initialize
        coordinates = molecule.coordinates.copy()
        
        # Calculate initial energy and forces
        energy, forces, dipole = self._calculate_energy_gradient(
            molecule.symbols, coordinates,
            molecule.charge, molecule.multiplicity
        )
        
        logger.info(f"Initial energy: {energy:.6f} Hartree ({energy * HARTREE_TO_KCAL:.2f} kcal/mol)")
        
        # Save initial state
        coordinates_list.append(coordinates.copy())
        energies_list.append(energy)
        forces_list.append(forces.copy())
        if self.config.save_dipole and dipole is not None:
            dipoles_list.append(dipole.copy())
        times_list.append(0.0)
        
        # MD loop
        start_time = time.time()
        
        for step in range(1, self.config.n_steps + 1):
            # Velocity Verlet step 1: update positions and half-step velocities
            coordinates, half_step_vels = self.integrator.step(
                coordinates, velocities, forces, masses
            )
            
            # Calculate energy and forces at new position
            energy, forces, dipole = self._calculate_energy_gradient(
                molecule.symbols, coordinates,
                molecule.charge, molecule.multiplicity
            )
            
            # Velocity Verlet step 2: complete velocity update
            velocities = self.integrator.complete_step(half_step_vels, forces, masses)
            
            # Apply thermostat
            current_temp = self.calculate_temperature(velocities, masses)
            velocities = self.apply_thermostat(velocities, masses, current_temp)
            
            # Save data at specified frequency
            if step % self.config.output_frequency == 0:
                coordinates_list.append(coordinates.copy())
                energies_list.append(energy)
                forces_list.append(forces.copy())
                if self.config.save_dipole and dipole is not None:
                    dipoles_list.append(dipole.copy())
                times_list.append(step * self.config.timestep)
                
                # Progress update
                if step % (self.config.output_frequency * 10) == 0:
                    elapsed = time.time() - start_time
                    rate = step / elapsed
                    remaining = (self.config.n_steps - step) / rate
                    
                    logger.info(
                        f"Step {step}/{self.config.n_steps} | "
                        f"E = {energy * HARTREE_TO_KCAL:.2f} kcal/mol | "
                        f"T = {current_temp:.1f} K | "
                        f"ETA: {remaining:.1f}s"
                    )
        
        total_time = time.time() - start_time
        logger.info(f"MD completed in {total_time:.1f}s ({self.config.n_steps/total_time:.1f} steps/s)")
        
        # Create TrajectoryData
        trajectory = TrajectoryData(
            symbols=molecule.symbols,
            coordinates=np.array(coordinates_list),
            energies=np.array(energies_list),
            forces=np.array(forces_list),
            dipoles=np.array(dipoles_list) if dipoles_list else None,
            times=np.array(times_list),
            metadata={
                'molecule': molecule.name,
                'method': self.config.method,
                'basis': self.config.basis,
                'temperature': self.config.temperature,
                'timestep': self.config.timestep,
                'n_steps': self.config.n_steps,
                'thermostat': self.config.thermostat
            }
        )
        
        logger.info(f"Generated trajectory: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms")
        
        return trajectory


def run_direct_md(molecule: TestMolecule, config: DirectMDConfig = None,
                 output_dir: str = None, save_format: str = 'extxyz') -> TrajectoryData:
    """
    Convenience function to run direct MD and save results.
    
    Args:
        molecule: Molecular structure
        config: MD configuration (uses defaults if None)
        output_dir: Output directory
        save_format: Format for saving trajectory
        
    Returns:
        TrajectoryData object
        
    Example:
        >>> from .test_molecules import get_molecule
        >>> water = get_molecule('water')
        >>> config = DirectMDConfig(method='HF', basis='STO-3G', n_steps=100)
        >>> traj = run_direct_md(water, config, output_dir='md_data')
    """
    if config is None:
        config = DirectMDConfig()
    
    if output_dir is None:
        output_dir = './md_output'
    
    runner = DirectMDRunner(config, output_dir)
    trajectory = runner.run(molecule)
    
    # Save trajectory
    output_path = Path(output_dir) / f"{molecule.name}_trajectory.{save_format}"
    save_trajectory(trajectory, str(output_path), format=save_format)
    
    # Save metadata
    metadata_path = Path(output_dir) / f"{molecule.name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(trajectory.metadata, f, indent=2)
    
    logger.info(f"Saved trajectory to {output_path}")
    
    return trajectory


if __name__ == "__main__":
    print("🚀 Direct MD with PSI4 - Demonstration")
    print("=" * 60)
    
    from .test_molecules import get_molecule
    
    # Get test molecule
    water = get_molecule('water')
    print(f"\n🧪 Test molecule: {water.name}")
    print(f"   Formula: {water.formula}")
    print(f"   Atoms: {len(water.symbols)}")
    
    # Configure MD
    config = DirectMDConfig(
        method='HF',  # Use HF for faster testing
        basis='STO-3G',  # Minimal basis for speed
        temperature=300.0,
        timestep=0.5,
        n_steps=50,  # Short for testing
        output_frequency=5,
        thermostat='berendsen',
        calculate_dipole=True
    )
    
    print(f"\n⚙️  MD Configuration:")
    print(f"   Method: {config.method}/{config.basis}")
    print(f"   Steps: {config.n_steps} x {config.timestep} fs")
    print(f"   Temperature: {config.temperature} K")
    
    # Run MD
    output_dir = '/home/claude/psi4md_framework/outputs/direct_md_test'
    
    print(f"\n🏃 Running direct MD...")
    trajectory = run_direct_md(water, config, output_dir, save_format='extxyz')
    
    print(f"\n✅ MD Complete!")
    print(f"   Frames generated: {trajectory.n_frames}")
    print(f"   Energy range: {trajectory.energies.min():.6f} to {trajectory.energies.max():.6f} Hartree")
    print(f"   Average energy: {trajectory.energies.mean():.6f} Hartree")
    
    if trajectory.dipoles is not None:
        dipole_magnitudes = np.linalg.norm(trajectory.dipoles, axis=1)
        print(f"   Dipole range: {dipole_magnitudes.min():.3f} to {dipole_magnitudes.max():.3f} Debye")
    
    print(f"\n📁 Output saved to: {output_dir}/")
    print(f"   Trajectory: {water.name}_trajectory.extxyz")
    print(f"   Metadata: {water.name}_metadata.json")
    
    print("\n✅ Direct MD demonstration complete!")
