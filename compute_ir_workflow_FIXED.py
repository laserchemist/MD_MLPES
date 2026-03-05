#!/usr/bin/env python3
"""
Complete IR Spectrum Workflow - FIXED
======================================
Handles ML-PES models saved as dicts (from complete_workflow_v2.2.py)

Author: Jonathan
Date: 2026-01-17
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle


# Import or define the wrapper
try:
    from ml_pes_wrapper import load_ml_pes_model
except ImportError:
    # Define wrapper inline if not available
    class MLPESModelWrapper:
        """Wrapper for ML-PES models providing predict_with_forces interface."""
        
        def __init__(self, model_data):
            self.model_data = model_data
            self._setup_components()
        
        def _setup_components(self):
            """Extract model components."""
            if isinstance(self.model_data, dict):
                self.model = self.model_data.get('model')
                self.scaler_X = self.model_data.get('scaler_X')
                self.scaler_y = self.model_data.get('scaler_y')
                self.descriptor = self.model_data.get('descriptor')
                self.symbols = self.model_data.get('symbols', [])
                
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
            """Predict energy."""
            if hasattr(self.model_data, 'predict'):
                return self.model_data.predict(self.symbols, coords)
            else:
                desc = self.descriptor.compute(self.symbols, coords)
                desc_scaled = self.scaler_X.transform([desc])
                e_scaled = self.model.predict(desc_scaled)
                energy = self.scaler_y.inverse_transform([[e_scaled[0]]])[0, 0]
                return energy
        
        def predict_with_forces(self, coords):
            """Predict energy and forces via finite differences."""
            energy = self.predict(coords)
            forces = self._compute_forces_finite_diff(coords)
            return energy, forces
        
        def _compute_forces_finite_diff(self, coords, delta=0.001):
            """Compute forces via finite differences."""
            n_atoms = coords.shape[0]
            forces = np.zeros_like(coords)
            
            for i in range(n_atoms):
                for j in range(3):
                    coords_plus = coords.copy()
                    coords_plus[i, j] += delta
                    e_plus = self.predict(coords_plus)
                    
                    coords_minus = coords.copy()
                    coords_minus[i, j] -= delta
                    e_minus = self.predict(coords_minus)
                    
                    forces[i, j] = -(e_plus - e_minus) / (2 * delta)
            
            return forces
    
    def load_ml_pes_model(filepath):
        """Load ML-PES model and return wrapper."""
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
        return MLPESModelWrapper(loaded)


def train_dipole_workflow(training_data_path, output_model='dipole_surface.pkl'):
    """Train ML dipole surface from training data."""
    try:
        from ml_dipole_surface_FIXED import train_dipole_surface_from_file
    except ImportError:
        from ml_dipole_surface import train_dipole_surface_from_file
    
    print("\n" + "="*80)
    print("WORKFLOW STEP 1: Train ML Dipole Surface")
    print("="*80)
    
    model = train_dipole_surface_from_file(training_data_path, output_model)
    
    print("\n✅ Dipole model training complete!")
    print(f"   Model saved: {output_model}")
    print(f"   Test MAE: {model.training_stats['magnitude_test_mae']:.4f} Debye")
    
    return model


# ===== FIXED MD INTEGRATION WITH PROPER UNIT CONVERSIONS =====

# Physical constants and unit conversions
HARTREE_TO_JOULE = 4.3597447222071e-18  # J
ANGSTROM_TO_METER = 1e-10  # m
AMU_TO_KG = 1.66053906660e-27  # kg

# Derived conversion for forces: F (Hartree/Å) / m (amu) → a (m/s²)
FORCE_CONVERSION = (HARTREE_TO_JOULE / ANGSTROM_TO_METER) / AMU_TO_KG
# = 2.6255e19 m/s² per (Hartree/Å)/amu

KB_SI = 1.380649e-23  # J/K (Boltzmann constant)


def run_md_with_dipole_prediction(
    ml_pes_model,
    dipole_model,
    initial_coords,
    symbols,
    temperature=300.0,
    n_steps=10000,
    timestep_fs=0.5,
    output_frequency=10
):
    """
    Run molecular dynamics with ML-PES and ML-dipole predictions.
    
    FIXED: Proper unit conversions in Velocity Verlet integrator!
    
    Args:
        ml_pes_model: ML-PES model with predict_with_forces() method
        dipole_model: ML-dipole model with predict() method
        initial_coords: Initial coordinates (N_atoms, 3) in Angstrom
        symbols: Atomic symbols
        temperature: Target temperature (K)
        n_steps: Number of MD steps
        timestep_fs: Timestep in femtoseconds
        output_frequency: Save every N steps
        
    Returns:
        Dictionary with trajectory data
    """
    print("\n" + "="*60)
    print("Running MD with Dipole Prediction")
    print("="*60)
    
    print(f"\nTemperature: {temperature} K")
    print(f"Steps: {n_steps}")
    print(f"Timestep: {timestep_fs} fs")
    
    # Initialize
    coords = initial_coords.copy()
    n_atoms = len(symbols)
    
    # Atomic masses (amu)
    masses_dict = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 
                   'F': 18.998, 'S': 32.06, 'Cl': 35.45}
    mass_array = np.array([masses_dict.get(s, 12.0) for s in symbols])
    
    # Timestep in seconds
    dt = timestep_fs * 1e-15
    
    # Initialize velocities (Maxwell-Boltzmann distribution)
    velocities = np.random.randn(n_atoms, 3)
    velocities = velocities - velocities.mean(axis=0)  # Remove COM motion
    
    # Scale velocities to target temperature
    # KE = (3/2) * N * kB * T in SI units (Joules)
    KE_target = 1.5 * n_atoms * KB_SI * temperature
    
    # Current KE: (1/2) * sum(m * v²) in SI units
    mass_kg = mass_array * AMU_TO_KG  # Convert to kg
    KE_current = 0.5 * np.sum(mass_kg[:, np.newaxis] * velocities**2)
    
    if KE_current > 0:
        velocities *= np.sqrt(KE_target / KE_current)
    
    # Storage
    trajectory_coords = []
    trajectory_dipoles = []
    trajectory_energies = []
    trajectory_temps = []
    
    print(f"\nRunning MD...")
    
    try:
        from tqdm import tqdm as tqdm_bar
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    iterator = tqdm_bar(range(n_steps), desc="MD") if use_tqdm else range(n_steps)
    
    for step in iterator:
        # ===== VELOCITY VERLET INTEGRATION (FIXED!) =====
        
        # Get forces at current position
        energy, forces = ml_pes_model.predict_with_forces(coords)
        
        # Convert forces: Hartree/Å → acceleration in m/s²
        # forces / mass_amu * FORCE_CONVERSION → m/s²
        accel = (forces / mass_array[:, np.newaxis]) * FORCE_CONVERSION
        
        # Half-step velocity update
        # v(t + dt/2) = v(t) + 0.5 * a(t) * dt
        velocities += 0.5 * accel * dt
        
        # Position update (convert m → Å)
        # r(t + dt) = r(t) + v(t + dt/2) * dt
        coords += velocities * dt / ANGSTROM_TO_METER
        
        # Get forces at new position
        energy_new, forces_new = ml_pes_model.predict_with_forces(coords)
        
        # Convert new forces
        accel_new = (forces_new / mass_array[:, np.newaxis]) * FORCE_CONVERSION
        
        # Complete velocity update
        # v(t + dt) = v(t + dt/2) + 0.5 * a(t + dt) * dt
        velocities += 0.5 * accel_new * dt
        
        # ===== SAVE DATA =====
        if step % output_frequency == 0:
            # Predict dipole
            dipole = dipole_model.predict(coords)
            
            # Calculate instantaneous temperature
            KE = 0.5 * np.sum(mass_kg[:, np.newaxis] * velocities**2)
            temp_inst = (2.0 / (3.0 * n_atoms * KB_SI)) * KE
            
            trajectory_coords.append(coords.copy())
            trajectory_dipoles.append(dipole)
            trajectory_energies.append(energy_new)
            trajectory_temps.append(temp_inst)
    
    print(f"\n✓ MD complete")
    print(f"  Frames saved: {len(trajectory_coords)}")
    print(f"  Average T: {np.mean(trajectory_temps):.1f} K")
    
    return {
        'coordinates': np.array(trajectory_coords),
        'dipoles': np.array(trajectory_dipoles),
        'energies': np.array(trajectory_energies),
        'temperatures': np.array(trajectory_temps),
        'symbols': symbols,
        'timestep_fs': timestep_fs,
        'output_frequency': output_frequency
    }


class IRSpectrumCalculator:
    """Calculate IR spectrum from dipole trajectory."""
    
    def __init__(self, temperature=300.0, timestep_fs=1.0):
        """
        Initialize calculator.
        
        Args:
            temperature: Temperature in K
            timestep_fs: Effective timestep in fs (accounting for output frequency)
        """
        self.temperature = temperature
        self.timestep_fs = timestep_fs
        self.timestep_s = timestep_fs * 1e-15
    
    def compute_dipole_autocorrelation(self, dipoles):
        """
        Compute dipole moment autocorrelation function.
        
        Args:
            dipoles: Array of dipole moments (N_frames, 3)
            
        Returns:
            Autocorrelation function
        """
        n_frames = len(dipoles)
        
        # Center dipoles
        dipoles_centered = dipoles - dipoles.mean(axis=0)
        
        # Compute ACF
        acf = np.zeros(n_frames)
        
        for t in range(n_frames):
            if t == 0:
                acf[t] = np.sum(dipoles_centered * dipoles_centered)
            else:
                acf[t] = np.sum(dipoles_centered[:-t] * dipoles_centered[t:])
        
        # Normalize
        acf /= acf[0]
        
        return acf
    
    def compute_ir_spectrum(self, acf):
        """
        Compute IR spectrum from autocorrelation function.
        
        Args:
            acf: Dipole autocorrelation function
            
        Returns:
            frequencies (cm⁻¹), intensities
        """
        # Apply window function to reduce spectral leakage
        window = np.blackman(len(acf))
        acf_windowed = acf * window
        
        # FFT
        spectrum = np.fft.rfft(acf_windowed)
        intensities = np.abs(spectrum)**2
        
        # Frequency axis
        n_points = len(acf)
        freq_hz = np.fft.rfftfreq(n_points, d=self.timestep_s)
        
        # Convert Hz → cm⁻¹
        c_cm_per_s = 2.99792458e10  # Speed of light in cm/s
        frequencies_cm = freq_hz / c_cm_per_s
        
        # Apply quantum correction factor
        h = 6.62607015e-34  # J·s
        c = 2.99792458e8  # m/s
        kB = 1.380649e-23  # J/K
        
        # Avoid division by zero
        freq_hz_safe = np.where(freq_hz > 0, freq_hz, 1e-10)
        
        quantum_factor = freq_hz_safe * (1 - np.exp(-h * freq_hz_safe / (kB * self.temperature)))
        intensities *= quantum_factor
        
        # Normalize
        if intensities.max() > 0:
            intensities /= intensities.max()
        
        return frequencies_cm, intensities
    
    def identify_peaks(self, frequencies, intensities, threshold=0.1):
        """Identify major peaks in spectrum."""
        try:
            from scipy.signal import find_peaks
            
            # Find peaks above threshold
            peaks, properties = find_peaks(intensities, height=threshold, prominence=threshold/2)
            
            # Sort by intensity
            peak_intensities = intensities[peaks]
            sorted_indices = np.argsort(peak_intensities)[::-1]
            
            results = []
            for idx in sorted_indices:
                peak_idx = peaks[idx]
                results.append({
                    'frequency': frequencies[peak_idx],
                    'intensity': intensities[peak_idx],
                    'rel_intensity': intensities[peak_idx] / intensities.max() if intensities.max() > 0 else 0
                })
            
            return results
        except ImportError:
            # Fallback without scipy
            return []
    
    def plot_ir_spectrum(self, frequencies, intensities, freq_range=(0, 4000), output_path=None):
        """Plot IR spectrum."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot spectrum
        mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        ax.plot(frequencies[mask], intensities[mask], 'b-', linewidth=2)
        ax.fill_between(frequencies[mask], 0, intensities[mask], alpha=0.3)
        
        ax.set_xlabel('Frequency (cm⁻¹)', fontsize=14, fontweight='bold')
        ax.set_ylabel('IR Intensity (arb. units)', fontsize=14, fontweight='bold')
        ax.set_title(f'Infrared Spectrum (T = {self.temperature} K)', fontsize=16, fontweight='bold')
        ax.set_xlim(freq_range)
        ax.set_ylim(0, intensities[mask].max() * 1.1 if intensities[mask].max() > 0 else 1)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Spectrum saved: {output_path}")
        
        return fig


def compute_ir_workflow(ml_pes_path, dipole_model_path, 
                       training_data_path=None,
                       temperature=300.0, n_steps=50000,
                       timestep=0.5, output_frequency=10,
                       output_dir='ir_spectrum_output'):
    """Compute IR spectrum using ML-PES and ML-dipole."""
    try:
        from ml_dipole_surface_FIXED import DipoleSurfaceModel
    except ImportError:
        from ml_dipole_surface import DipoleSurfaceModel
    
    # Use embedded fixed MD code (don't import buggy compute_ir_spectrum)
    
    print("\n" + "="*80)
    print("WORKFLOW STEP 2: Compute IR Spectrum")
    print("="*80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\nOutput directory: {output_path}")
    
    # Load models
    print("\n" + "="*60)
    print("Loading Models")
    print("="*60)
    
    print(f"\nLoading ML-PES: {ml_pes_path}")
    ml_pes_model = load_ml_pes_model(ml_pes_path)
    print("✓ ML-PES loaded")
    
    print(f"\nLoading dipole model: {dipole_model_path}")
    dipole_model = DipoleSurfaceModel.load(dipole_model_path)
    print("✓ Dipole model loaded")
    
    # Get initial structure
    if training_data_path:
        try:
            from data_loading_utils import load_training_data_with_dipoles
            print(f"\nLoading initial structure from: {training_data_path}")
            data_dict = load_training_data_with_dipoles(training_data_path)
            initial_coords = data_dict['coordinates'][0]
            symbols = data_dict['symbols']
        except:
            # Fallback - load as npz
            print(f"\nLoading initial structure from: {training_data_path}")
            data = np.load(training_data_path, allow_pickle=True)
            symbols = data['symbols'].tolist() if hasattr(data['symbols'], 'tolist') else list(data['symbols'])
            initial_coords = data['coordinates'][0]
    else:
        # Use dipole model's symbols
        print("\nUsing molecule from dipole model")
        symbols = dipole_model.symbols
        # Create a reasonable starting geometry
        # For ammonia (N H H H): pyramidal structure
        if symbols == ['N', 'H', 'H', 'H']:
            initial_coords = np.array([
                [0.0, 0.0, 0.0],      # N
                [0.0, 0.94, 0.38],    # H
                [0.81, -0.47, 0.38],  # H
                [-0.81, -0.47, 0.38]  # H
            ])
        else:
            raise ValueError(f"No initial structure for molecule: {symbols}")
    
    print(f"✓ Initial structure: {' '.join(symbols)}")
    
    # Run MD with dipole prediction
    print("\n" + "="*60)
    print("Running Molecular Dynamics")
    print("="*60)
    
    trajectory = run_md_with_dipole_prediction(
        ml_pes_model=ml_pes_model,
        dipole_model=dipole_model,
        initial_coords=initial_coords,
        symbols=symbols,
        temperature=temperature,
        n_steps=n_steps,
        timestep_fs=timestep,
        output_frequency=output_frequency
    )
    
    # Save trajectory
    traj_file = output_path / 'md_trajectory.npz'
    np.savez(
        traj_file,
        coordinates=trajectory['coordinates'],
        dipoles=trajectory['dipoles'],
        energies=trajectory['energies'],
        temperatures=trajectory['temperatures'],
        symbols=trajectory['symbols'],
        timestep_fs=timestep,
        output_frequency=output_frequency
    )
    print(f"\n✓ Trajectory saved: {traj_file}")
    
    # Compute IR spectrum
    print("\n" + "="*60)
    print("Computing IR Spectrum")
    print("="*60)
    
    calculator = IRSpectrumCalculator(
        temperature=temperature,
        timestep_fs=timestep * output_frequency
    )
    
    # Dipole autocorrelation
    print("\nComputing dipole autocorrelation function...")
    dipoles = trajectory['dipoles']
    acf = calculator.compute_dipole_autocorrelation(dipoles)
    
    # Plot ACF
    fig_acf, ax = plt.subplots(figsize=(10, 6))
    time_ps = np.arange(len(acf)) * timestep * output_frequency / 1000
    ax.plot(time_ps[:min(1000, len(acf))], acf[:min(1000, len(acf))], 'b-', linewidth=2)
    ax.set_xlabel('Time (ps)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Autocorrelation', fontsize=12, fontweight='bold')
    ax.set_title('Dipole Moment Autocorrelation Function', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    acf_plot = output_path / 'dipole_acf.png'
    plt.savefig(acf_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ACF plot saved: {acf_plot}")
    
    # IR spectrum
    print("\nComputing IR spectrum...")
    frequencies, intensities = calculator.compute_ir_spectrum(acf)
    
    # Save spectrum data
    spectrum_file = output_path / 'ir_spectrum.txt'
    np.savetxt(
        spectrum_file,
        np.column_stack([frequencies, intensities]),
        header='Frequency (cm-1)\tIntensity (arb. units)',
        fmt='%.2f\t%.6e'
    )
    print(f"✓ Spectrum data saved: {spectrum_file}")
    
    # Plot spectrum
    print("\nPlotting IR spectrum...")
    fig_ir = calculator.plot_ir_spectrum(
        frequencies, intensities,
        freq_range=(0, 4000),
        output_path=output_path / 'ir_spectrum.png'
    )
    plt.close()
    
    # Identify peaks
    print("\nIdentifying major peaks...")
    peaks = calculator.identify_peaks(frequencies, intensities, threshold=0.1)
    
    print("\n📊 Major IR Peaks:")
    print("-" * 60)
    for i, peak in enumerate(peaks[:10], 1):
        print(f"  {i:2d}. {peak['frequency']:7.1f} cm⁻¹  "
              f"(intensity: {peak['rel_intensity']:.2f})")
    
    # Summary
    print("\n" + "="*80)
    print("IR SPECTRUM COMPUTATION COMPLETE")
    print("="*80)
    
    print(f"\n✅ Results saved to: {output_path}/")
    print(f"   MD trajectory: md_trajectory.npz")
    print(f"   Dipole ACF plot: dipole_acf.png")
    print(f"   IR spectrum plot: ir_spectrum.png")
    print(f"   Spectrum data: ir_spectrum.txt")
    
    print(f"\n📊 Statistics:")
    print(f"   MD frames: {len(trajectory['coordinates'])}")
    print(f"   Average T: {np.mean(trajectory['temperatures']):.1f} K")
    print(f"   Frequency range: 0-4000 cm⁻¹")
    print(f"   Major peaks identified: {len(peaks)}")
    
    return trajectory, frequencies, intensities, peaks


def main():
    parser = argparse.ArgumentParser(
        description='Complete IR spectrum workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Training options
    parser.add_argument('--train-dipole', action='store_true',
                       help='Train dipole surface model')
    parser.add_argument('--training-data', type=str,
                       help='Training data file (with dipoles)')
    parser.add_argument('--dipole-output', type=str, default='dipole_surface.pkl',
                       help='Output path for trained dipole model')
    
    # Computation options
    parser.add_argument('--ml-pes', type=str,
                       help='Trained ML-PES model file')
    parser.add_argument('--dipole-model', type=str,
                       help='Trained dipole surface model')
    parser.add_argument('--temp', type=float, default=300.0,
                       help='MD temperature (K)')
    parser.add_argument('--steps', type=int, default=50000,
                       help='Number of MD steps')
    parser.add_argument('--timestep', type=float, default=0.5,
                       help='MD timestep (fs)')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save frequency')
    parser.add_argument('--output-dir', type=str, default='ir_spectrum_output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.train_dipole:
        if not args.training_data:
            parser.error("--train-dipole requires --training-data")
        
        # Train dipole model
        train_dipole_workflow(args.training_data, args.dipole_output)
        
        # If not computing IR, we're done
        if not args.ml_pes:
            return
        
        # Use the just-trained model
        args.dipole_model = args.dipole_output
    
    # Compute IR spectrum
    if args.ml_pes:
        if not args.dipole_model:
            parser.error("Computing IR requires --dipole-model")
        
        compute_ir_workflow(
            ml_pes_path=args.ml_pes,
            dipole_model_path=args.dipole_model,
            training_data_path=args.training_data,
            temperature=args.temp,
            n_steps=args.steps,
            timestep=args.timestep,
            output_frequency=args.save_every,
            output_dir=args.output_dir
        )
    else:
        if not args.train_dipole:
            parser.error("Must specify either --train-dipole or --ml-pes")
            parser.print_help()


if __name__ == '__main__':
    main()
