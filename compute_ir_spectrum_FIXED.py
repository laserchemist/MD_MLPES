#!/usr/bin/env python3
"""
IR Spectrum Calculation - FIXED UNIT CONVERSIONS
=================================================
Computes IR spectrum from ML-PES + ML-dipole MD simulation.

CRITICAL FIX: Proper unit conversions in MD integrator!

Author: Jonathan
Date: 2026-01-17
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Physical constants and unit conversions
HARTREE_TO_JOULE = 4.3597447222071e-18  # J
ANGSTROM_TO_METER = 1e-10  # m
AMU_TO_KG = 1.66053906660e-27  # kg
BOHR_TO_ANGSTROM = 0.529177210903  # Å
AU_TO_DEBYE = 2.541746473  # Debye

# Derived conversion for forces
# F (Hartree/Å) / m (amu) → a (m/s²)
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
        'symbols': symbols
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
        # ν (cm⁻¹) = ν (Hz) / c (cm/s)
        c_cm_per_s = 2.99792458e10  # Speed of light in cm/s
        frequencies_cm = freq_hz / c_cm_per_s
        
        # Apply quantum correction factor
        # I(ν) ∝ ν * [1 - exp(-hcν/kT)] * |FFT|²
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


if __name__ == "__main__":
    print("IR Spectrum Calculator - Fixed Unit Conversions")
    print("=" * 60)
    print("\nThis module provides:")
    print("  - run_md_with_dipole_prediction() - Fixed MD integrator")
    print("  - IRSpectrumCalculator - Compute IR from dipoles")
    print("\nKey fixes:")
    print("  ✓ Proper Hartree/Å → m/s² conversion")
    print("  ✓ Correct unit handling in Velocity Verlet")
    print("  ✓ Molecules now actually move!")
