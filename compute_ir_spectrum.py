#!/usr/bin/env python3
"""
Compute IR Spectrum from ML-PES Molecular Dynamics
==================================================
Run MD on ML-PES, predict dipoles with ML-dipole model, compute IR spectrum.

The IR spectrum is computed from the Fourier transform of the dipole moment
autocorrelation function.

Author: Jonathan
Date: 2026-01-15
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import pickle


class IRSpectrumCalculator:
    """
    Calculate IR spectrum from molecular dynamics trajectory.
    
    Uses ML-PES for fast dynamics and ML-dipole model for dipole predictions.
    """
    
    def __init__(self, temperature=300.0, timestep_fs=0.5):
        """
        Initialize IR calculator.
        
        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
        timestep_fs : float
            MD timestep in femtoseconds
        """
        self.temperature = temperature
        self.timestep_fs = timestep_fs
        self.timestep_au = timestep_fs * 41.341  # fs to a.u.
        
    def compute_dipole_autocorrelation(self, dipoles):
        """
        Compute dipole moment autocorrelation function.
        
        Parameters
        ----------
        dipoles : ndarray, shape (n_steps, 3)
            Dipole moment time series in Debye
            
        Returns
        -------
        acf : ndarray, shape (n_steps,)
            Autocorrelation function
        """
        n_steps = len(dipoles)
        
        # Center the dipoles (remove DC component)
        dipoles_centered = dipoles - dipoles.mean(axis=0)
        
        # Compute autocorrelation for each component
        acf_components = []
        
        for i in range(3):
            # Use FFT for efficient computation
            dipole_fft = fft(dipoles_centered[:, i], n=2*n_steps)
            power = np.abs(dipole_fft)**2
            acf_full = np.real(fft(power))[:n_steps]
            acf_full = acf_full / acf_full[0]  # Normalize
            acf_components.append(acf_full)
        
        # Total ACF is sum of components
        acf = np.sum(acf_components, axis=0) / 3.0
        
        return acf
    
    def compute_ir_spectrum(self, acf):
        """
        Compute IR spectrum from autocorrelation function.
        
        Parameters
        ----------
        acf : ndarray
            Autocorrelation function
            
        Returns
        -------
        frequencies : ndarray
            Frequencies in cm^-1
        intensities : ndarray
            IR intensities (arbitrary units)
        """
        n_points = len(acf)
        
        # Apply window function to reduce spectral leakage
        window = signal.windows.hann(n_points)
        acf_windowed = acf * window
        
        # Fourier transform
        spectrum = fft(acf_windowed)
        
        # Frequencies
        freq_au = fftfreq(n_points, d=self.timestep_au)
        
        # Convert to cm^-1 (1 a.u. = 219474.63 cm^-1)
        AU_TO_CM = 219474.63
        freq_cm = freq_au * AU_TO_CM
        
        # Take positive frequencies only
        positive_mask = freq_cm > 0
        frequencies = freq_cm[positive_mask]
        intensities = np.abs(spectrum[positive_mask])**2
        
        # Apply quantum correction factor
        # I(ω) ∝ ω * [1 - exp(-ℏω/kT)] * |μ(ω)|²
        h_bar = 1.054571817e-34  # J·s
        k_B = 1.380649e-23       # J/K
        c = 2.99792458e10        # cm/s
        
        omega_joules = 2 * np.pi * c * frequencies * h_bar
        correction = frequencies * (1 - np.exp(-omega_joules / (k_B * self.temperature)))
        intensities = intensities * correction
        
        return frequencies, intensities
    
    def plot_ir_spectrum(self, frequencies, intensities, 
                        freq_range=(0, 4000), output_path=None):
        """
        Plot IR spectrum.
        
        Parameters
        ----------
        frequencies : ndarray
            Frequencies in cm^-1
        intensities : ndarray
            IR intensities
        freq_range : tuple
            Frequency range to plot (min, max) in cm^-1
        output_path : str, optional
            Save plot to file
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Filter to frequency range
        mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        freq_plot = frequencies[mask]
        int_plot = intensities[mask]
        
        # Plot
        ax.plot(freq_plot, int_plot, 'b-', linewidth=1.5)
        ax.fill_between(freq_plot, 0, int_plot, alpha=0.3)
        
        ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12, fontweight='bold')
        ax.set_ylabel('IR Intensity (arb. units)', fontsize=12, fontweight='bold')
        ax.set_title(f'Infrared Spectrum (T = {self.temperature} K)', 
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim(freq_range)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Spectrum saved: {output_path}")
        
        return fig
    
    def identify_peaks(self, frequencies, intensities, 
                      threshold=0.1, freq_range=(0, 4000)):
        """
        Identify major peaks in IR spectrum.
        
        Parameters
        ----------
        frequencies : ndarray
            Frequencies in cm^-1
        intensities : ndarray
            IR intensities
        threshold : float
            Minimum relative intensity (0-1)
        freq_range : tuple
            Frequency range to search
            
        Returns
        -------
        peaks : list of dict
            Peak information (frequency, intensity)
        """
        # Filter to range
        mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        freq_search = frequencies[mask]
        int_search = intensities[mask]
        
        # Normalize
        int_norm = int_search / int_search.max()
        
        # Find peaks
        peak_indices, properties = signal.find_peaks(
            int_norm,
            height=threshold,
            distance=20  # Minimum 20 points between peaks
        )
        
        # Extract peak info
        peaks = []
        for idx in peak_indices:
            peaks.append({
                'frequency': freq_search[idx],
                'intensity': int_search[idx],
                'rel_intensity': int_norm[idx]
            })
        
        # Sort by intensity
        peaks = sorted(peaks, key=lambda x: x['intensity'], reverse=True)
        
        return peaks


def run_md_with_dipole_prediction(ml_pes_model, dipole_model,
                                  initial_coords, symbols,
                                  temperature=300.0, n_steps=10000,
                                  timestep_fs=0.5, output_frequency=10):
    """
    Run MD on ML-PES with ML dipole predictions.
    
    Parameters
    ----------
    ml_pes_model : MLPESModel
        Trained ML potential energy surface
    dipole_model : DipoleSurfaceModel
        Trained ML dipole surface
    initial_coords : ndarray
        Starting coordinates
    symbols : list
        Atomic symbols
    temperature : float
        Temperature in K
    n_steps : int
        Number of MD steps
    timestep_fs : float
        Timestep in femtoseconds
    output_frequency : int
        Save every N steps
        
    Returns
    -------
    trajectory : dict
        MD trajectory with dipoles
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
    masses = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'S': 32.06}
    mass_array = np.array([masses[s] for s in symbols])
    
    # Velocity Verlet parameters
    dt = timestep_fs * 1e-15  # Convert to seconds
    kB = 1.380649e-23  # J/K
    
    # Initialize velocities (Maxwell-Boltzmann)
    velocities = np.random.randn(n_atoms, 3)
    velocities = velocities - velocities.mean(axis=0)  # Remove COM motion
    
    # Scale to temperature
    KE_target = 1.5 * n_atoms * kB * temperature
    KE_current = 0.5 * np.sum(mass_array[:, np.newaxis] * velocities**2)
    velocities *= np.sqrt(KE_target / KE_current)
    
    # Storage
    trajectory_coords = []
    trajectory_dipoles = []
    trajectory_energies = []
    trajectory_temps = []
    
    print(f"\nRunning MD...")
    
    from tqdm import tqdm as tqdm_bar
    
    for step in tqdm_bar(range(n_steps), desc="MD"):
        # Predict energy and forces
        energy, forces = ml_pes_model.predict_with_forces(coords)
        
        # Velocity Verlet integration
        # v(t + dt/2) = v(t) + 0.5 * a(t) * dt
        accel = forces / mass_array[:, np.newaxis]
        velocities += 0.5 * accel * dt
        
        # r(t + dt) = r(t) + v(t + dt/2) * dt
        coords += velocities * dt * 1e10  # Convert m to Angstrom
        
        # Predict forces at new position
        energy_new, forces_new = ml_pes_model.predict_with_forces(coords)
        
        # v(t + dt) = v(t + dt/2) + 0.5 * a(t + dt) * dt
        accel_new = forces_new / mass_array[:, np.newaxis]
        velocities += 0.5 * accel_new * dt
        
        # Save if needed
        if step % output_frequency == 0:
            # Predict dipole
            dipole = dipole_model.predict(coords)
            
            # Temperature
            KE = 0.5 * np.sum(mass_array[:, np.newaxis] * velocities**2)
            temp_inst = (2.0 / (3.0 * n_atoms * kB)) * KE
            
            trajectory_coords.append(coords.copy())
            trajectory_dipoles.append(dipole)
            trajectory_energies.append(energy_new)
            trajectory_temps.append(temp_inst)
    
    trajectory = {
        'coordinates': np.array(trajectory_coords),
        'dipoles': np.array(trajectory_dipoles),
        'energies': np.array(trajectory_energies),
        'temperatures': np.array(trajectory_temps),
        'symbols': symbols,
        'timestep_fs': timestep_fs,
        'output_frequency': output_frequency
    }
    
    print(f"\n✓ MD complete")
    print(f"  Frames saved: {len(trajectory_coords)}")
    print(f"  Average T: {np.mean(trajectory_temps):.1f} K")
    
    return trajectory


if __name__ == "__main__":
    print("IR Spectrum Calculator")
    print("Use compute_ir_workflow.py for complete workflow")
