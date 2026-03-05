#!/usr/bin/env python3
"""
IR Spectrum Visualization Module

Creates publication-quality visualizations for IR spectroscopy analysis:
- IR spectrum with peak labels
- Dipole autocorrelation function
- Dipole moment evolution
- Energy trajectory
- Multi-panel dashboard

All plots are publication-ready (300 DPI, proper units, clear labels).

Author: PSI4-MD ML-PES Framework
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional
from scipy.signal import find_peaks

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3
})


def plot_ir_spectrum(frequencies: np.ndarray, 
                     intensities: np.ndarray,
                     output_file: str,
                     peak_threshold: float = 0.1,
                     title: str = None,
                     verbose: bool = True):
    """
    Plot IR spectrum with peak labels.
    
    Args:
        frequencies: Frequencies in cm⁻¹
        intensities: Normalized intensities
        output_file: Output filename
        peak_threshold: Threshold for peak labeling (0-1)
        title: Custom title
        verbose: Print progress
    """
    if verbose:
        print(f"\n📈 Creating IR spectrum plot...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot spectrum
    ax.plot(frequencies, intensities, 'b-', linewidth=1.5, label='IR Spectrum')
    ax.fill_between(frequencies, intensities, alpha=0.3)
    
    # Find and label peaks
    peak_indices, properties = find_peaks(
        intensities,
        height=peak_threshold,
        distance=20
    )
    
    # Get peak frequencies and intensities
    peak_freqs = frequencies[peak_indices]
    peak_ints = intensities[peak_indices]
    
    # Sort by intensity to label strongest peaks
    sorted_indices = np.argsort(peak_ints)[::-1]
    
    # Label top 10 peaks
    for i in sorted_indices[:10]:
        freq = peak_freqs[i]
        intensity = peak_ints[i]
        
        ax.annotate(f'{freq:.0f}',
                   xy=(freq, intensity),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Mark peaks
    ax.plot(peak_freqs, peak_ints, 'ro', markersize=6, label=f'Peaks (>{peak_threshold:.1%})')
    
    # Labels and formatting
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Intensity (arbitrary units)', fontsize=14, fontweight='bold')
    
    if title is None:
        title = 'Infrared Spectrum'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(0, frequencies.max())
    ax.set_ylim(0, intensities.max() * 1.15)
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add region labels
    regions = [
        (0, 500, 'Torsion'),
        (500, 1000, 'C-O\nC-N'),
        (1000, 1500, 'C-H\nbend'),
        (1500, 1800, 'C=C\nC=O'),
        (1800, 2500, 'C≡C\nC≡N'),
        (2500, 3000, 'C-H\nstretch'),
        (3000, 4000, 'O-H\nN-H')
    ]
    
    y_pos = intensities.max() * 1.08
    for start, end, label in regions:
        if start >= frequencies.min() and end <= frequencies.max():
            mid = (start + end) / 2
            ax.text(mid, y_pos, label, ha='center', va='bottom',
                   fontsize=8, style='italic', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"   ✅ Saved: {output_file}")
        print(f"   Found {len(peak_freqs)} peaks above threshold")


def plot_dipole_autocorrelation(lags: np.ndarray,
                                autocorr: np.ndarray,
                                timestep: float,
                                output_file: str,
                                verbose: bool = True):
    """
    Plot dipole autocorrelation function.
    
    Args:
        lags: Time lags in steps
        autocorr: Autocorrelation values
        timestep: Timestep in fs
        output_file: Output filename
        verbose: Print progress
    """
    if verbose:
        print(f"\n📈 Creating autocorrelation plot...")
    
    # Convert lags to time in ps
    time_ps = lags * timestep / 1000
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot autocorrelation
    ax.plot(time_ps, autocorr, 'b-', linewidth=2, label='C(t)')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    # Highlight where it crosses zero
    zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
    if len(zero_crossings) > 0:
        first_zero = zero_crossings[0]
        ax.axvline(x=time_ps[first_zero], color='r', linestyle='--', 
                  linewidth=1, alpha=0.7, label=f'First zero: {time_ps[first_zero]:.2f} ps')
    
    # Labels
    ax.set_xlabel('Time (ps)', fontsize=14, fontweight='bold')
    ax.set_ylabel('C(t) = ⟨μ(0)·μ(t)⟩ / ⟨μ(0)·μ(0)⟩', fontsize=14, fontweight='bold')
    ax.set_title('Dipole Moment Autocorrelation Function', fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(0, time_ps.max())
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"   ✅ Saved: {output_file}")


def plot_dipole_evolution(dipoles: np.ndarray,
                         timestep: float,
                         output_file: str,
                         verbose: bool = True):
    """
    Plot dipole moment components over time.
    
    Args:
        dipoles: Dipole moments (n_frames, 3) in Debye
        timestep: Timestep in fs
        output_file: Output filename
        verbose: Print progress
    """
    if verbose:
        print(f"\n📈 Creating dipole evolution plot...")
    
    # Time axis
    time_ps = np.arange(len(dipoles)) * timestep / 1000
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Individual components
    components = ['μx', 'μy', 'μz']
    colors = ['r', 'g', 'b']
    
    for i, (comp, color) in enumerate(zip(components, colors)):
        ax = axes[i]
        ax.plot(time_ps, dipoles[:, i], color=color, linewidth=1, alpha=0.7)
        ax.set_ylabel(f'{comp} (Debye)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend([comp], loc='upper right')
        
        # Show mean and std
        mean = dipoles[:, i].mean()
        std = dipoles[:, i].std()
        ax.axhline(y=mean, color=color, linestyle='--', linewidth=1, alpha=0.5)
        ax.text(0.02, 0.95, f'μ̄ = {mean:.3f} D\nσ = {std:.3f} D',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Total magnitude
    ax = axes[3]
    magnitude = np.linalg.norm(dipoles, axis=1)
    ax.plot(time_ps, magnitude, 'k-', linewidth=1, alpha=0.7)
    ax.set_ylabel('|μ| (Debye)', fontweight='bold')
    ax.set_xlabel('Time (ps)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(['|μ|'], loc='upper right')
    
    mean_mag = magnitude.mean()
    std_mag = magnitude.std()
    ax.axhline(y=mean_mag, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(0.02, 0.95, f'|μ̄| = {mean_mag:.3f} D\nσ = {std_mag:.3f} D',
           transform=ax.transAxes, fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('Dipole Moment Evolution', fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"   ✅ Saved: {output_file}")


def plot_energy_trajectory(energies: np.ndarray,
                          timestep: float,
                          output_file: str,
                          verbose: bool = True):
    """
    Plot energy trajectory.
    
    Args:
        energies: Energies in Hartree
        timestep: Timestep in fs  
        output_file: Output filename
        verbose: Print progress
    """
    if verbose:
        print(f"\n📈 Creating energy trajectory plot...")
    
    # Time axis and convert to kcal/mol
    time_ps = np.arange(len(energies)) * timestep / 1000
    energies_kcal = energies * 627.509
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot energy
    ax.plot(time_ps, energies_kcal, 'b-', linewidth=1, alpha=0.7)
    
    # Running average
    window = min(50, len(energies) // 10)
    if window > 1:
        running_avg = np.convolve(energies_kcal, np.ones(window)/window, mode='valid')
        time_avg = time_ps[window//2:len(running_avg)+window//2]
        ax.plot(time_avg, running_avg, 'r-', linewidth=2, label=f'Running average ({window} frames)')
    
    # Statistics
    mean_e = energies_kcal.mean()
    std_e = energies_kcal.std()
    
    ax.axhline(y=mean_e, color='g', linestyle='--', linewidth=1, alpha=0.7)
    
    # Labels
    ax.set_xlabel('Time (ps)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Energy (kcal/mol)', fontsize=14, fontweight='bold')
    ax.set_title('Energy Conservation During Production MD', fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Statistics box
    stats_text = f'Mean: {mean_e:.2f} kcal/mol\n'
    stats_text += f'Std: {std_e:.2f} kcal/mol\n'
    stats_text += f'Range: {energies_kcal.max() - energies_kcal.min():.2f} kcal/mol'
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"   ✅ Saved: {output_file}")


def create_ir_dashboard(frequencies: np.ndarray,
                       intensities: np.ndarray,
                       dipoles: np.ndarray,
                       autocorr: np.ndarray,
                       energies: np.ndarray,
                       timestep: float,
                       temperature: float,
                       output_file: str,
                       verbose: bool = True):
    """
    Create comprehensive 6-panel IR spectroscopy dashboard.
    
    Args:
        frequencies: IR frequencies in cm⁻¹
        intensities: IR intensities
        dipoles: Dipole moments (n_frames, 3) in Debye
        autocorr: Autocorrelation function
        energies: Energies in Hartree
        timestep: Timestep in fs
        temperature: Temperature in K
        output_file: Output filename
        verbose: Print progress
    """
    if verbose:
        print(f"\n📊 Creating comprehensive dashboard...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Convert units
    time_ps = np.arange(len(dipoles)) * timestep / 1000
    energies_kcal = energies * 627.509
    dipole_magnitude = np.linalg.norm(dipoles, axis=1)
    
    # Panel 1: IR Spectrum (large, top left)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(frequencies, intensities, 'b-', linewidth=2)
    ax1.fill_between(frequencies, intensities, alpha=0.3)
    
    # Find peaks
    peaks, _ = find_peaks(intensities, height=0.1, distance=20)
    peak_freqs = frequencies[peaks]
    peak_ints = intensities[peaks]
    ax1.plot(peak_freqs, peak_ints, 'ro', markersize=6)
    
    # Label top 5 peaks
    sorted_idx = np.argsort(peak_ints)[::-1]
    for i in sorted_idx[:5]:
        ax1.annotate(f'{peak_freqs[i]:.0f}',
                    xy=(peak_freqs[i], peak_ints[i]),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax1.set_xlabel('Frequency (cm⁻¹)', fontweight='bold')
    ax1.set_ylabel('Intensity', fontweight='bold')
    ax1.set_title('IR Spectrum', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Statistics (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    stats_text = f"""
    SIMULATION PARAMETERS
    {'='*30}
    Temperature:     {temperature:.1f} K
    Timestep:        {timestep:.2f} fs
    Total time:      {time_ps[-1]:.2f} ps
    Frames:          {len(dipoles)}
    
    DIPOLE STATISTICS
    {'='*30}
    ⟨μx⟩ = {dipoles[:, 0].mean():8.3f} ± {dipoles[:, 0].std():.3f} D
    ⟨μy⟩ = {dipoles[:, 1].mean():8.3f} ± {dipoles[:, 1].std():.3f} D
    ⟨μz⟩ = {dipoles[:, 2].mean():8.3f} ± {dipoles[:, 2].std():.3f} D
    ⟨|μ|⟩ = {dipole_magnitude.mean():7.3f} ± {dipole_magnitude.std():.3f} D
    
    ENERGY STATISTICS
    {'='*30}
    Mean:    {energies_kcal.mean():.2f} kcal/mol
    Std:     {energies_kcal.std():.2f} kcal/mol
    Range:   {energies_kcal.max()-energies_kcal.min():.2f} kcal/mol
    
    SPECTRUM STATISTICS
    {'='*30}
    Peaks found:     {len(peaks)}
    Freq. range:     0-{frequencies.max():.0f} cm⁻¹
    Resolution:      {frequencies[1]-frequencies[0]:.2f} cm⁻¹
    """
    
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=9, family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Panel 3: Autocorrelation (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    time_autocorr = np.arange(len(autocorr)) * timestep / 1000
    ax3.plot(time_autocorr, autocorr, 'b-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (ps)', fontweight='bold')
    ax3.set_ylabel('C(t)', fontweight='bold')
    ax3.set_title('Dipole Autocorrelation', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Dipole magnitude (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time_ps, dipole_magnitude, 'k-', linewidth=1, alpha=0.7)
    ax4.axhline(y=dipole_magnitude.mean(), color='r', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Time (ps)', fontweight='bold')
    ax4.set_ylabel('|μ| (Debye)', fontweight='bold')
    ax4.set_title('Dipole Magnitude', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Energy (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(time_ps, energies_kcal, 'g-', linewidth=1, alpha=0.7)
    ax5.axhline(y=energies_kcal.mean(), color='r', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Time (ps)', fontweight='bold')
    ax5.set_ylabel('Energy (kcal/mol)', fontweight='bold')
    ax5.set_title('Energy Conservation', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Dipole components (bottom, spanning)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.plot(time_ps, dipoles[:, 0], 'r-', linewidth=1, alpha=0.7, label='μx')
    ax6.plot(time_ps, dipoles[:, 1], 'g-', linewidth=1, alpha=0.7, label='μy')
    ax6.plot(time_ps, dipoles[:, 2], 'b-', linewidth=1, alpha=0.7, label='μz')
    ax6.set_xlabel('Time (ps)', fontweight='bold')
    ax6.set_ylabel('Dipole Component (Debye)', fontweight='bold')
    ax6.set_title('Dipole Components Evolution', fontsize=12, fontweight='bold')
    ax6.legend(loc='upper right', ncol=3)
    ax6.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'IR Spectroscopy Analysis - {temperature:.0f} K', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"   ✅ Saved: {output_file}")


def create_ir_visualizations(frequencies: np.ndarray,
                            intensities: np.ndarray,
                            dipoles: np.ndarray,
                            trajectory: 'TrajectoryData',
                            timestep: float,
                            output_dir: str,
                            verbose: bool = True):
    """
    Create complete set of IR spectroscopy visualizations.
    
    Args:
        frequencies: IR frequencies in cm⁻¹
        intensities: IR intensities
        dipoles: Dipole moments (n_frames, 3) in Debye
        trajectory: Trajectory data
        timestep: Timestep in fs
        output_dir: Output directory
        verbose: Print progress
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("\n" + "=" * 80)
        print("  CREATING IR SPECTROSCOPY VISUALIZATIONS")
        print("=" * 80)
    
    temperature = trajectory.metadata.get('temperature', 300.0)
    
    # Compute autocorrelation
    from ir_spectroscopy import IRSpectrumCalculator
    calc = IRSpectrumCalculator(temperature)
    lags, autocorr = calc.compute_autocorrelation(dipoles, verbose=False)
    
    # Individual plots
    plot_ir_spectrum(
        frequencies, intensities,
        str(output_dir / 'ir_spectrum.png'),
        title=f'IR Spectrum at {temperature:.0f} K',
        verbose=verbose
    )
    
    plot_dipole_autocorrelation(
        lags, autocorr, timestep,
        str(output_dir / 'dipole_autocorrelation.png'),
        verbose=verbose
    )
    
    plot_dipole_evolution(
        dipoles, timestep,
        str(output_dir / 'dipole_evolution.png'),
        verbose=verbose
    )
    
    plot_energy_trajectory(
        trajectory.energies, timestep,
        str(output_dir / 'energy_trajectory.png'),
        verbose=verbose
    )
    
    # Comprehensive dashboard
    create_ir_dashboard(
        frequencies, intensities,
        dipoles, autocorr,
        trajectory.energies, timestep, temperature,
        str(output_dir / 'ir_dashboard.png'),
        verbose=verbose
    )
    
    if verbose:
        print("\n✅ All visualizations complete!")
        print(f"   Files saved in: {output_dir}")


if __name__ == '__main__':
    print("This module provides visualization functions for IR spectroscopy.")
    print("Import and use the functions in your analysis scripts.")
