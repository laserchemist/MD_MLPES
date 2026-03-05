#!/usr/bin/env python3
"""
Analyze MD Trajectory and Dipoles
==================================
Check if dipoles are varying properly during MD simulation.

Author: Jonathan
Date: 2026-01-17
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_trajectory(traj_file):
    """Analyze MD trajectory and dipole predictions."""
    print("\n" + "="*70)
    print("MD TRAJECTORY ANALYSIS")
    print("="*70)
    
    # Load trajectory
    print(f"\nLoading: {traj_file}")
    data = np.load(traj_file)
    
    # Extract data
    coords = data['coordinates']
    dipoles = data['dipoles']
    energies = data['energies']
    temps = data['temperatures']
    
    n_frames = len(coords)
    n_atoms = coords.shape[1]
    
    print(f"\n📊 Trajectory Information:")
    print(f"   Frames: {n_frames}")
    print(f"   Atoms: {n_atoms}")
    print(f"   Coordinates shape: {coords.shape}")
    print(f"   Dipoles shape: {dipoles.shape}")
    
    # Analyze dipoles
    print(f"\n🔍 Dipole Analysis:")
    
    # Dipole magnitude
    dipole_mag = np.linalg.norm(dipoles, axis=1)
    
    print(f"   Mean |μ|: {dipole_mag.mean():.4f} Debye")
    print(f"   Std  |μ|: {dipole_mag.std():.4f} Debye")
    print(f"   Min  |μ|: {dipole_mag.min():.4f} Debye")
    print(f"   Max  |μ|: {dipole_mag.max():.4f} Debye")
    print(f"   Range:    {dipole_mag.max() - dipole_mag.min():.4f} Debye")
    
    # Check if dipoles are varying
    if dipole_mag.std() < 0.01:
        print(f"\n   ⚠️  WARNING: Dipoles are nearly constant!")
        print(f"      Std deviation is very small ({dipole_mag.std():.6f} Debye)")
        print(f"      This will produce a flat IR spectrum")
    else:
        print(f"\n   ✓ Dipoles are varying")
    
    # Dipole components
    print(f"\n   Dipole components:")
    for i, label in enumerate(['X', 'Y', 'Z']):
        mean = dipoles[:, i].mean()
        std = dipoles[:, i].std()
        print(f"      {label}: {mean:7.4f} ± {std:.4f} Debye")
    
    # Energy analysis
    print(f"\n⚡ Energy Analysis:")
    print(f"   Mean: {energies.mean():.6f} Hartree")
    print(f"   Std:  {energies.std():.6f} Hartree")
    print(f"   Min:  {energies.min():.6f} Hartree")
    print(f"   Max:  {energies.max():.6f} Hartree")
    print(f"   Range: {(energies.max()-energies.min())*627.509:.2f} kcal/mol")
    
    # Temperature
    print(f"\n🌡️  Temperature:")
    print(f"   Mean: {temps.mean():.1f} K")
    print(f"   Std:  {temps.std():.1f} K")
    
    # Geometry analysis
    print(f"\n📐 Geometry Analysis:")
    
    # Check if structure is moving
    first_coords = coords[0]
    last_coords = coords[-1]
    displacement = np.linalg.norm(last_coords - first_coords, axis=1)
    
    print(f"   Atom displacements (first → last):")
    for i, disp in enumerate(displacement):
        print(f"      Atom {i}: {disp:.4f} Å")
    
    total_disp = displacement.sum()
    print(f"   Total displacement: {total_disp:.4f} Å")
    
    if total_disp < 0.1:
        print(f"\n   ⚠️  WARNING: Very small displacement!")
        print(f"      MD might not be sampling vibrations properly")
    
    # Create diagnostic plots
    print(f"\n📊 Creating diagnostic plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Dipole magnitude over time
    ax = axes[0, 0]
    ax.plot(dipole_mag, 'b-', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Frame', fontweight='bold')
    ax.set_ylabel('|μ| (Debye)', fontweight='bold')
    ax.set_title('Dipole Magnitude vs Time', fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Plot 2: Dipole components
    ax = axes[0, 1]
    ax.plot(dipoles[:, 0], 'r-', linewidth=0.5, alpha=0.5, label='μx')
    ax.plot(dipoles[:, 1], 'g-', linewidth=0.5, alpha=0.5, label='μy')
    ax.plot(dipoles[:, 2], 'b-', linewidth=0.5, alpha=0.5, label='μz')
    ax.set_xlabel('Frame', fontweight='bold')
    ax.set_ylabel('μ (Debye)', fontweight='bold')
    ax.set_title('Dipole Components', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Energy
    ax = axes[1, 0]
    ax.plot(energies * 627.509, 'purple', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Frame', fontweight='bold')
    ax.set_ylabel('Energy (kcal/mol)', fontweight='bold')
    ax.set_title('Total Energy', fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Plot 4: Dipole histogram
    ax = axes[1, 1]
    ax.hist(dipole_mag, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(dipole_mag.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {dipole_mag.mean():.2f}')
    ax.set_xlabel('|μ| (Debye)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Dipole Magnitude Distribution', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path(traj_file).parent / 'trajectory_diagnostics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_file}")
    plt.close()
    
    # Summary
    print(f"\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    
    issues = []
    
    if dipole_mag.std() < 0.01:
        issues.append("Dipoles are nearly constant (std < 0.01)")
    
    if total_disp < 0.1:
        issues.append("Very small atomic displacements")
    
    if energies.std() * 627.509 < 0.1:
        issues.append("Energy is nearly constant")
    
    if issues:
        print("\n⚠️  ISSUES DETECTED:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print("\n💡 Possible causes:")
        print("   • ML-dipole model predicting constant values")
        print("   • MD not sampling vibrational modes")
        print("   • Starting geometry too close to minimum")
        print("   • Temperature too low for vibrations")
        
        print("\n🔧 Suggested fixes:")
        print("   • Check ML-dipole model predictions on test data")
        print("   • Increase temperature (try 500K)")
        print("   • Add initial velocity/displacement")
        print("   • Run longer simulation")
    else:
        print("\n✅ Trajectory looks reasonable!")
        print("   Dipoles are varying")
        print("   Structure is moving")
        print("   Energy is fluctuating")
    
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage: python3 analyze_md_trajectory.py <trajectory.npz>")
        print("\nExample:")
        print("  python3 analyze_md_trajectory.py ir_spectrum_output/md_trajectory.npz")
        sys.exit(1)
    
    traj_file = sys.argv[1]
    
    if not Path(traj_file).exists():
        print(f"\n❌ File not found: {traj_file}")
        sys.exit(1)
    
    analyze_trajectory(traj_file)
