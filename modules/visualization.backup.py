#!/usr/bin/env python3
"""
Visualization Module for PSI4-MD Framework

Provides publication-quality visualizations for:
- Energy and force trajectories
- Molecular structures
- Dipole moments
- Temperature and dynamics properties
- ML model training curves

All plots are well-labeled with units and publication styling.

Classes:
    TrajectoryVisualizer: Main visualization class
    
Functions:
    plot_energy_trajectory: Plot energy vs time
    plot_forces: Plot force magnitudes
    plot_3d_structure: 3D molecular structure
    plot_training_curves: ML training progress
    
Author: PSI4-MD Framework
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings

# Try to import seaborn for better styling
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from data_formats import TrajectoryData

# Physical constants for unit conversions
HARTREE_TO_KCAL = 627.509474
HARTREE_TO_EV = 27.2114
BOHR_TO_ANGSTROM = 0.529177


class TrajectoryVisualizer:
    """
    Main class for visualizing MD trajectories and analysis results.
    """
    
    def __init__(self, trajectory: TrajectoryData, output_dir: str = None):
        """
        Initialize visualizer.
        
        Args:
            trajectory: Trajectory data to visualize
            output_dir: Directory for saving plots
        """
        self.trajectory = trajectory
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set publication-quality defaults
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'sans-serif',
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.figsize': (10, 6),
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'lines.linewidth': 2,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    
    def plot_energy_trajectory(self, unit: str = 'kcal/mol',
                               show_reference: bool = True,
                               save: bool = True) -> Figure:
        """
        Plot energy vs time trajectory.
        
        Args:
            unit: Energy unit ('hartree', 'kcal/mol', 'ev')
            show_reference: Show reference line at mean energy
            save: Save figure to output directory
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert energies to requested unit
        if unit.lower() == 'kcal/mol':
            energies = self.trajectory.energies * HARTREE_TO_KCAL
            ylabel = 'Energy (kcal/mol)'
        elif unit.lower() == 'ev':
            energies = self.trajectory.energies * HARTREE_TO_EV
            ylabel = 'Energy (eV)'
        else:
            energies = self.trajectory.energies
            ylabel = 'Energy (Hartree)'
        
        # Get time array
        if self.trajectory.times is not None:
            times = self.trajectory.times
            xlabel = 'Time (fs)'
        else:
            times = np.arange(self.trajectory.n_frames)
            xlabel = 'Frame'
        
        # Plot energy
        ax.plot(times, energies, 'b-', linewidth=2, label='Total Energy')
        
        # Add reference line
        if show_reference:
            mean_energy = np.mean(energies)
            ax.axhline(mean_energy, color='r', linestyle='--', linewidth=1.5,
                      label=f'Mean: {mean_energy:.4f}')
        
        # Labels and formatting
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        
        # Add title with metadata
        method = self.trajectory.metadata.get('method', 'Unknown')
        basis = self.trajectory.metadata.get('basis', 'Unknown')
        molecule = self.trajectory.metadata.get('molecule', 'Unknown')
        ax.set_title(f'Energy Trajectory: {molecule} ({method}/{basis})',
                    fontsize=16, fontweight='bold')
        
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / 'energy_trajectory.png'
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved energy trajectory plot to {output_file}")
        
        return fig
    
    def plot_force_magnitudes(self, atom_indices: Optional[List[int]] = None,
                             save: bool = True) -> Figure:
        """
        Plot force magnitudes over time.
        
        Args:
            atom_indices: Specific atoms to plot (None for all)
            save: Save figure to output directory
            
        Returns:
            Matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Calculate force magnitudes
        force_mags = np.linalg.norm(self.trajectory.forces, axis=2)  # (n_frames, n_atoms)
        
        # Convert to kcal/mol/Å
        force_mags_kcal = force_mags * HARTREE_TO_KCAL
        
        # Get time array
        if self.trajectory.times is not None:
            times = self.trajectory.times
            xlabel = 'Time (fs)'
        else:
            times = np.arange(self.trajectory.n_frames)
            xlabel = 'Frame'
        
        # Plot individual atom forces
        if atom_indices is None:
            atom_indices = range(min(self.trajectory.n_atoms, 10))  # Limit to 10 atoms
        
        for idx in atom_indices:
            ax1.plot(times, force_mags_kcal[:, idx],
                    label=f'{self.trajectory.symbols[idx]}{idx+1}',
                    alpha=0.7, linewidth=1.5)
        
        ax1.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax1.set_ylabel('Force Magnitude (kcal/mol/Å)', fontsize=12, fontweight='bold')
        ax1.set_title('Individual Atom Forces', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', ncol=2, frameon=True)
        ax1.grid(True, alpha=0.3)
        
        # Plot statistics: max, mean, min forces
        max_forces = np.max(force_mags_kcal, axis=1)
        mean_forces = np.mean(force_mags_kcal, axis=1)
        min_forces = np.min(force_mags_kcal, axis=1)
        
        ax2.plot(times, max_forces, 'r-', label='Maximum', linewidth=2)
        ax2.plot(times, mean_forces, 'b-', label='Mean', linewidth=2)
        ax2.plot(times, min_forces, 'g-', label='Minimum', linewidth=2)
        ax2.fill_between(times, min_forces, max_forces, alpha=0.2)
        
        ax2.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax2.set_ylabel('Force Magnitude (kcal/mol/Å)', fontsize=12, fontweight='bold')
        ax2.set_title('Force Statistics', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', frameon=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / 'force_magnitudes.png'
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved force magnitudes plot to {output_file}")
        
        return fig
    
    def plot_dipole_trajectory(self, save: bool = True) -> Optional[Figure]:
        """
        Plot dipole moment trajectory.
        
        Args:
            save: Save figure to output directory
            
        Returns:
            Matplotlib Figure object or None if no dipole data
        """
        if self.trajectory.dipoles is None:
            warnings.warn("No dipole data available in trajectory")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Get time array
        if self.trajectory.times is not None:
            times = self.trajectory.times
            xlabel = 'Time (fs)'
        else:
            times = np.arange(self.trajectory.n_frames)
            xlabel = 'Frame'
        
        # Plot components
        ax1.plot(times, self.trajectory.dipoles[:, 0], 'r-', label='μₓ', linewidth=2)
        ax1.plot(times, self.trajectory.dipoles[:, 1], 'g-', label='μᵧ', linewidth=2)
        ax1.plot(times, self.trajectory.dipoles[:, 2], 'b-', label='μz', linewidth=2)
        
        ax1.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax1.set_ylabel('Dipole Moment (Debye)', fontsize=12, fontweight='bold')
        ax1.set_title('Dipole Moment Components', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', frameon=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        
        # Plot magnitude
        dipole_mag = np.linalg.norm(self.trajectory.dipoles, axis=1)
        
        ax2.plot(times, dipole_mag, 'purple', linewidth=2.5, label='|μ|')
        ax2.axhline(np.mean(dipole_mag), color='orange', linestyle='--',
                   linewidth=1.5, label=f'Mean: {np.mean(dipole_mag):.3f} D')
        
        ax2.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax2.set_ylabel('|μ| (Debye)', fontsize=12, fontweight='bold')
        ax2.set_title('Dipole Moment Magnitude', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', frameon=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / 'dipole_trajectory.png'
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved dipole trajectory plot to {output_file}")
        
        return fig
    
    def plot_3d_structure(self, frame_index: int = 0,
                         show_forces: bool = True,
                         force_scale: float = 1.0,
                         save: bool = True) -> Figure:
        """
        Plot 3D molecular structure with optional forces.
        
        Args:
            frame_index: Frame to visualize
            show_forces: Show force vectors
            force_scale: Scaling factor for force vectors
            save: Save figure to output directory
            
        Returns:
            Matplotlib Figure object
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get frame data
        coords = self.trajectory.coordinates[frame_index]
        forces = self.trajectory.forces[frame_index]
        
        # Atomic colors (CPK coloring)
        colors = {
            'H': 'white', 'C': 'gray', 'N': 'blue', 'O': 'red',
            'F': 'green', 'S': 'yellow', 'Cl': 'green', 'Br': 'brown'
        }
        
        # Atomic sizes (van der Waals radii)
        sizes = {
            'H': 120, 'C': 170, 'N': 155, 'O': 152,
            'F': 147, 'S': 180, 'Cl': 175, 'Br': 185
        }
        
        # Plot atoms
        for i, (symbol, coord) in enumerate(zip(self.trajectory.symbols, coords)):
            color = colors.get(symbol, 'gray')
            size = sizes.get(symbol, 150)
            
            ax.scatter(coord[0], coord[1], coord[2],
                      c=color, s=size, edgecolors='black',
                      linewidths=1.5, alpha=0.9,
                      label=symbol if i == 0 or symbol not in self.trajectory.symbols[:i] else '')
        
        # Plot bonds (simple distance-based)
        bond_threshold = 1.8  # Angstroms
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < bond_threshold:
                    ax.plot([coords[i, 0], coords[j, 0]],
                           [coords[i, 1], coords[j, 1]],
                           [coords[i, 2], coords[j, 2]],
                           'k-', linewidth=2, alpha=0.6)
        
        # Plot forces as arrows
        if show_forces:
            for i, (coord, force) in enumerate(zip(coords, forces)):
                # Force in kcal/mol/Å
                force_vec = -force * HARTREE_TO_KCAL * force_scale
                
                ax.quiver(coord[0], coord[1], coord[2],
                         force_vec[0], force_vec[1], force_vec[2],
                         color='red', arrow_length_ratio=0.3,
                         linewidth=2, alpha=0.7)
        
        # Labels and formatting
        ax.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (Å)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (Å)', fontsize=12, fontweight='bold')
        
        molecule = self.trajectory.metadata.get('molecule', 'Unknown')
        energy = self.trajectory.energies[frame_index] * HARTREE_TO_KCAL
        
        ax.set_title(f'{molecule} - Frame {frame_index}\nE = {energy:.2f} kcal/mol',
                    fontsize=14, fontweight='bold')
        
        ax.legend(loc='best', frameon=True)
        
        # Equal aspect ratio
        max_range = np.array([coords[:, 0].max() - coords[:, 0].min(),
                             coords[:, 1].max() - coords[:, 1].min(),
                             coords[:, 2].max() - coords[:, 2].min()]).max() / 2.0
        
        mid_x = (coords[:, 0].max() + coords[:, 0].min()) * 0.5
        mid_y = (coords[:, 1].max() + coords[:, 1].min()) * 0.5
        mid_z = (coords[:, 2].max() + coords[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / f'structure_frame_{frame_index}.png'
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved 3D structure plot to {output_file}")
        
        return fig
    
    def plot_summary_dashboard(self, save: bool = True) -> Figure:
        """
        Create a comprehensive summary dashboard.
        
        Args:
            save: Save figure to output directory
            
        Returns:
            Matplotlib Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Get time array
        if self.trajectory.times is not None:
            times = self.trajectory.times
            xlabel = 'Time (fs)'
        else:
            times = np.arange(self.trajectory.n_frames)
            xlabel = 'Frame'
        
        # 1. Energy trajectory
        ax1 = fig.add_subplot(gs[0, :])
        energies_kcal = self.trajectory.energies * HARTREE_TO_KCAL
        ax1.plot(times, energies_kcal, 'b-', linewidth=2)
        ax1.set_xlabel(xlabel, fontweight='bold')
        ax1.set_ylabel('Energy (kcal/mol)', fontweight='bold')
        ax1.set_title('Energy Trajectory', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Force statistics
        ax2 = fig.add_subplot(gs[1, 0])
        force_mags = np.linalg.norm(self.trajectory.forces, axis=2) * HARTREE_TO_KCAL
        max_forces = np.max(force_mags, axis=1)
        mean_forces = np.mean(force_mags, axis=1)
        
        ax2.plot(times, max_forces, 'r-', label='Max', linewidth=1.5)
        ax2.plot(times, mean_forces, 'b-', label='Mean', linewidth=1.5)
        ax2.fill_between(times, 0, max_forces, alpha=0.2)
        ax2.set_xlabel(xlabel, fontweight='bold')
        ax2.set_ylabel('Force (kcal/mol/Å)', fontweight='bold')
        ax2.set_title('Force Magnitudes', fontsize=14, fontweight='bold')
        ax2.legend(frameon=True)
        ax2.grid(True, alpha=0.3)
        
        # 3. Energy distribution
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(energies_kcal, bins=30, edgecolor='black', alpha=0.7)
        ax3.axvline(np.mean(energies_kcal), color='r', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(energies_kcal):.2f}')
        ax3.set_xlabel('Energy (kcal/mol)', fontweight='bold')
        ax3.set_ylabel('Count', fontweight='bold')
        ax3.set_title('Energy Distribution', fontsize=14, fontweight='bold')
        ax3.legend(frameon=True)
        ax3.grid(True, alpha=0.3)
        
        # 4. Dipole moments (if available)
        if self.trajectory.dipoles is not None:
            ax4 = fig.add_subplot(gs[2, :])
            dipole_mag = np.linalg.norm(self.trajectory.dipoles, axis=1)
            ax4.plot(times, dipole_mag, 'purple', linewidth=2)
            ax4.axhline(np.mean(dipole_mag), color='orange', linestyle='--',
                       linewidth=1.5, label=f'Mean: {np.mean(dipole_mag):.3f} D')
            ax4.set_xlabel(xlabel, fontweight='bold')
            ax4.set_ylabel('|μ| (Debye)', fontweight='bold')
            ax4.set_title('Dipole Moment Magnitude', fontsize=14, fontweight='bold')
            ax4.legend(frameon=True)
            ax4.grid(True, alpha=0.3)
        
        # Add overall title
        molecule = self.trajectory.metadata.get('molecule', 'Unknown')
        method = self.trajectory.metadata.get('method', 'Unknown')
        basis = self.trajectory.metadata.get('basis', 'Unknown')
        
        fig.suptitle(f'MD Trajectory Summary: {molecule} ({method}/{basis})',
                    fontsize=18, fontweight='bold', y=0.995)
        
        if save:
            output_file = self.output_dir / 'trajectory_summary.png'
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved summary dashboard to {output_file}")
        
        return fig


def plot_training_curves(training_history: Dict, save_path: Optional[str] = None) -> Figure:
    """
    Plot ML model training curves.
    
    Args:
        training_history: Dictionary with 'train_loss' and 'val_loss' arrays
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = np.arange(1, len(training_history['train_loss']) + 1)
    
    # Training and validation loss
    ax1.plot(epochs, training_history['train_loss'], 'b-o',
            label='Training Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, training_history['val_loss'], 'r-s',
            label='Validation Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss (RMSE in kcal/mol)', fontsize=14, fontweight='bold')
    ax1.set_title('Training Progress', fontsize=16, fontweight='bold')
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Learning curve (log-log)
    ax2.loglog(epochs, training_history['train_loss'], 'b-o',
              label='Training', linewidth=2, markersize=6)
    ax2.loglog(epochs, training_history['val_loss'], 'r-s',
              label='Validation', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Loss (log scale)', fontsize=14, fontweight='bold')
    ax2.set_title('Learning Curve (Log-Log)', fontsize=16, fontweight='bold')
    ax2.legend(frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    return fig


if __name__ == "__main__":
    print("📊 Visualization Module Demonstration")
    print("=" * 60)
    
    # Create test trajectory
    n_frames = 100
    n_atoms = 3
    
    test_trajectory = TrajectoryData(
        symbols=['O', 'H', 'H'],
        coordinates=np.random.randn(n_frames, n_atoms, 3) * 0.05 + 
                   np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
        energies=np.random.randn(n_frames) * 0.001 - 76.0 + 0.01 * np.sin(np.arange(n_frames) * 0.1),
        forces=np.random.randn(n_frames, n_atoms, 3) * 0.01,
        dipoles=np.random.randn(n_frames, 3) * 0.1 + np.array([0.5, 0.5, 0]),
        times=np.arange(n_frames) * 0.5,
        metadata={'molecule': 'water', 'method': 'B3LYP', 'basis': '6-31G*'}
    )
    
    # Create visualizer
    output_dir = '/home/claude/psi4md_framework/outputs/visualizations'
    visualizer = TrajectoryVisualizer(test_trajectory, output_dir)
    
    print("\n📈 Creating visualizations...")
    
    # Energy trajectory
    print("\n1. Energy trajectory...")
    visualizer.plot_energy_trajectory(unit='kcal/mol', save=True)
    
    # Force magnitudes
    print("2. Force magnitudes...")
    visualizer.plot_force_magnitudes(save=True)
    
    # Dipole trajectory
    print("3. Dipole trajectory...")
    visualizer.plot_dipole_trajectory(save=True)
    
    # 3D structure
    print("4. 3D molecular structure...")
    visualizer.plot_3d_structure(frame_index=0, save=True)
    
    # Summary dashboard
    print("5. Summary dashboard...")
    visualizer.plot_summary_dashboard(save=True)
    
    # Training curves example
    print("6. ML training curves...")
    mock_training = {
        'train_loss': np.exp(-np.linspace(0, 3, 50)) * 10 + np.random.randn(50) * 0.1,
        'val_loss': np.exp(-np.linspace(0, 2.8, 50)) * 10 + np.random.randn(50) * 0.15
    }
    plot_training_curves(mock_training, 
                        save_path=str(Path(output_dir) / 'training_curves.png'))
    
    print(f"\n✅ All visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    for file in Path(output_dir).glob('*.png'):
        print(f"  - {file.name}")
