#!/usr/bin/env python3
"""
Visualize MD Trajectories

Creates publication-quality plots from MD trajectory files.

Usage:
    python3 visualize_trajectory.py trajectory.npz
    python3 visualize_trajectory.py trajectory.npz --output analysis/
    python3 visualize_trajectory.py trajectory.npz --all
"""

import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

try:
    from modules.data_formats import load_trajectory
    from modules.visualization import TrajectoryVisualizer
    FRAMEWORK = True
    print("✅ Framework loaded")
except ImportError as e:
    print(f"❌ Framework import error: {e}")
    print("   Falling back to basic visualization")
    FRAMEWORK = False

def plot_basic_energy(energies, timestep, output_path=None):
    """Basic energy plot if framework not available."""
    time_ps = np.arange(len(energies)) * timestep / 1000  # Convert to ps
    energies_kcal = energies * 627.509
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_ps, energies_kcal, linewidth=1, alpha=0.8)
    plt.xlabel('Time (ps)')
    plt.ylabel('Energy (kcal/mol)')
    plt.title('MD Trajectory - Energy vs Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"   Saved: {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize MD trajectory')
    parser.add_argument('trajectory', help='Trajectory file (.npz, .xyz, .extxyz)')
    parser.add_argument('--output', default=None, help='Output directory for plots')
    parser.add_argument('--all', action='store_true', help='Create all available plots')
    parser.add_argument('--energy', action='store_true', help='Energy trajectory')
    parser.add_argument('--structure', action='store_true', help='3D structure')
    parser.add_argument('--temperature', action='store_true', help='Temperature evolution')
    parser.add_argument('--forces', action='store_true', help='Force magnitudes')
    parser.add_argument('--summary', action='store_true', help='Summary dashboard')
    
    args = parser.parse_args()
    
    # Check file exists
    traj_file = Path(args.trajectory)
    if not traj_file.exists():
        print(f"❌ Trajectory file not found: {traj_file}")
        sys.exit(1)
    
    print("=" * 80)
    print("  TRAJECTORY VISUALIZATION")
    print("=" * 80)
    
    print(f"\n📂 Loading: {traj_file}")
    
    if not FRAMEWORK:
        # Basic fallback
        print("⚠️  Using basic visualization (framework not available)")
        try:
            data = np.load(traj_file, allow_pickle=True)
            energies = data['energies']
            
            # Get timestep from metadata if available
            metadata = data.get('metadata', None)
            if metadata and hasattr(metadata, 'item'):
                metadata = metadata.item()
                timestep = metadata.get('timestep_fs', 0.5)
            else:
                timestep = 0.5  # Default
            
            output_path = Path(args.output) / 'energy_trajectory.png' if args.output else 'energy_trajectory.png'
            plot_basic_energy(energies, timestep, output_path)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
        
        return
    
    # Load trajectory with framework
    try:
        trajectory = load_trajectory(str(traj_file))
        print(f"   ✅ Loaded {trajectory.n_frames} frames")
        print(f"   Molecule: {' '.join(trajectory.symbols)} ({trajectory.n_atoms} atoms)")
    except Exception as e:
        print(f"❌ Error loading trajectory: {e}")
        sys.exit(1)
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = traj_file.parent / f"{traj_file.stem}_analysis"
        output_dir.mkdir(exist_ok=True)
    
    print(f"\n📊 Output directory: {output_dir}")
    
    # Create visualizer
    viz = TrajectoryVisualizer(trajectory, output_dir=str(output_dir))
    
    # Determine what to plot
    plot_all = args.all
    if not (args.energy or args.structure or args.temperature or args.forces or args.summary or plot_all):
        # Default: energy + summary
        args.energy = True
        args.summary = True
    
    print(f"\n📈 Creating plots...")
    
    # Energy trajectory
    if args.energy or plot_all:
        print("\n   Creating energy trajectory plot...")
        try:
            fig = viz.plot_energy_trajectory(unit='kcal/mol', save=True)
            print(f"   ✅ energy_trajectory.png")
            plt.close(fig)
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Temperature evolution (if available)
    if (args.temperature or plot_all) and hasattr(trajectory, 'temperatures'):
        print("\n   Creating temperature plot...")
        try:
            fig = viz.plot_temperature(save=True)
            print(f"   ✅ temperature.png")
            plt.close(fig)
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Force magnitudes (if available)
    if (args.forces or plot_all) and trajectory.forces is not None:
        print("\n   Creating force plot...")
        try:
            fig = viz.plot_forces(save=True)
            print(f"   ✅ forces.png")
            plt.close(fig)
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # 3D structure
    if args.structure or plot_all:
        print("\n   Creating 3D structure visualization...")
        try:
            # Plot first, middle, and last frames
            frames = [0, len(trajectory.coordinates) // 2, len(trajectory.coordinates) - 1]
            for i, frame_idx in enumerate(frames):
                fig = viz.plot_3d_structure(frame=frame_idx, save=True)
                print(f"   ✅ structure_frame_{frame_idx}.png")
                plt.close(fig)
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary dashboard
    if args.summary or plot_all:
        print("\n   Creating summary dashboard...")
        try:
            fig = viz.plot_summary_dashboard(save=True)
            print(f"   ✅ summary_dashboard.png")
            plt.close(fig)
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("  TRAJECTORY STATISTICS")
    print("=" * 80)
    
    energies_kcal = trajectory.energies * 627.509
    
    print(f"\n📊 Basic statistics:")
    print(f"   Frames: {trajectory.n_frames}")
    print(f"   Atoms: {trajectory.n_atoms}")
    print(f"   Molecule: {' '.join(trajectory.symbols)}")
    
    # Get metadata
    metadata = trajectory.metadata if hasattr(trajectory, 'metadata') and trajectory.metadata else {}
    
    if 'timestep_fs' in metadata:
        timestep = metadata['timestep_fs']
        total_time = trajectory.n_frames * timestep / 1000
        print(f"   Timestep: {timestep} fs")
        print(f"   Total time: {total_time:.2f} ps")
    
    if 'temperature_target' in metadata:
        print(f"   Target temperature: {metadata['temperature_target']} K")
        if 'temperature_mean' in metadata:
            print(f"   Mean temperature: {metadata['temperature_mean']:.1f} ± {metadata.get('temperature_std', 0):.1f} K")
    
    print(f"\n⚡ Energy statistics:")
    print(f"   Mean: {energies_kcal.mean():.2f} kcal/mol")
    print(f"   Std: {energies_kcal.std():.2f} kcal/mol")
    print(f"   Min: {energies_kcal.min():.2f} kcal/mol")
    print(f"   Max: {energies_kcal.max():.2f} kcal/mol")
    print(f"   Range: {energies_kcal.max() - energies_kcal.min():.2f} kcal/mol")
    
    if trajectory.forces is not None:
        force_magnitudes = np.linalg.norm(trajectory.forces, axis=-1) * 627.509
        print(f"\n💪 Force statistics:")
        print(f"   Mean magnitude: {force_magnitudes.mean():.2f} kcal/mol/Å")
        print(f"   Max magnitude: {force_magnitudes.max():.2f} kcal/mol/Å")
    
    print("\n" + "=" * 80)
    print("  VISUALIZATION COMPLETE")
    print("=" * 80)
    
    print(f"\n✅ All plots saved to: {output_dir}")
    print(f"\n💡 You can now:")
    print(f"   • View plots in {output_dir}/")
    print(f"   • Create custom plots with TrajectoryVisualizer")
    print(f"   • Further analyze with load_trajectory('{traj_file}')")

if __name__ == '__main__':
    main()
