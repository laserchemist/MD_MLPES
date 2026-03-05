#!/usr/bin/env python3
"""
Complete IR Spectrum Workflow
==============================
Train ML-dipole model, run MD on ML-PES, compute IR spectrum.

This script integrates:
1. ML-PES (fast dynamics)
2. ML-dipole surface (fast dipole prediction)
3. IR spectrum from dipole autocorrelation

Usage:
    # Train dipole model
    python compute_ir_workflow.py --train-dipole --training-data data.npz

    # Compute IR spectrum
    python compute_ir_workflow.py \\
        --ml-pes model.pkl \\
        --dipole-model dipole.pkl \\
        --temp 300 --steps 50000

Author: Jonathan
Date: 2026-01-15
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    from ml_dipole_surface import DipoleSurfaceModel, train_dipole_surface_from_file
    from compute_ir_spectrum import IRSpectrumCalculator, run_md_with_dipole_prediction
except ImportError:
    print("Error: Required modules not found")
    print("Make sure ml_dipole_surface.py and compute_ir_spectrum.py are in the same directory")
    sys.exit(1)


def train_dipole_workflow(training_data_path, output_model='dipole_surface.pkl'):
    """
    Train ML dipole surface from training data.
    
    Parameters
    ----------
    training_data_path : str
        Path to training_data.npz with dipoles
    output_model : str
        Where to save trained model
    """
    print("\n" + "="*80)
    print("WORKFLOW STEP 1: Train ML Dipole Surface")
    print("="*80)
    
    model = train_dipole_surface_from_file(training_data_path, output_model)
    
    print("\n✅ Dipole model training complete!")
    print(f"   Model saved: {output_model}")
    print(f"   Test MAE: {model.training_stats['magnitude_test_mae']:.4f} Debye")
    
    return model


def compute_ir_workflow(ml_pes_path, dipole_model_path, 
                       training_data_path=None,
                       temperature=300.0, n_steps=50000,
                       timestep=0.5, output_frequency=10,
                       output_dir='ir_spectrum_output'):
    """
    Compute IR spectrum using ML-PES and ML-dipole.
    
    Parameters
    ----------
    ml_pes_path : str
        Path to trained ML-PES model
    dipole_model_path : str
        Path to trained dipole surface model
    training_data_path : str, optional
        Path to get initial structure
    temperature : float
        MD temperature in K
    n_steps : int
        Number of MD steps
    timestep : float
        MD timestep in fs
    output_frequency : int
        Save every N steps
    output_dir : str
        Output directory
    """
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
    import pickle
    with open(ml_pes_path, 'rb') as f:
        ml_pes_model = pickle.load(f)
    print("✓ ML-PES loaded")
    
    print(f"\nLoading dipole model: {dipole_model_path}")
    dipole_model = DipoleSurfaceModel.load(dipole_model_path)
    print("✓ Dipole model loaded")
    
    # Get initial structure
    if training_data_path:
        from modules.data_formats import load_trajectory
        print(f"\nLoading initial structure from: {training_data_path}")
        data = load_trajectory(training_data_path)
        initial_coords = data.coordinates[0]
        symbols = data.symbols
    else:
        # Use test molecule
        from modules.test_molecules import get_molecule
        print("\nUsing water molecule as initial structure")
        mol = get_molecule('water')
        initial_coords = mol.coordinates
        symbols = mol.symbols
    
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
    time_ps = np.arange(len(acf)) * timestep * output_frequency / 1000  # Convert to ps
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
    for i, peak in enumerate(peaks[:10], 1):  # Top 10 peaks
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train dipole model
  python compute_ir_workflow.py --train-dipole \\
      --training-data training_data.npz

  # Compute IR spectrum
  python compute_ir_workflow.py \\
      --ml-pes model.pkl \\
      --dipole-model dipole_surface.pkl \\
      --temp 300 --steps 50000

  # Full workflow
  python compute_ir_workflow.py \\
      --train-dipole \\
      --training-data training_data.npz \\
      --ml-pes model.pkl \\
      --temp 300 --steps 50000
        """
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
