#!/usr/bin/env python3
"""
Simple Production MD Script

Use your refined ML-PES model to run production molecular dynamics!

Usage:
    python3 simple_production_md.py \
        --model outputs/.../retrained_mlpes_model.pkl \
        --training-data outputs/.../augmented_training_data.npz \
        --temp 300 \
        --steps 10000 \
        --output production_md_300K.npz
"""

import sys
import numpy as np
import pickle
from pathlib import Path
import argparse
from datetime import datetime

try:
    from tqdm import tqdm
    TQDM = True
except:
    TQDM = False
    def tqdm(x, **kwargs):
        return x

print("=" * 80)
print("  SIMPLE PRODUCTION MD WITH ML-PES")
print("=" * 80)

# Try to import framework
try:
    from modules.data_formats import load_trajectory, save_trajectory, TrajectoryData
    print("✅ Framework loaded")
except ImportError:
    print("❌ Framework not found - make sure you're in MD_MLPES directory")
    sys.exit(1)

# Try to import two_phase_workflow for MLPESPredictor
try:
    from two_phase_workflow import MLPESPredictor
    print("✅ ML-PES predictor available")
except ImportError:
    print("⚠️  MLPESPredictor not found - will use basic prediction")
    MLPESPredictor = None

def main():
    parser = argparse.ArgumentParser(description='Simple production MD with ML-PES')
    parser.add_argument('--model', required=True, help='Refined ML-PES model (.pkl)')
    parser.add_argument('--training-data', required=True, help='Training data for starting geometry')
    parser.add_argument('--temp', type=float, default=300.0, help='Temperature (K)')
    parser.add_argument('--steps', type=int, default=10000, help='Number of MD steps')
    parser.add_argument('--timestep', type=float, default=0.5, help='Timestep (fs)')
    parser.add_argument('--save-every', type=int, default=10, help='Save interval')
    parser.add_argument('--output', default=None, help='Output file (.npz)')
    
    args = parser.parse_args()
    
    # Load model
    print(f"\n📂 Loading ML-PES model: {args.model}")
    with open(args.model, 'rb') as f:
        model_data = pickle.load(f)
    
    metadata = model_data.get('metadata', {})
    model_info = metadata.get('model', {})
    
    print(f"   ✅ Loaded: {model_data.get('version', 'unknown')}")
    print(f"   Theory: {metadata.get('theory', {}).get('method', 'unknown')}/{metadata.get('theory', {}).get('basis', 'unknown')}")
    print(f"   RMSE: {model_info.get('test_rmse_kcal', 'N/A')} kcal/mol")
    print(f"   Molecule: {' '.join(model_data['symbols'])}")
    
    # Load training data for starting geometry
    print(f"\n📊 Loading training data: {args.training_data}")
    training_data = load_trajectory(args.training_data)
    
    # Use lowest energy configuration as starting point
    lowest_idx = np.argmin(training_data.energies)
    initial_coords = training_data.coordinates[lowest_idx].copy()
    
    print(f"   ✅ Using configuration {lowest_idx} (lowest energy)")
    print(f"   Starting energy: {training_data.energies[lowest_idx]*627.509:.2f} kcal/mol")
    
    # Setup MD
    print(f"\n⚙️  MD Configuration:")
    print(f"   Temperature: {args.temp} K")
    print(f"   Steps: {args.steps}")
    print(f"   Timestep: {args.timestep} fs")
    print(f"   Total time: {args.steps * args.timestep / 1000:.1f} ps")
    print(f"   Save interval: {args.save_every} steps")
    
    # Atomic masses (amu)
    mass_dict = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998}
    masses = np.array([mass_dict[s] for s in model_data['symbols']])
    
    # Initialize predictor
    if MLPESPredictor:
        predictor = MLPESPredictor(args.model)
        print(f"\n✅ Using MLPESPredictor")
    else:
        # Fallback: direct prediction
        print(f"\n⚠️  Using fallback prediction")
        from two_phase_workflow import compute_coulomb_matrix
        
        def predict_energy(coords):
            desc = compute_coulomb_matrix(model_data['symbols'], coords)
            desc_scaled = model_data['scaler_X'].transform(desc.reshape(1, -1))
            e_scaled = model_data['model'].predict(desc_scaled)[0]
            return model_data['scaler_y'].inverse_transform([[e_scaled]])[0, 0]
        
        def predict_forces(coords, delta=0.001):
            """Numerical forces."""
            forces = np.zeros_like(coords)
            e0 = predict_energy(coords)
            for i in range(len(coords)):
                for j in range(3):
                    coords_plus = coords.copy()
                    coords_plus[i, j] += delta
                    e_plus = predict_energy(coords_plus)
                    forces[i, j] = -(e_plus - e0) / delta
            return forces
        
        class SimplePred:
            def predict_energy(self, coords):
                return predict_energy(coords)
            def predict_forces(self, coords):
                return predict_forces(coords)
        
        predictor = SimplePred()
    
    # Initialize velocities (Maxwell-Boltzmann)
    kb = 3.1668114e-6  # Hartree/K
    velocities = np.random.randn(len(initial_coords), 3)
    velocities *= np.sqrt(kb * args.temp / masses[:, None])
    
    # Remove center of mass velocity
    velocities -= velocities.mean(axis=0)
    
    # MD trajectory
    coords = initial_coords.copy()
    trajectory = []
    energies = []
    forces_traj = []
    temperatures = []
    
    print(f"\n🚀 Starting MD simulation...")
    
    for step in tqdm(range(args.steps), desc="MD Steps"):
        # Velocity Verlet integration
        # Step 1: v(t + dt/2)
        forces = predictor.predict_forces(coords)
        velocities += 0.5 * forces / masses[:, None] * args.timestep
        
        # Step 2: x(t + dt)
        coords += velocities * args.timestep
        
        # Step 3: v(t + dt)
        forces = predictor.predict_forces(coords)
        velocities += 0.5 * forces / masses[:, None] * args.timestep
        
        # Temperature control (velocity rescaling)
        ke = 0.5 * (masses[:, None] * velocities**2).sum()
        current_T = 2 * ke / (3 * len(coords) * kb)
        
        if step % 10 == 0:  # Rescale every 10 steps
            scale = np.sqrt(args.temp / current_T)
            velocities *= scale
            current_T = args.temp
        
        # Save
        if step % args.save_every == 0:
            trajectory.append(coords.copy())
            energy = predictor.predict_energy(coords)
            energies.append(energy)
            forces_traj.append(forces.copy())
            temperatures.append(current_T)
    
    print(f"\n✅ MD simulation complete!")
    
    # Save trajectory
    if args.output is None:
        args.output = f"production_md_{args.temp}K_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    
    traj_data = TrajectoryData(
        symbols=model_data['symbols'],
        coordinates=np.array(trajectory),
        energies=np.array(energies),
        forces=np.array(forces_traj),
        metadata={
            'production_md': True,
            'model': args.model,
            'temperature_target': args.temp,
            'temperature_mean': np.mean(temperatures),
            'temperature_std': np.std(temperatures),
            'timestep_fs': args.timestep,
            'n_steps': args.steps,
            'total_time_ps': args.steps * args.timestep / 1000,
            'theory': metadata.get('theory', {})
        }
    )
    
    save_trajectory(traj_data, args.output)
    
    print(f"\n💾 Saved trajectory: {args.output}")
    print(f"\n📊 Statistics:")
    print(f"   Frames: {len(trajectory)}")
    print(f"   Total time: {args.steps * args.timestep / 1000:.1f} ps")
    print(f"   Temperature: {np.mean(temperatures):.1f} ± {np.std(temperatures):.1f} K")
    
    energies_kcal = np.array(energies) * 627.509
    print(f"   Energy range: {energies_kcal.min():.1f} to {energies_kcal.max():.1f} kcal/mol")
    print(f"   Energy std: {energies_kcal.std():.2f} kcal/mol")
    
    forces_kcal = np.array(forces_traj) * 627.509
    force_magnitudes = np.linalg.norm(forces_kcal, axis=-1).mean(axis=-1)
    print(f"   Force magnitude: {force_magnitudes.mean():.2f} ± {force_magnitudes.std():.2f} kcal/mol/Å")
    
    print(f"\n💡 You can now:")
    print(f"   • Visualize: python3 visualize_trajectory.py {args.output}")
    print(f"   • Analyze: Load {args.output} with load_trajectory()")
    print(f"   • Continue: Run more MD starting from final geometry")

if __name__ == '__main__':
    main()
