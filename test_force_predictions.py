#!/usr/bin/env python3
"""
Test ML-PES Force Predictions
==============================
Check if forces are actually being computed correctly.

Author: Jonathan
Date: 2026-01-17
"""

import numpy as np
import pickle
from pathlib import Path
import sys


def test_force_predictions(model_path, dipole_model_path):
    """Test that ML-PES forces are non-zero and reasonable."""
    print("\n" + "="*70)
    print("FORCE PREDICTION TEST")
    print("="*70)
    
    # Load models (need the wrapper)
    sys.path.insert(0, str(Path(__file__).parent))
    
    print(f"\nLoading ML-PES: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Load dipole model for geometry
    print(f"Loading dipole model: {dipole_model_path}")
    with open(dipole_model_path, 'rb') as f:
        dipole_data = pickle.load(f)
    
    symbols = dipole_data['symbols']
    print(f"Molecule: {' '.join(symbols)}")
    
    # Test geometry (ammonia pyramidal)
    coords = np.array([
        [0.0, 0.0, 0.0],      # N
        [0.0, 0.94, 0.38],    # H
        [0.81, -0.47, 0.38],  # H
        [-0.81, -0.47, 0.38]  # H
    ])
    
    print(f"\nTest geometry:")
    for i, (s, c) in enumerate(zip(symbols, coords)):
        print(f"  {s}: [{c[0]:7.4f}, {c[1]:7.4f}, {c[2]:7.4f}]")
    
    # Create wrapper
    from ml_pes_wrapper import MLPESModelWrapper
    wrapper = MLPESModelWrapper(model_data)
    
    # Test 1: Energy prediction
    print(f"\n{'='*70}")
    print("TEST 1: Energy Prediction")
    print("="*70)
    
    energy = wrapper.predict(coords)
    print(f"Energy: {energy:.6f} Hartree = {energy*627.509:.2f} kcal/mol")
    
    # Test 2: Force prediction
    print(f"\n{'='*70}")
    print("TEST 2: Force Prediction")
    print("="*70)
    
    energy_with_f, forces = wrapper.predict_with_forces(coords)
    
    print(f"Energy (with forces): {energy_with_f:.6f} Hartree")
    print(f"\nForces (Hartree/Å):")
    for i, (s, f) in enumerate(zip(symbols, forces)):
        mag = np.linalg.norm(f)
        print(f"  {s}{i}: [{f[0]:9.6f}, {f[1]:9.6f}, {f[2]:9.6f}]  |F| = {mag:.6f}")
    
    # Check force magnitude
    force_mags = np.linalg.norm(forces, axis=1)
    max_force = force_mags.max()
    mean_force = force_mags.mean()
    
    print(f"\nForce Statistics:")
    print(f"  Max |F|:  {max_force:.6f} Hartree/Å")
    print(f"  Mean |F|: {mean_force:.6f} Hartree/Å")
    print(f"  RMS F:    {np.sqrt((forces**2).mean()):.6f} Hartree/Å")
    
    if max_force < 1e-6:
        print(f"\n⚠️  WARNING: Forces are essentially ZERO!")
        print(f"   This explains why MD doesn't move")
        print(f"   Possible causes:")
        print(f"     • Model is predicting flat PES")
        print(f"     • Finite difference delta too small")
        print(f"     • Model not sensitive to geometry changes")
    elif max_force < 1e-3:
        print(f"\n⚠️  WARNING: Forces are very small")
        print(f"   MD will be very slow")
    else:
        print(f"\n✓ Forces look reasonable")
    
    # Test 3: Perturbed geometry
    print(f"\n{'='*70}")
    print("TEST 3: Perturbed Geometry")
    print("="*70)
    
    # Displace one H atom
    coords_pert = coords.copy()
    coords_pert[1, 2] += 0.1  # Move H1 up by 0.1 Å
    
    print(f"\nPerturbation: H1 displaced +0.1 Å in Z")
    
    energy_pert = wrapper.predict(coords_pert)
    energy_diff = (energy_pert - energy) * 627.509  # kcal/mol
    
    print(f"Energy (original): {energy:.6f} Hartree")
    print(f"Energy (perturbed): {energy_pert:.6f} Hartree")
    print(f"ΔE: {energy_diff:.4f} kcal/mol")
    
    if abs(energy_diff) < 0.01:
        print(f"\n⚠️  WARNING: Energy barely changes with displacement!")
        print(f"   Model is not sensitive to geometry")
        print(f"   This explains zero forces")
    else:
        print(f"\n✓ Model responds to geometry changes")
    
    # Test 4: Check descriptor computation
    print(f"\n{'='*70}")
    print("TEST 4: Descriptor Computation")
    print("="*70)
    
    desc1 = wrapper.descriptor.compute(symbols, coords)
    desc2 = wrapper.descriptor.compute(symbols, coords_pert)
    
    print(f"Descriptor shape: {desc1.shape}")
    print(f"Descriptor range: [{desc1.min():.2f}, {desc1.max():.2f}]")
    
    desc_diff = np.abs(desc2 - desc1)
    print(f"\nDescriptor change with perturbation:")
    print(f"  Max change: {desc_diff.max():.6f}")
    print(f"  Mean change: {desc_diff.mean():.6f}")
    print(f"  Changed features: {(desc_diff > 1e-6).sum()} / {len(desc1)}")
    
    if desc_diff.max() < 1e-6:
        print(f"\n⚠️  WARNING: Descriptor doesn't change!")
        print(f"   Descriptor calculation may be broken")
    else:
        print(f"\n✓ Descriptor responds to geometry changes")
    
    # Test 5: Finite difference delta
    print(f"\n{'='*70}")
    print("TEST 5: Finite Difference Step Size")
    print("="*70)
    
    if hasattr(wrapper, 'force_delta'):
        delta = wrapper.force_delta
        print(f"Current delta: {delta} Å")
    else:
        delta = 0.001  # Default
        print(f"Assumed delta: {delta} Å")
    
    # Manual finite difference test
    coords_plus = coords.copy()
    coords_plus[0, 0] += delta  # Move N in X
    
    e_center = wrapper.predict(coords)
    e_plus = wrapper.predict(coords_plus)
    
    f_fd = -(e_plus - e_center) / delta
    
    print(f"\nManual FD test (N, X-direction):")
    print(f"  E(x): {e_center:.8f} Ha")
    print(f"  E(x+δ): {e_plus:.8f} Ha")
    print(f"  F_x = -(E+ - E) / δ = {f_fd:.6f} Ha/Å")
    
    if abs(f_fd) < 1e-6:
        print(f"\n⚠️  Finite difference gives zero force")
        print(f"   Energy doesn't change enough")
    
    # Summary
    print(f"\n{'='*70}")
    print("DIAGNOSIS")
    print("="*70)
    
    issues = []
    
    if max_force < 1e-6:
        issues.append("Forces are essentially zero")
    
    if abs(energy_diff) < 0.01:
        issues.append("Energy insensitive to geometry changes")
    
    if desc_diff.max() < 1e-6:
        issues.append("Descriptor doesn't change with geometry")
    
    if issues:
        print(f"\n⚠️  PROBLEMS DETECTED:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print(f"\n💡 Root cause:")
        print(f"   Your ML-PES model is predicting a FLAT potential energy surface!")
        print(f"   This could mean:")
        print(f"     • Model was trained on data with very small geometry variations")
        print(f"     • Model is overfitted to equilibrium geometry")
        print(f"     • Descriptor isn't capturing geometry differences")
        print(f"     • Hyperparameters (kernel width) too large")
        
        print(f"\n🔧 Solutions:")
        print(f"   1. Check training data has diverse geometries")
        print(f"   2. Retrain ML-PES with better hyperparameters")
        print(f"   3. Use smaller kernel width (alpha)")
        print(f"   4. Ensure descriptors are geometry-dependent")
    else:
        print(f"\n✅ ML-PES looks good!")
        print(f"   Forces are non-zero")
        print(f"   Model responds to geometry")
        print(f"   Should produce dynamics")
    
    print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nUsage: python3 test_force_predictions.py <mlpes_model.pkl> <dipole_model.pkl>")
        print("\nExample:")
        print("  python3 test_force_predictions.py \\")
        print("    outputs/refined_*/mlpes_model_energy_forces.pkl \\")
        print("    dipole_surface_ammonia.pkl")
        sys.exit(1)
    
    model_path = sys.argv[1]
    dipole_path = sys.argv[2]
    
    if not Path(model_path).exists():
        print(f"\n❌ ML-PES model not found: {model_path}")
        sys.exit(1)
    
    if not Path(dipole_path).exists():
        print(f"\n❌ Dipole model not found: {dipole_path}")
        sys.exit(1)
    
    test_force_predictions(model_path, dipole_path)
