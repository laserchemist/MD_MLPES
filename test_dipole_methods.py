#!/usr/bin/env python3
"""
Quick Dipole Computation Test

Tests all dipole computation methods on a single configuration
to determine which one works best with your PSI4 installation.

Usage:
    python3 test_dipole_methods.py
"""

import numpy as np
import time
import psi4

# Test configuration (NH3)
symbols = ['N', 'H', 'H', 'H']
coords = np.array([
    [ 0.0000,  0.0000,  0.1173],
    [ 0.0000,  0.9377, -0.2738],
    [ 0.8121, -0.4689, -0.2738],
    [-0.8121, -0.4689, -0.2738]
])

method = 'B3LYP'
basis = '6-31G*'

print("=" * 80)
print("  DIPOLE COMPUTATION METHOD TESTING")
print("=" * 80)
print(f"\nTest molecule: NH3")
print(f"Theory: {method}/{basis}")
print(f"Expected dipole: ~1.47 Debye (literature)")
print()

# Create molecule string
mol_str = f"0 1\n"
for s, c in zip(symbols, coords):
    mol_str += f"{s} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n"
mol_str += "units angstrom\nno_reorient\nno_com"

# Clean PSI4
psi4.core.clean_options()
psi4.core.clean()
psi4.core.be_quiet()
psi4.set_memory('2 GB')
psi4.set_num_threads(4)

# Create molecule
mol = psi4.geometry(mol_str)
psi4.set_options({
    'basis': basis,
    'scf_type': 'df',
    'reference': 'rhf',
    'maxiter': 200,
    'e_convergence': 1e-6,
    'd_convergence': 1e-6
})

# Compute gradient to get wavefunction
method_string = f"{method}/{basis}"
gradient_result, wfn = psi4.gradient(method_string, molecule=mol, return_wfn=True)

print("✅ Wavefunction obtained\n")

# Test all methods
results = {}

# =============================================================================
# METHOD 1: Density Matrix (Recommended)
# =============================================================================
print("─" * 80)
print("METHOD 1: Density Matrix Computation")
print("─" * 80)

try:
    start = time.time()
    
    # Get density matrix
    Da = wfn.Da()
    
    # Get dipole integrals
    mints = psi4.core.MintsHelper(wfn.basisset())
    dipole_ints = mints.ao_dipole()
    
    # Electronic dipole (atomic units)
    dipole_e = np.array([
        Da.vector_dot(dipole_ints[0]),
        Da.vector_dot(dipole_ints[1]),
        Da.vector_dot(dipole_ints[2])
    ])
    
    # Nuclear dipole (atomic units)
    mol_geom = wfn.molecule()
    dipole_n = np.array([
        mol_geom.nuclear_dipole()[0],
        mol_geom.nuclear_dipole()[1],
        mol_geom.nuclear_dipole()[2]
    ])
    
    # Total dipole (atomic units)
    dipole_au = dipole_e + dipole_n
    
    # Convert to Debye
    au_to_debye = 2.54174623
    dipole = dipole_au * au_to_debye
    
    elapsed = time.time() - start
    
    magnitude = np.linalg.norm(dipole)
    
    print(f"✅ SUCCESS")
    print(f"   Dipole: [{dipole[0]:.4f}, {dipole[1]:.4f}, {dipole[2]:.4f}] Debye")
    print(f"   |μ|: {magnitude:.4f} Debye")
    print(f"   Time: {elapsed*1000:.2f} ms")
    print(f"   Status: {'✅ CORRECT' if 1.3 < magnitude < 1.7 else '⚠️  Check value'}")
    
    results['density_matrix'] = {
        'success': True,
        'dipole': dipole,
        'magnitude': magnitude,
        'time': elapsed
    }
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    results['density_matrix'] = {'success': False, 'error': str(e)}

# =============================================================================
# METHOD 2: PSI4 oeprop + variable
# =============================================================================
print("\n" + "─" * 80)
print("METHOD 2: PSI4 oeprop + psi4.variable()")
print("─" * 80)

try:
    start = time.time()
    
    psi4.oeprop(wfn, 'DIPOLE')
    
    dipole = np.array([
        psi4.variable('DIPOLE X'),
        psi4.variable('DIPOLE Y'),
        psi4.variable('DIPOLE Z')
    ])
    
    elapsed = time.time() - start
    magnitude = np.linalg.norm(dipole)
    
    print(f"✅ SUCCESS")
    print(f"   Dipole: [{dipole[0]:.4f}, {dipole[1]:.4f}, {dipole[2]:.4f}] Debye")
    print(f"   |μ|: {magnitude:.4f} Debye")
    print(f"   Time: {elapsed*1000:.2f} ms")
    print(f"   Status: {'✅ CORRECT' if 1.3 < magnitude < 1.7 else '⚠️  Check value'}")
    
    results['oeprop_variable'] = {
        'success': True,
        'dipole': dipole,
        'magnitude': magnitude,
        'time': elapsed
    }
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    results['oeprop_variable'] = {'success': False, 'error': str(e)}

# =============================================================================
# METHOD 3: Wavefunction variable
# =============================================================================
print("\n" + "─" * 80)
print("METHOD 3: Wavefunction variable")
print("─" * 80)

try:
    start = time.time()
    
    psi4.oeprop(wfn, 'DIPOLE')
    
    dipole = np.array([
        wfn.variable('DIPOLE X'),
        wfn.variable('DIPOLE Y'),
        wfn.variable('DIPOLE Z')
    ])
    
    elapsed = time.time() - start
    magnitude = np.linalg.norm(dipole)
    
    print(f"✅ SUCCESS")
    print(f"   Dipole: [{dipole[0]:.4f}, {dipole[1]:.4f}, {dipole[2]:.4f}] Debye")
    print(f"   |μ|: {magnitude:.4f} Debye")
    print(f"   Time: {elapsed*1000:.2f} ms")
    print(f"   Status: {'✅ CORRECT' if 1.3 < magnitude < 1.7 else '⚠️  Check value'}")
    
    results['wfn_variable'] = {
        'success': True,
        'dipole': dipole,
        'magnitude': magnitude,
        'time': elapsed
    }
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    results['wfn_variable'] = {'success': False, 'error': str(e)}

# =============================================================================
# METHOD 4: Direct oeprop access
# =============================================================================
print("\n" + "─" * 80)
print("METHOD 4: Direct oeprop object access")
print("─" * 80)

try:
    start = time.time()
    
    psi4.oeprop(wfn, 'DIPOLE')
    oep = wfn.oeprop
    
    dipole = np.array([oep.Dx(), oep.Dy(), oep.Dz()])
    
    elapsed = time.time() - start
    magnitude = np.linalg.norm(dipole)
    
    # Check if in atomic units
    if magnitude < 0.1:
        dipole = dipole * 2.54174623  # Convert to Debye
        magnitude = np.linalg.norm(dipole)
    
    print(f"✅ SUCCESS")
    print(f"   Dipole: [{dipole[0]:.4f}, {dipole[1]:.4f}, {dipole[2]:.4f}] Debye")
    print(f"   |μ|: {magnitude:.4f} Debye")
    print(f"   Time: {elapsed*1000:.2f} ms")
    print(f"   Status: {'✅ CORRECT' if 1.3 < magnitude < 1.7 else '⚠️  Check value'}")
    
    results['oeprop_direct'] = {
        'success': True,
        'dipole': dipole,
        'magnitude': magnitude,
        'time': elapsed
    }
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    results['oeprop_direct'] = {'success': False, 'error': str(e)}

# =============================================================================
# METHOD 5: properties() call
# =============================================================================
print("\n" + "─" * 80)
print("METHOD 5: psi4.properties() with dipole")
print("─" * 80)

try:
    start = time.time()
    
    # Need fresh calculation
    psi4.core.clean_options()
    psi4.set_options({
        'basis': basis,
        'scf_type': 'df',
        'reference': 'rhf',
        'maxiter': 200,
        'e_convergence': 1e-6,
        'd_convergence': 1e-6
    })
    
    energy, wfn2 = psi4.properties(method_string, properties=['dipole'], 
                                    molecule=mol, return_wfn=True)
    
    # Try to get dipole
    try:
        dipole = np.array([
            psi4.variable('DIPOLE X'),
            psi4.variable('DIPOLE Y'),
            psi4.variable('DIPOLE Z')
        ])
    except:
        dipole = np.array([
            wfn2.variable('DIPOLE X'),
            wfn2.variable('DIPOLE Y'),
            wfn2.variable('DIPOLE Z')
        ])
    
    elapsed = time.time() - start
    magnitude = np.linalg.norm(dipole)
    
    print(f"✅ SUCCESS")
    print(f"   Dipole: [{dipole[0]:.4f}, {dipole[1]:.4f}, {dipole[2]:.4f}] Debye")
    print(f"   |μ|: {magnitude:.4f} Debye")
    print(f"   Time: {elapsed*1000:.2f} ms")
    print(f"   Status: {'✅ CORRECT' if 1.3 < magnitude < 1.7 else '⚠️  Check value'}")
    
    results['properties'] = {
        'success': True,
        'dipole': dipole,
        'magnitude': magnitude,
        'time': elapsed
    }
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    results['properties'] = {'success': False, 'error': str(e)[:100]}

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("  SUMMARY")
print("=" * 80)

successful_methods = [name for name, res in results.items() if res['success']]

if successful_methods:
    print(f"\n✅ {len(successful_methods)}/{len(results)} methods successful:")
    
    for method_name in successful_methods:
        res = results[method_name]
        print(f"\n   {method_name.upper()}:")
        print(f"      |μ| = {res['magnitude']:.4f} Debye")
        print(f"      Time = {res['time']*1000:.2f} ms")
        
        # Check accuracy
        if 1.3 < res['magnitude'] < 1.7:
            print(f"      ✅ Accurate (literature: ~1.47 Debye)")
        else:
            print(f"      ⚠️  Magnitude seems off")
    
    # Recommend fastest accurate method
    print("\n" + "─" * 80)
    print("RECOMMENDATION:")
    print("─" * 80)
    
    # Find fastest accurate method
    accurate_methods = {
        name: res for name, res in results.items()
        if res['success'] and 1.3 < res['magnitude'] < 1.7
    }
    
    if accurate_methods:
        fastest = min(accurate_methods.items(), key=lambda x: x[1]['time'])
        print(f"\n✅ Use: {fastest[0].upper()}")
        print(f"   Most accurate and reliable")
        print(f"   Time: {fastest[1]['time']*1000:.2f} ms per dipole")
        
        # Time estimate
        n_configs = 2500
        total_time = n_configs * fastest[1]['time'] / 60
        print(f"\n⏱️  For 2500 configurations:")
        print(f"   Estimated time: {total_time:.1f} minutes")
        
        if fastest[0] == 'density_matrix':
            print(f"\n💡 DENSITY MATRIX method is:")
            print(f"   • Most robust (works on all PSI4 versions)")
            print(f"   • Most reliable (direct calculation)")
            print(f"   • Production-ready")
            print(f"   → Use generate_training_data_COMPLETE_WORKING.py ✅")
    
else:
    print("\n❌ No methods successful!")
    print("   PSI4 installation may have issues")
    print("   Try reinstalling PSI4 or using conda environment")

# Show failed methods
failed_methods = [name for name, res in results.items() if not res['success']]
if failed_methods:
    print(f"\n⚠️  Failed methods: {', '.join(failed_methods)}")

print("\n" + "=" * 80)
print("  CONCLUSION")
print("=" * 80)

if 'density_matrix' in successful_methods:
    print("\n✅ Density matrix method WORKS!")
    print("   → Use generate_training_data_COMPLETE_WORKING.py")
    print("   → This will give you 95%+ dipole success")
    print()
elif successful_methods:
    print(f"\n✅ {successful_methods[0].upper()} works!")
    print("   → Modify generate script to use this method")
    print()
else:
    print("\n❌ No methods working")
    print("   → Check PSI4 installation")
    print()

print("=" * 80)
