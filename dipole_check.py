import psi4
import numpy as np

psi4.set_memory("500 MB")
psi4.core.set_output_file("output.dat", False)

mol = psi4.geometry("""
0 1
O
H 1 0.96
H 1 0.96 2 104.5
""")

psi4.set_options({
    'basis': 'cc-pvdz',
    'scf_type': 'pk'
})

# Request properties explicitly
energy, wfn = psi4.energy(
    "scf",
    molecule=mol,
    return_wfn=True,
    properties=["dipole"]
)

# Retrieve dipole vector (a.u.)
dipole_vec = psi4.variable("SCF DIPOLE")

# Or equivalently:
dipole_vec2 = wfn.variable("SCF DIPOLE")

print("Dipole (a.u.):", dipole_vec)

# Convert to Debye if desired
AU_TO_DEBYE = 2.541746
dipole_debye = np.array(dipole_vec) * AU_TO_DEBYE
print("Dipole (Debye):", dipole_debye)
print("Magnitude (Debye):", np.linalg.norm(dipole_debye))

