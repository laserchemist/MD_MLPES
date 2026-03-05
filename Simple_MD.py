from modules.test_molecules import get_molecule
from modules.direct_md import DirectMDConfig, run_direct_md

# Get a test molecule
water = get_molecule('water')

# Configure MD
config = DirectMDConfig(
    method='B3LYP',
    basis='6-31G*',
    temperature=300.0,  # Kelvin
    timestep=0.5,       # femtoseconds
    n_steps=1000,
    thermostat='berendsen'
)

# Run MD and save trajectory
trajectory = run_direct_md(
    water, 
    config, 
    output_dir='outputs/water_md',
    save_format='extxyz'
)

print(f"Generated {trajectory.n_frames} frames")