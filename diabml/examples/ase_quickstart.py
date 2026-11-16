"""Short ASE example with a diabml model: single point, geometry optimisation, and a short trajectory on S1.

    python examples/ase_quickstart.py runs/01_baseline/models/random_framed.pt

Writes quickstart_opt.traj and quickstart_S1.traj next to the model; view them with  ase gui <file>.traj
"""
import sys
from pathlib import Path

import numpy as np
from ase import units
from ase.io import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS

from diabml.models import load_model
from diabml.dynamics import CoordinateSurface, DiabmlCalculator, ch2oo_atoms, ch2oo_coordinates, reduced_restraints

path = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/01_baseline/models/random_framed.pt')
model = load_model(path)
names = model.meta['coord_names']                                   # e.g. ['r_OO', 'a_COO']

# 1. molecule + calculator (atoms in the order C, O1, O2, H, H)
atoms = ch2oo_atoms(r_oo=1.40, a_coo=115.0)
surface = reduced_restraints(CoordinateSurface(model, ch2oo_coordinates(names)), atoms.positions, free=names)
atoms.calc = DiabmlCalculator(surface, state=0)

# 2. single point: every state at once
print('energies S0..S3 (eV):', np.round(atoms.calc.state_energies(atoms), 3))
print('zwitterion weight in S0:', round(float(atoms.calc.results['structure_weights'][0, -1]), 2))

# 3. relax on S0 (ASE optimisers work as with any calculator)
BFGS(atoms, trajectory=str(path.with_name('quickstart_opt.traj')), logfile=None).run(fmax=0.01)
print(f'S0 minimum: r(O-O) = {atoms.get_distance(1, 2):.3f} Å, angle C-O-O = {atoms.get_angle(0, 1, 2):.1f}°, '
      f'E = {atoms.get_potential_energy():.3f} eV')

# 4. vertical excitation: switch to S1 and run 100 fs at constant energy
atoms.calc.state = 1
MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(0))
traj = Trajectory(str(path.with_name('quickstart_S1.traj')), 'w', atoms)
md = VelocityVerlet(atoms, timestep=0.2 * units.fs)
md.attach(traj.write, interval=5)


def report():
    E = atoms.calc.results['energies']
    print(f'{md.nsteps * 0.2:6.1f} fs  r(O-O) {atoms.get_distance(1, 2):5.2f} Å  E_tot {atoms.get_total_energy():8.4f} eV  '
          f'S1-S0 gap {E[1] - E[0]:6.3f} eV')


md.attach(report, interval=50)
md.run(500)
print('view with: ase gui', path.with_name('quickstart_S1.traj'))
