"""Constant-energy trajectories on a trained CH2OO model (reduced-dimensional test).

    python examples/ch2oo_nve.py runs/01_baseline/models/random_framed.pt --temperature 1500 --steps 4000

What it does
  * loads a model saved by the workflow (load_model reads weights + spec; no pickles),
  * builds planar CH2OO at equilibrium and restrains everything the model does not describe with stiff springs
    (planar CH2O frame; also r_CO and/or the C-O-O angle if the model does not have them),
  * draws Maxwell-Boltzmann velocities, runs velocity Verlet on the chosen state, and
  * writes <out>.csv (time, energies of all states, r_OO, angle, ionic weight of the active state) and <out>.traj,
    and prints the total-energy drift, the first test any surface must pass before production dynamics.

The model is a function of 2-3 internal coordinates only, so this is a test of smoothness, energy conservation and
state structure, not a full-dimensional simulation (see "From surface to dynamics" in RUNNING_CH2OO.html).
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', help='a .pt file written by the workflow (models/*.pt)')
    ap.add_argument('--state', type=int, default=0, help='adiabatic state to run on (0 = ground state)')
    ap.add_argument('--temperature', type=float, default=1000.0, help='K, for the initial velocities')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--dt', type=float, default=0.2, help='fs (restraints are stiff: keep it at 0.25 or below)')
    ap.add_argument('--r_oo', type=float, default=1.35, help='starting O-O distance (Angstrom)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='ch2oo_nve')
    a = ap.parse_args(argv)

    from ase import units
    from ase.io import Trajectory
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    from ase.md.verlet import VelocityVerlet
    from diabml.models import load_model
    from diabml.dynamics import CoordinateSurface, DiabmlCalculator, ch2oo_atoms, ch2oo_coordinates, reduced_restraints

    model = load_model(a.model)
    names = model.meta.get('coord_names', ['r_OO', 'a_COO'])
    fixed = {k.lower(): v for k, v in model.meta.get('fixed_coordinates', {}).items()}
    atoms = ch2oo_atoms(**dict(fixed, r_oo=a.r_oo))
    surface = reduced_restraints(CoordinateSurface(model, ch2oo_coordinates(names)), atoms.positions, free=names)
    atoms.calc = DiabmlCalculator(surface, state=a.state)
    MaxwellBoltzmannDistribution(atoms, temperature_K=a.temperature, rng=np.random.default_rng(a.seed))
    Stationary(atoms); ZeroRotation(atoms)

    ion = len(model.frame.diabats) - 1 if hasattr(model, 'frame') else None     # the ion-pair structure is last
    dyn = VelocityVerlet(atoms, timestep=a.dt * units.fs)
    traj = Trajectory(f'{a.out}.traj', 'w', atoms)
    rows, t0 = [], time.time()

    def record():
        res = atoms.calc.results
        epot, ekin = atoms.get_potential_energy(), atoms.get_kinetic_energy()
        w = res['structure_weights'][a.state, ion] if ion is not None else np.nan
        rows.append([dyn.nsteps * a.dt, epot, ekin, epot + ekin, *res['energies'], atoms.get_distance(1, 2),
                     atoms.get_angle(0, 1, 2), w])

    dyn.attach(record, interval=1)
    dyn.attach(traj.write, interval=10)
    dyn.run(a.steps)
    rows = np.array(rows)
    head = ['t_fs', 'E_pot', 'E_kin', 'E_tot'] + [f'E{s}' for s in range(len(atoms.calc.results['energies']))] + \
           ['r_OO', 'a_COO', 'ionic_weight']
    with open(f'{a.out}.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(head); w.writerows(np.round(rows, 6))
    etot = rows[:, 3]
    gap = np.min(rows[:, 5] - rows[:, 4]) if a.state == 0 else np.nan
    print(f'{a.steps} steps of {a.dt} fs on S{a.state} in {time.time() - t0:.0f} s; '
          f'r_OO {rows[:, -3].min():.2f}-{rows[:, -3].max():.2f} Å, angle {rows[:, -2].min():.0f}-{rows[:, -2].max():.0f}°')
    print(f'total energy drift {1000 * (etot[-1] - etot[0]):+.2f} meV, rms fluctuation {1000 * etot.std():.2f} meV '
          f'(kinetic energy fluctuates by {1000 * rows[:, 2].std():.0f} meV); smallest S1-S0 gap {gap:.3f} eV')
    print(f'wrote {a.out}.csv and {a.out}.traj')


if __name__ == '__main__':
    main()
