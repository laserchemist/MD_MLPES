"""Constant-energy trajectory on a trained CH2OO model, watched, with an HTML log.

    python examples/ch2oo_nve.py runs/13_md/models/production.pt --r_oo 1.7 --temperature 1500 --steps 4000 \
        --refs runs/13_md/data/refs.npz --out runs/13_md/nve_hot

What it does
  * loads a model saved by the workflow (a single model or a production ensemble),
  * builds planar CH2OO, restrains what a reduced model does not describe (planar CH2O frame) with stiff springs,
  * runs velocity Verlet on the chosen adiabatic state (--state, default S0) with diabml.dynamics.monitor watching
    every step: the energies of all states, energy conservation, the ensemble members' disagreement, the distance to
    the reference data (with --refs), the gaps and nonadiabatic couplings to neighbouring states,
  * stops early when the surface becomes untrustworthy (members disagree by --max-spread eV, too far from the data,
    an energy jump or an extreme force), unless --no-stop,
  * writes <out>.traj (view: ase gui <out>.traj), <out>.csv, and <out>.html (figures and tables of the run).

The dynamics is Born-Oppenheimer on one adiabatic state (an eigenvalue of the learned matrix): no state changes.
The HTML log marks the nonadiabatic regions where surface hopping would be needed.
"""
from __future__ import annotations

import argparse
import csv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', help='a .pt file written by the workflow (models/*.pt)')
    ap.add_argument('--state', type=int, default=0, help='adiabatic state to run on (0 = ground state)')
    ap.add_argument('--temperature', type=float, default=1000.0, help='K, for the initial velocities')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--dt', type=float, default=0.1, help='fs (restraints are stiff: 0.1 recommended)')
    ap.add_argument('--r_oo', type=float, default=1.35, help='starting O-O distance (Angstrom)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='ch2oo_nve')
    ap.add_argument('--refs', help='refs.npz the model was trained on: enables the distance-to-data check and plots')
    ap.add_argument('--max-spread', type=float, default=0.25, help='eV: stop when the ensemble members disagree this much')
    ap.add_argument('--no-stop', action='store_true', help='run to the end whatever the monitor says')
    a = ap.parse_args(argv)

    import numpy as np
    from diabml.models import load_model
    from diabml.dynamics.monitor import run_trajectory, summarize
    from diabml.dynamics.report import write_html

    model = load_model(a.model)
    data, scales = None, None
    if a.refs:
        from diabml.refset import RefSet
        ref = RefSet.load(a.refs)
        data = ref.coords
        scales = np.array([{'r_OO': 0.1, 'a_COO': 10.0, 'r_CO': 0.05}.get(n, 1.0) for n in ref.coord_names])
    rec = run_trajectory(model, r_oo=a.r_oo, temperature=a.temperature, steps=a.steps, dt=a.dt, seed=a.seed,
                         state=a.state, triggers={'spread_eV': a.max_spread}, data_coords=data, scales=scales,
                         traj_path=f'{a.out}.traj', stop=not a.no_stop)
    names = rec['names']
    head = (['t_fs', 'E_pot', 'E_kin', 'E_tot'] + [f'E{s}' for s in range(rec['E'].shape[1])] + names +
            ['ionic_weight', 'member_spread', 'gap_to_neighbour', 'coupling_dv', 'f_max', 'data_distance'])
    gap = np.fmin(np.nan_to_num(rec['gap_up'], nan=np.inf), np.nan_to_num(rec['gap_down'], nan=np.inf))
    cpl = np.fmax(np.nan_to_num(rec['coupling_up']), np.nan_to_num(rec['coupling_down']))
    rows = np.column_stack([rec['t'], rec['E_pot'], rec['E_kin'], rec['E_tot'], rec['E'], rec['coords'],
                            rec['ionic_weight'], rec['spread'][:, a.state], gap, cpl, rec['f_max'], rec['data_distance']])
    with open(f'{a.out}.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(head); w.writerows(np.round(rows, 6))
    s = summarize(rec)
    write_html([rec], f'{a.out}.html', title=f'Dynamics log: {a.out}', data_coords=data,
               notes=f'Model {a.model}; start r_OO {a.r_oo} Å, {a.temperature} K, seed {a.seed}, S{a.state}.')
    if rec['stop'] and not a.no_stop:
        print(f'stopped at {s["time_fs"]:.1f} fs: {rec["stop"]}. The surface is not trustworthy there: add reference '
              f'points (python -m diabml.workflow.ch2oo ... --template explore) or rerun with --no-stop.')
    print(f'{s["time_fs"]:.1f} fs on S{a.state} in {rec["seconds"]:.0f} s; r_OO {s["r_OO_range"][0]:.2f}-'
          f'{s["r_OO_range"][1]:.2f} Å; total energy drift {s["drift_meV"]:+.2f} meV, rms {s["rms_meV"]:.2f} meV; '
          f'largest member spread {s["max_spread_meV"]:.0f} meV; smallest gap {s["min_gap_eV"]:.3f} eV; '
          f'{s["nonadiabatic_events"]} nonadiabatic regions')
    print(f'wrote {a.out}.traj, {a.out}.csv, {a.out}.html')


if __name__ == '__main__':
    main()
