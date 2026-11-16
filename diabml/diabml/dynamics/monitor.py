"""Watched trajectories: run constant-energy dynamics on a learned surface and stop *before* it goes wrong.

Every step records the energies of all states, the ensemble members' disagreement, the internal coordinates, the
energy gap and nonadiabatic coupling strength to the neighbouring states, the largest force, and how far the geometry
is from the reference data. Two kinds of signal are checked:

  uncertain surface (stops the trajectory, after a short look-ahead):
      spread     the ensemble members disagree on the active state by more than spread_eV
      distance   the geometry is more than data_distance grid steps from every reference point
      jump       the total energy changes by more than jump_meV in one step (a rough or broken surface)
      force      a force above force_max eV/Angstrom (a hole or a wall the data never saw)
  nonadiabatic region (recorded, the trajectory continues on its state):
      gap        the active state comes within gap_eV of a neighbour
      coupling   Massey parameter hbar |d_kl . v| / (E_l - E_k) above `massey`: the nuclei move too fast for the
                 electrons to follow adiabatically (a surface-hopping method would let the state change)

The geometries where the surface became uncertain are what active learning from dynamics computes next (see
diabml/workflow/explore.py): the trajectory is stopped before it can fall into a hole the data does not rule out.
"""
from __future__ import annotations

import time

import numpy as np

DEFAULT_TRIGGERS = {'spread_eV': 0.25, 'data_distance': 1.5, 'jump_meV': 5.0, 'force_max': 25.0,
                    'gap_eV': 0.25, 'massey': 0.1, 'lookahead_fs': 5.0}
HBAR_EV_FS = 0.6582119569


class _Stop(Exception):
    pass


def run_trajectory(model, *, r_oo=1.35, a_coo=None, r_co=None, temperature=1000.0, steps=2000, dt=0.1, seed=0,
                   state=0, triggers=None, data_coords=None, scales=None, traj_path=None, stop=True, label=None):
    """Velocity-Verlet trajectory of planar CH2OO on `model` (any diabml matrix model or ensemble).

    Returns a dict of per-step arrays ('t', 'E_pot', 'E_kin', 'E_tot', 'E' (n, ns), 'spread' (n, ns), 'coords'
    (n, d), 'gap_up', 'gap_down', 'coupling_up', 'coupling_down', 'f_max', 'ionic_weight', 'data_distance'),
    the stop reason, the trigger time, the flagged geometries ('flagged', model coordinates) and the
    nonadiabatic events."""
    import torch
    from ase import units
    from ase.io import Trajectory
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    from ase.md.verlet import VelocityVerlet
    from . import CoordinateSurface, ch2oo_atoms, ch2oo_coordinates, reduced_restraints
    from .ase_calc import DiabmlCalculator

    tr = dict(DEFAULT_TRIGGERS, **(triggers or {}))
    names = model.meta.get('coord_names', ['r_OO', 'a_COO'])
    fixed = {k.lower(): v for k, v in model.meta.get('fixed_coordinates', {}).items()}
    geo = dict(fixed, r_oo=r_oo)
    if a_coo is not None:
        geo['a_coo'] = a_coo
    if r_co is not None:
        geo['r_co'] = r_co
    atoms = ch2oo_atoms(**geo)
    coord_fn = ch2oo_coordinates(names)
    surface = reduced_restraints(CoordinateSurface(model, coord_fn), atoms.positions, free=names)
    atoms.calc = DiabmlCalculator(surface, state=state)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=np.random.default_rng(seed))
    Stationary(atoms); ZeroRotation(atoms)
    ion = len(model.frame.diabats) - 1 if getattr(model, 'frame', None) is not None else None
    D = None if data_coords is None else np.asarray(data_coords, float) / np.asarray(scales, float)
    dyn = VelocityVerlet(atoms, timestep=dt * units.fs)
    traj = Trajectory(str(traj_path), 'w', atoms) if traj_path else None
    rec = {k: [] for k in ('t', 'E_pot', 'E_kin', 'E_tot', 'E', 'spread', 'coords', 'gap_up', 'gap_down',
                           'coupling_up', 'coupling_down', 'massey', 'f_max', 'ionic_weight', 'data_distance')}
    info = {'stop': None, 't_trigger': None, 'flagged': [], 'flag_reason': [], 'events': [], 'end_after': None}
    look = int(round(tr['lookahead_fs'] / dt))

    def hard(reason):
        return reason.startswith('force') or reason.startswith('jump')

    def record():
        res = atoms.calc.results
        E = np.asarray(res['energies'])
        ns = len(E)
        epot, ekin = atoms.get_potential_energy(), atoms.get_kinetic_energy()
        sp = np.std(res['member_energies'], 0) if 'member_energies' in res else np.full(ns, np.nan)
        x = coord_fn(torch.tensor(atoms.positions)).numpy()
        v = atoms.get_velocities() * units.fs                       # Angstrom / fs
        d = res.get('couplings')
        cu = cd = np.nan
        if d is not None:
            if state + 1 < ns:
                cu = abs(float(np.sum(d[state, state + 1] * v)))
            if state > 0:
                cd = abs(float(np.sum(d[state, state - 1] * v)))
        dd = float(np.sqrt(np.min(np.sum((D - x / np.asarray(scales)) ** 2, 1)))) if D is not None else np.nan
        t = dyn.nsteps * dt
        rec['t'].append(t); rec['E_pot'].append(epot); rec['E_kin'].append(ekin); rec['E_tot'].append(epot + ekin)
        rec['E'].append(E); rec['spread'].append(sp); rec['coords'].append(x)
        rec['gap_up'].append(E[state + 1] - E[state] if state + 1 < ns else np.nan)
        rec['gap_down'].append(E[state] - E[state - 1] if state > 0 else np.nan)
        rec['coupling_up'].append(cu); rec['coupling_down'].append(cd)
        rec['f_max'].append(float(np.max(np.linalg.norm(res['forces'], axis=1))))
        rec['ionic_weight'].append(res['structure_weights'][state, ion] if ion is not None else np.nan)
        rec['data_distance'].append(dd)
        # nonadiabatic region: record, keep going
        near = min([g for g in (rec['gap_up'][-1], rec['gap_down'][-1]) if np.isfinite(g)], default=np.inf)
        xi = max([HBAR_EV_FS * c / g for c, g in ((cu, rec['gap_up'][-1]), (cd, rec['gap_down'][-1]))
                  if np.isfinite(c) and np.isfinite(g) and g > 0], default=0.0)
        rec['massey'].append(xi)
        if near < tr['gap_eV'] or xi > tr['massey']:
            if not info['events'] or t - info['events'][-1]['t_end'] > 2.0:
                info['events'].append({'t': t, 't_end': t, 'gap_eV': near, 'massey': xi, 'coords': x.tolist()})
            else:
                ev = info['events'][-1]
                ev['t_end'] = t
                ev['gap_eV'] = min(ev['gap_eV'], near)
                ev['massey'] = max(ev['massey'], xi)
        # uncertain surface
        reason = None
        if np.isfinite(sp[state]) and sp[state] > tr['spread_eV']:
            reason = f'spread {sp[state] * 1000:.0f} meV'
        elif np.isfinite(dd) and dd > tr['data_distance']:
            reason = f'distance {dd:.1f} grid steps from the data'
        elif len(rec['E_tot']) > 1 and abs(rec['E_tot'][-1] - rec['E_tot'][-2]) * 1000 > tr['jump_meV']:
            reason = f'jump {abs(rec["E_tot"][-1] - rec["E_tot"][-2]) * 1000:.1f} meV in one step'
        elif rec['f_max'][-1] > tr['force_max']:
            reason = f'force {rec["f_max"][-1]:.0f} eV/Å'
        if reason and info['t_trigger'] is None:
            info['t_trigger'], info['stop'] = t, reason
            info['flagged'].append(x.tolist()); info['flag_reason'].append(reason)
            info['end_after'] = dyn.nsteps + (0 if hard(reason) else look)
        if info['t_trigger'] is not None and stop:
            if reason and hard(reason) and dyn.nsteps > 0:
                raise _Stop
            if dyn.nsteps >= info['end_after']:
                raise _Stop

    dyn.attach(record, interval=1)
    if traj is not None:
        dyn.attach(traj.write, interval=10)
    t0 = time.time()
    try:
        dyn.run(steps)
    except _Stop:
        pass
    if traj is not None:
        traj.close()
    out = {k: np.array(v) for k, v in rec.items()}
    # where it was heading: the most uncertain geometry after the trigger (the look-ahead)
    if info['t_trigger'] is not None:
        after = out['t'] >= info['t_trigger']
        sp = out['spread'][:, state]
        if np.isfinite(sp[after]).any():
            j = np.where(after)[0][int(np.nanargmax(sp[after]))]
            if out['t'][j] > info['t_trigger']:
                info['flagged'].append(out['coords'][j].tolist()); info['flag_reason'].append('look-ahead, largest spread')
    out.update(info, seconds=time.time() - t0, names=names, state=state, dt=dt, steps=steps,
               start={'r_oo': r_oo, 'a_coo': a_coo, 'r_co': r_co, 'temperature': temperature, 'seed': seed},
               label=label or f'r_OO {r_oo:g} Å, {temperature:g} K, seed {seed}', triggers=tr,
               completed=info['t_trigger'] is None or not stop)
    return out


def summarize(rec):
    """One line of numbers for a trajectory (used by the HTML log and the explore table)."""
    e = rec['E_tot']
    s = rec['spread'][:, rec['state']] if len(rec['spread']) else np.array([np.nan])
    return {'trajectory': rec['label'], 'time_fs': float(rec['t'][-1]) if len(rec['t']) else 0.0,
            'stopped': rec['stop'] or ('completed' if rec['completed'] else ''),
            'drift_meV': float((e[-1] - e[0]) * 1000) if len(e) else np.nan,
            'rms_meV': float(np.std(e) * 1000) if len(e) else np.nan,
            'max_spread_meV': float(np.nanmax(s) * 1000) if np.isfinite(s).any() else np.nan,
            'min_gap_eV': float(np.nanmin(np.r_[rec['gap_up'], rec['gap_down']])) if len(rec['gap_up']) else np.nan,
            'r_OO_range': [float(rec['coords'][:, 0].min()), float(rec['coords'][:, 0].max())] if len(rec['coords']) else None,
            'nonadiabatic_events': len(rec['events']), 'flagged': len(rec['flagged'])}
