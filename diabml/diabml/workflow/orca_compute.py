"""The compute stage of active learning with ORCA: new reference points that cannot land on a wrong SA-CASSCF solution.

    [compute]
    engine = "orca"          # in the run's config.toml (the active and explore templates set it)

and ORCA found as for the audit ([compute] orca, [audit] orca, --orca, or the ORCA_PATH environment variable).

Every proposed geometry is computed several times in parallel (`jobs` ORCA runs at once):
  scratch     ORCA's own starting orbitals
  neighbour   MORead from the converged orbitals of each of the `neighbours` nearest points already in the data
              (the orbital library orca_orbitals.json, made by the repair stage from the audit and extended by every
              ORCA compute round)
and the solution with the lowest state-averaged SA-CASSCF energy is kept: SA-CASSCF minimises that average, so the
lowest one found is the right one (the audit's rule). Points where the attempts found different solutions are counted
and listed, so you can see where the surface has several.

Energies are at the level of the base data: SC-NEVPT2 ('nevpt2', as PySCF computes it; the audit showed agreement to
about 2 µEh), QD-NEVPT2 ('qd_nevpt2', so the QD set can now be extended too) or CASSCF, all on the base data's zero.
SC- and QD-NEVPT2 are always both stored (extra E_nevpt2_eV, E_qd_nevpt2_eV). Not computed with ORCA: per-state
terminal-O charges (NaN: the charge restraint skips those points) and couplings (no rows added to nacs.npz).

Writes data/new_refs.npz (merged by the merge stage as before), data/new_orca_orbitals.json and the run folder
data/compute/ (ORCA inputs and outputs); results/active.json 'compute' and figures/A2_new_points.png as before.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import time
from pathlib import Path

import numpy as np

from . import active as A
from . import audit as AU
from . import ch2oo as W

HARTREE_EV = AU.HARTREE_EV
CODE_ORCA_COMPUTE = 4          # extra['audit'] for points computed by this stage


# ------------------------------------------------------------------------------------------------ orbital library
def load_library(path):
    """(coords (n, d), [gbw paths]) of an orca_orbitals.json, keeping entries whose .gbw still exists."""
    p = Path(path)
    if not p.exists():
        return np.zeros((0, 0)), []
    d = json.loads(p.read_text())
    pts = [q for q in d.get('points', []) if Path(q['gbw']).exists()]
    return np.array([q['coords'] for q in pts], float).reshape(len(pts), -1), [q['gbw'] for q in pts]


def save_library(path, coord_names, coords, gbws):
    Path(path).write_text(json.dumps({'coord_names': list(coord_names), 'points': [
        {'coords': [float(v) for v in c], 'gbw': str(g)} for c, g in zip(coords, gbws)]}, indent=0))


# ------------------------------------------------------------------------------------------------ stage
def _exe(run):
    cc = run.cfg['compute']
    return AU._orca_exe(run, cc.get('orca') or run.cfg['audit'].get('orca') or None)


def stage_compute_orca(run):
    from ..geometry import ch2oo_xyz
    from ..refset import RefSet
    cc = run.cfg['compute']
    z = np.load(run.p('data', 'proposed.npz'))
    X = z['coords']
    N = len(X)
    base = RefSet.load(run.p('data', 'base_refs.npz'))
    spec, zero = base.meta['spec'], base.meta['zero_hartree']
    level = base.meta.get('energy_level', 'casscf')
    if level not in ('casscf', 'nevpt2', 'qd_nevpt2'):
        raise SystemExit(f'unknown energy level {level} in the base data')
    if level != 'casscf' and level not in zero:
        raise SystemExit(f'the base data have no {level} energy zero (meta zero_hartree)')
    nr = int(spec['nroots'])
    exe = _exe(run)
    jobs = int(cc['jobs'])
    wd = run.p('data', 'compute')
    wd.mkdir(exist_ok=True)
    sc = A._scales(base.coord_names, A._ac(run))
    LX, LG = load_library(run.p('data', 'orca_orbitals_base.json'))
    nn = int(cc['neighbours']) if len(LG) else 0
    run.log(f'{N} new points with ORCA {exe}: {"scratch + " if cc["scratch"] else ""}{nn} nearest-neighbour '
            f'orbitals each (library {len(LG)} points), {jobs} parallel runs; level {level}')
    if not len(LG):
        run.log('  no ORCA orbital library next to the base data (run the repair stage after an audit to make one): '
                'scratch orbitals only')
    # inputs: one per (point, attempt)
    tasks = []                                           # (point, attempt label, name)
    for k, x in enumerate(X):
        atoms = ch2oo_xyz(**W._geometry_kwargs(base, x))
        att = [('scratch', None)] if cc['scratch'] or not nn else []
        if nn:
            d = np.sum(((LX - x) / sc) ** 2, 1)
            for j in np.argsort(d)[:nn]:
                att.append((f'neighbour {np.round(LX[j], 3).tolist()}', LG[j]))
        for a, (label, gbw) in enumerate(att):
            name = f'c{k:03d}_{a}'
            f = wd / f'{name}.inp'
            if not f.exists():
                f.write_text(AU.orca_input(atoms, spec, moread=gbw, qd=True, maxcore=cc['maxcore']))
            tasks.append((k, label, name))
    t0 = time.time()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(AU._run_one, exe, wd, nm, float(cc['timeout_s'])) for _, _, nm in tasks]
        for f in cf.as_completed(futs):
            done += 1
            if done % 10 == 0 or done == len(tasks):
                el = time.time() - t0
                run.log(f'  {done}/{len(tasks)} ORCA runs, {el / 60:.1f} min, about '
                        f'{el / done * (len(tasks) - done) / 60:.0f} min left')
    # choose the lowest state-averaged SA-CASSCF solution per point
    need = {'casscf': 'casscf', 'nevpt2': 'sc', 'qd_nevpt2': 'qd'}[level]
    best, seconds = [None] * N, np.zeros(N)
    several, rows = [], []
    for k in range(N):
        found = []
        for kk, label, nm in tasks:
            if kk != k:
                continue
            r = AU.parse_orca(wd / f'{nm}.out', nr)
            if r is None or len(r[need]) < nr or len(r['casscf']) < nr:
                continue
            found.append((float(np.mean(r['casscf'])), label, nm, r))
        if not found:
            rows.append((k, 'failed', ''))
            continue
        found.sort(key=lambda t: t[0])
        best[k] = found[0]
        gap = (found[-1][0] - found[0][0]) * 1000
        if gap > float(run.cfg['audit']['tol_uEh']) * 1e-3:
            several.append((k, gap, found[0][1]))
        rows.append((k, found[0][1], f'{gap:.3f}'))
    ok = np.array([b is not None for b in best])
    def arr(key, sort):
        a = np.full((N, nr), np.nan)
        for k in np.where(ok)[0]:
            v = np.asarray(best[k][3][key][:nr], float)
            if len(v) == nr:
                a[k] = np.sort(v) if sort else v
        return a
    Ec, Esc, Eqd = arr('casscf', True), arr('sc', False), arr('qd', True)   # SC per root, in ORCA's root order
    ev = {'casscf': (Ec - zero['casscf']) * HARTREE_EV}
    if 'nevpt2' in zero:
        ev['nevpt2'] = (Esc - zero['nevpt2']) * HARTREE_EV
    if 'qd_nevpt2' in zero:
        ev['qd_nevpt2'] = (Eqd - zero['qd_nevpt2']) * HARTREE_EV
    elif 'nevpt2' in zero:
        # the QD energies on the SC zero (the repair stage moves the QD zero only by the tiny QD shift there)
        ev['qd_nevpt2'] = (Eqd - zero['nevpt2']) * HARTREE_EV
    rnd = int(z['round'])
    extra = {'q_O_terminal': np.full((N, nr), np.nan), 'n_unpaired': np.full((N, nr), np.nan),
             'converged': ok.copy(), 'active_space_consistent': ok.copy(), 'seconds': seconds,
             'E_casscf_eV': ev['casscf'], 'al_round': np.full(N, rnd, int), 'from_orca': ok.copy()}
    if 'audit' in base.extra:                     # the base data went through an audit: mark these as ORCA-computed
        extra['audit'] = np.where(ok, CODE_ORCA_COMPUTE, 3)
    if 'nevpt2' in ev:
        extra['E_nevpt2_eV'] = ev['nevpt2']
        extra['casci_mismatch_hartree'] = np.zeros(N)
    if 'qd_nevpt2' in ev and ('E_qd_nevpt2_eV' in base.extra or level == 'qd_nevpt2'):
        extra['E_qd_nevpt2_eV'] = ev['qd_nevpt2']
    E = ev[level]
    new = RefSet(base.name, X, E, list(base.coord_names), extra=extra, meta=dict(base.meta, active_round=rnd))
    new.save(run.p('data', 'new_refs.npz'))
    save_library(run.p('data', 'new_orca_orbitals.json'), base.coord_names, X[ok],
                 [str((wd / f'{best[k][2]}.gbw').resolve()) for k in np.where(ok)[0]])
    for f in ('new_orbitals.npz', 'new_nacs.npz'):            # PySCF products of an earlier compute: not this round's
        run.p('data', f).unlink(missing_ok=True)
    with open(run.p('results', 'compute_orca.csv'), 'w') as fh:
        fh.write(','.join(['point'] + base.coord_names + ['kept', 'spread_of_attempts_mEh']) + '\n')
        for k, label, gap in rows:
            fh.write(','.join([str(k)] + [f'{v:.4f}' for v in X[k]] + [f'"{label}"', gap]) + '\n')
    # how wrong was the committee where it was least sure?
    E_new = np.sort(E, 1) if run.cfg['train'].get('sort_states', True) else E
    good = ok & np.all(np.isfinite(E_new[:, :4]), 1)
    err = (z['mean'] - E_new[:, :4])[good] * A.EV_KCAL
    hours = (time.time() - t0) / 3600
    stats = {'engine': 'orca', 'n_new': N, 'n_usable_new': int(good.sum()), 'hours': hours,
             'orca_runs': len(tasks), 'points_with_several_solutions': len(several),
             'kept_from_scratch': int(sum(1 for b in best if b is not None and b[1] == 'scratch')),
             'error_at_proposed_rmse_kcal': np.sqrt(np.mean(err ** 2, 0)).tolist() if len(err) else None,
             'error_at_proposed_max_kcal': np.abs(err).max(0).tolist() if len(err) else None,
             'spread_at_proposed_kcal': (z['std'][good].mean(0) * A.EV_KCAL).tolist() if good.any() else None,
             'started_from_scratch': int(not len(LG))}
    A._write_json(run, stats, 'compute')
    run.log(f'{int(good.sum())}/{N} new points usable; {len(tasks)} ORCA runs in {hours * 60:.1f} min; the attempts '
            f'found different solutions at {len(several)} points (lowest kept; {stats["kept_from_scratch"]} from '
            f'scratch orbitals)')
    for k, gap, label in several[:10]:
        run.log(f'  point {k} {np.round(X[k], 3).tolist()}: attempts differ by {gap:.1f} mEh (state average); kept: {label}')
    if len(err):
        run.log(f'committee error at the proposed points, before training on them: RMSE '
                f'{np.round(stats["error_at_proposed_rmse_kcal"], 2)} kcal/mol (spread there '
                f'{np.round(stats["spread_at_proposed_kcal"], 2)})')
    if N:
        A._figure_new_points(run, z['mean'], z['std'], E_new, good)
