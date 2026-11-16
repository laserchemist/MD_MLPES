"""Active learning from dynamics: let trajectories find the weak parts of the surface, and heal them.

    python -m diabml.workflow.ch2oo init runs/14_md1 --template explore     # refs.file = the data to extend
    python -m diabml.workflow.ch2oo all  runs/14_md1     # refs, ladder, explore, compute, merge, train, figures,
                                                         # tables, report  (+ runs/14_md1/dynamics.html)

explore  trains the committee (as the propose stage does), runs watched trajectories on its average (one per start in
         [explore]: O-O distances x temperatures x seeds) and stops each one where the surface becomes uncertain
         (members disagree, too far from the data, an energy jump or an extreme force; see diabml/dynamics/monitor.py),
         after a few fs of look-ahead to see where it was heading. The flagged geometries, de-duplicated against the
         data and each other, become data/proposed.npz, so the compute and merge stages of active learning add them
         at exactly the level of the base data. Nonadiabatic regions (small gaps, strong |d.v|) are logged, not
         stopped on: they tell you where single-state dynamics needs surface hopping.

Repeat with a new folder per round (refs.file = the previous round's data/refs.npz). Trajectories restart from the
same initial conditions, so each round runs further before it needs new data; the loop has converged when every
trajectory completes. Every round writes dynamics.html: energies of all states, energy conservation, coordinates,
member spread, distance to the data, gaps and couplings, and the paths over the reference geometries.
"""
from __future__ import annotations

import itertools
import json

import numpy as np

from . import active as A
from . import ch2oo as W

EV_KCAL = W.EV_KCAL


def _starts(ec):
    out = []
    for r, T, s in itertools.product(ec['r_oo'], ec['temperatures'], range(int(ec['seeds']))):
        out.append({'r_oo': float(r), 'temperature': float(T), 'seed': int(s)})
    return out


def stage_explore(run, force=False):
    from ..dynamics.monitor import run_trajectory, summarize
    from ..dynamics.report import write_html
    from ..models import EnsembleModel
    from ..refset import RefSet
    out = run.p('data', 'proposed.npz')
    if out.exists() and not force:
        run.log(f'{out.name} exists (use --force to explore again); nothing to do')
        return
    run.threads()
    ac, ec, tc = run.cfg['active'], run.cfg['explore'], run.cfg['train']
    ref = RefSet.load(run.p('data', 'base_refs.npz'))
    if tc.get('sort_states', True):
        ref = W.adiabatic_order(ref)
    U = W.usable_points(run, ref)
    models, val = A.train_committee(run, ref, U)
    ens = EnsembleModel(models)
    ens.meta = {'coord_names': ref.coord_names, 'fixed_coordinates': ref.meta.get('fixed', {})}
    sc = A._scales(ref.coord_names, ac)
    triggers = {k: float(ec[k]) for k in ('spread_eV', 'data_distance', 'jump_meV', 'force_max', 'gap_eV',
                                          'massey', 'lookahead_fs')}
    starts = _starts(ec)
    run.log(f'{len(starts)} trajectories of up to {ec["steps"]} x {ec["dt"]} fs on S{ec["state"]}; stop at spread '
            f'{triggers["spread_eV"] * 1000:.0f} meV or {triggers["data_distance"]} grid steps from the data')
    tdir = run.p('results', 'trajectories')
    tdir.mkdir(exist_ok=True)
    recs = []
    for k, st in enumerate(starts):
        rec = run_trajectory(ens, **st, steps=int(ec['steps']), dt=float(ec['dt']), state=int(ec['state']),
                             triggers=triggers, data_coords=ref.coords[U], scales=sc,
                             traj_path=tdir / f'traj_{k:02d}.traj')
        recs.append(rec)
        s = summarize(rec)
        run.log(f'  {k:2d} {rec["label"]:34s} {s["time_fs"]:6.1f} fs  {s["stopped"]:42s} drift {s["drift_meV"]:+6.2f} '
                f'meV  max spread {s["max_spread_meV"]:4.0f} meV  {s["nonadiabatic_events"]} nonadiabatic  '
                f'({rec["seconds"]:.0f} s)')
        np.savez(tdir / f'traj_{k:02d}.npz', **{key: rec[key] for key in ('t', 'E_pot', 'E_kin', 'E_tot', 'E', 'spread',
                                                                       'coords', 'gap_up', 'gap_down', 'coupling_up',
                                                                       'coupling_down', 'massey', 'f_max', 'ionic_weight',
                                                                       'data_distance')})
    # flagged geometries -> proposals (not on top of the data, not on top of each other)
    F = [(np.array(x), why, i) for i, r in enumerate(recs) for x, why in zip(r['flagged'], r['flag_reason'])]
    existing = ref.coords / sc
    keep = []
    for x, why, i in F:
        z = x / sc
        if np.min(np.sum((existing - z) ** 2, 1)) < float(ac['min_distance']) ** 2 * 0.25:
            continue
        if keep and np.min([np.sum((k[0] / sc - z) ** 2) for k in keep]) < float(ac['min_distance']) ** 2 * 0.25:
            continue
        keep.append((x, why, i))
    keep = keep[:int(ec['max_new'])]
    X = np.array([k[0] for k in keep]).reshape(-1, len(ref.coord_names))
    P = ens.member_predictions(X) if len(X) else np.zeros((len(models), 0, 4))
    allx = np.concatenate([r['coords'] for r in recs])
    alls = np.concatenate([r['spread'][:, int(ec['state'])] for r in recs])
    np.savez(out, coords=X, score=P.std(0).max(1) if len(X) else np.zeros(0), mean=P.mean(0), std=P.std(0),
             cand=allx, cand_score=alls, round=A._round_of(ref))
    rows = [summarize(r) for r in recs]
    stats = {'round': A._round_of(ref), 'n_base': len(ref), 'n_usable': int(len(U)), 'committee': len(models),
             'committee_val_rmse_kcal': val, 'trajectories': rows, 'n_completed': int(sum(r['completed'] for r in recs)),
             'n_flagged': len(F), 'n_proposed': int(len(X)), 'reasons': [k[1] for k in keep],
             'nonadiabatic_events': int(sum(len(r['events']) for r in recs)), 'triggers': triggers}
    run.p('results', 'explore.json').write_text(json.dumps(stats, indent=1, default=float))
    write_html(recs, run.p('dynamics.html'), title=f'Dynamics log: {run.path.name}', data_coords=ref.coords[U],
               notes=f'Committee of {len(models)} framed models trained on {len(U)} reference points (round '
                     f'{stats["round"]}). {stats["n_completed"]}/{len(recs)} trajectories completed; '
                     f'{len(X)} geometries proposed for new reference calculations.')
    with open(run.p('results', 'proposed.csv'), 'w') as fh:
        fh.write(','.join(ref.coord_names + ['why', 'trajectory']) + '\n')
        for x, why, i in keep:
            fh.write(','.join(f'{v:.4f}' for v in x) + f',{why},{recs[i]["label"]}\n')
    run.log(f'{stats["n_completed"]}/{len(recs)} trajectories completed; {len(F)} flagged geometries, '
            f'{len(X)} new after removing near-duplicates -> {out.name}; log: {run.p("dynamics.html")}')
    if stats['n_completed'] == len(recs):
        run.log('every trajectory ran to the end without leaving the trusted surface: this set of starts is converged '
                '(try hotter or more stretched starts, or longer runs)')


def table_rows(run):
    path = run.p('results', 'explore.json')
    if not path.exists():
        return None
    e = json.loads(path.read_text())
    rows = [['round', e['round']], ['committee members', e['committee']],
            ['trajectories completed', f'{e["n_completed"]} / {len(e["trajectories"])}'],
            ['flagged geometries (new after de-duplication)', f'{e["n_flagged"]} ({e["n_proposed"]})'],
            ['nonadiabatic regions visited', e['nonadiabatic_events']]]
    for t in e['trajectories']:
        rows.append([t['trajectory'], f'{t["time_fs"]:.0f} fs, {t["stopped"]}, max spread {t["max_spread_meV"]:.0f} meV'])
    return rows
