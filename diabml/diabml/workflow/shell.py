"""The validated spectroscopy surface as a fixed switch, and the CCSD(T) shell beyond the well for the Delta model.

    python -m diabml.workflow.ch2oo freeze      RUN --name benchmark   # save the current well surface, unchanged
    python -m diabml.workflow.ch2oo shell       RUN --batch 1          # CCSD(T) on one batch of shell points
    python -m diabml.workflow.ch2oo shell_check RUN                    # what the shell says, nothing changed

1. The switch. `freeze` fits the well polynomial exactly as vib_levels does now ([vib] source, order, window,
   [gs_grid.delta]) and writes it to models/surfaces/<name>.npz with a manifest (<name>.json: the data files and
   their checksums, the fit, the fundamentals of the last vib_levels run). A frozen surface never changes when data
   are added. Use it with

       [vib] source = "frozen:benchmark"        [ir] surface = "frozen:benchmark"

   and flip back to it at any time; every later surface (the Delta-corrected network) is tested against it.

2. The shell. CCSD(T) points beyond the dense well grid, where the Delta correction (CCSD(T) - diabatic network)
   has to be defined by data and then switched off. Candidates: a coarse product grid over [shell] box with [shell]
   step, minus everything inside the dense grid's range; kept if a cheap energy estimate (the run's diabatic network
   S0 for the reference-data coordinates plus a harmonic term for any coordinate it lacks, curvature from the well
   polynomial) lies in [shell] window_eV above the minimum. Of those, n_points are chosen by farthest-point
   sampling in grid-step units (evenly spread, no clusters) and split into `batches` in that order, so each batch
   covers the whole shell at a coarser density: batch 1 alone is already a usable test. Each batch is computed and
   merged into data/shell.npz (coords, E_total_hartree, T1, batch); finished points are reused.

3. The check. `shell_check` touches no model. For every shell point it compares CCSD(T) with (a) the frozen
   benchmark polynomial where it is inside its fitted box (how far the validated surface can be trusted), and (b)
   the diabatic network S0 (the Delta the next model will learn): size, smoothness, how it grows with r(O-O) and
   with T1. Writes results/shell_check.{json,csv} and figures/SH1_shell_check.png.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import itertools
import json
import time

import numpy as np

from . import audit as AU
from . import ch2oo as W
from .gs_grid import _geometry

HARTREE_EV = AU.HARTREE_EV


# ------------------------------------------------------------------------------------------------ the switch
def _md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest() if p and p.exists() else None


def frozen_path(run, name):
    return run.p('models', 'surfaces', f'{name}.npz')


def load_frozen(run, spec):
    """spec 'frozen:<name>' -> (PolyWell, manifest)."""
    from ..wellfit import PolyWell
    name = spec.split(':', 1)[1]
    p = frozen_path(run, name)
    if not p.exists():
        raise SystemExit(f'{p} not found: make it with  freeze {run.path} --name {name}')
    man = p.with_suffix('.json')
    return PolyWell.load(p), (json.loads(man.read_text()) if man.exists() else {})


def stage_freeze(run, name=None):
    from .vib_levels import _fit, _refs
    name = name or 'benchmark'
    vc = run.cfg['vib']
    if str(vc.get('source', '')).startswith('frozen:'):
        raise SystemExit('[vib] source is already a frozen surface; freeze from the data ("gs_grid" or "refs")')
    p = frozen_path(run, name)
    if p.exists():
        raise SystemExit(f'{p} exists; frozen surfaces are not overwritten (choose another --name)')
    ref = _refs(run, vc)
    well, info = _fit(run, ref, vc['order'], vc['window_eV'])
    p.parent.mkdir(parents=True, exist_ok=True)
    well.save(p)
    dc = run.cfg['gs_grid'].get('delta') or {}
    dpath = None
    if dc.get('run'):
        q = W.C.resolve(str(dc['run']), run.path)
        dpath = q / 'data' / 'gs_grid.npz' if q.is_dir() else q
    vl = run.p('results', 'vib_levels.json')
    fund = json.loads(vl.read_text()).get('fundamentals_cm-1') if vl.exists() else None
    man = {'name': name, 'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'run': str(run.path),
           'source': vc.get('source', 'refs'), 'order': int(vc['order']), 'window_eV': float(vc['window_eV']),
           'delta': dc or None, 'fit': info, 'method': ref.meta.get('method'),
           'data': {'gs_grid.npz': _md5(run.p('data', 'gs_grid.npz')), 'delta': str(dpath) if dpath else None,
                    'delta_md5': _md5(dpath) if dpath else None},
           'fundamentals_cm-1_at_freeze': fund}
    p.with_suffix('.json').write_text(json.dumps(man, indent=1, default=float))
    run.log(f'froze the well surface as models/surfaces/{name}.npz ({info["n_points"]} points, order {info["order"]}, '
            f'residual {info["residual_rms_meV"]:.2f} meV' + (f', with the delta from {dc["run"]}' if dc else '')
            + f'). Use it with [vib] source = "frozen:{name}" and [ir] surface = "frozen:{name}".')


# ------------------------------------------------------------------------------------------------ the shell
def _estimator(run, sc, names):
    """Cheap S0 estimate (eV, relative) for candidates: the diabatic network on its coordinates + a harmonic term
    (curvature from the well polynomial) for coordinates it lacks. None if no network is configured."""
    from ..models import load_model
    mp = sc.get('energy_model')
    if not mp:
        return None, 'none'
    mpath = W.C.resolve(str(mp), run.path)
    if not mpath.exists():
        run.log(f'[shell] energy_model {mpath} not found: no energy filter')
        return None, 'none'
    model = load_model(mpath)
    mn = list(model.meta.get('coord_names', names[:3]))
    fixed = {k.lower(): v for k, v in model.meta.get('fixed_coordinates', {}).items()}
    extra = [n for n in names if n not in mn]
    k_extra = {}
    if extra:
        from ..vibdvr import _harmonic, minimise
        from .vib_levels import _fit, _refs
        ref = _refs(run, dict(run.cfg['vib'], source='gs_grid'))
        well, _ = _fit(run, ref, 4, 0.6)
        Vf = lambda X: np.asarray(well.predict(X))[:, 0]
        x0 = minimise(Vf, well.center.numpy(), [tuple(well.box[c]) for c in well.names])
        K = _harmonic(Vf, x0, well.names, {})[0]
        for n in extra:
            j = well.names.index(n)
            k_extra[n] = (float(x0[j]), float(K[j, j]) * (np.pi / 180) ** 2 if n.startswith('a_') else float(K[j, j]))
    idx = [names.index(n) for n in mn]

    def est(X):
        X = np.atleast_2d(X)
        e = np.asarray(model.predict(X[:, idx]))[:, 0]
        for n, (c, k) in k_extra.items():
            e = e + 0.5 * k * (X[:, names.index(n)] - c) ** 2
        return e
    return est, f'{mpath.name} on {mn}' + (f' + harmonic in {extra}' if extra else '')


def _fps(Z, n, start):
    """Farthest-point sampling: indices of n points of Z, the first `start`."""
    sel = [int(start)]
    d = np.sum((Z - Z[start]) ** 2, 1)
    for _ in range(min(n, len(Z)) - 1):
        k = int(np.argmax(d))
        sel.append(k)
        d = np.minimum(d, np.sum((Z - Z[k]) ** 2, 1))
    return np.array(sel)


def design(run):
    """The shell candidates and their batches (deterministic: same config, same points)."""
    sc, gc = run.cfg['shell'], run.cfg['gs_grid']
    g = np.load(run.p('data', 'gs_grid.npz'), allow_pickle=False)
    names = [str(n) for n in g['coord_names']]
    Xg = g['coords']
    box, step = sc['box'], sc['step']
    miss = [n for n in names if n not in box or n not in step]
    if miss:
        raise SystemExit(f'[shell] box / step need every grid coordinate; missing {miss}')
    axes = [np.arange(float(box[n][0]), float(box[n][1]) + 1e-9, float(step[n])) for n in names]
    C = np.array(list(itertools.product(*axes)), float)
    lo, hi = Xg.min(0), Xg.max(0)
    inside = np.all((C >= lo - 1e-9) & (C <= hi + 1e-9), 1)
    C = C[~inside]
    est, how = _estimator(run, sc, names)
    E = est(C) if est is not None else np.zeros(len(C))
    if est is not None:
        Eg = est(Xg[np.isfinite(g['E_total_hartree'][:, 0])] if 'E_total_hartree' in g.files else Xg)
        e0 = float(np.nanmin(Eg))
        E = E - e0
        w0, w1 = map(float, sc['window_eV'])
        keep = (E >= w0) & (E <= w1)
        C, E = C[keep], E[keep]
    ls = np.array([float(step[n]) for n in names])
    Z = (C - Xg.mean(0)) / ls
    n = min(int(sc['n_points']), len(C))
    start = int(np.argmin(np.sum(Z ** 2, 1)))
    order = _fps(Z, n, start)
    nb = max(1, int(sc['batches']))
    batch = np.empty(n, int)
    for b in range(nb):                                         # round-robin over the FPS order: each batch spans all
        batch[b::nb] = b + 1
    return {'names': names, 'coords': C[order], 'estimate_eV': E[order], 'batch': batch, 'estimator': how,
            'n_candidates': int(len(C)), 'grid_range': [lo.tolist(), hi.tolist()], 'cand': C, 'cand_estimate_eV': E,
            'est': est}


def _delta_training(run, names, est, shell_X=None, shell_E=None, n_grid=400):
    """(X, Delta) for a GP: Delta = CCSD(T) - network estimate, both from their minimum over the dense grid, at a
    spread-out subset of the grid points plus the shell points given."""
    g = np.load(run.p('data', 'gs_grid.npz'), allow_pickle=False)
    Xg = g['coords']
    Eg = (g['E_total_hartree'] if 'E_total_hartree' in g.files else g['E_nevpt2_hartree'])[:, 0] * HARTREE_EV
    ok = np.isfinite(Eg)
    Xg, Eg = Xg[ok], Eg[ok]
    e0 = float(Eg.min())
    nn_g = est(Xg)
    n0 = float(nn_g.min())
    ls = np.array([float(run.cfg['shell']['step'].get(n, 1.0)) for n in names])
    if n_grid > 0:
        sub = _fps((Xg - Xg.mean(0)) / ls, min(n_grid, len(Xg)), int(np.argmin(Eg)))
    else:
        sub = np.zeros(0, int)                                 # shell points only
    X = Xg[sub].reshape(-1, len(names))
    y = (Eg[sub] - e0) - (nn_g[sub] - n0)
    if shell_X is not None and len(shell_X):
        X = np.vstack([X, shell_X])
        y = np.r_[y, (shell_E - e0) - (est(shell_X) - n0)]
    return X, y, e0, n0


def _gp_batch(run, d, b, finished_X, finished_E):
    """Batch b (>= 2) chosen where a GP of Delta fitted to what is done is least certain; cached in
    data/shell_gp_batch<b>.npz so the batch is the same when the stage is run again."""
    from ..gpdelta import GPDelta, select_uncertain
    cp = run.p('data', f'shell_gp_batch{b}.npz')
    if cp.exists():
        return np.load(cp)['coords']
    X, y, _, _ = _delta_training(run, d['names'], d['est'], finished_X, finished_E)
    gp = GPDelta.fit(X, y, d['names'])
    C = d['cand']
    done = {tuple(np.round(x, 4)) for x in finished_X}
    free = np.array([tuple(np.round(c, 4)) not in done for c in C])
    n = int(np.ceil(int(run.cfg['shell']['n_points']) / max(1, int(run.cfg['shell']['batches']))))
    pick = select_uncertain(gp, C[free], n)
    Xb = C[free][pick]
    _, sd = gp.predict(Xb)
    np.savez(cp, coords=Xb, gp_std_eV=sd)
    run.log(f'  batch {b} chosen by GP uncertainty: GP on {len(X)} points (length scales '
            + ', '.join(f'{k} {v:.3g}' for k, v in gp.meta['length_scales'].items())
            + f'); picked {len(Xb)} of {int(free.sum())} candidates, their std {sd.max() * 1000:.0f}-{sd.min() * 1000:.0f} meV')
    return Xb


def stage_shell(run, jobs=None, orca=None, batch=None):
    from ..geometry import ch2oo_xyz
    from .method_cut import cc_input, parse_cc
    sc, cc, gc = run.cfg['shell'], run.cfg['compute'], run.cfg['gs_grid']
    jobs = int(jobs or sc.get('jobs') or gc.get('jobs') or cc['jobs'])
    d = design(run)
    names, X = d['names'], d['coords']
    g = np.load(run.p('data', 'gs_grid.npz'), allow_pickle=False)
    fixed = json.loads(str(g['fixed']))
    basis = str(sc.get('basis') or (g['basis'] if 'basis' in g.files else 'aug-cc-pVDZ'))
    mode = str(sc.get('select', 'fps'))
    if mode == 'gp' and d['est'] is None:
        run.log('[shell] select = "gp" needs [shell] energy_model (the Delta is CCSD(T) - network): using fps')
        mode = 'fps'
    want = sorted(set(d['batch'])) if batch in (None, 0, 'all') else [int(batch)]
    if mode == 'gp':
        if batch in (None, 0, 'all'):
            raise SystemExit('[shell] select = "gp": run the batches one at a time (--batch 1, then 2, ...): each GP '
                             'batch is chosen from the results of the ones before')

        def put(bb, Xo):                                    # GP batch bb replaces fps batch bb
            m = d['batch'] != bb
            d['coords'] = np.vstack([d['coords'][m], Xo])
            d['estimate_eV'] = np.r_[d['estimate_eV'][m], d['est'](Xo)]
            d['batch'] = np.r_[d['batch'][m], np.full(len(Xo), bb)]
        for bb in range(2, int(sc['batches']) + 1):         # GP batches chosen earlier, as they were chosen
            cp = run.p('data', f'shell_gp_batch{bb}.npz')
            if cp.exists():
                put(bb, np.load(cp)['coords'])
        if int(batch) >= 2 and not run.p('data', f'shell_gp_batch{int(batch)}.npz').exists():
            prev = run.p('data', 'shell.npz')
            if not prev.exists():
                raise SystemExit('select = "gp": run --batch 1 first')
            pz = np.load(prev)
            put(int(batch), _gp_batch(run, d, int(batch), pz['coords'], pz['E_total_hartree'][:, 0] * HARTREE_EV))
    sel = np.where(np.isin(d['batch'], want))[0]
    names, X = d['names'], d['coords']
    run.log(f'shell design: {d["n_candidates"]} candidates outside the grid range, estimate {d["estimator"]}, '
            f'window {sc["window_eV"]} eV -> {len(X)} points in {int(sc["batches"])} batches (select: {mode}); '
            f'computing batch(es) {want}: {len(sel)} points, RHF-CCSD(T)/{basis}, {jobs} parallel')
    for j, n in enumerate(names):
        run.log(f'  {n}: {X[sel, j].min():.3f}-{X[sel, j].max():.3f}')
    exe = AU._orca_exe(run, orca or cc.get('orca') or run.cfg['audit'].get('orca') or None)
    wd = run.p('data', 'shell')
    wd.mkdir(exist_ok=True)
    key = lambda x: 's_' + '_'.join(f'{v:.4f}' for v in x).replace('-', 'm').replace('.', 'p')
    for k in sel:
        nm = key(X[k])
        if not (wd / f'{nm}.inp').exists():
            (wd / f'{nm}.inp').write_text(cc_input(ch2oo_xyz(**_geometry(names, X[k], fixed)), basis, cc['maxcore']))

    def one(k):
        nm = key(X[k])
        if parse_cc(wd / f'{nm}.out') is not None:
            return 'reused'
        return AU._run_one(exe, wd, nm, float(cc['timeout_s']))[1]
    t0 = time.time()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        for _ in ex.map(one, sel):
            done += 1
            if done % max(1, len(sel) // 10) == 0:
                run.log(f'  {done}/{len(sel)} points ({(time.time() - t0) / 60:.1f} min)')
    # merge: every shell point finished so far (earlier rows of data/shell.npz are kept) plus this run's
    rows = {}
    sp = run.p('data', 'shell.npz')
    if sp.exists():
        z = np.load(sp)
        for x, e, t, b, es in zip(z['coords'], z['E_total_hartree'][:, 0], z['T1'], z['batch'], z['estimate_eV']):
            rows[key(x)] = (x, e, t, b, es)
    for k in range(len(X)):
        r = parse_cc(wd / f'{key(X[k])}.out')
        if r is not None:
            rows[key(X[k])] = (X[k], r['E'], r.get('T1', np.nan), d['batch'][k], d['estimate_eV'][k])
    R = list(rows.values())
    Xs = np.array([r[0] for r in R]).reshape(-1, len(names))
    np.savez(sp, coords=Xs, coord_names=np.array(names), E_total_hartree=np.array([r[1] for r in R])[:, None],
             T1=np.array([r[2] for r in R]), batch=np.array([r[3] for r in R], int),
             estimate_eV=np.array([r[4] for r in R]), basis=basis, method='ccsd_t', fixed=json.dumps(fixed))
    T1a = np.array([r[2] for r in R])
    run.log(f'{len(R)} shell points done in total (batches {sorted(set(int(r[3]) for r in R))}); T1 '
            f'{np.nanmin(T1a):.3f}-{np.nanmax(T1a):.3f}; wrote data/shell.npz. Nothing else changed: next  shell_check')


def stage_shell_check(run):
    from ..wellfit import PolyWell
    sc = run.cfg['shell']
    p = run.p('data', 'shell.npz')
    if not p.exists():
        raise SystemExit(f'{p} not found: run the shell stage first')
    s = np.load(p, allow_pickle=False)
    names = [str(n) for n in s['coord_names']]
    X, Ecc = s['coords'], s['E_total_hartree'][:, 0] * HARTREE_EV
    g = np.load(run.p('data', 'gs_grid.npz'), allow_pickle=False)
    Eg = (g['E_total_hartree'] if 'E_total_hartree' in g.files else g['E_nevpt2_hartree'])[:, 0] * HARTREE_EV
    e0 = float(np.nanmin(Eg))                                   # same method and basis: one energy zero
    E = Ecc - e0
    out = {'n': int(len(X)), 'batches': sorted(set(s['batch'].tolist())), 'T1': [float(np.nanmin(s['T1'])),
                                                                                 float(np.nanmax(s['T1']))]}
    rows = {'E_ccsd_t_eV': E, 'T1': s['T1'], 'estimate_eV': s['estimate_eV']}
    # (a) the frozen benchmark polynomial: how far out it still holds
    bench = sc.get('benchmark', 'benchmark')
    bp = frozen_path(run, bench)
    if bp.exists():
        well = PolyWell.load(bp)
        Xw = X[:, [names.index(n) for n in well.names]]
        pw = well.predict(Xw)[:, 0]
        ing = well.inside(Xw)
        # put the polynomial on the CCSD(T) zero through the grid points it was fitted to
        gx = g['coords'][:, [names.index(n) for n in well.names]]
        okg = np.isfinite(Eg) & well.inside(gx)
        off = float(np.median((Eg[okg] - e0) - well.predict(gx[okg])[:, 0])) if okg.any() else 0.0
        err = pw + off - E
        rows['benchmark_minus_ccsd_t_eV'] = err
        rows['inside_benchmark_box'] = ing.astype(float)
        bins = [(0.0, 0.6), (0.6, 1.0), (1.0, 1.5), (1.5, 9.0)]
        def _rms(sel):
            return {f'{a}-{b}': (float(np.sqrt(np.mean(err[(E >= a) & (E < b) & sel] ** 2)) * 1000)
                                 if ((E >= a) & (E < b) & sel).any() else None) for a, b in bins}
        out['benchmark'] = {'name': bench, 'inside_box': int(ing.sum()),
                            'rms_meV_by_energy': _rms(ing),                       # inside its box (usually none here)
                            'rms_meV_by_energy_outside': _rms(~ing)}             # extrapolation: how far it can be trusted
        txt = lambda d: ', '.join(f'{k} eV {v:.0f} meV' for k, v in d.items() if v is not None) or 'none'
        run.log(f'benchmark {bench}: {int(ing.sum())}/{len(X)} shell points inside its fitted box (RMS by CCSD(T) '
                f'energy: {txt(out["benchmark"]["rms_meV_by_energy"])}); outside it (extrapolated): '
                f'{txt(out["benchmark"]["rms_meV_by_energy_outside"])}')
    else:
        run.log(f'no frozen benchmark {bp.name}: make one with  freeze {run.path} --name {bench}')
    # (b) the Delta the network will need
    est, how = _estimator(run, sc, names)
    if est is not None:
        nn = est(X)
        nn_g = est(g['coords'][np.isfinite(Eg)])
        ref0 = float(np.nanmin(nn_g))
        delta = E - (nn - ref0)
        rows['nn_estimate_eV'] = nn - ref0
        rows['delta_eV'] = delta
        jo = names.index('r_OO')
        bins = np.arange(1.15, 1.81, 0.05)
        stat = []
        for a, b in zip(bins[:-1], bins[1:]):
            m = (X[:, jo] >= a) & (X[:, jo] < b)
            if m.any():
                stat.append({'r_OO': [float(a), float(b)], 'n': int(m.sum()), 'mean_meV': float(delta[m].mean() * 1000),
                             'std_meV': float(delta[m].std() * 1000), 'T1_max': float(np.nanmax(s['T1'][m]))})
        out['delta'] = {'estimator': how, 'mean_meV': float(delta.mean() * 1000), 'std_meV': float(delta.std() * 1000),
                        'by_r_OO': stat,
                        'corr_with_T1': float(np.corrcoef(delta, s['T1'])[0, 1]) if len(X) > 2 else None}
        run.log(f'Delta = CCSD(T) - network ({how}): mean {delta.mean() * 1000:.0f}, spread (std) '
                f'{delta.std() * 1000:.0f} meV over {len(X)} points; by r(O-O): '
                + '; '.join(f'{d["r_OO"][0]:.2f} {d["mean_meV"]:+.0f}±{d["std_meV"]:.0f} (T1 ≤ {d["T1_max"]:.3f})'
                            for d in stat))
    # (c) a GP of Delta: how well it predicts each batch it has not seen, and whether its std is honest
    if est is not None and len(X) >= 8:
        from ..gpdelta import GPDelta
        bt = s['batch']
        cal = []
        for b in sorted(set(bt.tolist())):
            tr = bt != b
            if tr.sum() < 4 or (~tr).sum() < 2:
                continue
            Xt, yt, _, _ = _delta_training(run, names, est, X[tr], Ecc[tr])
            gp = GPDelta.fit(Xt, yt, names)
            _, yh, _, _ = _delta_training(run, names, est, X[~tr], Ecc[~tr], n_grid=0)
            mu, sd = gp.predict(X[~tr])
            err = mu - yh
            cal.append({'held_out_batch': int(b), 'n': int((~tr).sum()), 'rmse_meV': float(np.sqrt(np.mean(err ** 2)) * 1000),
                        'mean_std_meV': float(sd.mean() * 1000), 'within_2std': float(np.mean(np.abs(err) <= 2 * sd))})
        if cal:
            out['gp_calibration'] = cal
            run.log('GP of Delta, each batch held out: ' + '; '.join(
                f'batch {c["held_out_batch"]}: RMSE {c["rmse_meV"]:.1f} meV, mean std {c["mean_std_meV"]:.1f} meV, '
                f'{100 * c["within_2std"]:.0f}% within 2 std' for c in cal)
                + ' (about 95% within 2 std means the std can be trusted)')
        Xa, ya, _, _ = _delta_training(run, names, est, X, Ecc)
        gp = GPDelta.fit(Xa, ya, names)
        gp.save(run.p('models', 'delta_gp_candidate.npz'))
        out['gp'] = gp.meta
        run.log(f'GP of Delta on all {len(Xa)} points -> models/delta_gp_candidate.npz (a candidate only: no stage uses '
                'it); length scales ' + ', '.join(f'{k} {v:.3g}' for k, v in gp.meta['length_scales'].items()))
    run.p('results', 'shell_check.json').write_text(json.dumps(out, indent=1, default=float))
    with open(run.p('results', 'shell_check.csv'), 'w') as fh:
        cols = list(rows)
        fh.write(','.join(names + cols) + '\n')
        for i in range(len(X)):
            fh.write(','.join([f'{v:.4f}' for v in X[i]] + [f'{float(rows[c][i]):.6f}' for c in cols]) + '\n')
    _figure(run, X, names, rows, s['T1'])
    run.log('wrote results/shell_check.{json,csv}, figures/SH1_shell_check.png (no model changed)')


def _figure(run, X, names, rows, T1):
    plt = W._mpl()
    jo = names.index('r_OO')
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    sc0 = ax[0].scatter(X[:, jo], rows['E_ccsd_t_eV'], c=T1, s=10, cmap='viridis')
    fig.colorbar(sc0, ax=ax[0], label='T1')
    ax[0].set(xlabel='r(O–O) (Å)', ylabel='CCSD(T) energy above the minimum (eV)', title='Shell points')
    if 'benchmark_minus_ccsd_t_eV' in rows:
        m = rows['inside_benchmark_box'] > 0
        ax[1].scatter(rows['E_ccsd_t_eV'][~m], rows['benchmark_minus_ccsd_t_eV'][~m] * 1000, s=8, color='#C8873A',
                      label='outside its box (extrapolated)')
        if m.any():
            ax[1].scatter(rows['E_ccsd_t_eV'][m], rows['benchmark_minus_ccsd_t_eV'][m] * 1000, s=10, color='#2F6DB5',
                          label='inside its box')
        ax[1].axhline(0, color='0.5', lw=0.8)
        ax[1].legend(frameon=False, fontsize=9)
        ax[1].set(xlabel='CCSD(T) energy (eV)', ylabel='benchmark − CCSD(T) (meV)',
                  title='Frozen benchmark polynomial at the shell points')
    else:
        ax[1].text(0.5, 0.5, 'no frozen benchmark', ha='center', transform=ax[1].transAxes)
    if 'delta_eV' in rows:
        s2 = ax[2].scatter(X[:, jo], rows['delta_eV'] * 1000, c=T1, s=10, cmap='viridis')
        fig.colorbar(s2, ax=ax[2], label='T1')
        ax[2].set(xlabel='r(O–O) (Å)', ylabel='Δ = CCSD(T) − network S0 (meV)', title='What the Δ model must learn')
    else:
        ax[2].text(0.5, 0.5, 'no [shell] energy_model', ha='center', transform=ax[2].transAxes)
    fig.tight_layout()
    fig.savefig(run.p('figures', 'SH1_shell_check.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)
