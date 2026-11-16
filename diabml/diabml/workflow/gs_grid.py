"""A dense, smooth ground-state grid around the S0 minimum, for vibrational spectroscopy.

    python -m diabml.workflow.ch2oo gs_grid    RUN [--jobs 6]   # ORCA -> RUN/data/gs_grid.npz, results/gs_grid.json
    python -m diabml.workflow.ch2oo vib_levels RUN              # with [vib] source = "gs_grid"

Why: the reference data of the quasi-diabatic surfaces are SA4-CASSCF(8,6)+NEVPT2. The top root of that average
(9-11 eV up for CH2OO) changes character between neighbouring geometries; the averaged orbitals change with it and S0
moves by 5-50 meV from point to point. That is harmless for the excited-state surfaces but destroys the curvature at
the bottom of the well, which is all a vibrational frequency depends on (vib_levels: the harmonic O-O frequency of the
24_spec data ranges over 915-1161 cm-1 depending on how the polynomial is fitted).

What it does: a regular grid of (2 half_points + 1) values per coordinate around `center` (default: the minimum of the
cubic well polynomial of the run's data) with spacing `step`, computed with ORCA as a ground-state CASSCF(8,6)
(`nroots` roots averaged; 1 = state-specific) + SC-NEVPT2 in the run's basis. The points are visited along a spanning
tree from the centre; each starts from the converged orbitals of its parent and of its nearest other finished point
(MORead), and the lower CASSCF energy is kept (state-specific CASSCF: the lower solution is the ground state). The
seed starts from the run's orbital library (the SA4 orbitals nearest the centre) if there is one.

Extra coordinates: [gs_grid.extra] adds coordinates the reference data hold fixed, e.g. the symmetric H-C-O angle
(both H atoms; the CH2 scissor that mixes with the C-O stretch):

    [gs_grid.extra.a_HCO]
    center = 120.0
    step = 3.0
    half_points = 4

The grid is then 4-D and so are the well fits of vib_levels and ir that use it.

Method: [gs_grid] method = "ccsd_t" computes RHF-CCSD(T) in the same basis instead (no active space, points run
independently, T1 diagnostic kept): the smooth single-reference ground state for band positions. The method_cut stage
showed that SC- and FIC-NEVPT2 on the (8,6) reference stiffen the outer O-O wall where CCSD(T) does not.

Output: data/gs_grid.npz (coords, coord_names, E_casscf_hartree, E_nevpt2_hartree, fixed), results/gs_grid.json
(smoothness: RMS third difference of S0 along grid lines; residual of quartic fits; how often the two starts
disagreed).
"""
from __future__ import annotations

import concurrent.futures as cf
import itertools
import json
import time

import numpy as np

from . import audit as AU
from . import ch2oo as W

HARTREE_EV = AU.HARTREE_EV


def grid_points(names, center, step, half_points):
    hp = np.broadcast_to(np.asarray(half_points, int), (len(names),))
    axes = [center[j] + step[j] * np.arange(-hp[j], hp[j] + 1) for j in range(len(names))]
    return np.array(list(itertools.product(*axes)), float), axes


def _geometry(names, x, fixed):
    kw = {W.KW.get(k, k.lower()): float(v) for k, v in fixed.items()}
    kw.update({W.KW[n]: float(v) for n, v in zip(names, x)})
    return kw


def _center(run, ref, gc):
    """Centre in the reference data's coordinates ([gs_grid] center, else the minimum of their cubic well fit)."""
    if gc.get('center'):
        return np.array([float(gc['center'][n]) for n in ref.coord_names])
    from ..vibdvr import harmonic_frequencies
    from ..wellfit import fit_well
    w, _ = fit_well(ref, 3, 0.6)
    _, x0 = harmonic_frequencies(w)
    return np.round(x0, 3)


def stage_gs_grid(run, jobs=None, orca=None):
    from ..geometry import ch2oo_xyz
    from ..refset import RefSet
    from .orca_compute import load_library
    from .track import line_roughness, spanning_tree
    gc, cc = run.cfg['gs_grid'], run.cfg['compute']
    jobs = int(jobs or gc.get('jobs') or cc['jobs'])
    ref = W.load_refs(run)
    extra = gc.get('extra') or {}
    names = list(ref.coord_names) + [n for n in extra if n not in ref.coord_names]
    unknown = [n for n in extra if n not in W.KW]
    if unknown:
        raise SystemExit(f'[gs_grid.extra]: unknown coordinates {unknown}; allowed {sorted(W.KW)}')
    fixed = {k: v for k, v in ref.meta.get('fixed', {}).items() if k not in names}
    spec = dict(ref.meta['spec'], nroots=int(gc['nroots']))
    if gc.get('basis'):
        spec['basis'] = str(gc['basis'])                         # e.g. aug-cc-pVDZ: a basis-set check of the curvature
    c = np.concatenate([_center(run, ref, gc), [float(extra[n]['center']) for n in names[len(ref.coord_names):]]])
    step = np.array([float(gc['step'].get(n, 0.02)) for n in ref.coord_names] +
                    [float(extra[n]['step']) for n in names[len(ref.coord_names):]])
    hp = [int(gc['half_points'])] * len(ref.coord_names) + [int(extra[n].get('half_points', gc['half_points']))
                                                           for n in names[len(ref.coord_names):]]
    X, axes = grid_points(names, c, step, hp)
    exe = AU._orca_exe(run, orca or cc.get('orca') or run.cfg['audit'].get('orca') or None)
    Z = (X - c) / step
    seed = int(np.argmin(np.sum(Z ** 2, 1)))
    parent, depth = spanning_tree(Z, seed)
    wd = run.p('data', 'gs_grid')
    wd.mkdir(exist_ok=True)
    nr = int(spec['nroots'])
    atoms = [ch2oo_xyz(**_geometry(names, x, fixed)) for x in X]
    method = str(gc.get('method', 'nevpt2')).lower()
    if method == 'ccsd_t':
        return _cc_grid(run, gc, cc, X, names, axes, step, c, fixed, spec, atoms, exe, jobs, wd)
    if method != 'nevpt2':
        raise SystemExit(f'[gs_grid] method = "{method}": use "nevpt2" or "ccsd_t"')
    run.log(f'{len(X)} points: ' + ', '.join(f'{n} {a[0]:.3f}-{a[-1]:.3f} step {s:g}' for n, a, s in
                                              zip(names, axes, step))
            + f'; SA{nr}-CASSCF({spec["nelecas"]},{spec["ncas"]})/{spec["basis"]} + SC-NEVPT2; ORCA {exe}, {jobs} '
              f'parallel; spanning tree of depth {depth.max()}')
    lib_x, lib_g = load_library(run.p('data', 'orca_orbitals.json'))

    def run_batch(tasks):
        with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
            list(ex.map(lambda nm: AU._run_one(exe, wd, nm, float(cc['timeout_s'])), tasks))

    def write(k, a, moread):
        nm = f'g{k:04d}_{a}'
        if not (wd / f'{nm}.inp').exists():
            (wd / f'{nm}.inp').write_text(AU.orca_input(atoms[k], spec, moread=moread, qd=False,
                                                        maxcore=cc['maxcore']))
        return nm

    def ok(r):
        return r is not None and len(r['casscf']) >= nr and len(r['sc']) >= nr

    t0 = time.time()
    res, gbw = {}, {}
    disagree = np.full(len(X), np.nan)
    starts = [None]
    if len(lib_g):
        # the library holds the reference data's coordinates only (extra coordinates sit at their fixed values there)
        nl = np.asarray(lib_x).shape[1]
        sc = np.where(step[:nl] > 0, step[:nl], 1.0)
        starts.append(lib_g[int(np.argmin(np.sum(((np.asarray(lib_x) - c[:nl]) / sc) ** 2, 1)))])
    names_seed = [write(seed, a, s) for a, s in enumerate(starts)]
    run_batch(names_seed)
    cand = [(AU.parse_orca(wd / f'{nm}.out', nr), nm) for nm in names_seed]
    cand = [(r, nm) for r, nm in cand if ok(r)]
    if not cand:
        raise SystemExit(f'the centre point failed (see {wd}/{names_seed[0]}.out)')
    cand.sort(key=lambda t: t[0]['casscf'][0])
    res[seed], gbw[seed] = cand[0][0], str((wd / f'{cand[0][1]}.gbw').resolve())
    if len(cand) > 1:
        disagree[seed] = (cand[-1][0]['casscf'][0] - cand[0][0]['casscf'][0]) * 1000
    for lev in range(1, depth.max() + 1):
        layer = [k for k in range(len(X)) if depth[k] == lev]
        done = np.array(sorted(gbw))
        plan, tasks = {}, []
        for k in layer:
            p = parent[k] if parent[k] in gbw else int(done[np.argmin(np.sum((Z[done] - Z[k]) ** 2, 1))])
            st = [p]
            o = done[done != p]
            if len(o) and int(gc['neighbours']) > 1:
                st += [int(x) for x in o[np.argsort(np.sum((Z[o] - Z[k]) ** 2, 1))[:int(gc['neighbours']) - 1]]]
            plan[k] = [write(k, a, gbw[s]) for a, s in enumerate(st)]
            tasks += plan[k]
        run_batch(tasks)
        for k in layer:
            cand = [(AU.parse_orca(wd / f'{nm}.out', nr), nm) for nm in plan[k]]
            cand = [(r, nm) for r, nm in cand if ok(r)]
            if not cand:
                continue
            cand.sort(key=lambda t: t[0]['casscf'][0])
            res[k], gbw[k] = cand[0][0], str((wd / f'{cand[0][1]}.gbw').resolve())
            if len(cand) > 1:
                disagree[k] = (cand[-1][0]['casscf'][0] - cand[0][0]['casscf'][0]) * 1000
        run.log(f'  level {lev}/{depth.max()}: {len(layer)} points ({len(res)}/{len(X)} done, '
                f'{(time.time() - t0) / 60:.1f} min)')
    Ec = np.full((len(X), nr), np.nan)
    En = np.full((len(X), nr), np.nan)
    for k, r in res.items():
        Ec[k] = np.sort(r['casscf'][:nr])
        En[k] = r['sc'][:nr]
    np.savez(run.p('data', 'gs_grid.npz'), coords=X, coord_names=np.array(names), E_casscf_hartree=Ec,
             E_nevpt2_hartree=En, center=c, step=step, fixed=json.dumps(fixed), basis=str(spec['basis']),
             spec=json.dumps(spec, default=str))
    info = summarize(X, names, En[:, 0], Ec[:, 0], step)
    info.update({'n': int(len(X)), 'n_done': int(len(res)), 'center': c.tolist(), 'step': step.tolist(),
                 'nroots': nr, 'minutes': (time.time() - t0) / 60,
                 'starts_disagree_mEh': {'n_compared': int(np.isfinite(disagree).sum()),
                                         'n_above_0.05': int(np.nansum(disagree > 0.05)),
                                         'max': float(np.nanmax(disagree)) if np.isfinite(disagree).any() else 0.0}})
    run.p('results', 'gs_grid.json').write_text(json.dumps(info, indent=1))
    run.log(f'{len(res)}/{len(X)} points in {info["minutes"]:.0f} min; two starts differed by > 0.05 mEh at '
            f'{info["starts_disagree_mEh"]["n_above_0.05"]} points (max {info["starts_disagree_mEh"]["max"]:.3f} mEh)')
    run.log(f'smoothness (RMS third difference along grid lines): NEVPT2 S0 {info["roughness_meV"]["nevpt2"]:.2f} meV, '
            f'CASSCF {info["roughness_meV"]["casscf"]:.2f} meV; quartic fit residual '
            f'{info["quartic_residual_meV"]:.2f} meV')
    run.log(f'wrote data/gs_grid.npz; next: vib_levels with [vib] source = "gs_grid"')


def summarize(X, names, En, Ec, step, labels=('nevpt2', 'casscf')):
    ok = np.isfinite(En) & np.isfinite(Ec)
    rn, nl = line_roughness_safe(X[ok], (En[ok] - np.nanmin(En)) * HARTREE_EV * 1000)
    rc, _ = line_roughness_safe(X[ok], (Ec[ok] - np.nanmin(Ec)) * HARTREE_EV * 1000)
    from ..wellfit import _powers
    z = (X[ok] - X[ok].mean(0)) / step
    A = np.prod(z[:, None, :] ** np.array(_powers(len(names), 4))[None], 2)
    y = (En[ok] - np.nanmin(En)) * HARTREE_EV
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {'roughness_meV': {labels[0]: rn, labels[1]: rc}, 'grid_lines': nl,
            'quartic_residual_meV': float(np.sqrt(np.mean((A @ coef - y) ** 2)) * 1000),
            'S0_range_eV': float(np.nanmax(y))}


def line_roughness_safe(X, E):
    from .track import line_roughness
    if len(X) < 4:
        return float('nan'), 0
    return line_roughness(X, E)


def load_gs_grid(run, ref):
    """The grid as a RefSet with one state (S0 at the grid's level: NEVPT2, or CCSD(T) with [gs_grid] method =
    "ccsd_t"; eV above the grid minimum), for fit_well."""
    from ..refset import RefSet
    p = run.p('data', 'gs_grid.npz')
    if not p.exists():
        raise SystemExit(f'{p} not found: run the gs_grid stage first')
    z = np.load(p, allow_pickle=False)
    E = (z['E_total_hartree'] if 'E_total_hartree' in z.files else z['E_nevpt2_hartree'])[:, :1] * HARTREE_EV
    X = z['coords']
    names = [str(n) for n in z['coord_names']]
    dc = run.cfg['gs_grid'].get('delta') or {}
    method = 'gs_grid'
    if dc.get('run'):
        dE, info = delta_correction(run, X, names, E[:, 0], dc)
        info.pop('_function', None)
        E = E + dE[:, None]
        method = f'gs_grid + delta ({info["basis"]} - {info["basis_base"]}, order {info["order"]})'
    E = E - np.nanmin(E)
    ok = np.isfinite(E[:, 0])
    meta = dict(ref.meta, fixed=json.loads(str(z['fixed'])), method=method)
    return RefSet('gs_grid', X[ok], E[ok], names, extra={}, meta=meta)


def delta_correction(run, X, names, E_eV, dc):
    """[gs_grid.delta] run = "<another run with a gs_grid at a higher level, on a subset of this grid's points>":
    the difference (theirs - ours) at the shared points, fitted by a polynomial of `order` (default 2) in the
    coordinates and added to every point of this grid. A small high-level grid (e.g. CCSD(T)/aug-cc-pVTZ, 3^4 points)
    then corrects the curvature of a dense lower-level one (CCSD(T)/aug-cc-pVDZ). Returns (correction in eV, info);
    logs the fit residual and how large the correction is."""
    from ..wellfit import _SCALE, _powers
    other = W.C.resolve(str(dc['run']), run.path)
    q = other / 'data' / 'gs_grid.npz' if other.is_dir() else other
    if not q.exists():
        raise SystemExit(f'[gs_grid.delta] run: {q} not found')
    zo = np.load(q, allow_pickle=False)
    no = [str(n) for n in zo['coord_names']]
    if no != names:
        raise SystemExit(f'[gs_grid.delta]: coordinates {no} differ from this grid\'s {names}')
    Eo = (zo['E_total_hartree'] if 'E_total_hartree' in zo.files else zo['E_nevpt2_hartree'])[:, 0] * HARTREE_EV
    key = {tuple(np.round(x, 5)): i for i, x in enumerate(X)}
    pairs = [(key[tuple(np.round(x, 5))], j) for j, x in enumerate(zo['coords'])
             if tuple(np.round(x, 5)) in key and np.isfinite(Eo[j])]
    pairs = [(i, j) for i, j in pairs if np.isfinite(E_eV[i])]
    order = int(dc.get('order', 2))
    pw = np.array(_powers(len(names), order))
    if len(pairs) < len(pw) + 3:
        raise SystemExit(f'[gs_grid.delta]: {len(pairs)} shared points, too few for an order-{order} fit '
                         f'({len(pw)} terms); the delta grid must lie on this grid\'s points')
    ii, jj = np.array(pairs).T
    d = Eo[jj] - E_eV[ii]
    c = X[ii].mean(0)
    sc = np.array([_SCALE.get(n, 1.0) for n in names])
    A = lambda Y: np.prod(((Y - c) / sc)[:, None, :] ** pw[None], 2)
    coef, *_ = np.linalg.lstsq(A(X[ii]), d, rcond=None)
    res = A(X[ii]) @ coef - d
    out = np.full(len(X), np.nan)
    good = np.all(np.isfinite(X), 1)
    out[good] = A(X[good]) @ coef
    shift = float(np.nanmean(out[ii]))
    out -= shift                                                 # only the shape matters
    span = {n: [float(X[ii, k].min()), float(X[ii, k].max())] for k, n in enumerate(names)}
    info = {'run': str(q), 'n_shared': int(len(pairs)), 'order': order, 'residual_meV': float(np.sqrt(np.mean(res ** 2))
            * 1000), 'range_on_shared_meV': float(np.ptp(d) * 1000), 'span': span,
            'basis': str(zo['basis']) if 'basis' in zo.files else '?',
            'basis_base': str(np.load(run.p('data', 'gs_grid.npz'))['basis']) if 'basis' in np.load(run.p('data', 'gs_grid.npz')).files else '?'}
    run.log(f'delta correction from {q.parent.parent.name if q.name == "gs_grid.npz" else q}: {info["basis"]} - '
            f'{info["basis_base"]} at {info["n_shared"]} shared points, order-{order} fit, residual '
            f'{info["residual_meV"]:.2f} meV, range {info["range_on_shared_meV"]:.0f} meV over '
            + ', '.join(f'{n} {a:g}-{b:g}' for n, (a, b) in span.items())
            + ' (outside that span the correction is extrapolated)')
    info['_function'] = lambda Y: A(np.atleast_2d(np.asarray(Y, float))) @ coef - shift
    return out, info


def delta_function(run, dc=None):
    """The [gs_grid.delta] correction as a function of this run's grid coordinates: f(X) in eV, the same
    polynomial delta_correction adds to the grid (fitted at the shared grid points, zero on average there), for
    evaluating it at geometries that are not grid points (well9d). Returns (f, info, grid coordinate names), or
    (None, None, names) when no [gs_grid.delta] run is configured."""
    z = np.load(run.p('data', 'gs_grid.npz'), allow_pickle=False)
    names = [str(n) for n in z['coord_names']]
    if dc is None:
        dc = run.cfg['gs_grid'].get('delta') or {}
    if not dc.get('run'):
        return None, None, names
    E = (z['E_total_hartree'] if 'E_total_hartree' in z.files else z['E_nevpt2_hartree'])[:, 0] * HARTREE_EV
    _, info = delta_correction(run, z['coords'], names, E, dc)
    f = info.pop('_function')
    return f, info, names


def _cc_grid(run, gc, cc, X, names, axes, step, c, fixed, spec, atoms, exe, jobs, wd):
    """[gs_grid] method = "ccsd_t": RHF-CCSD(T) at every grid point. Single reference, no orbitals to follow, so the
    points run independently; ORCA's T1 diagnostic is kept per point (CH2OO: about 0.04-0.05, above the 0.02 rule of
    thumb for closed shells, so read CCSD(T) as the better ground-state well, not as exact)."""
    from .method_cut import cc_input, parse_cc
    basis = str(spec['basis'])
    run.log(f'{len(X)} points: ' + ', '.join(f'{n} {a[0]:.3f}-{a[-1]:.3f} step {s:g}' for n, a, s in
                                              zip(names, axes, step))
            + f'; RHF-CCSD(T)/{basis}; ORCA {exe}, {jobs} parallel')
    names_k = []
    for k in range(len(X)):
        nm = f'c{k:04d}'
        if not (wd / f'{nm}.inp').exists():
            (wd / f'{nm}.inp').write_text(cc_input(atoms[k], basis, cc['maxcore']))
        names_k.append(nm)
    t0 = time.time()
    k0 = int(np.argmin(np.sum(((X - c) / step) ** 2, 1)))

    def one(nm):
        if parse_cc(wd / f'{nm}.out') is not None:
            return nm, 'reused'
        return AU._run_one(exe, wd, nm, float(cc['timeout_s']))
    one(names_k[k0])                                        # the centre first: stop early on a bad input
    r0 = parse_cc(wd / f'{names_k[k0]}.out')
    if r0 is None:
        raise SystemExit(f'CCSD(T) at the centre failed or its output was not understood: see {wd}/{names_k[k0]}.out')
    run.log(f'  centre: CCSD(T) {r0["E"]:.8f} Eh' + (f', T1 {r0["T1"]:.4f}' if 'T1' in r0 else ''))
    done = 0
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        for _ in ex.map(one, [nm for k, nm in enumerate(names_k) if k != k0]):
            done += 1
            if done % max(1, len(X) // 20) == 0:
                run.log(f'  {done + 1}/{len(X)} points ({(time.time() - t0) / 60:.1f} min)')
    E = np.full((len(X), 1), np.nan)
    Es = np.full((len(X), 1), np.nan)
    T1 = np.full(len(X), np.nan)
    for k, nm in enumerate(names_k):
        r = parse_cc(wd / f'{nm}.out')
        if r is None:
            continue
        E[k, 0] = r['E']
        Es[k, 0] = r.get('E_scf', np.nan)
        T1[k] = r.get('T1', np.nan)
    n_ok = int(np.isfinite(E[:, 0]).sum())
    np.savez(run.p('data', 'gs_grid.npz'), coords=X, coord_names=np.array(names), E_total_hartree=E,
             E_scf_hartree=Es, T1=T1, method='ccsd_t', center=c, step=step, fixed=json.dumps(fixed), basis=basis,
             spec=json.dumps(dict(spec, method='ccsd_t'), default=str))
    info = summarize(X, names, E[:, 0], np.where(np.isfinite(Es[:, 0]), Es[:, 0], E[:, 0]), step,
                     labels=('ccsd_t', 'scf'))
    info.update({'method': 'ccsd_t', 'basis': basis, 'n': int(len(X)), 'n_done': n_ok, 'center': c.tolist(),
                 'step': step.tolist(), 'minutes': (time.time() - t0) / 60,
                 'T1': {'min': float(np.nanmin(T1)) if np.isfinite(T1).any() else None,
                        'max': float(np.nanmax(T1)) if np.isfinite(T1).any() else None}})
    run.p('results', 'gs_grid.json').write_text(json.dumps(info, indent=1))
    run.log(f'{n_ok}/{len(X)} points in {info["minutes"]:.0f} min; T1 {info["T1"]["min"]}-{info["T1"]["max"]}')
    run.log(f'smoothness (RMS third difference along grid lines): CCSD(T) {info["roughness_meV"]["ccsd_t"]:.2f} meV, '
            f'SCF {info["roughness_meV"]["scf"]:.2f} meV; quartic fit residual {info["quartic_residual_meV"]:.2f} meV')
    run.log('wrote data/gs_grid.npz (CCSD(T)); next: vib_levels with [vib] source = "gs_grid"')
