"""Follow one SA-CASSCF solution through a region, so the reference energies there are smooth.

    python -m diabml.workflow.ch2oo track RUN [--jobs 6]     # -> RUN/data/tracked/refs.npz (+ orbitals, couplings)

Why: near the top of the state average (S3 of SA4-CASSCF lies 9-11 eV up, among many states) SA-CASSCF has several
solutions that differ in which upper state is averaged. Keeping the lowest state average at every point (audit,
compute) can switch solutions between neighbouring points; the upper states jump by 0.3-1 eV and S0 by 10-30 meV,
which ruins the curvature a vibrational spectrum needs.

What it does, for every reference point inside the [track] box (default: around the S0 minimum):
  1. a spanning tree over the points (nearest neighbours, in grid steps of [active.length_scales]) grown from a seed
     (default: the point with the lowest S0 in the box), processed level by level, `jobs` ORCA runs at once;
  2. each point is computed with ORCA from the converged orbitals of its parent in the tree and of its nearest other
     already-tracked point (MORead); of these, the solution whose SA-CASSCF excitation energies continue those of the
     parent most closely is kept, and its orbitals are passed on;
  3. a jump larger than jump_eV in any excitation energy between a point and its parent is flagged (the solution
     could not be followed there).
The tracked energies (SA-CASSCF, SC- and QD-NEVPT2, at the data's level and zero) replace the stored ones inside the
box; points outside are unchanged. Per-state PySCF diagnostics and couplings at points whose energies changed are
dropped (they belong to the other solution). Writes RUN/data/tracked/ (refs.npz, orca_orbitals.json, nacs.npz),
results/track.csv, results/track.json and figures/TR1_track.png (S0 along grid lines before and after; S3).

Smoothness is measured on grid lines: the RMS third difference of S0 along every line of at least four equally
spaced points (meV), before and after (curvature does not count, steps and kinks do).
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import time

import numpy as np

from . import active as A
from . import audit as AU
from . import ch2oo as W

HARTREE_EV = AU.HARTREE_EV
CODE_TRACKED = 5
DEFAULT_BOX = {'r_OO': [1.2, 1.5], 'a_COO': [105.0, 130.0], 'r_CO': [1.15, 1.40]}


def spanning_tree(Z, seed):
    """Prim's minimum spanning tree over points Z (scaled), rooted at `seed`. Returns (parent, depth)."""
    n = len(Z)
    parent, depth = np.full(n, -1), np.zeros(n, int)
    inside = np.zeros(n, bool)
    inside[seed] = True
    best = np.sum((Z - Z[seed]) ** 2, 1)
    src = np.full(n, seed)
    best[seed] = np.inf
    for _ in range(n - 1):
        j = int(np.argmin(np.where(inside, np.inf, best)))
        inside[j] = True
        parent[j], depth[j] = src[j], depth[src[j]] + 1
        d = np.sum((Z - Z[j]) ** 2, 1)
        upd = (~inside) & (d < best)
        best[upd], src[upd] = d[upd], j
    return parent, depth


def line_roughness(X, E, tol=1e-6):
    """RMS third difference (same units as E) along grid lines: runs of >= 4 equally spaced points that differ in
    one coordinate only. A smooth curve, curvature included, has small third differences; a solution switch (a step
    or kink) gives large ones. Returns (rms, number of third differences)."""
    d3 = []
    n, k = X.shape
    for j in range(k):
        others = [c for c in range(k) if c != j]
        key = {}
        for i in range(n):
            key.setdefault(tuple(np.round(X[i, others], 6)), []).append(i)
        for idx in key.values():
            if len(idx) < 4:
                continue
            idx = sorted(idx, key=lambda i: X[i, j])
            x = X[idx, j]
            for a in range(len(idx) - 3):
                h = np.diff(x[a:a + 4])
                if h.min() > 0 and np.ptp(h) < tol * max(1.0, h.max()) + 1e-9:
                    e = E[idx[a:a + 4]]
                    d3.append(e[3] - 3 * e[2] + 3 * e[1] - e[0])
    d3 = np.array(d3)
    return (float(np.sqrt(np.mean(d3 ** 2))) if len(d3) else float('nan')), int(len(d3))


def stage_track(run, jobs=None, orca=None):
    from ..geometry import ch2oo_xyz
    from ..refset import RefSet
    from .orca_compute import load_library, save_library
    tc, cc = run.cfg['track'], run.cfg['compute']
    jobs = int(jobs or tc.get('jobs') or cc['jobs'])
    ref = RefSet.load(run.p('data', 'refs.npz'))
    spec, zero = ref.meta['spec'], ref.meta['zero_hartree']
    level = ref.meta.get('energy_level', 'casscf')
    nr = int(spec['nroots'])
    X = ref.coords
    box = tc.get('box') or DEFAULT_BOX
    inb = np.ones(len(ref), bool)
    for n, (lo, hi) in box.items():
        if n in ref.coord_names:
            inb &= (X[:, ref.coord_names.index(n)] >= float(lo)) & (X[:, ref.coord_names.index(n)] <= float(hi))
    pts = np.where(inb & np.isfinite(ref.energies[:, 0]))[0]
    if len(pts) < 2:
        raise SystemExit(f'fewer than 2 reference points inside the box {box}')
    exe = AU._orca_exe(run, orca or cc.get('orca') or run.cfg['audit'].get('orca') or None)
    sc = A._scales(ref.coord_names, A._ac(run))
    Z = X[pts] / sc
    E0 = np.sort(ref.energies[pts], 1)[:, 0]
    if tc.get('seed'):
        s = np.array([float(tc['seed'][n]) for n in ref.coord_names]) / sc
        seed = int(np.argmin(np.sum((Z - s) ** 2, 1)))
    else:
        seed = int(np.argmin(E0))
    parent, depth = spanning_tree(Z, seed)
    lib_x, lib_g = load_library(run.p('data', 'orca_orbitals.json'))
    wd = run.p('data', 'track')
    wd.mkdir(exist_ok=True)
    run.log(f'{len(pts)} points in {box}; seed {np.round(X[pts[seed]], 3).tolist()}; spanning tree of depth '
            f'{depth.max()}; ORCA {exe}, {jobs} parallel runs; level {level}')
    atoms = {k: ch2oo_xyz(**W._geometry_kwargs(ref, X[pts[k]])) for k in range(len(pts))}
    result, gbw, score, choice = {}, {}, np.full(len(pts), np.nan), [''] * len(pts)
    need = {'casscf': 'casscf', 'nevpt2': 'sc', 'qd_nevpt2': 'qd'}[level]

    def exc(r):
        e = np.sort(np.asarray(r['casscf'], float))
        return (e - e[0]) * HARTREE_EV

    def ok(r):
        return r is not None and len(r['casscf']) >= nr and len(r[need]) >= nr

    def run_batch(tasks):
        with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
            list(ex.map(lambda nm: AU._run_one(exe, wd, nm, float(cc['timeout_s'])), [t[1] for t in tasks]))

    # the seed: its own orbitals from the library if there are any (the solution the data already has), else scratch
    t0 = time.time()
    seed_moread = None
    if len(lib_g):
        d = np.sum((lib_x / sc - Z[seed]) ** 2, 1)
        if d.min() < 1e-10:
            seed_moread = lib_g[int(np.argmin(d))]
    nm = f't{seed:04d}_0'
    if not (wd / f'{nm}.inp').exists():
        (wd / f'{nm}.inp').write_text(AU.orca_input(atoms[seed], spec, moread=seed_moread, qd=True,
                                                    maxcore=cc['maxcore']))
    run_batch([(seed, nm)])
    r = AU.parse_orca(wd / f'{nm}.out', nr)
    if not ok(r):
        raise SystemExit(f'the seed point failed (see {wd / nm}.out)')
    result[seed], gbw[seed], score[seed], choice[seed] = r, str((wd / f'{nm}.gbw').resolve()), 0.0, 'seed'
    for lev in range(1, depth.max() + 1):
        layer = [k for k in range(len(pts)) if depth[k] == lev]
        tasks, plan = [], {}
        tracked = np.array(sorted(gbw))
        for k in layer:
            p = parent[k]
            if p not in gbw:                                  # parent failed: the nearest tracked point instead
                p = int(tracked[np.argmin(np.sum((Z[tracked] - Z[k]) ** 2, 1))])
            starts = [p]
            others = [t for t in tracked if t != p]
            if int(tc['neighbours']) > 1 and others:
                o = np.array(others)
                starts += [int(x) for x in o[np.argsort(np.sum((Z[o] - Z[k]) ** 2, 1))[:int(tc['neighbours']) - 1]]]
            plan[k] = (p, [])
            for a, s in enumerate(starts):
                nm = f't{k:04d}_{a}'
                if not (wd / f'{nm}.inp').exists():
                    (wd / f'{nm}.inp').write_text(AU.orca_input(atoms[k], spec, moread=gbw[s], qd=True,
                                                                maxcore=cc['maxcore']))
                tasks.append((k, nm))
                plan[k][1].append((s, nm))
        run_batch(tasks)
        for k in layer:
            p, att = plan[k]
            ref_exc = exc(result[p])
            cands = []
            for s, nm in att:
                r = AU.parse_orca(wd / f'{nm}.out', nr)
                if ok(r):
                    # the top root of the average moves steeply (~15 eV/A for CH2OO S3): compare the lower roots
                    lo_ = slice(1, max(2, nr - 1))
                    cands.append((float(np.max(np.abs(exc(r)[lo_] - ref_exc[lo_]))), s, nm, r))
            if not cands:
                choice[k] = 'failed'
                continue
            cands.sort(key=lambda c: c[0])
            score[k], s, nm, r = cands[0]
            result[k], gbw[k] = r, str((wd / f'{nm}.gbw').resolve())
            choice[k] = f'from {np.round(X[pts[s]], 3).tolist()}'
        done = len(gbw)
        run.log(f'  level {lev}/{depth.max()}: {len(layer)} points ({done}/{len(pts)} tracked, '
                f'{(time.time() - t0) / 60:.1f} min)')
    # ------------------------------------------------------------------ new data set
    jump = float(tc['jump_eV'])
    good = np.array([k in result for k in range(len(pts))])
    flagged = good & (score > jump)
    Eh = {key: np.full((len(pts), nr), np.nan) for key in ('casscf', 'sc', 'qd')}
    for k, r in result.items():
        Eh['casscf'][k] = np.sort(r['casscf'][:nr])
        Eh['sc'][k] = r['sc'][:nr] if len(r['sc']) >= nr else np.nan
        Eh['qd'][k] = r['qd'][:nr] if len(r['qd']) >= nr else np.nan
    ex = {k: np.array(v, copy=True) for k, v in ref.extra.items()}
    E = np.array(ref.energies, float)
    old_lvl = np.sort(E[pts], 1)
    use = good & ~(flagged if tc.get('drop_flagged', False) else np.zeros_like(flagged))
    rows = pts[use]
    ev_cas = (Eh['casscf'] - zero['casscf']) * HARTREE_EV
    ex['E_casscf_eV'][rows] = ev_cas[use]
    if 'nevpt2' in zero and 'E_nevpt2_eV' in ex:
        ex['E_nevpt2_eV'][rows] = ((Eh['sc'] - zero['nevpt2']) * HARTREE_EV)[use]
    if 'qd_nevpt2' in zero and 'E_qd_nevpt2_eV' in ex:
        ex['E_qd_nevpt2_eV'][rows] = ((Eh['qd'] - zero['qd_nevpt2']) * HARTREE_EV)[use]
    lv = {'casscf': ex['E_casscf_eV'], 'nevpt2': ex.get('E_nevpt2_eV'), 'qd_nevpt2': ex.get('E_qd_nevpt2_eV')}[level]
    E[rows] = lv[rows][:, :E.shape[1]]
    new_lvl = np.sort(E[pts], 1)
    changed = np.zeros(len(ref), bool)
    changed[pts] = use & (np.nanmax(np.abs(new_lvl - old_lvl), 1) > 1e-4)
    for k in W.PER_STATE:
        if k in ex and np.ndim(ex[k]) == 2:
            v = np.array(ex[k], float); v[changed] = np.nan; ex[k] = v
    if 'casci_mismatch_hartree' in ex:
        ex['casci_mismatch_hartree'][rows] = 0.0
    for k in ('converged', 'active_space_consistent'):
        if k in ex:
            ex[k][rows] = True
    if 'from_orca' in ex:
        ex['from_orca'][rows] = True
    if 'audit' in ex:
        ex['audit'][rows] = CODE_TRACKED
    if tc.get('drop_flagged', False) and flagged.any():
        E[pts[flagged]] = np.nan                                   # could not be followed: leave them out
    meta = dict(ref.meta, method=f'{ref.meta.get("method", ref.name)} [tracked: {int(use.sum())} points]',
                track={'box': box, 'n': int(len(pts)), 'n_tracked': int(use.sum()), 'n_flagged': int(flagged.sum()),
                       'seed': X[pts[seed]].tolist()})
    out = run.p('data', 'tracked')
    out.mkdir(exist_ok=True)
    RefSet(ref.name, X.copy(), E, list(ref.coord_names), extra=ex, meta=meta).save(out / 'refs.npz')
    # orbital library: tracked points use their tracked orbitals; couplings at changed points are dropped
    gx = {tuple(np.round(X[pts[k]], 6)): gbw[k] for k in gbw}
    keep = [(c, g) for c, g in zip(lib_x, lib_g) if tuple(np.round(c, 6)) not in gx]
    allc = [c for c, _ in keep] + [X[pts[k]] for k in gbw]
    save_library(out / 'orca_orbitals.json', ref.coord_names, np.array(allc).reshape(-1, X.shape[1]),
                 [g for _, g in keep] + [gbw[k] for k in gbw])
    from .repair import _filter_nacs, _filter_orbitals
    _filter_nacs(run.p('data', 'nacs.npz'), out / 'nacs.npz', changed)
    _filter_orbitals(run.p('data', 'orbitals.npz'), out / 'orbitals.npz', X[changed])
    # ------------------------------------------------------------------ smoothness and report
    Xp = X[pts]
    r_old, n2 = line_roughness(Xp, old_lvl[:, 0] * 1000)
    r_new, _ = line_roughness(Xp[use], new_lvl[use, 0] * 1000)
    s3_old, _ = line_roughness(Xp, np.sort(ref.extra['E_casscf_eV'][pts], 1)[:, nr - 1] * 1000)
    s3_new, _ = line_roughness(Xp[use], ev_cas[use, nr - 1] * 1000)
    dS0 = (new_lvl[:, 0] - old_lvl[:, 0]) * 1000
    info = {'n': int(len(pts)), 'n_tracked': int(use.sum()), 'n_failed': int((~good).sum()),
            'n_flagged': int(flagged.sum()), 'jump_eV': jump, 'grid_third_differences': n2,
            'S0_roughness_meV': {'before': r_old, 'after': r_new},
            f'S{nr - 1}_casscf_roughness_meV': {'before': s3_old, 'after': s3_new},
            'S0_change_meV': {'median_abs': float(np.nanmedian(np.abs(dS0[use]))), 'max_abs': float(np.nanmax(np.abs(dS0[use])))},
            'n_changed': int(changed.sum()), 'minutes': (time.time() - t0) / 60}
    run.p('results', 'track.json').write_text(json.dumps(info, indent=1))
    with open(run.p('results', 'track.csv'), 'w') as fh:
        fh.write(','.join(['index'] + ref.coord_names + ['depth', 'kept', 'excitation_jump_eV', 'S0_change_meV']) + '\n')
        for k in range(len(pts)):
            fh.write(','.join([str(pts[k])] + [f'{v:.4f}' for v in X[pts[k]]] +
                              [str(depth[k]), f'"{choice[k]}"', f'{score[k]:.3f}', f'{dS0[k]:.2f}']) + '\n')
    run.log(f'{int(use.sum())}/{len(pts)} points tracked ({int(flagged.sum())} jumps > {jump} eV flagged and left out, '
            f'{int((~good).sum())} failed) in {info["minutes"]:.0f} min; S0 changed at {int(changed.sum())} points '
            f'(median |change| {info["S0_change_meV"]["median_abs"]:.1f} meV, max {info["S0_change_meV"]["max_abs"]:.0f})')
    run.log(f'smoothness on grid lines (RMS of {n2} third differences): S0 {r_old:.1f} -> {r_new:.1f} meV RMS; '
            f'S{nr - 1} (CASSCF) {s3_old:.0f} -> {s3_new:.0f} meV RMS')
    _figure(run, ref, pts, old_lvl, new_lvl, ev_cas, use, nr)
    rel = f'../{run.path.name}/data/tracked/refs.npz'
    run.log(f'wrote {out / "refs.npz"}; next: init a run with --refs {rel}, then refs, merge, production')


def _figure(run, ref, pts, old, new, ev_cas, use, nr):
    plt = W._mpl()
    X = ref.coords[pts]
    names = ref.coord_names
    j = names.index('r_CO') if 'r_CO' in names else len(names) - 1
    others = [c for c in range(len(names)) if c != j]
    groups = {}
    for k in range(len(pts)):
        groups.setdefault(tuple(np.round(X[k, others], 4)), []).append(k)
    lines = sorted([g for g in groups.values() if len(g) >= 6], key=lambda g: -len(g))[:4]
    fig, axes = plt.subplots(2, max(1, len(lines)), figsize=(4.4 * max(1, len(lines)), 7.5), squeeze=False)
    for c, g in enumerate(lines):
        g = sorted(g, key=lambda k: X[k, j])
        x = X[g, j]
        ax = axes[0, c]
        ax.plot(x, (old[g, 0] - np.nanmin(old[g, 0])) * 1000, 'o--', color=W.SCOL[2], label='lowest SA average')
        gu = [k for k in g if use[k]]
        ax.plot(X[gu, j], (new[gu, 0] - np.nanmin(new[gu, 0])) * 1000, 'o-', color=W.SCOL[1], label='tracked')
        lab = ', '.join(f'{names[o]} {X[g[0], o]:g}' for o in others)
        ax.set(title=lab, xlabel=names[j], ylabel='S0 (meV above the lowest on the line)')
        ax.legend(fontsize=8)
        ax = axes[1, c]
        ax.plot(x, np.sort(ref.extra['E_casscf_eV'][pts[g]], 1)[:, nr - 1], 'o--', color=W.SCOL[2], label='before')
        ax.plot(X[gu, j], ev_cas[gu, nr - 1], 'o-', color=W.SCOL[1], label='tracked')
        ax.set(xlabel=names[j], ylabel=f'S{nr - 1}, SA-CASSCF (eV)')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run.p('figures', 'TR1_track.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)
    run.log('figures/TR1_track.png')
