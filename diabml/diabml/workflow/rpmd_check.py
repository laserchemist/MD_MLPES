"""Is the 9-D surface right where the ring polymer goes? CCSD(T) at bead geometries outside the fitted range.

    python -m diabml.workflow.ch2oo rpmd_check RUN [--jobs N] [--orca PATH] [-v]
        -> results/rpmd_check.json, data/rpmd_check.npz, figures/RPMD2_check.png (and a section of dynamics_9d.html)

Path-integral beads spread over the zero-point width of every atom (about 0.16 A for H), so part of the bead sampling
lies outside the range of the points the surface was fitted to, where the polynomial is extrapolated. rpmd9d keeps a
random sample of the bead geometries it visits, inside and outside that range. This stage takes [rpmd] check_points
of the outside ones, spread by farthest-point sampling, plus check_controls inside ones, computes CCSD(T) at them in
the surface's basis (outputs in data/rpmd_check/, finished ones reused), adds the same 9-D basis-set correction the
surface was fitted with, and compares with the surface:

    error = [E_surface - E_CCSD(T)] - median over the inside controls of the same difference,

so a constant offset between the two energy zeros drops out. Reported: RMS and largest error outside and inside, by the
coordinate that is out of range and by energy above the minimum, against the surface's own held-out error. If the
outside error is much larger than that, the points in data/rpmd_check.npz are where the surface needs data.
"""
from __future__ import annotations

import json

import numpy as np

from . import ch2oo as W


def _fps(X, n, seed=0):
    """Farthest-point selection of n rows of X (already scaled)."""
    if len(X) <= n:
        return np.arange(len(X))
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(X)))]
    d = np.linalg.norm(X - X[idx[0]], axis=1)
    for _ in range(n - 1):
        k = int(np.argmax(d))
        idx.append(k)
        d = np.minimum(d, np.linalg.norm(X - X[k], axis=1))
    return np.array(idx)


def stage_rpmd_check(run, jobs=None, orca=None):
    from ..geometry import NINE
    from ..wellfit import PolyWell
    from . import audit as AU
    from .well9d import HARTREE_EV, _basis, _compute, delta9, refine_minimum
    rc, wc, cc = run.cfg['rpmd'], run.cfg['well9d'], run.cfg['compute']
    P = int(rc.get('check_beads') or max(int(b) for b in rc['beads']))
    f = run.p('results', f'rpmd_9d_P{P}.npz')
    if not f.exists():
        raise SystemExit(f'{f} not found: run rpmd9d (with P = {P}) first')
    z = np.load(f)
    if 'out_x' not in z.files or len(z['out_x']) == 0:
        raise SystemExit(f'{f} has no bead-geometry sample (written by an older rpmd9d, or no bead left the fitted range): '
                         f'rerun rpmd9d')
    model = PolyWell.load(run.p('models', str(run.cfg['ir'].get('model9d', 'well9d.npz'))))
    lo = np.array([model.box[k][0] for k in NINE])
    hi = np.array([model.box[k][1] for k in NINE])
    sc = np.asarray(model.scale, float)
    n_out, n_in = int(rc.get('check_points', 40)), int(rc.get('check_controls', 10))
    Xo = z['out_x'][_fps(z['out_x'] / sc, n_out)]
    Xi = z['in_x'][_fps(z['in_x'] / sc, n_in, seed=1)] if len(z['in_x']) else np.zeros((0, 9))
    X = np.concatenate([Xo, Xi])
    kind = np.r_[np.ones(len(Xo), bool), np.zeros(len(Xi), bool)]
    basis = _basis(run, wc)
    jobs = int(jobs or rc.get('check_jobs') or wc.get('jobs') or run.cfg['gs_grid'].get('jobs') or cc['jobs'])
    exe = AU._orca_exe(run, orca or cc.get('orca') or run.cfg['audit'].get('orca') or None)
    run.log(f'checking the surface at {len(Xo)} bead geometries outside the fitted range and {len(Xi)} inside (P = {P} '
            f'sample of {len(z["out_x"])} outside, {len(z["in_x"])} inside): RHF-CCSD(T)/{basis}, {jobs} parallel')
    Eh, T1 = _compute(run, exe, run.p('data', 'rpmd_check'), X, basis, cc, jobs, 'rpmd_check')
    ok = np.isfinite(Eh)
    ref = Eh * HARTREE_EV
    dlt, info = delta9(run, wc, X)
    if dlt is not None:
        ref = ref + dlt
    xm, Em, _ = refine_minimum(model, model.center.numpy() if hasattr(model.center, 'numpy') else model.center)
    Es = model.predict(X)[:, 0]
    diff = Es - ref
    ctrl = ok & ~kind
    if ctrl.sum() < 3:
        raise SystemExit('fewer than 3 inside controls finished: cannot fix the energy offset')
    off = float(np.median(diff[ctrl]))
    err = (diff - off) * 1000                                   # meV
    above = (Es - Em) * 1000                                    # meV above the minimum (surface)
    which = [NINE[int(np.argmax(np.maximum(lo - x, x - hi) / sc))] for x in X]
    fit_rms = (json.loads(run.p('results', 'well9d.json').read_text()).get('orders', {}).get(
        str(model.meta.get('order')), {}).get('heldout_rms_meV') if run.p('results', 'well9d.json').exists() else None)

    def stats(m):
        e = err[m & ok]
        return {'n': int(len(e)), 'rms_meV': float(np.sqrt(np.mean(e ** 2))) if len(e) else None,
                'max_meV': float(np.abs(e).max()) if len(e) else None}
    out = {'beads': P, 'basis': basis, 'delta': bool(dlt is not None), 'offset_meV': off * 1000,
           'outside': stats(kind), 'inside': stats(~kind), 'fit_heldout_rms_meV': fit_rms, 'by_coordinate': {},
           'by_energy': [], 'T1_max': float(np.nanmax(T1)) if np.isfinite(T1).any() else None}
    for c in sorted(set(np.array(which)[kind])):
        out['by_coordinate'][c] = stats(kind & (np.array(which) == c))
    win = float(model.meta.get('window_eV') or wc.get('fit_window_eV', 0.8)) * 1000     # meV
    edges = sorted({0.0, 200.0, 400.0, 600.0, 1000.0, win, 2000.0} - {e for e in (2000.0,) if e <= win})
    for a, b in zip(edges, edges[1:] + [1e9]):
        s = stats(kind & (above >= a) & (above < b))
        if s['n']:
            out['by_energy'].append({'range_meV': [int(a), int(b) if b < 1e9 else None], **s})
    o, i = out['outside'], out['inside']
    share = None
    rj = run.p('results', 'rpmd_9d.json')
    if rj.exists():
        share = json.loads(rj.read_text()).get('beads', {}).get(str(P), {}).get('outside_box')
    ow, ob = stats(kind & (above < win)), stats(kind & (above >= win))
    out.update({'window_meV': win, 'outside_within_window': ow, 'outside_beyond_window': ob, 'share_outside': share})
    if share is not None and o['rms_meV'] is not None and i['rms_meV'] is not None:
        # the sample is spread to the extremes, so this over-weights the far tail: an upper bound for the error the
        # beads actually see
        out['bead_weighted_rms_meV'] = float(np.sqrt((1 - share) * i['rms_meV'] ** 2 + share * o['rms_meV'] ** 2))
    run.log(f'surface - CCSD(T){" (with the 9-D basis-set correction)" if dlt is not None else ""}, offset removed '
            f'with the inside controls: outside the fitted range RMS {o["rms_meV"]:.2f} meV, max {o["max_meV"]:.2f} meV '
            f'({o["n"]} points); inside RMS {i["rms_meV"]:.2f} meV ({i["n"]}); the fit\'s own held-out RMS '
            + (f'{fit_rms:.2f} meV' if fit_rms else '?'))
    run.log('  by the coordinate furthest out of range: ' + ', '.join(
        f'{c} {s["rms_meV"]:.2f} meV RMS ({s["n"]})' for c, s in out['by_coordinate'].items()))
    run.log('  by energy above the minimum: ' + ', '.join(
        f'{r["range_meV"][0]}-{r["range_meV"][1] or "∞"} meV: {r["rms_meV"]:.2f} RMS ({r["n"]})' for r in out['by_energy']))
    ref_rms = max(fit_rms or 0.6, 0.5)
    sh = f'{100 * share:.1f}% of bead geometries' if share is not None else 'the bead geometries'
    bw = out.get('bead_weighted_rms_meV')
    bad_in = ow['rms_meV'] is not None and ow['rms_meV'] > 3 * ref_rms
    bad_avg = bw is not None and bw > 2 * ref_rms
    if o['rms_meV'] is None:
        verdict = 'no outside geometry finished: nothing to judge'
    elif (bad_in and (share is None or share > 0.01)) or bad_avg:
        verdict = (f'{sh} leave the fitted range and the surface is {o["rms_meV"] / ref_rms:.0f}x less accurate there '
                   f'(within the {win / 1000:g} eV fit window: {ow["rms_meV"] or float("nan"):.1f} meV RMS'
                   + (f'; averaged over all beads about {bw:.1f} meV' if bw is not None else '') +
                   '): add these points to the fit (well9d_extend, or data/rpmd_check.npz)')
    else:
        parts = [f'only {sh} leave the fitted range']
        if ow['n']:
            parts.append(f'within the {win / 1000:g} eV fit window the error there is {ow["rms_meV"]:.1f} meV RMS ({ow["n"]})')
        if ob['n']:
            parts.append(f'beyond it {ob["rms_meV"]:.1f} meV RMS ({ob["n"]}, max {ob["max_meV"]:.0f} meV), a rare tail')
        if bw is not None:
            parts.append(f'averaged over all beads at most about {bw:.1f} meV')
        verdict = '; '.join(parts) + ': no new data needed for the TRPMD runs'
    out['verdict'] = verdict
    if ow['n'] or ob['n']:
        run.log(f'  outside the fitted range, within the {win / 1000:g} eV fit window: '
                + (f'{ow["rms_meV"]:.2f} meV RMS ({ow["n"]})' if ow['n'] else 'none')
                + '; beyond it: ' + (f'{ob["rms_meV"]:.2f} meV RMS ({ob["n"]})' if ob['n'] else 'none')
                + (f'; share of bead geometries outside {100 * share:.2f}%' if share is not None else ''))
    run.log('  ' + verdict)
    np.savez(run.p('data', 'rpmd_check.npz'), coords=X, outside=kind, E_hartree=Eh, T1=T1, surface_eV=Es,
             error_meV=err, above_min_meV=above, basis=basis)
    run.p('results', 'rpmd_check.json').write_text(json.dumps(out, indent=1, default=float))
    _figure(run, err, above, kind, ok, which, ref_rms)
    run.log('wrote results/rpmd_check.json, data/rpmd_check.npz, figures/RPMD2_check.png')
    from .dynamics_report import stage_dynamics_report
    stage_dynamics_report(run)


def _figure(run, err, above, kind, ok, which, ref_rms):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.6))
    cs = sorted(set(np.array(which)[kind & ok]))
    cmap = plt.get_cmap('tab10')
    for k, c in enumerate(cs):
        m = kind & ok & (np.array(which) == c)
        ax.scatter(above[m], err[m], s=26, color=cmap(k % 10), label=f'outside: {c}')
    m = ~kind & ok
    ax.scatter(above[m], err[m], s=30, facecolor='none', edgecolor='0.3', label='inside (controls)')
    ax.axhspan(-ref_rms, ref_rms, color='0.9', zorder=0, label=f'± fit held-out RMS ({ref_rms:.2f} meV)')
    ax.axhline(0, color='0.5', lw=0.6)
    ax.set_xlabel('surface energy above the minimum (meV)')
    ax.set_ylabel('surface − CCSD(T) (meV)')
    ax.set_title('Surface error at bead geometries visited by TRPMD', fontsize=11)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(run.p('figures', 'RPMD2_check.png'), dpi=150)
    plt.close(fig)
