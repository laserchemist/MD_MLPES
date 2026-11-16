"""Extend the 9-D surface to higher energy, where the ring-polymer beads and the C-H overtones go.

    python -m diabml.workflow.ch2oo well9d_extend RUN [--jobs N] [--orca PATH] [-v]
        -> data/well9d.npz (new batches), data/well9d_tz.npz (more points), data/well9d_extend_design.npz,
           results/well9d_extend.json, figures/W9X_extend.png, and (extend_refit) a refitted models/well9d.npz

The surface was fitted to CCSD(T) points within [well9d] fit_window_eV (0.8 eV) of the minimum. rpmd_check showed
that its error grows with energy beyond that (a few meV at 1 eV, tens of meV above 1.5 eV), and that the beads of a
300 K ring polymer go there, mostly along the C-H stretches. This stage adds data there:

  1. candidates: the bead geometries every rpmd9d run saved (results/rpmd_9d_P*.npz; those outside the fitted range
     or at least extend_e_min_eV up), and a harmonic sample of the well (normal-mode directions, energies up to
     extend_e_max_eV, weighted towards the top by extend_e_power < 1);
  2. extend_points of them by farthest-point sampling in the energy-scaled normal coordinates, starting from all the
     points already computed, so the new ones fill what is missing: extend_bead_fraction from the beads, the rest
     from the harmonic sample. The rpmd_check points (already computed) are added as they are;
  3. CCSD(T) in the surface's basis (data/well9d_extend/, finished outputs reused), added to data/well9d.npz as
     extend_batches new batches, so the held-out error by batch still applies;
  4. extend_tz_points of them in the larger basis (data/well9d_tz/), added to data/well9d_tz.npz so the 9-D
     basis-set correction covers the same range;
  5. the old surface is tested at the new points (a true out-of-sample test, by energy), then, with extend_refit,
     saved as models/well9d_pre_extend.npz (once) and refitted with the window raised to extend_fit_window_eV
     (and [well9d] fit_weight_eV, if set, weighting the low-energy points more).

The selection is saved the first time and reused, so a rerun only computes what is unfinished. To check the refit:
the held-out error within fit_low_eV (the region the fundamentals depend on) should stay near its old value; then
rerun well9d_vpt2, well9d_dipole is unchanged, well9d_vci, well9d_bands, well9d_isotopes, rpmd9d and rpmd_check.
"""
from __future__ import annotations

import json
import shutil

import numpy as np

from . import ch2oo as W

BANDS_EV = (0.0, 0.4, 0.8, 1.0, 1.5, 2.0, 1e9)


def _min_dist(Z, Y, chunk=4000):
    """Distance from each row of Z to the nearest row of Y."""
    if len(Y) == 0:
        return np.full(len(Z), np.inf)
    out = np.empty(len(Z))
    for i in range(0, len(Z), chunk):
        z = Z[i:i + chunk]
        d = np.full(len(z), np.inf)
        for j in range(0, len(Y), chunk):
            y = Y[j:j + chunk]
            d2 = np.sum(z ** 2, 1)[:, None] - 2 * z @ y.T + np.sum(y ** 2, 1)[None]
            d = np.minimum(d, np.sqrt(np.maximum(d2.min(1), 0.0)))
        out[i:i + chunk] = d
    return out


def fps_from(Z, n, Y):
    """n rows of Z by farthest-point sampling, each as far as possible from the rows of Y and those already taken."""
    d = _min_dist(Z, Y)
    if not np.isfinite(d).any():
        d = np.sum((Z - Z.mean(0)) ** 2, 1)
    sel = []
    for _ in range(min(n, len(Z))):
        k = int(np.argmax(d))
        if d[k] <= 1e-9:                                         # only duplicates left
            break
        sel.append(k)
        d = np.minimum(d, np.linalg.norm(Z - Z[k], axis=1))
        d[k] = -1.0
    return np.array(sel, int)


def _bead_candidates(run, model, lo, hi, e_min, e_max, n_keep, rng):
    """Bead geometries from every saved rpmd9d run: outside the fitted range, or at least e_min up (surface energy)."""
    from .well9d import _physical
    files = sorted(run.p('results').glob('rpmd_9d_P*.npz'))
    X = [np.load(f)[k] for f in files for k in ('out_x', 'in_x') if k in np.load(f).files]
    X = np.concatenate([x for x in X if len(x)]) if any(len(x) for x in X) else np.zeros((0, 9))
    if not len(X):
        return X, np.zeros(0), [f.name for f in files]
    X = X[_physical(X)]
    Es = model.predict(X)[:, 0]
    out = np.any((X < lo) | (X > hi), axis=1)
    keep = (out | (Es >= e_min)) & (Es <= e_max)
    X, Es = X[keep], Es[keep]
    if len(X) > n_keep:
        k = rng.choice(len(X), n_keep, replace=False)
        X, Es = X[k], Es[k]
    return X, Es, [f.name for f in files]


def _design(run, wc, rc, hz, model, Xold):
    from .well9d import _metric, sample_pool
    dp = run.p('data', 'well9d_extend_design.npz')
    keys = ('extend_points', 'extend_bead_fraction', 'extend_e_min_eV', 'extend_e_max_eV', 'extend_e_power',
            'extend_batches', 'extend_pool', 'seed')
    sig = json.dumps({k: wc.get(k) for k in keys}, sort_keys=True, default=str)
    if dp.exists():
        z = np.load(dp, allow_pickle=False)
        if str(z['signature']) != sig:
            run.log('note: data/well9d_extend_design.npz was made with other [well9d] extend_* settings; it is kept so '
                    'the new batches stay the same (move it away to choose again)')
        return {k: z[k] for k in z.files}, True
    from ..geometry import NINE
    lo = np.array([model.box[k][0] for k in NINE])
    hi = np.array([model.box[k][1] for k in NINE])
    n = int(wc.get('extend_points', 400))
    e_min, e_max = float(wc.get('extend_e_min_eV', 0.6)), float(wc.get('extend_e_max_eV', 2.0))
    n_pool = int(wc.get('extend_pool', 20000))
    rng = np.random.default_rng(int(wc.get('seed', 0)) + 17)
    Xb, Eb, files = _bead_candidates(run, model, lo, hi, e_min, e_max, n_pool, rng)
    Xh, Eh, _ = sample_pool(hz['center'], hz['lam'], hz['L'], n_pool, e_max, float(wc.get('extend_e_power', 0.5)),
                            float(wc.get('d_max', 45.0)), int(wc.get('seed', 0)) + 23)
    mh = Eh >= e_min
    Xh = Xh[mh]
    # rpmd_check points: computed already, in the same basis
    cp = run.p('data', 'rpmd_check.npz')
    Xc, Ec, Tc = np.zeros((0, 9)), np.zeros(0), np.zeros(0)
    if cp.exists():
        c = np.load(cp, allow_pickle=False)
        ok = np.isfinite(c['E_hartree'])
        Xc, Ec, Tc = c['coords'][ok], c['E_hartree'][ok], c['T1'][ok]
    from .well9d import _key
    have = {_key(x) for x in np.concatenate([Xold, Xc])}
    Xb = Xb[np.array([_key(x) not in have for x in Xb], bool)] if len(Xb) else Xb
    taken = np.concatenate([_metric(hz, Xold), _metric(hz, Xc)]) if len(Xc) else _metric(hz, Xold)
    nb = int(round(float(wc.get('extend_bead_fraction', 0.5)) * n)) if len(Xb) else 0
    nb = min(nb, len(Xb))
    sb = fps_from(_metric(hz, Xb), nb, taken) if nb else np.zeros(0, int)
    taken = np.concatenate([taken, _metric(hz, Xb[sb])]) if nb else taken
    sh = fps_from(_metric(hz, Xh), n - nb, taken)
    X = np.concatenate([Xb[sb], Xh[sh]])
    src = np.array(['beads'] * len(sb) + ['harmonic'] * len(sh))
    order = fps_from(_metric(hz, X), len(X), _metric(hz, Xold))      # round-robin batches over a spread order
    X, src = X[order], src[order]
    k = max(1, int(wc.get('extend_batches', 2)))
    b0 = int(np.load(run.p('data', 'well9d.npz'))['batch'].max()) + 1
    batch = b0 + np.arange(len(X)) % k
    out = dict(coords=X, source=src, batch=batch, check_coords=Xc, check_E_hartree=Ec, check_T1=Tc,
               check_batch=b0 + np.arange(len(Xc)) % k, signature=sig, bead_files=np.array(files or ['']),
               n_bead_candidates=len(Xb), n_harmonic_candidates=len(Xh))
    np.savez(dp, **out)
    run.log(f'design: {len(Xb)} bead geometries from {", ".join(files) or "no rpmd9d run"} (outside the fitted range '
            f'or >= {e_min:g} eV, <= {e_max:g} eV) and {len(Xh)} harmonic candidates ({e_min:g}-{e_max:g} eV) -> '
            f'{len(sb)} + {len(sh)} new points, spread from the {len(Xold)} computed ones, in batches {b0}-{b0 + k - 1}; '
            f'{len(Xc)} rpmd_check points added as they are; wrote data/well9d_extend_design.npz')
    return out, False


def _bands(E, err):
    rows = []
    for a, b in zip(BANDS_EV[:-1], BANDS_EV[1:]):
        m = (E >= a) & (E < b) & np.isfinite(err)
        if m.any():
            rows.append({'range_eV': [a, b if b < 1e9 else None], 'n': int(m.sum()),
                         'rms_meV': float(np.sqrt(np.mean(err[m] ** 2))), 'max_meV': float(np.abs(err[m]).max())})
    return rows


def stage_well9d_extend(run, jobs=None, orca=None):
    from ..geometry import NINE
    from ..wellfit import PolyWell
    from . import audit as AU
    from .well9d import (HARTREE_EV, _basis, _compute, _key, _merge, _metric, delta9, harmonic_model, save_tz,
                         stage_well9d_fit)
    wc, rc, cc, gc = run.cfg['well9d'], run.cfg.get('rpmd', {}), run.cfg['compute'], run.cfg['gs_grid']
    p = run.p('data', 'well9d.npz')
    mp = run.p('models', str(run.cfg['ir'].get('model9d', 'well9d.npz')))
    if not p.exists() or not mp.exists():
        raise SystemExit('data/well9d.npz or models/well9d.npz not found: run well9d and well9d_fit first')
    hz = np.load(run.p('data', 'well9d_hessian.npz'), allow_pickle=False)
    pre = run.p('models', 'well9d_pre_extend.npz')
    model = PolyWell.load(pre if pre.exists() else mp)             # the surface before any extension
    with np.load(p, allow_pickle=False) as f:
        z = {k: f[k] for k in f.files}
    d, reused = _design(run, wc, rc, hz, model, z['coords'])
    ext = set(np.unique(np.r_[d['batch'], d['check_batch']]).astype(int).tolist())
    old = ~np.isin(z['batch'], list(ext))
    Xold, Ehold = z['coords'][old], z['E_total_hartree'][old, 0]
    basis = _basis(run, wc)
    jobs = int(jobs or wc.get('jobs') or gc.get('jobs') or cc['jobs'])
    exe = AU._orca_exe(run, orca or cc.get('orca') or run.cfg['audit'].get('orca') or None)
    X = d['coords']
    run.log(f'{"reusing the saved selection: " if reused else ""}{len(X)} new points, RHF-CCSD(T)/{basis}, {jobs} '
            f'parallel (data/well9d_extend/)')
    for j, nm in enumerate(NINE):
        run.detail(f'{nm:7s} new {X[:, j].min():9.3f} - {X[:, j].max():9.3f}   fitted box {model.box[nm][0]:9.3f} - '
                   f'{model.box[nm][1]:9.3f}')
    E, T1 = _compute(run, exe, run.p('data', 'well9d_extend'), X, basis, cc, jobs, 'well9d_extend')
    Xn = np.concatenate([X, d['check_coords']])
    En = np.concatenate([E, d['check_E_hartree']])
    Tn = np.concatenate([T1, d['check_T1']])
    bn = np.concatenate([d['batch'], d['check_batch']]).astype(int)
    srcn = np.concatenate([d['source'], np.array(['rpmd_check'] * len(d['check_coords']))])
    ok = np.isfinite(En)
    # the old surface at the new points, on the energy zero it was fitted with (old data + the old 9-D correction)
    dall, _ = delta9(run, wc, np.concatenate([Xold, Xn]))
    Eall = np.concatenate([Ehold, En]) * HARTREE_EV + (dall if dall is not None else 0.0)
    zero = float(np.nanmin(Eall[:len(Xold)]))
    Eref = Eall[len(Xold):] - zero
    err = (model.predict(Xn)[:, 0] - Eref) * 1000
    test = _bands(Eref, err)
    run.log(f'the surface before extension at the {int(ok.sum())} new points (out of sample), by CCSD(T) energy above '
            f'the minimum: ' + ', '.join(f'{r["range_eV"][0]:g}-{r["range_eV"][1] or "∞"} eV {r["rms_meV"]:.1f} meV '
                                         f'RMS ({r["n"]})' for r in test))
    f = harmonic_model(hz['center'], hz['g'], hz['K'])
    rows = {_key(x): (x, e, t, int(b), float(f(x[None])[0])) for x, e, t, b in zip(Xn[ok], En[ok], Tn[ok], bn[ok])}
    R = _merge(run, rows)
    run.log(f'data/well9d.npz: {len(R)} points (was {len(Xold)}), new batches {sorted(ext)}; T1 of the new points '
            f'{np.nanmin(Tn):.3f}-{np.nanmax(Tn):.3f}' + (' (above 0.02: check the multireference character)'
                                                          if np.nanmax(Tn) > 0.02 else ''))
    # the larger basis at a spread subset of the new points
    ntz = int(wc.get('extend_tz_points', 50))
    tz_info = None
    if ntz > 0 and ok.any():
        tbasis = str(wc.get('tz_basis', 'aug-cc-pVTZ'))
        tp = run.p('data', 'well9d_tz.npz')
        Xt0 = np.load(tp)['coords'] if tp.exists() else np.zeros((0, 9))
        cand = np.where(ok & (Eref <= float(wc.get('extend_e_max_eV', 2.0))))[0]
        pick = cand[fps_from(_metric(hz, Xn[cand]), min(ntz, len(cand)), _metric(hz, Xt0))]
        tjobs = int(wc.get('tz_jobs') or jobs)
        run.log(f'{tbasis}: {len(pick)} of the new points (spread from the {len(Xt0)} already in data/well9d_tz.npz), '
                f'{tjobs} parallel')
        Etz, Ttz = _compute(run, exe, run.p('data', 'well9d_tz'), Xn[pick], tbasis, cc, tjobs, tbasis)
        g = np.isfinite(Etz)
        nt = save_tz(run, Xn[pick][g], Etz[g], En[pick][g], Ttz[g], tbasis, str(hz['basis']))
        dd = (Etz[g] - En[pick][g]) * HARTREE_EV * 1000
        tz_info = {'basis': tbasis, 'n_new': int(g.sum()), 'n_total': int(nt),
                   'range_meV': float(np.ptp(dd)) if len(dd) else None}
        run.log(f'data/well9d_tz.npz: {nt} points ({int(g.sum())} new; {tbasis} - {hz["basis"]} spans '
                f'{tz_info["range_meV"] or 0:.0f} meV over them)')
    out = {'basis': basis, 'n_new': int(ok.sum()), 'n_failed': int((~ok).sum()), 'n_before': int(len(Xold)),
           'batches': sorted(ext), 'sources': {s: int(np.sum(srcn[ok] == s)) for s in np.unique(srcn)},
           'energy_eV': {'min': float(np.nanmin(Eref)), 'median': float(np.nanmedian(Eref)),
                         'max': float(np.nanmax(Eref))},
           'old_surface_test': test, 'tz': tz_info, 'T1_max': float(np.nanmax(Tn)) if ok.any() else None,
           'bead_files': [str(s) for s in d['bead_files'] if str(s)]}
    # refit with the wider window
    before = json.loads(run.p('results', 'well9d.json').read_text()) if run.p('results', 'well9d.json').exists() else {}
    if wc.get('extend_refit', True):
        if not pre.exists():
            shutil.copy(mp, pre)
            if run.p('results', 'well9d.json').exists():
                shutil.copy(run.p('results', 'well9d.json'), run.p('results', 'well9d_pre_extend.json'))
            run.log('saved the surface before extension as models/well9d_pre_extend.npz (and '
                    'results/well9d_pre_extend.json)')
        else:
            pj = run.p('results', 'well9d_pre_extend.json')
            before = json.loads(pj.read_text()) if pj.exists() else before
        win = float(wc.get('extend_fit_window_eV', 1.5))
        run.log(f'refitting with the window raised from {float(wc["fit_window_eV"]):g} to {win:g} eV')
        stage_well9d_fit(run, window_eV=win)
        after = json.loads(run.p('results', 'well9d.json').read_text())
        out['refit'] = _compare(run, before, after, model, PolyWell.load(mp), Xn[ok], Eref[ok], win)
    run.p('results', 'well9d_extend.json').write_text(json.dumps(out, indent=1, default=float))
    _figure(run, Eref, err, srcn, ok, z, old, out.get('refit'))
    run.log('wrote results/well9d_extend.json, figures/W9X_extend.png')
    nxt = 'well9d_vpt2, well9d_vci, well9d_bands, well9d_isotopes, rpmd9d, rpmd_check'
    if wc.get('extend_refit', True):
        run.log(f'next: set [well9d] fit_window_eV = {float(wc.get("extend_fit_window_eV", 1.5)):g} in config.toml so a '
                f'later well9d_fit keeps the wider window, then rerun {nxt}')
    else:
        run.log(f'next: raise [well9d] fit_window_eV (e.g. 1.5), run well9d_fit, then {nxt}')


def _compare(run, before, after, m_old, m_new, Xn, En, win):
    """Held-out errors and harmonic frequencies before and after the refit."""
    o_b, o_a = before.get('order'), after.get('order')
    cb = before.get('orders', {}).get(str(o_b), {}) if before else {}
    ca = after.get('orders', {}).get(str(o_a), after.get('orders', {}).get(o_a, {}))
    hb = [h['cm-1'] for h in before.get('harmonic_cm-1', [])]
    ha = [h['cm-1'] for h in after.get('harmonic_cm-1', [])]
    labels = [h.get('label', '') for h in after.get('harmonic_cm-1', [])]
    res = (m_new.predict(Xn)[:, 0] - En) * 1000
    inwin = En <= win
    r = {'window_eV': [before.get('window_eV'), win], 'order': [o_b, o_a],
         'heldout_rms_meV': [cb.get('heldout_rms_meV'), ca.get('heldout_rms_meV')],
         'heldout_low_rms_meV': ca.get('heldout_low_rms_meV'), 'low_eV': ca.get('low_eV'),
         'new_points_residual_rms_meV': float(np.sqrt(np.mean(res[inwin] ** 2))) if inwin.any() else None,
         'harmonic_cm-1': [{'label': l, 'before': b, 'after': a} for l, b, a in zip(labels, hb, ha)]}
    low = f'{r["heldout_low_rms_meV"]:.2f} meV within {r["low_eV"]:g} eV' if r['heldout_low_rms_meV'] else ''
    run.log(f'refit: held-out RMS {cb.get("heldout_rms_meV") or float("nan"):.2f} meV (within '
            f'{before.get("window_eV") or "?"} eV, before) -> {ca.get("heldout_rms_meV") or float("nan"):.2f} meV within '
            f'{win:g} eV' + (f', {low}' if low else '') + ' (the old value is the one to compare with: the spectra '
            f'depend most on the low-energy part)')
    if hb and ha:
        run.log('  harmonic frequencies (before -> after, cm-1): ' + ', '.join(
            f'{l} {b:.1f}->{a:.1f}' for l, b, a in zip(labels, hb, ha)))
    return r


def _figure(run, Eref, err, src, ok, z, old, refit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .well9d import HARTREE_EV
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    Eall = z['E_total_hartree'][:, 0] * HARTREE_EV
    e0 = np.nanmin(Eall)
    bins = np.linspace(0, max(2.2, float(np.nanmax(Eref)) + 0.1), 45)
    ax[0].hist(Eall[old] - e0, bins, color='0.6', label=f'before ({int(old.sum())})')
    ax[0].hist(Eref[ok], bins, color='C3', alpha=0.75, label=f'added ({int(ok.sum())})')
    ax[0].set_xlabel('CCSD(T) energy above the lowest point (eV)')
    ax[0].set_ylabel('points')
    ax[0].set_title('Where the data are', fontsize=11)
    ax[0].legend(frameon=False, fontsize=8)
    for k, s in enumerate(('beads', 'harmonic', 'rpmd_check')):
        m = ok & (src == s)
        if m.any():
            ax[1].scatter(Eref[m], err[m], s=12, color=f'C{k}', label=s, alpha=0.8)
    ax[1].axhline(0, color='0.5', lw=0.6)
    ax[1].set_yscale('symlog', linthresh=5)
    ax[1].set_xlabel('CCSD(T) energy above the minimum (eV)')
    ax[1].set_ylabel('surface before extension − CCSD(T) (meV)')
    ax[1].set_title('Old surface at the new points (out of sample)', fontsize=11)
    ax[1].legend(frameon=False, fontsize=8)
    if refit and refit.get('window_eV') and refit['window_eV'][0]:
        ax[1].axvline(refit['window_eV'][0], color='0.4', ls=':', lw=0.8)
        ax[1].text(refit['window_eV'][0], ax[1].get_ylim()[1], ' old fit window', fontsize=7, va='top', color='0.3')
    fig.tight_layout()
    fig.savefig(run.p('figures', 'W9X_extend.png'), dpi=150)
    plt.close(fig)
