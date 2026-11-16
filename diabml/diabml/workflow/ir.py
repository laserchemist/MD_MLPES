"""IR spectra of hot CH2OO from dynamics on the production surface and the dipole surface.

    python -m diabml.workflow.ch2oo ir RUN [--jobs 6]     # needs models/production.pt and models/dipole.pt
                                                          # -> RUN/ir.html, results/ir_*.npz, results/ir_spectrum.csv

For every temperature in [ir] temperatures, `trajectories` trajectories (different seeds, run in parallel): Langevin
for equil_fs at that temperature, then constant energy for run_fs on S0, the lab-frame dipole recorded from the
dipole surface. The spectrum is the averaged power spectrum of the dipole's time derivative (harmonic quantum
correction; diabml/dynamics/ir.py). The harmonic spectrum at the S0 minimum (same surfaces) is drawn as sticks for
reference: the shift and broadening of the bands with temperature is the anharmonic, hot-molecule effect.

Quality per trajectory: energy drift (the surface and time step) and the largest ensemble spread on S0 (whether the
trajectory stayed where the surface is trusted; above ~0.25 eV it did not, see the explore stage).

Limits of a reduced model: only the coordinates the surface describes carry dipole changes (for the 3-D CH2OO surface:
O-O stretch, C-O stretch, COO bend and their overtones and combinations). The CH2 frame is held by restraint
springs; harmonic modes that are mostly H motion are those springs, marked and left out of the comparison.
"""
from __future__ import annotations

import concurrent.futures as cf
import html
import json
import time

import numpy as np

from . import ch2oo as W


_SFX = ''                     # '_well' (polynomial well) or '_9d' (9-D well): outputs side by side


def _suffix(surf):
    return '_9d' if surf == 'well9d' else '_well' if surf == 'well' or surf.startswith('frozen:') else ''


def _traj(args):
    from ..dynamics.ir import run_ir_trajectory
    return run_ir_trajectory(*args[:2], **args[2])


def stage_ir(run, jobs=None):
    from ..dynamics.ir import find_peaks, harmonic_ir, ir_spectrum
    ic = run.cfg['ir']
    jobs = int(jobs or ic['jobs'])
    global _SFX
    surf = str(ic.get('surface', 'production'))
    _SFX = _suffix(surf)
    mp = run.p('models', ic['model'])
    well = None
    if surf == 'well9d':
        from ..wellfit import PolyWell
        mp = run.p('models', str(ic.get('model9d', 'well9d.npz')))
        if not mp.exists():
            raise SystemExit(f'{mp} not found: run well9d and well9d_fit first')
        well = PolyWell.load(mp)
        run.log(f'surface: the 9-D well {mp.name} ({well.meta.get("source", "")}, order {well.meta.get("order")}); '
                f'all nine modes free, no restraint springs')
    elif surf.startswith('frozen:'):
        from .shell import frozen_path, load_frozen
        well, man = load_frozen(run, surf)
        mp = frozen_path(run, surf.split(':', 1)[1])
        run.log(f'surface: the frozen {surf} (made {man.get("created")}; no refit) -> {mp}')
    elif _SFX:
        from ..wellfit import fit_well
        if ic.get('well_source', 'refs') == 'gs_grid':
            from .gs_grid import load_gs_grid
            wref = load_gs_grid(run, W.load_refs(run))
            run.log(f'well polynomial fitted to the ground-state grid data/gs_grid.npz ({len(wref)} points)')
        else:
            wref = W.load_refs(run)
        well, winfo = fit_well(wref, int(ic['well_order']), float(ic['well_window_eV']))
        mp = well.save(run.p('models', 'well_poly.npz'))
        run.log(f'surface: polynomial well fit (order {winfo["order"]}, {winfo["n_points"]} points with S0 within '
                f'{winfo["window_eV"]} eV, {winfo["n_terms"]} terms), residual RMS {winfo["residual_rms_meV"]:.1f} meV; '
                f'valid in {dict((k, [round(v[0], 3), round(v[1], 3)]) for k, v in winfo["box"].items())} '
                f'-> models/well_poly.npz')
    dp = run.p('models', ic['dipole'] if surf != 'well9d' else str(ic.get('dipole9d', 'dipole9d.npz')))
    for p, what in ((mp, 'production'), (dp, 'dipole_fit' if surf != 'well9d' else 'well9d_dipole')):
        if not p.exists():
            raise SystemExit(f'{p} not found: run the {what} stage first')
    conds = [(f'{float(T):.0f} K', {'temperature': float(T), 'equil_fs': float(ic['equil_fs'])})
             for T in ic['temperatures']]
    conds += [(f'{float(E):g} kJ/mol', {'energy_eV': float(E) / 96.485, 'rot_temperature': float(ic['rot_temperature'])})
              for E in ic.get('energies_kJmol', [])]
    temps = [c[0] for c in conds]                                     # condition labels, in order
    nt = int(ic['trajectories'])
    run.log(f'{nt} trajectories for each of {temps}: thermal = {ic["equil_fs"]} fs Langevin, energy = microcanonical '
            f'normal-mode sampling above the S0 minimum (+ rotation at {ic["rot_temperature"]} K); then '
            f'{ic["run_fs"]} fs at constant energy, dt {ic["dt"]} fs; models {mp.name}, {dp.name}; {jobs} processes')
    rs = float(ic.get('restraint_scale', 1.0))
    if surf == 'well9d':
        from ..dynamics.ir import harmonic_ir9
        rows, x0 = harmonic_ir9(str(mp), str(dp))
    else:
        rows, x0 = harmonic_ir(str(mp), str(dp), restraint_scale=rs)
    run.log('harmonic at the S0 minimum ' + str({n: round(float(v), 3) for n, v in zip(['r_OO', 'a_COO', 'r_CO'], x0)})
            + ': ' + ', '.join(f'{r["cm-1"]:.0f} cm-1 (I {r["intensity"]:.2f}' + (', H springs' if r.get('spring', r['H_fraction'] > 0.5)
                                                                                    else '') + ')' for r in rows))
    tasks = [(str(mp), str(dp), dict(kw, seed=s, label=lab, run_fs=float(ic['run_fs']), dt=float(ic['dt']),
                                     restraint_scale=rs,
                                     every=int(ic['every'])))
             for lab, kw in conds for s in range(nt)]
    out = {T: [] for T in temps}
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_traj, t): t for t in tasks}
        for k, f in enumerate(cf.as_completed(futs), 1):
            r = f.result()
            out[r['label']].append(r)
            drift = (r['E_tot'][-1] - r['E_tot'][0]) * 1000
            above = '' if r['E_min'] is None else f', {(r["E_tot"].mean() - r["E_min"]) * 96.485:.1f} kJ/mol above min'
            run.log(f'  {r["label"]} seed {r["seed"]}: drift {drift:+.2f} meV, rms '
                    f'{np.std(r["E_tot"]) * 1000:.2f} meV, '
                    + (f'max spread {np.nanmax(r["spread"]) * 1000:.0f} meV, ' if np.isfinite(r["spread"]).any() else '')
                    + f'r_OO {r["x"][:, 0].min():.2f}-{r["x"][:, 0].max():.2f} Å{above} ({r["seconds"]:.0f} s; '
                    f'{k}/{len(tasks)}, {(time.time() - t0) / 60:.1f} min)')
    spectra, summary = {}, {'harmonic': rows, 'minimum': x0.tolist(), 'temperatures': {}}
    fine = {}                                  # the same spectra at zoom_broaden_cm, for the band-contour figures
    for T in temps:
        R = sorted(out[T], key=lambda r: r['seed'])
        n = min(len(r['mu']) for r in R)
        mu = np.stack([r['mu'][:n] for r in R])
        nu, I = ir_spectrum(mu, R[0]['dt_record'], broaden_cm=float(ic['broaden_cm']), max_cm=float(ic['max_cm']),
                            min_cm=float(ic['min_cm']))
        spectra[T] = (nu, I)
        fine[T] = ir_spectrum(mu, R[0]['dt_record'], broaden_cm=float(ic.get('zoom_broaden_cm', ic['broaden_cm'])),
                              max_cm=float(ic['max_cm']), min_cm=float(ic['min_cm']))
        pk = find_peaks(nu, I, min_cm=float(ic['min_cm']))
        bad = [r['seed'] for r in R if (np.nanmax(r['spread']) > float(ic['max_spread_eV'])
                                        if well is None else not well.inside(r['x']).all())]
        summary['temperatures'][T] = {
            'peaks_cm-1': pk, 'trajectories': len(R), 'left_trusted_surface': bad,
            'drift_meV': [float((r['E_tot'][-1] - r['E_tot'][0]) * 1000) for r in R],
            'max_spread_meV': [float(np.nanmax(r['spread']) * 1000) if np.isfinite(r['spread']).any() else 0.0
                               for r in R],
            'mean_total_energy_eV': float(np.mean([r['E_tot'].mean() for r in R]))}
        tag = T.replace(' ', '').replace('/', 'per')
        th = max(1, int(ic.get('save_every', 10)))    # the dipole at every record (spectra); the rest thinned, float32
        np.savez(run.p('results', f'ir{_SFX}_{tag}.npz'), t=R[0]['t'][:n], mu=mu.astype(np.float32),
                 x=np.stack([r['x'][:n:th] for r in R]).astype(np.float32),
                 E_tot=np.stack([r['E_tot'][:n:th] for r in R]), spread=np.stack([r['spread'][:n:th] for r in R]),
                 save_every=th, nu=nu, intensity=I, intensity_fine=fine[T][1])
        run.log(f'{T}: peaks ' + ', '.join(f'{p:.0f} ({a:.2f})' for p, a in pk)
                + (f'; trajectories {bad} left the trusted surface' if bad else ''))
    with open(run.p('results', f'ir{_SFX}_spectrum.csv'), 'w') as fh:
        nu = spectra[temps[0]][0]
        fh.write('wavenumber_cm-1,' + ','.join('I_' + T.replace(' ', '') for T in temps) + '\n')
        for k in range(len(nu)):
            fh.write(f'{nu[k]:.2f},' + ','.join(f'{np.interp(nu[k], *spectra[T]):.5f}' for T in temps) + '\n')
    run.p('results', f'ir{_SFX}.json').write_text(json.dumps(summary, indent=1))
    _figures(run, spectra, rows, out, temps, fine)
    _html(run, ic, summary, temps)
    run.log(f'wrote {run.p(f"ir{_SFX}.html")}, results/ir{_SFX}_spectrum.csv, results/ir{_SFX}.json, '
            f'results/ir{_SFX}_*.npz')


def _figures(run, spectra, rows, out, temps, fine=None):
    plt = W._mpl()
    cols = ['#2F6DB5', '#C8873A', '#B03A2E', '#2E9A6B', '#6B4FA0']
    fig, ax = plt.subplots(figsize=(11, 5))
    off = 0.0
    for k, T in enumerate(temps):
        nu, I = spectra[T]
        ax.plot(nu, np.minimum(I, 1.05) + off, color=cols[k % len(cols)], lw=1.4, label=T)
        off += 1.1
    ref = _reference(run)
    if ref is not None:
        nu_r, I_r, lab = ref
        ax.plot(nu_r, I_r + off, color='0.25', lw=1.1, label=lab)
        off += 1.1
    ymax = off
    for r in rows:
        ls = ':' if r.get('spring', r['H_fraction'] > 0.5) else '-'
        ax.vlines(r['cm-1'], -0.35, -0.35 + 0.3 * r['intensity'], color='k', lw=2, linestyles=ls)
    ax.axhline(-0.02, color='0.7', lw=0.6)
    ax.axvspan(0, run.cfg['ir']['min_cm'], color='0.92', zorder=0)
    ax.text(10, ymax, 'rotation\n(clipped)', fontsize=8, va='top', color='0.4')
    ax.set(xlabel='wavenumber (cm⁻¹)', ylabel='IR intensity (normalised, offset)', ylim=(-0.4, ymax + 0.1),
           title='CH₂OO IR from dynamics (curves) and harmonic at the minimum (sticks'
                 + (')' if _SFX == '_9d' else '; dotted: CH₂ restraint springs)'))
    ax.set_xlim(0, min(4000, max(spectra[temps[0]][0])))
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(run.p('figures', f'IR1_spectrum{_SFX}.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for k, T in enumerate(temps):
        r = sorted(out[T], key=lambda r: r['seed'])[0]
        c = cols[k % len(cols)]
        ax[0].plot(r['t'] / 1000, (r['E_tot'] - r['E_tot'][0]) * 1000, color=c, lw=0.8, label=T)
        m = r['t'] <= 1000
        ax[1].plot(r['t'][m], np.linalg.norm(r['mu'][m], axis=1) * 4.8032, color=c, lw=0.8)
        ax[2].plot(r['x'][:, 0], r['x'][:, 1], color=c, lw=0.3, alpha=0.7)
    ax[0].set(xlabel='t (ps)', ylabel='E_tot − E_tot(0) (meV)', title='Energy conservation (seed 0)')
    ax[0].legend(fontsize=8)
    ax[1].set(xlabel='t (fs)', ylabel='|μ| (D)', title='Dipole, first ps')
    ax[2].set(xlabel='r(O–O) (Å)', ylabel='∠COO (°)', title='Where the trajectories went')
    fig.tight_layout()
    fig.savefig(run.p('figures', f'IR2_trajectories{_SFX}.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)
    run.log(f'figures/IR1_spectrum{_SFX}.png, figures/IR2_trajectories{_SFX}.png')
    _zoom(run, fine or spectra, rows, temps)
    _contour(run, fine or spectra, rows, temps)


def _zoom(run, spectra, rows, temps, lo=750.0, hi=1550.0):
    """IR3: the 750-1550 cm-1 region, each spectrum normalised inside the window, against the experiment (with its
    band centres) and the harmonic sticks of the surface."""
    plt = W._mpl()
    cols = ['#2F6DB5', '#C8873A', '#B03A2E', '#2E9A6B', '#6B4FA0']
    fig, ax = plt.subplots(figsize=(11, 5.6))
    off = 0.0
    for k, T in enumerate(temps):
        nu, I = spectra[T]
        m = (nu >= lo) & (nu <= hi)
        ax.plot(nu[m], I[m] / I[m].max() + off, color=cols[k % len(cols)], lw=1.5, label=T)
        off += 1.15
    ref = _reference(run)
    if ref is not None:
        nu_r, I_r, lab = ref
        m = (nu_r >= lo) & (nu_r <= hi)
        y = I_r[m] - np.percentile(I_r[m], 5)
        ax.plot(nu_r[m], y / y.max() + off, color='0.2', lw=1.1, label=lab)
        off += 1.15
    b = _reference_bands(run)
    if b is not None:
        for md in b['modes']:
            if md[2] and lo <= md[2][0] <= hi:
                ax.axvline(md[2][0], color='0.55', lw=0.8, ls=':')
                ax.text(md[2][0], off + 0.02, f'{md[0]} {md[2][0]}', rotation=90, fontsize=8, ha='center', va='bottom',
                        color='0.35')
            h = md[3]
            if h and lo <= h <= hi and md[0] in b.get('in_3d_model', {}):
                ax.plot([h, h], [-0.62, -0.42], color='#6B4FA0', lw=2)
    for r in rows:
        if lo <= r['cm-1'] <= hi:
            ax.plot([r['cm-1']] * 2, [-0.3, -0.3 + 0.25 * max(r['intensity'], 0.08)], color='k', lw=2,
                    ls=':' if r.get('spring', r['H_fraction'] > 0.5) else '-')
    dvr = _dvr(run)
    for l, v in (dvr or {}).items():
        if lo <= v <= hi:
            ax.plot([v], [-0.5], marker='v', color='#B03A2E', ms=7)
    if dvr:
        ax.plot([], [], marker='v', color='#B03A2E', ls='', label='this surface, ' + _quantum_label())
    ax.plot([], [], color='k', lw=2, label='this surface, harmonic at the minimum'
            + ('' if _SFX == '_9d' else ' (dotted: CH₂ springs)'))
    if b is not None:
        ax.plot([], [], color='#6B4FA0', lw=2, label='NEVPT2/aVDZ harmonic (Su et al.), modes in the 3-D model')
    ax.set(xlim=(hi, lo), ylim=(-0.7, off + 0.55), xlabel='wavenumber (cm⁻¹)',
           ylabel='IR intensity (normalised in this window, offset)', title='CH₂OO, 750–1550 cm⁻¹')
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(run.p('figures', f'IR3_zoom{_SFX}.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)
    run.log(f'figures/IR3_zoom{_SFX}.png')


def _vpt2(run):
    """For the 9-D surface: {label: (GVPT2 fundamental, harmonic)} from results/well9d_vpt2.json, or None."""
    p = run.p('results', 'well9d_vpt2.json')
    if _SFX != '_9d' or not p.exists():
        return None
    return {m['label']: (float(m['gvpt2']), float(m['harmonic'])) for m in json.loads(p.read_text())['modes']}


def _quantum_label():
    return 'anharmonic fundamentals (well9d_vpt2, GVPT2)' if _SFX == '_9d' else 'quantum fundamentals (vib_levels, DVR)'


def _dvr(run):
    """Quantum fundamentals {mode: cm-1}: GVPT2 for the 9-D surface (well9d_vpt2), else results/vib_levels.json
    (the vib_levels stage), if it has run."""
    v = _vpt2(run)
    if v is not None:
        return {k: f for k, (f, _) in v.items()}
    p = run.p('results', 'vib_levels.json')
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {k: float(v) for k, v in d.get('fundamentals_cm-1', {}).items()}


def _centroid(nu, I, lo, hi):
    m = (nu >= lo) & (nu <= hi)
    y = np.clip(I[m] - np.percentile(I[m], 5), 0, None)
    return float((nu[m] * y).sum() / y.sum()) if y.sum() > 0 else float('nan')


def _contour(run, spectra, rows, temps, lo=840.0, hi=1000.0):
    """IR4: the O-O band contour against experiment. Left: as computed. Right ([ir] contour_align):
    "centroid" (default) moves each computed band so that its intensity-weighted centre sits on the quantum O-O
    fundamental of the same surface (vib_levels), so only the shape is compared with experiment and nothing is
    counted twice: classical dynamics already carries part of the anharmonic shift at the bottom of the band;
    "shift" moves it by the full zero-point shift (fundamental - harmonic), the older choice, which double-counts
    that part when the classical band sits below the harmonic frequency."""
    plt = W._mpl()
    cols = ['#2F6DB5', '#C8873A', '#B03A2E', '#2E9A6B', '#6B4FA0']
    align = str(run.cfg['ir'].get('contour_align', 'centroid'))
    p = run.p('results', 'vib_levels.json')
    fund = harm = None
    v9 = _vpt2(run)
    if v9 is not None and 'O-O stretch' in v9:
        fund, harm = v9['O-O stretch']
    elif p.exists():
        d = json.loads(p.read_text())
        h = dict(zip(d.get('harmonic_labels', []), d.get('harmonic_cm-1', [])))
        f = d.get('fundamentals_cm-1', {})
        if 'O-O' in h and 'O-O' in f:
            fund, harm = float(f['O-O']), float(h['O-O'])
    wlo, whi = lo + 20, hi - 20                                  # the window the band centres are taken over
    ref = _reference(run)
    cen = {T: _centroid(*spectra[T], wlo, whi) for T in temps}
    cen_ref = _centroid(ref[0], ref[1], wlo, whi) if ref is not None else None
    if fund is not None:
        dxs = {T: (fund - cen[T]) if align == 'centroid' else (fund - harm) for T in temps}
    fig, axes = plt.subplots(1, 2 if fund is not None else 1, figsize=(13 if fund is not None else 7, 4.6),
                             squeeze=False)
    for j, ax in enumerate(axes[0]):
        if ref is not None:
            nu_r, I_r, lab = ref
            m = (nu_r >= lo) & (nu_r <= hi)
            y = I_r[m] - np.percentile(I_r[m], 5)
            ax.plot(nu_r[m], y / y.max(), color='0.2', lw=1.1,
                    label=lab + (f'; centre {cen_ref:.0f}' if cen_ref is not None and j == 0 else ''))
        for k, T in enumerate(temps):
            dx = dxs[T] if j == 1 else 0.0
            nu, I = spectra[T]
            m = (nu + dx >= lo) & (nu + dx <= hi)
            ax.plot(nu[m] + dx, I[m] / I[m].max(), color=cols[k % len(cols)], lw=1.5,
                    label=T + (f' (moved {dx:+.0f} cm⁻¹)' if j == 1 else f'; centre {cen[T]:.0f}'))
        if j == 0:
            for r in rows:
                if lo <= r['cm-1'] <= hi and not r.get('spring', r['H_fraction'] > 0.5):
                    ax.axvline(r['cm-1'], color='k', lw=0.8, ls='--')
        else:
            ax.axvline(fund, color='#B03A2E', lw=1.0, ls=':')
        ttl = ('O–O band, as computed' if j == 0 else
               (f'O–O band centred on the quantum fundamental ({fund:.0f} cm⁻¹)' if align == 'centroid'
                else 'O–O band, moved by the quantum anharmonic shift'))
        ax.set(xlim=(hi, lo), ylim=(0, 1.15), xlabel='wavenumber (cm⁻¹)', ylabel='normalised in window', title=ttl)
        ax.legend(fontsize=8, loc='upper right')
    fig.text(0.01, 0.005, f'computed spectra at {run.cfg["ir"].get("zoom_broaden_cm", run.cfg["ir"]["broaden_cm"])} '
             f'cm⁻¹ broadening; centres: intensity-weighted over {wlo:.0f}–{whi:.0f} cm⁻¹; left dashed: harmonic O–O; '
             'right dotted: O–O fundamental (' + ('well9d_vpt2' if _SFX == '_9d' else 'vib_levels') + ')', fontsize=8, color='0.4')
    fig.tight_layout()
    fig.savefig(run.p('figures', f'IR4_contour{_SFX}.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)
    run.log(f'figures/IR4_contour{_SFX}.png: O-O band centres ' + ', '.join(f'{T} {c:.1f}' for T, c in cen.items())
            + (f', experiment {cen_ref:.1f}' if cen_ref is not None else '') + ' cm-1'
            + (f'; quantum fundamental {fund:.1f}, harmonic {harm:.1f} (align: {align})' if fund is not None else ''))
    return {'centres_cm-1': cen, 'experiment_centre_cm-1': cen_ref, 'fundamental_cm-1': fund, 'harmonic_cm-1': harm,
            'align': align}


def stage_ir_replot(run):
    """Redraw the zoom and contour figures (IR3, IR4) and ir.html from the saved spectra of the last ir run, without
    running trajectories again (e.g. after vib_levels or after changing [ir] contour_align or zoom_broaden_cm)."""
    global _SFX
    ic = run.cfg['ir']
    surf = str(ic.get('surface', 'production'))
    _SFX = _suffix(surf)
    sp = run.p('results', f'ir{_SFX}.json')
    if not sp.exists():
        raise SystemExit(f'{sp} not found: run the ir stage first')
    summary = json.loads(sp.read_text())
    temps = list(summary['temperatures'])
    spectra, fine = {}, {}
    for T in temps:
        z = np.load(run.p('results', f'ir{_SFX}_{T.replace(" ", "").replace("/", "per")}.npz'))
        spectra[T] = (z['nu'], z['intensity'])
        fine[T] = (z['nu'], z['intensity_fine'] if 'intensity_fine' in z.files else z['intensity'])
    rows = summary['harmonic']
    _zoom(run, fine, rows, temps)
    summary['contour'] = _contour(run, fine, rows, temps)
    sp.write_text(json.dumps(summary, indent=1, default=float))
    _html(run, ic, summary, temps)
    run.log(f'redrew figures/IR3_zoom{_SFX}.png, figures/IR4_contour{_SFX}.png and ir{_SFX}.html')


def _reference(run):
    """(wavenumbers, intensity normalised to 1, label) of [ir] reference (a CSV: wavenumber, intensity, ...), or None."""
    path = run.cfg['ir'].get('reference')
    if not path:
        return None
    p = W.C.resolve(path, run.path)
    if not p.exists():
        run.log(f'reference spectrum {p} not found: not drawn')
        return None
    a = np.loadtxt(p, delimiter=',', comments='#')
    a = a[np.argsort(a[:, 0])]
    I = a[:, 1] - np.percentile(a[:, 1], 10)
    return a[:, 0], I / I.max(), run.cfg['ir'].get('reference_label', p.stem)


def _reference_bands(run):
    path = run.cfg['ir'].get('reference')
    if not path:
        return None
    p = W.C.resolve(path, run.path)
    b = p.with_name(p.name.replace('_fig2B.csv', '_bands.json'))
    return json.loads(b.read_text()) if b.exists() else None


def _html(run, ic, s, temps):
    import base64
    sf = str(ic.get('surface', 'production'))
    surf_label = html.escape(f'models/surfaces/{sf.split(":", 1)[1]}.npz (frozen)' if sf.startswith('frozen:')
                             else f'models/{ic.get("model9d", "well9d.npz")}' if _SFX == '_9d'
                             else ('models/well_poly.npz' if _SFX else f'models/{ic["model"]}'))
    nine = _SFX == '_9d'
    labels = _mode_labels(run) if nine else {}
    img = lambda n: base64.b64encode(run.p('figures', n).read_bytes()).decode()
    harm = ''.join(f'<tr><td>{r["cm-1"]:.0f}</td><td>{r["intensity"]:.3f}</td><td>{r["H_fraction"]:.2f}</td>'
                   f'<td>{_harm_label(r, nine, labels)}</td></tr>'
                   for r in s['harmonic'])
    title = {'_9d': ' (9-D surface: all nine internal coordinates, no restraints)',
             '_well': ' (polynomial well surface, CH₂ held by restraint springs)'}.get(_SFX, '')
    note = ('The surface describes all nine internal coordinates (r(O–O), ∠COO, r(C–O), both C–H bonds, both HCO '
            'angles and both out-of-plane dihedrals); every atom moves freely, there are no restraint springs, and '
            'all nine harmonic modes are physical.' if nine else
            'The surface describes r(O–O), ∠COO and r(C–O) only; the CH₂ frame is held by restraint springs, whose '
            'modes carry no dipole change here.')
    quantum = 'well9d_vpt2 (GVPT2)' if nine else 'vib_levels'
    tem = ''
    for T in temps:
        d = s['temperatures'][T]
        tem += (f'<tr><td>{html.escape(T)}</td><td>{", ".join(f"{p:.0f}" for p, a in d["peaks_cm-1"])}</td>'
                f'<td>{d["mean_total_energy_eV"]:.2f}</td><td>{max(abs(x) for x in d["drift_meV"]):.2f}</td>'
                f'<td>{max(d["max_spread_meV"]):.0f}</td><td>{d["left_trusted_surface"] or "none"}</td></tr>')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CH2OO IR from dynamics</title><style>
:root{{--paper:#F6F4F0;--ink:#2E2A36;--muted:#7A7480;--rule:#DAD3C8;--panel:#EFEBF3}}
@media (prefers-color-scheme:dark){{:root{{--paper:#17131f;--ink:#ece8f2;--muted:#a69cb6;--rule:#3d3250;--panel:#241b33}}}}
body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,-apple-system,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:20px 16px}} h1{{font:600 28px Georgia,serif}} h2{{font:600 21px Georgia,serif;margin-top:30px}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0}} th,td{{border-bottom:1px solid var(--rule);padding:6px 8px;text-align:left}}
th{{background:var(--panel)}} img{{max-width:100%;background:#fff;border:1px solid var(--rule)}} p.n{{color:var(--muted);font-size:14px}}
</style></head><body><main>
<h1>CH₂OO infrared spectrum from dynamics: {html.escape(run.path.name)}{title}</h1>
<p>Surface <code>{surf_label}</code>, dipole <code>models/{html.escape(ic["dipole"])}</code>.
{ic["trajectories"]} trajectories per temperature: {ic["equil_fs"]} fs Langevin then {ic["run_fs"]} fs at constant energy (dt {ic["dt"]} fs).
Spectrum: power spectrum of dμ/dt (harmonic quantum correction), {ic["broaden_cm"]} cm⁻¹ Gaussian broadening.</p>
<img src="data:image/png;base64,{img(f"IR1_spectrum{_SFX}.png")}" alt="IR spectra">
<img src="data:image/png;base64,{img(f"IR3_zoom{_SFX}.png")}" alt="IR spectra, 750-1550 cm-1">
<h2>O–O band contour</h2>
<p class="n">Spectra at {ic.get("zoom_broaden_cm", ic["broaden_cm"])} cm⁻¹ broadening. Classical dynamics gives the band shape of a
rotating, vibrating molecule (rotational P/Q/R contour, hot-band anharmonicity) but not the full zero-point shift;
where the {quantum} stage has run, the right panel centres each computed band (intensity-weighted) on the quantum O–O
fundamental of the same surface, so that only the shape is compared with experiment ([ir] contour_align = "centroid";
"shift" moves it by fundamental − harmonic instead, which double-counts the part of the shift the classical band
already has).</p>
<img src="data:image/png;base64,{img(f"IR4_contour{_SFX}.png")}" alt="O-O band contour">
<h2>Bands by condition</h2>
<table><tr><th>condition</th><th>peaks (cm⁻¹)</th><th>mean total energy (eV)</th><th>largest |drift| (meV)</th><th>largest spread (meV)</th><th>left trusted surface</th></tr>{tem}</table>
{_ref_table(run, nine)}
<h2>Harmonic spectrum at the S0 minimum</h2>
<table><tr><th>cm⁻¹</th><th>relative intensity</th><th>H fraction</th><th></th></tr>{harm}</table>
<p class="n">{note}</p>
<h2>Trajectories</h2>
<img src="data:image/png;base64,{img(f"IR2_trajectories{_SFX}.png")}" alt="trajectory diagnostics">
</main></body></html>'''
    run.p(f'ir{_SFX}.html').write_text(page)


def _mode_labels(run):
    """Herzberg names and descriptions of the 9-D harmonic modes, by harmonic wavenumber (well9d_vpt2)."""
    p = run.p('results', 'well9d_vpt2.json')
    if not p.exists():
        return {}
    return {round(m['harmonic']): f'{m["mode"]} {m["label"]}' for m in json.loads(p.read_text())['modes']}


def _harm_label(r, nine, labels):
    if nine:
        k = round(r['cm-1'])
        return next((v for h, v in labels.items() if abs(h - k) <= 1), 'normal mode')
    return 'CH₂ restraint spring (not physical)' if r.get('spring', r['H_fraction'] > 0.5) else 'model mode'


def _ref_table(run, nine=False):
    b = _reference_bands(run)
    if b is None:
        return ''
    rows = ''
    for m in b['modes']:
        e = f'{m[2][0]} ({m[2][1]})' if m[2] else ''
        inm = '' if nine else f'<td>{html.escape(b.get("in_3d_model", {}).get(m[0], ""))}</td>'
        rows += (f'<tr><td>{m[0]}</td><td>{html.escape(m[1])}</td><td>{e}</td><td>{m[3]}</td><td>{m[4][0]} ({m[4][1]})</td>'
                 f'<td>{m[6]}</td><td>{html.escape(m[7])}</td>{inm}</tr>')
    return (f'<h2>Reference: experiment and published calculations</h2><p class="n">{html.escape(b["citation"])}. '
            f'{html.escape(b["spectrum"])}.</p><table><tr><th>mode</th><th>sym.</th><th>expt. (rel. int.)</th>'
            f'<th>NEVPT2/aVDZ harm.</th><th>NEVPT2/aVDZ anharm.</th><th>CCSD(T)/aVTZ</th><th>description</th>'
            + ('' if nine else '<th>in the 3-D model</th>') + f'</tr>{rows}</table>')
