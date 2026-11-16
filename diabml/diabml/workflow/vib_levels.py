"""Quantum vibrational levels of the reduced CH2OO model on the polynomial well surface (diabml/vibdvr.py).

    python -m diabml.workflow.ch2oo vib_levels RUN    # -> RUN/vib_levels.html, results/vib_levels.{json,csv}

The well polynomial ([vib] order, window_eV; the same fit as [ir] surface = "well") is refitted from the run's
reference data, and the J = 0 vibrational Schroedinger equation for r_OO, angle COO and r_CO (CH2 rigid) is solved on
a grid. Output: the fundamentals, overtones and combinations with assignments and intensities from the dipole surface,
next to experiment and the NEVPT2 harmonic / anharmonic values of Su et al. (2013) ([ir] reference) and, if the ir
stage has run on the well surface, the classical band positions.

Sensitivity: the harmonic frequencies are also computed for polynomials of several orders and energy windows
([vib] check_orders, check_windows_eV). If they disagree by more than the effect being measured, the reference data
do not determine the curvature of the well (scatter between points, e.g. from different SA-CASSCF solutions), and the
levels are only as good as that spread.
"""
from __future__ import annotations

import html
import json

import numpy as np

from . import ch2oo as W

_LAB = {'r_OO': 'r(O–O)', 'a_COO': '∠COO', 'r_CO': 'r(C–O)', 'a_HCO': '∠HCO (both H)', 'r_ch': 'r(C–H)',
        'r_CH': 'r(C–H)', 'a_hco': '∠HCO'}
_SU = {'O-O': 'v6', 'COO': 'v7', 'C-O': 'v4', 'HCO': 'v3'}          # Su et al. Table 1 modes of the 3-D model's coordinates


def _fit(run, ref, order, window):
    from ..wellfit import fit_well
    return fit_well(ref, int(order), float(window))


def _refs(run, vc):
    ref = W.load_refs(run)
    if vc.get('source', 'refs') == 'gs_grid':
        from .gs_grid import load_gs_grid
        g = load_gs_grid(run, ref)
        run.log(f'fitting the ground-state grid data/gs_grid.npz ({len(g)} points), not the reference data')
        return g
    codes = vc.get('only_audit') or []
    if codes and 'audit' in ref.extra:
        keep = np.where(np.isin(np.asarray(ref.extra['audit']), [int(c) for c in codes]))[0]
        run.log(f'using the {len(keep)} reference points with audit code in {codes}')
        ref = ref.subset(keep)
    return ref


def stage_vib_levels(run):
    from ..dipole import DipoleModel
    from ..vibdvr import harmonic_frequencies, vib_levels
    vc = run.cfg['vib']
    frozen = str(vc.get('source', '')).startswith('frozen:')
    if frozen:
        from .shell import load_frozen
        well, man = load_frozen(run, vc['source'])
        info = dict(man.get('fit', {}), frozen=vc['source'])
        ref = None
        run.log(f'surface: the frozen {vc["source"]} (made {man.get("created")}, {info.get("n_points")} points, order '
                f'{info.get("order")}); no refit, no sensitivity check')
    else:
        ref = _refs(run, vc)
        well, info = _fit(run, ref, vc['order'], vc['window_eV'])
        well.save(run.p('models', 'well_poly_vib.npz'))
        run.log(f'well polynomial: order {info["order"]}, {info["n_points"]} points with S0 within {info["window_eV"]} eV, '
                f'{info["n_terms"]} terms, residual RMS {info["residual_rms_meV"]:.1f} meV -> models/well_poly_vib.npz')
    dp = run.p('models', vc['dipole'])
    dip = DipoleModel.load(dp) if dp.exists() else None
    if dip is None:
        run.log(f'{dp} not found: levels without intensities')
    pts = vc['points']
    out = vib_levels(well, dip, n=[int(v) for v in pts] if isinstance(pts, (list, tuple)) else int(pts), n_states=int(vc['n_states']), ecut_eV=float(vc['ecut_eV']),
                     log=run.log)
    out['fit'] = info
    if not out['reliable']:
        run.log('warning: the levels are NOT reliable: the polynomial dips below its minimum on the DVR grid or the low '
                'states reach the grid edge (largest edge density '
                f'{max(s["edge_density"] for s in out["states"][:len(out["names"]) + 1]):.0e}). The data do not cover '
                'the wavefunctions: extend or recentre the gs_grid around the minimum '
                f'({", ".join(f"{n} {v:.3f}" for n, v in zip(out["names"], out["minimum"]))}), or use a lower order')
    run.log(f'grid {out["grid"]["n"]} over ' + ', '.join(f'{n} {lo:.3f}-{hi:.3f}' for n, (lo, hi) in
                                                          zip(out['names'], out['grid']['ranges']))
            + f'; harmonic test on the grid within {out["harmonic_grid_error_cm-1"]:.2f} cm-1')
    run.log('harmonic ' + ', '.join(f'{l} {v:.0f}' for l, v in zip(out['harmonic_labels'], out['harmonic_cm-1']))
            + f' cm-1; ZPE {out["zpe_cm-1"]:.0f} (harmonic {out["zpe_harmonic_cm-1"]:.0f}) cm-1')
    for s in out['states'][1:]:
        run.log(f'  {s["cm-1"]:7.1f} cm-1  {s["label"]:<14} overlap {s["overlap"]:.2f}'
                + (f'  I {s["intensity"]:.3f}' if 'intensity' in s else '')
                + (f'  (edge {s["edge_density"]:.0e})' if s['edge_density'] > 1e-3 else ''))
    # sensitivity of the curvature to the fit
    sens = []
    for o in ([] if frozen else vc['check_orders']):
        for wdw in vc['check_windows_eV']:
            try:
                w, inf = _fit(run, ref, o, wdw)
                f, x0 = harmonic_frequencies(w)
                sens.append({'order': int(o), 'window_eV': float(wdw), 'n_points': inf['n_points'],
                             'residual_rms_meV': inf['residual_rms_meV'], 'harmonic_cm-1': f})
            except Exception as e:                                   # too few points for an order
                run.log(f'  sensitivity order {o} window {wdw}: {e}')
    out['sensitivity'] = sens
    spread = {}
    for l in out['harmonic_labels']:
        v = [s['harmonic_cm-1'].get(l) for s in sens if l in s['harmonic_cm-1']]
        if v:
            spread[l] = [float(min(v)), float(max(v))]
    out['harmonic_spread_cm-1'] = spread
    run.log('harmonic range over fits (orders ' + ', '.join(map(str, vc['check_orders'])) + '; windows '
            + ', '.join(map(str, vc['check_windows_eV'])) + ' eV): '
            + ', '.join(f'{l} {a:.0f}-{b:.0f}' for l, (a, b) in spread.items()) + ' cm-1')
    worst = max((b - a for a, b in spread.values()), default=0)
    if worst > 40:
        run.log(f'warning: the harmonic frequencies depend on the fit by up to {worst:.0f} cm-1: the reference data do '
                'not pin the curvature of the well this closely (scatter between points); see vib_levels.html')
    run.p('results', 'vib_levels.json').write_text(json.dumps(out, indent=1, default=float))
    with open(run.p('results', 'vib_levels.csv'), 'w') as fh:
        fh.write('n,cm-1,assignment,quanta,overlap,intensity,edge_density\n')
        for s in out['states']:
            fh.write(f'{s["n"]},{s["cm-1"]:.2f},{s["label"]},{"-".join(map(str, s["quanta"]))},{s["overlap"]:.3f},'
                     f'{s.get("intensity", float("nan")):.4f},{s["edge_density"]:.2e}\n')
    _rot_log(run, out)
    _figure(run, out)
    _html(run, vc, out)
    run.log(f'wrote {run.p("vib_levels.html")}, results/vib_levels.json, results/vib_levels.csv')


def _classical(run):
    for name in ('ir_well.json', 'ir.json'):
        p = run.p('results', name)
        if p.exists():
            d = json.loads(p.read_text())
            return name, {T: [pk[0] for pk in v['peaks_cm-1']] for T, v in d['temperatures'].items()}
    return None, {}


_MODE_OF = {'C-O': 'v4', 'O-O': 'v6', 'HCO': 'v3'}


def _rot_reference():
    p = W.C.PACKAGE_ROOT / 'data' / 'reference' / 'ch2oo_rotational.json'
    return json.loads(p.read_text()) if p.exists() else None


def _rot_rows(out):
    """[(state label, inertial (A, B, C), with Coriolis (A, B, C) or None, ratios to v = 0 with Coriolis (or
    inertial if none), experiment row or None)] for the equilibrium, the ground state and the fundamentals."""
    rot = out.get('rotational_cm-1')
    if not rot:
        return []
    ref = _rot_reference()
    st = out['states']
    cor = lambda x: x.get('ABC_with_coriolis_cm-1')
    g, gc = st[0]['ABC_cm-1'], cor(st[0])
    rows = [('equilibrium', rot['equilibrium'], None, None, None),
            ('ground (v = 0)', g, gc, None, ref['states']['ground'] if ref else None)]
    for lab, f in out['fundamentals_cm-1'].items():
        s = min((x for x in st if x['label'] == lab and 'ABC_cm-1' in x), key=lambda x: abs(x['cm-1'] - f), default=None)
        if s is None:
            continue
        a, ac = s['ABC_cm-1'], cor(s)
        base, top = (gc, ac) if ac is not None and gc is not None else (g, a)
        e = None
        if ref:
            key = _MODE_OF.get(lab)
            e = ref['states'].get(key) if key else None
        rows.append((f'{lab} = 1 ({s["cm-1"]:.0f})', a, ac, [top[k] / base[k] for k in range(3)], e))
    return rows


def _rot_log(run, out):
    rows = _rot_rows(out)
    if not rows:
        return
    for lab, a, ac, rat, e in rows:
        line = f'  rotational {lab}: A {a[0]:.5f}, B {a[1]:.5f}, C {a[2]:.5f} cm-1'
        if ac is not None:
            line += f' (+ Coriolis: {ac[0]:.5f}, {ac[1]:.5f}, {ac[2]:.5f})'
        if rat is not None:
            line += f'; ratios to v = 0: {rat[0]:.5f} {rat[1]:.5f} {rat[2]:.5f}'
            if e and 'ratios' in e:
                line += f' (expt. {e["ratios"]["A"]:.5f} {e["ratios"]["B"]:.5f} {e["ratios"]["C"]:.5f})'
        elif e:
            line += f' (expt. {e["A"]:.5f} {e["B"]:.5f} {e["C"]:.5f})'
        run.log(line)
    c = out['rotational_cm-1'].get('coriolis')
    if c:
        z = np.asarray(c['zeta'])[2]
        lab = out['harmonic_labels']
        pairs = sorted(((abs(z[r, s_]), lab[r], lab[s_]) for r in range(len(lab)) for s_ in range(r + 1, len(lab))),
                       reverse=True)
        run.log('  Coriolis zeta^c (in-plane pairs): ' + ', '.join(f'{a}/{b} {v:.3f}' for v, a, b in pairs))


def _rot_html(out):
    rows = _rot_rows(out)
    if not rows:
        return ''
    ref = _rot_reference()
    f3 = lambda v: '' if v is None else ' / '.join(f'{x:.5f}' for x in v)
    body = ''
    for lab, a, ac, rat, e in rows:
        er = ''
        if e and 'ratios' in e:
            er = f'{e["ratios"]["A"]:.5f} / {e["ratios"]["B"]:.5f} / {e["ratios"]["C"]:.5f}'
        elif e:
            er = f'{e["A"]:.5f} / {e["B"]:.5f} / {e["C"]:.5f}'
        body += (f'<tr><td>{html.escape(lab)}</td><td>{f3(a)}</td><td>{f3(ac)}</td><td>{f3(rat)}</td>'
                 f'<td>{er}</td></tr>')
    cit = html.escape(ref['citation']) if ref else ''
    c = out['rotational_cm-1'].get('coriolis')
    zt = ''
    if c:
        lab = out['harmonic_labels']
        z = np.asarray(c['zeta'])[2]
        zt = ('<p class="n">Coriolis ζ<sup>c</sup> between the model’s modes: '
              + ', '.join(f'{lab[r]}/{lab[s_]} {z[r, s_]:+.3f}' for r in range(len(lab)) for s_ in range(r + 1, len(lab)))
              + '.</p>')
    return (f'<h2>Rotational constants</h2><div class="wrap"><table><tr><th>state</th><th>A / B / C, averaged</th>'
            f'<th>+ Coriolis</th><th>ratio to v = 0</th><th>experiment (constants, or ratios)</th></tr>{body}</table></div>'
            '<p class="n">cm⁻¹, main isotopologue (¹²C, ¹⁶O, ¹H). "Averaged": ⟨v|ħ²/2I<sub>g</sub>|v⟩ over the DVR states '
            'with the principal moments of each grid geometry (the anharmonic and inertial part of the '
            'vibration–rotation interaction). "+ Coriolis": plus −Σ<sub>r</sub> α<sub>r</sub><sup>Cor</sup>(v<sub>r</sub>+½) '
            'from second-order perturbation theory with the model’s harmonic modes (Mills). A planar model has only '
            'in-plane modes, so only C gets a Coriolis term; the partners outside the model (CH₂ rock ν5, C–H stretches, '
            f'out-of-plane ν8, ν9) are missing. Experiment: {cit}.</p>' + zt)


def _figure(run, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .ir import _reference
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ref = _reference(run)
    if ref is not None:
        nu, I, lab = ref
        m = (nu > 400) & (nu < 2500)
        ax.plot(nu[m], I[m], color='0.45', lw=1.0, label=lab)
    S = [s for s in out['states'][1:] if 'intensity' in s and s['cm-1'] < 2500]
    if S:
        mk, st, bl = ax.stem([s['cm-1'] for s in S], [max(s['intensity'], 0.004) for s in S], linefmt='C3-',
                             markerfmt='C3o', basefmt=' ', label='quantum levels (this surface), intensity')
        plt.setp(mk, ms=4)
    for l, v in zip(out['harmonic_labels'], out['harmonic_cm-1']):
        ax.axvline(v, color='C0', ls=':', lw=1)
        ax.text(v, 1.06, f'harm. {l}', color='C0', fontsize=8, ha='center')
    ax.set_xlim(400, 2500)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel('wavenumber (cm$^{-1}$)')
    ax.set_ylabel('relative intensity')
    ax.legend(loc='upper right', fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(run.p('figures', 'VL1_levels.png'), dpi=150)
    plt.close(fig)


def _html(run, vc, out):
    import base64
    from .ir import _reference_bands
    b = _reference_bands(run)
    su = {m[0]: m for m in b['modes']} if b else {}
    cname, cl = _classical(run)
    fund = out['fundamentals_cm-1']
    harm = dict(zip(out['harmonic_labels'], out['harmonic_cm-1']))
    spread = out.get('harmonic_spread_cm-1', {})
    rows = ''
    for l in out['harmonic_labels']:
        m = su.get(_SU.get(l, ''))
        e = f'{m[2][0]}' if m and m[2] else '–'
        rows += (f'<tr><td>{l}</td><td>{harm[l]:.0f}</td><td>{spread.get(l, [np.nan, np.nan])[0]:.0f}–'
                 f'{spread.get(l, [np.nan, np.nan])[1]:.0f}</td><td><b>{fund.get(l, float("nan")):.0f}</b></td>'
                 f'<td>{fund.get(l, float("nan")) - harm[l]:+.0f}</td><td>{e}</td>'
                 f'<td>{m[3] if m else "–"}</td><td>{m[4][0] if m else "–"}</td>'
                 f'<td>{(m[4][0] - m[3]) if m else 0:+}</td></tr>')
    lev = ''.join(f'<tr><td>{s["cm-1"]:.1f}</td><td>{html.escape(s["label"])}</td><td>{s["overlap"]:.2f}</td>'
                  f'<td>{s.get("intensity", float("nan")):.3f}</td><td>{s["edge_density"]:.0e}</td></tr>'
                  for s in out['states'][1:])
    sen = ''.join(f'<tr><td>{s["order"]}</td><td>{s["window_eV"]}</td><td>{s["n_points"]}</td>'
                  f'<td>{s["residual_rms_meV"]:.1f}</td>'
                  + ''.join(f'<td>{s["harmonic_cm-1"].get(l, float("nan")):.0f}</td>' for l in out['harmonic_labels'])
                  + '</tr>' for s in out.get('sensitivity', []))
    ctab = ''
    if cl:
        ctab = (f'<p class="n">Classical band positions from <code>results/{cname}</code>: '
                + '; '.join(f'{html.escape(T)}: ' + ', '.join(f'{p:.0f}' for p in v) for T, v in cl.items())
                + ' cm⁻¹. Classical dynamics at low energy stays near the harmonic values; the quantum levels include '
                  'the zero-point anharmonicity.</p>')
    img = base64.b64encode(run.p('figures', 'VL1_levels.png').read_bytes()).decode()
    f = out['fit']
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CH2OO vibrational levels</title><style>
:root{{--paper:#F6F4F0;--ink:#2E2A36;--muted:#7A7480;--rule:#DAD3C8;--panel:#EFEBF3}}
@media (prefers-color-scheme:dark){{:root{{--paper:#17131f;--ink:#ece8f2;--muted:#a69cb6;--rule:#3d3250;--panel:#241b33}}}}
body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,-apple-system,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:20px 16px}} h1{{font:600 28px Georgia,serif}} h2{{font:600 21px Georgia,serif;margin-top:30px}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0}} th,td{{border-bottom:1px solid var(--rule);padding:6px 8px;text-align:left}}
th{{background:var(--panel)}} img{{max-width:100%;background:#fff;border:1px solid var(--rule)}} p.n{{color:var(--muted);font-size:14px}}
.wrap{{overflow-x:auto}}
</style></head><body><main>
<h1>CH₂OO vibrational levels: {html.escape(run.path.name)}</h1>
<p>J = 0 levels of the {len(out["names"])}-D model ({html.escape(", ".join(_LAB.get(n, n) for n in out["names"]))};
everything else held at {html.escape(", ".join(f"{_LAB.get(k, k)} {v:g}" for k, v in out["fixed"].items()) or "reference values")}) on the well polynomial (order {f["order"]},
{f["n_points"]} points with S0 within {f["window_eV"]} eV, residual {f["residual_rms_meV"]:.1f} meV), sinc-DVR grid
{"×".join(map(str, out["grid"]["n"]))}, exact G matrix. ZPE {out["zpe_cm-1"]:.0f} cm⁻¹ (harmonic {out["zpe_harmonic_cm-1"]:.0f}).
Harmonic check on the grid: {out["harmonic_grid_error_cm-1"]:.2f} cm⁻¹.</p>
{"" if out.get("reliable", True) else '<p style="border-left:4px solid #B03A2E;padding:8px 12px;background:var(--panel)"><b>Levels not reliable.</b> The polynomial dips below its minimum on the DVR grid or the low states reach the grid edge: the data do not cover the wavefunctions. Harmonic values and the sensitivity table are still valid. Extend or recentre the grid around the minimum.</p>'}
<img src="data:image/png;base64,{img}" alt="levels and experiment">
<h2>Fundamentals</h2><div class="wrap">
<table><tr><th>mode</th><th>harmonic</th><th>harmonic, range over fits</th><th>quantum fundamental</th><th>anharm. shift</th>
<th>expt.</th><th>NEVPT2/aVDZ harm.</th><th>NEVPT2/aVDZ anharm.</th><th>their shift</th></tr>{rows}</table></div>
<p class="n">Experiment and NEVPT2/aVDZ: Su, Huang, Witek &amp; Lee, Science 340, 174 (2013). In the molecule C–O
(v4) and the CH₂ scissor (v3, HCO here) are mixed; without ∠HCO in the model C–O carries both. The range over fits is the spread of the harmonic
frequency over polynomial orders and energy windows (below): when it is wider than the anharmonic shift, the reference
data, not the method, limit the comparison.</p>
{ctab}
{_rot_html(out)}
<h2>Levels</h2><div class="wrap">
<table><tr><th>cm⁻¹ above ground</th><th>assignment</th><th>overlap with harmonic state</th><th>relative intensity</th><th>density at grid edge</th></tr>{lev}</table></div>
<h2>Sensitivity of the curvature to the fit</h2><div class="wrap">
<table><tr><th>order</th><th>window (eV)</th><th>points</th><th>residual (meV)</th>{"".join(f"<th>{l} harm.</th>" for l in out["harmonic_labels"])}</tr>{sen}</table></div>
</main></body></html>'''
    run.p('vib_levels.html').write_text(page)
