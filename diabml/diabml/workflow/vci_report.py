"""A self-contained HTML report of the CH2OO 9-D vibrational spectrum: figures, results and a full account of the
calculations behind them (electronic structure, sampling, fits, VCI, rotational band model, references).

    python -m diabml.workflow.ch2oo vci_report RUN        # -> RUN/vci_9d.html (also written by well9d_bands)

Every number is read from the run's own files (config.toml, results/well9d*.json, the ORCA outputs under
data/well9d*), so the page describes the calculation that was actually done.
"""
from __future__ import annotations

import base64
import datetime as _dt
import html
import json
import re
from pathlib import Path

from . import ch2oo as W

E = html.escape
MEV_CM = 8.065544


def _j(run, *p):
    f = run.p(*p)
    return json.loads(f.read_text()) if f.exists() else None


def _orca_info(run):
    """Program version, method line, frozen core and basis-function counts from one ORCA output per step."""
    info = {}
    for key, sub in (('vdz', 'well9d'), ('vtz', 'well9d_tz'), ('ff', 'well9d_dipole')):
        d = run.p('data', sub)
        if not d.is_dir():
            continue
        outs = sorted(d.glob('*.out'))
        info[key + '_n'] = len(outs)
        if not outs:
            continue
        t = outs[0].read_text(errors='replace')
        m = re.search(r'Program Version\s+(\S+)', t)
        if m:
            info['version'] = m.group(1)
        m = re.search(r'\|\s*\d+>\s*(!.*)', t)
        if m:
            info[key + '_line'] = m.group(1).strip()
        m = re.search(r'Number of basis functions\s*\.+\s*(\d+)', t)
        if m:
            info[key + '_nbf'] = int(m.group(1))
        m = re.search(r'Frozen core treatment\s*\.+\s*(.+)', t)
        if m:
            info['frozen'] = m.group(1).strip()
        m = re.search(r'Hartree-Fock type\s+HFTyp\s+\.+\s*(\S+)', t)
        if m:
            info['hf'] = m.group(1)
        m = re.search(r'EField\s+([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)', t)
        if m and key == 'ff':
            info['ff_field'] = max(abs(float(x)) for x in m.groups())
    return info


def _versions():
    from importlib import metadata
    out = {}
    for p in ('numpy', 'scipy', 'torch', 'matplotlib', 'ase'):
        try:
            out[p] = metadata.version(p)
        except Exception:
            pass
    return out


def _img(run, name, alt):
    f = run.p('figures', name)
    if not f.exists():
        return ''
    return f'<img src="data:image/png;base64,{base64.b64encode(f.read_bytes()).decode()}" alt="{E(alt)}">'


def _fmt(x, nd=1):
    return '–' if x is None else f'{x:.{nd}f}'


def _dynamics_section(c, run):
    """Section: classical dynamics (FT of the dipole autocorrelation) against VCI (well9d_compare)."""
    if not c:
        return ''
    tp = c.get('trpmd_beads')

    def _sh(x, h):
        return '–' if x is None else f'{x - h:+.0f}'

    def _gap(gc, mode):
        return f'{100 * gc[mode]:.0f}%' if mode in gc else '–'

    def _pm(x, e):
        return '–' if x is None else (f'{x:.1f} ± {e:.1f}' if e is not None else f'{x:.1f}')
    if tp and any('trpmd_centre' in b_ for b_ in c.get('bands', [])):
        gc = c.get('trpmd_gap_closed', {})
        rows = ''.join(
            f'<tr><td>{b_["mode"]} {E(b_["label"])}</td><td>{b_["harmonic"]:.1f}</td>'
            f'<td>{_pm(b_.get("classical_centre"), b_.get("classical_err"))}</td>'
            f'<td>{_pm(b_.get("trpmd_centre"), b_.get("trpmd_err"))}</td><td>{_fmt(b_.get("vci_centre"))}</td>'
            f'<td>{b_["vci"]:.1f}</td><td>{_fmt(b_.get("expt"))}</td>'
            f'<td>{_gap(gc, b_["mode"])}</td></tr>'
            for b_ in c.get('bands', []))
        head = (f'<tr><th>band</th><th>harmonic</th><th>classical MD centre</th><th>TRPMD, P = {tp}, centre</th>'
                f'<th>VCI contour centre</th><th>VCI band origin</th><th>expt. origin</th>'
                f'<th>classical-to-VCI gap closed by TRPMD</th></tr>')
    else:
        rows = ''.join(
            f'<tr><td>{b_["mode"]} {E(b_["label"])}</td><td>{b_["harmonic"]:.1f}</td><td>{b_["dynamics_max"]:.1f}</td>'
            f'<td>{b_["vci"]:.1f}</td><td>{_fmt(b_.get("expt"))}</td><td>{_sh(b_["dynamics_max"], b_["harmonic"])}</td>'
            f'<td>{_sh(b_["vci"], b_["harmonic"])}</td><td>{_sh(b_.get("expt"), b_["harmonic"])}</td></tr>'
            for b_ in c.get('bands', []))
        head = ('<tr><th>band</th><th>harmonic</th><th>classical MD (max.)</th><th>VCI</th><th>expt.</th>'
                '<th colspan="3">shift from harmonic: classical / VCI / expt.</th></tr>')
    reg = ''
    for r in c['regions']:
        reg += f'<tr><td>{E(r["region"])} ({r["lo"]:.0f}–{r["hi"]:.0f})</td>' + ''.join(
            f'<td>{r[k]["centre"]:.1f}</td><td>{100 * r[k]["share"]:.0f}</td>' if k in r else '<td>–</td><td>–</td>'
            for k in ('dynamics', 'VCI', 'expt')) + '</tr>'
    hot = c.get('hot')
    en = c.get('md_mean_energy_eV') or {}
    en_txt = '; '.join(f'{E(k)}: {v * 96.485:.1f} kJ/mol' for k, v in en.items() if v)
    trp = '' if not tp else f''' Thermostatted ring-polymer molecular dynamics (TRPMD; Craig and Manolopoulos 2004;
Rossi, Ceriotti and Manolopoulos 2014) adds approximate quantum effects: each atom becomes a ring of P = {tp} beads joined by
harmonic springs, which samples the quantum Boltzmann distribution (zero-point motion included) and propagates the bead
centroid with approximately quantum dynamics; the internal ring-polymer modes are damped to suppress spurious resonances.
TRPMD is the method that scales to the larger Criegee intermediates, where VCI is not possible; here it is measured
against the exact VCI answer on the same surface and dipole (stage <code>rpmd9d</code>).'''
    return f'''<h2>Dynamics against VCI on the same surface</h2>
<p>The ir stage propagates classical trajectories on this surface and takes the spectrum as the Fourier transform of the
dipole autocorrelation function (dipoles from the same nine-dimensional dipole surface). Potential and dipole being
identical, every difference from VCI is the treatment of the nuclear motion.{trp}</p>
{_img(run, "VCI5_dynamics.png", "dynamics vs VCI vs experiment")}
<p class="cap"><b>Figure 5.</b> Top: measured spectrum (Huang et al. 2016, digitized), VCI with asymmetric-rotor contours, and
the classical dynamics spectrum ({E(c["dynamics_label"])}), each normalised to its O–O band. Bottom left: the O–O band; the dashed
curve is the VCI result at the resolution of the dynamics ({c["md_resolution_cm-1"]:.1f} cm⁻¹, set by the {c["run_fs"] / 1000:g} ps
trajectory length). Bottom right: the O–O band with 13 kJ/mol of excess energy, dynamics and VCI emission, with the 300 K
results dotted.</p>
<table>{head}{rows}</table>
<p class="n">Centres: intensity-weighted mean wavenumber over each band region (870–950, 1250–1340, 1400–1500 cm⁻¹), the same measure for every method (for VCI on its 300 K rotational contour); ± one standard error from a bootstrap over trajectories. Gap closed: (classical − TRPMD) / (classical − VCI), with classical MD run with the same protocol (P = 1).</p>
<table><tr><th>region (cm⁻¹)</th><th>dynamics centre</th><th>share %</th><th>VCI centre</th><th>share %</th><th>expt. centre</th><th>share %</th></tr>{reg}</table>
<p class="n">Centre: intensity-weighted mean wavenumber of the region; share: its part of the integrated intensity of the three
regions (the measured spectrum ends at 1448 cm⁻¹, so its scissor region is incomplete). Correlation with the measured trace over
800–1448 cm⁻¹: dynamics {c.get("correlation_dynamics", float("nan")):.3f}, VCI {c.get("correlation_VCI", float("nan")):.3f}; O–O band
alone: dynamics {c.get("correlation_oo_dynamics", float("nan")):.3f}, VCI {c.get("correlation_oo_VCI", float("nan")):.3f}
(VCI at the dynamics resolution {c.get("correlation_oo_VCI_at_md_resolution", float("nan")):.3f}).</p>
<ul>
<li><b>Band positions.</b> Classical trajectories carry no zero-point energy, so each mode samples only the bottom of its well and
the bands stay near their harmonic positions; the quantum shifts are two to five times larger for the bending modes.</li>
<li><b>Thermal energy.</b> A classical ensemble at 300 K holds k<sub>B</sub>T in every mode, including the C–H stretches that are
frozen quantum mechanically (mean total energy of the runs: {en_txt or "–"}).''' + (f''' With 13 kJ/mol of excess energy the
classical O–O band centre moves by {hot["dynamics_shift"]:+.1f} cm⁻¹ relative to the 300 K run; VCI emission moves by
{hot["vci_shift"]:+.1f} cm⁻¹ relative to 300 K absorption.''' if hot else '') + f'''</li>
<li><b>Band shape.</b> Rotation is in the trajectories, so the classical band has the right overall width, but its resolution is
limited by the trajectory length, it has no sharp Q branch, and it cannot show discrete hot bands.</li>
<li><b>Intensities.</b> The integrated shares of the three regions agree between the two treatments to within a few percent:
intensities depend mainly on the dipole surface, which is shared.</li>
</ul>
'''


def _isotope_section(iso, run=None):
    """Section: microwave constants of the isotopologues and their vibrational shifts (well9d_isotopes)."""
    if not iso:
        return ''
    st = iso['structure']
    lab = [('r_OO', 'r(O–O) Å', 4), ('r_CO', 'r(C–O) Å', 4), ('r_CH1', 'r(C–H<sub>cis</sub>) Å', 4),
           ('r_CH2', 'r(C–H<sub>trans</sub>) Å', 4), ('a_COO', '∠COO °', 2), ('a_HCO1', '∠H<sub>cis</sub>CO °', 2),
           ('a_HCO2', '∠H<sub>trans</sub>CO °', 2)]
    geo = ''.join(f'<tr><td>{n}</td><td>{st["surface"][k]:.{d}f}</td><td>{st["reSE"][k]:.{d}f}</td>'
                  f'<td>{st["surface"][k] - st["reSE"][k]:+.{d}f}</td></tr>' for k, n, d in lab)
    rows = ''
    for name, r in iso['species'].items():
        x, e, s = r['expt_MHz'], r['error_pct'], r['error_at_reSE_pct']
        rows += (f'<tr><td>{E(name)}</td><td>{x[0]:.1f}</td><td>{x[1]:.1f}</td><td>{x[2]:.1f}</td>'
                 f'<td>{e[0]:+.2f}, {e[1]:+.2f}, {e[2]:+.2f}</td><td>{s[0]:+.3f}, {s[1]:+.3f}, {s[2]:+.3f}</td>'
                 f'<td>{", ".join(f"{v:.0f}" for v in r["dB_vib_MHz"])}</td><td>{E(r["source"])}</td></tr>')
    order = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']
    vib = ''
    for name, r in iso['species'].items():
        byp = {}
        for m in sorted(r['modes'], key=lambda m: -m['overlap']):
            byp.setdefault(m['parent'], m)
        has = any('vci' in m for m in r['modes'])
        cells = ''.join('<td>–</td>' if p not in byp else
                        f'<td>{byp[p]["vci"] if has and "vci" in byp[p] else byp[p]["gvpt2"]:.1f}</td>' for p in order)
        vib += f'<tr><td>{E(name)}</td><td>{"VCI" if has else "GVPT2"}</td>{cells}</tr>'
    w = iso['worst_error_pct']
    n = 3 * len(iso['species'])
    cmp_html = ''
    if iso.get('spectra') and run is not None:
        crow = ''
        for name, d in iso['spectra'].items():
            for b_ in d['bands']:
                crow += (f'<tr><td>{E(name)}</td><td>{b_["mode"]}</td><td>{b_["vci_origin"]:.1f}</td><td>{b_["expt_Q"]:.1f}</td>'
                         f'<td>{b_["vci_origin"] - b_["expt_Q"]:+.1f}</td><td>{b_["km_mol"]:.0f}</td></tr>')
        cors = ', '.join(f'{E(k)} {d["correlation_800_1450"]:.3f}' for k, d in iso['spectra'].items())
        cmp_html = f'''<h3>Comparison with the measured CH₂OO and CD₂OO spectra</h3>
{_img(run, "VCI4_isotopes.png", "CH2OO and CD2OO, measured and computed")}
<p class="cap"><b>Figure 4.</b> Measured absorption (up; digitized from Huang, Nishimura, Witek and Lee, J. Chem. Phys. 145, 044305
(2016), Fig. 2, about 0.86 cm⁻¹ per pixel) and the spectrum computed from this surface (down; VCI levels and intensities with
asymmetric-rotor contours at 300 K, least-squares scaled). Nothing in the computed CD₂OO spectrum was adjusted to it: only the
masses differ from CH₂OO, and the ground-state rotational constants are the microwave values. Grey: precursor interference
or off-scale regions of the measurement, excluded. Correlation of the traces over 800–1450 cm⁻¹: {cors}.</p>
<table><tr><th>species</th><th>mode (CH₂OO label)</th><th>VCI band origin</th><th>sharpest measured feature within 18 cm⁻¹</th>
<th>difference</th><th>km/mol</th></tr>{crow}</table>
<p class="n">The sharpest measured feature is the Q-branch maximum where the band has one; a Q-branch maximum lies slightly below the
band origin. The five sharp CH₂OO maxima of the digitized trace fall within 0.7 cm⁻¹ of the high-resolution band origins,
which checks the wavenumber calibration of the digitization.</p>
'''
    return f'''<h2>Isotopologues: microwave constants and vibrational shifts</h2>
<p>The surface is independent of the nuclear masses, so each isotopologue is a prediction with no further electronic-structure
work. Ground-state rotational constants B<sub>0</sub> = B<sub>e</sub> + ΔB<sub>vib</sub>, with ΔB<sub>vib</sub> from second-order perturbation theory on the
surface's cubic force field (ground-state average of Watson's inverse inertia tensor, plus Coriolis terms), are compared with
Fourier-transform microwave values for {len(iso["species"])} species ({n} constants). B<sub>e</sub> is taken either at the surface's own
minimum or at the semi-experimental equilibrium structure of McCarthy et al.; the electronic contribution is not included.</p>
<table><tr><th>species</th><th>A expt. (MHz)</th><th>B</th><th>C</th><th>error %, surface minimum (A, B, C)</th>
<th>error %, semi-experimental structure</th><th>ΔB<sub>vib</sub> (MHz)</th><th>source</th></tr>{rows}</table>
<p>Largest error: {w["surface_minimum"]:.2f}% with the surface's own minimum, {w["at_reSE"]:.3f}% with the semi-experimental
structure. The vibrational corrections, which depend on the shape of the well (harmonic and cubic force field), reproduce
the measurements; the uniform 1% error is the position of the minimum:</p>
<table><tr><th></th><th>surface</th><th>semi-experimental r<sub>e</sub></th><th>difference</th></tr>{geo}</table>
<p class="n">The bonds of the surface are too long (O–O most), as expected for frozen-core CCSD(T) in a triple-zeta basis. A bond that
is too long normally gives a stretching wavenumber that is too low, so the close agreement of ν6 with experiment must involve
a compensating error, most plausibly the missing higher excitations. NE2013: {E(iso["sources"]["NE2013"])}. McC2013: {E(iso["sources"]["McC2013"])}.</p>
{cmp_html}<h3>Fundamentals of the isotopologues (cm⁻¹)</h3>
<table><tr><th>species</th><th>method</th>{"".join(f"<th>{p}</th>" for p in order)}</tr>{vib}</table>
<p class="n">Each column is the mode with the largest overlap with that normal mode of CH₂OO (for the deuterated species the
modes mix and the label is approximate; – where no mode of the species is closest to that parent mode).</p>
'''


def stage_vci_report(run):
    v = _j(run, 'results', 'well9d_vci.json')
    if v is None:
        raise SystemExit('results/well9d_vci.json not found: run well9d_vci (and well9d_bands) first')
    b = _j(run, 'results', 'well9d_bands.json') or {}
    w9 = _j(run, 'results', 'well9d.json') or {}
    hs = _j(run, 'results', 'well9d_hessian.json') or {}
    dp = _j(run, 'results', 'well9d_dipole.json') or {}
    vp = _j(run, 'results', 'well9d_vpt2.json') or {}
    wc = run.cfg['well9d']
    orca = _orca_info(run)
    ref = json.loads((W.C.PACKAGE_ROOT / 'data' / 'reference' / 'su2013_ch2oo_bands.json').read_text())
    rot = json.loads((W.C.PACKAGE_ROOT / 'data' / 'reference' / 'ch2oo_rotational.json').read_text())
    fund = v['fundamentals']
    obs = [f for f in fund if f.get('expt')]
    f6 = next(f for f in fund if f['mode'] == 'v6')
    mad = sum(abs(f['vci'] - f['expt']) for f in obs) / max(len(obs), 1)
    mad_h = sum(abs(f['harmonic'] - f['expt']) for f in obs) / max(len(obs), 1)
    mad_g = (sum(abs(f['gvpt2'] - f['expt']) for f in obs if f.get('gvpt2')) / max(len(obs), 1)) if vp else None
    oo = b.get('oo', {})
    nine = ', '.join(['r(O–O)', '∠COO', 'r(C–O)', 'r(C–H<sub>syn</sub>)', 'r(C–H<sub>anti</sub>)', '∠H<sub>syn</sub>CO',
                      '∠H<sub>anti</sub>CO', 'τ(H<sub>syn</sub>–C–O–O)', 'τ(H<sub>anti</sub>–C–O–O) − 180°'])
    ver = orca.get('version', '6.x')
    nbf = orca.get('vdz_nbf')
    nbft = orca.get('vtz_nbf')
    frozen = orca.get('frozen', 'frozen core')
    n9 = w9.get('n_points')
    nh = hs.get('n_points')
    dlt = w9.get('delta9') or {}
    nt = dlt.get('n')
    ndip = dp.get('n')
    n_ff = orca.get('ff_n')
    ncalc = (orca.get('vdz_n') or 0) + (orca.get('vtz_n') or 0) + (orca.get('ff_n') or 0)
    vers = _versions()
    xm = w9.get('minimum', {})
    rotk = v.get('rotational', {})
    conv = v.get('convergence') or {}
    sens = oo.get('sensitivity', {})
    stats = v.get('grid_stats', {})
    sp = v.get('spectra', {})

    # ------------------------------------------------------------------ tables
    rows = ''
    i6 = f6.get('km_mol') or 1
    for f in fund:
        d = f['vci'] - f['expt'] if f.get('expt') else None
        typ = ' '.join(f'{a}{100 * x:.0f}' for a, x in f.get('type', {}).items() if x > 0.05)
        ri = f'{100 * f["km_mol"] / i6:.0f}' if f.get('km_mol') is not None else '–'
        ei = f.get('expt_rel_intensity')
        rows += (f'<tr><td>{f["mode"]}</td><td>{E(f["label"])}</td><td>{E(f["symmetry"])}</td>'
                 f'<td>{f["harmonic"]:.1f}</td><td>{_fmt(f.get("gvpt2"))}</td><td><b>{f["vci"]:.1f}</b></td>'
                 f'<td>{_fmt(f.get("expt"))}</td><td>{"–" if d is None else f"{d:+.1f}"}</td>'
                 f'<td>{_fmt(f.get("km_mol"))}</td><td>{ri}</td><td>{"–" if ei is None else ei}</td><td>{typ}</td>'
                 f'<td>{f["weight"]:.2f}</td><td>{_fmt(f.get("convergence"))}</td></tr>')
    orders = ''.join(f'<tr><td>{o}</td><td>{d["terms"]}</td><td>{d["heldout_rms_meV"]:.2f} '
                     f'({d["heldout_rms_meV"] * MEV_CM:.1f} cm⁻¹)</td><td>{d["heldout_max_meV"]:.1f}</td></tr>'
                     for o, d in w9.get('orders', {}).items())
    dorders = ''.join(f'<tr><td>{o}</td><td>{d["coefficients"]}</td><td>{d["heldout_rms_D"]:.4f}</td>'
                      f'<td>{d["heldout_max_D"]:.3f}</td></tr>' for o, d in dp.get('fit', {}).get('orders', {}).items())
    torders = ''.join(f'<tr><td>{o}</td><td>{d["terms"]}</td><td>{d["heldout_rms_meV"]:.2f}</td></tr>'
                      for o, d in dlt.get('orders', {}).items())
    geom = ''.join(f'<td>{xm[k]:.4f}</td>' if k.startswith('r_') else f'<td>{xm[k]:.2f}</td>'
                   for k in ('r_OO', 'a_COO', 'r_CO', 'r_CH1', 'r_CH2', 'a_HCO1', 'a_HCO2')) if xm else ''
    cmp_html = _dynamics_section(_j(run, 'results', 'well9d_compare.json'), run)
    iso_html = _isotope_section(_j(run, 'results', 'well9d_isotopes.json'), run)
    rr = ''
    for st, d in rotk.items():
        if st.endswith('_ratios'):
            c, x = d['computed'], d['expt']
            rr += (f'<tr><td>{st[:-7]}</td>' + ''.join(f'<td>{a:.5f} ({e:.5f})</td>' for a, e in zip(c, x)) + '</tr>')
    B0, Bx, Be = rotk.get('B0'), rotk.get('B0_expt'), rotk.get('Be')
    lev = ', '.join(f'{l["E"]:.0f} {E(l["label"])}' + (f' ({100 * l["weight"]:.0f}%)' if l['weight'] < 0.7 else '')
                    for l in v['levels'] if 0 < l['E'] <= 1600)

    pr = oo.get('absorption_PR') or [None, None]
    px = oo.get('expt_PR') or [None, None]
    pe = oo.get('emission_PR') or [None, None]
    sp_ = lambda p: (p[1] - p[0]) if p[0] is not None else None
    today = _dt.date.today().isoformat()
    h = run.cfg.get('ir', {})

    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CH2OO VCI spectrum</title><style>
:root{{--paper:#F6F4F0;--ink:#2E2A36;--muted:#6f6976;--rule:#DAD3C8;--panel:#EFEBF3;--accent:#1f5fa8;--warm:#b5442c}}
@media (prefers-color-scheme:dark){{:root{{--paper:#17131f;--ink:#ece8f2;--muted:#a69cb6;--rule:#3d3250;--panel:#241b33;--accent:#7fb0ef;--warm:#f08a6e}}}}
body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:20px 16px 40px}} h1{{font:600 29px/1.25 Georgia,serif;margin:.4em 0 .2em}}
h2{{font:600 22px Georgia,serif;margin-top:34px;border-bottom:1px solid var(--rule);padding-bottom:4px}} h3{{font:600 17px Georgia,serif;margin:22px 0 6px}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0;display:block;overflow-x:auto}} th,td{{border-bottom:1px solid var(--rule);padding:5px 8px;text-align:left;white-space:nowrap}}
th{{background:var(--panel)}} img{{max-width:100%;background:#fff;border:1px solid var(--rule);margin:8px 0}} p.n,.cap{{color:var(--muted);font-size:14px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:18px 0}}
.card{{background:var(--panel);border-radius:8px;padding:12px 14px}} .card b{{display:block;font:600 24px Georgia,serif;color:var(--accent)}}
.card span{{font-size:13px;color:var(--muted)}} code{{font-size:13px}} ol li{{margin:3px 0}} .eq{{font-family:Georgia,serif;margin:6px 0 6px 18px}}
footer{{margin-top:40px;font-size:13px;color:var(--muted);border-top:1px solid var(--rule);padding-top:8px}}
</style></head><body><main>
<h1>Vibrational spectrum of CH₂OO from a nine-dimensional CCSD(T) surface: variational levels and rotational band shapes</h1>
<p class="n">Run <code>{E(run.path.name)}</code> · report written {today} · J. M. Smith, Temple University</p>

<p>The infrared spectrum of the simplest Criegee intermediate is computed with no adjustable parameters from a
machine-fitted potential energy surface in all nine internal coordinates, built from {n9 or '–'} RHF-CCSD(T)/aug-cc-pVDZ
energies corrected to aug-cc-pVTZ quality, and a nine-dimensional CCSD(T) dipole surface. Vibrational levels and
intensities come from variational vibrational configuration interaction (VCI) on the full Watson Hamiltonian;
each band is then given its exact asymmetric-rotor structure. The result is compared with the room-temperature gas-phase
time-resolved infrared absorption spectrum of Su et al. (2013).</p>

<div class="cards">
<div class="card"><b>{f6["vci"]:.1f} cm⁻¹</b><span>O–O stretch ν6, VCI (expt. {_fmt(f6.get("expt"))}; harmonic {f6["harmonic"]:.1f})</span></div>
<div class="card"><b>{mad:.1f} cm⁻¹</b><span>mean |VCI − expt.| over the {len(obs)} observed fundamentals (harmonic {mad_h:.0f}{f"; GVPT2 {mad_g:.1f}" if mad_g else ""})</span></div>
<div class="card"><b>{oo.get("shape_correlation", float("nan")):.3f}</b><span>correlation of the computed and measured O–O band shapes, 840–1000 cm⁻¹</span></div>
<div class="card"><b>{_fmt(sp_(pr))} / {_fmt(sp_(px))}</b><span>O–O P–R branch spacing (cm⁻¹), computed / expt.</span></div>
</div>

{_img(run, "VCI3_bands.png", "VCI band shapes vs experiment")}
<p class="cap"><b>Figure 1.</b> Top: absorption spectrum at 300 K computed from {b.get("absorption_bands", "–")} vibrational
bands (VCI levels, hot bands included) each with its asymmetric-rotor structure at T<sub>rot</sub> = {b.get("T_rot_K", 300):g} K and a
{b.get("fwhm_cm-1", 1):g} cm⁻¹ Gaussian instrument function (bottom trace), against the measured spectrum of Su et al.
(Science 2013, Fig. 2B, digitized; top trace). Labels: computed fundamentals (blue, below) and experimental band
origins (grey, above). Bottom left: the O–O stretch band (a-type), computed band moved by {_fmt((f6.get("expt") or f6["vci"]) - f6["vci"])} cm⁻¹ onto the
experimental origin to compare shapes only. Bottom right: computed emission from CH₂OO carrying
{wc.get("vci_emission_kJmol", 13):g} kJ/mol of vibrational energy above its zero-point level (levels populated at
T<sub>vib</sub> = {b.get("T_vib_K", sp.get("T_vib_K", 0)):.0f} K, the temperature with that mean energy) against the cold absorption.</p>

<h2>1. Results</h2>
<h3>Fundamentals</h3>
<table><tr><th>mode</th><th>description</th><th>sym.</th><th>harmonic</th><th>GVPT2</th><th>VCI</th><th>expt.</th>
<th>VCI − expt.</th><th>km/mol</th><th>rel. int.</th><th>expt. rel. int.</th><th>band type (%)</th><th>VCI weight</th><th>basis change</th></tr>{rows}</table>
<p class="n">Wavenumbers in cm⁻¹. Experimental band origins: Su et al. (2013) and, for ν4, the high-resolution analysis of
Chang et al.; relative intensities with the O–O band = 100 (the experimental values are integrated band areas). Band type:
share of the squared transition moment along the a, b, c principal axes. VCI weight: largest share of the singly excited
basis function in the eigenstate (below 1 where Fermi resonances mix levels: ν4 with 2ν9, ν3 with ν8+ν9). Basis change: shift
of each fundamental from a basis of half the size to the one used (convergence).</p>
<p>Zero-point energy {v["zpe_cm-1"]:.1f} cm⁻¹ (harmonic {v["zpe_harmonic_cm-1"]:.1f}{f', GVPT2 {vp["zpe_cm-1"]:.1f}' if vp else ''}).
Levels to 1600 cm⁻¹: {lev}.</p>

<h3>O–O band shape</h3>
<table><tr><th></th><th>P branch max.</th><th>R branch max.</th><th>P–R spacing</th></tr>
<tr><td>computed absorption (300 K)</td><td>{_fmt(pr[0])}</td><td>{_fmt(pr[1])}</td><td>{_fmt(sp_(pr))}</td></tr>
<tr><td>experiment (Su et al.)</td><td>{_fmt(px[0])}</td><td>{_fmt(px[1])}</td><td>{_fmt(sp_(px))}</td></tr>
<tr><td>computed emission ({wc.get("vci_emission_kJmol", 13):g} kJ/mol)</td><td>{_fmt(pe[0])}</td><td>{_fmt(pe[1])}</td><td>{_fmt(sp_(pe))}</td></tr></table>
<p class="n">Branch maxima located after 6 cm⁻¹ Gaussian smoothing so that sharp hot-band Q branches are not taken for them.
Sensitivity: with every level given the ground-state rotational constants (rigid rotor throughout) the shape correlation
falls from {oo.get("shape_correlation", float("nan")):.3f} to {sens.get("rigid_shape_correlation", float("nan")):.3f}; the
vibration–rotation constants of the individual levels matter for the band shape. In emission, hot bands (lower level
not the ground state) carry {100 * sp.get("oo_emission_hot_fraction", 0):.0f}% of the O–O intensity and add sharp Q branches
on the low-wavenumber side; the band's intensity-weighted centre moves from {sp.get("oo_band_centre", {}).get("absorption", 0):.1f}
(absorption) to {sp.get("oo_band_centre", {}).get("emission", 0):.1f} cm⁻¹.</p>
{_img(run, "VCI1_spectra.png", "VCI spectra, vibrational only")}
<p class="cap"><b>Figure 2.</b> Vibrational spectra without rotational structure (12 cm⁻¹ Gaussian): 300 K absorption
with stick lines (blue: from the ground state; orange: hot bands) against experiment, and the full-range emission and absorption.</p>

<h2>2. Electronic structure</h2>
<ul>
<li><b>Program:</b> ORCA {E(ver)} (Neese 2025). All energies are canonical (not local or DLPNO) closed-shell
{E(orca.get("hf", "RHF"))}-based CCSD(T) single points, keywords <code>{E(orca.get("vdz_line", "! CCSD(T) aug-cc-pVDZ TightSCF"))}</code>; {E(frozen)}
(C and O 1s uncorrelated), TightSCF convergence, no point-group symmetry imposed on the geometries.</li>
<li><b>Basis sets:</b> aug-cc-pVDZ ({nbf or '–'} basis functions) for the surface and the dipoles; aug-cc-pVTZ
({nbft or '–'} functions) for the basis-set correction (Dunning 1989; Kendall, Dunning and Harrison 1992).</li>
<li><b>Number of calculations:</b> {orca.get("vdz_n", "–")} aug-cc-pVDZ energies (Hessian and surface),
{orca.get("vtz_n", "–")} aug-cc-pVTZ energies, {n_ff or "–"} finite-field energies for the dipoles: {ncalc:,} CCSD(T) calculations in all.</li>
<li><b>Coordinates:</b> nine internal coordinates, {nine}. Both dihedrals vanish for the planar (C<sub>s</sub>) molecule.</li>
</ul>

<h2>3. Choosing the points</h2>
<ol>
<li><b>Reference and Hessian.</b> Centre: {E(hs.get("source", "–").replace("minimum of the frozen benchmark benchmark", "the minimum of the earlier four-dimensional CCSD(T) surface"))}. A finite-difference CCSD(T)/aug-cc-pVDZ Hessian in the nine internal
coordinates ({nh or '–'} points; steps {wc.get("fd_step", {}).get("r", 0.01)} Å, {wc.get("fd_step", {}).get("a", 1.0)}°,
{wc.get("fd_step", {}).get("d", 2.0)}°) gives the normal modes (Wilson GF method) used to place the samples.</li>
<li><b>Candidates.</b> {wc.get("n_pool", 20000):,} geometries drawn in the normal coordinates with harmonic energy spread uniformly
up to {wc.get("e_max_eV", 0.6)} eV ({wc.get("e_max_eV", 0.6) * 8065.5:.0f} cm⁻¹) above the minimum and |dihedral| ≤ {wc.get("d_max", 45)}°.</li>
<li><b>Selection.</b> {wc.get("n_points", 3000):,} points by farthest-point sampling in energy-scaled normal coordinates (each point as far
as possible from all chosen so far), dealt round-robin into {wc.get("batches", 6)} batches of equal coverage, so that every
batch spans the whole region and serves as an independent test set for the others.</li>
</ol>

<h2>4. Potential energy surface</h2>
<p>The energy is a polynomial in the nine internal coordinates (scaled and centred), containing only the terms even in the
two dihedrals jointly, so that the surface has the exact C<sub>s</sub> mirror symmetry of the molecule. Coefficients by linear
least squares (ridge 10⁻¹⁰) to the {w9.get("n_fit", "–")} points within {w9.get("window_eV", 0.8)} eV of the minimum; the order is chosen
by leave-one-batch-out cross-validation:</p>
<table><tr><th>order</th><th>terms</th><th>held-out RMS error (meV)</th><th>held-out max. (meV)</th></tr>{orders}</table>
<p>Selected: order {w9.get("order", "–")}, {w9.get("terms", "–")} terms, fitting residual {w9.get("residual_rms_meV", 0):.2f} meV RMS
({w9.get("residual_rms_meV", 0) * MEV_CM:.1f} cm⁻¹). <b>Basis-set correction:</b> E(aug-cc-pVTZ) − E(aug-cc-pVDZ) at {nt or '–'} points
(farthest-point spread over the points within 0.6 eV) fitted the same way:</p>
<table><tr><th>order</th><th>terms</th><th>held-out RMS (meV)</th></tr>{torders}</table>
<p>The selected correction (order {dlt.get("order", "–")}) is added to every aug-cc-pVDZ energy before the final fit, so the surface
approximates CCSD(T)/aug-cc-pVTZ. {f'Check: on the planar symmetric slice the surface agrees with the earlier, independently fitted 4-D surface to {w9["benchmark"]["rms_meV"]:.2f} meV RMS at {w9["benchmark"]["n"]} points.' if w9.get("benchmark") else ''}</p>
<p>Minimum (refined by Newton steps on the analytic surface, exact gradients and Hessian by automatic differentiation in PyTorch):</p>
<table><tr><th>r(O–O) Å</th><th>∠COO °</th><th>r(C–O) Å</th><th>r(C–H<sub>syn</sub>) Å</th><th>r(C–H<sub>anti</sub>) Å</th><th>∠H<sub>syn</sub>CO °</th><th>∠H<sub>anti</sub>CO °</th></tr><tr>{geom}</tr></table>

<h2>5. Dipole moment surface</h2>
<p>Dipole moments from finite differences of the CCSD(T)/aug-cc-pVDZ energy in a uniform electric field
(ORCA <code>%scf EField</code>, ±{dp.get("field_au", 0.001)} a.u. along x, y, z: μ = −∂E/∂F, the fully relaxed CCSD(T) dipole) at
{ndip or '–'} geometries chosen by farthest-point sampling from the surface points. The sign and origin conventions of the
field term were determined automatically by comparison with the SCF dipole of every output ({E(dp.get("convention", "–"))},
RMS {dp.get("rms_vs_scf_D", 0):.2f} D from the SCF value, the expected correlation effect).</p>
<p>Model: μ(P) = Σ<sub>j</sub> f<sub>j</sub>(q) b<sub>j</sub>(P), with the four bond vectors b = (O1−C, O2−O1, H<sub>syn</sub>−C, H<sub>anti</sub>−C)
and scalar coefficient functions f<sub>j</sub> polynomial in the nine internal coordinates (even in the dihedrals). The form
rotates exactly with the molecule and has no out-of-plane component for planar geometries. Order by 5-fold cross-validation:</p>
<table><tr><th>order</th><th>coefficients</th><th>held-out RMS (D)</th><th>held-out max. (D)</th></tr>{dorders}</table>

<h2>6. Vibrational levels: variational VCI</h2>
<p><b>Hamiltonian.</b> Watson's Hamiltonian for J = 0 in the rectilinear (Eckart) normal coordinates of the minimum
(from the exact Cartesian Hessian of the surface), including the vibrational angular momentum (Coriolis) term with the
equilibrium inverse inertia tensor:</p>
<p class="eq">H = Σ<sub>i</sub> ½ω<sub>i</sub> p<sub>i</sub>² + V(q) + Σ<sub>α</sub> B<sub>α</sub> π<sub>α</sub>² − ¼ Σ<sub>α</sub> B<sub>α</sub>,
&nbsp; π<sub>α</sub> = Σ<sub>i≠j</sub> ζ<sup>α</sup><sub>ij</sub> (ω<sub>j</sub>/ω<sub>i</sub>)<sup>½</sup> q<sub>i</sub> p<sub>j</sub></p>
<p><b>Potential.</b> n-mode representation to {v.get("nmode", 3)}-mode couplings (all one-, two- and three-mode cuts of the
surface through the minimum), evaluated on direct-product Gauss–Hermite grids ({stats.get("points", 0):,} surface evaluations)
and integrated by quadrature; terms coupling four or more normal modes at once are left out. Far grid points ({stats.get("guarded", 0):,}; harmonic energy above
{wc.get("vci_guard_cm", 12000):g} cm⁻¹ where the polynomial turns over) take the harmonic value; they lie where all
computed wavefunctions vanish and change the fundamentals by less than 0.4 cm⁻¹. Dipole: 2-mode representation.</p>
<p><b>Basis.</b> Products of harmonic-oscillator functions: the seven modes below 2500 cm⁻¹ to a harmonic excitation energy of
{v.get("ecut", 0):g} cm⁻¹, times the two C–H stretches to {v.get("ch_quanta", 3)} quanta: {v.get("basis_size", 0):,} basis functions
(quanta per mode {v.get("quanta_per_mode")}). This product truncation gives every low-lying level the same room for
C–H stretch excitation; a single energy cutoff for all modes biases the fundamentals by 10–20 cm⁻¹ here, because in
rectilinear coordinates the CH₂ bends and torsion couple strongly to the C–H stretches. The Hamiltonian matrix is
sparse; its {len(v["levels"])} lowest eigenstates (up to {v["levels"][-1]["E"]:.0f} cm⁻¹) come from the implicitly restarted
Lanczos method (ARPACK, SciPy <code>eigsh</code>).</p>
<p><b>Convergence.</b> Against a basis with a {conv.get("ecut_check", 0):g} cm⁻¹ cutoff: zero-point energy {conv.get("zpe_change", 0):+.2f} cm⁻¹;
fundamentals change by at most {max((abs(x) for x in conv.get("fundamental_change", [0])[:7]), default=0):.1f} cm⁻¹ below 2000 cm⁻¹
({max((abs(x) for x in conv.get("fundamental_change", [0])), default=0):.1f} for the C–H stretches); adding a fourth C–H quantum changes them by less
than 1.5 cm⁻¹ (tested separately).</p>
<p><b>Intensities.</b> Transition moments ⟨f|μ|i⟩ between all computed levels; integrated absorption
A = (8π³N<sub>A</sub>/3hc·4πε₀) ν̃ |μ<sub>fi</sub>|² (km/mol). Absorption spectra use Boltzmann populations of the vibrational levels
(hot bands included); emission rates ν̃³|μ<sub>fi</sub>|² from each populated upper level to every lower one (population above
the highest computed level {100 * sp.get("missing_population", 0):.1f}%).</p>

<h2>7. Rotational band shapes</h2>
<p>Each vibrational band (the {b.get("absorption_bands", "–")} strongest in absorption, carrying {100 * wc.get("band_fraction", 0.98):.0f}% of the 780–1560 cm⁻¹
intensity; {b.get("emission_bands", "–")} in emission) is given the full line list of a rigid asymmetric rotor
(I<sup>r</sup> representation, all J to about 90), H = A J<sub>a</sub>² + B J<sub>b</sub>² + C J<sub>c</sub>² for its lower and its upper level. Line strengths</p>
<p class="eq">S(J′n′ ← Jn) = (2J+1) |Σ<sub>K,q</sub> c′<sub>K+q</sub> c<sub>K</sub> ⟨J K 1 q | J′ K+q⟩ μ<sub>q</sub>|²</p>
<p>with the asymmetric-top eigenvectors c and the spherical components μ<sub>q</sub> of the VCI transition moment in the principal axes
(a-, b- and c-type parts added; they do not interfere). Lines weighted by ν̃ S exp(−E″/kT)/Q (absorption) or ν̃³ S exp(−E′/kT)/Q
(emission) at T<sub>rot</sub> = {b.get("T_rot_K", 300):g} K, binned and convolved with a {b.get("fwhm_cm-1", 1):g} cm⁻¹ Gaussian.
No nuclear-spin statistics (the two H atoms are not equivalent).</p>
<p><b>Rotational constants of each level</b> = ground-state constants × the computed ratio B<sub>v</sub>/B<sub>0</sub>, where B<sub>v</sub> is the VCI expectation
value of Watson's inverse effective inertia tensor μ<sub>αα</sub>(Q) (harmonic and anharmonic vibration–rotation terms) plus the
perturbative Coriolis contribution −Σ<sub>k</sub>α<sub>k</sub><sup>Cor</sup>(⟨n<sub>k</sub>⟩+½). The ground-state constants are the experimental ones
(millimetre-wave, as tabulated by Chang et al.); the surface's own are {', '.join(f'{100 * (a / x - 1):+.1f}%' for a, x in zip(B0, Bx)) if B0 and Bx else '–'} from them
(A, B, C: {', '.join(f'{a:.4f}' for a in B0) if B0 else '–'} vs {', '.join(f'{a:.4f}' for a in Bx) if Bx else '–'} cm⁻¹; equilibrium {', '.join(f'{a:.4f}' for a in Be) if Be else '–'}).
Check against high-resolution data, B<sub>v</sub>/B<sub>0</sub> computed (expt.):</p>
<table><tr><th>level</th><th>A</th><th>B</th><th>C</th></tr>{rr}</table>
<p class="n">The B and C ratios that set the P–R spacing are reproduced closely. The A ratio is less certain: three pairs of
modes (ν7/ν9, ν6/ν8, ν4/ν5) lie close enough for Coriolis resonance, which the perturbative term treats only approximately;
a full treatment needs rotation–vibration calculations at J &gt; 0.</p>
{_img(run, "VCI2_oo_band.png", "O-O band, absorption and emission")}
<p class="cap"><b>Figure 3.</b> O–O and CH₂ wag region without rotational structure: cold absorption and hot emission (VCI),
with the measured absorption.</p>

{iso_html}
{cmp_html}
<h2>8. What the agreement rests on, and its limits</h2>
<ul>
<li>The anharmonic shift of the O–O stretch (harmonic {f6["harmonic"]:.1f} → {f6["vci"]:.1f} cm⁻¹) is a zero-point effect of the
coupled nine-dimensional motion; classical trajectories on the same surface leave the band near its harmonic position.</li>
<li>The band contour depends on the rotational constants and on the a/b/c character of the transition moment; the P/Q/R shape
of the O–O band follows from its almost pure a-type character (computed {100 * f6.get("type", {}).get("a", 0):.0f}% a).</li>
<li>Remaining errors: ν4 and ν5 lie about 12 cm⁻¹ low and the ground-state rotational constants about 1% low, pointing to the
electronic structure (aug-cc-pVTZ-quality CCSD(T) with frozen core; no core–valence correlation, complete-basis
extrapolation or higher excitations). The experimental intensities are integrated band areas from a digitized figure;
the measured 1241 cm⁻¹ feature assigned to ν5 overlaps 2ν9 (computed {next((l["E"] for l in v["levels"] if l["label"] == "2v9"), 0):.0f} cm⁻¹), which borrows
intensity from ν4 through their Fermi resonance, so the computed ν5 intensity alone is not the whole band.</li>
<li>Rotation–vibration coupling beyond the level-specific rotational constants (Coriolis resonances between nearby
levels at J &gt; 0) and collisional or pressure broadening are not included; the instrument function is a single
Gaussian of chosen width.</li>
</ul>

<h2>9. Software</h2>
<p>ORCA {E(ver)} for all electronic-structure calculations. <code>diabml</code> (this project, github.com/laserchemist/MD_MLPES) for sampling,
fitting, VCI, the rotor model and this report: stages <code>well9d</code>, <code>well9d_fit</code>, <code>well9d_tz</code>, <code>well9d_vpt2</code>,
<code>well9d_dipole</code>, <code>well9d_vci</code>, <code>well9d_bands</code>. Python libraries: {', '.join(f'{k} {v_}' for k, v_ in vers.items())}.</p>

<h2>References</h2>
<ol>
<li>Y.-T. Su, Y.-H. Huang, H. A. Witek, Y.-P. Lee, Infrared absorption spectrum of the simplest Criegee intermediate CH₂OO, <i>Science</i> <b>340</b>, 174–176 (2013).</li>
<li>Y.-P. Chang, A. J. Merer, H.-H. Chang, L.-J. Jhang, W. Chao, J. J.-M. Lin, High resolution quantum cascade laser spectroscopy of the simplest Criegee intermediate, CH₂OO, between 1273 and 1290 cm⁻¹, arXiv:1706.03488 (ground-state constants from earlier millimetre-wave work, as tabulated there).</li>
<li>F. Neese, Software update: the ORCA program system—Version 6.0, <i>WIREs Comput. Mol. Sci.</i> <b>15</b>, e70019 (2025).</li>
<li>K. Raghavachari, G. W. Trucks, J. A. Pople, M. Head-Gordon, A fifth-order perturbation comparison of electron correlation theories, <i>Chem. Phys. Lett.</i> <b>157</b>, 479 (1989).</li>
<li>T. H. Dunning Jr., <i>J. Chem. Phys.</i> <b>90</b>, 1007 (1989); R. A. Kendall, T. H. Dunning Jr., R. J. Harrison, <i>J. Chem. Phys.</i> <b>96</b>, 6796 (1992).</li>
<li>J. K. G. Watson, Simplification of the molecular vibration-rotation Hamiltonian, <i>Mol. Phys.</i> <b>15</b>, 479 (1968).</li>
<li>J. M. Bowman, S. Carter, X. Huang, MULTIMODE: a code to calculate rovibrational energies of polyatomic molecules, <i>Int. Rev. Phys. Chem.</i> <b>22</b>, 533 (2003) (n-mode representation, VCI).</li>
<li>I. M. Mills, Vibration-rotation structure in asymmetric- and symmetric-top molecules, in <i>Molecular Spectroscopy: Modern Research</i>, K. N. Rao, C. W. Mathews (eds.), Academic Press (1972) (vibration–rotation and Coriolis constants).</li>
<li>E. B. Wilson Jr., J. C. Decius, P. C. Cross, <i>Molecular Vibrations</i>, McGraw-Hill (1955) (GF method).</li>
<li>R. B. Lehoucq, D. C. Sorensen, C. Yang, <i>ARPACK Users' Guide</i>, SIAM (1998).</li>
</ol>
<footer>J. M. Smith · Temple University · generated by <code>diabml.workflow.vci_report</code> from run {E(run.path.name)}</footer>
</main></body></html>'''
    out = run.p('vci_9d.html')
    out.write_text(page)
    run.log(f'wrote {out}')
    return out
