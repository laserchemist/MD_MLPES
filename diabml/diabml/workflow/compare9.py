"""Classical dynamics against the variational quantum result on the same 9-D surface and dipole.

    python -m diabml.workflow.ch2oo well9d_compare RUN
        -> results/well9d_compare.json, figures/VCI5_dynamics.png (and a section of vci_9d.html)

Two routes from one potential and one dipole surface to the infrared spectrum of CH2OO:
  dynamics  classical trajectories (ir stage, [ir] surface = "well9d"): the spectrum is the Fourier transform of the
            dipole autocorrelation function, averaged over trajectories; rotation is in the trajectories.
  VCI       quantum vibrational levels and transition moments (well9d_vci), each band given its asymmetric-rotor
            structure (well9d_bands).
Since the surface and the dipole are the same, every difference is the treatment of the nuclear motion: zero-point
energy and quantisation (band positions, resonances), and the classical thermal populations (hot bands, widths).
Reported per band region: the intensity-weighted centre and the share of the integrated intensity, for dynamics,
VCI and the measured spectrum (digitized, Huang et al. 2016), and the correlation of each computed trace with it.
"""
from __future__ import annotations

import json

import numpy as np

from . import ch2oo as W

REGIONS = [('O–O stretch ν6 (+ ν8 wag)', 835.0, 960.0), ('ν5, 2ν9, C–O stretch ν4', 1150.0, 1345.0),
           ('CH₂ scissor ν3', 1385.0, 1520.0)]
MASK = (1085.0, 1135.0)                                  # precursor interference in the measurement


def _centre(nu, y, lo, hi):
    m = (nu >= lo) & (nu <= hi)
    yy = np.clip(y[m], 0, None)
    return float((nu[m] * yy).sum() / max(yy.sum(), 1e-300)), float(np.trapezoid(yy, nu[m]))


def stage_well9d_compare(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .isotopes9 import _load_ref
    bp, sp = run.p('results', 'well9d_bands.npz'), run.p('results', 'ir_9d_spectrum.csv')
    for f, how in ((bp, 'well9d_bands'), (sp, 'ir with [ir] surface = "well9d"')):
        if not f.exists():
            raise SystemExit(f'{f} not found: run {how} first')
    z = np.load(bp)
    g, va, ve = z['grid'], z['absorption'], z['emission']
    hdr = sp.read_text().splitlines()[0].split(',')
    d = np.loadtxt(sp, delimiter=',', skiprows=1)
    md = {h.replace('I_', ''): d[:, j] for j, h in enumerate(hdr) if j}
    mnu = d[:, 0]
    kT = next((k for k in md if k.endswith('K')), list(md)[0])
    kE = next((k for k in md if 'kJ' in k), None)
    run_fs = float(run.cfg['ir'].get('run_fs', 10000.0))
    zb = float(run.cfg['ir'].get('zoom_broaden_cm', 3.0))
    # resolution of a Hann-windowed power spectrum of one trajectory (FWHM 1.44 / T), and of the saved spectrum,
    # which is further smoothed by a Gaussian of standard deviation zoom_broaden_cm
    res_md = float(np.hypot(1.44 * 33356.4 / run_fs, 2.3548 * zb))
    raw = {}
    for k in list(md):                       # recompute from the saved dipoles with no extra smoothing
        f = run.p('results', 'ir_9d_' + k.replace(' ', '').replace('/', 'per') + '.npz')
        if f.exists():
            from ..dynamics.ir import ir_spectrum
            with np.load(f) as zz:
                if 'mu' in zz.files and 't' in zz.files:
                    tt = zz['t']
                    raw[k] = ir_spectrum(zz['mu'], float(tt[1] - tt[0]), broaden_cm=float(run.cfg['well9d'].get(
                        'compare_broaden_cm', 0.0)), max_cm=float(run.cfg['ir'].get('max_cm', 4000.0)),
                        min_cm=float(run.cfg['ir'].get('min_cm', 200.0)))
                    run_fs = float(tt[-1] - tt[0])
    if len(raw) == len(md):
        mnu = raw[kT][0]
        md = {k: v[1] for k, v in raw.items()}
        res_md = float(np.hypot(1.44 * 33356.4 / run_fs, 2.3548 * float(run.cfg['well9d'].get('compare_broaden_cm', 0.0))))
        run.log('dynamics spectra recomputed from the saved dipoles without extra smoothing')
    sg = res_md / 2.3548 / (g[1] - g[0])
    hw = min(int(4 * sg) + 1, len(g) // 2 - 1)
    kk = np.arange(-hw, hw + 1)
    gk = np.exp(-0.5 * (kk / sg) ** 2)
    va_s = np.convolve(va, gk / gk.sum(), mode='same')    # the VCI trace at the dynamics resolution
    ref = _load_ref('CH2OO')
    vci = json.loads(run.p('results', 'well9d_vci.json').read_text())
    ir = json.loads(run.p('results', 'ir_9d.json').read_text()) if run.p('results', 'ir_9d.json').exists() else {}
    traces = {'dynamics': (mnu, md[kT]), 'VCI': (g, va)}
    rp_csv = run.p('results', 'rpmd_9d_spectrum.csv')
    trpmd_P = None
    if rp_csv.exists():                                  # TRPMD: the largest bead number
        h2 = rp_csv.read_text().splitlines()[0].split(',')
        d2 = np.loadtxt(rp_csv, delimiter=',', skiprows=1)
        cols = {int(h.replace('I_P', '')): j for j, h in enumerate(h2) if h.startswith('I_P')}
        trpmd_P = max(cols)
        if trpmd_P > 1:
            traces['TRPMD'] = (d2[:, 0], d2[:, cols[trpmd_P]])
        else:
            trpmd_P = None
    if ref is not None:
        keep = ~((ref[0] >= MASK[0]) & (ref[0] <= MASK[1]))
        traces['expt'] = (ref[0][keep], ref[1][keep])
    out = {'dynamics_label': kT, 'regions': [], 'md_resolution_cm-1': res_md, 'run_fs': run_fs}
    run.log(f'dynamics: {run_fs / 1000:g} ps trajectories, resolution about {res_md:.1f} cm-1 (FWHM; Hann-windowed '
            f'transform, 1.44 / T); the VCI trace is also shown at that resolution')
    tot = {k: sum(_centre(*t, lo, hi)[1] for _, lo, hi in REGIONS) for k, t in traces.items()}
    run.log(f'{"region":28s} ' + ' '.join(f'{k + " centre":>16s} {"share %":>8s}' for k in traces))
    for name, lo, hi in REGIONS:
        row = {'region': name, 'lo': lo, 'hi': hi}
        for k, t in traces.items():
            c, a = _centre(*t, lo, hi)
            row[k] = {'centre': c, 'share': a / tot[k]}
        out['regions'].append(row)
        run.log(f'{name:28s} ' + ' '.join(f'{row[k]["centre"]:16.1f} {100 * row[k]["share"]:8.1f}' for k in traces))
    if ref is not None:
        x = traces['expt'][0]
        m = (x >= 800) & (x <= 1448)
        for k in [k for k in ('dynamics', 'TRPMD', 'VCI') if k in traces]:
            yc = np.interp(x, traces[k][0], traces[k][1])
            out[f'correlation_{k}'] = float(np.corrcoef(yc[m], traces['expt'][1][m])[0, 1])
            mo = (x >= 860) & (x <= 960)
            out[f'correlation_oo_{k}'] = float(np.corrcoef(yc[mo], traces['expt'][1][mo])[0, 1])
        ys = np.interp(x, g, va_s)
        out['correlation_VCI_at_md_resolution'] = float(np.corrcoef(ys[m], traces['expt'][1][m])[0, 1])
        mo = (x >= 860) & (x <= 960)
        out['correlation_oo_VCI_at_md_resolution'] = float(np.corrcoef(ys[mo], traces['expt'][1][mo])[0, 1])
        out['correlation_oo_dynamics_vs_VCI_smoothed'] = float(np.corrcoef(
            np.interp(g[(g >= 860) & (g <= 960)], mnu, md[kT]), va_s[(g >= 860) & (g <= 960)])[0, 1])
        run.log('correlation with the measured spectrum, 800-1448 cm-1: ' + ', '.join(
            f'{k} {out["correlation_" + k]:.3f}' for k in ('dynamics', 'TRPMD', 'VCI') if 'correlation_' + k in out)
            + '; O-O band (860-960): ' + ', '.join(
            f'{k} {out["correlation_oo_" + k]:.3f}' for k in ('dynamics', 'TRPMD', 'VCI') if 'correlation_oo_' + k in out)
            + f'; classical dynamics against VCI at the same resolution (O-O band) '
              f'{out["correlation_oo_dynamics_vs_VCI_smoothed"]:.3f}')
    # band by band: where the classical band sits between the harmonic and the quantum position
    out['bands'] = []
    if trpmd_P:
        out['trpmd_beads'] = trpmd_P
        rj = run.p('results', 'rpmd_9d.json')
        if rj.exists():
            out['trpmd_gap_closed'] = json.loads(rj.read_text()).get('gap_closed', {})
    run.log(f'{"band":6s} {"harmonic":>9s} {"dynamics":>9s} ' + (f'{"TRPMD":>8s} ' if trpmd_P else '') +
            f'{"VCI":>8s} {"expt.":>8s}   shift from harmonic: dynamics / ' + ('TRPMD / ' if trpmd_P else '') + 'VCI / expt.')
    for f in vci['fundamentals']:
        if not (800 <= f['harmonic'] <= 1550) or f.get('km_mol', 0) < 30:     # the three strong, separate bands
            continue
        m = (mnu >= f['harmonic'] - 60) & (mnu <= f['harmonic'] + 15)
        pk = float(mnu[m][np.argmax(md[kT][m])])
        r = {'mode': f['mode'], 'label': f['label'], 'harmonic': f['harmonic'], 'dynamics_max': pk, 'vci': f['vci'],
             'expt': f.get('expt')}
        rj = run.p('results', 'rpmd_9d.json')
        if trpmd_P and rj.exists():
            rr = json.loads(rj.read_text())
            bd = rr['beads'].get(str(trpmd_P), {}).get('bands', {}).get(f['mode'])
            if bd:
                r['trpmd_centre'], r['trpmd_err'] = bd['centre'], bd['err']
                b1 = rr['beads'].get('1', {}).get('bands', {}).get(f['mode'])
                if b1:
                    r['classical_centre'], r['classical_err'] = b1['centre'], b1['err']
                r['vci_centre'] = (rr.get('vci_centre') or {}).get(f['mode'])
            xr, yr = traces['TRPMD']
            mr = (xr >= f['harmonic'] - 80) & (xr <= f['harmonic'] + 15)
            r['trpmd_max'] = float(xr[mr][np.argmax(yr[mr])])
        out['bands'].append(r)
        run.log(f'{f["mode"]:6s} {f["harmonic"]:9.1f} {pk:9.1f} ' + (f'{r["trpmd_max"]:8.1f} ' if trpmd_P else '') +
                f'{f["vci"]:8.1f} {f["expt"] if f.get("expt") else float("nan"):8.1f}   {pk - f["harmonic"]:+.0f} / ' +
                (f'{r["trpmd_max"] - f["harmonic"]:+.0f} / ' if trpmd_P else '') +
                f'{f["vci"] - f["harmonic"]:+.0f} / ' + (f'{f["expt"] - f["harmonic"]:+.0f}' if f.get('expt') else '-'))
    f6 = next(f for f in vci['fundamentals'] if f['mode'] == 'v6')
    out['oo'] = {'harmonic': f6['harmonic'], 'vci_origin': f6['vci'], 'expt_origin': f6.get('expt'),
                 'dynamics_centre': out['regions'][0]['dynamics']['centre'], 'vci_centre': out['regions'][0]['VCI']['centre']}
    if 'expt' in traces:
        out['oo']['expt_centre'] = out['regions'][0]['expt']['centre']
    if kE:
        out['hot'] = {'label': kE, 'dynamics_centre': _centre(mnu, md[kE], 870, 950)[0],
                      'vci_emission_centre': _centre(g, ve, 870, 950)[0],
                      'dynamics_shift': _centre(mnu, md[kE], 870, 950)[0] - _centre(mnu, md[kT], 870, 950)[0],
                      'vci_shift': _centre(g, ve, 870, 950)[0] - _centre(g, va, 870, 950)[0]}
        run.log(f'O-O band, hot ({kE}) minus {kT} (870-950 cm-1 centre): dynamics {out["hot"]["dynamics_shift"]:+.1f} cm-1, '
                f'VCI (emission minus absorption) {out["hot"]["vci_shift"]:+.1f} cm-1')
    if ir.get('temperatures'):
        out['md_mean_energy_eV'] = {k: v.get('mean_total_energy_eV') for k, v in ir['temperatures'].items()}
    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(13, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.55, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    lo, hi = 800, 1520
    col = {'expt': '0.2', 'VCI': '#1f5fa8', 'dynamics': '#c8641e', 'TRPMD': '#2a7f62'}
    lab = {'expt': 'expt. (Huang et al. 2016, digitized)', 'VCI': 'VCI + asymmetric rotor (exact quantum, this surface)',
           'dynamics': f'classical MD, {kT}: FT of the dipole autocorrelation',
           'TRPMD': f'TRPMD, P = {trpmd_P} beads, 300 K: Kubo-transformed dipole autocorrelation'}
    order = [k for k in ('expt', 'VCI', 'TRPMD', 'dynamics') if k in traces]
    nlev = len(order)
    for i, k in enumerate(order):
        x, y = traces[k]
        m = (x >= lo) & (x <= hi)
        yy = y[m] / np.nanmax(y[(x >= 860) & (x <= 960)])
        if k == 'expt':
            yy = np.where(np.r_[False, np.diff(x[m]) > 3], np.nan, yy)
        ax.plot(x[m], yy + 1.25 * (nlev - 1 - i), color=col[k], lw=1.0, label=lab[k])
    lab_y = 1.25 * (nlev - 1) + 1.3
    top = lab_y + 0.3
    for f in vci['fundamentals']:
        if f.get('expt') and lo <= f['expt'] <= hi:
            ax.plot([f['expt']] * 2, [-0.1, lab_y - 0.08], color='0.75', ls=':', lw=0.8)
            ax.text(f['expt'], lab_y, f['mode'], ha='center', fontsize=8, color='0.35')
    ax.set_xlim(hi, lo)
    ax.set_ylim(-0.1, top)
    ax.set_yticks([])
    ax.set_xlabel('wavenumber (cm⁻¹)')
    ax.set_title(f'One surface, one dipole, {nlev - 1 if "expt" in traces else nlev} treatments of the nuclear motion '
                 f'(each trace normalised to its O–O band)',
                 fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.19))
    ax = fig.add_subplot(gs[1, 0])
    a, b = 860, 960
    for k in order:
        x, y = traces[k]
        m = (x >= a) & (x <= b)
        ax.plot(x[m], y[m] / y[m].max(), color=col[k], lw=1.3 if k != 'expt' else 1.0,
                label={'dynamics': 'classical MD', 'TRPMD': f'TRPMD (P = {trpmd_P})'}.get(k, k))
    m = (g >= a) & (g <= b)
    ax.plot(g[m], va_s[m] / va_s[m].max(), color='#1f5fa8', lw=1.2, ls='--', label=f'VCI at {res_md:.0f} cm⁻¹ resolution')
    ax.axvline(f6['harmonic'], color='0.5', ls='--', lw=0.8)
    ax.text(f6['harmonic'] - 0.6, 1.1, 'harmonic', fontsize=7, color='0.4', ha='left')
    ax.set_xlim(b, a)
    ax.set_ylim(-0.03, 1.2)
    ax.set_yticks([])
    ax.set_xlabel('wavenumber (cm⁻¹)')
    ax.set_title(f'O–O band, {kT}', fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    ax = fig.add_subplot(gs[1, 1])
    if kE:
        m = (mnu >= a) & (mnu <= b)
        ax.plot(mnu[m], md[kT][m] / md[kT][m].max(), color='#c8641e', lw=1.0, ls=':', label=f'dynamics, {kT}')
        ax.plot(mnu[m], md[kE][m] / md[kE][m].max(), color='#c8641e', lw=1.3, label=f'dynamics, {kE}')
        m = (g >= a) & (g <= b)
        ax.plot(g[m], va[m] / va[m].max(), color='#1f5fa8', lw=1.0, ls=':', label='VCI absorption, 300 K')
        ax.plot(g[m], ve[m] / ve[m].max(), color='#1f5fa8', lw=1.3, label='VCI emission, 13 kJ/mol above zero point')
        ax.legend(fontsize=7.5, frameon=False, loc='upper right')
    ax.set_xlim(b, a)
    ax.set_ylim(-0.03, 1.45)
    ax.set_yticks([])
    ax.set_xlabel('wavenumber (cm⁻¹)')
    ax.set_title('O–O band with 13 kJ/mol of excess energy', fontsize=11)
    fig.savefig(run.p('figures', 'VCI5_dynamics.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    run.p('results', 'well9d_compare.json').write_text(json.dumps(out, indent=1, default=float))
    run.log('wrote results/well9d_compare.json, figures/VCI5_dynamics.png')
    from .vci_report import stage_vci_report
    stage_vci_report(run)
