"""Diagnostics report of the dynamics on the 9-D surface: classical MD (ir stage) and TRPMD (rpmd9d stage).

    python -m diabml.workflow.ch2oo dynamics_report RUN        # -> RUN/dynamics_9d.html (also written by rpmd9d)

What was run and what it cost (trajectories, time step, lengths, steps, resolution, wall time, time per step), and
what the trajectories did: geometry and dipole along a sample trajectory, energy conservation (classical), and for
TRPMD the centroid geometry, the spread of the ring polymer on each atom (radius of gyration: the quantum width),
centroid temperature and potential energy against time, the distribution of bead geometries against the classical
one, and the dipole autocorrelation functions. Figures: figures/DYN1_classical.png, DYN2_trpmd.png,
DYN3_quantum_widths.png. Sections appear for the stages whose results exist.
"""
from __future__ import annotations

import base64
import datetime as _dt
import html
import json
import re

import numpy as np

from . import ch2oo as W

E = html.escape
NAMES = ['r(O–O)', '∠COO', 'r(C–O)', 'r(C–H syn)', 'r(C–H anti)', '∠HCO syn', '∠HCO anti', 'τ H syn', 'τ H anti']
ATOMS = ['C', 'O (central)', 'O (terminal)', 'H syn', 'H anti']


def _img(run, name, alt):
    f = run.p('figures', name)
    return f'<img src="data:image/png;base64,{base64.b64encode(f.read_bytes()).decode()}" alt="{E(alt)}">' \
        if f.exists() else ''


def _acf(mu, nmax):
    """Normalised dipole autocorrelation (average over trajectories and components) to nmax lags, by FFT."""
    d = mu - mu.mean(1, keepdims=True)
    n = d.shape[1]
    f = np.fft.rfft(d, 2 * n, axis=1)
    c = np.fft.irfft(f * np.conj(f), axis=1)[:, :nmax].sum(2).mean(0)
    c = c / (n - np.arange(nmax))
    return c / c[0]


def _classical(run, plt):
    """Figure and summary of the classical ir runs (results/ir_9d_*.npz)."""
    ic = run.cfg['ir']
    rows, fig_done = [], False
    logf = run.p('logs', 'ir.log')
    wall = None
    if logf.exists():
        m = re.findall(r'(\d+)/(\d+), ([\d.]+) min\)', logf.read_text())
        if m:
            wall = float(m[-1][2])
    files = sorted(run.p('results').glob('ir_9d_*.npz'))
    summ = json.loads(run.p('results', 'ir_9d.json').read_text()) if run.p('results', 'ir_9d.json').exists() else {}
    for f in files:
        with np.load(f) as z:
            if 'mu' not in z.files:
                continue
            t, mu, x, Et = z['t'], z['mu'], z['x'], z['E_tot']
        label = f.stem.replace('ir_9d_', '').replace('per', '/')
        dt_rec = float(t[1] - t[0])
        every = int(ic.get('every', 2))
        dt = dt_rec / every
        drift = (Et[:, -1] - Et[:, 0]) * 1000
        rms = Et.std(1) * 1000
        rows.append({'label': label, 'trajectories': int(mu.shape[0]), 'dt': dt, 'record': dt_rec,
                     'equil': float(ic.get('equil_fs', 0)), 'run': float(t[-1] - t[0]),
                     'steps': int(round((float(ic.get('equil_fs', 0)) + t[-1] - t[0]) / dt)),
                     'resolution': 1.44 * 33356.4 / float(t[-1] - t[0]), 'drift_max': float(np.abs(drift).max()),
                     'drift_rms': float(np.sqrt((drift ** 2).mean())), 'fluct': float(rms.mean()),
                     'E_mean_kJ': float(Et.mean() * 96.485)})
        if not fig_done:
            fig_done = True
            tx = np.linspace(t[0], t[-1], x.shape[1])
            fig, axs = plt.subplots(3, 2, figsize=(13, 9), sharex='col')
            for col, tmax in enumerate((min(1000.0, t[-1]), t[-1])):
                mx, mt = tx <= tmax, t <= tmax
                ax = axs[0, col]
                for j, c in ((0, '#1f5fa8'), (2, '#b5442c'), (3, '#2a7f62'), (4, '#8a6fb0')):
                    ax.plot(tx[mx], x[0, mx, j], lw=0.9, color=c, label=NAMES[j])
                ax.set_ylabel('bond length (Å)')
                ax.margins(y=0.22)
                ax.set_title(f'sample trajectory, {label}: ' + ('first picosecond' if col == 0 else 'whole run'), fontsize=10)
                ax = axs[1, col]
                for j, c in ((1, '#1f5fa8'), (5, '#b5442c'), (6, '#2a7f62')):
                    ax.plot(tx[mx], x[0, mx, j], lw=0.9, color=c, label=NAMES[j])
                a2 = ax.twinx()
                for j, c in ((7, '0.45'), (8, '0.7')):
                    a2.plot(tx[mx], x[0, mx, j], lw=0.7, color=c, label=NAMES[j])
                ax.set_ylabel('angle (°)')
                ax.margins(y=0.22)
                a2.margins(y=0.22)
                a2.set_ylabel('out-of-plane dihedral (°)', color='0.4')
                if col == 0:
                    h1, l1 = ax.get_legend_handles_labels()
                    h2, l2 = a2.get_legend_handles_labels()
                    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, ncol=5, loc='upper left')
                    axs[0, col].legend(fontsize=7, frameon=False, ncol=4, loc='upper left')
                ax = axs[2, col]
                for a, c in enumerate(('#1f5fa8', '#b5442c', '#2a7f62')):
                    ax.plot(t[mt], mu[0, mt, a], lw=0.6, color=c, label=f'μ{"xyz"[a]} (space-fixed)')
                ax.plot(t[mt], np.linalg.norm(mu[0, mt], axis=1), lw=0.9, color='0.2', label='|μ|')
                ax.set_ylabel('dipole (D)')
                ax.margins(y=0.2)
                ax.set_xlabel('time (fs)')
                if col == 0:
                    ax.legend(fontsize=7, frameon=False, ncol=4, loc='lower left')
            fig.tight_layout()
            fig.savefig(run.p('figures', 'DYN1_classical.png'), dpi=140)
            plt.close(fig)
    return rows, wall, summ


def _trpmd(run, plt):
    rj = run.p('results', 'rpmd_9d.json')
    if not rj.exists():
        return None
    r = json.loads(rj.read_text())
    beads = sorted(int(b) for b in r['beads'])
    data = {}
    for P in beads:
        f = run.p('results', f'rpmd_9d_P{P}.npz')
        if f.exists():
            with np.load(f) as z:
                if 'centroid_x' in z.files:
                    data[P] = {k: z[k] for k in z.files}
    if not data:
        return {'json': r, 'beads': beads, 'rg': {}}
    Pm = max(data)
    d = data[Pm]
    td = np.arange(d['centroid_x'].shape[1]) * float(d['diag_dt'])
    tm = np.arange(d['mu'].shape[1]) * float(d['dt_record'])
    fig, axs = plt.subplots(3, 2, figsize=(13, 9))
    ax = axs[0, 0]
    for j, c in ((0, '#1f5fa8'), (2, '#b5442c'), (3, '#2a7f62'), (4, '#8a6fb0')):
        ax.plot(td, d['centroid_x'][0, :, j], lw=0.8, color=c, label=NAMES[j])
    ax.set_ylabel('centroid bond length (Å)')
    ax.margins(y=0.22)
    ax.set_title(f'TRPMD, P = {Pm}: centroid geometry of a sample trajectory', fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=4, loc='upper left')
    ax = axs[1, 0]
    for j, c in ((1, '#1f5fa8'), (5, '#b5442c'), (6, '#2a7f62')):
        ax.plot(td, d['centroid_x'][0, :, j], lw=0.8, color=c, label=NAMES[j])
    a2 = ax.twinx()
    for j, c in ((7, '0.45'), (8, '0.7')):
        a2.plot(td, d['centroid_x'][0, :, j], lw=0.7, color=c, label=NAMES[j])
    ax.set_ylabel('centroid angle (°)')
    ax.margins(y=0.22)
    a2.margins(y=0.22)
    a2.set_ylabel('out-of-plane dihedral (°)', color='0.4')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, ncol=5, loc='upper left')
    ax = axs[2, 0]
    m = tm <= min(1000.0, tm[-1])
    for a, c in enumerate(('#1f5fa8', '#b5442c', '#2a7f62')):
        ax.plot(tm[m], d['mu'][0, m, a], lw=0.7, color=c, label=f'μ{"xyz"[a]} (bead average)')
    ax.plot(tm[m], np.linalg.norm(d['mu'][0, m], axis=1), lw=0.9, color='0.2', label='|μ|')
    ax.set_ylabel('dipole (D)')
    ax.margins(y=0.2)
    ax.set_xlabel('time (fs)')
    ax.set_title('bead-averaged dipole, first picosecond', fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=4, loc='lower left')
    ax = axs[0, 1]
    for a, c in enumerate(('#444444', '#b5442c', '#e08a6e', '#1f5fa8', '#7fb0ef')):
        ax.plot(td, d['rg'][:, :, a].mean(0), lw=0.9, color=c, label=ATOMS[a])
    ax.set_ylabel('radius of gyration (Å)')
    ax.set_title('spread of the ring polymer on each atom (trajectory average)', fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=5, loc='center')
    ax = axs[1, 1]
    for P in sorted(data):
        ax.plot(td, data[P]['T_t'], lw=0.8, label=f'P = {P}')
    ax.axhline(r['temperature_K'], color='0.5', ls='--', lw=0.8)
    ax.set_ylabel('centroid temperature (K)')
    ax.set_title('centroid kinetic temperature (batch average; unthermostatted during the run)', fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=4)
    ax = axs[2, 1]
    for P in sorted(data):
        ax.plot(td, data[P]['V_t'] * 8065.5, lw=0.8, label=f'P = {P}')
    ax.set_ylabel('mean potential energy (cm⁻¹)')
    ax.set_xlabel('time (fs)')
    ax.set_title('potential energy (bead and trajectory average): zero-point energy held through the run', fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(run.p('figures', 'DYN2_trpmd.png'), dpi=140)
    plt.close(fig)
    # distributions and correlation functions
    from .rpmd9 import HIST, HIST_BINS
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    cols = ['#c8641e', '#7a3fa0', '#2a7f62', '#1f5fa8', '#b02a6b']
    lab = {'r_OO': 'r(O–O) − rₑ (Å)', 'r_CO': 'r(C–O) − rₑ (Å)', 'r_CH1': 'r(C–H syn) − rₑ (Å)',
           'a_COO': '∠COO − eq. (°)', 'd_H1': 'out-of-plane dihedral τ H syn (°)'}
    widths = {}
    for h, (name, half) in enumerate(HIST):
        ax = axs.flat[h]
        xb = np.linspace(-half, half, HIST_BINS + 1)
        xc = 0.5 * (xb[1:] + xb[:-1])
        for i, P in enumerate(sorted(data)):
            y = data[P]['hist'][h]
            y = y / max(y.sum(), 1) / (xb[1] - xb[0])
            ax.plot(xc, y, lw=1.2, color=cols[i % len(cols)], label='classical (P = 1)' if P == 1 else f'P = {P}')
            mean = (xc * y).sum() / y.sum()
            widths.setdefault(name, {})[P] = float(np.sqrt(((xc - mean) ** 2 * y).sum() / y.sum()))
        ax.set_xlabel(lab[name])
        ax.set_yticks([])
        if h == 0:
            ax.legend(fontsize=8, frameon=False)
    ax = axs.flat[5]
    for i, P in enumerate(sorted(data)):
        dtr = float(data[P]['dt_record'])
        n = min(int(1500 / dtr), data[P]['mu'].shape[1] // 2)
        ax.plot(np.arange(n) * dtr, _acf(data[P]['mu'].astype(float), n), lw=0.9, color=cols[i % len(cols)],
                label='classical (P = 1)' if P == 1 else f'P = {P}')
    ax.set_xlabel('time lag (fs)')
    ax.set_title('dipole autocorrelation (normalised)', fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle('Distribution of bead geometries (quantum, zero-point widths) against the classical distribution',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(run.p('figures', 'DYN3_quantum_widths.png'), dpi=140)
    plt.close(fig)
    rg = {P: data[P]['rg'].mean((0, 1)).tolist() for P in sorted(data)}
    return {'json': r, 'beads': beads, 'rg': rg, 'widths': widths, 'Pm': Pm}


def _check_split(c):
    """The outside error split at the fit window, and how often the beads go there (newer rpmd_check.json only)."""
    if 'window_meV' not in c:
        return ''
    w, b = c.get('outside_within_window') or {}, c.get('outside_beyond_window') or {}
    t = f'Of the outside points, {w.get("n", 0)} lie within the {c["window_meV"] / 1000:g} eV fit window'
    t += f' (RMS {w["rms_meV"]:.2f} meV)' if w.get('n') else ''
    t += f' and {b.get("n", 0)} beyond it' + (f' (RMS {b["rms_meV"]:.2f} meV)' if b.get('n') else '') + '. '
    if c.get('share_outside') is not None:
        t += (f'Only {100 * c["share_outside"]:.2f}% of all bead geometries are outside the fitted range, and the sample '
              f'above is spread deliberately to the extremes, so it over-represents the far tail')
        t += (f'; weighting by how often the beads go there gives at most about {c["bead_weighted_rms_meV"]:.2f} meV'
              if c.get('bead_weighted_rms_meV') is not None else '')
        t += '. '
    return t


def _check_section(run, r):
    """Where the beads leave the fitted range, and (rpmd_check) whether the surface is right there."""
    P = max(int(b) for b in r['beads'])
    ob = r['beads'][str(P)].get('outside_by_coordinate')
    rows = ''
    if ob:
        rows = ''.join(f'<tr><td>{E(k)}</td><td>{100 * v:.2f}</td></tr>' for k, v in sorted(ob.items(), key=lambda kv: -kv[1])
                       if v > 0.0005)
        rows = (f'<h3>Where the beads leave the fitted range (P = {P})</h3><table><tr><th>coordinate</th>'
                f'<th>bead geometries beyond the fitted range (%)</th></tr>{rows}</table>')
    cj = run.p('results', 'rpmd_check.json')
    if not cj.exists():
        return rows + ('<p class="n">Not yet checked against CCSD(T): run <code>rpmd_check</code> to compute the surface error '
                       'at a sample of these geometries.</p>' if rows else '')
    c = json.loads(cj.read_text())
    o, i = c['outside'], c['inside']
    bc = ''.join(f'<tr><td>{E(k)}</td><td>{s["n"]}</td><td>{s["rms_meV"]:.2f}</td><td>{s["max_meV"]:.2f}</td></tr>'
                 for k, s in c['by_coordinate'].items())
    be = ''.join(f'<tr><td>{x["range_meV"][0]}–{x["range_meV"][1] or "∞"}</td><td>{x["n"]}</td><td>{x["rms_meV"]:.2f}</td>'
                 f'<td>{x["max_meV"]:.2f}</td></tr>' for x in c['by_energy'])
    return rows + f'''<h3>Surface check at bead geometries (stage <code>rpmd_check</code>)</h3>
<p>RHF-CCSD(T)/{E(c["basis"])}{" plus the 9-D basis-set correction" if c["delta"] else ""} at {o["n"]} bead geometries outside the
fitted range (farthest-point selection from the P = {c["beads"]} sample) and {i["n"]} inside as controls; the median
difference at the controls removes the offset between the energy zeros. Error outside: RMS {o["rms_meV"]:.2f} meV, largest
{o["max_meV"]:.2f} meV; inside: RMS {i["rms_meV"]:.2f} meV; the fit's own held-out RMS {c["fit_heldout_rms_meV"] or float("nan"):.2f} meV.
{_check_split(c)}<b>{E(c["verdict"])}.</b></p>
{_img(run, "RPMD2_check.png", "surface error at bead geometries")}
<table><tr><th>coordinate furthest out of range</th><th>points</th><th>RMS error (meV)</th><th>largest (meV)</th></tr>{bc}</table>
<table><tr><th>surface energy above the minimum (meV)</th><th>points</th><th>RMS error (meV)</th><th>largest (meV)</th></tr>{be}</table>
'''


def stage_dynamics_report(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    crow, wall, summ = _classical(run, plt)
    tr = _trpmd(run, plt)
    if not crow and tr is None:
        raise SystemExit('no dynamics results found: run ir (with [ir] surface = "well9d") or rpmd9d first')
    today = _dt.date.today().isoformat()
    sec = ''
    if crow:
        rows = ''.join(
            f'<tr><td>{E(c["label"])}</td><td>{c["trajectories"]}</td><td>{c["dt"]:g}</td><td>{c["equil"]:g}</td>'
            f'<td>{c["run"] / 1000:g}</td><td>{c["steps"]:,}</td><td>{c["record"]:g}</td><td>{c["resolution"]:.1f}</td>'
            f'<td>{c["E_mean_kJ"]:.1f}</td><td>{c["drift_rms"]:.3f}</td><td>{c["drift_max"]:.3f}</td>'
            f'<td>{c["fluct"]:.3f}</td></tr>' for c in crow)
        nfe = sum(c['trajectories'] * c['steps'] for c in crow)
        sec += f'''<h2>1. Classical molecular dynamics (stage <code>ir</code>)</h2>
<p>Newtonian trajectories on the 9-D surface, all atoms free, exact forces; started from Langevin equilibration
(thermal runs) or normal-mode sampling at fixed vibrational energy with thermal rotation; then constant energy
(velocity Verlet), the dipole recorded from the 9-D dipole surface. Spectrum: Fourier transform of the dipole
autocorrelation.</p>
<table><tr><th>run</th><th>trajectories</th><th>time step (fs)</th><th>equilibration (fs)</th><th>recorded (ps)</th>
<th>steps each</th><th>dipole every (fs)</th><th>resolution (cm⁻¹)</th><th>mean total energy (kJ/mol)</th>
<th>energy drift, RMS (meV)</th><th>drift, max. (meV)</th><th>energy fluctuation (meV)</th></tr>{rows}</table>
<p class="n">Resolution: full width at half maximum of the Hann-windowed transform, 1.44 / (recorded time). Energy drift: last minus
first total energy of a trajectory; fluctuation: its standard deviation along the trajectory (mean over trajectories).
{f"Wall time of the stage: {wall:.1f} min" if wall else "Wall time not recorded"} for {nfe:,} force evaluations in all.</p>
{_img(run, "DYN1_classical.png", "classical trajectory: geometry and dipole")}
<p class="cap"><b>Figure D1.</b> One classical trajectory: bond lengths, angles (out-of-plane dihedrals on the right axis) and the
dipole moment in the space-fixed frame, over the first picosecond (left) and the whole run (right). The slow modulation of
the dipole components is the rotation of the molecule; the fast ripple is vibration.</p>
'''
    if tr is not None:
        r = tr['json']
        rows = ''
        for P in tr['beads']:
            b = r['beads'][str(P)]
            rows += (f'<tr><td>{P}{" (classical)" if P == 1 else ""}</td><td>{b.get("trajectories", r["trajectories"])}</td>'
                     f'<td>{r["dt_fs"]:g}</td><td>{r.get("equil_fs", 0):g}</td><td>{r["run_fs"] / 1000:g}</td>'
                     f'<td>{b.get("steps_per_trajectory", 0):,}</td><td>{b.get("force_evaluations", 0):,}</td>'
                     f'<td>{b["minutes"]:.1f}</td><td>{b.get("ms_per_step_per_trajectory", float("nan")):.2f}</td>'
                     f'<td>{b["T_centroid_K"]:.0f}</td><td>{b["V_mean_eV"] * 8065.5:.0f}</td>'
                     f'<td>{100 * b["outside_box"]:.1f}</td><td>{1000 * b["E_min_eV"]:+.0f}</td></tr>')
        bands = list(next(iter(r['beads'].values()))['bands'])
        brow = ''
        for P in tr['beads']:
            b = r['beads'][str(P)]['bands']
            brow += f'<tr><td>P = {P}</td>' + ''.join(
                f'<td>{b[k]["centre"]:.1f}' + (f' ± {b[k]["err"]:.1f}' if b[k]['err'] is not None else '') + '</td>'
                for k in bands) + '</tr>'
        if r.get('vci_centre'):
            brow += '<tr><td>VCI (rotational contour, 300 K)</td>' + ''.join(
                f'<td>{r["vci_centre"][k]:.1f}</td>' for k in bands) + '</tr>'
        lrow = ''
        if r.get('lam_scan'):
            for k, v in sorted(r['lam_scan']['values'].items(), key=lambda kv: float(kv[0])):
                lrow += (f'<tr><td>λ = {k}{" (RPMD)" if float(k) == 0 else ""}</td>' + ''.join(
                    f'<td>{v["bands"][b]["centre"]:.1f} ± {v["bands"][b]["err"]:.1f}</td>' for b in bands) + '</tr>')
        rgrow = ''.join(f'<tr><td>P = {P}</td>' + ''.join(f'<td>{x:.4f}</td>' for x in v) + '</tr>'
                        for P, v in tr['rg'].items())
        wrow = ''
        if tr.get('widths'):
            Ps = sorted(next(iter(tr['widths'].values())))
            wrow = ('<table><tr><th>standard deviation of</th>' + ''.join(f'<th>P = {P}</th>' for P in Ps) + '</tr>' +
                    ''.join(f'<tr><td>{E(n)}</td>' + ''.join(f'<td>{w[P]:.4f}</td>' for P in Ps) + '</tr>'
                            for n, w in tr['widths'].items()) + '</table>')
        sec += f'''<h2>{2 if crow else 1}. Thermostatted ring-polymer molecular dynamics (stage <code>rpmd9d</code>)</h2>
<p>Each atom is a ring of P beads joined by harmonic springs (ring-polymer Hamiltonian at P × T). Protocol at
{r["temperature_K"]:g} K: path-integral MD with a Langevin thermostat on every ring-polymer normal mode (PILE-L; centroid
time constant {r.get("centroid_tau_fs", 100):g} fs) for {r.get("equil_fs", 0):g} fs from the minimum, then TRPMD for
{r["run_fs"] / 1000:g} ps: centroid free, internal modes damped with friction 2λω<sub>k</sub>, λ = {r["lam"]:g}. Integrator:
OBABO with exact free-ring-polymer propagation in normal modes, time step {r["dt_fs"]:g} fs. The bead-averaged dipole is
recorded every {r.get("every", 4) * r["dt_fs"]:g} fs; spectral resolution {r.get("resolution_cm-1", 1.44 * 33356.4 / r["run_fs"]):.1f} cm⁻¹.
{r.get("batch", "–")} trajectories per process (all their beads in one batched surface evaluation), {r.get("jobs", "–")} processes.</p>
<table><tr><th>beads P</th><th>trajectories</th><th>time step (fs)</th><th>equilibration (fs)</th><th>recorded (ps)</th>
<th>steps each</th><th>force evaluations</th><th>wall time (min)</th><th>ms per step per trajectory</th>
<th>centroid T (K)</th><th>⟨V⟩ (cm⁻¹)</th><th>beads outside fitted box (%)</th><th>lowest bead energy (meV)</th></tr>{rows}</table>
<p class="n">⟨V⟩: potential energy above the minimum, averaged over beads, trajectories and time; classically 9 × k<sub>B</sub>T/2 ≈ 940 cm⁻¹
at 300 K, quantum mechanically about half the zero-point energy plus the thermal part. Beads outside the fitted box: share of
bead geometries with any internal coordinate outside the range of the fitted points (the surface is extrapolated there); a
lowest bead energy below zero would mean a bead found a hole of the polynomial.</p>
{_img(run, "DYN2_trpmd.png", "TRPMD diagnostics")}
<p class="cap"><b>Figure D2.</b> TRPMD at the largest bead number. Left: centroid bond lengths and angles of one trajectory and its
bead-averaged dipole. Right: radius of gyration of the ring polymer on each atom (the quantum position uncertainty: largest
for H), centroid temperature and mean potential energy against time for every bead number.</p>
{_img(run, "DYN3_quantum_widths.png", "bead distributions")}
<p class="cap"><b>Figure D3.</b> Distributions of the bead geometries relative to the minimum for each bead number, and the
normalised dipole autocorrelation functions. The quantum distributions are wider than the classical ones (zero-point motion),
most of all for the C–H bond; convergence with P shows as curves that stop changing.</p>
<table><tr><th>radius of gyration (Å)</th>{"".join(f"<th>{a}</th>" for a in ATOMS)}</tr>{rgrow}</table>
{wrow}
<h3>Band centres</h3>
<table><tr><th></th>{"".join(f"<th>{b} (cm⁻¹)</th>" for b in bands)}</tr>{brow}</table>
{f'<h3>Friction scan at P = {r["lam_scan"]["beads"]}</h3><table><tr><th></th>' + "".join(f"<th>{b} (cm⁻¹)</th>" for b in bands) + f'</tr>{lrow}</table><p class="n">λ = 0 is plain RPMD (no damping of the internal modes: spurious resonances possible); larger λ broadens the bands.</p>' if lrow else ''}
<p class="n">Intensity-weighted band centres (870–950, 1250–1340, 1400–1500 cm⁻¹) ± one standard error from a bootstrap over
trajectories. More trajectories shrink the error bar (as 1/√N); a longer run improves the resolution but does not move a
centre; more beads matter only until the centres stop changing.</p>
{_img(run, "RPMD1_spectra.png", "TRPMD spectra")}
{_check_section(run, r)}
'''
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CH2OO dynamics diagnostics</title><style>
:root{{--paper:#F6F4F0;--ink:#2E2A36;--muted:#6f6976;--rule:#DAD3C8;--panel:#EFEBF3}}
@media (prefers-color-scheme:dark){{:root{{--paper:#17131f;--ink:#ece8f2;--muted:#a69cb6;--rule:#3d3250;--panel:#241b33}}}}
body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:20px 16px 40px}} h1{{font:600 28px/1.25 Georgia,serif;margin:.4em 0 .2em}}
h2{{font:600 22px Georgia,serif;margin-top:34px;border-bottom:1px solid var(--rule);padding-bottom:4px}} h3{{font:600 17px Georgia,serif;margin:22px 0 6px}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0;display:block;overflow-x:auto}} th,td{{border-bottom:1px solid var(--rule);padding:5px 8px;text-align:left;white-space:nowrap}}
th{{background:var(--panel)}} img{{max-width:100%;background:#fff;border:1px solid var(--rule);margin:8px 0}} p.n,.cap{{color:var(--muted);font-size:14px}}
footer{{margin-top:40px;font-size:13px;color:var(--muted);border-top:1px solid var(--rule);padding-top:8px}} code{{font-size:13px}}
</style></head><body><main>
<h1>Dynamics on the nine-dimensional CH₂OO surface: what was run and what the trajectories did</h1>
<p class="n">Run <code>{E(run.path.name)}</code> · report written {today} · J. M. Smith, Temple University · the spectra and their
comparison with VCI and experiment are in <code>vci_9d.html</code> and <code>ir_9d.html</code></p>
{sec}
<footer>J. M. Smith · Temple University · generated by <code>diabml.workflow.dynamics_report</code> from run {E(run.path.name)}</footer>
</main></body></html>'''
    out = run.p('dynamics_9d.html')
    out.write_text(page)
    run.log(f'wrote {out}')
    return out
