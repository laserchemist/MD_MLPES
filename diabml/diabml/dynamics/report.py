"""HTML log of watched trajectories (diabml.dynamics.monitor): one self-contained page with figures and a table.

    from diabml.dynamics.report import write_html
    write_html([rec1, rec2, ...], 'dynamics.html', title='...', data_coords=ref.coords)
"""
from __future__ import annotations

import base64
import html
import io
import time

import numpy as np

from .monitor import summarize

SCOL = ['#2E2A36', '#2F6DB5', '#E06C2B', '#2E9A6B', '#8E44AD', '#7F8C8D']


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def figures(recs, data_coords=None):
    """(name, caption, base64 png) for the standard dynamics diagnostics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = []
    n = len(recs)
    ncol = min(3, n); nrow = int(np.ceil(n / ncol))

    def grid(h=3.0):
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, h * nrow), squeeze=False)
        for ax in list(axes.flat)[n:]:
            ax.axis('off')
        return fig, axes

    def marks(ax, r, y=None):
        if r['t_trigger'] is not None:
            ax.axvline(r['t_trigger'], color='#B03A2E', lw=1.2, ls='--')
        for ev in r['events']:
            ax.axvspan(ev['t'], max(ev['t_end'], ev['t'] + 0.5), color='#E9B872', alpha=0.4, lw=0)

    # D1 energies of every state
    fig, axes = grid(3.2)
    for ax, r in zip(axes.flat, recs):
        for s in range(r['E'].shape[1]):
            ax.plot(r['t'], r['E'][:, s], color=SCOL[s % len(SCOL)], lw=1.2 if s == r['state'] else 0.8,
                    label=f'S{s}' + (' (active)' if s == r['state'] else ''))
        ax.plot(r['t'], r['E_tot'], color='k', lw=0.8, ls=':', label='total')
        marks(ax, r)
        ax.set(title=r['label'], xlabel='t (fs)', ylabel='E (eV)')
    axes.flat[0].legend(fontsize=8, ncol=2)
    out.append(('D1_energies', 'Energies of all states along each trajectory (active state bold, dotted: total '
                'energy). Dashed red line: the surface became uncertain; shaded: nonadiabatic region (small gap or '
                'strong coupling).', _png(fig)))
    # D2 conservation
    fig, axes = grid(2.6)
    for ax, r in zip(axes.flat, recs):
        ax.plot(r['t'], (r['E_tot'] - r['E_tot'][0]) * 1000, color=SCOL[1], lw=1)
        ax.plot(r['t'], (r['E_kin'] - r['E_kin'].mean()) * 1000 / 20, color='0.7', lw=0.6, label='kinetic / 20')
        marks(ax, r)
        ax.set(title=r['label'], xlabel='t (fs)', ylabel='E_tot − E_tot(0) (meV)')
    axes.flat[0].legend(fontsize=8)
    out.append(('D2_energy_conservation', 'Total-energy deviation (meV); the grey trace is the kinetic energy scaled '
                'down 20 times for comparison. Drift or steps here mean a rough surface or too long a time step.',
                _png(fig)))
    # D3 coordinates
    names = recs[0]['names']
    fig, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 3.4), squeeze=False)
    for j, nm in enumerate(names):
        ax = axes[0, j]
        for i, r in enumerate(recs):
            ax.plot(r['t'], r['coords'][:, j], lw=0.9, color=plt.cm.viridis(i / max(1, n - 1)))
        if data_coords is not None:
            lo, hi = np.min(data_coords[:, j]), np.max(data_coords[:, j])
            ax.axhspan(lo, hi, color='0.92', zorder=0, label='range of the data')
        ax.set(xlabel='t (fs)', ylabel=nm)
    axes[0, 0].legend(fontsize=8)
    out.append(('D3_coordinates', 'Internal coordinates of every trajectory (one colour each); grey band: the range '
                'covered by the reference data.', _png(fig)))
    # D4 uncertainty and distance to data
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for i, r in enumerate(recs):
        c = plt.cm.viridis(i / max(1, n - 1))
        axes[0].plot(r['t'], r['spread'][:, r['state']] * 1000, lw=0.9, color=c)
        axes[1].plot(r['t'], r['data_distance'], lw=0.9, color=c)
    axes[0].axhline(recs[0]['triggers']['spread_eV'] * 1000, color='#B03A2E', ls='--', label='stop threshold')
    axes[1].axhline(recs[0]['triggers']['data_distance'], color='#B03A2E', ls='--', label='stop threshold')
    axes[0].set(xlabel='t (fs)', ylabel='member spread, active state (meV)')
    axes[1].set(xlabel='t (fs)', ylabel='distance to nearest data point (grid steps)')
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    out.append(('D4_uncertainty', 'Left: how much the ensemble members disagree on the active state; right: distance '
                'from the nearest reference geometry. Crossing a dashed line stops the trajectory.', _png(fig)))
    # D5 nonadiabatic indicators
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for i, r in enumerate(recs):
        c = plt.cm.viridis(i / max(1, n - 1))
        g = np.fmin(np.nan_to_num(r['gap_up'], nan=np.inf), np.nan_to_num(r['gap_down'], nan=np.inf))
        axes[0].plot(r['t'], np.where(np.isfinite(g), g, np.nan), lw=0.9, color=c)
        axes[1].plot(r['t'], r['massey'], lw=0.9, color=c)
    axes[0].axhline(recs[0]['triggers']['gap_eV'], color='#E06C2B', ls='--', label='nonadiabatic region')
    axes[1].axhline(recs[0]['triggers']['massey'], color='#E06C2B', ls='--', label='nonadiabatic region')
    axes[0].set(xlabel='t (fs)', ylabel='gap to the nearest other state (eV)', yscale='log')
    axes[1].set(xlabel='t (fs)', ylabel='Massey parameter ħ|d·v| / ΔE')
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    out.append(('D5_nonadiabatic', 'Where Born–Oppenheimer dynamics on one state stops being adequate: the energy gap '
                'to the nearest other state and the Massey parameter ħ|d·v|/ΔE (how fast the nuclei move compared with '
                'the electronic time scale). Below / above the dashed lines a surface-hopping method would let the '
                'trajectory change state.', _png(fig)))
    # D6 paths over the data
    fig, axes = plt.subplots(1, max(1, len(names) - 1), figsize=(6 * max(1, len(names) - 1), 4.4), squeeze=False)
    for k in range(1, len(names)):
        ax = axes[0, k - 1]
        if data_coords is not None:
            ax.plot(data_coords[:, 0], data_coords[:, k], '+', color='0.65', ms=4, label='reference data')
        for i, r in enumerate(recs):
            c = plt.cm.viridis(i / max(1, n - 1))
            ax.plot(r['coords'][:, 0], r['coords'][:, k], lw=0.7, color=c)
            if r['t_trigger'] is not None:
                F = np.array(r['flagged'])
                ax.plot(F[:, 0], F[:, k], 'x', color='#B03A2E', ms=9, mew=2)
        ax.set(xlabel=names[0], ylabel=names[k], xlim=(0.9, 3.2) if names[0] == 'r_OO' else None)
    axes[0, 0].legend(fontsize=8)
    out.append(('D6_paths', 'Trajectory paths over the reference geometries (grey +); red ×: geometries flagged for new '
                'reference calculations (where the surface became uncertain, and where it was heading).', _png(fig)))
    return out


def write_html(recs, path, title='Dynamics log', data_coords=None, notes=''):
    rows = [summarize(r) for r in recs]
    figs = figures(recs, data_coords)
    esc = html.escape
    head = ['trajectory', 'time reached (fs)', 'stopped because', 'E_tot drift (meV)', 'E_tot rms (meV)',
            'max spread (meV)', 'min gap (eV)', 'r_OO range (Å)', 'nonadiabatic events', 'flagged geometries']
    body = ''
    for s in rows:
        rr = s['r_OO_range']
        body += ('<tr>' + ''.join(f'<td>{esc(str(c))}</td>' for c in [
            s['trajectory'], f'{s["time_fs"]:.1f}', s['stopped'], f'{s["drift_meV"]:+.2f}', f'{s["rms_meV"]:.2f}',
            f'{s["max_spread_meV"]:.0f}', f'{s["min_gap_eV"]:.3f}', f'{rr[0]:.2f}–{rr[1]:.2f}' if rr else '',
            s['nonadiabatic_events'], s['flagged']]) + '</tr>')
    ev = ''
    for r in recs:
        for e in r['events']:
            ev += (f'<tr><td>{esc(r["label"])}</td><td>{e["t"]:.1f}–{e["t_end"]:.1f}</td><td>{e["gap_eV"]:.3f}</td>'
                   f'<td>{e["massey"]:.3f}</td><td>{esc(str(np.round(e["coords"], 3).tolist()))}</td></tr>')
    figs_html = ''.join(f'<figure><img src="data:image/png;base64,{b}" alt="{esc(c)}"><figcaption><b>{esc(n)}.</b> '
                        f'{esc(c)}</figcaption></figure>' for n, c, b in figs)
    tr = recs[0]['triggers']
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>
:root{{--paper:#F6F4F0;--ink:#2E2A36;--muted:#7A7480;--rule:#DAD3C8;--plum:#3D2C58;--copper:#C8873A;--panel:#EFEBF3}}
@media (prefers-color-scheme:dark){{:root{{--paper:#17131f;--ink:#ece8f2;--muted:#a69cb6;--rule:#3d3250;--panel:#241b33}}}}
body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,-apple-system,sans-serif}}
header{{background:var(--plum);color:#fff;border-bottom:5px solid var(--copper);padding:24px}}
header h1{{margin:0;font:600 28px Georgia,serif}} header p{{margin:6px 0 0;color:#E9B872}}
main{{max-width:1180px;margin:0 auto;padding:16px}} h2{{font:600 22px Georgia,serif;border-bottom:1px solid var(--rule)}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:14px}} th,td{{border-bottom:1px solid var(--rule);padding:5px 8px;text-align:left}}
th{{background:var(--panel)}} img{{max-width:100%;background:#fff;border:1px solid var(--rule)}} figcaption{{color:var(--muted);font-size:14px}}
</style></head><body><header><h1>{esc(title)}</h1><p>{len(recs)} trajectories · {time.strftime('%Y-%m-%d %H:%M')}</p></header><main>
<p>{esc(notes)}</p>
<p>Stop when the members disagree by more than {tr["spread_eV"] * 1000:.0f} meV, the geometry is more than {tr["data_distance"]} grid steps from the data, the total energy jumps {tr["jump_meV"]} meV in one step, or a force exceeds {tr["force_max"]} eV/Å (then {tr["lookahead_fs"]} fs of look-ahead).
Nonadiabatic region: gap below {tr["gap_eV"]} eV or Massey parameter ħ|d·v|/ΔE above {tr["massey"]}.</p>
<h2>Trajectories</h2><div class="tw"><table><thead><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr></thead><tbody>{body}</tbody></table></div>
<h2>Nonadiabatic regions visited</h2>{'<div class="tw"><table><thead><tr><th>trajectory</th><th>t (fs)</th><th>smallest gap (eV)</th><th>largest Massey parameter</th><th>geometry</th></tr></thead><tbody>' + ev + '</tbody></table></div>' if ev else '<p>None.</p>'}
<h2>Figures</h2>{figs_html}</main></body></html>'''
    with open(path, 'w') as fh:
        fh.write(page)
    return rows
