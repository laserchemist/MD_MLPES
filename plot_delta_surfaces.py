#!/usr/bin/env python3
"""
plot_delta_surfaces.py — 3D visualization of CASSCF(4,4) delta-ML correction
surfaces from casscf_wB97X_nm_grid.py results (works with partial runs).

Three surfaces plotted:
    δ_S0(mode, amplitude)    — wB97X→CASSCF ground-state correction
    Δgap_S1(mode, amplitude) — S0→S1 adiabatic gap
    Δgap_T1(mode, amplitude) — S0→T1 adiabatic gap (ISC)

X-axis: NM displacement amplitude (signed: −2.0 to +2.0 × a_thermal)
Y-axis: Mode frequency (cm⁻¹) — represents vibrational character
Z-axis: Correction / gap (kcal/mol)

Usage
-----
    python3 plot_delta_surfaces.py
    python3 plot_delta_surfaces.py --results outputs/casscf_wB97X_nm_grid_<ts>
    python3 plot_delta_surfaces.py --no-show   # save only, no interactive window
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_RESULTS = 'outputs/casscf_wB97X_nm_grid_20260407_184904'
EQ_GAP_S1 = 29.0   # kcal/mol at equilibrium (from test_casscf_equilibrium.py)
EQ_GAP_T1 = 26.5

SIGNED_FACTORS = [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]


# ── load and reshape data ─────────────────────────────────────────────────────

def load_grid(results_dir):
    """
    Returns dict keyed by mode_idx, each containing:
        freq_cm1  : float
        amplitude : list of signed factors with data
        delta_s0  : dict {signed_factor: value_kcal}
        gap_s1    : dict {signed_factor: value_kcal}
        gap_t1    : dict {signed_factor: value_kcal}
    """
    with open(Path(results_dir) / 'results.json') as f:
        raw = json.load(f)

    modes = {}
    for rec in raw.values():
        if rec.get('status') != 'ok':
            continue
        k   = rec['mode_idx']
        sf  = rec['sign'] * rec['factor']   # signed factor
        if k not in modes:
            modes[k] = {'freq_cm1': rec['freq_cm1'],
                        'delta_s0': {}, 'gap_s1': {}, 'gap_t1': {}}
        modes[k]['delta_s0'][sf] = rec.get('delta_s0_kcal')
        modes[k]['gap_s1'][sf]   = rec.get('gap_s1_kcal')
        modes[k]['gap_t1'][sf]   = rec.get('gap_t1_kcal')
    return modes


def build_grid_arrays(modes, signed_factors=SIGNED_FACTORS):
    """
    Build 2D arrays (n_modes × n_amplitudes) for each surface.
    Missing values → NaN.  Returns (freqs, sf_arr, Z_delta, Z_s1, Z_t1).
    """
    mode_keys = sorted(modes.keys())
    freqs     = np.array([modes[k]['freq_cm1'] for k in mode_keys])
    sf_arr    = np.array(signed_factors)

    n_m, n_a = len(mode_keys), len(signed_factors)
    Z_d  = np.full((n_m, n_a), np.nan)
    Z_s1 = np.full((n_m, n_a), np.nan)
    Z_t1 = np.full((n_m, n_a), np.nan)

    for i, k in enumerate(mode_keys):
        for j, sf in enumerate(signed_factors):
            Z_d[i, j]  = modes[k]['delta_s0'].get(sf, np.nan)
            Z_s1[i, j] = modes[k]['gap_s1'].get(sf, np.nan)
            Z_t1[i, j] = modes[k]['gap_t1'].get(sf, np.nan)

    return freqs, sf_arr, Z_d, Z_s1, Z_t1


# ── plotting ──────────────────────────────────────────────────────────────────

def surface_plot(ax, freqs, sf_arr, Z, title, zlabel, cmap, vcenter=None):
    """
    3D surface + scatter for one correction surface.
    NaN values are masked out; available points plotted as scatter if surface
    is incomplete.
    """
    # Meshgrid: X = amplitude, Y = frequency
    X, Y = np.meshgrid(sf_arr, freqs)          # (n_modes, n_amp)

    mask = ~np.isnan(Z)
    frac = mask.sum() / Z.size

    if vcenter is not None:
        vmin, vmax = np.nanmin(Z), np.nanmax(Z)
        vmin = min(vmin, vcenter - 0.1)
        vmax = max(vmax, vcenter + 0.1)
        norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    else:
        norm = None

    # Draw surface where we have complete rows
    complete_rows = [i for i in range(len(freqs))
                     if np.sum(~np.isnan(Z[i])) == len(sf_arr)]
    if len(complete_rows) >= 3:
        Xi = X[complete_rows]
        Yi = Y[complete_rows]
        Zi = Z[complete_rows]
        kwargs = dict(cmap=cmap, alpha=0.75, linewidth=0.3, edgecolor='k')
        if norm:
            kwargs['norm'] = norm
        ax.plot_surface(Xi, Yi, Zi, **kwargs)

    # Scatter all available points on top
    xs = X[mask].ravel()
    ys = Y[mask].ravel()
    zs = Z[mask].ravel()
    sc_kwargs = dict(c=zs, cmap=cmap, s=40, depthshade=True, zorder=5)
    if norm:
        sc_kwargs['norm'] = norm
    sc = ax.scatter(xs, ys, zs, **sc_kwargs)

    ax.set_xlabel('Amplitude (×a_thermal)', fontsize=9, labelpad=6)
    ax.set_ylabel('Mode freq (cm⁻¹)',       fontsize=9, labelpad=6)
    ax.set_zlabel(zlabel,                    fontsize=9, labelpad=6)
    ax.set_title(f'{title}\n({mask.sum()}/{Z.size} frames)', fontsize=10)
    ax.tick_params(labelsize=7)

    # Equilibrium reference plane
    if vcenter is not None:
        ax.axhspan = None   # not available in 3D; skip
    return sc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default=DEFAULT_RESULTS)
    parser.add_argument('--out', default=None,
                        help='Output PNG path (default: <results>/surfaces_3d.png)')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    out_path = args.out or str(Path(args.results) / 'surfaces_3d.png')

    print(f'Loading: {args.results}/results.json')
    modes = load_grid(args.results)
    print(f'Modes with data: {sorted(modes.keys())}  ({len(modes)} / 30)')

    freqs, sf_arr, Z_d, Z_s1, Z_t1 = build_grid_arrays(modes)

    n_ok = int(np.sum(~np.isnan(Z_d)))
    print(f'Frames available: {n_ok} / 240')
    print(f'δ_S0   range: [{np.nanmin(Z_d):+.2f}, {np.nanmax(Z_d):+.2f}] kcal/mol')
    print(f'Δgap_S1 range: [{np.nanmin(Z_s1):.2f}, {np.nanmax(Z_s1):.2f}] kcal/mol')
    print(f'Δgap_T1 range: [{np.nanmin(Z_t1):.2f}, {np.nanmax(Z_t1):.2f}] kcal/mol')

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 5.5))
    fig.suptitle(
        'CASSCF(4,4)/6-31G* delta-ML surfaces vs wB97X-D  '
        f'[{n_ok}/240 frames, modes 0–{max(modes.keys())}]',
        fontsize=11, y=1.01)

    # Panel 1 — δ_S0
    ax1 = fig.add_subplot(131, projection='3d')
    sc1 = surface_plot(ax1, freqs, sf_arr, Z_d,
                       title='δ_S0 = ΔE_CASSCF − ΔE_wB97X',
                       zlabel='δ_S0 (kcal/mol)',
                       cmap='RdBu_r', vcenter=0.0)
    ax1.set_zlim(np.nanmin(Z_d) - 0.5, np.nanmax(Z_d) + 0.5)
    fig.colorbar(sc1, ax=ax1, shrink=0.55, pad=0.1,
                 label='kcal/mol', orientation='horizontal')

    # Panel 2 — Δgap_S1
    ax2 = fig.add_subplot(132, projection='3d')
    sc2 = surface_plot(ax2, freqs, sf_arr, Z_s1,
                       title='Δgap_S1  (S0→S1 adiabatic)',
                       zlabel='Gap (kcal/mol)',
                       cmap='plasma', vcenter=EQ_GAP_S1)
    fig.colorbar(sc2, ax=ax2, shrink=0.55, pad=0.1,
                 label='kcal/mol', orientation='horizontal')

    # Panel 3 — Δgap_T1
    ax3 = fig.add_subplot(133, projection='3d')
    sc3 = surface_plot(ax3, freqs, sf_arr, Z_t1,
                       title='Δgap_T1  (S0→T1, ISC)',
                       zlabel='Gap (kcal/mol)',
                       cmap='viridis', vcenter=EQ_GAP_T1)
    fig.colorbar(sc3, ax=ax3, shrink=0.55, pad=0.1,
                 label='kcal/mol', orientation='horizontal')

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f'\nSaved: {out_path}')

    if not args.no_show:
        matplotlib.use('TkAgg') if False else None   # keep Agg for headless
        print('(Use --no-show to suppress this message in headless environments)')


if __name__ == '__main__':
    main()
