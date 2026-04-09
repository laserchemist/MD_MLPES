#!/usr/bin/env python3
"""
plot_wB97X_surface.py — 3D visualization of wB97X-D/6-31G* PES on the
NM-coordinate grid computed by casscf_wB97X_nm_grid.py.

One primary surface:
    ΔE_wB97X(mode, amplitude) — wB97X energy relative to equilibrium (kcal/mol)

A transparent harmonic reference surface is overlaid for direct comparison.

X-axis: NM displacement amplitude (signed: −2.0 to +2.0 × a_thermal)
Y-axis: Mode frequency (cm⁻¹)
Z-axis: ΔE (kcal/mol)

Usage
-----
    python3 plot_wB97X_surface.py
    python3 plot_wB97X_surface.py --results outputs/casscf_wB97X_nm_grid_<ts>
    python3 plot_wB97X_surface.py --zmax 20    # cap color scale at 20 kcal/mol
    python3 plot_wB97X_surface.py --no-show
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_RESULTS = 'outputs/casscf_wB97X_nm_grid_20260407_184904'
E_WB97X_EQ_HA   = -306.2099845838      # equilibrium reference from test_casscf_equilibrium.py
KCAL_PER_HA     = 627.509474
SIGNED_FACTORS  = [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]


# ── load and reshape ──────────────────────────────────────────────────────────

def load_wb97x_grid(results_dir):
    """
    Returns dict keyed by mode_idx:
        freq_cm1  : float
        dE        : {signed_factor: kcal/mol relative to equilibrium}
    """
    with open(Path(results_dir) / 'results.json') as f:
        raw = json.load(f)

    modes = {}
    for rec in raw.values():
        if rec.get('status') != 'ok':
            continue
        k  = rec['mode_idx']
        sf = rec['sign'] * rec['factor']
        if k not in modes:
            modes[k] = {'freq_cm1': rec['freq_cm1'], 'dE': {}}
        e_wb = rec.get('e_wb97x_ha')
        if e_wb is not None:
            dE = (e_wb - E_WB97X_EQ_HA) * KCAL_PER_HA
            modes[k]['dE'][sf] = dE
    return modes


def build_grid_arrays(modes, signed_factors=SIGNED_FACTORS):
    mode_keys = sorted(modes.keys())
    freqs     = np.array([modes[k]['freq_cm1'] for k in mode_keys])
    sf_arr    = np.array(signed_factors)

    n_m, n_a = len(mode_keys), len(signed_factors)
    Z = np.full((n_m, n_a), np.nan)

    for i, k in enumerate(mode_keys):
        for j, sf in enumerate(signed_factors):
            Z[i, j] = modes[k]['dE'].get(sf, np.nan)

    return freqs, sf_arr, Z


def build_harmonic_surface(freqs, sf_arr):
    """
    Harmonic energy in kcal/mol: E_harm = 0.5 * (2π ν)² × a_thermal² × f²
    In reduced NM coordinates q = f × a_thermal, E_harm = 0.5 × ω² × q².
    In thermal amplitude units (f = q/a_thermal):
        E_harm(f) = 0.5 × k_B × T × f²   with T=300 K (a_thermal² = kT/ω²)
    So E_harm(f) = kT × (f²/2) = 0.5 × kBT [kcal/mol] × f²
    kBT at 300 K = 0.5961 kcal/mol → 0.5 × 0.5961 × f² = 0.2981 kcal/mol × f²
    This is independent of mode frequency (thermal amplitude absorbs ω).
    """
    KB_KCAL = 0.0019872  # kcal/mol per K
    T_REF   = 300.0      # K
    kT_half = 0.5 * KB_KCAL * T_REF   # = 0.29808 kcal/mol

    X, Y = np.meshgrid(sf_arr, freqs)   # (n_modes, n_amp)
    Z_harm = kT_half * X**2             # same for all frequencies
    return X, Y, Z_harm


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_wb97x(ax, freqs, sf_arr, Z, zmax=None):
    X, Y = np.meshgrid(sf_arr, freqs)
    mask = ~np.isnan(Z)

    # Cap color scale for readability; still draw full surface
    vmin, vmax = 0.0, (zmax if zmax else max(np.nanmax(Z), 1.0))

    # Surface for complete rows
    complete_rows = [i for i in range(len(freqs))
                     if np.sum(~np.isnan(Z[i])) == len(sf_arr)]
    if len(complete_rows) >= 3:
        Xi = X[complete_rows]; Yi = Y[complete_rows]; Zi = Z[complete_rows]
        Zi_clamp = np.clip(Zi, vmin, vmax)
        ax.plot_surface(Xi, Yi, Zi, facecolors=plt.cm.hot_r(Zi_clamp / vmax),
                        alpha=0.75, linewidth=0.2, edgecolor='k')

    # Scatter all available points
    xs = X[mask].ravel(); ys = Y[mask].ravel(); zs = Z[mask].ravel()
    zs_clamp = np.clip(zs, vmin, vmax)
    sc = ax.scatter(xs, ys, zs, c=zs_clamp, cmap='hot_r',
                    vmin=vmin, vmax=vmax, s=35, depthshade=True, zorder=5)

    # Harmonic reference surface (transparent wireframe)
    _, _, Z_harm = build_harmonic_surface(freqs, sf_arr)
    ax.plot_wireframe(X, Y, Z_harm, color='steelblue', alpha=0.25,
                      linewidth=0.8, linestyle='--', label='harmonic')

    ax.set_xlabel('Amplitude (×a_thermal)', fontsize=9, labelpad=6)
    ax.set_ylabel('Mode freq (cm⁻¹)',       fontsize=9, labelpad=6)
    ax.set_zlabel('ΔE_wB97X (kcal/mol)',    fontsize=9, labelpad=6)
    ax.tick_params(labelsize=7)

    color_lim = f'(color capped at {vmax:.0f} kcal/mol)' if zmax else ''
    ax.set_title(
        f'wB97X-D/6-31G*  ΔE surface\n'
        f'({mask.sum()}/{Z.size} frames)  {color_lim}',
        fontsize=10)

    return sc, vmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default=DEFAULT_RESULTS)
    parser.add_argument('--out', default=None)
    parser.add_argument('--zmax', type=float, default=20.0,
                        help='Color scale cap in kcal/mol (default 20); '
                             'set 0 for auto (uses full range)')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    zmax = args.zmax if args.zmax > 0 else None
    out_path = args.out or str(Path(args.results) / 'wB97X_surface_3d.png')

    print(f'Loading: {args.results}/results.json')
    modes = load_wb97x_grid(args.results)
    print(f'Modes with data: {sorted(modes.keys())}  ({len(modes)} / 30)')

    freqs, sf_arr, Z = build_grid_arrays(modes)
    n_ok = int(np.sum(~np.isnan(Z)))
    print(f'Frames available : {n_ok} / 240')
    print(f'ΔE_wB97X range   : [{np.nanmin(Z):+.2f}, {np.nanmax(Z):+.2f}] kcal/mol')
    print(f'Color scale cap  : {zmax} kcal/mol')

    # single-panel figure
    fig = plt.figure(figsize=(9, 7))
    fig.suptitle(
        'wB97X-D/6-31G* PES on CASSCF NM grid  '
        f'[{n_ok}/240 frames, modes 0–{max(modes.keys())}]\n'
        'Blue wireframe = harmonic reference (kT/2 × f²)',
        fontsize=11, y=1.01)

    ax = fig.add_subplot(111, projection='3d')
    sc, vmax_used = plot_wb97x(ax, freqs, sf_arr, Z, zmax=zmax)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.60, pad=0.10,
                        label='ΔE_wB97X (kcal/mol)', orientation='vertical')
    if zmax:
        cbar.set_label(f'ΔE_wB97X (kcal/mol, capped at {vmax_used:.0f})')

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f'\nSaved: {out_path}')

    if not args.no_show:
        print('(Use --no-show to suppress this message in headless environments)')


if __name__ == '__main__':
    main()
