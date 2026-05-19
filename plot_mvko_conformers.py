#!/usr/bin/env python3
"""
2×2 figure of the four MVKO conformers (syn/anti × trans/cis),
modelled after Figure 3 of Barber et al. 2018 (Phys. Chem. Chem. Phys.).

syn  = O2 (distal oxygen) on the same side as CH3 relative to the C1-O1 axis
anti = O2 on the same side as the vinyl (C=C) group

Geometries are loaded from the best available source in priority order:
  1. PSI4 wB97X-D/6-31G* optimised eq_coords.npy from the NM-PES build output
  2. Approximate starting geometry from modules/test_molecules.py

Status label per panel:
  (no marker) = PSI4 wB97X-D/6-31G* optimised
  *           = approximate starting geometry (optimisation in progress or pending)

Each panel shows two sets of relative energies:
  wB97X-D/6-31G* (this work) — loaded from PSI4 eq_energy in state.json files
  CCSD(T)/aug-cc-pVTZ//B3LYP (Barber et al. 2018)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from pathlib import Path

KCAL = 627.509474


# ---------------------------------------------------------------------------
# Atom appearance
# ---------------------------------------------------------------------------
ATOM_COLORS = {'C': '#555555', 'O': '#CC3333', 'H': '#EEEEEE'}
ATOM_RADII  = {'C': 0.40,      'O': 0.35,      'H': 0.22}
ATOM_ZS     = {'C': 10,        'O': 10,         'H': 5}
COVALENT_R  = {'C': 0.77,      'O': 0.73,       'H': 0.31}
BOND_TOL    = 0.30
DOUBLE_BOND_PAIRS = {(3, 4)}   # C2=C3 (vinyl double bond, 0-based indices)


# ---------------------------------------------------------------------------
# Geometry loading — try optimised output first, fall back to test_molecules
# ---------------------------------------------------------------------------
def _symbols_from_tm(key):
    from modules.test_molecules import get_molecule
    return list(get_molecule(key).symbols)


def _load_best(opt_paths, tm_key):
    """
    Returns (coords, symbols, status_str).
    Tries each path in opt_paths for an eq_coords.npy file; falls back to
    test_molecules[tm_key] if none found.
    """
    syms = _symbols_from_tm(tm_key)
    for p in opt_paths:
        p = Path(p)
        if p.exists():
            return np.load(str(p)), syms, 'wB97X-D opt'
    from modules.test_molecules import get_molecule
    mol = get_molecule(tm_key)
    return mol.coordinates.copy(), syms, 'starting geometry*'


def load_our_energies():
    """
    Load wB97X-D/6-31G* equilibrium energies from PSI4 state.json files.
    Returns dict {name: energy_ha} or empty dict if files not found.
    """
    sources = {
        'syn-trans':  'outputs/syn_trans_nm_pes_20260513/state.json',
        'syn-cis':    'outputs/syn_cis_nm_pes_20260513v2/state.json',
        'anti-trans': 'outputs/anti_trans_nm_pes_20260421/state.json',
        'anti-cis':   'outputs/anti_cis_nm_pes_20260513/state.json',
    }
    raw = {}
    for name, path in sources.items():
        p = Path(path)
        if p.exists():
            s = json.loads(p.read_text())
            if 'eq_energy' in s:
                raw[name] = s['eq_energy']

    if not raw:
        return {}

    e0 = raw.get('syn-trans', min(raw.values()))
    return {name: (e - e0) * KCAL for name, e in raw.items()}


def load_casscf_gaps():
    """
    Load CASSCF S1/T1 gaps at equilibrium for each conformer.
    Returns dict {name: {'gap_s1': float, 'gap_t1': float, 'active_space': str}}
    or empty dict for conformers without data.
    """
    sources = {
        'syn-trans': ('outputs/syn_trans_casscf22_eq_ref.json', '(2,2)'),
        'anti-cis':  ('outputs/casscf_wB97X_nm_grid_20260407_184904/eq_reference.json', '(4,4)'),
    }
    out = {}
    for name, (path, cas) in sources.items():
        p = Path(path)
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        e_s0 = r.get('e_casscf_s0_eq_ha') or r.get('singlet', {}).get('e_s0')
        gap_s1 = r.get('gap_s1_eq_kcal')
        gap_t1 = r.get('gap_t1_eq_kcal')
        if gap_s1 is None:
            s = r.get('singlet', {}); t = r.get('triplet', {})
            gap_s1 = (s['e_s1'] - e_s0) * KCAL if s.get('e_s1') else None
            gap_t1 = (t['e_t1'] - e_s0) * KCAL if t.get('e_t1') else None
        if e_s0 is not None:
            out[name] = {'gap_s1': gap_s1, 'gap_t1': gap_t1, 'active_space': cas}
    return out


def load_conformer_geometries():
    """Return dict of {name: (coords, symbols, status)} for the four conformers."""

    st_coords, st_syms, st_status = _load_best(
        ['outputs/syn_trans_nm_pes_20260513/eq_coords.npy'],
        'mvko_syn_trans',
    )

    sc_coords, sc_syms, sc_status = _load_best(
        ['outputs/syn_cis_nm_pes_20260513v2/eq_coords.npy'],
        'mvko_syn_cis',
    )

    at_coords, at_syms, at_status = _load_best(
        ['outputs/anti_trans_nm_pes_20260421/eq_coords.npy'],
        'mvko_anti_trans',
    )

    ac_coords, ac_syms, ac_status = _load_best(
        [
            'outputs/anti_cis_nm_pes_20260513/eq_coords.npy',   # dedicated build
            'outputs/wB97X_nm_model_v5/eq_coords.npy',          # from pkl below
        ],
        'mvko_anti_cis',
    )
    # Special case: anti-cis eq coords live inside the pkl
    if ac_status == 'starting geometry*':
        pkl = Path('outputs/wB97X_nm_model_v5/mlpes_wB97X_nm.pkl')
        if pkl.exists():
            with open(pkl, 'rb') as f:
                m = pickle.load(f)
            ac_coords = np.array(m['eq_coords_ang'])
            ac_syms   = list(m['symbols'])
            ac_status = 'wB97X-D opt'

    return {
        'syn-trans':  (st_coords, st_syms, st_status),
        'syn-cis':    (sc_coords, sc_syms, sc_status),
        'anti-trans': (at_coords, at_syms, at_status),
        'anti-cis':   (ac_coords, ac_syms, ac_status),
    }


# ---------------------------------------------------------------------------
# Bond detection
# ---------------------------------------------------------------------------
def _get_bonds(coords, symbols):
    bonds = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            r_cut = COVALENT_R[symbols[i]] + COVALENT_R[symbols[j]] + BOND_TOL
            if np.linalg.norm(coords[i] - coords[j]) < r_cut:
                bonds.append((i, j))
    return bonds


# ---------------------------------------------------------------------------
# Draw one conformer panel
# ---------------------------------------------------------------------------
def _draw_conformer(ax, coords, symbols, label, barber_kcal, status,
                    our_kcal=None, casscf_gaps=None):
    xy = coords[:, :2]
    bonds = _get_bonds(coords, symbols)

    # bonds
    for (i, j) in bonds:
        x0, y0 = xy[i]; x1, y1 = xy[j]
        key = (min(i, j), max(i, j))
        lw = 1.8
        if key in DOUBLE_BOND_PAIRS:
            dx, dy = x1 - x0, y1 - y0
            length = np.hypot(dx, dy)
            nx, ny = -dy / length, dx / length
            off = 0.08
            for sgn in (-1, +1):
                ax.plot([x0 + sgn*off*nx, x1 + sgn*off*nx],
                        [y0 + sgn*off*ny, y1 + sgn*off*ny],
                        color='#333333', lw=lw, zorder=3, solid_capstyle='round')
        else:
            ax.plot([x0, x1], [y0, y1],
                    color='#333333', lw=lw, zorder=3, solid_capstyle='round')

    # atoms
    for sym, (x, y) in zip(symbols, xy):
        ec = '#444444' if sym != 'H' else '#999999'
        ax.add_patch(plt.Circle((x, y), ATOM_RADII[sym],
                                facecolor=ATOM_COLORS[sym], edgecolor=ec,
                                linewidth=0.8, zorder=ATOM_ZS[sym]))
        if sym in ('C', 'O'):
            ax.text(x, y, sym, ha='center', va='center',
                    fontsize=5.5, fontweight='bold', color='white',
                    zorder=ATOM_ZS[sym] + 1)

    # axis
    margin = 1.1
    ax.set_xlim(xy[:, 0].min() - margin, xy[:, 0].max() + margin)
    ax.set_ylim(xy[:, 1].min() - margin, xy[:, 1].max() + margin)
    ax.set_aspect('equal')
    ax.axis('off')

    # panel title
    opt_flag = '' if 'opt' in status else '*'
    ax.set_title(f'{label}{opt_flag}', fontsize=11, fontweight='bold', pad=4)

    # our wB97X-D energy (primary, larger)
    if our_kcal is not None:
        ax.text(0.5, -0.03,
                f'$\\Delta E$ = {our_kcal:+.2f} kcal mol$^{{-1}}$ (wB97X-D)',
                transform=ax.transAxes, ha='center', va='top', fontsize=8.5,
                color='#1a1a1a')
        ax.text(0.5, -0.12,
                f'$\\Delta E$ = {barber_kcal:.2f} kcal mol$^{{-1}}$ (Barber CCSD(T))',
                transform=ax.transAxes, ha='center', va='top', fontsize=7.5,
                color='#666666')
        y_cas = -0.21
    else:
        ax.text(0.5, -0.03, f'$\\Delta E$ = {barber_kcal:.2f} kcal mol$^{{-1}}$',
                transform=ax.transAxes, ha='center', va='top', fontsize=9.5)
        y_cas = -0.11

    # CASSCF gap line
    if casscf_gaps is not None:
        cas = casscf_gaps['active_space']
        s1  = casscf_gaps['gap_s1']
        t1  = casscf_gaps['gap_t1']
        gap_txt = (f'CASSCF{cas}: '
                   f'$\\Delta E_{{S1}}$={s1:.1f}, $\\Delta E_{{T1}}$={t1:.1f} kcal mol$^{{-1}}$')
        ax.text(0.5, y_cas, gap_txt,
                transform=ax.transAxes, ha='center', va='top', fontsize=7.0,
                color='#1155aa')
        y_status = y_cas - 0.10
    else:
        ax.text(0.5, y_cas, 'CASSCF: pending',
                transform=ax.transAxes, ha='center', va='top', fontsize=7.0,
                color='#aaaaaa', style='italic')
        y_status = y_cas - 0.10

    # status line (small, grey)
    ax.text(0.5, y_status, status,
            transform=ax.transAxes, ha='center', va='top',
            fontsize=6.5, color='#888888', style='italic')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    geoms      = load_conformer_geometries()
    our_de     = load_our_energies()    # wB97X-D/6-31G* ΔE (kcal/mol), may be empty
    casscf_de  = load_casscf_gaps()     # CASSCF S1/T1 gaps, keyed by conformer name

    # Barber et al. 2018 CCSD(T)/aug-cc-pVTZ//B3LYP relative energies
    BARBER_DE = {'syn-trans': 0.00, 'syn-cis': 1.76,
                 'anti-trans': 2.57, 'anti-cis': 3.05}

    layout = [
        (0, 0, 'syn-trans'),
        (0, 1, 'syn-cis'),
        (1, 0, 'anti-trans'),
        (1, 1, 'anti-cis'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 9.0),
                              gridspec_kw={'wspace': 0.10, 'hspace': 0.52})

    any_approx = False
    for row, col, name in layout:
        coords, symbols, status = geoms[name]
        if 'starting' in status:
            any_approx = True
        _draw_conformer(axes[row][col], coords, symbols,
                        name, BARBER_DE[name], status,
                        our_kcal=our_de.get(name),
                        casscf_gaps=casscf_de.get(name))

    # row / column headers
    for ax, hdr in zip(axes[:, 0], ['syn', 'anti']):
        ax.text(-0.10, 0.5, hdr, transform=ax.transAxes,
                ha='right', va='center', fontsize=13, fontweight='bold',
                rotation=90)
    for ax, hdr in zip(axes[0, :], ['trans', 'cis']):
        ax.text(0.5, 1.12, hdr, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=13, fontweight='bold')

    # legend
    legend_elements = [
        mpatches.Patch(facecolor=ATOM_COLORS['C'], edgecolor='#444444', label='C'),
        mpatches.Patch(facecolor=ATOM_COLORS['O'], edgecolor='#444444', label='O'),
        mpatches.Patch(facecolor=ATOM_COLORS['H'], edgecolor='#999999', label='H'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))

    # title
    fig.text(0.5, 0.98,
             'MVKO Conformers — wB97X-D, Barber CCSD(T), and CASSCF gaps',
             ha='center', va='top', fontsize=11)
    if any_approx:
        fig.text(0.5, 0.945,
                 '* starting geometry — PSI4 wB97X-D/6-31G* optimisation in progress',
                 ha='center', va='top', fontsize=7.5, color='#888888', style='italic')

    out_png = 'outputs/mvko_conformers_figure.png'
    out_pdf = 'outputs/mvko_conformers_figure.pdf'
    Path('outputs').mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out_png}')
    print(f'Saved: {out_pdf}')

    # Report geometry sources
    print('\nGeometry sources:')
    for row, col, name in layout:
        _, _, status = geoms[name]
        print(f'  {name:12s}: {status}')


if __name__ == '__main__':
    main()
