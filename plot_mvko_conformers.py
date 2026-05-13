#!/usr/bin/env python3
"""
2×2 figure of the four MVKO conformers (syn/anti × trans/cis),
modelled after Figure 3 of Barber et al. 2018 (Phys. Chem. Chem. Phys.).

Geometries:
  syn-trans  — wB97X-D/6-31G* opt (outputs/wB97X_nm_model_v5/)
  anti-trans — wB97X-D/6-31G* opt (outputs/anti_trans_nm_pes_20260421/)
  syn-cis    — approximate (180° vinyl rotation from syn-trans; see test_molecules.py)
  anti-cis   — approximate (180° vinyl rotation from anti-trans; see test_molecules.py)

Relative energies shown are from Barber et al. 2018 (CCSD(T)/aug-cc-pVTZ//B3LYP).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pickle
from pathlib import Path


# ---------------------------------------------------------------------------
# Atom appearance
# ---------------------------------------------------------------------------
ATOM_COLORS = {'C': '#555555', 'O': '#CC3333', 'H': '#EEEEEE'}
ATOM_RADII  = {'C': 0.40,      'O': 0.35,      'H': 0.22}   # Å, visual only
ATOM_ZS     = {'C': 10,        'O': 10,         'H': 5}
COVALENT_R  = {'C': 0.77,      'O': 0.73,       'H': 0.31}   # Å, for bond detection

BOND_TOL = 0.30   # Å extra tolerance on top of sum of covalent radii
DOUBLE_BOND_PAIRS = {(3, 4)}   # C2=C3 by 0-based index (always the vinyl bond)


# ---------------------------------------------------------------------------
# Load geometries
# ---------------------------------------------------------------------------
def _load_syn_trans():
    path = Path('outputs/wB97X_nm_model_v5/mlpes_wB97X_nm.pkl')
    if path.exists():
        with open(path, 'rb') as f:
            m = pickle.load(f)
        return np.array(m['eq_coords_ang']), list(m['symbols'])
    from modules.test_molecules import get_molecule
    mol = get_molecule('mvko')
    return mol.coordinates.copy(), list(mol.symbols)


def _load_anti_trans():
    path = Path('outputs/anti_trans_nm_pes_20260421/eq_coords.npy')
    if path.exists():
        coords = np.load(str(path))
        from modules.test_molecules import get_molecule
        symbols = list(get_molecule('mvko_anti_trans').symbols)
        return coords, symbols
    from modules.test_molecules import get_molecule
    mol = get_molecule('mvko_anti_trans')
    return mol.coordinates.copy(), list(mol.symbols)


def _load_from_test_molecules(key):
    from modules.test_molecules import get_molecule
    mol = get_molecule(key)
    return mol.coordinates.copy(), list(mol.symbols)


# ---------------------------------------------------------------------------
# Bond detection
# ---------------------------------------------------------------------------
def _get_bonds(coords, symbols):
    n = len(symbols)
    bonds = []
    for i in range(n):
        for j in range(i + 1, n):
            r_cut = COVALENT_R[symbols[i]] + COVALENT_R[symbols[j]] + BOND_TOL
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < r_cut:
                bonds.append((i, j))
    return bonds


# ---------------------------------------------------------------------------
# Draw one conformer panel
# ---------------------------------------------------------------------------
def _draw_conformer(ax, coords, symbols, label, energy_str, optimized=True):
    # Work in the xy plane (all conformers are planar or near-planar in z)
    xy = coords[:, :2]

    bonds = _get_bonds(coords, symbols)

    # --- draw bonds ---
    for (i, j) in bonds:
        x0, y0 = xy[i]
        x1, y1 = xy[j]
        key = (min(i, j), max(i, j))
        lw = 1.8
        if key in DOUBLE_BOND_PAIRS:
            # Draw two parallel lines for double bond
            dx, dy = x1 - x0, y1 - y0
            length = np.sqrt(dx**2 + dy**2)
            nx, ny = -dy / length, dx / length   # normal vector
            offset = 0.08
            for sign in (-1, +1):
                ax.plot([x0 + sign*offset*nx, x1 + sign*offset*nx],
                        [y0 + sign*offset*ny, y1 + sign*offset*ny],
                        color='#333333', lw=lw, zorder=3, solid_capstyle='round')
        else:
            ax.plot([x0, x1], [y0, y1],
                    color='#333333', lw=lw, zorder=3, solid_capstyle='round')

    # --- draw atoms ---
    for idx, (sym, (x, y)) in enumerate(zip(symbols, xy)):
        radius = ATOM_RADII[sym]
        facecolor = ATOM_COLORS[sym]
        edgecolor = '#444444' if sym != 'H' else '#999999'
        circle = plt.Circle((x, y), radius,
                             facecolor=facecolor, edgecolor=edgecolor,
                             linewidth=0.8, zorder=ATOM_ZS[sym])
        ax.add_patch(circle)

        # Atom label (only heavy atoms for clarity)
        if sym in ('C', 'O'):
            fontsize = 5.5
            ax.text(x, y, sym,
                    ha='center', va='center',
                    fontsize=fontsize, fontweight='bold',
                    color='white' if sym == 'C' else 'white',
                    zorder=ATOM_ZS[sym] + 1)

    # --- axis formatting ---
    margin = 1.1
    xmin, xmax = xy[:, 0].min() - margin, xy[:, 0].max() + margin
    ymin, ymax = xy[:, 1].min() - margin, xy[:, 1].max() + margin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.axis('off')

    # --- panel title and energy ---
    opt_marker = '' if optimized else '*'
    ax.set_title(label + opt_marker, fontsize=11, fontweight='bold', pad=4)
    ax.text(0.5, -0.04, energy_str, transform=ax.transAxes,
            ha='center', va='top', fontsize=9.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    syn_trans_coords,  syn_trans_sym  = _load_syn_trans()
    anti_trans_coords, anti_trans_sym = _load_anti_trans()
    syn_cis_coords,    syn_cis_sym    = _load_from_test_molecules('mvko_syn_cis')
    anti_cis_coords,   anti_cis_sym   = _load_from_test_molecules('mvko_anti_cis')

    # Barber et al. 2018 CCSD(T)/aug-cc-pVTZ//B3LYP relative energies
    conformers = [
        # (row, col, coords, symbols, label, energy_kcal, optimized)
        (0, 0, syn_trans_coords,  syn_trans_sym,  'syn-trans',  0.00, True),
        (0, 1, syn_cis_coords,    syn_cis_sym,    'syn-cis',    1.76, False),
        (1, 0, anti_trans_coords, anti_trans_sym, 'anti-trans', 2.57, True),
        (1, 1, anti_cis_coords,   anti_cis_sym,   'anti-cis',   3.05, False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.0),
                              gridspec_kw={'wspace': 0.08, 'hspace': 0.20})

    for row, col, coords, symbols, label, energy, optimized in conformers:
        ax = axes[row][col]
        energy_str = f'$\\Delta E$ = {energy:.2f} kcal mol$^{{-1}}$'
        _draw_conformer(ax, coords, symbols, label, energy_str, optimized)

    # --- row / column headers ---
    for ax, header in zip(axes[:, 0], ['syn', 'anti']):
        ax.text(-0.08, 0.5, header, transform=ax.transAxes,
                ha='right', va='center', fontsize=12, fontweight='bold',
                rotation=90)

    for ax, header in zip(axes[0, :], ['trans', 'cis']):
        ax.text(0.5, 1.10, header, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # --- legend ---
    legend_elements = [
        mpatches.Patch(facecolor=ATOM_COLORS['C'], edgecolor='#444444', label='C'),
        mpatches.Patch(facecolor=ATOM_COLORS['O'], edgecolor='#444444', label='O'),
        mpatches.Patch(facecolor=ATOM_COLORS['H'], edgecolor='#999999', label='H'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))

    fig.text(0.5, 0.97,
             'MVKO Conformers — Barber et al. 2018 relative energies',
             ha='center', va='top', fontsize=11)
    fig.text(0.5, 0.93,
             r'* approximate geometry (PSI4 re-optimisation pending)',
             ha='center', va='top', fontsize=7.5, color='#666666', style='italic')

    out_path = 'outputs/mvko_conformers_figure.png'
    Path('outputs').mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out_path}')

    # Also save PDF
    pdf_path = out_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f'Saved: {pdf_path}')


if __name__ == '__main__':
    main()
