"""H3 references: two lowest doublets by FCI/cc-pVDZ, plus H2 singlet/triplet fragment curves.

Data sets produced (all energies in eV relative to three free H atoms):
  h3_cut      the Shu et al. (JCTC 2025) cut: r23 = 1.08 A, r12, r13 in [0.88, 1.40]  (441 points)
  h3_global   collinear 16x16 grid + bent (60, 90, 120 deg) 9x9 grids, out to 3.6 A   (499 points)
  h2_curves   H2 singlet and triplet, relative to two free H atoms                     (78 points)
"""
from __future__ import annotations

import numpy as np

from ..geometry import h3_xy, h3_from_bend
from ..refset import RefSet
from ..units import HARTREE_EV
from .fci import fci_states


def h3_two_states(r12, r13, r23):
    return fci_states(h3_xy(r12, r13, r23), 0, (2, 1), nroots=2, ss=0.75)


def cut_coords():
    g = np.linspace(0.88, 1.40, 21)
    return np.array([(a, b, 1.08) for a in g for b in g])


def global_geometries():
    lin, bent = np.linspace(0.6, 3.6, 16), np.linspace(0.6, 3.0, 9)
    geo = [(a, b, 180.0) for a in lin for b in lin] + [(a, b, th) for th in (60.0, 90.0, 120.0) for a in bent for b in bent]
    return np.array(geo)


def generate(outdir, log=print):
    E_H = fci_states([[0, 0, 0]], 0, (1, 0))[0]
    r = np.concatenate([np.linspace(0.40, 3.0, 53), np.linspace(3.1, 8.0, 25)])
    S = np.array([fci_states([[0, 0, 0], [x, 0, 0]], 0, (1, 1), 1, 0.0)[0] for x in r])
    T = np.array([fci_states([[0, 0, 0], [x, 0, 0]], 0, (2, 0), 1, 2.0)[0] for x in r])
    h2 = RefSet('H2 FCI/cc-pVDZ', r[:, None], np.c_[S, T] - 2 * E_H, ['r'], meta={'zero': '2 H', 'states': ['singlet', 'triplet']})
    h2.energies = h2.energies * HARTREE_EV
    h2.save(f'{outdir}/h2_curves.npz')
    log('H2 done')
    cut = cut_coords()
    Ec = np.array([h3_two_states(*d) for d in cut])
    RefSet('H3 CI cut FCI/cc-pVDZ', cut, (Ec - 3 * E_H) * HARTREE_EV, ['r12', 'r13', 'r23'],
           meta={'zero': '3 H', 'E_H_hartree': E_H}).save(f'{outdir}/h3_cut.npz')
    log('cut done')
    geo = global_geometries()
    R = np.array([h3_from_bend(*x) for x in geo])
    Eg = np.array([h3_two_states(*d) for d in R])
    RefSet('H3 global FCI/cc-pVDZ', R, (Eg - 3 * E_H) * HARTREE_EV, ['r12', 'r13', 'r23'], extra={'bend_geometry': geo},
           meta={'zero': '3 H', 'E_H_hartree': E_H}).save(f'{outdir}/h3_global.npz')
    log('global done')


def from_legacy(npz, outdir):
    """Convert the cached arrays from the exploratory notebooks (h3_two_state_refs.npz)."""
    z = np.load(npz)
    E_H = float(z['E_H'])
    h2 = RefSet('H2 FCI/cc-pVDZ', z['r_di'][:, None], np.c_[z['H2_S'], z['H2_T']] - 2 * E_H, ['r'],
                meta={'zero': '2 H', 'states': ['singlet', 'triplet']})
    h2.energies = h2.energies * HARTREE_EV
    h2.save(f'{outdir}/h2_curves.npz')
    RefSet('H3 CI cut FCI/cc-pVDZ', z['cut_R'], (z['cut_E'] - 3 * E_H) * HARTREE_EV, ['r12', 'r13', 'r23'],
           meta={'zero': '3 H', 'E_H_hartree': E_H}).save(f'{outdir}/h3_cut.npz')
    RefSet('H3 global FCI/cc-pVDZ', z['glob_R'], (z['glob_E'] - 3 * E_H) * HARTREE_EV, ['r12', 'r13', 'r23'],
           extra={'bend_geometry': z['glob_geo']}, meta={'zero': '3 H', 'E_H_hartree': E_H}).save(f'{outdir}/h3_global.npz')
