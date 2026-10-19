"""H2O references: frozen-core FCI/6-31G for O and H atoms, eight valence OH states, H2, and 149 H2O geometries.

Data sets produced:
  oh_curves   eight OH states (columns OH_STATES), eV relative to O(3P) + H; meta holds the O term energies
  h2_631g     H2 singlet/triplet in 6-31G, eV relative to 2 H
  h2o         149 geometries (r1, r2, theta), ground singlet, eV relative to O(3P) + 2 H
"""
from __future__ import annotations

import numpy as np

from ..refset import RefSet
from ..units import HARTREE_EV
from .fci import fc_fci

OH_STATES = ['2S-', '2P1', '2P2', '4S-', '4P', '2S+1', '2S+2', '2D']
OH_LABELS = {'2P1': 'X 2Pi (3P)', '4S-': '4Sigma- (3P)', '2S-': '2Sigma- (3P)', '4P': '4Pi (3P)',
             '2S+1': 'A 2Sigma+ (1D)', '2D': '2Delta (1D)', '2P2': '2Pi (1D)', '2S+2': '2Sigma+ (1S)'}


def oh_states(r):
    """All eight valence OH states at bond length r, labelled with C2v symmetry (Sigma+ A1, Sigma- A2, Pi B1,
    Delta = one A1 + one A2 component)."""
    at = f'O 0 0 0; H 0 0 {r}'
    a1 = fc_fci(at, 0, 5, 4, nroots=5, ss=0.75, sym='C2v', wfnsym='A1', ref_charge=-1)
    a2 = fc_fci(at, 0, 5, 4, nroots=3, ss=0.75, sym='C2v', wfnsym='A2', ref_charge=-1)
    b1 = fc_fci(at, 0, 5, 4, nroots=3, ss=0.75, sym='C2v', wfnsym='B1', ref_charge=-1)
    q2 = fc_fci(at, 0, 6, 3, nroots=2, ss=3.75, sym='C2v', wfnsym='A2', ref_charge=-1)
    qb = fc_fci(at, 0, 6, 3, nroots=2, ss=3.75, sym='C2v', wfnsym='B1', ref_charge=-1)
    d = [np.min(np.abs(a1 - e)) for e in a2]
    delta, sig_minus = a2[int(np.argmin(d))], np.delete(a2, int(np.argmin(d)))[0]
    sig_plus = np.delete(a1, int(np.argmin(np.abs(a1 - delta))))
    return {'2S-': sig_minus, '2P1': b1[0], '2P2': b1[1], '4S-': q2[0], '4P': qb[0],
            '2S+1': sig_plus[0], '2S+2': sig_plus[1], '2D': delta}


def from_legacy(npz, outdir):
    """Convert the cached arrays from the exploratory H2O notebook (h2o_refs.npz)."""
    z = np.load(npz)
    EO, EH = float(z['E_O3P']), float(z['E_H'])
    meta = {'E_O3P': EO, 'E_O1D': float(z['E_O1D']), 'E_O1S': float(z['E_O1S']), 'E_H': EH, 'basis': '6-31G',
            'method': 'frozen-core FCI'}
    r = z['r_di']
    oh = np.stack([z['OH_' + k] for k in OH_STATES], 1)
    RefSet('OH states fc-FCI/6-31G', r[:, None], (oh - EO - EH) * HARTREE_EV, ['r'],
           meta=dict(meta, zero='O(3P) + H', states=OH_STATES)).save(f'{outdir}/oh_curves.npz')
    RefSet('H2 fc-FCI/6-31G', r[:, None], (np.c_[z['H2_S'], z['H2_T']] - 2 * EH) * HARTREE_EV, ['r'],
           meta=dict(meta, zero='2 H', states=['singlet', 'triplet'])).save(f'{outdir}/h2_631g.npz')
    RefSet('H2O fc-FCI/6-31G', z['h2o_geo'], (z['h2o_E'][:, None] - EO - 2 * EH) * HARTREE_EV, ['r1', 'r2', 'theta'],
           extra={'tag': z['h2o_tag']}, meta=dict(meta, zero='O(3P) + 2 H')).save(f'{outdir}/h2o.npz')
