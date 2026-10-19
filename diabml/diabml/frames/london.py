"""London / LEPS frame for H3: the 2x2 covalent valence-bond matrix built from exact H2 singlet and triplet curves.

    Omega = (Q12 + Q13 + Q23) 1 + [[a, b], [b, -a]],   a = J12 - (J13 + J23)/2,   b = (sqrt 3 / 2)(J13 - J23)
    Q = (1E + 3E)/2,  J = (1E - 3E)/2  per pair.

(a, b) transform as the E irrep of D3h, so the two eigenvalues are exactly degenerate at every equilateral geometry,
and the eigenvalues are invariant to any permutation of the atoms. When one atom leaves, the matrix reduces to the
exact H2 + H energies.

Slots (learned corrections that keep all of the above):
  sato   per pair, input r_ij: a LEPS Sato function d(r) mixing singlet and triplet curves (bounded by tanh),
         switched on only when the third atom is near. (Classic LEPS applies it everywhere, which leaves the ground
         state exact but scales the triplet curve by (1-d)/(1+d), breaking the excited state's H2 + H limit.)
  three  per pair, input r_ij: corrections (dQ, dJ) to Q and J, multiplied by a three-body envelope that vanishes
         when any of the three atoms leaves
"""
from __future__ import annotations

import numpy as np
import torch

from .base import Frame, Curve1D, slater_overlap

SQ3 = np.sqrt(3.0) / 2


class LondonFrame(Frame):
    name = 'London (H3, 2x2 covalent VB)'
    n_diabats = 2
    diabat_labels = ['VB structure A', 'VB structure B']
    slots = {'sato': (1, 1), 'three': (1, 2)}

    def __init__(self, h2_refset, use_sato=True, use_three=True):
        r = h2_refset.coords[:, 0]
        self.S = Curve1D(r, h2_refset.energies[:, 0])
        self.T = Curve1D(r, h2_refset.energies[:, 1])
        self.use_sato, self.use_three = use_sato, use_three

    def prepare(self, R):
        R = np.atleast_2d(np.asarray(R, float))
        t = lambda x: torch.tensor(np.asarray(x, float))
        Rt = t(R)
        return {'R': Rt, 'S': t(self.S(R)), 'T': t(self.T(R)),
                'slot_in': {'sato': Rt[..., None], 'three': Rt[..., None]}}

    def assemble(self, c, out):
        S, T, R = c['S'], c['T'], c['R']
        sR = slater_overlap(R)
        # proximity of the third atom to pair (i, j): ~1 when it is close, -> 0 when it leaves
        near = torch.stack([sR[:, 1] + sR[:, 2], sR[:, 0] + sR[:, 2], sR[:, 0] + sR[:, 1]], 1)
        prox = near / (near + 0.02)
        # Sato mixing only in the presence of a third atom, so both H2 + H limits (singlet and triplet) stay exact
        d = (torch.tanh(out['sato'][..., 0]) * 0.9 * sR / (sR + 0.05) * prox) if self.use_sato else torch.zeros_like(R)
        Q = ((1 + d) * S + (1 - d) * T) / 2 / (1 + d)
        J = ((1 + d) * S - (1 - d) * T) / 2 / (1 + d)
        if self.use_three:
            env = sR * torch.stack([sR[:, 1] + sR[:, 2], sR[:, 0] + sR[:, 2], sR[:, 0] + sR[:, 1]], 1) / 2
            Q = Q + out['three'][..., 0] * env
            J = J + out['three'][..., 1] * env
        q = Q.sum(1)
        a = J[:, 0] - (J[:, 1] + J[:, 2]) / 2
        b = SQ3 * (J[:, 1] - J[:, 2])
        M = torch.stack([torch.stack([q + a, b], 1), torch.stack([b, q - a], 1)], 1)
        return M
