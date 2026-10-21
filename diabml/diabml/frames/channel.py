"""Fragment-channel frame for a molecule that breaks into two fragments A + B along one coordinate r.

Each diabat is a *structure* labelled by the fragment states it dissociates to, e.g. for CH2OO -> CH2O + O:

    covalent   CH2O(S0) + O(1D)   (three A' components, degenerate at r -> infinity)
    biradical  CH2O(T1) + O(3P)   singlet-coupled
    ionic      CH2O(+)  + O(-)    the zwitterion structure H2C=O(+)-O(-), with a Coulomb tail -14.40 q_A q_B / r

The asymptote of every diabat is fixed by fragment (ladder) energies: neutral, excited, cation and anion fragment
calculations give exactly the numbers a diabatic model needs at long range. The frame then guarantees:

    Omega_ii(r -> inf) = E_inf_i + tail_i(r),      Omega_ij(r -> inf) = 0,

because every learned part is multiplied by an overlap envelope env(r) that decays to 0 as the fragments separate.

Slots:
  diag   one evaluation per geometry, input = standardised internal coordinates, n_diabats outputs (eV)
  coup   one evaluation per geometry, same input, n_diabats (n_diabats - 1) / 2 outputs (eV)
Optionally diagonal elements can carry a fixed short-range prior (a callable of the coordinates, in eV).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .base import Frame, slater_overlap
from ..units import COULOMB_EV_ANG


@dataclass
class Diabat:
    label: str
    asymptote: float              # eV, same zero as the reference data
    charge_product: float = 0.0   # q_A * q_B for a Coulomb tail (e.g. -1 for an ion pair)
    kind: str = 'covalent'        # covalent | biradical | ionic   (for plotting / bookkeeping)


class ChannelFrame(Frame):
    """Frame for one dissociation channel A + B along coordinate r (column r_index of the coordinates).

    Learned parts enter as   Omega_ii += env(r) * f_i(x),   Omega_ij = env(r) * g_ij(x)
    with env(r) = S(r) / S(r_ref), a Slater-overlap envelope (orbital overlap decays roughly the way exchange and
    resonance couplings do), and network inputs x that *saturate* as the fragments separate: overlaps of r at a few
    exponents plus the other coordinates (standardised), and optionally r itself (raw_r) for short-range flexibility.
    Outside the training range the overlap inputs stop changing and the exponentially decaying envelope removes the
    correction (it beats any growth of a GELU network in r), so long-range behaviour is fixed by physics.
    """

    def __init__(self, diabats: list[Diabat], n_coords, r_index=0, coord_scale=None, zeta_env=1.0, r_ref=1.35,
                 feature_zetas=(0.6, 1.0, 1.6), raw_r=False, coulomb_softening=0.5, name='fragment-channel frame',
                 asymptote_fn=None):
        self.diabats = diabats
        # optional: asymptote_fn(coords) -> (N, n_diabats) asymptotic energies that depend on the coordinates that do
        # not separate the fragments (e.g. a C-O stretch inside CH2O); default = the constant Diabat.asymptote values
        self.asymptote_fn = asymptote_fn
        self.n_diabats = len(diabats)
        self.diabat_labels = [d.label for d in diabats]
        self.r_index = r_index
        self.coord_scale = coord_scale            # (mean, std) of the non-r coordinates; set by fit_scaler
        self.zeta_env, self.r_ref, self.fz, self.raw_r = zeta_env, r_ref, tuple(feature_zetas), raw_r
        self.soft = coulomb_softening             # Angstrom; keeps the Coulomb tail finite at short range
        self.name = name
        self.iu = np.triu_indices(self.n_diabats, 1)
        n_in = len(self.fz) + n_coords - 1 + int(raw_r)
        self.slots = {'diag': (n_in, self.n_diabats), 'coup': (n_in, len(self.iu[0]))}

    def fit_scaler(self, coords):
        other = np.delete(np.asarray(coords, float), self.r_index, axis=1)
        self.coord_scale = (other.mean(0), other.std(0) + 1e-9)
        return self

    def envelope(self, r):
        return slater_overlap(np.asarray(r, float), self.zeta_env) / slater_overlap(self.r_ref, self.zeta_env)

    def features(self, coords):
        coords = np.atleast_2d(np.asarray(coords, float))
        r = coords[:, self.r_index]
        ov = np.stack([slater_overlap(r, z) / slater_overlap(self.r_ref, z) for z in self.fz], 1)
        other = (np.delete(coords, self.r_index, axis=1) - self.coord_scale[0]) / self.coord_scale[1]
        parts = [ov, other] + ([(r[:, None] - self.r_ref)] if self.raw_r else [])
        return np.concatenate(parts, 1)

    def tails(self, r, coords=None):
        r = np.asarray(r, float)
        if self.asymptote_fn is not None and coords is not None:
            asym = np.asarray(self.asymptote_fn(np.atleast_2d(coords)), float)
        else:
            asym = np.array([d.asymptote for d in self.diabats])[None, :] * np.ones((len(r), 1))
        q = np.array([d.charge_product for d in self.diabats])[None, :]
        return asym + q * COULOMB_EV_ANG / np.sqrt(r[:, None] ** 2 + self.soft ** 2)

    def prepare(self, coords):
        coords = np.atleast_2d(np.asarray(coords, float))
        if self.coord_scale is None:
            self.fit_scaler(coords)
        r = coords[:, self.r_index]
        t = lambda a: torch.tensor(np.asarray(a, float))
        x = t(self.features(coords))[:, None, :]
        return {'base': t(self.tails(r, coords)), 'sw': t(self.envelope(r)), 'slot_in': {'diag': x, 'coup': x}}

    def assemble(self, c, out):
        N, n = c['base'].shape
        sw = c['sw'][:, None]
        M = torch.diag_embed(c['base'] + sw * out['diag'][:, 0])
        U = torch.zeros(N, n, n)
        U[:, self.iu[0], self.iu[1]] = sw * out['coup'][:, 0]
        return M + U + U.transpose(1, 2)

    def diabatic_energies(self, model, coords):
        """Diagonal elements (diabatic energies) of a trained model's matrix, for plotting."""
        M = model.matrix(coords).detach().numpy()
        return np.diagonal(M, axis1=1, axis2=2)
