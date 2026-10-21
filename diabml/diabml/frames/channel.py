"""Fragment-channel frame for a molecule that breaks into two fragments A + B along one coordinate r.

Each diabat is a *structure* labelled by the fragment states it dissociates to, e.g. for CH2OO -> CH2O + O:

    covalent   CH2O(S0) + O(1D)   (three A' components, degenerate at r -> infinity)
    biradical  CH2O(T1) + O(3P)   singlet-coupled
    ionic      CH2O(+)  + O(-)    the zwitterion structure H2C=O(+)-O(-), with a Coulomb tail -14.40 q_A q_B / r

The asymptote of every diabat is fixed by fragment (ladder) energies: neutral, excited, cation and anion fragment
calculations give exactly the numbers a diabatic model needs at long range. The frame then guarantees:

    Omega_ii(r -> inf) = E_inf_i + tail_i(r),      Omega_ij(r -> inf) = 0,

and, with a short-range wall on each structure (Born-Mayer for the ion pair, balanced against the Coulomb pull at
the equilibrium distance), no structure can fall into an unphysical hole below the data.

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
    wall_A: float = 0.0           # short-range repulsion wall_A * exp(-r / wall_rho) in eV (Born-Mayer form)
    wall_rho: float = 0.3         # Angstrom

    @staticmethod
    def born_mayer_balance(r_eq, charge_product=-1.0, rho=0.3, softening=0.5):
        """wall_A that makes Coulomb + Born-Mayer stationary at r_eq (an ion pair with its minimum there)."""
        coulomb_slope = -charge_product * COULOMB_EV_ANG * r_eq / (r_eq ** 2 + softening ** 2) ** 1.5
        return float(coulomb_slope * rho * np.exp(r_eq / rho))


class AsymptoteTable:
    """Asymptotic energies of a channel's structures as a function of one coordinate that survives dissociation.

    vals[k, s] is the energy of structure s at knot k of coordinate column j (e.g. r_CO, which sets the CH2O fragment
    energy); between knots the energies are interpolated linearly and they are held constant outside. With j = None
    the table is a single row of constants. extra = list of (structure index, list of source columns, offset): an
    entry (4, [0, 1, 2], dE) makes structure 4 sit dE above the mean of structures 0-2 (the ion pair of CH2OO).
    Works on numpy arrays and on torch tensors (differentiable), and serialises to plain JSON (spec / from_spec)."""

    def __init__(self, knots, vals, j=None, extra=()):
        self.knots = np.atleast_1d(np.asarray(knots, float))
        self.vals = np.atleast_2d(np.asarray(vals, float))
        self.j = j
        self.extra = [(int(i), [int(c) for c in cols], float(off)) for i, cols, off in extra]

    def _base_np(self, coords):
        if self.j is None or len(self.knots) == 1:
            return np.repeat(self.vals[:1], len(coords), 0)
        x = coords[:, self.j]
        return np.stack([np.interp(x, self.knots, self.vals[:, s]) for s in range(self.vals.shape[1])], 1)

    def _base_torch(self, coords):
        V = torch.as_tensor(self.vals, dtype=coords.dtype)
        if self.j is None or len(self.knots) == 1:
            return V[:1].expand(coords.shape[0], -1)
        K = torch.as_tensor(self.knots, dtype=coords.dtype)
        x = coords[:, self.j].clamp(K[0], K[-1])
        i = torch.searchsorted(K, x.detach()).clamp(1, len(K) - 1)
        t = ((x - K[i - 1]) / (K[i] - K[i - 1]))[:, None]
        return V[i - 1] + t * (V[i] - V[i - 1])

    def __call__(self, coords):
        is_t = isinstance(coords, torch.Tensor)
        base = self._base_torch(coords) if is_t else self._base_np(np.atleast_2d(coords))
        cols = [base[:, s] for s in range(base.shape[1])]
        for i, src, off in self.extra:
            val = sum(cols[c] for c in src) / len(src) + off
            if i < len(cols):
                cols[i] = val
            else:
                cols.append(val)
        return torch.stack(cols, 1) if is_t else np.stack(cols, 1)

    def spec(self):
        return {'type': 'AsymptoteTable', 'knots': self.knots.tolist(), 'vals': self.vals.tolist(), 'j': self.j,
                'extra': [list(e) for e in self.extra]}

    @classmethod
    def from_spec(cls, d):
        return cls(d['knots'], d['vals'], d['j'], d.get('extra', ()))


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
        self.n_coords = n_coords
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
        A = np.array([d.wall_A for d in self.diabats])[None, :]
        rho = np.array([d.wall_rho for d in self.diabats])[None, :]
        return asym + q * COULOMB_EV_ANG / np.sqrt(r[:, None] ** 2 + self.soft ** 2) + A * np.exp(-r[:, None] / rho)

    def prepare(self, coords):
        coords = np.atleast_2d(np.asarray(coords, float))
        if self.coord_scale is None:
            self.fit_scaler(coords)
        r = coords[:, self.r_index]
        t = lambda a: torch.tensor(np.asarray(a, float))
        x = t(self.features(coords))[:, None, :]
        return {'base': t(self.tails(r, coords)), 'sw': t(self.envelope(r)), 'slot_in': {'diag': x, 'coup': x}}

    def prepare_torch(self, coords):
        """Like prepare, but from a torch tensor and differentiable with respect to the coordinates (for forces)."""
        if self.coord_scale is None:
            raise RuntimeError('fit_scaler (or train) the frame before prepare_torch')
        r = coords[:, self.r_index]
        ov = torch.stack([slater_overlap(r, z) / float(slater_overlap(self.r_ref, z)) for z in self.fz], 1)
        keep = [j for j in range(coords.shape[1]) if j != self.r_index]
        mean = torch.as_tensor(np.asarray(self.coord_scale[0], float), dtype=coords.dtype)
        std = torch.as_tensor(np.asarray(self.coord_scale[1], float), dtype=coords.dtype)
        other = (coords[:, keep] - mean) / std
        parts = [ov, other] + ([(r[:, None] - self.r_ref)] if self.raw_r else [])
        x = torch.cat(parts, 1)[:, None, :]
        if self.asymptote_fn is not None:
            asym = self.asymptote_fn(coords)
        else:
            asym = torch.as_tensor([d.asymptote for d in self.diabats], dtype=coords.dtype)[None, :].expand(len(r), -1)
        q = torch.as_tensor([d.charge_product for d in self.diabats], dtype=coords.dtype)[None, :]
        A = torch.as_tensor([d.wall_A for d in self.diabats], dtype=coords.dtype)[None, :]
        rho = torch.as_tensor([d.wall_rho for d in self.diabats], dtype=coords.dtype)[None, :]
        base = (asym + q * COULOMB_EV_ANG / torch.sqrt(r[:, None] ** 2 + self.soft ** 2)
                + A * torch.exp(-r[:, None] / rho))
        sw = slater_overlap(r, self.zeta_env) / float(slater_overlap(self.r_ref, self.zeta_env))
        return {'base': base, 'sw': sw, 'slot_in': {'diag': x, 'coup': x}}

    def spec(self):
        """Everything needed to rebuild this frame, as plain JSON-serialisable data."""
        if self.asymptote_fn is not None and not hasattr(self.asymptote_fn, 'spec'):
            raise TypeError('asymptote_fn must be an AsymptoteTable (or have spec()) to save the frame')
        return {'type': 'ChannelFrame', 'n_coords': self.n_coords,
                'diabats': [d.__dict__ for d in self.diabats], 'r_index': self.r_index,
                'coord_scale': None if self.coord_scale is None else [np.asarray(v, float).tolist() for v in self.coord_scale],
                'zeta_env': self.zeta_env, 'r_ref': self.r_ref, 'feature_zetas': list(self.fz), 'raw_r': self.raw_r,
                'coulomb_softening': self.soft, 'name': self.name,
                'asymptote_fn': None if self.asymptote_fn is None else self.asymptote_fn.spec()}

    @classmethod
    def from_spec(cls, d):
        cs = d['coord_scale']
        return cls([Diabat(**x) for x in d['diabats']], d['n_coords'], r_index=d['r_index'],
                   coord_scale=None if cs is None else (np.array(cs[0]), np.array(cs[1])), zeta_env=d['zeta_env'],
                   r_ref=d['r_ref'], feature_zetas=tuple(d['feature_zetas']), raw_r=d['raw_r'],
                   coulomb_softening=d['coulomb_softening'], name=d['name'],
                   asymptote_fn=None if d['asymptote_fn'] is None else AsymptoteTable.from_spec(d['asymptote_fn']))

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
