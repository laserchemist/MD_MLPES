"""A smooth local ground-state surface for spectroscopy: a low-order polynomial fitted to the reference S0 energies
in the well.

Why: a neural-network surface fitted to energies with errors of tens of meV across the whole surface does not fix the
curvature at the bottom of the well (vibrational frequencies depend on meV over hundredths of an Angstrom); members of
an ensemble disagree there and its harmonic frequencies can be far off even when its energies are good. A cubic (or
quartic) polynomial fitted by least squares to the few hundred reference points in the well averages over their
scatter and has the curvature the data imply. For CH2OO (SA4-CASSCF(8,6)+SC-NEVPT2/cc-pVDZ, 24_spec data) a cubic
in r_OO, angle COO, r_CO over the lowest 0.6 eV gives harmonic COO bend / O-O / C-O frequencies of 554 / 905 / 1434
cm-1 against 536 / 916 / 1338 (C-O mixed with the CH2 scissor) from NEVPT2/aVDZ in Su et al., Science 340, 174 (2013).

It is a one-state model with the interface the dynamics code uses (matrix_t, predict), so it runs in the same ASE
calculator; use it for low-energy spectra (trajectories that stay in the fitted region), the neural-network surface
for anything that leaves the well.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import torch

_SCALE = {'r_OO': 0.1, 'a_COO': 10.0, 'r_CO': 0.05, 'r_CH': 0.05, 'a_HCO': 10.0}


def _powers(n, order):
    return [p for d in range(order + 1) for p in itertools.product(range(d + 1), repeat=n) if sum(p) == d]


class PolyWell(torch.nn.Module):
    n_states = 1

    def __init__(self, names, center, scale, powers, coef, box, meta=None):
        super().__init__()
        self.names = list(names)
        self.center = torch.as_tensor(np.asarray(center, float))
        self.scale = torch.as_tensor(np.asarray(scale, float))
        self.pw = torch.as_tensor(np.asarray(powers, float))
        self.coef = torch.as_tensor(np.asarray(coef, float))
        self.box = {k: list(map(float, v)) for k, v in box.items()}
        self.meta = dict(meta or {})

    def energy_t(self, q):
        z = (q - self.center) / self.scale
        return torch.prod(z[:, None, :] ** self.pw[None], 2) @ self.coef

    def matrix_t(self, q):
        return self.energy_t(q)[:, None, None]

    def predict(self, q):
        with torch.no_grad():
            return self.energy_t(torch.as_tensor(np.atleast_2d(np.asarray(q, float))))[:, None].numpy()

    def inside(self, X):
        X = np.atleast_2d(X)
        ok = np.ones(len(X), bool)
        for j, n in enumerate(self.names):
            lo, hi = self.box[n]
            ok &= (X[:, j] >= lo) & (X[:, j] <= hi)
        return ok

    def save(self, path):
        np.savez(path, names=np.array(self.names), center=self.center.numpy(), scale=self.scale.numpy(),
                 powers=self.pw.numpy(), coef=self.coef.numpy(), box=json.dumps(self.box), meta=json.dumps(self.meta))
        return path

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls([str(n) for n in z['names']], z['center'], z['scale'], z['powers'], z['coef'],
                   json.loads(str(z['box'])), json.loads(str(z['meta'])))


def fit_well(ref, order=3, window_eV=0.6, half_width=None):
    """Least-squares polynomial of `order` in the model coordinates through the reference S0 energies within
    window_eV of the lowest one and inside a box around it (half_width per coordinate; default r 0.2 / 0.15 A,
    angle 20 deg). Returns (PolyWell, info)."""
    X = np.asarray(ref.coords, float)
    E = np.sort(np.asarray(ref.energies, float), 1)[:, 0]
    ok = np.isfinite(E)
    i0 = int(np.nanargmin(np.where(ok, E, np.nan)))
    c = X[i0].copy()
    hw = half_width or {'r_OO': 0.2, 'a_COO': 20.0, 'r_CO': 0.15}
    m = ok & (E - E[i0] <= window_eV)
    box = {}
    for j, n in enumerate(ref.coord_names):
        h = float(hw.get(n, 3 * _SCALE.get(n, 1.0)))
        m &= np.abs(X[:, j] - c[j]) <= h
        box[n] = [c[j] - h, c[j] + h]
    s = np.array([_SCALE.get(n, 1.0) for n in ref.coord_names])
    pw = np.array(_powers(len(ref.coord_names), order))
    z = (X[m] - c) / s
    A = np.prod(z[:, None, :] ** pw[None], 2)
    coef, *_ = np.linalg.lstsq(A, E[m], rcond=None)
    res = A @ coef - E[m]
    # the box actually covered by the data used
    box = {n: [float(X[m, j].min()), float(X[m, j].max())] for j, n in enumerate(ref.coord_names)}
    info = {'order': order, 'window_eV': window_eV, 'n_points': int(m.sum()), 'n_terms': int(len(pw)),
            'residual_rms_meV': float(np.sqrt(np.mean(res ** 2)) * 1000), 'box': box}
    model = PolyWell(ref.coord_names, c, s, pw, coef, box,
                     meta={'coord_names': ref.coord_names, 'fixed_coordinates': ref.meta.get('fixed', {}),
                           'experiment': f'polynomial well fit, order {order}, S0 within {window_eV} eV', **info})
    return model, info
