"""The Frame interface: a physics-built matrix Omega(geometry) with named slots for learned corrections.

A frame answers three questions that a generic CDNN has to learn from data:
  * which structures (diabats) span the matrix, and how many,
  * how the matrix elements behave under symmetry and at the dissociation limits,
  * where learned corrections may enter without breaking those properties.

Each learned correction is a *slot*: a small function learned by a network. ``slots`` maps a slot name to
(n_in, n_out); ``prepare`` returns the per-geometry inputs for every slot as an (N, k, n_in) tensor, where k is
how many times the slot is evaluated per geometry (e.g. once per atom pair). ``assemble`` receives the slot outputs
(N, k, n_out) and builds the (N, n, n) matrix. With all slot outputs zero, ``assemble`` returns the pure physics
prior, so a model built on the frame starts from physics and learns only what physics misses.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import CubicSpline

torch.set_default_dtype(torch.float64)


class Frame:
    name = 'frame'
    n_diabats: int = 0
    diabat_labels: list[str] = []
    slots: dict[str, tuple[int, int]] = {}

    def prepare(self, coords: np.ndarray) -> dict:
        raise NotImplementedError

    def assemble(self, cache: dict, out: dict) -> torch.Tensor:
        raise NotImplementedError

    def zero_outputs(self, cache):
        return {k: torch.zeros(*cache['slot_in'][k].shape[:2], n_out) for k, (n_in, n_out) in self.slots.items()}

    def prior_matrix(self, coords) -> np.ndarray:
        c = self.prepare(np.atleast_2d(coords))
        return self.assemble(c, self.zero_outputs(c)).detach().numpy()

    def prior_energies(self, coords, n_states=None) -> np.ndarray:
        w = np.linalg.eigvalsh(self.prior_matrix(coords))
        return w[:, :n_states] if n_states else w

    def __repr__(self):
        return f'{type(self).__name__}(n_diabats={self.n_diabats}, slots={self.slots})'


class Curve1D:
    """Spline through a fragment curve; held at its last value beyond the data (the asymptote)."""

    def __init__(self, r, E):
        self.r = np.asarray(r, float)
        self.rmax = self.r[-1]
        self.sp = CubicSpline(self.r, np.asarray(E, float))
        self.asymptote = float(E[-1])

    def __call__(self, r):
        r = np.asarray(r, float)
        return np.where(r >= self.rmax, self.asymptote, self.sp(np.minimum(r, self.rmax)))


def eigvals_sorted(M: torch.Tensor) -> torch.Tensor:
    """Ascending eigenvalues of a batch of symmetric matrices. Closed form for 2x2 (stable at degeneracies)."""
    if M.shape[-1] == 2:
        m = (M[:, 0, 0] + M[:, 1, 1]) / 2
        d = (M[:, 0, 0] - M[:, 1, 1]) / 2
        s = torch.sqrt(d * d + M[:, 0, 1] ** 2 + 1e-12)
        return torch.stack([m - s, m + s], 1)
    return torch.linalg.eigvalsh(M)


def slater_overlap(r_ang, zeta=1.0):
    """Slater 1s-1s overlap at distance r (Angstrom) for exponent zeta (1/bohr); works on numpy or torch."""
    lib = torch if isinstance(r_ang, torch.Tensor) else np
    p = r_ang / 0.529177210903 * zeta
    return lib.exp(-p) * (1 + p + p * p / 3)


def switch(r, r_c=3.2, w=0.35):
    """Smooth switch from 1 (short range) to 0 (long range): 0.5 at r_c, below 1e-5 beyond r_c + 12 w."""
    lib = torch if isinstance(r, torch.Tensor) else np
    return 1.0 / (1.0 + lib.exp((r - r_c) / w))
