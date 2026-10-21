"""Surfaces: Cartesian positions (Angstrom) -> the learned matrix Omega(R) (eV), differentiably.

Everything a dynamics code needs follows from Omega(R) and its derivative dOmega/dR (see ase_calc.py):
energies E_k = eigenvalues, forces F_k = -v_k^T (dOmega/dR) v_k, and couplings v_k^T (dOmega/dR) v_l.

A surface only has to supply `matrix(R)` for a single geometry, R a float64 torch tensor of shape (n_atoms, 3),
returning an (n, n) symmetric tensor built from R with torch operations (so autograd can differentiate it).
Any model with `matrix_t(coords)` (CDNN, FramedModel) plugs in through a coordinate map.
"""
from __future__ import annotations

import torch

torch.set_default_dtype(torch.float64)


def pair_distances(R):
    """All interatomic distances r_ij (i < j) in row-major order: (0,1), (0,2), ..., (n-2, n-1)."""
    n = R.shape[0]
    i, j = torch.triu_indices(n, n, 1)
    return torch.linalg.norm(R[i] - R[j], dim=1)


class Surface:
    n_states: int = 1
    n_atoms: int | None = None

    def matrix(self, R: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CoordinateSurface(Surface):
    """A model of chosen coordinates (internal coordinates, distances, ...) plus a differentiable map R -> coords.

    coord_fn(R) returns a 1-D tensor in the order the model was trained on. Reduced-dimensional models (e.g. the
    planar CH2OO models of the workflow, functions of r_OO, angle COO, r_CO) give forces only along the coordinates
    they know; every other degree of freedom feels no force, so hold those near fixed values with restraints
    (diabml.dynamics.reduced_restraints)."""

    def __init__(self, model, coord_fn, n_states=None, n_atoms=None):
        self.model, self.coord_fn = model, coord_fn
        self.n_states = n_states or model.n_states
        self.n_atoms = n_atoms

    def matrix(self, R):
        return self.model.matrix_t(self.coord_fn(R)[None, :])[0]


class DistanceSurface(CoordinateSurface):
    """Full-dimensional surface whose model inputs are all interatomic distances (pair_distances order).

    permutations: optional list of atom orderings that are physically equivalent (e.g. swapping the two H atoms of
    CH2OO: [[0, 1, 2, 3, 4], [0, 1, 2, 4, 3]]). The matrix is averaged over them, which makes energies, forces and
    couplings exactly invariant, provided the matrix's structures do not themselves relabel under the permutation
    (true for the CH2OO channel structures under H swap). If they do (as in H3), build the symmetry into the frame
    instead (see LondonFrame)."""

    def __init__(self, model, permutations=None, n_states=None):
        perms = permutations or [None]
        super().__init__(model, pair_distances, n_states, n_atoms=None if perms[0] is None else len(perms[0]))
        self.perms = [None if p is None else torch.as_tensor(p) for p in perms]

    def matrix(self, R):
        Ms = [self.model.matrix_t(pair_distances(R if p is None else R[p])[None, :])[0] for p in self.perms]
        return torch.stack(Ms).mean(0)
