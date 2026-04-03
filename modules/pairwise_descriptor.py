#!/usr/bin/env python3
"""
Pairwise distance descriptor for ML-PES.

Uses all unique interatomic distances {r_ij, i<j} as the descriptor vector —
the same representation used by sGDML (Chmiela et al.).  No Z_i Z_j weighting,
so the Coulomb 1/r stiffness factor is reduced, but global KRR RMSE is slightly
worse (~0.23 vs 0.15 kcal/mol for MVKO) because chemical element information
is lost.

For MVKO (12 atoms): 12×11/2 = 66 features — same compact scale as Coulomb (78).

This module exists as a reference implementation for comparison with:
  - CoulombMatrixDescriptor (ml_pes.py): 78 features, best global KRR RMSE
  - ACSFDescriptor (acsf_descriptor.py): designed for NN, poor with global KRR
  - sGDML (sgdml package): uses same distance descriptor but with force learning

Key finding (2026-04-03): for global KRR + energy learning, Coulomb matrix
outperforms both pairwise distances and ACSF.  The stiffness artifact fix
requires force learning (sGDML) or per-atom neural networks (ANI-2x), not
just descriptor substitution.
"""

import numpy as np
from itertools import combinations

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(x, **kw): return x


class PairwiseDistanceDescriptor:
    """
    All unique interatomic distances as a descriptor vector.

    Matches the compute() / compute_batch() API of CoulombMatrixDescriptor.

    Feature count: N*(N-1)/2  (e.g. 66 for 12 atoms).
    Atom ordering must be consistent across all calls (same as Coulomb matrix
    convention already used throughout this project).
    """

    def __init__(self):
        self._pairs = None   # cached list of (i,j) pairs for a given N

    def _get_pairs(self, n):
        if self._pairs is None or len(self._pairs[0]) != n*(n-1)//2:
            i_idx, j_idx = np.triu_indices(n, k=1)
            self._pairs = (i_idx, j_idx)
        return self._pairs

    def compute(self, symbols: list, coords: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance descriptor for a single geometry.

        Parameters
        ----------
        symbols : list of str  (unused — included for API compatibility)
        coords  : np.ndarray, shape (N, 3), Ångström

        Returns
        -------
        np.ndarray, shape (N*(N-1)/2,)
        """
        n = len(coords)
        i_idx, j_idx = self._get_pairs(n)
        diff = coords[i_idx] - coords[j_idx]          # (n_pairs, 3)
        return np.linalg.norm(diff, axis=1)            # (n_pairs,)

    def compute_batch(self, symbols: list, coords_batch: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance descriptors for a batch of geometries.

        Parameters
        ----------
        symbols      : list of str  (unused — for API compatibility)
        coords_batch : np.ndarray, shape (M, N, 3)

        Returns
        -------
        np.ndarray, shape (M, N*(N-1)/2)
        """
        M, N, _ = coords_batch.shape
        i_idx, j_idx = self._get_pairs(N)
        diff = coords_batch[:, i_idx] - coords_batch[:, j_idx]   # (M, n_pairs, 3)
        return np.linalg.norm(diff, axis=2)                       # (M, n_pairs)
