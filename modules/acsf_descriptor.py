#!/usr/bin/env python3
"""
ACSF (Atom-Centered Symmetry Functions) descriptor for ML-PES.

Implements the same compute() / compute_batch() API as CoulombMatrixDescriptor
in ml_pes.py so it can be used as a drop-in replacement for the KRR trainer.

Descriptor strategy for a fixed single molecule (MVKO, 12 atoms):
  - Compute per-atom ACSF vectors via DScribe
  - Concatenate atomic vectors in fixed atom order → 1D descriptor per geometry
  - Total size: n_atoms × n_features_per_atom

This preserves per-atom chemical environment information without requiring
permutation invariance (atom ordering is fixed by the Coulomb matrix convention
already in use throughout this project).

ACSF parameter defaults (Behler 2011, organic molecules):
  G2 radial:  8 (η, Rs) pairs, Rs=0, η covering 0.5–36 Å⁻²
  G4 angular: 12 (η, λ, ζ) triples, η=[0.01, 0.1], λ=[1,-1], ζ=[1,2,4]
  R_cut: 6.0 Å (captures all bonded + close non-bonded pairs in MVKO)

For MVKO {H, C, O} (3 species):
  G2 features per atom: 8 × 3 species = 24
  G4 features per atom: 12 × 6 species-pairs = 72
  Total per atom: 96
  Total for 12 atoms: 1152

Hyperparameter guidance:
  The feature space is ~15× larger than Coulomb matrix (1152 vs 78).
  Start γ grid from 1e-5 to 1e-3 (expect optimal ~5e-5 to 3e-4).
  α grid: 1e-6 to 1e-3 (same as current KRR practice).

References:
  Behler & Parrinello, PRL 2007, 98, 146401
  Behler, JCP 2011, 134, 074106
  Himanen et al., CPC 2020, 247, 106949 (DScribe)
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(x, **kw): return x

try:
    from dscribe.descriptors import ACSF as DScribeACSF
    from ase import Atoms as ASEAtoms
    DSCRIBE_AVAILABLE = True
except ImportError:
    DSCRIBE_AVAILABLE = False

# Default ACSF parameter sets
# ─────────────────────────────────────────────────────────────────────────────
# Radial G2: (η [Å⁻²], Rs [Å]) pairs
# Covers near-bond (η large) through mid-range (η small) distances
_DEFAULT_G2 = [
    [0.5,  0.0],
    [1.0,  0.0],
    [2.0,  0.0],
    [3.5,  0.0],
    [6.0,  0.0],
    [10.0, 0.0],
    [18.0, 0.0],
    [36.0, 0.0],
]

# Angular G4: (η [Å⁻²], λ, ζ) triples
# λ=+1: cos-peak at θ=0 (collinear), λ=-1: peak at θ=π
# ζ: angular resolution (higher = narrower)
_DEFAULT_G4 = [
    [0.01, 1,  1], [0.01, -1,  1],
    [0.01, 1,  2], [0.01, -1,  2],
    [0.01, 1,  4], [0.01, -1,  4],
    [0.1,  1,  1], [0.1,  -1,  1],
    [0.1,  1,  2], [0.1,  -1,  2],
    [0.1,  1,  4], [0.1,  -1,  4],
]


class ACSFDescriptor:
    """
    ACSF descriptor with the same API as CoulombMatrixDescriptor.

    Parameters
    ----------
    species : list of str
        Element symbols present in the molecule (order irrelevant — DScribe
        handles species internally). Default: ['H', 'C', 'O'].
    r_cut : float
        Cutoff radius in Ångström. Default: 6.0.
    g2_params : list of [eta, Rs] or None
        Radial symmetry function parameters. Uses _DEFAULT_G2 if None.
    g4_params : list of [eta, lambda, zeta] or None
        Angular symmetry function parameters. Uses _DEFAULT_G4 if None.
    aggregate : str
        How to combine per-atom ACSF vectors into a single descriptor:
        - 'concatenate' : stack all atomic vectors → (N * n_feat,)
          Preserves per-atom identity. Requires fixed atom ordering.
          Descriptor size: 12 × 99 = 1188 for MVKO.
          NOTE: too large for KRR with ~800 training points → poor RMSE.
        - 'sum' : sum atomic vectors over all atoms → (n_feat,)
          Compact (~99 dim), permutation-invariant, good KRR conditioning.
          Loses per-atom identity but captures global environment well.
        - 'sum_by_species' : sum separately per element, then concatenate
          → (n_species * n_feat,). Best balance: 3 × 99 = 297 for MVKO.
          Permutation-invariant within each element. Recommended default.
        Default: 'sum_by_species'
    """

    def __init__(self, species=None, r_cut=6.0, g2_params=None, g4_params=None,
                 aggregate='sum_by_species'):
        if not DSCRIBE_AVAILABLE:
            raise ImportError(
                "dscribe and ase are required: pip install dscribe"
            )

        self.species   = sorted(species or ['H', 'C', 'O'])
        self.r_cut     = r_cut
        self.g2_params = g2_params if g2_params is not None else _DEFAULT_G2
        self.g4_params = g4_params if g4_params is not None else _DEFAULT_G4
        self.aggregate = aggregate

        if aggregate not in ('concatenate', 'sum', 'sum_by_species'):
            raise ValueError(f"aggregate must be 'concatenate', 'sum', or 'sum_by_species'; got {aggregate!r}")

        self._acsf = DScribeACSF(
            species   = self.species,
            r_cut     = self.r_cut,
            g2_params = self.g2_params,
            g4_params = self.g4_params,
            periodic  = False,
        )

        # Compute feature dimension on a dummy molecule
        dummy = ASEAtoms(
            symbols   = self.species,
            positions = np.eye(len(self.species)) * 1.5,
        )
        _dummy_out = self._acsf.create(dummy)
        self.n_features_per_atom = _dummy_out.shape[1]

        logger.info(
            f"ACSFDescriptor({aggregate}): {len(self.g2_params)} G2 + {len(self.g4_params)} G4, "
            f"R_cut={r_cut} Å, {self.n_features_per_atom} features/atom"
        )

    def _to_ase(self, symbols, coords):
        """Convert (symbols, coords_Å) to ASE Atoms."""
        return ASEAtoms(symbols=list(symbols), positions=coords)

    def _aggregate(self, per_atom: np.ndarray, symbols: list) -> np.ndarray:
        """
        Aggregate (N, n_feat) per-atom array to a 1D descriptor vector.

        per_atom : np.ndarray, shape (N, n_feat)
        symbols  : list of str, length N
        """
        if self.aggregate == 'concatenate':
            return per_atom.flatten()

        if self.aggregate == 'sum':
            return per_atom.sum(axis=0)

        # sum_by_species: sum per element, concatenate species sums
        parts = []
        for sp in self.species:
            idx = [i for i, s in enumerate(symbols) if s == sp]
            if idx:
                parts.append(per_atom[idx].sum(axis=0))
            else:
                parts.append(np.zeros(self.n_features_per_atom))
        return np.concatenate(parts)

    def compute(self, symbols: list, coords: np.ndarray) -> np.ndarray:
        """
        Compute ACSF descriptor for a single geometry.

        Parameters
        ----------
        symbols : list of str
            Atomic symbols (length N).
        coords : np.ndarray, shape (N, 3)
            Atomic coordinates in Ångström.

        Returns
        -------
        np.ndarray, 1D
            Aggregated descriptor vector. Length depends on `aggregate`:
            - 'concatenate'    : N × n_features_per_atom
            - 'sum'            : n_features_per_atom
            - 'sum_by_species' : n_species × n_features_per_atom
        """
        atoms = self._to_ase(symbols, coords)
        per_atom = self._acsf.create(atoms)          # (N, n_features_per_atom)
        return self._aggregate(per_atom, symbols)

    def compute_batch(self, symbols: list, coords_batch: np.ndarray) -> np.ndarray:
        """
        Compute ACSF descriptors for a batch of geometries.

        Parameters
        ----------
        symbols : list of str
            Atomic symbols (length N) — same for all frames.
        coords_batch : np.ndarray, shape (M, N, 3)
            Coordinates in Ångström for M frames.

        Returns
        -------
        np.ndarray, shape (M, descriptor_dim)
        """
        atoms_list = [self._to_ase(symbols, c) for c in coords_batch]

        # DScribe batch-compute
        result = self._acsf.create(atoms_list, n_jobs=1)  # n_jobs=1: avoid Apple Silicon OOM

        # result is (M, N, n_feat) or (N, n_feat) for single frame
        if result.ndim == 2:
            result = result[np.newaxis]  # (1, N, n_feat)

        M = result.shape[0]
        syms = list(symbols)
        return np.array([self._aggregate(result[i], syms) for i in range(M)])

    def describe(self):
        n_g2 = len(self.g2_params)
        n_g4 = len(self.g4_params)
        n_sp = len(self.species)
        n_pairs = n_sp * (n_sp + 1) // 2
        per_atom = n_g2 * n_sp + n_g4 * n_pairs + n_sp  # +n_sp for G1 terms DScribe adds
        if self.aggregate == 'concatenate':
            total = f"N_atoms × {self.n_features_per_atom}"
        elif self.aggregate == 'sum':
            total = str(self.n_features_per_atom)
        else:
            total = f"{n_sp} × {self.n_features_per_atom} = {n_sp * self.n_features_per_atom}"
        return (
            f"ACSF({n_g2} G2 × {n_sp} species = {n_g2*n_sp} radial, "
            f"{n_g4} G4 × {n_pairs} pairs = {n_g4*n_pairs} angular, "
            f"{self.n_features_per_atom} feat/atom, aggregate={self.aggregate!r}, "
            f"total={total})"
        )
