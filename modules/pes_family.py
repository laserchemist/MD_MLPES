#!/usr/bin/env python3
"""
pes_family.py — Multi-surface PES family with softmin blending
==============================================================
Manages a collection of trained ML-PES models, one per conformer or
electronic state, and blends their predictions smoothly using the
softmin (log-sum-exp) scheme so forces are continuous across surfaces.

Blending scheme
---------------
Given K surfaces with energies E_k and forces F_k at geometry R:

    Z        = Σ_k  exp(-β E_k)
    w_k      = exp(-β E_k) / Z            (softmin weights)
    E_blend  = -log(Z) / β  =  Σ_k w_k E_k  - entropy term
    F_blend  = Σ_k w_k F_k

where β = 1 / blend_width (kcal/mol).  When ΔE ≫ blend_width the lowest
surface dominates (approaches hard assignment).  Typical blend_width is
1–5 kcal/mol — smaller means sharper switching.

Conformer assignment
--------------------
`PESFamily.assign_conformer(coords)` returns the index of the surface with
lowest predicted energy (hard assignment, cheap).  Useful for bookkeeping
in long MD trajectories.

Typical use
-----------
    from modules.pes_family import PESFamily

    family = PESFamily.from_model_paths(
        symbols,
        {'s-cis': 'outputs/mvko_scis/mlpes_initial.pkl',
         's-trans': 'outputs/mvko_strans/mlpes_initial.pkl'},
        blend_width=3.0,       # kcal/mol
    )
    e, f = family.energy_and_forces(coords, method='fd', dx=0.005)
    label = family.assign_conformer(coords)

    # Save / reload
    family.save('family.pkl')
    family2 = PESFamily.load('family.pkl')
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HARTREE_TO_KCAL = 627.509474


# ── Internal helper ───────────────────────────────────────────────────────────

def _fd_forces(symbols: List[str], coords: np.ndarray,
               energy_fn, dx: float = 0.005) -> np.ndarray:
    """
    Central-difference forces from scalar energy_fn(symbols, coords).

    Parameters
    ----------
    dx : float
        Displacement in Angstrom (default 0.005 Å).

    Returns
    -------
    forces : (n_atoms, 3)  in Hartree/Angstrom
    """
    forces = np.zeros_like(coords)
    for a in range(len(symbols)):
        for j in range(3):
            coords_p = coords.copy(); coords_p[a, j] += dx
            coords_m = coords.copy(); coords_m[a, j] -= dx
            ep = energy_fn(symbols, coords_p)
            em = energy_fn(symbols, coords_m)
            forces[a, j] = -(ep - em) / (2.0 * dx)
    return forces


# ── ConformerPES ──────────────────────────────────────────────────────────────

class ConformerPES:
    """
    Thin wrapper around a trained MLPESTrainer for one conformer.

    Parameters
    ----------
    label : str
        Human-readable name (e.g. 's-cis', 'gauche').
    trainer : MLPESTrainer
        A fully trained model object.
    reference_energy : float
        Energy offset (Hartree) to align this surface to a common reference.
        Typically the absolute energy at the global minimum.  Set to 0 to
        leave energies as-is (default).
    """

    def __init__(self, label: str, trainer, reference_energy: float = 0.0):
        self.label = label
        self.trainer = trainer
        self.reference_energy = reference_energy

    # ------------------------------------------------------------------
    def predict_energy(self, symbols: List[str], coords: np.ndarray) -> float:
        """Return energy in Hartree (reference-corrected)."""
        return float(self.trainer.predict(symbols, coords)) - self.reference_energy

    def predict_batch(self, symbols: List[str],
                      coords_batch: np.ndarray) -> np.ndarray:
        """Return energies for a batch (Hartree, reference-corrected)."""
        return self.trainer.predict_batch(symbols, coords_batch) - self.reference_energy


# ── PESFamily ─────────────────────────────────────────────────────────────────

class PESFamily:
    """
    Collection of ConformerPES surfaces with softmin blending.

    Parameters
    ----------
    symbols : list of str
        Atomic symbols (must match all member models).
    members : dict[str, ConformerPES]
        Surface label → ConformerPES mapping.
    blend_width : float
        Blending width β⁻¹ in kcal/mol.  Larger = softer transitions.
    """

    def __init__(self,
                 symbols: List[str],
                 members: Dict[str, 'ConformerPES'],
                 blend_width: float = 3.0):
        self.symbols = symbols
        self.members = members        # OrderedDict preserved in Python 3.7+
        self.blend_width = blend_width
        self._labels = list(members.keys())

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_model_paths(cls,
                         symbols: List[str],
                         paths: Dict[str, str],
                         blend_width: float = 3.0,
                         reference_energies: Optional[Dict[str, float]] = None
                         ) -> 'PESFamily':
        """
        Build a PESFamily by loading pre-trained .pkl models.

        Parameters
        ----------
        symbols : list of str
        paths : dict[label → pkl_path]
        blend_width : float  (kcal/mol)
        reference_energies : dict[label → float] (Hartree), optional.
            If None, no offset is applied.
        """
        try:
            from .ml_pes import MLPESTrainer
        except ImportError:
            from ml_pes import MLPESTrainer

        ref = reference_energies or {}
        members = {}
        for label, pkl_path in paths.items():
            trainer = MLPESTrainer.load(str(pkl_path))
            members[label] = ConformerPES(label, trainer, ref.get(label, 0.0))
            print(f"  Loaded surface '{label}' from {pkl_path}")
        return cls(symbols, members, blend_width)

    @classmethod
    def from_trainers(cls,
                      symbols: List[str],
                      trainers: Dict[str, object],
                      blend_width: float = 3.0,
                      reference_energies: Optional[Dict[str, float]] = None
                      ) -> 'PESFamily':
        """Build from already-loaded MLPESTrainer objects."""
        ref = reference_energies or {}
        members = {lbl: ConformerPES(lbl, tr, ref.get(lbl, 0.0))
                   for lbl, tr in trainers.items()}
        return cls(symbols, members, blend_width)

    # ── Core blending ─────────────────────────────────────────────────────────

    def _weights(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute softmin weights and per-surface energies.

        Returns
        -------
        energies_ha : (K,)  Hartree, one per surface
        weights     : (K,)  softmin weights, sum to 1
        """
        beta = HARTREE_TO_KCAL / self.blend_width  # Hartree⁻¹
        energies_ha = np.array([m.predict_energy(self.symbols, coords)
                                 for m in self.members.values()])
        # Numerically stable softmin: subtract minimum before exp
        shifted = -beta * (energies_ha - energies_ha.min())
        exp_vals = np.exp(shifted)
        weights = exp_vals / exp_vals.sum()
        return energies_ha, weights

    def blend_energy(self, coords: np.ndarray) -> float:
        """
        Blended energy at a single geometry (Hartree).

        Returns the softmin log-sum-exp energy:
            E_blend = -log(Σ_k exp(-β E_k)) / β
        """
        beta = HARTREE_TO_KCAL / self.blend_width
        energies_ha = np.array([m.predict_energy(self.symbols, coords)
                                 for m in self.members.values()])
        e_min = energies_ha.min()
        log_z = np.log(np.exp(-beta * (energies_ha - e_min)).sum())
        return float(e_min - log_z / beta)

    def blend_energy_and_weights(self, coords: np.ndarray
                                 ) -> Tuple[float, np.ndarray]:
        """Return (E_blend, weights) for further use (e.g. force blending)."""
        energies_ha, weights = self._weights(coords)
        beta = HARTREE_TO_KCAL / self.blend_width
        e_min = energies_ha.min()
        shifted = -beta * (energies_ha - e_min)
        log_z = np.log(np.exp(shifted).sum())
        e_blend = float(e_min - log_z / beta)
        return e_blend, weights

    # ── Forces ────────────────────────────────────────────────────────────────

    def energy_and_forces(self,
                          coords: np.ndarray,
                          method: str = 'fd',
                          dx: float = 0.005
                          ) -> Tuple[float, np.ndarray]:
        """
        Compute blended energy (Hartree) and forces (Hartree/Å).

        Parameters
        ----------
        method : 'fd' | 'analytic'
            'fd'       — finite-difference on the blended energy surface
            'analytic' — blend forces from each surface analytically
                         (requires analytic_forces() on all member trainers)
        dx : float
            FD step in Angstrom (used only when method='fd').
        """
        if method == 'analytic':
            return self._energy_forces_analytic(coords)
        else:
            return self._energy_forces_fd(coords, dx)

    def _energy_forces_fd(self, coords: np.ndarray,
                          dx: float) -> Tuple[float, np.ndarray]:
        """FD forces on the blended scalar surface."""
        e = self.blend_energy(coords)
        forces = _fd_forces(self.symbols, coords, self.blend_energy, dx)
        return e, forces

    def _energy_forces_analytic(self, coords: np.ndarray
                                 ) -> Tuple[float, np.ndarray]:
        """
        Analytic blended forces: F_blend = Σ_k w_k F_k.

        Requires that each member trainer has analytic_forces() (via bakken
        MLPESDriver or a compatible wrapper that accepts coords).
        """
        e_blend, weights = self.blend_energy_and_weights(coords)
        forces_blend = np.zeros((len(self.symbols), 3))
        for w, member in zip(weights, self.members.values()):
            if hasattr(member.trainer, 'analytic_forces'):
                f_k = member.trainer.analytic_forces(coords)
            else:
                # Fall back to FD on this member's surface
                f_k = _fd_forces(self.symbols, coords,
                                  member.predict_energy, dx=0.005)
            forces_blend += w * f_k
        return e_blend, forces_blend

    # ── Assignment ────────────────────────────────────────────────────────────

    def assign_conformer(self, coords: np.ndarray) -> str:
        """Return the label of the lowest-energy surface (hard assignment)."""
        energies_ha, _ = self._weights(coords)
        idx = int(np.argmin(energies_ha))
        return self._labels[idx]

    def assign_conformer_batch(self, coords_batch: np.ndarray) -> List[str]:
        """Assign conformers for a batch of geometries."""
        labels = []
        for coords in coords_batch:
            labels.append(self.assign_conformer(coords))
        return labels

    def surface_energies(self, coords: np.ndarray) -> Dict[str, float]:
        """Return dict of {label: energy_hartree} for inspection."""
        return {lbl: m.predict_energy(self.symbols, coords)
                for lbl, m in self.members.items()}

    # ── Batch blending ────────────────────────────────────────────────────────

    def blend_energy_batch(self, coords_batch: np.ndarray) -> np.ndarray:
        """Blended energies for a batch (Hartree)."""
        beta = HARTREE_TO_KCAL / self.blend_width
        K = len(self.members)
        N = len(coords_batch)
        energies_ha = np.zeros((K, N))
        for k, member in enumerate(self.members.values()):
            energies_ha[k] = member.predict_batch(self.symbols, coords_batch)
        e_min = energies_ha.min(axis=0)                  # (N,)
        shifted = -beta * (energies_ha - e_min[np.newaxis, :])  # (K, N)
        log_z = np.log(np.exp(shifted).sum(axis=0))      # (N,)
        return e_min - log_z / beta

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Pickle the entire PESFamily."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"  PESFamily ({len(self.members)} surfaces) saved → {path}")

    @classmethod
    def load(cls, path: str) -> 'PESFamily':
        """Load a previously saved PESFamily."""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"  PESFamily loaded from {path} "
              f"(surfaces: {list(obj.members.keys())})")
        return obj

    def __repr__(self) -> str:
        return (f"PESFamily(surfaces={self._labels}, "
                f"blend_width={self.blend_width} kcal/mol)")
