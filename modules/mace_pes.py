#!/usr/bin/env python3
"""
mace_pes — MACE ML-PES backend
================================
Wraps a trained MACE model (.pt) in the bakken-compatible driver interface
so it drops into any workflow that uses MLPESDriver.

Why MACE over Coulomb+KRR or sGDML
------------------------------------
Both KRR-based approaches (Coulomb matrix or sGDML) fail to produce physical
normal-mode frequencies because the descriptor Jacobian introduces unphysical
second-derivative stiffness: C-H modes appear at 10,000–38,000 cm⁻¹ rather
than ~3,000 cm⁻¹. This is intrinsic to 1/r-based descriptors under RBF/Matérn
kernels and cannot be fixed by tuning hyperparameters.

MACE (Multi-Atomic Cluster Expansion) avoids this by:
  • Decomposing the total energy into a sum of local atomic contributions
    (no global kernel over all atoms)
  • Enforcing SO(3) equivariance — the force field respects rotational,
    translational, and permutation symmetry exactly
  • Training on forces natively (forces_weight=100 standard), giving a
    smooth, physically consistent PES curvature near equilibrium

The Hessian is computed via central FD on MACE's analytic forces (each force
evaluation is a single neural-network forward pass). With n_atoms=12, this
costs 72 forward passes per Hessian — fast on MPS/CPU.

Units
-----
MACE model outputs eV (energy) and eV/Å (forces).  This module converts to
the pipeline-standard Hartree / Hartree per Angstrom on every call.
All inputs/outputs of public methods use the pipeline units (Hartree, Å).
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Optional

# ── Unit conversions ──────────────────────────────────────────────────────────
HA_TO_EV       = 27.211386245988      # Hartree → electron-volt
EV_TO_HA       = 1.0 / HA_TO_EV      # eV → Hartree

# ── Physical constants (mirrors bakken.py) ────────────────────────────────────
ATOMIC_MASSES = {
    'H':  1.00794,  'He': 4.002602, 'C':  12.011,   'N':  14.007,
    'O':  15.999,   'F':  18.9984,  'S':  32.06,    'Cl': 35.453,
    'Br': 79.904,   'I':  126.904,
}


def _pick_device(requested: str) -> str:
    """Select compute device: 'auto' picks MPS (Apple Silicon) then CPU."""
    import torch
    if requested != 'auto':
        return requested
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


# =============================================================================
# MACEDriver — bakken.py-compatible driver
# =============================================================================

class MACEDriver:
    """
    bakken-compatible PES driver backed by a trained MACE model.

    Implements the same interface as MLPESDriver and SGDMLDriver:
        .energy(coords)           → float  [Hartree]
        .forces(coords)           → (n_atoms, 3)  [Hartree/Å]
        .analytic_forces(coords)  → (n_atoms, 3)  [Hartree/Å]  (MACE is analytic)
        .analytic_hessian(coords) → (3N, 3N)  [Hartree/Å²]    (FD on analytic forces)

    Parameters
    ----------
    model_path : str
        Path to a MACE .pt model file (output of mace_run_train).
    symbols : list of str
        Atomic symbols in consistent order (same as training data).
    device : str
        'cpu' (default — float64 models cannot load on MPS; CPU also faster for
        small-molecule batches). Pass 'cuda' for GPU inference on CUDA hardware.
        'auto' selects MPS/CUDA/CPU but will fail for float64 models on Apple Silicon.
    """

    def __init__(self, model_path: str,
                 symbols: Optional[List[str]] = None,
                 device: str = 'cpu'):
        from mace.calculators import MACECalculator

        self.model_path = str(model_path)
        self.device     = _pick_device(device)

        self.calc = MACECalculator(
            model_paths=self.model_path,
            device=self.device,
            energy_units_to_eV=1.0,   # model trained in eV
            length_units_to_A=1.0,    # model trained in Angstrom
        )

        # Load symbols from companion metadata file if not supplied
        if symbols is None:
            meta_path = Path(self.model_path).with_suffix('.symbols.pkl')
            if meta_path.exists():
                with open(meta_path, 'rb') as f:
                    symbols = pickle.load(f)
            else:
                raise ValueError(
                    f"symbols not provided and no companion file {meta_path}. "
                    "Pass symbols= explicitly or use train_mace_model.py which saves it."
                )

        self.symbols      = list(symbols)
        self.n_atoms      = len(self.symbols)
        self.masses       = np.array([ATOMIC_MASSES[s] for s in self.symbols])
        self._has_analytic = True  # MACE forces are analytic

        print(f"  MACEDriver: {self.n_atoms}-atom molecule on {self.device}  "
              f"({self.model_path})")

    # ── ASE interface ────────────────────────────────────────────────────────

    def _make_atoms(self, coords: np.ndarray):
        """Build an ASE Atoms object from coords (n_atoms, 3) Angstrom."""
        from ase import Atoms
        atoms = Atoms(
            symbols=self.symbols,
            positions=coords,
            pbc=False,
        )
        atoms.calc = self.calc
        return atoms

    # ── Energy / force prediction ─────────────────────────────────────────────

    def energy(self, coords: np.ndarray) -> float:
        """Predict PES energy in Hartree.  coords: (n_atoms, 3) Angstrom."""
        atoms = self._make_atoms(coords)
        return float(atoms.get_potential_energy()) * EV_TO_HA

    def forces(self, coords: np.ndarray, delta: float = 0.005) -> np.ndarray:
        """Analytic forces in Hartree/Å.  delta is ignored (API compatibility)."""
        atoms = self._make_atoms(coords)
        return atoms.get_forces() * EV_TO_HA   # eV/Å → Ha/Å

    def analytic_forces(self, coords: np.ndarray) -> np.ndarray:
        """Analytic forces in Hartree/Å."""
        return self.forces(coords)

    def analytic_hessian(self, coords: np.ndarray,
                         delta: float = 0.005) -> np.ndarray:
        """
        Hessian via central FD on analytic MACE forces.

        Cost: 6N force evaluations (each is one MACE forward pass).
        For n_atoms=12: 72 forward passes ≈ <1 s on MPS.
        Returns (3N, 3N) Hartree/Å² — symmetrised.
        """
        n3 = self.n_atoms * 3
        H  = np.zeros((n3, n3))
        r  = coords.flatten()
        for i in range(n3):
            rp = r.copy(); rp[i] += delta
            rm = r.copy(); rm[i] -= delta
            fp = self.analytic_forces(rp.reshape(-1, 3)).flatten()
            fm = self.analytic_forces(rm.reshape(-1, 3)).flatten()
            H[i] = -(fp - fm) / (2.0 * delta)
        return 0.5 * (H + H.T)


# =============================================================================
# Data conversion utilities
# =============================================================================

def npz_to_extxyz(npz_path: str,
                  xyz_path: str,
                  energy_cutoff_kcal: float = 15.0,
                  energy_key: str = 'REF_energy',
                  forces_key: str = 'REF_forces') -> int:
    """
    Convert our standard npz training data to MACE-compatible extxyz format.

    Energies and forces are converted to eV / eV/Å.
    Frames with dE > energy_cutoff_kcal above the minimum are excluded.
    Large simulation box (20 Å cube, pbc=False) is written for each frame.

    Returns number of frames written.
    """
    from ase import Atoms
    from ase.io import write

    data     = np.load(npz_path, allow_pickle=True)
    symbols  = data['symbols'].tolist()
    coords   = data['coordinates']   # (n, n_atoms, 3) Å
    energies = data['energies']      # (n,) Ha
    forces   = data['forces']        # (n, n_atoms, 3) Ha/Å

    # Energy filter — keep near-equilibrium frames
    e_min  = energies.min()
    e_rel  = (energies - e_min) * 627.509474   # kcal/mol
    mask   = e_rel < energy_cutoff_kcal
    n_kept = int(mask.sum())
    n_drop = len(energies) - n_kept
    print(f"  {n_kept} frames kept (dE < {energy_cutoff_kcal:.0f} kcal/mol), "
          f"{n_drop} dropped")

    coords_f   = coords[mask]
    energies_f = energies[mask] * HA_TO_EV    # Ha → eV
    forces_f   = forces[mask]  * HA_TO_EV     # Ha/Å → eV/Å

    frames = []
    for i in range(n_kept):
        atoms = Atoms(
            symbols=symbols,
            positions=coords_f[i],
            pbc=False,
            cell=[20.0, 20.0, 20.0],
        )
        atoms.info[energy_key] = float(energies_f[i])
        atoms.arrays[forces_key] = forces_f[i].astype(np.float64)
        frames.append(atoms)

    write(xyz_path, frames, format='extxyz')
    print(f"  Wrote {n_kept} frames → {xyz_path}")
    return n_kept
