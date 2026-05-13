#!/usr/bin/env python3
"""
nm_pes.py — Full ML-PES using normal-mode coordinate KRR.

Why NM coordinates instead of Coulomb matrix?
----------------------------------------------
The Coulomb+RBF kernel has a fundamental stiffness artifact: its second
derivatives along Cartesian directions that don't move atoms far apart
are unphysically large.  This produces imaginary Hessian modes (spurious
negative curvature) and C-H frequencies at 5,000–15,000 cm⁻¹ instead of
~3,000 cm⁻¹, causing bond elongation during MD.

Normal-mode coordinates fix this:
  q = U_vib^T · M^{1/2} · (R − R_eq)   [sqrt(amu)·Bohr]

Properties of NM-KRR Hessian:
  H_q = Σ_i α_i K_i [4γ²Δq_i Δq_i^T − 2γ I]

At q=0 (reference equilibrium), Δq_i = −q_i (the training displacement):
  H_q(0) = Σ_i α_i K_i [4γ² q_i q_i^T − 2γ I]

For training data that spans all modes symmetrically (+ and − displacements),
the 4γ² outer-product term dominates and H_q ≻ 0 with the correct mode
structure.  The Cartesian Hessian is:

  H_cart = J^T · H_q · J   where J[3a+j, k] = U_vib[3a+j, k] · sqrt(m_a) · Å→Bohr

This has rank n_vib (30 for MVKO) with the 6 TR modes at exactly zero.

Classes
-------
NMKRRPESModel  — KRR for absolute energy in NM coordinates; analytic
                 forces and Hessian.  save()/load() use state-dict pickling
                 (avoids module-identity PicklingError).

NMPESDriver    — Wraps NMKRRPESModel with the same interface as MLPESDriver
                 (energy, forces, analytic_hessian, symbols, masses) so it
                 can be passed to run_md(), minimize_geometry(), etc.

Usage in ir_md_spectrum.py
--------------------------
  # New flag  --nm-pes-model path/to/mlpes_wB97X_nm.pkl
  # replaces  --model          path/to/mlpes_wB97X.pkl  (Coulomb-based)
  driver = NMPESDriver(nm_pes_model_path)
"""

import pickle
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── unit conversions ─────────────────────────────────────────────────────────
ANGSTROM_TO_BOHR = 1.88972612463
BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR
HARTREE_TO_KCAL  = 627.509474
FREQ_CONV        = 5140.48   # cm⁻¹ / sqrt(Ha / (Bohr² · amu))

# ── atomic masses (amu) ──────────────────────────────────────────────────────
ATOMIC_MASSES: dict = {
    'H': 1.00794, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'F': 18.998, 'S': 32.06, 'Cl': 35.453, 'Br': 79.904,
}


# =============================================================================
# NMKRRPESModel — kernel ridge regression for absolute energy in q-space
# =============================================================================

class NMKRRPESModel:
    """
    Full ML-PES in normal-mode coordinate space using KRR.

    E(R) = k(q(R))^T β + E_ref
    where q = U_vib^T · M^{1/2} · (R − R_eq) and E_ref is the mean training energy.

    Analytic forces and Hessian avoid the Coulomb+RBF stiffness artifact.
    """

    # Physical constants used for the harmonic wall
    _KB_HA_PER_K = 3.1668114e-6   # Boltzmann constant, Hartree/K
    _T_WALL      = 300.0          # Reference temperature for wall calibration (K)

    def __init__(
        self,
        *,
        eq_coords_ang: np.ndarray,   # (N, 3) reference geometry, Angstrom
        U_vib: np.ndarray,           # (3N, n_vib) mass-weighted NM eigenvectors
        sqrt_mass: np.ndarray,       # (3N,) sqrt(amu), repeated per atom
        freqs_vib: np.ndarray,       # (n_vib,) cm⁻¹  (diagnostic)
        symbols: List[str],
        gamma: float,                # RBF width  (1 / (amu·Bohr²))
        alpha_reg: float,            # KRR regularisation
        X_train_q: np.ndarray,       # (M, n_vib) training NM coordinates
        y_train_ha: np.ndarray,      # (M,) training energies, Hartree
        cv_rmse_kcal: float = None,
        wall_factor: float = 3.5,    # harmonic wall threshold per mode (see wall_mode)
        wall_stiffness: float = 2.0, # wall force constant = wall_stiffness × λ_k (mode eigenvalue)
        wall_mode: str = 'thermal',  # 'thermal': wall_factor × a_therm(T_WALL)
                                     # 'data':    wall_factor × max(|q_train|) per mode
        coord_scale: np.ndarray = None,  # per-mode scale applied before kernel: q_s = q / coord_scale
                                         # Use a_therm(T) to equalise all modes (recommended).
                                         # None → no scaling (raw sqrt(amu)·Bohr, legacy behaviour).
    ):
        self.eq_coords_ang = np.asarray(eq_coords_ang, dtype=float)
        self.U_vib         = np.asarray(U_vib,         dtype=float)
        self.sqrt_mass     = np.asarray(sqrt_mass,     dtype=float)
        self.freqs_vib     = np.asarray(freqs_vib,     dtype=float)
        self.symbols       = list(symbols)
        self.wall_factor   = float(wall_factor)
        self.wall_stiffness = float(wall_stiffness)
        self.wall_mode     = str(wall_mode)
        self.gamma         = float(gamma)
        self.alpha_reg     = float(alpha_reg)
        self.X_train_q     = np.asarray(X_train_q, dtype=float)
        self.y_train_ha    = np.asarray(y_train_ha, dtype=float)
        self.cv_rmse_kcal  = cv_rmse_kcal

        n_vib_local = self.X_train_q.shape[1]
        if coord_scale is None:
            self.coord_scale = np.ones(n_vib_local)
        else:
            self.coord_scale = np.asarray(coord_scale, dtype=float)
        assert self.coord_scale.shape == (n_vib_local,), "coord_scale must be (n_vib,)"

        # Scaled training coordinates used in all kernel computations
        # q_s = q / coord_scale  (dimensionless when coord_scale = a_therm)
        self._X_train_qs = self.X_train_q / self.coord_scale[None, :]  # (M, n_vib)

        # Centre y to avoid scale issues with absolute Hartree values
        self._y_mean = float(np.mean(self.y_train_ha))
        y_centred    = self.y_train_ha - self._y_mean

        # Solve (K + α I) β = y_centred  for dual coefficients β
        K = self._kernel_matrix(self._X_train_qs, self._X_train_qs)
        K[np.diag_indices_from(K)] += alpha_reg
        self._alpha_vec = np.linalg.solve(K, y_centred)   # (M,)

        self.n_atoms = len(symbols)
        self.n_vib   = U_vib.shape[1]

        # Jacobian of q w.r.t. Cartesian coords (fixed for a given eq geometry):
        #   J[3a+j, k] = U_vib[3a+j, k] · sqrt_mass[3a+j] · ANGSTROM_TO_BOHR
        #   shape: (3N, n_vib)
        # Converting from Å to Bohr because q is in sqrt(amu)·Bohr
        self._J = self.U_vib * (self.sqrt_mass[:, None] * ANGSTROM_TO_BOHR)  # (3N, n_vib)

        # ── Harmonic wall ──────────────────────────────────────────────────────
        # Activates per-mode when |q_k| > q_wall_k.
        # Prevents extrapolation into flat KRR regions beyond training data.
        # E_wall = Σ_k (λ_k/2) × wall_stiffness × max(0, |q_k| - q_wall_k)²
        # where λ_k = (FREQ_CONV)⁻² × ω_k² is the NM eigenvalue (Ha/(amu·Bohr²)).
        self._eigenvalues_ha = (self.freqs_vib / FREQ_CONV) ** 2   # (n_vib,) Ha/(amu·Bohr²)
        if wall_mode == 'thermal':
            # q_wall = wall_factor × a_therm(T_WALL)  — physically motivated.
            # a_therm = sqrt(2 k_B T / λ_k) is the classical 1σ amplitude at T_WALL.
            # With factor=3.5: C-H ZPE sits at 77-81% of wall (no clamping),
            #                   soft torsions are confined to ~3.5× thermal motion.
            _a_therm = np.sqrt(2 * self._KB_HA_PER_K * self._T_WALL / self._eigenvalues_ha)
            self._q_wall = wall_factor * _a_therm                   # (n_vib,)
        else:
            # 'data': wall_factor × max per-mode training-data amplitude (legacy)
            q_max_train = np.max(np.abs(self.X_train_q), axis=0)    # (n_vib,)
            self._q_wall = wall_factor * q_max_train                 # (n_vib,)
        # Wall spring constant (same units as KRR output: Ha/q²)
        self._wall_k = wall_stiffness * self._eigenvalues_ha        # (n_vib,)

    # ── projection ─────────────────────────────────────────────────────────

    def project(self, coords_ang: np.ndarray) -> np.ndarray:
        """
        Project Cartesian geometry to NM coordinates.

        Args:
            coords_ang : (N, 3) Angstrom
        Returns:
            q : (n_vib,) sqrt(amu)·Bohr
        """
        delta_ang  = np.asarray(coords_ang, dtype=float) - self.eq_coords_ang
        delta_bohr = delta_ang.flatten() * ANGSTROM_TO_BOHR
        delta_mw   = delta_bohr * self.sqrt_mass
        return self.U_vib.T @ delta_mw

    def project_batch(self, coords_ang: np.ndarray) -> np.ndarray:
        """Project a batch (M, N, 3) → (M, n_vib)."""
        coords_ang = np.asarray(coords_ang, dtype=float)
        delta_ang  = coords_ang - self.eq_coords_ang[None, :, :]    # (M, N, 3)
        delta_bohr = delta_ang.reshape(len(coords_ang), -1) * ANGSTROM_TO_BOHR  # (M, 3N)
        delta_mw   = delta_bohr * self.sqrt_mass[None, :]           # (M, 3N)
        return delta_mw @ self.U_vib                                 # (M, n_vib)

    # ── kernel helpers ──────────────────────────────────────────────────────

    def _kernel_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """RBF kernel K[i,j] = exp(−γ ||A_i − B_j||²). A:(m,d), B:(n,d)→(m,n)."""
        A2 = np.sum(A ** 2, axis=1, keepdims=True)   # (m,1)
        B2 = np.sum(B ** 2, axis=1, keepdims=True)   # (n,1)
        return np.exp(-self.gamma * (A2 + B2.T - 2.0 * A @ B.T))

    def _kernel_vec(self, q_s: np.ndarray) -> np.ndarray:
        """K_i = exp(−γ ||X_train_s[i] − q_s||²).  q_s must already be scaled. Returns (M,)."""
        diff2 = np.sum((self._X_train_qs - q_s[None, :]) ** 2, axis=1)
        return np.exp(-self.gamma * diff2)

    # ── prediction ─────────────────────────────────────────────────────────

    def _wall_energy(self, q: np.ndarray) -> float:
        """Harmonic wall energy (Hartree). Zero inside training range, quadratic outside."""
        excess = np.maximum(0.0, np.abs(q) - self._q_wall)   # (n_vib,)
        return float(0.5 * np.dot(self._wall_k, excess ** 2))

    def _wall_force_q(self, q: np.ndarray) -> np.ndarray:
        """Gradient of wall energy w.r.t. q (Ha / sqrt(amu)·Bohr)."""
        excess = np.maximum(0.0, np.abs(q) - self._q_wall)   # (n_vib,)
        return self._wall_k * excess * np.sign(q)             # (n_vib,)

    def predict_ha(self, q: np.ndarray) -> float:
        """Predict energy (Hartree) for one NM-coordinate vector q (unscaled sqrt(amu)·Bohr)."""
        q_s = q / self.coord_scale
        k = self._kernel_vec(q_s)
        return float(np.dot(self._alpha_vec, k)) + self._y_mean + self._wall_energy(q)

    def predict(self, symbols: list, coords_ang: np.ndarray) -> float:
        """
        Predict energy (Hartree) for one geometry.
        Same signature as MLPESTrainer.predict() for compatibility.
        """
        q = self.project(coords_ang)
        return self.predict_ha(q)

    def predict_forces_ha_ang(self, coords_ang: np.ndarray) -> np.ndarray:
        """
        Analytic forces (Hartree / Angstrom) via chain rule.

        F_{a,j} = − ∂E/∂R_{a,j} = − Σ_k (∂E/∂q_k) · J[3a+j, k]

        ∂E/∂q_k = Σ_i α_i K_i · (−2γ)(q_k − q_ik)
                = −2γ Σ_i α_i K_i (q_k − q_ik)

        Returns:
            F : (N, 3) Hartree/Angstrom
        """
        q   = self.project(coords_ang)
        q_s = q / self.coord_scale                      # scaled coordinates
        k   = self._kernel_vec(q_s)                     # (M,)
        g   = self._alpha_vec * k                       # (M,)
        # dE/dq_s_k = −2γ Σ_i g_i (q_s_k − q_s_ik)
        Δqs    = q_s[None, :] - self._X_train_qs        # (M, n_vib)  q_s − q_s_i
        dEdqs  = -2.0 * self.gamma * (g[:, None] * Δqs).sum(axis=0)  # (n_vib,)
        # Chain rule: dE/dq = dE/dq_s / coord_scale
        dEdq  = dEdqs / self.coord_scale
        dEdq += self._wall_force_q(q)                   # add wall gradient (in q space)
        # F_{3a+j} = − dEdq · J[3a+j, :]   →  (3N,)
        F_flat = -(self._J @ dEdq)                      # (3N,)
        return F_flat.reshape(self.n_atoms, 3)

    def predict_hessian_q(self, q: np.ndarray) -> np.ndarray:
        """
        Analytic Hessian in NM coordinate space (Hartree / (amu·Bohr²)).

        H_q[k,l] = Σ_i α_i K_i [4γ² (q_s_k−q_s_ik)(q_s_l−q_s_il) − 2γ δ_{kl}]
                   / (coord_scale_k · coord_scale_l)

        (The 1/(s_k·s_l) factor transforms the q_s Hessian back to the q Hessian
        via the chain rule d²E/dq_k dq_l = H_qs[k,l] / (s_k s_l).)

        Returns:
            H_q : (n_vib, n_vib) in Ha / (amu·Bohr²)
        """
        q_s = q / self.coord_scale
        k   = self._kernel_vec(q_s)                     # (M,)
        g   = self._alpha_vec * k                       # (M,)
        Δqs = q_s[None, :] - self._X_train_qs           # (M, n_vib)
        # H_qs: 4γ² Σ_i g_i Δqs_i Δqs_i^T  −  2γ δ_{kl} Σ_i g_i
        H_qs = 4.0 * self.gamma ** 2 * (g[:, None, None] * Δqs[:, :, None] * Δqs[:, None, :]).sum(0)
        H_qs -= 2.0 * self.gamma * np.sum(g) * np.eye(self.n_vib)
        # Transform to unscaled q-space
        s_outer = np.outer(self.coord_scale, self.coord_scale)   # (n_vib, n_vib)
        return H_qs / s_outer

    def predict_hessian_cart_ha_ang2(self, coords_ang: np.ndarray) -> np.ndarray:
        """
        Analytic Cartesian Hessian (Hartree / Angstrom²).

        H_cart = J · H_q · J^T    [shape (3N, 3N)]

        Rank n_vib; the 6 TR modes have exactly zero eigenvalue.
        """
        q   = self.project(coords_ang)
        H_q = self.predict_hessian_q(q)                 # (n_vib, n_vib)
        return self._J @ H_q @ self._J.T               # (3N, 3N)

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save as a plain state dict (avoids module-identity PicklingError)."""
        state = {
            'eq_coords_ang':  self.eq_coords_ang,
            'U_vib':          self.U_vib,
            'sqrt_mass':      self.sqrt_mass,
            'freqs_vib':      self.freqs_vib,
            'symbols':        self.symbols,
            'gamma':          self.gamma,
            'alpha_reg':      self.alpha_reg,
            'X_train_q':      self.X_train_q,
            'y_train_ha':     self.y_train_ha,
            'cv_rmse_kcal':   self.cv_rmse_kcal,
            'wall_factor':    self.wall_factor,
            'wall_stiffness': self.wall_stiffness,
            'wall_mode':      self.wall_mode,
            'coord_scale':    self.coord_scale,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=4)

    @classmethod
    def load(cls, path: str) -> 'NMKRRPESModel':
        """Load from path — supports both state-dict and object pickles."""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if isinstance(obj, cls):
            return obj
        # state dict — supply defaults for params added after initial release
        obj.setdefault('wall_factor', 4.0)
        obj.setdefault('wall_stiffness', 2.0)
        obj.setdefault('wall_mode', 'data')    # old saves used data-based wall
        obj.setdefault('coord_scale', None)    # old saves used unscaled coordinates
        return cls(**obj)


# =============================================================================
# NMPESDriver — MLPESDriver-compatible wrapper for NMKRRPESModel
# =============================================================================

class NMPESDriver:
    """
    Wraps NMKRRPESModel with the same interface as MLPESDriver so that
    it can be passed directly to run_md(), minimize_geometry(), and
    compute_mlpes_normal_modes().

    Analytic forces replace finite-difference: ~30× faster and exact.
    Analytic Hessian used when compute_mlpes_normal_modes() is called.
    """

    def __init__(self, model_path: str,
                 bond_wall_factor: float = 1.9,
                 bond_wall_stiffness: float = 0.2):
        """
        Args:
            model_path          : path to NMKRRPESModel pickle
            bond_wall_factor    : repulsive wall starts at this × eq bond length.
                                  Default 1.9 (190 % of equilibrium).  Targets
                                  heavy-atom (non-H) bonds only.  Prevents ML-PES
                                  flat-region wandering into dissociative geometries
                                  during 300 K IR simulations without retraining.
            bond_wall_stiffness : wall spring constant (Ha/Å²).  0.2 Ha/Å²
                                  gives ~1.6 kcal/mol at 0.1 Å past cutoff.
        """
        self._model        = NMKRRPESModel.load(model_path)
        self.symbols       = self._model.symbols
        self.n_atoms       = len(self.symbols)
        self.masses        = np.array([ATOMIC_MASSES[s] for s in self.symbols])
        self._has_analytic = True   # enables compute_mlpes_normal_modes analytic path
        print(f"  NMPESDriver loaded: {model_path}")
        print(f"    n_atoms={self.n_atoms}  n_vib={self._model.n_vib}  "
              f"γ={self._model.gamma}  α={self._model.alpha_reg}")
        if self._model.cv_rmse_kcal is not None:
            print(f"    LOO-CV RMSE: {self._model.cv_rmse_kcal:.4f} kcal/mol")

        # ── Bond-distance wall (heavy atoms only) ─────────────────────────────
        # Adds a soft harmonic repulsion when heavy-atom bonds elongate beyond
        # bond_wall_factor × eq_length.  This prevents flat KRR extrapolation
        # from allowing unphysical dissociation at 300K.
        self._bond_wall_k   = float(bond_wall_stiffness)   # Ha/Å²
        self._bond_pairs    = []  # list of (i, j, d_eq, d_cut)
        eq_coords = self._model.eq_coords_ang               # (N, 3) Ang
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                if self.symbols[i] == 'H' or self.symbols[j] == 'H':
                    continue
                d0 = float(np.linalg.norm(eq_coords[i] - eq_coords[j]))
                if d0 < 2.0:   # covalent heavy-atom bond threshold (Å)
                    self._bond_pairs.append((i, j, d0, bond_wall_factor * d0))
        if self._bond_pairs:
            print(f"    Bond wall: {len(self._bond_pairs)} heavy-atom bonds, "
                  f"cutoff={bond_wall_factor:.2f}×d_eq, k={bond_wall_stiffness} Ha/Å²")
            for i, j, d0, dc in self._bond_pairs:
                print(f"      {self.symbols[i]}{i}-{self.symbols[j]}{j}: "
                      f"d_eq={d0:.3f}Å, d_cut={dc:.3f}Å")

    def _bond_wall_energy(self, coords: np.ndarray) -> float:
        """Harmonic repulsion energy (Hartree) for elongated heavy-atom bonds."""
        E = 0.0
        for i, j, _, d_cut in self._bond_pairs:
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d > d_cut:
                E += 0.5 * self._bond_wall_k * (d - d_cut) ** 2
        return E

    def _bond_wall_forces(self, coords: np.ndarray) -> np.ndarray:
        """Force contribution (Ha/Å) from bond-distance wall.

        E = 0.5 k (d - d_cut)²  for d > d_cut
        F_i = −∂E/∂r_i = +k (d − d_cut) · (r_j − r_i) / d   (pulls i toward j)
        F_j = −∂E/∂r_j = −k (d − d_cut) · (r_j − r_i) / d   (pulls j toward i)
        """
        F = np.zeros_like(coords)
        for i, j, _, d_cut in self._bond_pairs:
            diff = coords[j] - coords[i]            # vector i→j
            d    = float(np.linalg.norm(diff))
            if d > d_cut and d > 1e-6:
                f_mag = self._bond_wall_k * (d - d_cut) / d   # Ha/Å per unit vector
                F[i] += f_mag * diff                # pulls i toward j
                F[j] -= f_mag * diff                # pulls j toward i
        return F

    def energy(self, coords: np.ndarray) -> float:
        """Predict ML-PES energy + bond-wall (Hartree)."""
        return self._model.predict(self.symbols, coords) + self._bond_wall_energy(coords)

    def forces(self, coords: np.ndarray, delta: float = None) -> np.ndarray:
        """
        Analytic forces (Hartree / Angstrom).
        The `delta` arg is accepted but ignored (kept for MLPESDriver API compat).
        """
        return self._model.predict_forces_ha_ang(coords) + self._bond_wall_forces(coords)

    def analytic_forces(self, coords: np.ndarray) -> np.ndarray:
        """Alias for forces(); provided for explicit naming."""
        return self._model.predict_forces_ha_ang(coords)

    def analytic_hessian(self, coords: np.ndarray) -> np.ndarray:
        """
        Analytic Cartesian Hessian (Hartree / Angstrom²), shape (3N, 3N).

        Used by compute_mlpes_normal_modes() and ZPE initialisation.
        """
        return self._model.predict_hessian_cart_ha_ang2(coords)

    def nm_frequencies(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute NM frequencies (cm⁻¹) at the given geometry using the
        analytic Hessian.  Negative values indicate imaginary modes.

        Unit conversion: H [Ha/Å²] → H [Ha/Bohr²] requires dividing by
        ANG_TO_BOHR² (since 1 Å = ANG_TO_BOHR Bohr, so 1 Ha/Å² = 1/ANG_TO_BOHR² Ha/Bohr²).
        """
        H_ang2        = self.analytic_hessian(coords)    # (3N, 3N) Ha/Å²
        H_bohr2       = H_ang2 / (ANGSTROM_TO_BOHR ** 2)  # Ha/Bohr²
        sqrt_mass_rep = np.repeat(self.masses, 3)         # (3N,)
        H_mw          = H_bohr2 / np.outer(sqrt_mass_rep, sqrt_mass_rep)
        evals, _      = np.linalg.eigh(H_mw)
        # FREQ_CONV = 5140.48 cm⁻¹ / sqrt(Ha/(Bohr²·amu))
        freqs = np.where(evals >= 0,
                         FREQ_CONV * np.sqrt(evals),
                         -FREQ_CONV * np.sqrt(-evals))
        return freqs

    @staticmethod
    def load_model(model_path: str) -> 'NMPESDriver':
        """Factory method — same name used in ir_md_spectrum.py."""
        return NMPESDriver(model_path)


# =============================================================================
# NMDipoleSurface — KRR dipole surface in normal-mode coordinate space
# =============================================================================

class NMDipoleSurface:
    """
    Three-component KRR dipole surface using NM coordinates q as descriptors.

    Overcomes the R²≈0.91 Coulomb+KRR ceiling:
    - Coulomb Z(H)=1 → near-zero sensitivity to C-H displacement
    - NM coordinate q_k is non-zero whenever mode k is displaced, giving
      direct C-H stretch (modes 25-30 for MVKO) sensitivity to the kernel.

    Hyperparameters selected by analytic LOO-CV (median heuristic for γ_center).
    LOO formula: e_i^LOO = β_i / (K+αI)^{-1}_{ii}   O(n^2) extra after Cholesky.

    Analytic dipole derivatives ∂μ_α/∂q_k identify IR-active modes directly.
    """

    _KB_HA_PER_K = 3.1668114e-6

    def __init__(
        self,
        *,
        eq_coords_ang: np.ndarray,
        U_vib: np.ndarray,
        sqrt_mass: np.ndarray,
        freqs_vib: np.ndarray,
        symbols: List[str],
        coord_scale: Optional[np.ndarray] = None,
    ):
        self.eq_coords_ang = np.asarray(eq_coords_ang, dtype=float)
        self.U_vib         = np.asarray(U_vib,         dtype=float)
        self.sqrt_mass     = np.asarray(sqrt_mass,     dtype=float)
        self.freqs_vib     = np.asarray(freqs_vib,     dtype=float)
        self.symbols       = list(symbols)
        self.n_atoms       = len(symbols)
        self.n_vib         = U_vib.shape[1]

        if coord_scale is None:
            self.coord_scale = np.ones(self.n_vib)
        else:
            self.coord_scale = np.asarray(coord_scale, dtype=float)

        # Jacobian (same as NMKRRPESModel): J[3a+j, k] = U[3a+j,k]·√m_a·Å→Bohr
        self._J = self.U_vib * (self.sqrt_mass[:, None] * ANGSTROM_TO_BOHR)  # (3N, n_vib)

        # Training state (populated by fit())
        self.gamma        = None
        self.alpha_reg    = None
        self._X_train_qs  = None  # (M, n_vib) scaled training coordinates
        self._beta        = None  # (M, 3) dual coefficients
        self._mu_mean     = None  # (3,) mean dipole (removed before fitting)
        self.cv_rmse_D    = None
        self.train_rmse_D = None
        self.test_rmse_D  = None
        self.r2_test      = None

    # ── projection ─────────────────────────────────────────────────────────

    def project(self, coords_ang: np.ndarray) -> np.ndarray:
        """Cartesian (N, 3) Å → NM coordinates (n_vib,) sqrt(amu)·Bohr."""
        delta_ang  = np.asarray(coords_ang, dtype=float) - self.eq_coords_ang
        delta_bohr = delta_ang.flatten() * ANGSTROM_TO_BOHR
        delta_mw   = delta_bohr * self.sqrt_mass
        return self.U_vib.T @ delta_mw

    def project_batch(self, coords_ang: np.ndarray) -> np.ndarray:
        """Batch projection (M, N, 3) → (M, n_vib)."""
        coords_ang = np.asarray(coords_ang, dtype=float)
        delta_ang  = coords_ang - self.eq_coords_ang[None, :, :]
        delta_bohr = delta_ang.reshape(len(coords_ang), -1) * ANGSTROM_TO_BOHR
        delta_mw   = delta_bohr * self.sqrt_mass[None, :]
        return delta_mw @ self.U_vib

    # ── kernel helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _rbf(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
        """RBF kernel K[i,j] = exp(−γ||A_i−B_j||²). A:(m,d), B:(n,d)→(m,n)."""
        A2 = np.sum(A ** 2, axis=1, keepdims=True)
        B2 = np.sum(B ** 2, axis=1, keepdims=True)
        return np.exp(-gamma * (A2 + B2.T - 2.0 * A @ B.T))

    @staticmethod
    def _loo_cv_rmse(K: np.ndarray, alpha_reg: float, y: np.ndarray) -> float:
        """
        Analytic LOO-CV RMSE for KRR with kernel matrix K and regularization α.

        Formula: e_i^LOO = β_i / (K+αI)^{-1}_{ii}
        where β = (K+αI)^{-1} y.

        Derivation:
          H = K(K+αI)^{-1} = I − α(K+αI)^{-1}   →   H_{ii} = 1 − α(A^{-1})_{ii}
          In-sample residual: r_i = (I−H)y = α β
          LOO error: r_i/(1−H_{ii}) = αβ_i / (α(A^{-1})_{ii}) = β_i/(A^{-1})_{ii}

        (A^{-1})_{ii} = ||col_i of L^{-1}||²  where K+αI = LL^T (Cholesky).
        y may be (n,) or (n, d); returned RMSE averages over all elements.
        """
        n = len(y)
        A = K.copy()
        A[np.diag_indices(n)] += alpha_reg
        try:
            L = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            return np.inf
        rhs = y if y.ndim == 2 else y[:, None]
        beta = np.linalg.solve(L.T, np.linalg.solve(L, rhs))  # (n, d)
        L_inv_cols = np.linalg.solve(L, np.eye(n))            # (n, n)
        A_inv_diag = np.sum(L_inv_cols ** 2, axis=0)          # (n,)
        loo_err    = beta / A_inv_diag[:, None]                # (n, d)
        return float(np.sqrt(np.mean(loo_err ** 2)))

    # ── training ────────────────────────────────────────────────────────────

    def fit(
        self,
        coords_ang: np.ndarray,    # (M, N, 3) training geometries
        dipoles_D:  np.ndarray,    # (M, 3) training dipoles, Debye
        test_fraction: float = 0.15,
        random_seed:   int   = 42,
        verbose:       bool  = True,
    ) -> 'NMDipoleSurface':
        """
        Fit NM-coordinate KRR dipole surface with analytic LOO-CV γ/α selection.

        Steps:
        1. Project coords → scaled NM space.
        2. Build γ grid from median pairwise distance (data-adaptive).
        3. For each (γ, α): analytic LOO-CV RMSE (O(n²) per γ after Cholesky).
        4. Refit on full training set with best (γ, α).
        5. Report train/test RMSE and R².
        """
        import math

        coords_ang = np.asarray(coords_ang, dtype=float)
        dipoles_D  = np.asarray(dipoles_D,  dtype=float)
        M = len(coords_ang)

        X_q  = self.project_batch(coords_ang)               # (M, n_vib)

        # If coord_scale was not set (all-ones default from PES model that
        # was trained without thermal normalization), auto-compute from the
        # per-mode standard deviation of training data.  This mode-equalizes
        # the kernel: low-frequency torsions (range ±10) and high-frequency
        # C-H modes (range ±0.3) contribute equally, which is essential for
        # the dipole surface to learn C-H stretch sensitivity.
        if np.allclose(self.coord_scale, 1.0):
            per_mode_std = np.std(X_q, axis=0)
            self.coord_scale = np.where(per_mode_std > 1e-8, per_mode_std, 1e-8)
            if verbose:
                print(f"  coord_scale auto-computed (per-mode std): "
                      f"[{self.coord_scale.min():.4f}, {self.coord_scale.max():.4f}]")

        X_qs = X_q / self.coord_scale[None, :]              # scaled, ~unit std per mode

        # ── train/test split ────────────────────────────────────────────────
        rng    = np.random.RandomState(random_seed)
        n_test = max(1, int(M * test_fraction))
        test_idx   = rng.choice(M, n_test, replace=False)
        train_mask = np.ones(M, dtype=bool)
        train_mask[test_idx] = False
        X_tr, y_tr = X_qs[train_mask],  dipoles_D[train_mask]
        X_te, y_te = X_qs[~train_mask], dipoles_D[~train_mask]

        # ── median heuristic for γ centre ──────────────────────────────────
        n_sub  = min(200, len(X_tr))
        i_sub  = rng.choice(len(X_tr), n_sub, replace=False)
        X_sub  = X_tr[i_sub]
        sq     = (np.sum(X_sub**2, axis=1, keepdims=True)
                  + np.sum(X_sub**2, axis=1, keepdims=True).T
                  - 2.0 * X_sub @ X_sub.T)
        upper  = sq[np.triu_indices(n_sub, k=1)]
        med_sq = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
        gamma_c = 1.0 / (2.0 * med_sq)

        log_c   = math.log10(gamma_c)
        g_grid  = sorted({round(10 ** (log_c + d), 8)
                          for d in np.linspace(-2, 2, 9)})
        a_grid  = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]

        if verbose:
            print(f"\n  NMDipoleSurface.fit: {len(X_tr)} train / {len(X_te)} test frames")
            print(f"  γ_center={gamma_c:.4g}  "
                  f"grid ({len(g_grid)}): {[f'{g:.3g}' for g in g_grid]}")

        # ── analytic LOO-CV grid search ─────────────────────────────────────
        mu_mean = y_tr.mean(axis=0)          # (3,)
        y_tr_c  = y_tr - mu_mean             # centred

        best_rmse, best_g, best_a = np.inf, g_grid[0], a_grid[0]
        for g in g_grid:
            K = self._rbf(X_tr, X_tr, g)    # reuse K for all α
            for a in a_grid:
                rmse = self._loo_cv_rmse(K, a, y_tr_c)
                if rmse < best_rmse:
                    best_rmse, best_g, best_a = rmse, g, a

        if verbose:
            print(f"  Best: γ={best_g:.4g}  α={best_a:.2g}  LOO-CV RMSE={best_rmse:.4f} D")

        # ── refit on full training set ──────────────────────────────────────
        self.gamma        = float(best_g)
        self.alpha_reg    = float(best_a)
        self._mu_mean     = mu_mean
        self._X_train_qs  = X_tr

        K_tr = self._rbf(X_tr, X_tr, self.gamma)
        K_tr[np.diag_indices(len(X_tr))] += self.alpha_reg
        self._beta = np.linalg.solve(K_tr, y_tr_c)          # (n_train, 3)

        # ── metrics ────────────────────────────────────────────────────────
        self.cv_rmse_D = float(best_rmse)

        K_tr_pred      = self._rbf(X_tr, X_tr, self.gamma)
        y_pred_tr      = K_tr_pred @ self._beta + mu_mean
        self.train_rmse_D = float(np.sqrt(np.mean((y_pred_tr - y_tr) ** 2)))

        K_te           = self._rbf(X_te, X_tr, self.gamma)
        y_pred_te      = K_te @ self._beta + mu_mean
        resid_te       = y_pred_te - y_te
        self.test_rmse_D = float(np.sqrt(np.mean(resid_te ** 2)))
        ss_res = float(np.sum(resid_te ** 2))
        ss_tot = float(np.sum((y_te - y_te.mean(axis=0)) ** 2))
        self.r2_test = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        if verbose:
            print(f"  Train RMSE: {self.train_rmse_D:.4f} D")
            print(f"  Test  RMSE: {self.test_rmse_D:.4f} D  R²={self.r2_test:.4f}")

        return self

    # ── prediction ─────────────────────────────────────────────────────────

    def _check_fitted(self):
        if self._beta is None:
            raise RuntimeError("NMDipoleSurface not fitted — call fit() first")

    def predict(self, coords_ang: np.ndarray) -> np.ndarray:
        """Predict dipole vector (3,) Debye for one geometry (N, 3)."""
        self._check_fitted()
        q   = self.project(coords_ang)
        q_s = q / self.coord_scale
        diff2 = np.sum((self._X_train_qs - q_s[None, :]) ** 2, axis=1)
        k = np.exp(-self.gamma * diff2)                      # (M,)
        return k @ self._beta + self._mu_mean               # (3,)

    def predict_batch(self, coords_ang: np.ndarray) -> np.ndarray:
        """Predict dipole vectors (M, 3) for a batch of geometries (M, N, 3)."""
        self._check_fitted()
        X_qs = self.project_batch(coords_ang) / self.coord_scale[None, :]
        K    = self._rbf(X_qs, self._X_train_qs, self.gamma)
        return K @ self._beta + self._mu_mean

    # ── analytic dipole derivatives ─────────────────────────────────────────

    def dipole_derivatives(
        self,
        coords_ang: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Analytic dipole derivatives ∂μ_α/∂q_k (Debye / sqrt(amu)·Bohr).

        Shape: (3, n_vib) — row α is the gradient of dipole component α
        w.r.t. each NM coordinate q_k.

        At the reference equilibrium (coords_ang=None, q=0) this gives the
        linear dipole derivative — directly proportional to harmonic IR intensity.

        Squared norm ||∂μ/∂q_k||² ∝ IR absorption intensity of mode k.
        """
        self._check_fitted()
        if coords_ang is None:
            q = np.zeros(self.n_vib)
        else:
            q = self.project(coords_ang)
        q_s   = q / self.coord_scale                        # (n_vib,)
        diff2 = np.sum((self._X_train_qs - q_s[None, :]) ** 2, axis=1)
        k_vec = np.exp(-self.gamma * diff2)                 # (M,)
        g     = k_vec[:, None] * self._beta                 # (M, 3)  weighted β
        Δqs   = q_s[None, :] - self._X_train_qs            # (M, n_vib)  q_s − q_s_i
        # ∂μ_α/∂q_s_k = −2γ Σ_i g_{i,α} Δqs_{i,k}
        dmu_dqs = -2.0 * self.gamma * (g.T @ Δqs)          # (3, n_vib)
        # chain rule: dμ/dq_k = (dμ/dq_s_k) / coord_scale_k
        return dmu_dqs / self.coord_scale[None, :]          # (3, n_vib)

    def ir_intensities(
        self,
        coords_ang: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        IR intensity per NM mode: I_k = ||∂μ/∂q_k||² (Debye²/(amu·Bohr²)).

        Shape: (n_vib,). Useful for identifying which modes are IR-active
        on the ML dipole surface vs. the Coulomb+KRR surface.
        """
        dmu = self.dipole_derivatives(coords_ang)           # (3, n_vib)
        return np.sum(dmu ** 2, axis=0)                     # (n_vib,)

    # ── metadata (DipoleSurface-compatible) ─────────────────────────────────

    @property
    def metadata(self) -> dict:
        """Dict compatible with DipoleSurface.metadata for ir_md_spectrum.py."""
        return {
            'model_type':   'NMDipoleSurface',
            'gamma':        self.gamma,
            'alpha_reg':    self.alpha_reg,
            'n_train':      len(self._X_train_qs) if self._X_train_qs is not None else 0,
            'cv_rmse':      self.cv_rmse_D,
            'train_rmse':   self.train_rmse_D,
            'test_rmse':    self.test_rmse_D,
            'r2_test':      self.r2_test,
        }

    # ── persistence ────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save as plain state dict (avoids module-identity PicklingError)."""
        state = {
            '_class':        'NMDipoleSurface',
            'eq_coords_ang': self.eq_coords_ang,
            'U_vib':         self.U_vib,
            'sqrt_mass':     self.sqrt_mass,
            'freqs_vib':     self.freqs_vib,
            'symbols':       self.symbols,
            'coord_scale':   self.coord_scale,
            'gamma':         self.gamma,
            'alpha_reg':     self.alpha_reg,
            'X_train_qs':    self._X_train_qs,
            'beta':          self._beta,
            'mu_mean':       self._mu_mean,
            'cv_rmse_D':     self.cv_rmse_D,
            'train_rmse_D':  self.train_rmse_D,
            'test_rmse_D':   self.test_rmse_D,
            'r2_test':       self.r2_test,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=4)

    @classmethod
    def load(cls, path: str) -> 'NMDipoleSurface':
        """Load from path — supports both state-dict and object pickles."""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if isinstance(obj, cls):
            return obj
        inst = cls(
            eq_coords_ang = obj['eq_coords_ang'],
            U_vib         = obj['U_vib'],
            sqrt_mass     = obj['sqrt_mass'],
            freqs_vib     = obj['freqs_vib'],
            symbols       = obj['symbols'],
            coord_scale   = obj.get('coord_scale'),
        )
        inst.gamma        = obj.get('gamma')
        inst.alpha_reg    = obj.get('alpha_reg')
        inst._X_train_qs  = obj.get('X_train_qs')
        inst._beta        = obj.get('beta')
        inst._mu_mean     = obj.get('mu_mean')
        inst.cv_rmse_D    = obj.get('cv_rmse_D')
        inst.train_rmse_D = obj.get('train_rmse_D')
        inst.test_rmse_D  = obj.get('test_rmse_D')
        inst.r2_test      = obj.get('r2_test')
        return inst

    @classmethod
    def from_nm_pes_model(cls, pes_model: 'NMKRRPESModel') -> 'NMDipoleSurface':
        """
        Create NMDipoleSurface sharing geometry/NM parameters from an
        existing NMKRRPESModel.  Call fit() afterwards to train.
        """
        return cls(
            eq_coords_ang = pes_model.eq_coords_ang,
            U_vib         = pes_model.U_vib,
            sqrt_mass     = pes_model.sqrt_mass,
            freqs_vib     = pes_model.freqs_vib,
            symbols       = pes_model.symbols,
            coord_scale   = pes_model.coord_scale,
        )


def load_dipole_surface(path: str):
    """
    Load a dipole surface model from path.  Auto-detects type:
    - NMDipoleSurface if state dict has '_class'=='NMDipoleSurface'
    - DipoleSurface otherwise (Coulomb+KRR legacy)
    """
    import pickle
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, NMDipoleSurface):
        return obj
    if isinstance(obj, dict) and obj.get('_class') == 'NMDipoleSurface':
        return NMDipoleSurface.load(path)
    # Fall back to DipoleSurface
    from ir_spectroscopy import DipoleSurface
    return DipoleSurface.load(path)
