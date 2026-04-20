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
