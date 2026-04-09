#!/usr/bin/env python3
"""
casscf_nm_delta.py — Delta-ML CASSCF correction using normal-mode coordinates.

Why normal modes instead of Coulomb matrix?
--------------------------------------------
The Coulomb matrix descriptor clusters all near-equilibrium geometries at
K ≈ 0.999 (nuclear charges dominate; small distortions move very little in
Coulomb space). High-energy CASSCF corrections (|δ| ≈ 10–28 kcal/mol at
ΔE_B3LYP > 30 kcal/mol) bleed into equilibrium because they are NOT far
away in Coulomb-matrix descriptor space. This corrupts the delta prediction
near q = 0 where the molecule actually lives at 300 K.

Normal mode coordinates fix this:
  - q = 0 exactly at the reference (B3LYP or CASSCF) minimum
  - ||q||² grows monotonically with distortion energy
  - RBF kernel K(q_i, q_j) = exp(−γ ||q_i − q_j||²) localises correctly:
    high-energy training frames are far from q = 0 → K ≈ 0 → no bleeding
  - Physically orthogonal modes → no descriptor clustering

The projection is:
    q_i = U_vib^T · M^{1/2} · (R_i − R_ref)
where:
  R_i   : Cartesian coordinates in Bohr (after converting from Angstrom)
  R_ref : reference (B3LYP or CASSCF) equilibrium geometry in Bohr
  M^{1/2}: diag(sqrt(m_1), sqrt(m_1), sqrt(m_1), sqrt(m_2), ...)   [sqrt(amu)]
  U_vib : (3N × n_vib) mass-weighted normal mode eigenvectors

q_i has units sqrt(amu)·Bohr. The KRR gamma is in 1/(amu·Bohr²).

Workflow
--------
1. Load 29 existing CASSCF single-point results (surface_results.json)
2. Retrieve training-set coordinates for each frame (combined_training_data.npz)
3. Load B3LYP equilibrium geometry (psi4_eq_coords.npy)
4. Obtain B3LYP Hessian (compute via PSI4 or load from .npy file)
5. Compute NM eigenvectors via existing modules/normal_modes.py
6. Project all frames: q_i = NM_project(R_i)
7. (Optional Fix A) Add NM-displaced CASSCF points at T=300 K for
   near-equilibrium anchoring
8. Train KRR δ(q) with gamma/alpha grid search + leave-one-out CV
9. Save NMKRRDeltaModel pickle → used by ir_md_spectrum.py --nm-delta-model

Usage
-----
  # Minimal: reuse existing 29 CASSCF points, compute B3LYP Hessian fresh
  python3 casscf_nm_delta.py \\
      --load-results outputs/casscf_surface_20260331_133413/surface_results.json \\
      --training-data outputs/mvko_20260319_081314/combined_training_data.npz \\
      --eq-coords outputs/mvko_20260319_081314/psi4_eq_coords.npy

  # With pre-saved B3LYP Hessian (fast; skip PSI4):
  python3 casscf_nm_delta.py \\
      --load-results ... --training-data ... --eq-coords ... \\
      --b3lyp-hessian outputs/casscf_nm_delta_<ts>/b3lyp_hessian.npy

  # + Fix A: add 15 near-equilibrium CASSCF points at 300 K NM amplitude
  python3 casscf_nm_delta.py \\
      --load-results ... --training-data ... --eq-coords ... \\
      --add-nm-points 15 --T-nm 300 \\
      --b3lyp-hessian outputs/casscf_nm_delta_<ts>/b3lyp_hessian.npy

  # CASSCF geometry optimization as reference (expensive):
  python3 casscf_nm_delta.py \\
      --load-results ... --training-data ... --eq-coords ... \\
      --casscf-opt

  # Reload a completed run (skip PSI4, reuse saved hessian + CASSCF results):
  python3 casscf_nm_delta.py \\
      --load-results outputs/casscf_nm_delta_<ts>/all_casscf_results.json \\
      --training-data ... \\
      --eq-coords outputs/casscf_nm_delta_<ts>/casscf_eq_coords.npy \\
      --b3lyp-hessian outputs/casscf_nm_delta_<ts>/b3lyp_hessian.npy

Output
------
  outputs/casscf_nm_delta_<ts>/
    nm_delta_model.pkl          ← NMKRRDeltaModel for --nm-delta-model flag
    b3lyp_hessian.npy           ← (3N,3N) B3LYP Hessian (save for reuse)
    casscf_eq_coords.npy        ← CASSCF-optimized eq geometry (if --casscf-opt)
    all_casscf_results.json     ← merged CASSCF results (original + Fix A)
    nm_descriptors.npy          ← (M, n_vib) projected NM coordinates
    diagnostics.png             ← delta vs ||q||, LOO-CV, NO occupations
    summary.json
"""

import argparse
import json
import pickle
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

HARTREE_TO_KCAL = 627.509474
ANGSTROM_TO_BOHR = 1.88972612463

# MVKO active space (same as casscf_surface_correction.py)
N_ELEC_ACTIVE     = 4
N_ORBS_ACTIVE     = 4
N_FROZEN_CORE     = 6
N_RESTRICTED_DOCC = 15


# ── NMKRRDeltaModel ───────────────────────────────────────────────────────────

class NMKRRDeltaModel:
    """
    Delta-ML correction: E_corrected = E_B3LYP + δ(q)

    δ is a KRR model in normal-mode coordinate space:
        q = U_vib^T · M^{1/2} · (R − R_ref)  [sqrt(amu)·Bohr]

    Interface: .predict(symbols, coords_ang) → delta in Hartree
    This matches the MLPESTrainer.predict() signature so that
    DeltaMLPESDriver (in ir_md_spectrum.py) can wrap it directly.
    """

    def __init__(
        self,
        *,
        eq_coords_ang: np.ndarray,   # (N, 3) reference geometry in Angstrom
        U_vib: np.ndarray,           # (3N, n_vib) mass-weighted eigenvectors
        sqrt_mass: np.ndarray,       # (3N,) sqrt(amu)
        freqs_vib: np.ndarray,       # (n_vib,) cm⁻¹ (diagnostic only)
        symbols: list,
        gamma: float,                # RBF kernel width (1/(amu·Bohr²))
        alpha_reg: float,            # KRR regularisation
        X_train_q: np.ndarray,       # (M, n_vib) training NM coordinates
        y_train_ha: np.ndarray,      # (M,) delta in Hartree
        e_b3lyp_ref_ha: float,       # B3LYP reference energy (Ha) for reporting
        e_cas_ref_ha: float,         # CASSCF reference energy (Ha) for reporting
        cv_rmse_kcal: float = None,  # leave-one-out CV RMSE (diagnostic)
    ):
        self.eq_coords_ang  = np.asarray(eq_coords_ang, dtype=float)
        self.U_vib          = np.asarray(U_vib, dtype=float)
        self.sqrt_mass      = np.asarray(sqrt_mass, dtype=float)
        self.freqs_vib      = np.asarray(freqs_vib, dtype=float)
        self.symbols        = list(symbols)
        self.gamma          = float(gamma)
        self.alpha_reg      = float(alpha_reg)
        self.X_train_q      = np.asarray(X_train_q, dtype=float)
        self.y_train_ha     = np.asarray(y_train_ha, dtype=float)
        self.e_b3lyp_ref_ha = float(e_b3lyp_ref_ha)
        self.e_cas_ref_ha   = float(e_cas_ref_ha)
        self.cv_rmse_kcal   = cv_rmse_kcal

        # Pre-compute KRR dual coefficients: solve (K + α I) α_vec = y
        K = self._kernel(X_train_q, X_train_q)
        K[np.diag_indices_from(K)] += alpha_reg
        self._alpha_vec = np.linalg.solve(K, y_train_ha)

    # ── projection ────────────────────────────────────────────────────────────

    def project(self, coords_ang: np.ndarray) -> np.ndarray:
        """
        Project one geometry into NM coordinate space.

        Args:
            coords_ang : (N, 3) Cartesian coordinates in Angstrom

        Returns:
            q : (n_vib,) displacement in sqrt(amu)·Bohr
        """
        delta_ang  = np.asarray(coords_ang, dtype=float) - self.eq_coords_ang
        delta_bohr = delta_ang.flatten() * ANGSTROM_TO_BOHR   # (3N,) Bohr
        delta_mw   = delta_bohr * self.sqrt_mass               # (3N,) sqrt(amu)·Bohr
        return self.U_vib.T @ delta_mw                         # (n_vib,)

    def project_batch(self, coords_ang: np.ndarray) -> np.ndarray:
        """Project a batch of geometries. coords_ang: (M, N, 3) → (M, n_vib)."""
        coords_ang = np.asarray(coords_ang, dtype=float)
        delta_ang  = coords_ang - self.eq_coords_ang[None, :, :]   # (M,N,3)
        delta_bohr = delta_ang.reshape(len(coords_ang), -1) * ANGSTROM_TO_BOHR  # (M,3N)
        delta_mw   = delta_bohr * self.sqrt_mass[None, :]           # (M,3N)
        return delta_mw @ self.U_vib                                 # (M,n_vib)

    # ── kernel ────────────────────────────────────────────────────────────────

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """RBF kernel K[i,j] = exp(−γ ||A_i − B_j||²). A:(m,d), B:(n,d)→(m,n)."""
        # ||A_i - B_j||² = ||A_i||² + ||B_j||² - 2 A_i·B_j
        A2 = np.sum(A ** 2, axis=1, keepdims=True)   # (m,1)
        B2 = np.sum(B ** 2, axis=1, keepdims=True)   # (n,1)
        return np.exp(-self.gamma * (A2 + B2.T - 2.0 * A @ B.T))

    # ── prediction ────────────────────────────────────────────────────────────

    def predict_delta_ha(self, q: np.ndarray) -> float:
        """Predict delta correction (Ha) for one NM-coordinate vector."""
        k = np.exp(-self.gamma * np.sum((self.X_train_q - q[None, :]) ** 2, axis=1))
        return float(np.dot(self._alpha_vec, k))

    def predict(self, symbols: list, coords_ang: np.ndarray) -> float:
        """
        Predict delta correction in Hartree for one geometry.

        Matches MLPESTrainer.predict(symbols, coords) interface so that
        DeltaMLPESDriver in ir_md_spectrum.py can wrap this model directly.
        """
        q = self.project(coords_ang)
        return self.predict_delta_ha(q)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        # Save as a plain state dict to avoid pickle module-identity errors
        # when casscf_nm_delta is imported from a script running as __main__.
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
            'e_b3lyp_ref_ha': self.e_b3lyp_ref_ha,
            'e_cas_ref_ha':   self.e_cas_ref_ha,
            'cv_rmse_kcal':   self.cv_rmse_kcal,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=4)
        print(f"  NMKRRDeltaModel saved: {path}")

    @classmethod
    def load(cls, path: str) -> 'NMKRRDeltaModel':
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        # Support both old full-object pickles and new state-dict pickles
        if isinstance(obj, cls):
            return obj
        return cls(**obj)


# ── KRR training helpers ──────────────────────────────────────────────────────

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    X2 = np.sum(X ** 2, axis=1, keepdims=True)
    Y2 = np.sum(Y ** 2, axis=1, keepdims=True)
    return np.exp(-gamma * (X2 + Y2.T - 2.0 * X @ Y.T))


def leave_one_out_cv(X_q: np.ndarray, y_ha: np.ndarray,
                     gamma: float, alpha_reg: float) -> float:
    """
    Leave-one-out CV RMSE in kcal/mol by explicit retraining.

    The hat-matrix shortcut  ŷ_{−i} = (ŷ_i − y_i) / (1 − h_{ii})  is
    numerically degenerate when alpha is very small (h_{ii} → 1), giving
    the false impression of 0 LOO error.  For the dataset sizes we use
    (M ≤ ~50), explicit retraining of M models is fast and exact.
    """
    n = len(X_q)
    loo_errors = np.zeros(n)
    try:
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_loo = X_q[mask]
            y_loo = y_ha[mask]
            K_loo = _rbf_kernel(X_loo, X_loo, gamma)
            K_loo[np.diag_indices_from(K_loo)] += alpha_reg
            alpha_vec = np.linalg.solve(K_loo, y_loo)
            k_pred = _rbf_kernel(X_q[i:i+1], X_loo, gamma)[0]
            loo_pred = float(np.dot(alpha_vec, k_pred))
            loo_errors[i] = loo_pred - y_ha[i]
        rmse = np.sqrt(np.mean(loo_errors ** 2)) * HARTREE_TO_KCAL
    except np.linalg.LinAlgError:
        rmse = np.inf
    return float(rmse)


def grid_search_krr(X_q: np.ndarray, y_ha: np.ndarray,
                    gammas: list, alphas: list) -> tuple:
    """
    Grid search over (gamma, alpha) pairs minimising LOO-CV RMSE.
    Returns (best_gamma, best_alpha, best_rmse, results_grid).
    """
    best_rmse  = np.inf
    best_gamma = gammas[0]
    best_alpha = alphas[0]
    grid = []

    for g in gammas:
        for a in alphas:
            rmse = leave_one_out_cv(X_q, y_ha, g, a)
            grid.append({'gamma': g, 'alpha': a, 'loo_rmse_kcal': rmse})
            if rmse < best_rmse:
                best_rmse, best_gamma, best_alpha = rmse, g, a

    return best_gamma, best_alpha, best_rmse, grid


# ── NM projection helpers ─────────────────────────────────────────────────────

def get_nm_eigvecs(symbols: list, hessian: np.ndarray):
    """
    Compute vibrational normal modes from Hessian.

    Uses modules/normal_modes.py compute_normal_modes().

    Returns:
        frequencies : (n_vib,) cm⁻¹
        U_vib       : (3N, n_vib) mass-weighted eigenvectors
        sqrt_mass   : (3N,) sqrt(amu)
    """
    from modules.normal_modes import compute_normal_modes, ATOMIC_MASSES
    frequencies, U_vib, eigenvalues, mass_vec = compute_normal_modes(symbols, hessian)
    sqrt_mass = np.sqrt(mass_vec)
    return frequencies, U_vib, sqrt_mass


def compute_b3lyp_hessian_psi4(symbols: list, eq_coords_ang: np.ndarray,
                                out_dir: Path, n_threads: int = 4,
                                memory: str = '6 GB') -> np.ndarray:
    """
    Compute B3LYP/6-31G* Cartesian Hessian (Hartree/Bohr²) via PSI4.
    Saves to out_dir/b3lyp_hessian.npy for reuse.
    """
    try:
        import psi4
    except ImportError:
        raise RuntimeError("PSI4 not available. Provide --b3lyp-hessian path.")

    out_file = str(out_dir / 'psi4_hessian.dat')
    psi4.core.set_output_file(out_file, False)
    psi4.set_memory(memory)
    psi4.set_num_threads(n_threads)
    psi4.core.clean()
    try:
        psi4.core.clean_options()
    except AttributeError:
        pass

    lines = ["0 1"]
    for s, (x, y, z) in zip(symbols, eq_coords_ang):
        lines.append(f"  {s}  {x:.10f}  {y:.10f}  {z:.10f}")
    lines += ["units angstrom", "no_reorient", "no_com", "symmetry c1"]
    psi4.geometry("\n".join(lines))

    psi4.set_options({
        'basis': '6-31G*', 'scf_type': 'df', 'reference': 'rhf',
        'maxiter': 200, 'e_convergence': 1e-8, 'd_convergence': 1e-8,
    })

    print("  Running PSI4 B3LYP/6-31G* Hessian...")
    H_psi4 = psi4.hessian('B3LYP/6-31G*')
    hess = np.array(H_psi4)   # (3N, 3N) Hartree/Bohr²
    out_path = out_dir / 'b3lyp_hessian.npy'
    np.save(str(out_path), hess)
    print(f"  B3LYP Hessian saved: {out_path}")
    return hess


def optimize_casscf_psi4(symbols: list, eq_coords_ang: np.ndarray,
                          out_dir: Path, n_threads: int = 4,
                          memory: str = '8 GB') -> np.ndarray:
    """
    Optimize CASSCF(4,4)/6-31G* geometry. Returns optimized coords (Angstrom).
    Also computes and saves the CASSCF Hessian (for NM modes).
    """
    try:
        import psi4
    except ImportError:
        raise RuntimeError("PSI4 not available for CASSCF optimization.")

    out_file = str(out_dir / 'psi4_casscf_opt.dat')
    psi4.core.set_output_file(out_file, False)
    psi4.set_memory(memory)
    psi4.set_num_threads(n_threads)
    psi4.core.clean()
    try:
        psi4.core.clean_options()
    except AttributeError:
        pass

    lines = ["0 1"]
    for s, (x, y, z) in zip(symbols, eq_coords_ang):
        lines.append(f"  {s}  {x:.10f}  {y:.10f}  {z:.10f}")
    lines += ["units angstrom", "no_reorient", "no_com", "symmetry c1"]
    mol = psi4.geometry("\n".join(lines))

    psi4.set_options({
        'basis': '6-31G*', 'scf_type': 'df', 'reference': 'rhf',
        'maxiter': 200, 'e_convergence': 1e-8, 'd_convergence': 1e-8,
        'frozen_docc':     [N_FROZEN_CORE],
        'restricted_docc': [N_RESTRICTED_DOCC],
        'active':          [N_ORBS_ACTIVE],
        'num_roots':       1, 'avg_states': [0], 'avg_weights': [1.0],
        'mcscf_algorithm': 'ah', 'mcscf_maxiter': 300,
        'mcscf_r_convergence': 1e-6, 'mcscf_e_convergence': 1e-9,
    })

    print("  Running PSI4 CASSCF(4,4)/6-31G* optimization (expensive)...")
    psi4.optimize('casscf', molecule=mol)
    geom = np.array(mol.geometry()).reshape(-1, 3) / ANGSTROM_TO_BOHR  # back to Angstrom
    out_path = out_dir / 'casscf_eq_coords.npy'
    np.save(str(out_path), geom)
    print(f"  CASSCF eq geometry saved: {out_path}")

    # Optionally compute CASSCF Hessian for better NM reference
    print("  Running PSI4 CASSCF Hessian...")
    H_psi4 = psi4.hessian('casscf', molecule=mol)
    hess = np.array(H_psi4)
    hess_path = out_dir / 'casscf_hessian.npy'
    np.save(str(hess_path), hess)
    print(f"  CASSCF Hessian saved: {hess_path}")
    return geom, hess


# ── Fix A: near-equilibrium NM-displaced CASSCF points ───────────────────────

def _geometry_string(symbols, coords, charge=0, mult=1):
    lines = [f"{charge} {mult}"]
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"  {sym}  {x:.10f}  {y:.10f}  {z:.10f}")
    lines += ["units angstrom", "no_reorient", "no_com", "symmetry c1"]
    return "\n".join(lines)


def _parse_no_occupations(output_text: str):
    matches = list(re.finditer(
        r'Active Space Natural occupation numbers:\s*\n\s*\n([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|\Z)',
        output_text))
    if matches:
        block = matches[0].group(1)
        nums = re.findall(r'[A-Za-z]+\s+([\d.]+)', block)
        if nums:
            return sorted([float(x) for x in nums], reverse=True)
    return None


def add_nm_casscf_points(symbols: list, eq_coords_ang: np.ndarray,
                          U_vib: np.ndarray, sqrt_mass: np.ndarray,
                          freqs_vib: np.ndarray, T_nm: float, n_points: int,
                          out_dir: Path, n_threads: int = 4,
                          memory: str = '6 GB',
                          e_b3lyp_ref_ha: float = 0.0) -> list:
    """
    Fix A: generate n_points SS-CASSCF single points at NM-displaced geometries.

    Geometries are ±1 thermal amplitude displacements along each mode (truncated
    to n_points modes, starting from the lowest-frequency ones). The reference
    energy e_b3lyp_ref_ha is used to fill the 'e_b3lyp' field (we store the
    B3LYP energy from the reference for the delta calculation; for these new
    points the user must run B3LYP separately or pass 0.0 — the delta is
    computed relative to the reference frame 0 anyway).

    Returns list of result dicts (same schema as surface_results.json).
    """
    try:
        import psi4
    except ImportError:
        print("  PSI4 not available — cannot generate Fix A points.")
        return []

    from modules.normal_modes import ATOMIC_MASSES, KB_HARTREE_PER_K

    masses  = np.array([ATOMIC_MASSES[s] for s in symbols])
    mass_vec = np.repeat(masses, 3)

    # Cartesian eigenvectors (not mass-weighted):  L_cart = U_vib / sqrt_mass
    L_cart = U_vib / sqrt_mass[:, None]      # (3N, n_vib)

    n_vib = len(freqs_vib)
    results = []
    frame_counter = [10000]  # synthetic frame indices start at 10000

    # Select modes to displace: favour low-frequency (thermally accessible)
    mode_order = np.argsort(np.abs(freqs_vib))

    print(f"  Fix A: generating CASSCF points for {min(n_points, n_vib)} modes "
          f"at T={T_nm} K...")

    for mode_i in mode_order[:n_points]:
        freq_cm1 = freqs_vib[mode_i]
        if np.abs(freq_cm1) < 50:
            continue

        # Classical thermal amplitude in Bohr·sqrt(amu)
        from modules.normal_modes import KB_HARTREE_PER_K, FREQ_CONV
        eigenval = (freq_cm1 / FREQ_CONV) ** 2    # Hartree/(Bohr²·amu)
        if eigenval <= 0:
            continue
        Q_thermal = np.sqrt(2.0 * KB_HARTREE_PER_K * T_nm / eigenval)  # Bohr·sqrt(amu)

        # Cartesian displacement in Angstrom:
        # Δr = Q · (L_cart[:, mode_i] / sqrt_mass) … but L_cart already has
        # that division; and Q has units Bohr·sqrt(amu), L_cart has 1/sqrt(amu)
        # so Δr_bohr = Q · L_cart (Bohr), then /ANGSTROM_TO_BOHR → Angstrom
        delta_bohr = Q_thermal * L_cart[:, mode_i]  # (3N,) Bohr
        delta_ang  = delta_bohr.reshape(-1, 3) / ANGSTROM_TO_BOHR

        for sign, label in [(+1, 'pos'), (-1, 'neg')]:
            new_coords = eq_coords_ang + sign * delta_ang
            fidx = frame_counter[0]
            frame_counter[0] += 1

            psi4.core.clean()
            try:
                psi4.core.clean_options()
            except AttributeError:
                pass

            fout = str(out_dir / f'psi4_nmfix_m{mode_i:02d}_{label}.dat')
            psi4.core.set_output_file(fout, False)
            psi4.set_memory(memory)
            psi4.set_num_threads(n_threads)
            psi4.geometry(_geometry_string(symbols, new_coords))

            result = {
                'frame_idx': fidx, 'nm_mode': int(mode_i),
                'nm_sign': sign, 'nm_freq_cm1': float(freq_cm1),
                'nm_amplitude_bohr_sqamu': float(Q_thermal),
                'e_b3lyp': e_b3lyp_ref_ha, 'e_b3lyp_rel': None,
                'e_rhf': None, 'e_casscf_ss': None,
                'no_occs': None, 'delta_kcal': None,
                'output_file': fout, 'error': None,
                '_coords': new_coords.tolist(),  # (N,3) Angstrom — needed by build_training_arrays
            }

            base_opts = {
                'basis': '6-31G*', 'scf_type': 'df', 'reference': 'rhf',
                'e_convergence': 1e-8, 'd_convergence': 1e-8, 'maxiter': 200,
            }
            psi4.set_options(base_opts)
            try:
                E_rhf, wfn_rhf = psi4.energy('hf', return_wfn=True)
                result['e_rhf'] = float(E_rhf)
            except Exception as exc:
                result['error'] = f'rhf: {exc}'
                results.append(result)
                print(f"    mode {mode_i:02d} {label}: RHF FAILED — {exc}")
                continue

            casscf_opts = {
                **base_opts,
                'frozen_docc':         [N_FROZEN_CORE],
                'restricted_docc':     [N_RESTRICTED_DOCC],
                'active':              [N_ORBS_ACTIVE],
                'num_roots': 1, 'avg_states': [0], 'avg_weights': [1.0],
                'mcscf_algorithm': 'ah', 'mcscf_maxiter': 200,
                'mcscf_diis_start': 3,
                'mcscf_r_convergence': 1e-5, 'mcscf_e_convergence': 1e-8,
            }
            psi4.set_options(casscf_opts)
            try:
                E_cas, _ = psi4.energy('casscf', return_wfn=True, ref_wfn=wfn_rhf)
                result['e_casscf_ss'] = float(E_cas)
                with open(fout) as f:
                    no_occs = _parse_no_occupations(f.read())
                if no_occs:
                    result['no_occs'] = no_occs[:N_ORBS_ACTIVE]
                print(f"    mode {mode_i:02d} {label}: CASSCF = {E_cas:.8f} Ha "
                      f"NO = {no_occs}")
            except Exception as exc:
                result['error'] = f'casscf: {exc}'
                print(f"    mode {mode_i:02d} {label}: CASSCF FAILED — {exc}")

            results.append(result)

    return results


# ── Load + filter CASSCF data ─────────────────────────────────────────────────

def load_casscf_results(json_path: str) -> list:
    """Load surface_results.json, filter frames with missing CASSCF energy."""
    with open(json_path) as f:
        raw = json.load(f)
    good = [r for r in raw
            if r.get('e_casscf_ss') is not None
            and r.get('e_b3lyp') is not None]
    n_bad = len(raw) - len(good)
    if n_bad:
        print(f"  Dropped {n_bad} frames with failed CASSCF (e.g. frame 713).")
    return good


def filter_results(results: list, max_energy_kcal: float,
                   delta_threshold_kcal: float) -> list:
    """
    Apply energy and delta magnitude filters.

    Frames with ΔE_B3LYP > max_energy_kcal are excluded (the kernel would
    give them tiny weight near equilibrium anyway, but excluding them avoids
    confusing the grid search with pathological high-energy CASSCF points
    whose active space may not represent the same electronic character).

    Frames with |delta| > delta_threshold_kcal are flagged as warnings but
    NOT automatically excluded — the NM kernel handles them geometrically.
    Set --delta-threshold 0 to use all frames.
    """
    e_b3lyp = np.array([r['e_b3lyp'] for r in results])
    e_b3_ref = e_b3lyp.min()
    e_cas    = np.array([r['e_casscf_ss'] for r in results])
    e_cas_ref = e_cas.min()

    kept = []
    for r in results:
        dE_b3 = (r['e_b3lyp'] - e_b3_ref) * HARTREE_TO_KCAL
        if dE_b3 > max_energy_kcal:
            continue

        delta = r.get('delta_kcal')
        if delta is None:
            # Recompute from raw energies (Fix A frames don't pre-store this)
            dE_cas = (r['e_casscf_ss'] - e_cas_ref) * HARTREE_TO_KCAL
            delta  = dE_cas - dE_b3
        if delta_threshold_kcal > 0 and abs(delta) > delta_threshold_kcal:
            print(f"  WARNING: frame {r['frame_idx']} |delta|={abs(delta):.1f} "
                  f"> {delta_threshold_kcal} kcal/mol — included but flagged.")

        kept.append(r)

    print(f"  After energy filter (<{max_energy_kcal} kcal/mol): "
          f"{len(kept)}/{len(results)} frames")
    return kept


def build_training_arrays(results: list, train_coords: np.ndarray,
                          train_energies: np.ndarray,
                          nm_results: list = None):
    """
    Assemble (coords, e_b3lyp_ha, e_cas_ha) arrays from filtered result list.

    For Fix A NM-displaced frames (frame_idx >= 10000), we need the geometry
    stored in the result dict (passed via nm_results which carries the coords).
    """
    # Original frames — coordinates from training data by frame_idx
    e_b3lyp_all = [r['e_b3lyp'] for r in results]
    e_cas_all   = [r['e_casscf_ss'] for r in results]
    coords_list = []

    for r in results:
        fidx = r['frame_idx']
        if fidx < 10000:
            coords_list.append(train_coords[fidx])
        else:
            # Synthetic NM-displaced geometry — must be stored in result['_coords']
            # (set by add_nm_casscf_points) or in nm_results list
            coords_found = False
            if r.get('_coords') is not None:
                coords_list.append(np.array(r['_coords']))
                coords_found = True
            elif nm_results is not None:
                match = next((x for x in nm_results
                              if x.get('frame_idx') == fidx
                              and x.get('_coords') is not None), None)
                if match:
                    coords_list.append(np.array(match['_coords']))
                    coords_found = True
            if not coords_found:
                raise RuntimeError(
                    f"No geometry found for synthetic Fix A frame {fidx}. "
                    "This indicates a bug: add_nm_casscf_points must store "
                    "'_coords' in each result dict."
                )

    coords_arr = np.array(coords_list)
    e_b3lyp_arr = np.array(e_b3lyp_all)
    e_cas_arr   = np.array(e_cas_all)

    return coords_arr, e_b3lyp_arr, e_cas_arr


# ── Diagnostics ───────────────────────────────────────────────────────────────

def plot_diagnostics(X_q: np.ndarray, delta_ha: np.ndarray,
                     freqs_vib: np.ndarray, grid_results: list,
                     best_gamma: float, best_alpha: float,
                     out_dir: Path, results: list):
    """
    3-panel diagnostic figure:
      Panel 1: delta (kcal/mol) vs ||q||² (NM amplitude²)
      Panel 2: gamma vs LOO-CV RMSE heatmap
      Panel 3: NO occupations vs ΔE_B3LYP
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping diagnostics figure.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={'wspace': 0.38})
    ax1, ax2, ax3 = axes

    delta_kcal = delta_ha * HARTREE_TO_KCAL
    q_norm2    = np.sum(X_q ** 2, axis=1)

    # Reference energy info
    e_b3lyp_arr = np.array([r['e_b3lyp'] for r in results])
    e_b3_ref    = e_b3lyp_arr.min()
    dE_b3_kcal  = (e_b3lyp_arr - e_b3_ref) * HARTREE_TO_KCAL

    sc = ax1.scatter(q_norm2, delta_kcal, c=dE_b3_kcal,
                     cmap='plasma', s=60, zorder=3)
    ax1.axhline(0, color='gray', lw=0.8, ls='--')
    ax1.set_xlabel('||q||²  (amu·Bohr²)', fontsize=11)
    ax1.set_ylabel('δ = ΔE_CASSCF − ΔE_B3LYP  (kcal/mol)', fontsize=11)
    ax1.set_title('Delta correction vs NM amplitude', fontsize=11)
    plt.colorbar(sc, ax=ax1, label='ΔE_B3LYP (kcal/mol)')

    # Panel 2: gamma vs CV RMSE
    gammas_g  = sorted(set(g['gamma'] for g in grid_results))
    alphas_g  = sorted(set(g['alpha'] for g in grid_results))
    if len(gammas_g) > 1 and len(alphas_g) > 1:
        rmse_mat = np.zeros((len(alphas_g), len(gammas_g)))
        gmap = {g: i for i, g in enumerate(gammas_g)}
        amap = {a: i for i, a in enumerate(alphas_g)}
        for row in grid_results:
            rmse_mat[amap[row['alpha']], gmap[row['gamma']]] = row['loo_rmse_kcal']
        rmse_mat = np.clip(rmse_mat, 0, 20)
        im = ax2.imshow(rmse_mat, aspect='auto', cmap='RdYlGn_r', origin='lower')
        ax2.set_xticks(range(len(gammas_g)))
        ax2.set_xticklabels([f'{g:.3g}' for g in gammas_g], rotation=45, fontsize=8)
        ax2.set_yticks(range(len(alphas_g)))
        ax2.set_yticklabels([f'{a:.0e}' for a in alphas_g], fontsize=8)
        ax2.set_xlabel('gamma  (1/(amu·Bohr²))', fontsize=10)
        ax2.set_ylabel('alpha (regularisation)', fontsize=10)
        ax2.set_title(f'LOO-CV RMSE (kcal/mol)\nbest γ={best_gamma:.3g}, α={best_alpha:.0e}',
                      fontsize=10)
        plt.colorbar(im, ax=ax2, label='RMSE (kcal/mol)')
    else:
        ax2.text(0.5, 0.5, f'γ={best_gamma}\nα={best_alpha}',
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('KRR parameters', fontsize=10)

    # Panel 3: NO occupations
    occ_data = [(r['no_occs'], (r['e_b3lyp'] - e_b3_ref) * HARTREE_TO_KCAL)
                for r in results if r.get('no_occs')]
    if occ_data:
        occs_arr = np.array([o for o, _ in occ_data])
        ens_arr  = np.array([e for _, e in occ_data])
        order = np.argsort(ens_arr)
        for col, lbl in enumerate(['NO 1', 'NO 2', 'NO 3', 'NO 4']):
            if col < occs_arr.shape[1]:
                ax3.plot(ens_arr[order], occs_arr[order, col], 'o-',
                         ms=5, lw=1.2, label=lbl)
        ax3.axhline(1.0, color='gray', lw=0.6, ls='--')
        ax3.set_xlabel('ΔE_B3LYP (kcal/mol)', fontsize=11)
        ax3.set_ylabel('NO occupation', fontsize=11)
        ax3.set_title('CASSCF natural orbital occupations', fontsize=11)
        ax3.legend(fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'No NO data', ha='center', va='center',
                 transform=ax3.transAxes)

    fig.suptitle('casscf_nm_delta diagnostics — NM coordinate delta-ML', fontsize=12)
    out_png = out_dir / 'diagnostics.png'
    fig.savefig(str(out_png), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Diagnostics: {out_png}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Delta-ML CASSCF correction with normal-mode coordinates')

    # Required inputs
    parser.add_argument('--load-results', required=True,
                        help='Path to surface_results.json from casscf_surface_correction.py')
    parser.add_argument('--training-data', required=True,
                        help='Path to combined_training_data.npz (B3LYP training set)')
    parser.add_argument('--eq-coords', required=True,
                        help='Path to .npy file with B3LYP (or CASSCF) equilibrium '
                             'geometry in Angstrom (e.g. psi4_eq_coords.npy)')

    # Hessian source
    hess_grp = parser.add_mutually_exclusive_group()
    hess_grp.add_argument('--b3lyp-hessian',
                          help='Pre-saved B3LYP Hessian .npy (Hartree/Bohr²). '
                               'If omitted, computed fresh via PSI4.')
    hess_grp.add_argument('--casscf-opt', action='store_true',
                          help='Run CASSCF(4,4) geometry opt + Hessian for NM reference '
                               '(expensive, ~hours). Overrides --eq-coords for projection.')

    # Fix A: near-equilibrium CASSCF points
    parser.add_argument('--add-nm-points', type=int, default=0, metavar='N',
                        help='Generate N additional SS-CASSCF points at 1-thermal-amplitude '
                             'NM displacements from equilibrium (Fix A, anchors near-eq). '
                             'Default: 0 (disabled).')
    parser.add_argument('--T-nm', type=float, default=300.0, metavar='K',
                        help='Temperature for NM displacement amplitude in Fix A (default 300 K)')

    # Filtering
    parser.add_argument('--max-energy', type=float, default=50.0, metavar='kcal/mol',
                        help='Exclude frames with ΔE_B3LYP above this (default 50 kcal/mol). '
                             'High-energy CASSCF often fails with wrong active space.')
    parser.add_argument('--delta-threshold', type=float, default=30.0, metavar='kcal/mol',
                        help='Warn on (but keep) frames with |delta| above this '
                             '(default 30). Set 0 to suppress warnings.')

    # KRR grid
    parser.add_argument('--gamma-values', default='0.01,0.05,0.1,0.5,1.0,5.0',
                        help='Comma-separated gamma values for grid search '
                             '(units: 1/(amu·Bohr²)). '
                             'Default: 0.01,0.05,0.1,0.5,1.0,5.0')
    parser.add_argument('--alpha-values', default='1e-8,1e-6,1e-4,1e-2',
                        help='Comma-separated alpha (regularisation) values for grid search. '
                             'Default: 1e-8,1e-6,1e-4,1e-2')

    # PSI4 settings
    parser.add_argument('--n-threads', type=int, default=4)
    parser.add_argument('--memory', default='6 GB')

    # Output
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: outputs/casscf_nm_delta_<ts>)')

    args = parser.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.out_dir) if args.out_dir else Path(f'outputs/casscf_nm_delta_{ts}')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== casscf_nm_delta.py  [{ts}] ===")
    print(f"Output directory: {out_dir}\n")

    # ── 1. Load CASSCF results ─────────────────────────────────────────────────
    print("Step 1: Loading CASSCF results...")
    results = load_casscf_results(args.load_results)
    print(f"  Loaded {len(results)} valid frames from {args.load_results}")

    # ── 2. Load training data (coordinates) ───────────────────────────────────
    print("\nStep 2: Loading B3LYP training data...")
    tdata = np.load(args.training_data, allow_pickle=True)
    train_coords   = tdata['coordinates']   # (N_train, n_atoms, 3) Angstrom
    train_energies = tdata['energies']      # (N_train,) Hartree
    symbols        = list(tdata['symbols'])
    n_atoms        = len(symbols)
    print(f"  Training data: {len(train_coords)} frames, {n_atoms} atoms")
    print(f"  Symbols: {symbols}")

    # ── 3. Equilibrium geometry ────────────────────────────────────────────────
    print("\nStep 3: Loading reference (equilibrium) geometry...")
    eq_coords_ang = np.load(args.eq_coords)
    print(f"  Loaded from: {args.eq_coords}  shape: {eq_coords_ang.shape}")
    if eq_coords_ang.shape != (n_atoms, 3):
        raise ValueError(f"eq_coords shape {eq_coords_ang.shape} != ({n_atoms}, 3)")

    # ── 4. B3LYP or CASSCF Hessian / NM modes ─────────────────────────────────
    print("\nStep 4: Getting normal mode eigenvectors...")
    casscf_eq_coords_ang = eq_coords_ang  # will update if --casscf-opt

    if args.casscf_opt:
        print("  Running CASSCF(4,4)/6-31G* geometry optimization + Hessian...")
        casscf_eq_coords_ang, hessian = optimize_casscf_psi4(
            symbols, eq_coords_ang, out_dir,
            n_threads=args.n_threads, memory=args.memory)
        np.save(str(out_dir / 'casscf_eq_coords.npy'), casscf_eq_coords_ang)
        eq_coords_ang = casscf_eq_coords_ang
        hess_label = 'CASSCF(4,4)/6-31G*'
    elif args.b3lyp_hessian:
        print(f"  Loading pre-saved B3LYP Hessian: {args.b3lyp_hessian}")
        hessian = np.load(args.b3lyp_hessian)
        hess_label = 'B3LYP/6-31G* (pre-saved)'
    else:
        print("  Computing B3LYP/6-31G* Hessian via PSI4...")
        hessian = compute_b3lyp_hessian_psi4(
            symbols, eq_coords_ang, out_dir,
            n_threads=args.n_threads, memory=args.memory)
        hess_label = 'B3LYP/6-31G* (computed)'

    if hessian.shape != (3 * n_atoms, 3 * n_atoms):
        raise ValueError(f"Hessian shape {hessian.shape} != ({3*n_atoms}, {3*n_atoms})")

    print(f"  Hessian source: {hess_label}  shape: {hessian.shape}")
    np.save(str(out_dir / 'hessian_used.npy'), hessian)

    freqs_vib, U_vib, sqrt_mass = get_nm_eigvecs(symbols, hessian)
    n_vib = len(freqs_vib)
    print(f"  Vibrational modes: {n_vib}  "
          f"freq range: {freqs_vib.min():.0f}–{freqs_vib.max():.0f} cm⁻¹")
    if np.any(freqs_vib < 0):
        n_imag = np.sum(freqs_vib < 0)
        print(f"  WARNING: {n_imag} imaginary modes "
              f"({freqs_vib[freqs_vib < 0].tolist()}) — reference is a saddle point!")

    # ── 5. Fix A: near-equilibrium CASSCF points ──────────────────────────────
    nm_fix_results = []
    if args.add_nm_points > 0:
        print(f"\nStep 5 (Fix A): generating {args.add_nm_points} near-equilibrium "
              f"CASSCF points at T={args.T_nm} K...")
        e_b3lyp_ref = min(r['e_b3lyp'] for r in results)
        nm_fix_results = add_nm_casscf_points(
            symbols, eq_coords_ang, U_vib, sqrt_mass, freqs_vib,
            T_nm=args.T_nm, n_points=args.add_nm_points,
            out_dir=out_dir, n_threads=args.n_threads, memory=args.memory,
            e_b3lyp_ref_ha=e_b3lyp_ref)
        print(f"  Fix A: {len(nm_fix_results)} NM CASSCF frames computed.")
        results = results + [r for r in nm_fix_results if r.get('e_casscf_ss') is not None]
    else:
        print("\nStep 5 (Fix A): skipped (--add-nm-points not set).")

    # Save merged results
    with open(str(out_dir / 'all_casscf_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ── 6. Filter ─────────────────────────────────────────────────────────────
    print("\nStep 6: Filtering frames...")
    gammas = [float(x) for x in args.gamma_values.split(',')]
    alphas = [float(x) for x in args.alpha_values.split(',')]

    results_filt = filter_results(results, args.max_energy, args.delta_threshold)
    if len(results_filt) < 4:
        raise ValueError(f"Only {len(results_filt)} frames after filtering — too few for KRR.")

    # ── 7. Assemble delta targets ──────────────────────────────────────────────
    print("\nStep 7: Assembling delta targets...")
    coords_arr, e_b3lyp_arr, e_cas_arr = build_training_arrays(
        results_filt, train_coords, train_energies, nm_fix_results)

    # Reference energies: use the minimum-energy frames as reference
    e_b3lyp_ref_ha = e_b3lyp_arr.min()
    e_cas_ref_ha   = e_cas_arr.min()

    dE_b3lyp_ha = e_b3lyp_arr - e_b3lyp_ref_ha   # (M,) Ha
    dE_cas_ha   = e_cas_arr   - e_cas_ref_ha       # (M,) Ha
    delta_ha    = dE_cas_ha   - dE_b3lyp_ha        # (M,) Ha  relative correction

    print(f"  Training frames: {len(delta_ha)}")
    print(f"  ΔE_B3LYP range: 0 – {dE_b3lyp_ha.max()*HARTREE_TO_KCAL:.1f} kcal/mol")
    print(f"  Delta range: {delta_ha.min()*HARTREE_TO_KCAL:.2f} – "
          f"{delta_ha.max()*HARTREE_TO_KCAL:.2f} kcal/mol")

    # ── 8. Project into NM coordinates ────────────────────────────────────────
    print("\nStep 8: Projecting geometries into NM coordinate space...")
    delta_ang_batch = coords_arr - eq_coords_ang[None, :, :]     # (M,N,3) Å
    delta_bohr_flat = delta_ang_batch.reshape(len(coords_arr), -1) * ANGSTROM_TO_BOHR
    delta_mw        = delta_bohr_flat * sqrt_mass[None, :]       # (M,3N) sqrt(amu)·Bohr
    X_q             = delta_mw @ U_vib                            # (M,n_vib)

    q_norms2 = np.sum(X_q ** 2, axis=1)
    print(f"  ||q||² range: {q_norms2.min():.4f} – {q_norms2.max():.2f}  "
          f"(amu·Bohr²)")
    print(f"  NM descriptor dimension: {X_q.shape[1]}")

    np.save(str(out_dir / 'nm_descriptors.npy'), X_q)
    np.save(str(out_dir / 'delta_ha.npy'), delta_ha)

    # ── 9. KRR grid search ────────────────────────────────────────────────────
    print(f"\nStep 9: KRR grid search "
          f"({len(gammas)} gammas × {len(alphas)} alphas = {len(gammas)*len(alphas)} fits)...")
    best_gamma, best_alpha, best_rmse, grid = grid_search_krr(X_q, delta_ha, gammas, alphas)
    print(f"\n  Best: gamma={best_gamma:.4g}  alpha={best_alpha:.2g}  "
          f"LOO-CV RMSE = {best_rmse:.4f} kcal/mol")

    # Print sorted grid summary
    grid_sorted = sorted(grid, key=lambda x: x['loo_rmse_kcal'])
    print("\n  Top 5 hyperparameter combinations:")
    for row in grid_sorted[:5]:
        print(f"    gamma={row['gamma']:.4g}  alpha={row['alpha']:.2g}  "
              f"RMSE={row['loo_rmse_kcal']:.4f} kcal/mol")

    # ── 10. Build and save model ───────────────────────────────────────────────
    print("\nStep 10: Building and saving NMKRRDeltaModel...")
    model = NMKRRDeltaModel(
        eq_coords_ang=eq_coords_ang,
        U_vib=U_vib,
        sqrt_mass=sqrt_mass,
        freqs_vib=freqs_vib,
        symbols=symbols,
        gamma=best_gamma,
        alpha_reg=best_alpha,
        X_train_q=X_q,
        y_train_ha=delta_ha,
        e_b3lyp_ref_ha=e_b3lyp_ref_ha,
        e_cas_ref_ha=e_cas_ref_ha,
        cv_rmse_kcal=best_rmse,
    )
    model_path = out_dir / 'nm_delta_model.pkl'
    model.save(str(model_path))

    # Sanity check: in-sample predictions
    y_pred_ha = np.array([model.predict_delta_ha(X_q[i]) for i in range(len(X_q))])
    train_rmse = np.sqrt(np.mean((y_pred_ha - delta_ha) ** 2)) * HARTREE_TO_KCAL
    print(f"  Train RMSE (in-sample): {train_rmse:.4f} kcal/mol")
    print(f"  LOO-CV   RMSE:          {best_rmse:.4f} kcal/mol")

    # ── 11. Diagnostics ───────────────────────────────────────────────────────
    print("\nStep 11: Generating diagnostics figure...")
    plot_diagnostics(X_q, delta_ha, freqs_vib, grid, best_gamma, best_alpha,
                     out_dir, results_filt)

    # ── 12. Summary ───────────────────────────────────────────────────────────
    summary = {
        'timestamp':           ts,
        'casscf_results_in':   args.load_results,
        'training_data':       args.training_data,
        'eq_coords':           args.eq_coords,
        'hessian_source':      hess_label,
        'n_frames_total':      len(results),
        'n_frames_filtered':   len(results_filt),
        'n_vib':               int(n_vib),
        'freqs_vib_min_cm1':   float(freqs_vib.min()),
        'freqs_vib_max_cm1':   float(freqs_vib.max()),
        'n_imag_modes':        int(np.sum(freqs_vib < 0)),
        'q_norm2_min':         float(q_norms2.min()),
        'q_norm2_max':         float(q_norms2.max()),
        'delta_min_kcal':      float(delta_ha.min() * HARTREE_TO_KCAL),
        'delta_max_kcal':      float(delta_ha.max() * HARTREE_TO_KCAL),
        'best_gamma':          best_gamma,
        'best_alpha':          best_alpha,
        'loo_cv_rmse_kcal':    best_rmse,
        'train_rmse_kcal':     float(train_rmse),
        'model_path':          str(model_path),
        'filter_max_energy_kcal': args.max_energy,
        'fix_a_nm_points':     len(nm_fix_results),
        'e_b3lyp_ref_ha':      float(e_b3lyp_ref_ha),
        'e_cas_ref_ha':        float(e_cas_ref_ha),
        'grid_search':         grid_sorted[:10],
    }
    summary_path = out_dir / 'summary.json'
    with open(str(summary_path), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done.  Output: {out_dir}")
    print(f"  Model:    {model_path}")
    print(f"  Hessian:  {out_dir / 'hessian_used.npy'}  (use with --b3lyp-hessian for reruns)")
    print(f"\nTo apply in IR-MD workflow:")
    print(f"  python3 ir_md_spectrum.py \\")
    print(f"      --model <energy_model.pkl> \\")
    print(f"      --training-data <training_with_dipoles.npz> \\")
    print(f"      --nm-delta-model {model_path} \\")
    print(f"      --steps 30000 --temp 300 --preminimize --zpe-min-freq 50 --zpe-max-freq 4000")
    print(f"\nLOO-CV RMSE: {best_rmse:.3f} kcal/mol "
          f"({'good' if best_rmse < 1.0 else 'marginal' if best_rmse < 3.0 else 'poor'} "
          f"for 300 K IR spectra where kT ≈ 0.59 kcal/mol)")


# ─────────────────────────────────────────────────────────────────────────────
# NEVPTKRRModel — two-layer delta-ML (CASSCF + SC-NEVPT2) in NM-coord space
# Defined here so it is always importable as casscf_nm_delta.NEVPTKRRModel,
# regardless of whether casscf_nevpt2_correction.py was run as __main__.
# ─────────────────────────────────────────────────────────────────────────────

class NEVPTKRRModel(NMKRRDeltaModel):
    """
    Two-layer NM-coordinate KRR delta model predicting δ_total(q) in Hartree.

    δ_total = E_NEVPT2(PySCF) − E_B3LYP(PSI4)  [referenced to equilibrium]
            = δ_CASSCF_rel + δ_NEVPT2_rel

    Inherits NMKRRDeltaModel interface (.predict / .project / .save / .load)
    so it is drop-in compatible with --nm-delta-model in ir_md_spectrum.py.

    Additional diagnostic attributes:
        y_train_casscf_ha   : (M,) relative δ_CASSCF per training frame
        y_train_nevpt2_ha   : (M,) relative δ_NEVPT2 per training frame
        e_nevpt2_ref_ha     : NEVPT2 total energy at reference geometry (Ha)
        casscf_cv_rmse_kcal : LOO-CV on δ_CASSCF alone
        nevpt2_cv_rmse_kcal : LOO-CV on δ_NEVPT2 alone
    """

    def __init__(
        self, *,
        eq_coords_ang, U_vib, sqrt_mass, freqs_vib, symbols,
        gamma, alpha_reg,
        X_train_q, y_train_ha,
        y_train_casscf_ha=None,
        y_train_nevpt2_ha=None,
        e_b3lyp_ref_ha=0.0,
        e_cas_ref_ha=0.0,
        e_nevpt2_ref_ha=0.0,
        cv_rmse_kcal=None,
        casscf_cv_rmse_kcal=None,
        nevpt2_cv_rmse_kcal=None,
    ):
        super().__init__(
            eq_coords_ang=eq_coords_ang,
            U_vib=U_vib,
            sqrt_mass=sqrt_mass,
            freqs_vib=freqs_vib,
            symbols=symbols,
            gamma=gamma,
            alpha_reg=alpha_reg,
            X_train_q=X_train_q,
            y_train_ha=y_train_ha,
            e_b3lyp_ref_ha=e_b3lyp_ref_ha,
            e_cas_ref_ha=e_cas_ref_ha,
            cv_rmse_kcal=cv_rmse_kcal,
        )
        self.y_train_casscf_ha   = (np.asarray(y_train_casscf_ha, dtype=float)
                                    if y_train_casscf_ha is not None else None)
        self.y_train_nevpt2_ha   = (np.asarray(y_train_nevpt2_ha, dtype=float)
                                    if y_train_nevpt2_ha is not None else None)
        self.e_nevpt2_ref_ha     = float(e_nevpt2_ref_ha)
        self.casscf_cv_rmse_kcal = casscf_cv_rmse_kcal
        self.nevpt2_cv_rmse_kcal = nevpt2_cv_rmse_kcal


if __name__ == '__main__':
    main()
