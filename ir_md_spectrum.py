#!/usr/bin/env python3
"""
ML-PES IR Spectrum via Dipole Autocorrelation

Workflow:
  1. Train ML dipole surface (KRR) on PSI4 training data
  2. Run dense ML-MD using the refined ML-PES
  3. Predict dipoles at every MD frame using the ML dipole surface
  4. Compute dipole ACF and FFT → IR spectrum
  5. Save spectrum as CSV and generate diagnostic figure

Usage:
  python3 ir_md_spectrum.py \\
      --model outputs/nm_training_20260308_203606/mlpes_model_nm.pkl \\
      --training-data outputs/clean_psi410_20260308_203552/training_data.npz \\
      --steps 10000 --temp 300

  # Re-use a previously trained dipole model:
  python3 ir_md_spectrum.py \\
      --model <pes.pkl> --training-data <data.npz> \\
      --dipole-model <dipole.pkl> --steps 20000 --temp 300

Units: coordinates Angstrom, energies Hartree, dipoles Debye,
       time femtoseconds.  (See CLAUDE.md for full table.)
"""

import sys
import os
import argparse
import csv
import json
import pickle
import datetime
import numpy as np
from pathlib import Path

# -------------------------------------------------------------------------
# Path setup
# -------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'modules'))

# -------------------------------------------------------------------------
# Physical constants (from direct_md.py — do not redefine)
# -------------------------------------------------------------------------
KB_HARTREE_PER_K = 3.1668114e-6
AMU_TO_AU        = 1822.888486
FS_TO_AU         = 41.341374575751
BOHR_TO_ANG      = 0.529177210903
ANG_TO_BOHR      = 1.0 / BOHR_TO_ANG
HARTREE_TO_KCAL  = 627.509474
CM_INV_PER_AU    = 219474.63       # a.u. frequency → cm⁻¹
FREQ_CONV        = 5140.48         # sqrt(Hartree/(Bohr²·amu)) → cm⁻¹

ATOMIC_MASSES = {
    'H': 1.00794, 'He': 4.002602, 'C': 12.011, 'N': 14.007,
    'O': 15.999,  'F': 18.9984,   'S': 32.06,  'Cl': 35.453,
}

# Lookup table: Hill-notation formula → common name
_MOLECULE_NAMES: dict = {
    'CH2O2':  'Criegee intermediate (CH₂OO)',
    'CH2O':   'formaldehyde',
    'H2O':    'water',
    'H2O2':   'hydrogen peroxide',
    'CH4':    'methane',
    'CO2':    'carbon dioxide',
    'NH3':    'ammonia',
    'C2H4':   'ethylene',
    'C2H2':   'acetylene',
    'C2H6':   'ethane',
    'C6H6':   'benzene',
    'CH3OH':  'methanol',
    'HNO3':   'nitric acid',
    'O3':     'ozone',
    'SO2':    'sulfur dioxide',
    'C4H6O2': 'methyl vinyl ketone oxide (MVKO)',
}

# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------
from data_formats import TrajectoryData, load_trajectory
from ml_pes import MLPESTrainer
from ir_spectroscopy import DipoleSurface, IRSpectrumCalculator
from normal_modes import compute_normal_modes
from bakken import (
    MLPESDriver,
    minimize_geometry,
    maxwell_boltzmann_velocities   as _maxwell_boltzmann,
    zpe_initialized_velocities     as _zpe_initialized_velocities,
    kinetic_temperature            as _kin_temp,
    run_md                         as _run_md_bakken,
)


# =============================================================================
# Molecule identification
# =============================================================================

def _hill_formula(symbols: list) -> str:
    """Return Hill-notation stoichiometry string (C, H first; rest alphabetical)."""
    from collections import Counter
    import re
    counts = Counter(str(s) for s in symbols)
    parts = []
    for el in ['C', 'H']:
        if el in counts:
            n = counts.pop(el)
            parts.append(el if n == 1 else f'{el}{n}')
    for el in sorted(counts):
        n = counts[el]
        parts.append(el if n == 1 else f'{el}{n}')
    return ''.join(parts)


def _unicode_subscripts(hill: str) -> str:
    """'CH2O2' → 'CH₂O₂' using Unicode subscript digits."""
    import re
    sub = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
    return re.sub(r'\d+', lambda m: m.group().translate(sub), hill)


def identify_molecule(symbols: list, coords: np.ndarray | None = None) -> dict:
    """
    Identify molecule from atomic symbols (and optionally coordinates).

    Returns a dict:
        hill    : 'CH2O2'
        unicode : 'CH₂O₂'
        name    : 'Criegee intermediate (CH₂OO)'  (or hill if unknown)
        label   : 'CH₂O₂ – Criegee intermediate (CH₂OO)'  for figure titles
        n_atoms : 5
    """
    hill = _hill_formula(list(symbols))
    uni  = _unicode_subscripts(hill)
    name = _MOLECULE_NAMES.get(hill, hill)
    if name == hill:
        label = uni
    else:
        label = f'{uni}  –  {name}'
    return {
        'hill':    hill,
        'unicode': uni,
        'name':    name,
        'label':   label,
        'n_atoms': len(symbols),
    }


# =============================================================================
# XYZ trajectory output
# =============================================================================

def save_trajectory_xyz(coords_traj: np.ndarray,
                         symbols: list,
                         times_fs: np.ndarray,
                         energies_ml: np.ndarray,
                         output_path: str,
                         mol_info: dict | None = None) -> None:
    """
    Save a multi-frame ML-MD trajectory to an extended .xyz file.

    Each frame is written as:
        <n_atoms>
        Frame N  t=X.XXfs  E=Y.YYYYYYHa  molecule=FORMULA
        El  x  y  z
        ...

    The comment line is compatible with ASE, OVITO, VMD, and Avogadro.

    Args:
        coords_traj  : (n_frames, n_atoms, 3) Angstrom
        symbols      : list of atomic symbols, length n_atoms
        times_fs     : (n_frames,) simulation time in fs
        energies_ml  : (n_frames,) ML-PES energies in Hartree
        output_path  : destination .xyz file path
        mol_info     : dict from identify_molecule() — used for the comment line
    """
    n_frames = len(coords_traj)
    n_atoms  = len(symbols)
    mol_label = mol_info['hill'] if mol_info else ''.join(str(s) for s in symbols)

    with open(output_path, 'w') as fh:
        for i, (coords, t, e) in enumerate(zip(coords_traj, times_fs, energies_ml)):
            fh.write(f'{n_atoms}\n')
            fh.write(
                f'Frame={i}  time={t:.3f}fs  energy={e:.8f}Ha  molecule={mol_label}\n'
            )
            for sym, (x, y, z) in zip(symbols, coords):
                fh.write(f'{sym:<2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}\n')

    size_kb = Path(output_path).stat().st_size / 1024
    print(f"\n  Trajectory XYZ     : {output_path}  ({n_frames} frames, {size_kb:.0f} KB)")


# MLPESDriver, minimize_geometry, and the core MD engine are provided by
# modules/bakken.py (Norwegian for "hill").  They are imported at the top
# of this file; no local definitions are needed.


# =============================================================================
# PESFamilyDriver — wraps PESFamily with the MLPESDriver interface
# =============================================================================

class PESFamilyDriver:
    """
    Thin adapter that gives a PESFamily the same interface as MLPESDriver
    so it can be dropped into run_ir_workflow as a drop-in replacement.

    FD forces only (no analytic Hessian for blended surface).
    `_has_analytic` is set to False so the Hessian path falls back to FD.
    """

    def __init__(self, family, delta_fd: float = 0.005):
        from modules.pes_family import PESFamily
        self.family       = family
        self.symbols      = family.symbols
        self.n_atoms      = len(self.symbols)
        self.masses       = np.array([ATOMIC_MASSES[s] for s in self.symbols])
        self._has_analytic = False
        self._delta_fd    = delta_fd

    def energy(self, coords: np.ndarray) -> float:
        return float(self.family.blend_energy(coords))

    def forces(self, coords: np.ndarray, delta: float = None) -> np.ndarray:
        dx = delta if delta is not None else self._delta_fd
        forces = np.zeros_like(coords)
        for a in range(self.n_atoms):
            for j in range(3):
                cp = coords.copy(); cp[a, j] += dx
                cm = coords.copy(); cm[a, j] -= dx
                forces[a, j] = -(self.family.blend_energy(cp) -
                                  self.family.blend_energy(cm)) / (2 * dx)
        return forces

    def predict(self, symbols, coords) -> float:
        return self.energy(coords)


# =============================================================================
# DeltaMLPESDriver — adds a CASSCF delta-ML correction to any base driver
# =============================================================================

class DeltaMLPESDriver:
    """
    Wraps a base ML-PES driver (MLPESDriver or PESFamilyDriver) and adds a
    delta-ML correction trained on CASSCF(4,4) − B3LYP relative energies.

    Corrected energy:
        E_corr(R) = E_base(R) + E_delta(R)

    Forces are computed as the sum of base forces and FD-differentiated delta
    forces (delta model is cheap — 39-point KRR — so FD adds negligible cost):
        F_corr(R) = F_base(R) + F_delta(R)

    The delta model predicts relative corrections in Hartree. The absolute
    CASSCF−B3LYP offset cancels in force differences and does not affect MD.
    """

    def __init__(self, base_driver, delta_model_path: str,
                 delta_fd: float = 0.005):
        self._base        = base_driver
        self._delta_model = MLPESTrainer.load(delta_model_path)
        self.symbols      = base_driver.symbols
        self.n_atoms      = base_driver.n_atoms
        self.masses       = base_driver.masses
        self._has_analytic = False   # delta forces via FD only
        self._delta_fd    = delta_fd
        print(f"  Delta-ML correction loaded: {delta_model_path}")

    def _delta_energy(self, coords: np.ndarray) -> float:
        return float(self._delta_model.predict(self.symbols, coords))

    def energy(self, coords: np.ndarray) -> float:
        return self._base.energy(coords) + self._delta_energy(coords)

    def forces(self, coords: np.ndarray, delta: float = None) -> np.ndarray:
        dx = delta if delta is not None else self._delta_fd
        # Base forces (may be analytic if base driver supports it)
        f_base = self._base.forces(coords) if delta is None else self._base.forces(coords, delta)
        # Delta forces via FD on delta model only (fast — small KRR)
        f_delta = np.zeros_like(coords)
        for a in range(self.n_atoms):
            for j in range(3):
                cp = coords.copy(); cp[a, j] += dx
                cm = coords.copy(); cm[a, j] -= dx
                f_delta[a, j] = -(self._delta_energy(cp) -
                                   self._delta_energy(cm)) / (2 * dx)
        return f_base + f_delta

    def predict(self, symbols, coords) -> float:
        return self.energy(coords)


# =============================================================================
# NMDeltaDriver — CASSCF correction using normal-mode coordinate KRR
# =============================================================================

class NMDeltaDriver:
    """
    Applies a CASSCF delta correction from a NMKRRDeltaModel (casscf_nm_delta.py).

    The NMKRRDeltaModel maps each geometry to mass-weighted NM displacements
    q = U_vib^T · M^{1/2} · (R − R_ref) and evaluates a KRR model in that
    space.  Because q = 0 at the reference geometry and ||q||² grows with
    distortion, the RBF kernel localises correctly — no Coulomb clustering.

    Forces are computed via FD on the combined (base + delta) surface.
    The FD cost is small: one extra KRR forward pass per atom per direction
    (18 × 6 = 108 KRR evaluations vs 36 for the base FD alone).
    """

    def __init__(self, base_driver, nm_delta_model_path: str,
                 delta_fd: float = 0.005):
        from casscf_nm_delta import NMKRRDeltaModel
        # NEVPTKRRModel is a subclass saved by casscf_nevpt2_correction.py;
        # importing it here registers it so pickle can reconstruct it.
        try:
            from casscf_nevpt2_correction import NEVPTKRRModel  # noqa: F401
        except ImportError:
            pass
        self._base         = base_driver
        self._nm_model     = NMKRRDeltaModel.load(nm_delta_model_path)
        self.symbols       = base_driver.symbols
        self.n_atoms       = base_driver.n_atoms
        self.masses        = base_driver.masses
        self._has_analytic = False
        self._delta_fd     = delta_fd
        print(f"  NM delta-ML model loaded: {nm_delta_model_path}")
        print(f"    KRR gamma={self._nm_model.gamma:.4g}  "
              f"alpha={self._nm_model.alpha_reg:.2g}  "
              f"n_train={len(self._nm_model.X_train_q)}  "
              f"n_vib={self._nm_model.U_vib.shape[1]}  "
              f"LOO-CV RMSE={self._nm_model.cv_rmse_kcal:.3f} kcal/mol"
              if self._nm_model.cv_rmse_kcal else "")

    def _delta_energy(self, coords: np.ndarray) -> float:
        return self._nm_model.predict(self.symbols, coords)

    def energy(self, coords: np.ndarray) -> float:
        return self._base.energy(coords) + self._delta_energy(coords)

    def forces(self, coords: np.ndarray, delta: float = None) -> np.ndarray:
        dx = delta if delta is not None else self._delta_fd
        f_base  = self._base.forces(coords) if delta is None else self._base.forces(coords, delta)
        f_delta = np.zeros_like(coords)
        for a in range(self.n_atoms):
            for j in range(3):
                cp = coords.copy(); cp[a, j] += dx
                cm = coords.copy(); cm[a, j] -= dx
                f_delta[a, j] = -(self._delta_energy(cp) - self._delta_energy(cm)) / (2 * dx)
        return f_base + f_delta

    def predict(self, symbols, coords) -> float:
        return self.energy(coords)


# =============================================================================
# EnergyDeltaDriver — 1D spline CASSCF correction as function of ΔE_B3LYP
# =============================================================================

class EnergyDeltaDriver:
    """
    Applies a CASSCF delta correction parameterised as a 1D function of the
    B3LYP relative energy:  δ(R) = f( E_base(R) − E_base_min )

    This sidesteps the Coulomb-descriptor clustering problem (all geometries
    look nearly identical in Coulomb space) by using the B3LYP relative energy
    as the sole input to the correction.  The spline is constrained to δ(0)=0
    so the equilibrium geometry is unperturbed.

    The correction file is a JSON produced by casscf_surface_correction.py:
        {"dE_b3lyp_kcal": [...], "delta_kcal": [...], "E_base_min_Ha": float}
    """

    HARTREE_TO_KCAL = 627.509

    def __init__(self, base_driver, delta_json_path: str,
                 delta_fd: float = 0.005,
                 max_dE_kcal: float = 40.0):
        from scipy.interpolate import CubicSpline
        import json

        self._base         = base_driver
        self.symbols       = base_driver.symbols
        self.n_atoms       = base_driver.n_atoms
        self.masses        = base_driver.masses
        self._has_analytic = False
        self._delta_fd     = delta_fd

        with open(delta_json_path) as f:
            dat = json.load(f)

        dE   = np.array(dat['dE_b3lyp_kcal'])
        delt = np.array(dat['delta_kcal'])

        # Only use points in the thermally relevant range and add the
        # exact anchor δ(0)=0.
        mask = dE <= max_dE_kcal
        dE_fit   = np.concatenate([[0.0], dE[mask]])
        delt_fit = np.concatenate([[0.0], delt[mask]])
        order = np.argsort(dE_fit)
        dE_fit, delt_fit = dE_fit[order], delt_fit[order]

        # Remove duplicate x-values
        _, u = np.unique(dE_fit, return_index=True)
        dE_fit, delt_fit = dE_fit[u], delt_fit[u]

        self._spline     = CubicSpline(dE_fit, delt_fit,
                                       bc_type=((1, 0.0), 'not-a-knot'))
        self._E_min      = dat.get('E_base_min_Ha', None)
        self._max_dE     = max_dE_kcal
        print(f"  Energy-delta correction loaded: {delta_json_path}")
        print(f"  Spline over {len(dE_fit)} points, ΔE_B3LYP ≤ {max_dE_kcal:.0f} kcal/mol")

    def _delta_energy(self, coords: np.ndarray) -> float:
        """Return delta correction in Hartree for given coordinates."""
        E_base = self._base.energy(coords)
        if self._E_min is None:
            return 0.0
        dE_kcal = (E_base - self._E_min) * self.HARTREE_TO_KCAL
        dE_kcal = min(dE_kcal, self._max_dE)  # clamp beyond training range
        dE_kcal = max(dE_kcal, 0.0)
        return float(self._spline(dE_kcal)) / self.HARTREE_TO_KCAL

    def energy(self, coords: np.ndarray) -> float:
        return self._base.energy(coords) + self._delta_energy(coords)

    def forces(self, coords: np.ndarray, delta: float = None) -> np.ndarray:
        dx = delta if delta is not None else self._delta_fd
        f_base = self._base.forces(coords) if delta is None else self._base.forces(coords, delta)
        f_delta = np.zeros_like(coords)
        for a in range(self.n_atoms):
            for j in range(3):
                cp = coords.copy(); cp[a, j] += dx
                cm = coords.copy(); cm[a, j] -= dx
                f_delta[a, j] = -(self._delta_energy(cp) -
                                   self._delta_energy(cm)) / (2 * dx)
        return f_base + f_delta

    def predict(self, symbols, coords) -> float:
        return self.energy(coords)


# =============================================================================
# ML-PES normal mode analysis (numerical or analytic Hessian)
# =============================================================================

def compute_mlpes_normal_modes(driver: MLPESDriver,
                                coords_eq: np.ndarray,
                                delta: float = 0.01,
                                analytic: bool = False) -> tuple:
    """
    Compute vibrational normal modes of the ML-PES at the equilibrium geometry.

    Two Hessian routes are available:

    Numerical (default, analytic=False):
        Central FD of ML forces with step δ (Hartree/Å²).
        Costs 6N energy+force evaluations; susceptible to KRR extrapolation
        artefacts that inflate high-frequency modes.

    Analytic (analytic=True):
        Exact chain rule through StandardScaler → RBF kernel → Coulomb matrix:
            H[q,r] = σ_y(-2γ)[Σ_k g_k J2_sc[k,q,r]
                               + J_sc.T (-2γ H_desc + E_sc I) J_sc [q,r]]
        Single forward pass; gives the true curvature of the KRR surface.
        Requires driver._has_analytic = True (sklearn KernelRidge attributes
        dual_coef_ and X_fit_ available).

    Both routes convert Hartree/Å² → Hartree/Bohr² before diagonalisation.

    Args:
        driver    : loaded MLPESDriver
        coords_eq : equilibrium geometry (n_atoms, 3) Angstrom
        delta     : FD displacement step in Angstrom (numerical route only)
        analytic  : use analytic Hessian instead of finite differences

    Returns:
        (frequencies, eigvecs_mw, eigenvalues, mass_vec)
          frequencies : (n_vib,)    cm⁻¹ (ascending)
          eigvecs_mw  : (3N, n_vib) mass-weighted eigenvectors
          eigenvalues : (n_vib,)    Hartree/(Bohr²·amu)
          mass_vec    : (3N,)       amu
    """
    n_atoms = len(driver.symbols)
    n_dof   = 3 * n_atoms

    if analytic and driver._has_analytic:
        print(f"\n  Computing ML-PES Hessian ({n_dof}×{n_dof}, analytic KRR) ...")
        # analytic_hessian returns Hartree/Ang²
        H_ang2 = driver.analytic_hessian(coords_eq)
        H_ang2 = 0.5 * (H_ang2 + H_ang2.T)   # enforce symmetry numerically
    else:
        if analytic:
            print("  [analytic Hessian unavailable; falling back to FD]")
        print(f"\n  Computing ML-PES Hessian ({n_dof}×{n_dof}, δ={delta} Å) ...")
        H_ang2 = np.zeros((n_dof, n_dof))   # Hartree / Angstrom²
        for i in range(n_dof):
            cp = coords_eq.flatten().copy(); cp[i] += delta
            cm = coords_eq.flatten().copy(); cm[i] -= delta
            F_p = driver.forces(cp.reshape(n_atoms, 3)).flatten()
            F_m = driver.forces(cm.reshape(n_atoms, 3)).flatten()
            H_ang2[:, i] = -(F_p - F_m) / (2.0 * delta)
        H_ang2 = 0.5 * (H_ang2 + H_ang2.T)

    # Unit conversion:  Hartree/Ang²  →  Hartree/Bohr²
    # 1 Å = ANG_TO_BOHR Bohr → 1 Ha/Å² = 1/ANG_TO_BOHR² Ha/Bohr²
    H_bohr2 = H_ang2 / ANG_TO_BOHR ** 2

    frequencies, eigvecs_mw, eigenvalues, mass_vec = compute_normal_modes(
        driver.symbols, H_bohr2
    )

    # Report frequencies and ZPE per mode
    print(f"\n  {'Mode':>5}  {'Freq (cm⁻¹)':>12}  {'ZPE (kcal/mol)':>15}")
    print(f"  {'─'*5}  {'─'*12}  {'─'*15}")
    zpe_total = 0.0
    for k, freq in enumerate(frequencies):
        zpe_k = 0.5 * abs(freq) / CM_INV_PER_AU * HARTREE_TO_KCAL   # kcal/mol
        zpe_total += zpe_k
        print(f"  {k+1:>5}  {freq:>12.1f}  {zpe_k:>15.3f}")
    print(f"  {'Total ZPE':>18}  {zpe_total:>15.3f} kcal/mol")

    return frequencies, eigvecs_mw, eigenvalues, mass_vec


# =============================================================================
# MD engine shim — delegates to bakken (modules/bakken.py)
# =============================================================================

def run_ml_md_dense(driver: MLPESDriver,
                    coords0: np.ndarray,
                    n_steps: int,
                    temperature: float,
                    timestep: float = 0.5,
                    save_every: int = 1,
                    thermostat_tau: float = 100.0,
                    seed: int = 42,
                    nm_data: tuple | None = None,
                    min_freq_zpe: float = 50.0,
                    max_freq_zpe: float = 4000.0,
                    preminimize: bool = False,
                    preminimize_steps: int = 300,
                    preminimize_tol: float = 0.005,
                    max_bond_extension: float = 0.0,
                    monitor_bonds: list = None,
                    print_every: int = 0) -> dict:
    """
    Velocity-Verlet ML-MD via the bakken engine (modules/bakken.py).

    When preminimize=True, runs steepest-descent on the ML-PES before
    velocity initialisation so the Hessian is evaluated at a true
    stationary point, preventing unphysical high-frequency artifacts.

    max_bond_extension: if > 0, stop when any covalent bond (initial length
        < 2.0 Å) extends beyond this multiple of its initial length.
        2.5 is a reasonable default (allows large-amplitude vibrations without
        permitting actual bond dissociation).  0 disables the check.
    monitor_bonds: list of (i, j) or (i, j, label) tuples — atom pairs whose
        distances are saved at every frame and printed periodically.
    print_every: print one-line diagnostic every N steps (0 = disabled).
    """
    return _run_md_bakken(
        driver, coords0, n_steps, temperature,
        timestep=timestep,
        save_every=save_every,
        thermostat_tau=thermostat_tau,
        seed=seed,
        nm_data=nm_data,
        min_freq_zpe=min_freq_zpe,
        max_freq_zpe=max_freq_zpe,
        preminimize=preminimize,
        preminimize_steps=preminimize_steps,
        preminimize_tol=preminimize_tol,
        max_bond_extension=max_bond_extension,
        monitor_bonds=monitor_bonds,
        print_every=print_every,
    )


# =============================================================================
# Dipole surface training
# =============================================================================

def train_dipole_surface(training_data_path: str,
                         output_path: str) -> DipoleSurface:
    """
    Train a KRR DipoleSurface on PSI4 training data.

    Args:
        training_data_path : .npz file with 'symbols', 'coordinates', 'dipoles'
        output_path        : where to save the trained model (.pkl)

    Returns:
        Trained DipoleSurface
    """
    print("\n" + "=" * 70)
    print("  TRAINING ML DIPOLE SURFACE")
    print("=" * 70)

    data = np.load(training_data_path, allow_pickle=True)
    symbols     = list(data['symbols'])
    coordinates = data['coordinates']   # (N, n_atoms, 3)  Angstrom
    dipoles     = data['dipoles']        # (N, 3)            Debye

    # Check for valid dipoles (PSI4 returns zeros if calculation failed)
    norms = np.linalg.norm(dipoles, axis=1)
    valid = norms > 1e-6
    n_valid = valid.sum()
    print(f"  Training frames    : {len(coordinates)}")
    print(f"  Valid dipoles      : {n_valid} (|μ| > 1e-6 D)")
    if n_valid < 10:
        raise RuntimeError(f"Too few valid dipoles ({n_valid}) for training")

    coords_v  = coordinates[valid]
    dipoles_v = dipoles[valid]

    print(f"  Dipole range (D)   : "
          f"μx [{dipoles_v[:,0].min():.3f}, {dipoles_v[:,0].max():.3f}]  "
          f"μy [{dipoles_v[:,1].min():.3f}, {dipoles_v[:,1].max():.3f}]  "
          f"μz [{dipoles_v[:,2].min():.3f}, {dipoles_v[:,2].max():.3f}]")

    surface = DipoleSurface(symbols)
    stats = surface.train(coords_v, dipoles_v, verbose=True)
    surface.save(output_path)

    print(f"\n  Dipole model saved : {output_path}")
    return surface


def _train_nm_dipole_surface(training_data_path: str,
                              output_path: str,
                              nm_pes_driver) -> 'NMDipoleSurface':
    """
    Train an NMDipoleSurface on PSI4 training data using NM coordinates
    from an existing NMPESDriver.

    Uses analytic LOO-CV for γ/α selection (median heuristic for γ centre).
    Expected to exceed the R²≈0.91 Coulomb+KRR ceiling by encoding C-H
    stretch modes directly as non-zero descriptors.
    """
    from modules.nm_pes import NMDipoleSurface

    print("\n" + "=" * 70)
    print("  TRAINING NM-COORDINATE DIPOLE SURFACE (NMDipoleSurface)")
    print("=" * 70)

    data        = np.load(training_data_path, allow_pickle=True)
    coordinates = data['coordinates']   # (N, n_atoms, 3)
    dipoles     = data['dipoles']       # (N, 3) Debye

    norms = np.linalg.norm(dipoles, axis=1)
    valid = norms > 1e-6
    n_valid = valid.sum()
    print(f"  Training frames    : {len(coordinates)}")
    print(f"  Valid dipoles      : {n_valid} (|μ| > 1e-6 D)")
    if n_valid < 10:
        raise RuntimeError(f"Too few valid dipoles ({n_valid}) for training")

    coords_v  = coordinates[valid]
    dipoles_v = dipoles[valid]
    print(f"  |μ| range          : {np.linalg.norm(dipoles_v, axis=1).min():.3f}–"
          f"{np.linalg.norm(dipoles_v, axis=1).max():.3f} D")

    nm = nm_pes_driver._model   # NMKRRPESModel
    surface = NMDipoleSurface.from_nm_pes_model(nm)
    surface.fit(coords_v, dipoles_v, verbose=True)
    surface.save(output_path)

    print(f"\n  NM dipole model saved : {output_path}")
    return surface


# =============================================================================
# Dipole prediction along trajectory
# =============================================================================

def predict_trajectory_dipoles(surface: DipoleSurface,
                                coords_traj: np.ndarray) -> np.ndarray:
    """
    Predict dipole vectors for every frame in a trajectory.

    Args:
        surface     : trained DipoleSurface
        coords_traj : (n_frames, n_atoms, 3) Angstrom

    Returns:
        dipoles_traj : (n_frames, 3) Debye
    """
    from tqdm import tqdm
    print(f"\n  Predicting dipoles for {len(coords_traj)} frames...")
    dipoles = []
    for coords in tqdm(coords_traj, desc="Dipole prediction"):
        dipoles.append(surface.predict(coords))
    return np.array(dipoles)


# =============================================================================
# IR spectrum
# =============================================================================

def compute_ir_spectrum(dipoles_traj: np.ndarray,
                        timestep_fs: float,
                        save_every: int,
                        temperature: float,
                        max_freq: float = 4500.0,
                        window: str = 'hann',
                        zero_padding: int = 4) -> tuple:
    """
    Compute IR spectrum from ML-dipole trajectory.

    The effective time step between saved frames = timestep_fs * save_every.

    Returns:
        (frequencies cm⁻¹, intensities a.u., acf_lags, acf_values)
    """
    dt_frame = timestep_fs * save_every   # effective Δt in fs

    print("\n" + "=" * 70)
    print("  IR SPECTRUM via DIPOLE ACF + FFT")
    print("=" * 70)
    print(f"  Frames used       : {len(dipoles_traj)}")
    print(f"  Effective Δt      : {dt_frame:.2f} fs")
    total_t = len(dipoles_traj) * dt_frame
    nyquist  = 1.0 / (2.0 * dt_frame * 1e-15) / 2.998e10  # cm⁻¹
    print(f"  Total trajectory  : {total_t:.0f} fs")
    print(f"  Nyquist frequency : {nyquist:.0f} cm⁻¹")

    calc = IRSpectrumCalculator(temperature=temperature)
    frequencies, intensities = calc.compute_ir_spectrum(
        dipoles_traj,
        timestep=dt_frame,
        max_freq=max_freq,
        window=window,
        zero_padding=zero_padding,
        verbose=True,
    )

    peaks = calc.find_peaks(threshold=0.05, verbose=True)

    # Also return raw ACF
    acf_lags, acf_values = calc.compute_autocorrelation(
        dipoles_traj, max_lag=len(dipoles_traj) // 2, verbose=False
    )

    return frequencies, intensities, acf_lags * dt_frame, acf_values, peaks


# =============================================================================
# CSV output
# =============================================================================

def save_spectrum_csv(frequencies: np.ndarray,
                      intensities: np.ndarray,
                      output_path: str,
                      metadata: dict) -> None:
    """Save IR spectrum to CSV file."""
    with open(output_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        # Header comments
        writer.writerow([f'# IR spectrum — {metadata.get("molecule", "CH2OO")}'])
        writer.writerow([f'# Temperature: {metadata.get("temperature", "?")} K'])
        writer.writerow([f'# MD steps: {metadata.get("n_steps", "?")}  '
                         f'Δt_eff: {metadata.get("dt_eff_fs", "?")} fs'])
        writer.writerow([f'# Method: ML-PES (KRR) + ML dipole surface (KRR)'])
        writer.writerow([f'# Date: {metadata.get("date", "")}'])
        writer.writerow(['frequency_cm-1', 'intensity_normalized'])
        for freq, inten in zip(frequencies, intensities):
            writer.writerow([f'{freq:.4f}', f'{inten:.8f}'])

    print(f"\n  Spectrum CSV saved : {output_path}")
    print(f"  Points             : {len(frequencies)}")
    print(f"  Frequency range    : {frequencies.min():.1f} – {frequencies.max():.1f} cm⁻¹")


# =============================================================================
# Standalone IR spectrum figure (publication-quality)
# =============================================================================

def plot_ir_spectrum(frequencies: np.ndarray,
                     intensities: np.ndarray,
                     acf_times: np.ndarray,
                     acf_values: np.ndarray,
                     peaks: list,
                     output_path: str,
                     temperature: float,
                     n_steps: int,
                     timestep: float,
                     dt_eff: float,
                     n_frames: int,
                     molecule: str = 'CH2OO',
                     nm_frequencies: np.ndarray | None = None) -> None:
    """
    Publication-quality two-panel IR spectrum figure.

    Top panel  : IR spectrum (cm⁻¹ vs normalised intensity)
                 with ZPE-floor ML-PES harmonic frequencies shown as
                 dashed reference lines and peak labels.
    Bottom panel: Dipole autocorrelation (first 500 fs of lag).

    Args:
        nm_frequencies : (n_vib,) cm⁻¹ normal mode frequencies from
                         compute_mlpes_normal_modes().  Drawn as vertical
                         dashed blue lines on the spectrum panel.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.40,
                           height_ratios=[3, 1],
                           left=0.10, right=0.96, top=0.91, bottom=0.09)

    total_ps = n_steps * timestep / 1000.0
    fig.suptitle(
        f"IR Spectrum  ·  {molecule}  ·  {temperature:.0f} K  ·  "
        f"{total_ps:.1f} ps ML-MD  ·  ML-PES KRR + ML Dipole KRR",
        fontsize=11, fontweight='bold',
    )

    # ── Top panel: IR spectrum ──────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])

    ax0.plot(frequencies, intensities, color='#c0392b', lw=1.5, zorder=3)
    ax0.fill_between(frequencies, 0, intensities, alpha=0.18,
                     color='#c0392b', zorder=2)

    # Normal mode harmonic reference lines
    if nm_frequencies is not None:
        labeled = False
        for freq in nm_frequencies:
            if 50 < freq < frequencies.max():
                lbl = 'Harmonic NM (ML-PES)' if not labeled else '_nolegend_'
                ax0.axvline(freq, color='steelblue', lw=1.0, ls='--',
                            alpha=0.65, zorder=1, label=lbl)
                labeled = True
        ax0.legend(fontsize=9, loc='upper right', framealpha=0.8)

    # Annotate top peaks
    top_peaks = sorted(peaks, key=lambda p: p[1], reverse=True)[:10]
    for freq, inten in top_peaks:
        if inten > 0.06:
            ax0.annotate(
                f'{freq:.0f}',
                xy=(freq, inten),
                xytext=(0, 9), textcoords='offset points',
                ha='center', fontsize=8, color='#7b241c', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#922b21', lw=0.7),
            )

    ax0.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11)
    ax0.set_ylabel('Intensity (arb. units)', fontsize=11)
    ax0.set_xlim(0, min(float(frequencies.max()), 4500.0))
    ax0.set_ylim(bottom=0)
    ax0.grid(True, alpha=0.25)
    ax0.tick_params(labelsize=10)

    # ── Bottom panel: dipole ACF ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    n_show = min(len(acf_times), int(500.0 / max(dt_eff, 0.5)) + 1)
    ax1.plot(acf_times[:n_show], acf_values[:n_show],
             color='#1a5276', lw=1.2)
    ax1.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.7)
    ax1.set_xlabel('Lag time (fs)', fontsize=10)
    ax1.set_ylabel('C(t) / C(0)', fontsize=10)
    ax1.set_title('Dipole Autocorrelation Function', fontsize=9)
    ax1.set_xlim(0, acf_times[n_show - 1])
    ax1.grid(True, alpha=0.25)
    ax1.tick_params(labelsize=9)

    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  IR spectrum figure  : {output_path}")


# =============================================================================
# Diagnostic figure
# =============================================================================

def plot_ir_diagnostics(md_data: dict,
                        dipoles_traj: np.ndarray,
                        frequencies: np.ndarray,
                        intensities: np.ndarray,
                        acf_times: np.ndarray,
                        acf_values: np.ndarray,
                        peaks: list,
                        dipole_surface: DipoleSurface,
                        training_data_path: str,
                        output_path: str,
                        nm_frequencies: np.ndarray | None = None) -> None:
    """
    6-panel diagnostic figure:
      [0,0] ML energy trajectory
      [0,1] Dipole components over time
      [0,2] Dipole autocorrelation function
      [1,0] IR spectrum with peak labels
      [1,1] Dipole surface parity (μ predicted vs PSI4)
      [1,2] Run summary
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    times_fs   = md_data['times_fs']
    energies   = md_data['energies_ml']
    symbols    = md_data['symbols']
    temperature = md_data['temperature']
    n_steps    = md_data['n_steps']
    timestep   = md_data['timestep']
    save_every = md_data['save_every']
    dt_eff     = timestep * save_every

    # Relative energies
    e_rel = (energies - energies.min()) * HARTREE_TO_KCAL

    # Dipole magnitude
    dip_mag = np.linalg.norm(dipoles_traj, axis=1)

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#f8f9fa')
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.42, wspace=0.38,
                           left=0.07, right=0.97, top=0.92, bottom=0.07)

    mol_info = identify_molecule(symbols)
    fig.suptitle(
        f"ML-PES IR Spectrum  ·  {mol_info['label']}  ·  "
        f"{temperature:.0f} K  ·  {n_steps} steps × {timestep} fs",
        fontsize=12, fontweight='bold', y=0.97,
    )

    # ── [0,0] ML energy trajectory ────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(times_fs, e_rel, color='steelblue', lw=0.8)
    ax0.set_xlabel('Time (fs)')
    ax0.set_ylabel('Relative energy (kcal/mol)')
    ax0.set_title('ML-PES Energy Trajectory', fontsize=10)
    ax0.grid(True, alpha=0.3)
    ax0.set_xlim(times_fs[0], times_fs[-1])

    # ── [0,1] Dipole components vs time ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    colors_mu = ['tab:blue', 'tab:orange', 'tab:green']
    labels_mu = ['μx', 'μy', 'μz']
    for k, (col, lbl) in enumerate(zip(colors_mu, labels_mu)):
        ax1.plot(times_fs, dipoles_traj[:, k], color=col, lw=0.7,
                 alpha=0.8, label=lbl)
    ax1.plot(times_fs, dip_mag, color='black', lw=0.9, ls='--',
             alpha=0.7, label='|μ|')
    ax1.set_xlabel('Time (fs)')
    ax1.set_ylabel('Dipole moment (Debye)')
    ax1.set_title('ML Dipole Trajectory', fontsize=10)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(times_fs[0], times_fs[-1])

    # ── [0,2] Dipole ACF ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    # Only show first 500 fs of ACF for clarity
    n_show = min(len(acf_times), int(500.0 / dt_eff) + 1)
    ax2.plot(acf_times[:n_show], acf_values[:n_show],
             color='darkorchid', lw=1.2)
    ax2.axhline(0, color='grey', lw=0.8, ls='--')
    ax2.set_xlabel('Lag time (fs)')
    ax2.set_ylabel('C(t) / C(0)')
    ax2.set_title('Dipole Autocorrelation', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, acf_times[n_show - 1])

    # ── [1,0] IR spectrum ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(frequencies, intensities, color='firebrick', lw=1.2)
    ax3.fill_between(frequencies, 0, intensities, alpha=0.18, color='firebrick')

    # Harmonic NM reference lines
    if nm_frequencies is not None:
        for nf in nm_frequencies:
            if 50 < nf < frequencies.max():
                ax3.axvline(nf, color='steelblue', lw=0.8, ls='--', alpha=0.55)

    # Label top peaks (by intensity)
    top_peaks = sorted(peaks, key=lambda p: p[1], reverse=True)[:8]
    for freq, inten in top_peaks:
        if inten > 0.1:
            ax3.annotate(f'{freq:.0f}', xy=(freq, inten),
                         xytext=(0, 6), textcoords='offset points',
                         ha='center', fontsize=7, color='darkred',
                         arrowprops=dict(arrowstyle='->', color='darkred',
                                         lw=0.6))

    ax3.set_xlabel('Frequency (cm⁻¹)')
    ax3.set_ylabel('Intensity (normalized)')
    ax3.set_title('IR Spectrum (ML dipole ACF)', fontsize=10)
    ax3.set_xlim(0, min(float(frequencies.max()), 4500.0))
    ax3.set_ylim(bottom=0)
    ax3.grid(True, alpha=0.3)

    # ── [1,1] Dipole surface parity ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    try:
        data = np.load(training_data_path, allow_pickle=True)
        coords_all = data['coordinates']
        dip_true   = data['dipoles']
        valid = np.linalg.norm(dip_true, axis=1) > 1e-6
        c_v, d_v = coords_all[valid], dip_true[valid]
        dip_pred = dipole_surface.predict(c_v)

        mag_true = np.linalg.norm(d_v, axis=1)
        mag_pred = np.linalg.norm(dip_pred, axis=1)
        ax4.scatter(mag_true, mag_pred, s=18, alpha=0.7, color='mediumseagreen',
                    edgecolors='none')
        lo_d = min(mag_true.min(), mag_pred.min()) - 0.05
        hi_d = max(mag_true.max(), mag_pred.max()) + 0.05
        ax4.plot([lo_d, hi_d], [lo_d, hi_d], 'k--', lw=1, alpha=0.5)
        rmse_d = np.sqrt(((mag_true - mag_pred)**2).mean())
        ax4.set_xlabel('PSI4 |μ| (Debye)')
        ax4.set_ylabel('ML |μ| (Debye)')
        ax4.set_title(f'Dipole Parity  |  RMSE={rmse_d:.4f} D', fontsize=10)
        ax4.grid(True, alpha=0.3)
    except Exception as exc:
        ax4.text(0.5, 0.5, f'Parity plot failed:\n{exc}',
                 ha='center', va='center', transform=ax4.transAxes, fontsize=8)
        ax4.set_title('Dipole Parity', fontsize=10)

    # ── [1,2] Summary ─────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    meta = dipole_surface.metadata
    peak_str = '\n'.join(
        f'  {f:.0f} cm⁻¹  ({i:.2f})'
        for f, i in sorted(top_peaks[:5], key=lambda p: p[0])
    ) or '  (none above threshold)'

    nm_str = ''
    if nm_frequencies is not None:
        nm_lines = [f'  {"k":>2}  {"freq":>7}  {"ZPE":>9}' ,
                    f'  {"─"*2}  {"─"*7}  {"─"*9}']
        for k, nf in enumerate(nm_frequencies):
            zpe_k = 0.5 * abs(nf) / CM_INV_PER_AU * HARTREE_TO_KCAL
            nm_lines.append(f'  {k+1:>2}  {nf:>7.1f}  {zpe_k:>7.3f} k')
        nm_str = '\nHarmonic NMs (cm⁻¹, ZPE kcal/mol)\n' + '\n'.join(nm_lines) + '\n'

    txt = (
        f"Run Summary\n"
        f"{'─' * 28}\n"
        f"Molecule   : {mol_info['label']}\n"
        f"Temperature: {temperature:.0f} K  (ZPE floor init)\n"
        f"MD steps   : {n_steps}  ({n_steps * timestep:.0f} fs)\n"
        f"Δt_eff     : {dt_eff:.2f} fs/frame\n"
        f"Frames (IR): {len(dipoles_traj)}\n\n"
        f"Dipole surface\n"
        f"  Train RMSE: {meta.get('train_rmse', float('nan')):.4f} D\n"
        f"  Test  RMSE: {meta.get('test_rmse', float('nan')):.4f} D\n"
        f"  R² (test) : {meta.get('r2_test', float('nan')):.4f}\n"
        f"  Hyperparams: {meta.get('hyperparameters', {})}\n\n"
        f"Top IR peaks (cm⁻¹, I)\n{peak_str}\n"
        f"{nm_str}"
    )
    ax5.text(0.04, 0.97, txt, transform=ax5.transAxes,
             fontsize=7.5, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       edgecolor='goldenrod', alpha=0.9))
    ax5.axis('off')
    ax5.set_title('Summary', fontsize=10)

    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Diagnostic figure  : {output_path}")


# =============================================================================
# Multi-trajectory dipole collection
# =============================================================================

def run_multi_trajectory_dipoles(
        driver,
        traj,
        dipole_surface,
        n_trajectories: int,
        n_steps: int,
        temperature: float,
        timestep: float,
        save_every: int,
        nm_data,
        min_freq_zpe: float,
        max_freq_zpe: float,
        preminimize: bool,
        preminimize_steps: int,
        preminimize_tol: float,
        max_bond_extension: float,
        monitor_bonds: list,
        print_every: int = 1000,
        output_dir = None,
        start_coords = None,
        thermostat_tau: float = 200.0) -> np.ndarray:
    """
    Run N independent ML-MD trajectories from the N lowest-energy training
    frames, each with a different RNG seed, and return the concatenated,
    per-trajectory-centred dipole array.

    Starting from diverse low-energy frames naturally samples different
    torsional conformers and provides conformational broadening of the IR
    spectrum.  Each trajectory's dipoles are mean-centred before
    concatenation so that conformer-dependent DC offsets do not alias the
    ACF.

    Dissociation guard: if max_bond_extension > 0 the MD is stopped early
    when any covalent bond extends beyond that multiple of its initial
    length, preventing unphysical dynamics from contaminating the ACF.

    Returns
    -------
    list of np.ndarray, each (n_frames_k, 3) Debye
        Per-trajectory dipole arrays.  Spectra are averaged independently
        per trajectory (not concatenated) to avoid ACF artifacts at boundaries.
    """
    n_atoms = len(driver.symbols)
    sort_idx    = np.argsort(traj.energies)
    n_starts    = min(n_trajectories, len(sort_idx))
    start_idxs  = sort_idx[:n_starts]

    print(f"\n{'=' * 70}")
    print(f"  MULTI-TRAJECTORY MD  ({n_starts} trajectories)")
    print(f"{'=' * 70}")
    if start_coords is not None:
        print(f"  Starting geometry  : user-supplied --start-coords (all trajectories)")
    else:
        print(f"  Starting frames (lowest energy): {start_idxs.tolist()}")
    print(f"  Steps per trajectory           : {n_steps}")
    print(f"  Temperature                    : {temperature:.0f} K")
    if max_bond_extension > 0:
        print(f"  Dissociation limit             : {max_bond_extension:.1f}× initial bond")

    all_dipoles = []   # list of (n_frames_k, 3) arrays, one per trajectory

    for k, fidx in enumerate(start_idxs):
        seed = 42 + k
        if start_coords is not None:
            coords0  = np.array(start_coords, dtype=float)
            e_start  = driver.energy(coords0) * HARTREE_TO_KCAL
            print(f"\n  --- Trajectory {k+1}/{n_starts}  "
                  f"(user start_coords, E={e_start:.2f} kcal/mol, seed={seed}) ---")
        else:
            coords0 = traj.coordinates[fidx].copy()
            e_start  = traj.energies[fidx] * HARTREE_TO_KCAL
            print(f"\n  --- Trajectory {k+1}/{n_starts}  "
                  f"(frame {fidx}, E={e_start:.2f} kcal/mol, seed={seed}) ---")

        md_data = run_ml_md_dense(
            driver, coords0, n_steps, temperature,
            timestep=timestep, save_every=save_every,
            nm_data=nm_data,
            min_freq_zpe=min_freq_zpe, max_freq_zpe=max_freq_zpe,
            preminimize=preminimize,
            preminimize_steps=preminimize_steps,
            preminimize_tol=preminimize_tol,
            seed=seed,
            max_bond_extension=max_bond_extension,
            monitor_bonds=monitor_bonds,
            print_every=print_every,
            thermostat_tau=thermostat_tau,
        )

        n_frames  = len(md_data['coords_traj'])
        dissoc    = md_data.get('dissociation_step')
        if dissoc:
            print(f"    Dissociation at step {dissoc} — {n_frames} frames kept")
        else:
            print(f"    Completed — {n_frames} frames")

        # Save per-trajectory XYZ
        traj_xyz = output_dir / f'traj_{k+1:02d}.xyz'
        save_trajectory_xyz(
            md_data['coords_traj'], driver.symbols,
            md_data['times_fs'], md_data['energies_ml'],
            str(traj_xyz),
        )

        # Predict dipoles for this trajectory
        dipoles = predict_trajectory_dipoles(dipole_surface,
                                              md_data['coords_traj'])
        all_dipoles.append(dipoles)
        mag = np.linalg.norm(dipoles, axis=-1)
        print(f"    Dipole |μ|: {mag.min():.3f}–{mag.max():.3f} D  "
              f"(mean {mag.mean():.3f} D)")

    print(f"\n  Trajectories collected: {len(all_dipoles)}")
    return all_dipoles   # return list, NOT concatenated


# =============================================================================
# Main workflow
# =============================================================================

def run_ir_workflow(model_path: str,
                    training_data_path: str,
                    dipole_model_path: str | None,
                    n_steps: int,
                    temperature: float,
                    timestep: float,
                    save_every: int,
                    max_freq: float,
                    window: str,
                    output_dir: Path,
                    use_zpe_init: bool = True,
                    min_freq_zpe: float = 50.0,
                    max_freq_zpe: float = 4000.0,
                    preminimize: bool = False,
                    preminimize_steps: int = 300,
                    preminimize_tol: float = 0.005,
                    analytic_hessian: bool = False,
                    family_manifest: str | None = None,
                    blend_width: float = 3.0,
                    start_coords: np.ndarray | None = None,
                    n_trajectories: int = 1,
                    max_bond_extension: float = 0.0,
                    monitor_bonds: list = None,
                    print_every: int = 0,
                    delta_model_path: str | None = None,
                    energy_delta_path: str | None = None,
                    nm_delta_model_path: str | None = None,
                    nm_pes_model_path: str | None = None,
                    nm_pes_bond_wall_factor: float = 1.6,
                    nm_pes_bond_wall_stiffness: float = 1.0,
                    sgdml_model_path: str | None = None,
                    mace_model_path: str | None = None,
                    nm_eigvec_model_path: str | None = None,
                    thermostat_tau: float = 200.0) -> None:
    """
    Full ML-PES IR spectrum workflow.

    Steps
    -----
    1. Train / load ML dipole surface.
    2. Compute ML-PES normal modes at equilibrium (numerical or analytic Hessian).
    3. Run ZPE-floor initialised ML-MD (dense frames).
    4. Predict ML dipoles along trajectory.
    5. Compute IR spectrum from dipole ACF.
    6. Save spectrum CSV.
    7. Save standalone IR spectrum figure  (ir_spectrum_clean.png).
    8. Save 6-panel diagnostic figure       (ir_spectrum_figure.png).
    9. Save JSON summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("  ML-PES IR SPECTRUM WORKFLOW")
    print(f"{'=' * 70}")
    print(f"  ML-PES model       : {nm_pes_model_path or model_path}")
    print(f"  Training data      : {training_data_path}")
    print(f"  Dipole model       : {dipole_model_path or '(will train now)'}")
    print(f"  MD steps           : {n_steps}  (dt={timestep} fs, save every {save_every})")
    print(f"  Temperature        : {temperature} K")
    zpe_label = ('yes' if use_zpe_init else 'no')
    if use_zpe_init:
        zpe_label += f'  (filter: [{min_freq_zpe:.0f}, {max_freq_zpe:.0f}] cm⁻¹)'
    print(f"  ZPE floor init     : {zpe_label}")
    premin_label = (f'yes  (max_steps={preminimize_steps}, tol={preminimize_tol} Ha/Å)'
                    if preminimize else 'no')
    print(f"  bakken pre-min     : {premin_label}")
    print(f"  Trajectories       : {n_trajectories}")
    if max_bond_extension > 0:
        print(f"  Bond dissoc guard  : {max_bond_extension:.1f}× initial length")
    print(f"  Output dir         : {output_dir}")

    # ── Identify molecule ─────────────────────────────────────────────
    traj = load_trajectory(training_data_path)
    if family_manifest:
        import json as _json
        from modules.pes_family import PESFamily
        with open(family_manifest) as _f:
            manifest = _json.load(_f)
        # manifest format: {"label": "path/to/model.pkl", ...}
        # optional keys: "_blend_width", "_reference_energies"
        _bw  = manifest.pop('_blend_width', blend_width)
        _ref = manifest.pop('_reference_energies', None)
        family = PESFamily.from_model_paths(
            traj.symbols, manifest, blend_width=_bw,
            reference_energies=_ref)
        driver = PESFamilyDriver(family)
        print(f"  PES family         : {list(family.members.keys())}  "
              f"(blend_width={family.blend_width} kcal/mol)")
    elif nm_pes_model_path:
        from modules.nm_pes import NMPESDriver
        driver = NMPESDriver(nm_pes_model_path,
                             bond_wall_factor=nm_pes_bond_wall_factor,
                             bond_wall_stiffness=nm_pes_bond_wall_stiffness)
    elif sgdml_model_path:
        from modules.sgdml_pes import SGDMLDriver
        driver = SGDMLDriver(sgdml_model_path)
        print(f"  sGDML model        : {sgdml_model_path}")
    elif mace_model_path:
        from modules.mace_pes import MACEDriver
        driver = MACEDriver(mace_model_path)
        print(f"  MACE model         : {mace_model_path}")
    else:
        driver = MLPESDriver(model_path)

    # Wrap with delta-ML correction if provided
    if nm_delta_model_path:
        print(f"  NM delta-ML model  : {nm_delta_model_path}")
        driver = NMDeltaDriver(driver, nm_delta_model_path)
    elif delta_model_path:
        print(f"  Delta-ML model     : {delta_model_path}")
        driver = DeltaMLPESDriver(driver, delta_model_path)
    elif energy_delta_path:
        print(f"  Energy-delta JSON  : {energy_delta_path}")
        driver = EnergyDeltaDriver(driver, energy_delta_path)

    mol = identify_molecule(driver.symbols, traj.coordinates[0])
    print(f"  Molecule           : {mol['label']}  ({mol['n_atoms']} atoms)")

    # ── Step 1: Dipole surface ────────────────────────────────────────
    dipole_pkl = output_dir / 'dipole_surface.pkl'

    # Detect whether the active PES driver is NM-based (NMPESDriver or a
    # delta wrapper around one).  If so, prefer NMDipoleSurface.
    _nm_pes_driver = None
    if hasattr(driver, '_model') and hasattr(driver._model, 'U_vib'):
        _nm_pes_driver = driver        # bare NMPESDriver
    elif hasattr(driver, '_base') and hasattr(getattr(driver, '_base', None), '_model'):
        _nm_pes_driver = driver._base  # NMDeltaDriver wrapping NMPESDriver

    # --nm-eigvec-model forces NMDipoleSurface even when PES is MACE/Coulomb+KRR.
    # The NM-PES pkl provides U_vib/freq_vib/eq_coords for the dipole surface.
    if nm_eigvec_model_path and _nm_pes_driver is None:
        import types
        from modules.nm_pes import NMKRRPESModel as _NMKRRPESModel
        _nm_eigvec_model_obj = _NMKRRPESModel.load(nm_eigvec_model_path)
        _nm_pes_driver = types.SimpleNamespace(_model=_nm_eigvec_model_obj)
        print(f"  NM eigvec model    : {nm_eigvec_model_path}  (NMDipoleSurface forced)")

    if dipole_model_path and Path(dipole_model_path).exists():
        print(f"\n  Loading existing dipole model from {dipole_model_path}")
        from modules.nm_pes import load_dipole_surface as _load_dipole
        dipole_surface = _load_dipole(dipole_model_path)
    elif _nm_pes_driver is not None:
        dipole_surface = _train_nm_dipole_surface(
            training_data_path, str(dipole_pkl), _nm_pes_driver
        )
    else:
        dipole_surface = train_dipole_surface(
            training_data_path, str(dipole_pkl)
        )

    # ── Step 2: ML-PES normal modes ───────────────────────────────────
    if start_coords is not None:
        coords0 = np.array(start_coords, dtype=float)
        print(f"\n  Starting geometry  : user-supplied (e.g. PSI4 equilibrium)  "
              f"shape={coords0.shape}")
    else:
        start_idx = np.argmin(traj.energies)
        coords0   = traj.coordinates[start_idx].copy()
        print(f"\n  Starting geometry  : frame {start_idx}  "
              f"(E = {traj.energies[start_idx] * HARTREE_TO_KCAL:.2f} kcal/mol)")

    nm_data = None
    nm_frequencies = None
    if use_zpe_init:
        # For NM-PES models (or delta wrapper around NM-PES), use the stored
        # PSI4 eigenvectors directly rather than recomputing via FD Hessian —
        # they are exact by construction and immediately available.
        _nm_pes_base = None
        if hasattr(driver, '_model') and hasattr(driver._model, 'U_vib'):
            _nm_pes_base = driver        # NMPESDriver directly
        elif hasattr(driver, '_base') and hasattr(getattr(driver, '_base', None), '_model'):
            _nm_pes_base = driver._base  # NMDeltaDriver wrapping NMPESDriver

        if _nm_pes_base is not None:
            print("\n--- Using stored PSI4 NM data for ZPE init (NM-PES model) ---")
            from modules.nm_pes import ATOMIC_MASSES
            _nm = _nm_pes_base._model
            _mass_vec = np.repeat(
                np.array([ATOMIC_MASSES[s] for s in driver.symbols]), 3)
            nm_data = (
                _nm.freqs_vib,          # (n_vib,) cm⁻¹
                _nm.U_vib,              # (3N, n_vib) mass-weighted eigenvectors
                _nm._eigenvalues_ha,    # (n_vib,) Ha/(Bohr²·amu)
                _mass_vec,              # (3N,) amu
            )
            nm_frequencies = nm_data[0]
            print(f"  n_vib={len(nm_frequencies)}  "
                  f"freq range [{nm_frequencies[0]:.1f}, {nm_frequencies[-1]:.1f}] cm⁻¹")
        elif nm_eigvec_model_path:
            # Use stored PSI4 eigenvectors from nm_eigvec_model (fast; avoids MACE Hessian)
            print("\n--- Using stored PSI4 NM data for ZPE init (--nm-eigvec-model) ---")
            from modules.nm_pes import NMKRRPESModel as _NMKRRPESModel2, ATOMIC_MASSES
            _nm_zpe = _NMKRRPESModel2.load(nm_eigvec_model_path)
            _mass_vec = np.repeat(
                np.array([ATOMIC_MASSES[s] for s in driver.symbols]), 3)
            nm_data = (
                _nm_zpe.freqs_vib,
                _nm_zpe.U_vib,
                _nm_zpe._eigenvalues_ha,
                _mass_vec,
            )
            nm_frequencies = nm_data[0]
            print(f"  n_vib={len(nm_frequencies)}  "
                  f"freq range [{nm_frequencies[0]:.1f}, {nm_frequencies[-1]:.1f}] cm⁻¹")
        else:
            print("\n--- ML-PES Normal Mode Analysis (for ZPE init) ---")
            nm_data = compute_mlpes_normal_modes(driver, coords0,
                                                 analytic=analytic_hessian)
            nm_frequencies = nm_data[0]   # (n_vib,) cm⁻¹

    # ── Step 3: ML-MD (dense) — single or multi-trajectory ───────────
    if n_trajectories > 1:
        # Multi-trajectory: compute spectrum for each independently, then
        # average the intensity arrays.  This avoids ACF discontinuities that
        # arise from concatenating trajectories with different starting phases.
        per_traj_dipoles = run_multi_trajectory_dipoles(
            driver, traj, dipole_surface,
            n_trajectories=n_trajectories,
            n_steps=n_steps,
            temperature=temperature,
            timestep=timestep, save_every=save_every,
            nm_data=nm_data,
            min_freq_zpe=min_freq_zpe, max_freq_zpe=max_freq_zpe,
            preminimize=preminimize,
            preminimize_steps=preminimize_steps,
            preminimize_tol=preminimize_tol,
            max_bond_extension=max_bond_extension,
            monitor_bonds=monitor_bonds,
            print_every=print_every,
            output_dir=output_dir,
            start_coords=start_coords,
            thermostat_tau=thermostat_tau,
        )
        print(f"\n--- Averaging IR spectra across {len(per_traj_dipoles)} trajectories ---")
        all_intensities = []
        for k, dip_k in enumerate(per_traj_dipoles):
            freq_k, int_k, _, _, _ = compute_ir_spectrum(
                dip_k, timestep_fs=timestep, save_every=save_every,
                temperature=temperature, max_freq=max_freq, window=window,
            )
            all_intensities.append(int_k)
            print(f"  Trajectory {k+1}/{len(per_traj_dipoles)} spectrum computed  "
                  f"({len(dip_k)} frames)")
        frequencies  = freq_k  # use last trajectory's grid as reference
        # Interpolate shorter trajectories onto the reference grid before averaging
        # (trajectories may have different lengths due to early dissociation truncation)
        all_int_interp = []
        for int_k in all_intensities:
            if len(int_k) == len(frequencies):
                all_int_interp.append(int_k)
            else:
                # Rebuild the frequency axis for this trajectory length
                freq_this = np.linspace(frequencies[0], frequencies[-1], len(int_k))
                all_int_interp.append(np.interp(frequencies, freq_this, int_k))
        intensities  = np.mean(all_int_interp, axis=0)
        # Normalise averaged spectrum to 1
        if intensities.max() > 0:
            intensities /= intensities.max()
        # Recompute peaks on averaged spectrum
        from ir_spectroscopy import IRSpectrumCalculator as _IRSC
        _calc = _IRSC(temperature=temperature)
        _calc.spectrum     = (frequencies, intensities)
        _calc.frequencies  = frequencies
        _calc.intensities  = intensities
        peaks = _calc.find_peaks(threshold=0.05, verbose=True)
        # Use last trajectory's ACF for diagnostic (approximation)
        _, _, acf_times, acf_values, _ = compute_ir_spectrum(
            per_traj_dipoles[-1], timestep_fs=timestep, save_every=save_every,
            temperature=temperature, max_freq=max_freq, window=window,
        )
        dipoles_traj = per_traj_dipoles[-1]   # for diagnostic figure (last traj)
        md_data = None  # no single md_data in multi-traj mode
    else:
        print(f"\n--- ML-MD  ({n_steps} steps, {temperature:.0f} K, "
              f"saving every {save_every} step) ---")

        md_data = run_ml_md_dense(
            driver, coords0, n_steps, temperature,
            timestep=timestep, save_every=save_every,
            nm_data=nm_data,
            min_freq_zpe=min_freq_zpe,
            max_freq_zpe=max_freq_zpe,
            preminimize=preminimize,
            preminimize_steps=preminimize_steps,
            preminimize_tol=preminimize_tol,
            max_bond_extension=max_bond_extension,
            monitor_bonds=monitor_bonds,
            print_every=print_every,
            thermostat_tau=thermostat_tau,
        )

        if md_data.get('dissociation_step'):
            print(f"  WARNING: dissociation at step {md_data['dissociation_step']}  "
                  f"— {len(md_data['coords_traj'])} frames used")

        # Save raw trajectory (pickle)
        md_pkl = output_dir / 'md_trajectory.pkl'
        with open(md_pkl, 'wb') as fh:
            pickle.dump(md_data, fh)
        print(f"  MD trajectory saved: {md_pkl}  ({len(md_data['coords_traj'])} frames)")

        # Save XYZ trajectory for external viewers (VMD, Avogadro, OVITO)
        xyz_path = output_dir / f'{mol["hill"]}_md_trajectory.xyz'
        save_trajectory_xyz(
            md_data['coords_traj'], md_data['symbols'],
            md_data['times_fs'], md_data['energies_ml'],
            str(xyz_path), mol_info=mol,
        )

        # ── Step 4: Predict dipoles along trajectory ──────────────────
        print("\n--- Predicting dipoles along MD trajectory ---")
        dipoles_traj = predict_trajectory_dipoles(
            dipole_surface, md_data['coords_traj']
        )

    # ── Step 5: Compute IR spectrum (single-trajectory path only) ────
    if n_trajectories == 1:
        print("\n--- Computing IR spectrum ---")
        frequencies, intensities, acf_times, acf_values, peaks = compute_ir_spectrum(
            dipoles_traj,
            timestep_fs=timestep,
            save_every=save_every,
            temperature=temperature,
            max_freq=max_freq,
            window=window,
        )
    # multi-trajectory: frequencies/intensities/peaks/acf_* already set above

    # ── Step 6: Save CSV ──────────────────────────────────────────────
    csv_path = output_dir / 'ir_spectrum.csv'
    dt_eff = timestep * save_every
    save_spectrum_csv(
        frequencies, intensities, str(csv_path),
        metadata={
            'molecule': mol['label'],
            'temperature': temperature,
            'n_steps': n_steps,
            'dt_eff_fs': dt_eff,
            'date': datetime.datetime.now().isoformat(),
        }
    )

    # ── Step 7: Standalone IR spectrum figure ─────────────────────────
    fig_spectrum_path = output_dir / 'ir_spectrum_clean.png'
    plot_ir_spectrum(
        frequencies, intensities,
        acf_times, acf_values, peaks,
        str(fig_spectrum_path),
        temperature=temperature,
        n_steps=n_steps,
        timestep=timestep,
        dt_eff=dt_eff,
        n_frames=len(dipoles_traj),
        molecule=mol['label'],
        nm_frequencies=nm_frequencies,
    )

    # ── Step 8: 6-panel diagnostic figure ────────────────────────────
    fig_diag_path = output_dir / 'ir_spectrum_figure.png'
    if md_data is not None:   # single-trajectory only; multi-traj skips full diagnostic
        plot_ir_diagnostics(
            md_data, dipoles_traj,
            frequencies, intensities,
            acf_times, acf_values, peaks,
            dipole_surface, training_data_path,
            str(fig_diag_path),
            nm_frequencies=nm_frequencies,
        )
    else:
        print(f"\n  Diagnostic figure  : skipped (multi-trajectory mode — see traj_*.xyz)")

    # ── Step 9: JSON summary ──────────────────────────────────────────
    top_peaks = sorted(peaks, key=lambda p: p[1], reverse=True)[:10]
    _xyz_path = str(xyz_path) if md_data is not None else str(output_dir / 'traj_01.xyz')
    summary = {
        'date':              datetime.datetime.now().isoformat(),
        'molecule_hill':     mol['hill'],
        'molecule_name':     mol['name'],
        'molecule_unicode':  mol['unicode'],
        'model_path':        str(nm_pes_model_path or model_path),
        'training_data':     str(training_data_path),
        'dipole_model':      str(dipole_pkl),
        'n_md_steps':        n_steps,
        'n_trajectories':    n_trajectories,
        'temperature_K':     temperature,
        'timestep_fs':       timestep,
        'save_every':        save_every,
        'dt_eff_fs':         dt_eff,
        'max_bond_extension': max_bond_extension,
        'zpe_floor_init':    use_zpe_init,
        'zpe_min_freq_cm-1': min_freq_zpe if use_zpe_init else None,
        'zpe_max_freq_cm-1': max_freq_zpe if use_zpe_init else None,
        'preminimized':      md_data.get('preminimized', False) if md_data else preminimize,
        'n_frames_ir':       len(dipoles_traj),
        'trajectory_xyz':    _xyz_path,
        'spectrum_csv':      str(csv_path),
        'figure_spectrum':   str(fig_spectrum_path),
        'figure_diagnostic': str(fig_diag_path),
        'dipole_surface_train_rmse_D': float(
            dipole_surface.metadata.get('train_rmse', float('nan'))),
        'dipole_surface_test_rmse_D':  float(
            dipole_surface.metadata.get('test_rmse', float('nan'))),
        'dipole_surface_r2_test':      float(
            dipole_surface.metadata.get('r2_test', float('nan'))),
        'top_peaks_cm-1': [
            {'frequency_cm-1': float(f), 'intensity': float(i)}
            for f, i in top_peaks
        ],
        'nm_frequencies_cm-1': (
            [float(f) for f in nm_frequencies] if nm_frequencies is not None else None
        ),
    }
    json_path = output_dir / 'ir_summary.json'
    with open(json_path, 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'=' * 70}")
    print("  IR WORKFLOW COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Molecule           : {mol['label']}")
    print(f"  Trajectory XYZ     : {_xyz_path}")
    print(f"  Spectrum CSV       : {csv_path}")
    print(f"  Spectrum figure    : {fig_spectrum_path}")
    print(f"  Diagnostic figure  : {fig_diag_path}")
    print(f"  Summary JSON       : {json_path}")

    if nm_frequencies is not None:
        print(f"\n  ML-PES harmonic frequencies (cm⁻¹):")
        for k, nf in enumerate(nm_frequencies):
            zpe_k = 0.5 * abs(nf) / CM_INV_PER_AU * HARTREE_TO_KCAL
            print(f"    Mode {k+1:2d}: {nf:8.1f} cm⁻¹   ZPE = {zpe_k:.3f} kcal/mol")

    print(f"\n  Top IR peaks (from ACF):")
    print(f"  {'Freq (cm⁻¹)':>14}  {'Intensity':>10}")
    print(f"  {'─'*14}  {'─'*10}")
    for f, i in sorted(top_peaks[:8], key=lambda p: p[0]):
        print(f"  {f:>14.1f}  {i:>10.4f}")


# =============================================================================
# CLI helpers
# =============================================================================

def _parse_monitor_bonds(s: str | None) -> list | None:
    """
    Parse --monitor-bonds string "i-j,i-j:label,..." into list of tuples.
    Formats supported:
      "0-1"          → (0, 1)
      "0-1:O1-O2"    → (0, 1, 'O1-O2')
      "0-1,2-9"      → [(0,1), (2,9)]
    """
    if not s:
        return None
    pairs = []
    for token in s.split(','):
        token = token.strip()
        if not token:
            continue
        if ':' in token:
            idx_part, label = token.split(':', 1)
        else:
            idx_part, label = token, None
        i, j = (int(x) for x in idx_part.split('-'))
        pairs.append((i, j, label) if label else (i, j))
    return pairs if pairs else None


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ML-PES IR spectrum via ML dipole surface + dipole ACF',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model',          required=False, default=None,
                        help='Coulomb-matrix ML-PES model (.pkl). '
                             'Either --model, --nm-pes-model, or --sgdml-model is required.')
    parser.add_argument('--sgdml-model',    default=None,
                        help='sGDML model (.pkl from train_sgdml_model.py). '
                             'Force-trained, symmetry-aware; physically correct Hessian.')
    parser.add_argument('--mace-model',     default=None,
                        help='MACE model (.pt from train_mace_model.py). '
                             'Equivariant MPNN; physical normal-mode frequencies; '
                             'no descriptor stiffness artifact. Preferred backend.')
    parser.add_argument('--nm-pes-model',   default=None,
                        help='NM-coordinate ML-PES model (.pkl, from train_wB97X_nm_model.py). '
                             'Replaces --model when provided. Avoids Coulomb+RBF imaginary-mode '
                             'artifact; analytic forces and Hessian are physical near equilibrium.')
    parser.add_argument('--nm-pes-bond-wall-factor',     type=float, default=1.6,
                        help='Bond cutoff as multiple of eq distance for NM-PES bond wall (default 1.6).')
    parser.add_argument('--nm-pes-bond-wall-stiffness',  type=float, default=1.0,
                        help='Bond wall spring constant Ha/Å² for NM-PES (default 1.0).')
    parser.add_argument('--training-data',  required=True,
                        help='Training data .npz with coordinates + dipoles')
    parser.add_argument('--dipole-model',   default=None,
                        help='Pre-trained dipole surface (.pkl); trains new if omitted')
    parser.add_argument('--steps',          type=int,   default=10000,
                        help='ML-MD steps')
    parser.add_argument('--temp',           type=float, default=300.0,
                        help='MD temperature (K)')
    parser.add_argument('--timestep',       type=float, default=0.5,
                        help='Timestep (fs)')
    parser.add_argument('--save-every',     type=int,   default=1,
                        help='Save frame every N steps (1 = every step for dense ACF)')
    parser.add_argument('--max-freq',       type=float, default=4500.0,
                        help='Maximum frequency in cm⁻¹')
    parser.add_argument('--window',         default='hann',
                        choices=['hann', 'hamming', 'blackman'],
                        help='ACF window function')
    parser.add_argument('--output-dir',     default=None,
                        help='Output directory (auto-timestamped if omitted)')
    parser.add_argument('--no-zpe-init',    action='store_true',
                        help='Disable ZPE-floor velocity initialisation '
                             '(use plain Maxwell-Boltzmann instead)')
    parser.add_argument('--zpe-min-freq',   type=float, default=50.0,
                        help='Min frequency (cm⁻¹) for ZPE boosting; modes below are skipped '
                             '(guards against near-zero/imaginary modes)')
    parser.add_argument('--zpe-max-freq',   type=float, default=4000.0,
                        help='Max frequency (cm⁻¹) for ZPE boosting; modes above are skipped '
                             '(guards against unphysical KRR Hessian artifacts outside '
                             'the training set hull; physical C-H max ~3200 cm⁻¹)')
    parser.add_argument('--start-coords',   default=None,
                        help='Path to .npy file containing starting geometry (n_atoms, 3) Å. '
                             'Overrides lowest-energy training frame. Use PSI4 equilibrium '
                             'geometry to avoid ML-PES saddle-point pre-minimization issues.')
    parser.add_argument('--n-trajectories', type=int, default=1,
                        help='Number of independent MD trajectories to run and average. '
                             'Each starts from one of the N lowest-energy training frames '
                             'with a different random seed (seed=42,43,...). Dipoles are '
                             'centred per-trajectory before concatenation, providing '
                             'conformational broadening. Default: 1 (single trajectory).')
    parser.add_argument('--max-bond-extension', type=float, default=0.0,
                        help='Stop trajectory when any covalent bond (initial length < 2 Å) '
                             'extends beyond this multiple of its initial length. '
                             '2.5 guards against dissociation while allowing large-amplitude '
                             'vibrations. 0 (default) disables detection.')
    parser.add_argument('--monitor-bonds', default=None,
                        help='Comma-separated atom-pair indices to monitor during MD, e.g. '
                             '"0-1,2-9,1-2". Distances saved at every frame and printed '
                             'periodically. Useful for tracking reaction coordinates.')
    parser.add_argument('--print-every', type=int, default=0,
                        help='Print one-line diagnostic (E, T, monitored distances) every N '
                             'steps during MD. 0 (default) disables periodic prints.')
    parser.add_argument('--preminimize',    action='store_true',
                        help='Run bakken steepest-descent pre-minimisation on the ML-PES '
                             'before Hessian/MD so the expansion point is a true stationary '
                             'point, preventing unphysical Hessian curvature')
    parser.add_argument('--preminimize-steps', type=int, default=300,
                        help='Max steps for bakken pre-minimiser (default 300)')
    parser.add_argument('--preminimize-tol',   type=float, default=0.005,
                        help='Force convergence threshold for pre-minimiser Ha/Å (default 0.005)')
    parser.add_argument('--analytic-hessian',  action='store_true',
                        help='Use analytic KRR Hessian (chain rule through Coulomb matrix + '
                             'RBF kernel) instead of numerical finite differences. '
                             'Exact, fast (single forward pass), no FD step-size tuning.')
    parser.add_argument('--delta-model',       default=None,
                        help='Path to delta-ML correction model .pkl '
                             '(from casscf_surface_correction.py). '
                             'Adds CASSCF(4,4)−B3LYP correction to energy and forces.')
    parser.add_argument('--energy-delta',      default=None,
                        help='Path to 1D energy-delta JSON '
                             '(from casscf_surface_correction.py energy_delta.json). '
                             'Applies a spline CASSCF correction as a function of '
                             'ΔE_B3LYP — more stable than --delta-model when Coulomb '
                             'descriptors cluster all geometries together.')
    parser.add_argument('--nm-delta-model',    default=None,
                        help='Path to NMKRRDeltaModel .pkl from casscf_nm_delta.py. '
                             'Applies CASSCF(4,4)−B3LYP correction in normal-mode '
                             'coordinate space — correct localisation near equilibrium '
                             '(preferred over --delta-model and --energy-delta).')
    parser.add_argument('--multi-surface',     action='store_true',
                        help='Use a PESFamily (multi-conformer) instead of a single ML-PES. '
                             'Requires --conformer-manifest.')
    parser.add_argument('--conformer-manifest', default=None,
                        help='JSON file mapping conformer label → model .pkl path. '
                             'Optional keys "_blend_width" (kcal/mol) and '
                             '"_reference_energies" ({label: float Hartree}). '
                             'Example: {"s-cis": "models/scis.pkl", "s-trans": "models/strans.pkl"}')
    parser.add_argument('--blend-width',       type=float, default=3.0,
                        help='Softmin blending width in kcal/mol (default 3.0). '
                             'Overridden by _blend_width key in --conformer-manifest.')
    parser.add_argument('--thermostat-tau',    type=float, default=200.0,
                        help='Berendsen thermostat coupling time in fs (default 200). '
                             'Increase to 2000+ to reduce ZPE leakage and retain C-H '
                             'stretch amplitude for longer in the ACF.')
    parser.add_argument('--nm-eigvec-model',   default=None,
                        help='Path to NMKRRPESModel .pkl used ONLY for normal-mode '
                             'eigenvectors (U_vib / freq_vib). Forces NMDipoleSurface '
                             'training and provides PSI4 NM data for ZPE init regardless '
                             'of which PES backend is active. Use with --mace-model to '
                             'combine stable MACE dynamics with physically meaningful '
                             'dipole derivatives (e.g. C-H stretch region).')
    args = parser.parse_args()

    if (args.model is None and args.nm_pes_model is None
            and args.sgdml_model is None and args.mace_model is None):
        parser.error('One of --model, --nm-pes-model, --sgdml-model, or --mace-model is required.')

    ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path(args.output_dir) if args.output_dir else \
          Path('outputs') / f'ir_spectrum_{ts}'

    run_ir_workflow(
        model_path          = args.model,
        nm_pes_model_path   = args.nm_pes_model,
        training_data_path  = args.training_data,
        dipole_model_path   = args.dipole_model,
        n_steps             = args.steps,
        temperature         = args.temp,
        timestep            = args.timestep,
        save_every          = args.save_every,
        max_freq            = args.max_freq,
        window              = args.window,
        output_dir          = out,
        use_zpe_init        = not args.no_zpe_init,
        min_freq_zpe        = args.zpe_min_freq,
        max_freq_zpe        = args.zpe_max_freq,
        preminimize         = args.preminimize,
        preminimize_steps   = args.preminimize_steps,
        preminimize_tol     = args.preminimize_tol,
        analytic_hessian    = args.analytic_hessian,
        family_manifest     = args.conformer_manifest if args.multi_surface else None,
        blend_width         = args.blend_width,
        start_coords        = np.load(args.start_coords) if args.start_coords else None,
        n_trajectories      = args.n_trajectories,
        max_bond_extension  = args.max_bond_extension,
        monitor_bonds       = _parse_monitor_bonds(args.monitor_bonds),
        print_every         = args.print_every,
        delta_model_path    = args.delta_model,
        energy_delta_path   = args.energy_delta,
        nm_delta_model_path         = args.nm_delta_model,
        nm_pes_bond_wall_factor     = args.nm_pes_bond_wall_factor,
        nm_pes_bond_wall_stiffness  = args.nm_pes_bond_wall_stiffness,
        sgdml_model_path            = args.sgdml_model,
        mace_model_path             = args.mace_model,
        nm_eigvec_model_path        = args.nm_eigvec_model,
        thermostat_tau              = args.thermostat_tau,
    )  # nm_pes_model_path already passed above


if __name__ == '__main__':
    main()
