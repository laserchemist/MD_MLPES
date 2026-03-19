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
}

# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------
from data_formats import TrajectoryData, load_trajectory
from ml_pes import MLPESTrainer
from ir_spectroscopy import DipoleSurface, IRSpectrumCalculator
from normal_modes import compute_normal_modes


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


# =============================================================================
# ML-PES driver (energy + finite-difference forces)
# =============================================================================

class MLPESDriver:
    """Thin wrapper around MLPESTrainer providing energy + FD forces."""

    def __init__(self, model_path: str):
        self.trainer = MLPESTrainer.load(model_path)
        self.symbols = self.trainer.symbols

    def energy(self, coords: np.ndarray) -> float:
        return self.trainer.predict(self.symbols, coords)

    def forces(self, coords: np.ndarray, delta: float = 0.005) -> np.ndarray:
        """Central finite-difference forces (Hartree/Angstrom)."""
        n_atoms = len(self.symbols)
        F = np.zeros_like(coords)
        for i in range(n_atoms):
            for ax in range(3):
                cp = coords.copy(); cp[i, ax] += delta
                cm = coords.copy(); cm[i, ax] -= delta
                F[i, ax] = -(self.energy(cp) - self.energy(cm)) / (2 * delta)
        return F


# =============================================================================
# ML-PES normal mode analysis (numerical Hessian via FD of forces)
# =============================================================================

def compute_mlpes_normal_modes(driver: MLPESDriver,
                                coords_eq: np.ndarray,
                                delta: float = 0.01) -> tuple:
    """
    Compute vibrational normal modes of the ML-PES at the equilibrium geometry.

    The Hessian is built by central finite-differences of ML forces
    (Hartree/Angstrom → Hartree/Angstrom²), then converted to Hartree/Bohr²
    before calling the canonical compute_normal_modes() from normal_modes.py.

    Args:
        driver    : loaded MLPESDriver
        coords_eq : equilibrium geometry (n_atoms, 3) Angstrom
        delta     : displacement step in Angstrom (default 0.01 Å)

    Returns:
        (frequencies, eigvecs_mw, eigenvalues, mass_vec)
          frequencies : (n_vib,)    cm⁻¹ (ascending)
          eigvecs_mw  : (3N, n_vib) mass-weighted eigenvectors
          eigenvalues : (n_vib,)    Hartree/(Bohr²·amu)
          mass_vec    : (3N,)       amu
    """
    n_atoms = len(driver.symbols)
    n_dof   = 3 * n_atoms

    print(f"\n  Computing ML-PES Hessian ({n_dof}×{n_dof}, δ={delta} Å) ...")
    H_ang2 = np.zeros((n_dof, n_dof))   # Hartree / Angstrom²

    for i in range(n_dof):
        cp = coords_eq.flatten().copy(); cp[i] += delta
        cm = coords_eq.flatten().copy(); cm[i] -= delta
        F_p = driver.forces(cp.reshape(n_atoms, 3)).flatten()   # Hartree/Ang
        F_m = driver.forces(cm.reshape(n_atoms, 3)).flatten()   # Hartree/Ang
        H_ang2[:, i] = -(F_p - F_m) / (2.0 * delta)            # Hartree/Ang²

    H_ang2 = 0.5 * (H_ang2 + H_ang2.T)             # symmetrize numerical noise

    # Unit conversion:  Hartree/Ang²  →  Hartree/Bohr²
    H_bohr2 = H_ang2 * ANG_TO_BOHR ** 2

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
# Velocity-Verlet MD engine (dense output — saves every `save_every` steps)
# =============================================================================

def _maxwell_boltzmann(masses_amu: np.ndarray, T: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Initial velocities (Ang/fs) from Maxwell-Boltzmann distribution."""
    masses_au = masses_amu * AMU_TO_AU
    v_au = np.zeros((len(masses_amu), 3))
    for i in range(len(masses_amu)):
        sigma = np.sqrt(KB_HARTREE_PER_K * T / masses_au[i])
        v_au[i] = rng.normal(0.0, sigma, 3)
    # Remove CoM drift
    total_p = (masses_au[:, None] * v_au).sum(axis=0)
    v_au -= total_p / masses_au.sum()
    return v_au * BOHR_TO_ANG / FS_TO_AU  # a.u. → Ang/fs


def _zpe_initialized_velocities(masses_amu: np.ndarray,
                                 T: float,
                                 rng: np.random.Generator,
                                 eigvecs_mw: np.ndarray,
                                 eigenvalues: np.ndarray,
                                 min_freq_zpe: float = 50.0,
                                 max_freq_zpe: float = 4000.0) -> np.ndarray:
    """
    Maxwell-Boltzmann velocities with a ZPE floor applied per normal mode.

    For each vibrational mode k (eigenvalue λ_k > 0):
      • Compute ZPE_k = ½ ω_k  [Hartree],  ω_k = sqrt(λ_k)·FREQ_CONV/CM_INV_PER_AU
      • Project current MB kinetic energy onto mode k.
      • If T_k < ZPE_k  →  set T_k = ZPE_k  (random sign preserved).

    Starting at the equilibrium geometry (PE = 0), assigning T_k = ZPE_k puts
    the total mode energy at the quantum ground state.

    The mass-weighted kinetic energy decomposition used here:
        q̇  = sqrt(m_amu) × ṙ_bohr_au          [Bohr√amu/au]
        Q̇_k = L̃_k · q̇                         [Bohr√amu/au]
        T_k = ½ · AMU_TO_AU · Q̇_k²            [Hartree]

    Args:
        masses_amu  : (n_atoms,) amu
        T           : target MB temperature (K)
        rng         : NumPy random generator
        eigvecs_mw  : (3N, n_vib) normalised mass-weighted eigenvectors
        eigenvalues : (n_vib,)    Hartree/(Bohr²·amu)
        min_freq_zpe: skip ZPE boosting for modes below this value (cm⁻¹).
                      Guards against near-zero/imaginary modes. Default 50.
        max_freq_zpe: skip ZPE boosting for modes above this value (cm⁻¹).
                      Guards against unphysical high-curvature KRR Hessian
                      artifacts outside the training set hull. Default 4000
                      (physical C-H stretch max ~3200 cm⁻¹ for CH₂OO).

    Returns:
        velocities (n_atoms, 3) Angstrom/fs
    """
    n_atoms  = len(masses_amu)
    n_vib    = eigvecs_mw.shape[1]
    mass_vec = np.repeat(masses_amu, 3)   # (3N,) amu
    sqrt_m   = np.sqrt(mass_vec)          # (3N,) sqrt(amu)

    # Start with Maxwell-Boltzmann
    v_mb   = _maxwell_boltzmann(masses_amu, T, rng)         # (n_atoms, 3) Ang/fs
    v_au   = v_mb * ANG_TO_BOHR * FS_TO_AU                  # (n_atoms, 3) Bohr/au
    v_flat = v_au.flatten()                                  # (3N,) Bohr/au

    # Mass-weighted velocities q̇  [Bohr√amu/au]
    q_dot = sqrt_m * v_flat

    # ── Step 1: project onto each vibrational mode ────────────────────
    # Q̇_k = L̃_k · q̇   (scalar normal-coordinate velocity per mode)
    Q_dot = eigvecs_mw.T @ q_dot                            # (n_vib,)

    # ── Step 2: apply ZPE floor independently per mode ───────────────
    # KE of mode k: T_k = ½ · AMU_TO_AU · Q̇_k²   [Hartree]
    # ZPE_k = ½ · ω_k[cm⁻¹] / CM_INV_PER_AU       [Hartree]
    # Minimum |Q̇_k|:  Q̇_k_min = sqrt(2·ZPE_k / AMU_TO_AU)
    Q_dot_new = Q_dot.copy()
    n_boosted = 0
    n_skipped = 0
    for k in range(n_vib):
        ev = eigenvalues[k]
        if ev <= 0:
            continue
        omega_k_cm = np.sqrt(ev) * FREQ_CONV                # cm⁻¹
        # Skip modes outside the physical frequency window — guards against
        # unphysical KRR Hessian curvature beyond the training set hull.
        if omega_k_cm < min_freq_zpe or omega_k_cm > max_freq_zpe:
            n_skipped += 1
            continue
        zpe_k      = 0.5 * omega_k_cm / CM_INV_PER_AU       # Hartree
        T_k        = 0.5 * AMU_TO_AU * Q_dot[k] ** 2        # Hartree (from MB)
        if T_k < zpe_k:
            sign           = np.sign(Q_dot[k]) if Q_dot[k] != 0 else rng.choice([-1.0, 1.0])
            Q_dot_new[k]   = sign * np.sqrt(2.0 * zpe_k / AMU_TO_AU)
            n_boosted     += 1

    # ── Step 3: reconstruct q̇ preserving trans/rot components ─────────
    # q̇ = q̇_TR + q̇_vib
    # q̇_TR  = q̇ − (L̃ · Q̇)        (non-vibrational subspace)
    # q̇_new = q̇_TR + (L̃ · Q̇_new)
    q_dot_vib_old = eigvecs_mw @ Q_dot                      # (3N,)
    q_dot_vib_new = eigvecs_mw @ Q_dot_new                  # (3N,)
    q_dot_new     = q_dot - q_dot_vib_old + q_dot_vib_new   # (3N,)

    # ── Step 4: back to Cartesian [Bohr/au], remove CoM drift ─────────
    v_flat_new = q_dot_new / sqrt_m
    v_au_new   = v_flat_new.reshape(n_atoms, 3)

    masses_au = masses_amu * AMU_TO_AU
    total_p   = (masses_au[:, None] * v_au_new).sum(axis=0)
    v_au_new -= total_p / masses_au.sum()

    if n_boosted or n_skipped:
        print(f"  ZPE floor applied  : {n_boosted}/{n_vib} modes boosted, "
              f"{n_skipped} skipped (outside [{min_freq_zpe:.0f}, {max_freq_zpe:.0f}] cm⁻¹)")

    return v_au_new * BOHR_TO_ANG / FS_TO_AU                # (n_atoms, 3) Ang/fs


def _kin_temp(velocities: np.ndarray, masses_amu: np.ndarray) -> float:
    masses_au = masses_amu * AMU_TO_AU
    v_au = velocities * ANG_TO_BOHR * FS_TO_AU
    ke = 0.5 * (masses_au[:, None] * v_au ** 2).sum()
    n_dof = 3 * len(masses_amu) - 6
    return 2.0 * ke / (n_dof * KB_HARTREE_PER_K)


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
                    max_freq_zpe: float = 4000.0) -> dict:
    """
    Velocity-Verlet ML-MD with Berendsen thermostat.
    Dense output: saves coordinates and ML energy every `save_every` steps.

    Args:
        nm_data : (frequencies, eigvecs_mw, eigenvalues, mass_vec) from
                  compute_mlpes_normal_modes().  When provided, velocities
                  are initialised with a ZPE floor so every vibrational mode
                  starts with at least its zero-point kinetic energy.
                  thermostat_tau is increased to 200 fs automatically to
                  preserve the quantum-corrected amplitudes.

    Returns dict:
        coords_traj  (n_frames, n_atoms, 3) Angstrom
        energies_ml  (n_frames,) Hartree
        times_fs     (n_frames,) femtoseconds
        symbols      List[str]
        nm_frequencies (n_vib,) cm⁻¹ or None
    """
    masses_amu = np.array([ATOMIC_MASSES[s] for s in driver.symbols])
    masses_au  = masses_amu * AMU_TO_AU

    rng    = np.random.default_rng(seed)
    coords = coords0.copy()

    if nm_data is not None:
        nm_frequencies, eigvecs_mw, eigenvalues_nm, _ = nm_data
        # Use a longer thermostat coupling time so ZPE-boosted modes are not
        # immediately quenched back to the classical k_BT equipartition value.
        thermostat_tau = max(thermostat_tau, 200.0)
        velocities = _zpe_initialized_velocities(
            masses_amu, temperature, rng, eigvecs_mw, eigenvalues_nm,
            min_freq_zpe=min_freq_zpe,
            max_freq_zpe=max_freq_zpe,
        )
        T_init = _kin_temp(velocities, masses_amu)
        print(f"  ZPE init T_eff     : {T_init:.0f} K  (thermostat target: {temperature:.0f} K, "
              f"τ={thermostat_tau:.0f} fs)")
    else:
        nm_frequencies = None
        velocities = _maxwell_boltzmann(masses_amu, temperature, rng)

    forces = driver.forces(coords)

    coords_list  = []
    energies_list = []
    times_list   = []

    from tqdm import tqdm
    pbar = tqdm(range(1, n_steps + 1), desc="ML-MD (dense)", unit="step")
    for step in pbar:
        # Half-step velocity
        F_bohr = forces * ANG_TO_BOHR
        acc_au = F_bohr / masses_au[:, None]
        v_au   = velocities * ANG_TO_BOHR * FS_TO_AU
        v_au  += 0.5 * acc_au * timestep * FS_TO_AU

        # Position update
        r_au = coords * ANG_TO_BOHR + v_au * timestep * FS_TO_AU
        coords = r_au * BOHR_TO_ANG

        # New forces
        forces = driver.forces(coords)
        F_bohr = forces * ANG_TO_BOHR
        acc_au = F_bohr / masses_au[:, None]
        v_au  += 0.5 * acc_au * timestep * FS_TO_AU
        velocities = v_au * BOHR_TO_ANG / FS_TO_AU

        # Berendsen thermostat
        T_curr = _kin_temp(velocities, masses_amu)
        if T_curr > 0:
            lam = np.sqrt(1.0 + (timestep / thermostat_tau) * (temperature / T_curr - 1.0))
            velocities *= lam

        # Save
        if step % save_every == 0:
            e = driver.energy(coords)
            coords_list.append(coords.copy())
            energies_list.append(e)
            times_list.append(step * timestep)

            if step % max(1, n_steps // 20) == 0:
                pbar.set_postfix({
                    'E': f'{e * HARTREE_TO_KCAL:.1f} kcal/mol',
                    'T': f'{T_curr:.0f} K',
                })

    return {
        'coords_traj':   np.array(coords_list),
        'energies_ml':   np.array(energies_list),
        'times_fs':      np.array(times_list),
        'symbols':       driver.symbols,
        'timestep':      timestep,
        'save_every':    save_every,
        'temperature':   temperature,
        'n_steps':       n_steps,
        'nm_frequencies': nm_frequencies,
    }


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
                    max_freq_zpe: float = 4000.0) -> None:
    """
    Full ML-PES IR spectrum workflow.

    Steps
    -----
    1. Train / load ML dipole surface.
    2. Compute ML-PES normal modes at equilibrium (numerical Hessian).
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
    print(f"  ML-PES model       : {model_path}")
    print(f"  Training data      : {training_data_path}")
    print(f"  Dipole model       : {dipole_model_path or '(will train now)'}")
    print(f"  MD steps           : {n_steps}  (dt={timestep} fs, save every {save_every})")
    print(f"  Temperature        : {temperature} K")
    zpe_label = ('yes' if use_zpe_init else 'no')
    if use_zpe_init:
        zpe_label += f'  (filter: [{min_freq_zpe:.0f}, {max_freq_zpe:.0f}] cm⁻¹)'
    print(f"  ZPE floor init     : {zpe_label}")
    print(f"  Output dir         : {output_dir}")

    # ── Identify molecule ─────────────────────────────────────────────
    driver = MLPESDriver(model_path)
    traj   = load_trajectory(training_data_path)
    mol    = identify_molecule(driver.symbols, traj.coordinates[0])
    print(f"  Molecule           : {mol['label']}  ({mol['n_atoms']} atoms)")

    # ── Step 1: Dipole surface ────────────────────────────────────────
    dipole_pkl = output_dir / 'dipole_surface.pkl'
    if dipole_model_path and Path(dipole_model_path).exists():
        print(f"\n  Loading existing dipole model from {dipole_model_path}")
        dipole_surface = DipoleSurface.load(dipole_model_path)
    else:
        dipole_surface = train_dipole_surface(
            training_data_path, str(dipole_pkl)
        )

    # ── Step 2: ML-PES normal modes ───────────────────────────────────
    start_idx = np.argmin(traj.energies)
    coords0   = traj.coordinates[start_idx].copy()
    print(f"\n  Starting geometry  : frame {start_idx}  "
          f"(E = {traj.energies[start_idx] * HARTREE_TO_KCAL:.2f} kcal/mol)")

    nm_data = None
    nm_frequencies = None
    if use_zpe_init:
        print("\n--- ML-PES Normal Mode Analysis (for ZPE init) ---")
        nm_data = compute_mlpes_normal_modes(driver, coords0)
        nm_frequencies = nm_data[0]   # (n_vib,) cm⁻¹

    # ── Step 3: ML-MD (dense) ─────────────────────────────────────────
    print(f"\n--- ML-MD  ({n_steps} steps, {temperature:.0f} K, "
          f"saving every {save_every} step) ---")

    md_data = run_ml_md_dense(
        driver, coords0, n_steps, temperature,
        timestep=timestep, save_every=save_every,
        nm_data=nm_data,
        min_freq_zpe=min_freq_zpe,
        max_freq_zpe=max_freq_zpe,
    )

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

    # ── Step 4: Predict dipoles along trajectory ──────────────────────
    print("\n--- Predicting dipoles along MD trajectory ---")
    dipoles_traj = predict_trajectory_dipoles(
        dipole_surface, md_data['coords_traj']
    )

    # ── Step 5: Compute IR spectrum ───────────────────────────────────
    print("\n--- Computing IR spectrum ---")
    frequencies, intensities, acf_times, acf_values, peaks = compute_ir_spectrum(
        dipoles_traj,
        timestep_fs=timestep,
        save_every=save_every,
        temperature=temperature,
        max_freq=max_freq,
        window=window,
    )

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
    plot_ir_diagnostics(
        md_data, dipoles_traj,
        frequencies, intensities,
        acf_times, acf_values, peaks,
        dipole_surface, training_data_path,
        str(fig_diag_path),
        nm_frequencies=nm_frequencies,
    )

    # ── Step 9: JSON summary ──────────────────────────────────────────
    top_peaks = sorted(peaks, key=lambda p: p[1], reverse=True)[:10]
    summary = {
        'date':              datetime.datetime.now().isoformat(),
        'molecule_hill':     mol['hill'],
        'molecule_name':     mol['name'],
        'molecule_unicode':  mol['unicode'],
        'model_path':        str(model_path),
        'training_data':     str(training_data_path),
        'dipole_model':      str(dipole_pkl),
        'n_md_steps':        n_steps,
        'temperature_K':     temperature,
        'timestep_fs':       timestep,
        'save_every':        save_every,
        'dt_eff_fs':         dt_eff,
        'zpe_floor_init':    use_zpe_init,
        'zpe_min_freq_cm-1': min_freq_zpe if use_zpe_init else None,
        'zpe_max_freq_cm-1': max_freq_zpe if use_zpe_init else None,
        'n_frames_ir':       len(dipoles_traj),
        'trajectory_xyz':    str(xyz_path),
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
    print(f"  Trajectory XYZ     : {xyz_path}")
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
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ML-PES IR spectrum via ML dipole surface + dipole ACF',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model',          required=True,
                        help='ML-PES model (.pkl)')
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
    args = parser.parse_args()

    ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path(args.output_dir) if args.output_dir else \
          Path('outputs') / f'ir_spectrum_{ts}'

    run_ir_workflow(
        model_path          = args.model,
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
    )


if __name__ == '__main__':
    main()
