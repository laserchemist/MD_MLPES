#!/usr/bin/env python3
"""
bakken — ML-PES Molecular Dynamics Engine
==========================================
Norwegian for "hill" (the potential energy surface).

Provides a self-contained Velocity-Verlet MD engine driven by a trained
ML-PES (KRR via MLPESTrainer), together with:

  • MLPESDriver       — thin wrapper: energy + central-FD forces
  • minimize_geometry — adaptive steepest-descent pre-minimiser
  • maxwell_boltzmann_velocities — MB velocity initialisation
  • zpe_initialized_velocities   — MB + per-mode ZPE floor
  • kinetic_temperature           — instantaneous T from velocities
  • run_md                        — Velocity-Verlet + Berendsen thermostat

Units (enforced throughout, per CLAUDE.md):
  coordinates  Angstrom
  energies     Hartree
  forces       Hartree/Angstrom
  velocities   Angstrom/fs
  time         femtoseconds
  mass         amu
  dipole       Debye
  frequency    cm⁻¹
"""

import numpy as np
from pathlib import Path
from typing import Optional

# Physical constants (canonical values from direct_md.py:42-48; CLAUDE.md §units)
KB_HARTREE_PER_K = 3.1668114e-6     # Boltzmann constant, Hartree/K
AMU_TO_AU        = 1822.888486      # amu → atomic mass units
FS_TO_AU         = 41.341374575751  # femtoseconds → a.u. time
ANGSTROM_TO_BOHR = 1.88972612456    # Å → Bohr
HARTREE_TO_KCAL  = 627.509474       # Hartree → kcal/mol

BOHR_TO_ANG     = 1.0 / ANGSTROM_TO_BOHR
ANG_TO_BOHR     = ANGSTROM_TO_BOHR
CM_INV_PER_AU   = 219474.63   # a.u. frequency → cm⁻¹
FREQ_CONV       = 5140.48     # sqrt(Hartree/(Bohr²·amu)) → cm⁻¹

ATOMIC_MASSES = {
    'H':  1.00794,  'He': 4.002602, 'C':  12.011,   'N':  14.007,
    'O':  15.999,   'F':  18.9984,  'S':  32.06,    'Cl': 35.453,
    'Br': 79.904,   'I':  126.904,
}


# =============================================================================
# ML-PES driver
# =============================================================================

class MLPESDriver:
    """
    Thin wrapper around a trained MLPESTrainer providing energy and
    central finite-difference forces.

    Args:
        model_path : path to a saved MLPESTrainer .pkl file
    """

    def __init__(self, model_path: str):
        # Lazy import: works both as package module and via sys.path
        import importlib, sys
        spec = importlib.util.find_spec('ml_pes') or importlib.util.find_spec('modules.ml_pes')
        MLPESTrainer = importlib.import_module(spec.name).MLPESTrainer
        self.trainer  = MLPESTrainer.load(model_path)
        self.symbols  = self.trainer.symbols
        self.n_atoms  = len(self.symbols)
        self.masses   = np.array([ATOMIC_MASSES[s] for s in self.symbols])

    def energy(self, coords: np.ndarray) -> float:
        """Predict ML-PES energy (Hartree)."""
        return float(self.trainer.predict(self.symbols, coords))

    def forces(self, coords: np.ndarray, delta: float = 0.005) -> np.ndarray:
        """
        Central finite-difference forces (Hartree/Angstrom).

        Args:
            coords : (n_atoms, 3) Angstrom
            delta  : FD displacement in Angstrom (default 0.005)
        """
        F = np.zeros_like(coords)
        for i in range(self.n_atoms):
            for ax in range(3):
                cp = coords.copy(); cp[i, ax] += delta
                cm = coords.copy(); cm[i, ax] -= delta
                F[i, ax] = -(self.energy(cp) - self.energy(cm)) / (2.0 * delta)
        return F


# =============================================================================
# Geometry pre-minimiser
# =============================================================================

def minimize_geometry(driver: MLPESDriver,
                       coords0: np.ndarray,
                       max_steps: int = 300,
                       force_tol: float = 0.005,
                       step_size: float = 0.01,
                       verbose: bool = True) -> tuple:
    """
    Adaptive steepest-descent minimisation on the ML-PES.

    Walks downhill along the force vector with an adaptive step size:
    accept the step and grow step_size by 20 % if energy decreased;
    reject the step and halve step_size if energy increased.
    Step size is clamped to [1e-4, 0.15] Å at all times.

    This finds a true ML-PES stationary point before computing the
    numerical Hessian, preventing unphysically high curvatures that
    arise when the Hessian expansion point is not a local minimum.

    Args:
        driver     : loaded MLPESDriver
        coords0    : starting geometry (n_atoms, 3) Angstrom
        max_steps  : maximum number of gradient steps (default 300)
        force_tol  : convergence threshold: max |F_ij| < force_tol
                     Hartree/Angstrom (default 0.005)
        step_size  : initial step size in Angstrom (default 0.01)
        verbose    : print convergence table

    Returns:
        (coords_min, E_min, n_steps)
          coords_min : (n_atoms, 3) Angstrom — minimised geometry
          E_min      : float Hartree — energy at minimum
          n_steps    : int — steps taken
    """
    coords   = coords0.copy()
    E        = driver.energy(coords)
    E0       = E
    n_steps  = 0

    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  bakken pre-minimiser  (tol={force_tol} Ha/Å, "
              f"max_steps={max_steps})")
        print(f"  {'Step':>6}  {'E (Ha)':>16}  {'ΔE (kcal/mol)':>15}  "
              f"{'|F|_max (Ha/Å)':>16}  {'step (Å)':>10}")
        print(f"  {'─'*6}  {'─'*16}  {'─'*15}  {'─'*16}  {'─'*10}")

    for step in range(1, max_steps + 1):
        F     = driver.forces(coords)
        f_max = float(np.abs(F).max())

        if verbose and (step == 1 or step % 20 == 0):
            dE = (E - E0) * HARTREE_TO_KCAL
            print(f"  {step:>6}  {E:>16.8f}  {dE:>+15.4f}  "
                  f"{f_max:>16.6f}  {step_size:>10.5f}")

        if f_max < force_tol:
            n_steps = step
            if verbose:
                dE = (E - E0) * HARTREE_TO_KCAL
                print(f"  {step:>6}  {E:>16.8f}  {dE:>+15.4f}  "
                      f"{f_max:>16.6f}  {step_size:>10.5f}")
                print(f"\n  Converged in {step} steps  "
                      f"ΔE = {dE:+.4f} kcal/mol  "
                      f"|F|_max = {f_max:.6f} Ha/Å")
            break

        # Trial step along the force (steepest descent)
        coords_trial = coords + step_size * F
        E_trial      = driver.energy(coords_trial)

        if E_trial < E:
            coords    = coords_trial
            E         = E_trial
            step_size = min(step_size * 1.2, 0.15)
        else:
            step_size = max(step_size * 0.5, 1e-4)

    else:
        n_steps = max_steps
        F       = driver.forces(coords)
        f_max   = float(np.abs(F).max())
        if verbose:
            dE = (E - E0) * HARTREE_TO_KCAL
            print(f"\n  Did not converge in {max_steps} steps  "
                  f"ΔE = {dE:+.4f} kcal/mol  "
                  f"|F|_max = {f_max:.6f} Ha/Å  (continuing anyway)")

    if verbose:
        print(f"  {'─'*60}")

    return coords, E, n_steps


# =============================================================================
# Velocity initialisation
# =============================================================================

def maxwell_boltzmann_velocities(masses_amu: np.ndarray,
                                  T: float,
                                  rng: np.random.Generator) -> np.ndarray:
    """
    Draw velocities from the Maxwell-Boltzmann distribution and remove
    centre-of-mass drift.

    Args:
        masses_amu : (n_atoms,) amu
        T          : temperature (K)
        rng        : NumPy random generator

    Returns:
        velocities (n_atoms, 3) Angstrom/fs
    """
    masses_au = masses_amu * AMU_TO_AU
    v_au      = np.zeros((len(masses_amu), 3))
    for i in range(len(masses_amu)):
        sigma    = np.sqrt(KB_HARTREE_PER_K * T / masses_au[i])
        v_au[i]  = rng.normal(0.0, sigma, 3)
    total_p = (masses_au[:, None] * v_au).sum(axis=0)
    v_au   -= total_p / masses_au.sum()
    return v_au * BOHR_TO_ANG / FS_TO_AU   # a.u. → Ang/fs


def zpe_initialized_velocities(masses_amu: np.ndarray,
                                T: float,
                                rng: np.random.Generator,
                                eigvecs_mw: np.ndarray,
                                eigenvalues: np.ndarray,
                                min_freq_zpe: float = 50.0,
                                max_freq_zpe: float = 4000.0) -> np.ndarray:
    """
    Maxwell-Boltzmann velocities with a per-mode ZPE floor.

    For each vibrational mode k with ω_k in [min_freq_zpe, max_freq_zpe]:
      • ZPE_k = ½ ω_k  [Hartree]
      • If the MB kinetic energy in mode k < ZPE_k, boost it to ZPE_k.

    Modes outside the frequency window are left at their MB amplitude.
    This prevents unphysically high frequencies from KRR Hessian
    extrapolation artifacts from injecting enormous kinetic energy.

    Args:
        masses_amu   : (n_atoms,) amu
        T            : MB target temperature (K)
        rng          : NumPy random generator
        eigvecs_mw   : (3N, n_vib) mass-weighted eigenvectors
        eigenvalues  : (n_vib,) Hartree/(Bohr²·amu)
        min_freq_zpe : lower frequency cutoff for ZPE boosting (cm⁻¹)
        max_freq_zpe : upper frequency cutoff for ZPE boosting (cm⁻¹)

    Returns:
        velocities (n_atoms, 3) Angstrom/fs
    """
    n_atoms  = len(masses_amu)
    n_vib    = eigvecs_mw.shape[1]
    mass_vec = np.repeat(masses_amu, 3)
    sqrt_m   = np.sqrt(mass_vec)

    v_mb   = maxwell_boltzmann_velocities(masses_amu, T, rng)
    v_au   = v_mb * ANG_TO_BOHR * FS_TO_AU
    q_dot  = sqrt_m * v_au.flatten()
    Q_dot  = eigvecs_mw.T @ q_dot

    Q_dot_new = Q_dot.copy()
    n_boosted = 0
    n_skipped = 0

    for k in range(n_vib):
        ev = eigenvalues[k]
        if ev <= 0:
            continue
        omega_k_cm = np.sqrt(ev) * FREQ_CONV
        if omega_k_cm < min_freq_zpe or omega_k_cm > max_freq_zpe:
            n_skipped += 1
            continue
        zpe_k = 0.5 * omega_k_cm / CM_INV_PER_AU
        T_k   = 0.5 * AMU_TO_AU * Q_dot[k] ** 2
        if T_k < zpe_k:
            sign          = np.sign(Q_dot[k]) if Q_dot[k] != 0 else rng.choice([-1.0, 1.0])
            Q_dot_new[k]  = sign * np.sqrt(2.0 * zpe_k / AMU_TO_AU)
            n_boosted    += 1

    q_dot_new  = q_dot - eigvecs_mw @ Q_dot + eigvecs_mw @ Q_dot_new
    v_flat_new = q_dot_new / sqrt_m
    v_au_new   = v_flat_new.reshape(n_atoms, 3)

    masses_au = masses_amu * AMU_TO_AU
    total_p   = (masses_au[:, None] * v_au_new).sum(axis=0)
    v_au_new -= total_p / masses_au.sum()

    if n_boosted or n_skipped:
        print(f"  ZPE floor applied  : {n_boosted}/{n_vib} modes boosted, "
              f"{n_skipped} skipped (outside [{min_freq_zpe:.0f}, "
              f"{max_freq_zpe:.0f}] cm⁻¹)")

    return v_au_new * BOHR_TO_ANG / FS_TO_AU


def kinetic_temperature(velocities: np.ndarray,
                         masses_amu: np.ndarray) -> float:
    """
    Instantaneous kinetic temperature (K) from Cartesian velocities.

    Uses 3N - 6 degrees of freedom (non-linear polyatomic).

    Args:
        velocities : (n_atoms, 3) Angstrom/fs
        masses_amu : (n_atoms,) amu

    Returns:
        temperature (K)
    """
    masses_au = masses_amu * AMU_TO_AU
    v_au      = velocities * ANG_TO_BOHR * FS_TO_AU
    ke        = 0.5 * (masses_au[:, None] * v_au ** 2).sum()
    n_dof     = 3 * len(masses_amu) - 6
    return float(2.0 * ke / (n_dof * KB_HARTREE_PER_K))


# =============================================================================
# Velocity-Verlet MD engine
# =============================================================================

def run_md(driver: MLPESDriver,
           coords0: np.ndarray,
           n_steps: int,
           temperature: float,
           timestep: float = 0.5,
           save_every: int = 1,
           thermostat_tau: float = 100.0,
           seed: int = 42,
           nm_data: Optional[tuple] = None,
           min_freq_zpe: float = 50.0,
           max_freq_zpe: float = 4000.0,
           preminimize: bool = False,
           preminimize_steps: int = 300,
           preminimize_tol: float = 0.005) -> dict:
    """
    Velocity-Verlet ML-MD with Berendsen thermostat (bakken engine).

    Dense output: saves coordinates and ML energy every `save_every` steps.

    Args:
        driver            : loaded MLPESDriver
        coords0           : starting geometry (n_atoms, 3) Angstrom
        n_steps           : number of MD steps
        temperature       : target temperature (K)
        timestep          : time step (fs)
        save_every        : save frame every N steps
        thermostat_tau    : Berendsen coupling time (fs)
        seed              : RNG seed
        nm_data           : (frequencies, eigvecs_mw, eigenvalues, mass_vec)
                            from compute_mlpes_normal_modes(). When provided,
                            ZPE-floor initialisation is used.
        min_freq_zpe      : lower ZPE filter cutoff (cm⁻¹)
        max_freq_zpe      : upper ZPE filter cutoff (cm⁻¹)
        preminimize       : run steepest-descent before MD to find true
                            ML-PES minimum (recommended before Hessian)
        preminimize_steps : max steps for pre-minimiser
        preminimize_tol   : force convergence threshold (Ha/Å)

    Returns dict:
        coords_traj   (n_frames, n_atoms, 3) Angstrom
        energies_ml   (n_frames,) Hartree
        times_fs      (n_frames,) femtoseconds
        symbols       List[str]
        timestep      float (fs)
        save_every    int
        temperature   float (K)
        n_steps       int
        nm_frequencies (n_vib,) cm⁻¹ or None
        coords_start  (n_atoms, 3) — geometry after pre-min (or coords0)
        preminimized  bool
    """
    masses_amu = driver.masses
    masses_au  = masses_amu * AMU_TO_AU
    rng        = np.random.default_rng(seed)

    # ── Optional pre-minimisation ─────────────────────────────────────────
    if preminimize:
        coords, E_min, n_min = minimize_geometry(
            driver, coords0,
            max_steps=preminimize_steps,
            force_tol=preminimize_tol,
            verbose=True,
        )
        print(f"  Pre-min geometry   : ΔE = "
              f"{(E_min - driver.energy(coords0)) * HARTREE_TO_KCAL:+.4f} kcal/mol "
              f"from starting frame")
    else:
        coords = coords0.copy()

    coords_start = coords.copy()

    # ── Velocity initialisation ───────────────────────────────────────────
    if nm_data is not None:
        nm_frequencies, eigvecs_mw, eigenvalues_nm, _ = nm_data
        thermostat_tau = max(thermostat_tau, 200.0)
        velocities = zpe_initialized_velocities(
            masses_amu, temperature, rng,
            eigvecs_mw, eigenvalues_nm,
            min_freq_zpe=min_freq_zpe,
            max_freq_zpe=max_freq_zpe,
        )
        T_init = kinetic_temperature(velocities, masses_amu)
        print(f"  ZPE init T_eff     : {T_init:.0f} K  "
              f"(target: {temperature:.0f} K, τ={thermostat_tau:.0f} fs)")
    else:
        nm_frequencies = None
        velocities     = maxwell_boltzmann_velocities(masses_amu, temperature, rng)

    forces = driver.forces(coords)

    coords_list   = []
    energies_list = []
    times_list    = []

    try:
        from tqdm import tqdm
        pbar = tqdm(range(1, n_steps + 1), desc="bakken MD", unit="step")
    except ImportError:
        pbar = range(1, n_steps + 1)

    for step in pbar:
        # ── Velocity-Verlet ────────────────────────────────────────────
        F_bohr  = forces * ANG_TO_BOHR
        acc_au  = F_bohr / masses_au[:, None]
        v_au    = velocities * ANG_TO_BOHR * FS_TO_AU
        v_au   += 0.5 * acc_au * timestep * FS_TO_AU

        r_au    = coords * ANG_TO_BOHR + v_au * timestep * FS_TO_AU
        coords  = r_au * BOHR_TO_ANG

        forces  = driver.forces(coords)
        F_bohr  = forces * ANG_TO_BOHR
        acc_au  = F_bohr / masses_au[:, None]
        v_au   += 0.5 * acc_au * timestep * FS_TO_AU
        velocities = v_au * BOHR_TO_ANG / FS_TO_AU

        # ── Berendsen thermostat ───────────────────────────────────────
        T_curr = kinetic_temperature(velocities, masses_amu)
        if T_curr > 0:
            lam        = np.sqrt(1.0 + (timestep / thermostat_tau) *
                                 (temperature / T_curr - 1.0))
            velocities *= lam

        # ── Save frame ────────────────────────────────────────────────
        if step % save_every == 0:
            e = driver.energy(coords)
            coords_list.append(coords.copy())
            energies_list.append(e)
            times_list.append(step * timestep)

            if hasattr(pbar, 'set_postfix') and step % max(1, n_steps // 20) == 0:
                pbar.set_postfix({
                    'E': f'{e * HARTREE_TO_KCAL:.1f} kcal/mol',
                    'T': f'{T_curr:.0f} K',
                })

    return {
        'coords_traj':    np.array(coords_list),
        'energies_ml':    np.array(energies_list),
        'times_fs':       np.array(times_list),
        'symbols':        driver.symbols,
        'timestep':       timestep,
        'save_every':     save_every,
        'temperature':    temperature,
        'n_steps':        n_steps,
        'nm_frequencies': nm_frequencies,
        'coords_start':   coords_start,
        'preminimized':   preminimize,
    }
