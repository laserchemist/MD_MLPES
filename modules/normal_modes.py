#!/usr/bin/env python3
"""
Normal Mode Analysis for ML-PES Training Data Generation.

Computes the PSI4 Hessian, diagonalises the mass-weighted Hessian to get
normal mode frequencies and vectors, then generates displaced geometries that
systematically cover the PES—especially the high-energy regions that plain
thermal MD misses.

Units (enforced throughout):
    Coordinates  : Angstrom  (input / output)
    Hessian      : Hartree / Bohr²  (PSI4 native)
    Eigenvalues  : Hartree / (Bohr² · amu)
    Frequencies  : cm⁻¹
    Amplitudes   : Angstrom  (Cartesian displacement)

Conversion constants must NOT be redefined elsewhere; see direct_md.py for
the canonical set.  Only NM-specific constants live here.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants (NM-specific)
# ---------------------------------------------------------------------------
BOHR_TO_ANGSTROM = 0.529177210903          # NIST 2018
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
KB_HARTREE_PER_K = 3.1668114e-6            # k_B in Hartree / K

# Frequency conversion: sqrt(Hartree / (Bohr² · amu))  →  cm⁻¹
# Derivation: FREQ_CONV = sqrt(E_h / (a_0² · m_u)) / (2π c · 100)
#   = sqrt(4.3597e-18 / (2.8003e-21 · 1.6605e-27)) / (2π · 2.9979e10 · 100)
#   ≈ 5140.48
FREQ_CONV = 5140.48

# Threshold below which an eigenvalue is treated as translation/rotation (cm⁻¹)
ROT_TRANS_THRESHOLD = 50.0

# Atomic masses in amu (sufficient for organic / bio molecules)
ATOMIC_MASSES: dict = {
    'H':  1.00794,  'He': 4.002602, 'Li': 6.941,    'Be': 9.012182,
    'B':  10.811,   'C':  12.011,   'N':  14.007,   'O':  15.999,
    'F':  18.99840, 'Ne': 20.1797,  'Na': 22.98977, 'Mg': 24.305,
    'Al': 26.98154, 'Si': 28.0855,  'P':  30.97376, 'S':  32.06,
    'Cl': 35.453,   'Ar': 39.948,   'Br': 79.904,   'I':  126.904,
}


# ---------------------------------------------------------------------------
# Hessian computation
# ---------------------------------------------------------------------------

def compute_hessian_psi4(
    symbols: List[str],
    coords: np.ndarray,
    method: str = 'B3LYP',
    basis: str = '6-31G*',
) -> np.ndarray:
    """
    Compute the Cartesian Hessian matrix using PSI4.

    Args:
        symbols : Atomic symbols, length N
        coords  : Equilibrium geometry in Angstrom, shape (N, 3)
        method  : QM method string (e.g. 'B3LYP', 'HF')
        basis   : Basis set string (e.g. '6-31G*')

    Returns:
        hessian : (3N, 3N) array in Hartree / Bohr²
    """
    try:
        import psi4
    except ImportError:
        raise ImportError("PSI4 is required for Hessian computation")

    mol_str = "0 1\n"
    for s, c in zip(symbols, coords):
        mol_str += f"{s}  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}\n"
    mol_str += "units angstrom\nno_reorient\nno_com"

    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.be_quiet()
    psi4.set_memory('2 GB')
    psi4.set_num_threads(4)
    psi4.set_options({
        'basis':         basis,
        'scf_type':      'df',
        'reference':     'rhf',
        'maxiter':       200,
        'e_convergence': 1e-8,
        'd_convergence': 1e-8,
    })

    mol = psi4.geometry(mol_str)
    logger.info(f"Computing Hessian: {method}/{basis}, {len(symbols)} atoms")
    H = psi4.hessian(f'{method}/{basis}', molecule=mol)
    return np.array(H)          # (3N, 3N)  Hartree / Bohr²


# ---------------------------------------------------------------------------
# Normal mode analysis
# ---------------------------------------------------------------------------

def compute_normal_modes(
    symbols: List[str],
    hessian: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Diagonalise the mass-weighted Hessian and return vibrational normal modes.

    The mass-weighted Hessian is:
        H_mw[i, j] = H[i, j] / sqrt(m_i · m_j)   [Hartree / (Bohr² · amu)]

    Its eigenvectors L̃ satisfy  H_mw · L̃ = λ · L̃  with  |L̃|² = 1  (orthonormal
    in mass-weighted space).  The Cartesian displacement for a displacement Q in
    the normal coordinate is:
        Δr[3i+α]  [Bohr] = Q [Bohr√amu] · L̃[3i+α] / sqrt(m_i [amu])

    Args:
        symbols  : Atomic symbols, length N
        hessian  : (3N, 3N) Hessian in Hartree / Bohr²

    Returns:
        frequencies  : (n_vib,) vibrational frequencies in cm⁻¹ (ascending)
        eigvecs_mw   : (3N, n_vib) mass-weighted eigenvectors, |col|² = 1
        eigenvalues  : (n_vib,) eigenvalues in Hartree / (Bohr² · amu)
        mass_vec     : (3N,) atomic masses in amu  (three entries per atom)
    """
    n_atoms = len(symbols)
    masses   = np.array([ATOMIC_MASSES[s] for s in symbols])   # (N,) amu
    mass_vec = np.repeat(masses, 3)                             # (3N,) amu

    # Mass-weight
    sqrt_m  = np.sqrt(mass_vec)
    hess_mw = hessian / np.outer(sqrt_m, sqrt_m)               # Hartree/(Bohr²·amu)

    # Diagonalise (eigh → real eigenvalues, sorted ascending)
    eigenvalues_all, eigvecs_all = np.linalg.eigh(hess_mw)

    # Convert eigenvalues → frequencies [cm⁻¹]
    freqs_all = np.where(
        eigenvalues_all >= 0,
        FREQ_CONV * np.sqrt(eigenvalues_all),
        -FREQ_CONV * np.sqrt(-eigenvalues_all),   # imaginary → negative
    )

    # Identify vibrational modes (|ω| > threshold)
    vib_mask = np.abs(freqs_all) > ROT_TRANS_THRESHOLD
    n_vib    = vib_mask.sum()
    expected = 3 * n_atoms - 6                  # non-linear molecule

    if n_vib != expected:
        logger.warning(
            f"Found {n_vib} modes above threshold (expected {expected}). "
            "Falling back to the {expected} highest-|ω| modes."
        )
        idx = np.argsort(np.abs(freqs_all))[::-1][:expected]
        idx = np.sort(idx)
    else:
        idx = np.where(vib_mask)[0]

    frequencies = freqs_all[idx]           # (n_vib,) cm⁻¹
    eigvecs_mw  = eigvecs_all[:, idx]      # (3N, n_vib)
    eigenvalues = eigenvalues_all[idx]     # (n_vib,) Hartree/(Bohr²·amu)

    logger.info(f"Normal modes: {len(frequencies)} vibrational, "
                f"{freqs_all.min():.0f}–{freqs_all.max():.0f} cm⁻¹ (all), "
                f"{frequencies.min():.0f}–{frequencies.max():.0f} cm⁻¹ (vib)")

    return frequencies, eigvecs_mw, eigenvalues, mass_vec


# ---------------------------------------------------------------------------
# Displacement generation
# ---------------------------------------------------------------------------

def thermal_amplitude_angstrom(
    eigenvalue: float,
    mass_vec: np.ndarray,
    eigvec_mw: np.ndarray,
    T: float,
) -> float:
    """
    Classical thermal amplitude for a single mode at temperature T.

    Returns the RMS Cartesian displacement in Angstrom:
        Q_cl  = sqrt(2 · k_B T / λ)   [Bohr√amu]
        |Δr|  = Q_cl · |L̃ / sqrt(m)|  [Bohr]  → [Angstrom]

    Args:
        eigenvalue : Hartree / (Bohr² · amu)
        mass_vec   : (3N,) amu
        eigvec_mw  : (3N,) mass-weighted eigenvector (normalised to 1)
        T          : Temperature in Kelvin

    Returns:
        amplitude in Angstrom
    """
    kT = T * KB_HARTREE_PER_K                         # Hartree
    Q_cl_bohr_sqamu = np.sqrt(2.0 * kT / eigenvalue)  # Bohr√amu

    # Cartesian un-mass-weighted displacement vector [Bohr]
    dr_bohr = Q_cl_bohr_sqamu * eigvec_mw / np.sqrt(mass_vec)
    amp_bohr = np.linalg.norm(dr_bohr)
    return amp_bohr * BOHR_TO_ANGSTROM                # Angstrom


def generate_nm_displacements(
    symbols:     List[str],
    coords_eq:   np.ndarray,
    eigvecs_mw:  np.ndarray,
    eigenvalues: np.ndarray,
    mass_vec:    np.ndarray,
    T:           float = 1000.0,
    n_amplitudes: int  = 4,
    max_factor:  float = 3.0,
) -> List[Tuple[np.ndarray, int, float]]:
    """
    Generate ±-displaced geometries along each normal mode.

    Displacement amplitude for step k (k = 1…n_amplitudes):
        a_k = (k / n_amplitudes) · max_factor · a_thermal(T)

    where a_thermal(T) is the classical RMS amplitude at temperature T
    (see thermal_amplitude_angstrom).

    Args:
        symbols      : Atomic symbols, length N
        coords_eq    : Equilibrium coordinates in Angstrom, shape (N, 3)
        eigvecs_mw   : (3N, n_vib) mass-weighted eigenvectors
        eigenvalues  : (n_vib,) in Hartree / (Bohr² · amu)
        mass_vec     : (3N,) amu
        T            : Temperature scale for amplitude (K). Use 1000 K to
                       cover anharmonic regions well.
        n_amplitudes : Number of positive amplitude steps per mode
        max_factor   : Maximum amplitude as a multiple of a_thermal

    Returns:
        List of (displaced_coords [Ang, shape (N,3)], mode_idx, factor) tuples
        (both + and − displacements are included for each amplitude step)
    """
    n_atoms  = len(symbols)
    n_vib    = eigvecs_mw.shape[1]
    factors  = np.linspace(max_factor / n_amplitudes, max_factor, n_amplitudes)

    displacements = []
    for mode_idx in range(n_vib):
        ev = eigenvalues[mode_idx]
        if ev <= 0:
            logger.debug(f"Mode {mode_idx}: imaginary/zero eigenvalue, skipping")
            continue

        L_mw    = eigvecs_mw[:, mode_idx]                         # (3N,)
        kT      = T * KB_HARTREE_PER_K
        Q_cl    = np.sqrt(2.0 * kT / ev)                          # Bohr√amu
        dr_bohr = Q_cl * L_mw / np.sqrt(mass_vec)                 # (3N,) Bohr
        dr_ang  = (dr_bohr * BOHR_TO_ANGSTROM).reshape(n_atoms, 3)  # (N,3) Å

        for factor in factors:
            displacements.append((coords_eq + factor * dr_ang,  mode_idx,  factor))
            displacements.append((coords_eq - factor * dr_ang,  mode_idx, -factor))

    n_modes_used = sum(1 for ev in eigenvalues if ev > 0)
    logger.info(
        f"Generated {len(displacements)} displaced geometries "
        f"({n_modes_used} modes × {n_amplitudes} amplitudes × 2 directions)"
    )
    return displacements


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def normal_modes_from_psi4(
    symbols: List[str],
    coords:  np.ndarray,
    method:  str = 'B3LYP',
    basis:   str = '6-31G*',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One-call wrapper: compute Hessian via PSI4 then return normal modes.

    Returns:
        frequencies  (n_vib,)       cm⁻¹
        eigvecs_mw   (3N, n_vib)    mass-weighted eigenvectors
        eigenvalues  (n_vib,)       Hartree / (Bohr² · amu)
        mass_vec     (3N,)          amu
    """
    hessian = compute_hessian_psi4(symbols, coords, method, basis)
    return compute_normal_modes(symbols, hessian)
