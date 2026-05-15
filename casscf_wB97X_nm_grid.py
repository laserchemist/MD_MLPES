#!/usr/bin/env python3
"""
casscf_wB97X_nm_grid.py — Systematic SA-2-CASSCF(4,4) + triplet NM grid
                            with ωB97X-D as the DFT reference level.

Architecture
------------
Three KRR models trained on NM coordinate descriptors (q ∈ ℝ³⁰):

    E_S0(R) = E_wB97X_ML(R) + δ_S0_ML(R)
    E_S1(R) = E_S0(R)        + Δgap_S1_ML(R)
    E_T1(R) = E_S0(R)        + Δgap_T1_ML(R)

where:
    δ_S0(R)    = [E_CASSCF_S0(R) − E_CASSCF_S0(R_eq)]
               − [E_wB97X(R)     − E_wB97X(R_eq)]          (0 at eq by construction)

    Δgap_S1(R) = E_CASSCF_S1(R) − E_CASSCF_S0(R)           (adiabatic S0→S1 gap)
    Δgap_T1(R) = E_CASSCF_T1(R) − E_CASSCF_S0(R)           (adiabatic S0→T1 gap)

Active space
------------
CASSCF(4,4)/6-31G* with COO biradical orbitals:
    {n⁺(O_terminal), n⁻(O_terminal), π(COO), π*(COO)}

Equilibrium NO occupations (from test_casscf_equilibrium.py):
    S0: [1.948, 1.724, 0.275, 0.052]    (moderate biradical character — physical)
    S1: [1.948, 1.000, 1.000, 0.052]    (pure open-shell singlet)
    T1: [1.949, 1.000, 1.000, 0.051]    (triplet counterpart of S1)

State-switch detection uses max |ΔNO_occ| > NO_OCC_SWITCH_THRESHOLD from the
equilibrium S0 reference above.  Unlike the old C-H active space, the COO
active space already has partial biradical character at equilibrium, so the
threshold is set to 0.20.

NM grid
-------
30 modes × [0.5, 1.0, 1.5, 2.0] × ±1 = 240 frames at T=300 K thermal amplitude.
Amplitudes capped at 2.0 × a_thermal(300 K) to stay within the wB97X ML-PES
reliable region and avoid CASSCF convergence failures at large displacements.

MO chaining: within each (mode, sign) direction, frames are processed in order
of increasing displacement factor.  The singlet SA-CASSCF MOs are chained to
the next frame.  The triplet SS-CASSCF is always seeded from the singlet MOs.
Each (mode, sign) chain is restarted from the equilibrium MOs.

Why no NEVPT2
-------------
wB97X-D already captures dynamic correlation via the range-separated functional.
Adding NEVPT2 on top of CASSCF when the reference is wB97X would double-count
dynamic correlation.  δ_S0 captures only the static (multi-reference/biradical)
correction that wB97X misses.

Usage
-----
    # Full run (default paths):
    python3 casscf_wB97X_nm_grid.py \\
        --eq-coords outputs/mvko_20260319_081314/psi4_eq_coords.npy \\
        --hessian   outputs/casscf_nm_delta_20260401_110049/hessian_used.npy

    # Resume a partial run:
    python3 casscf_wB97X_nm_grid.py \\
        --eq-coords outputs/mvko_20260319_081314/psi4_eq_coords.npy \\
        --hessian   outputs/casscf_nm_delta_20260401_110049/hessian_used.npy \\
        --resume    outputs/casscf_wB97X_nm_grid_<ts>

    # Retrain KRR only (skip CASSCF, reload saved results.json):
    python3 casscf_wB97X_nm_grid.py \\
        --eq-coords ... --hessian ... \\
        --resume outputs/casscf_wB97X_nm_grid_<ts> \\
        --retrain-only

Outputs (outputs/casscf_wB97X_nm_grid_<ts>/)
---------------------------------------------
    results.json            — per-frame checkpoint (PSI4 + CASSCF results)
    eq_reference.json       — equilibrium CASSCF S0, wB97X energies; S0-S1, S0-T1 gaps
    nm_delta_s0_model.pkl   — NMKRRDeltaModel for δ_S0 (Ha)
    nm_gap_s1_model.pkl     — NMKRRDeltaModel for Δgap_S1 (Ha)
    nm_gap_t1_model.pkl     — NMKRRDeltaModel for Δgap_T1 (Ha)
    summary.json            — statistics, LOO-CV RMSEs, filter counts
    diagnostics.png         — δ_S0 vs ||q||, gap vs ||q||, LOO-CV curves
"""

import argparse
import json
import pickle
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'modules'))

from normal_modes import compute_normal_modes, KB_HARTREE_PER_K
from data_formats import load_trajectory

HARTREE_TO_KCAL  = 627.509474
ANGSTROM_TO_BOHR = 1.88972612463
BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR
AU_TO_DEBYE      = 2.541746

MVKOO_SYMBOLS    = ['C', 'O', 'O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']

# Equilibrium SA-2-CASSCF(4,4) reference values from test_casscf_equilibrium.py
E_CASSCF_S0_EQ_HA = -304.5422196022
E_WB97X_EQ_HA     = -306.2099845838
GAP_S1_EQ_HA      =  (- 304.49597380 - E_CASSCF_S0_EQ_HA)   # +0.04624580 Ha
GAP_T1_EQ_HA      =  (-304.50002017  - E_CASSCF_S0_EQ_HA)   # +0.04219943 Ha

# Updated NO occ reference for COO active space (from equilibrium test)
EQ_NO_OCC_REF         = np.array([1.948, 1.724, 0.275, 0.052])
NO_OCC_SWITCH_THRESHOLD = 0.20   # looser than old 0.15 (COO orbitals shift more)


# ── NM grid generation ────────────────────────────────────────────────────────

def make_nm_grid(eq_coords_ang, frequencies, eigenvalues, eigvecs_mw, sqrt_mass,
                 symbols, amplitudes=(0.5, 1.0, 1.5, 2.0), T=300.0):
    """
    Generate systematic NM displacement geometries.

    Returns list of dicts with keys:
        coords_ang  : (N, 3) Angstrom
        mode_idx    : int
        sign        : +1 or -1
        factor      : float  (multiple of a_thermal)
        freq_cm1    : float
        q_nm        : (n_vib,) NM coordinate vector [sqrt(amu)·Bohr]
        a_thermal   : float  [sqrt(amu)·Bohr]
    """
    kT   = T * KB_HARTREE_PER_K
    n_at = len(symbols)
    n_vib = eigvecs_mw.shape[1]

    frames = []
    for k in range(n_vib):
        lam = eigenvalues[k]
        if lam <= 0:
            continue

        a_therm = np.sqrt(2 * kT / lam)

        for factor in amplitudes:
            for sign in (+1, -1):
                q_vec              = np.zeros(n_vib)
                q_vec[k]           = sign * factor * a_therm
                mw_disp            = eigvecs_mw @ q_vec              # (3N,) sqrt(amu)·Bohr
                cart_disp_bohr     = mw_disp / sqrt_mass             # (3N,) Bohr
                cart_disp_ang      = cart_disp_bohr.reshape(n_at, 3) * BOHR_TO_ANGSTROM
                frames.append({
                    'coords_ang': eq_coords_ang + cart_disp_ang,
                    'mode_idx':   k,
                    'sign':       sign,
                    'factor':     factor,
                    'freq_cm1':   float(frequencies[k]),
                    'q_nm':       q_vec,
                    'a_thermal':  float(a_therm),
                })
    return frames


def frame_key(frame):
    s = '+' if frame['sign'] == 1 else '-'
    return f"m{frame['mode_idx']:02d}_{s}_f{frame['factor']:.2f}"


# ── PSI4 wB97X-D single point ─────────────────────────────────────────────────

class _TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame_):
    raise _TimeoutError("PSI4 timeout")


def psi4_wb97x_point(symbols, coords_ang, basis='6-31G*',
                     method='wb97x-d', timeout_s=600):
    """
    PSI4 wB97X-D gradient at the given geometry.

    Returns (energy_Ha, forces_Ha_per_ang, error_str_or_None).
    Forces are returned for future force-training compatibility but not used
    in the delta-ML training here.
    """
    try:
        import psi4
    except ImportError:
        return None, None, 'PSI4 not available'

    mol_str = f"0 1\n"
    for s, c in zip(symbols, coords_ang):
        mol_str += f"{s}  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}\n"
    mol_str += "units angstrom\nno_reorient\nno_com\nsymmetry c1"

    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.be_quiet()
    psi4.set_memory('3 GB')
    psi4.set_num_threads(4)
    psi4.set_options({
        'basis':                basis,
        'scf_type':             'df',
        'reference':            'rhf',
        'maxiter':              200,
        'e_convergence':        1e-7,
        'd_convergence':        1e-7,
        'dft_spherical_points': 590,
        'dft_radial_points':    99,
    })

    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
        mol  = psi4.geometry(mol_str)
        gmat, wfn = psi4.gradient(f'{method}/{basis}', molecule=mol,
                                   return_wfn=True)
        signal.alarm(0)
        energy = float(wfn.energy())
        n      = len(symbols)
        grad   = np.array([[gmat.get(i, j) for j in range(3)] for i in range(n)])
        forces = -grad / ANGSTROM_TO_BOHR
        return energy, forces, None
    except _TimeoutError:
        signal.alarm(0)
        return None, None, f'timeout (>{timeout_s}s)'
    except Exception as exc:
        signal.alarm(0)
        return None, None, str(exc)


# ── PySCF SA-2-CASSCF(4,4) singlet ───────────────────────────────────────────

def run_singlet_casscf(symbols, coords_ang, basis='6-31g*',
                        n_active_orb=4, n_active_elec=4, n_states=2,
                        mo_coeff_init=None, verbose=0):
    """
    RHF → SA-n-CASSCF(4,4)/basis.

    Returns dict with e_s0, e_s1, no_occ_s0, no_occ_s1, mo_coeff,
    state_switched, converged, error.
    """
    result = {
        'e_s0': None, 'e_s1': None, 'e_hf': None,
        'no_occ_s0': None, 'no_occ_s1': None, 'mo_coeff': None,
        'state_switched': False, 'no_occ_max_dev': None,
        'converged': False, 'error': None,
    }
    try:
        from pyscf import scf, mcscf, gto
        atom_str = '; '.join(
            f'{s} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}'
            for s, c in zip(symbols, coords_ang))
        mol = gto.Mole()
        mol.atom    = atom_str
        mol.basis   = basis
        mol.charge  = 0
        mol.spin    = 0
        mol.verbose = verbose
        mol.build()

        mf = scf.RHF(mol)
        mf.max_cycle = 300
        mf.conv_tol  = 1e-9
        if mo_coeff_init is not None:
            mf.mo_coeff = mo_coeff_init
        mf.kernel()
        result['e_hf'] = float(mf.e_tot)

        weights = [1.0 / n_states] * n_states
        mc = mcscf.CASSCF(mf, n_active_orb, n_active_elec).state_average(weights)
        mc.max_cycle_macro = 300
        mc.conv_tol        = 1e-8
        mc.verbose         = verbose
        mc.kernel()

        result['converged'] = mc.converged
        result['mo_coeff']  = mc.mo_coeff.copy()
        e_states = np.array(mc.e_states)
        result['e_s0'] = float(e_states[0])
        result['e_s1'] = float(e_states[1]) if len(e_states) > 1 else None

        # Natural orbital occupations
        try:
            nelec = mc.nelecas
            if not isinstance(nelec, tuple):
                nelec = (nelec // 2, nelec - nelec // 2)
            if hasattr(mc.fcisolver, 'states_make_rdm1'):
                dm1_list = mc.fcisolver.states_make_rdm1(mc.ci, mc.ncas, nelec)
            else:
                dm1_list = [mc.fcisolver.make_rdm1(mc.ci[i], mc.ncas, nelec)
                            for i in range(len(e_states))]
            for idx, key in enumerate(('no_occ_s0', 'no_occ_s1')[:len(e_states)]):
                occ = np.sort(np.linalg.eigvalsh(dm1_list[idx]))[::-1]
                result[key] = occ
        except Exception:
            pass

        # State-switch detection on S0 (truncate reference to active space size)
        if result['no_occ_s0'] is not None:
            _n = len(result['no_occ_s0'])
            _ref = EQ_NO_OCC_REF[:_n] if _n <= len(EQ_NO_OCC_REF) else EQ_NO_OCC_REF
            dev = float(np.max(np.abs(result['no_occ_s0'] - _ref)))
            result['no_occ_max_dev'] = dev
            result['state_switched'] = dev > NO_OCC_SWITCH_THRESHOLD

    except Exception as exc:
        result['error'] = str(exc)
    return result


# ── PySCF SS-CASSCF(4,4) triplet ─────────────────────────────────────────────

def run_triplet_casscf(symbols, coords_ang, mo_coeff_seed=None,
                        basis='6-31g*', n_active_orb=4, n_active_elec=4,
                        verbose=0):
    """
    ROHF → SS-CASSCF(4,4)/basis triplet (spin=2).
    Seeded from singlet SA-CASSCF MOs to maintain active-space consistency.

    Returns dict with e_t1, no_occ_t1, converged, error.
    """
    result = {'e_t1': None, 'no_occ_t1': None,
              'converged': False, 'error': None}
    try:
        from pyscf import scf, mcscf, gto
        atom_str = '; '.join(
            f'{s} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}'
            for s, c in zip(symbols, coords_ang))
        mol = gto.Mole()
        mol.atom    = atom_str
        mol.basis   = basis
        mol.charge  = 0
        mol.spin    = 2
        mol.verbose = verbose
        mol.build()

        mf = scf.ROHF(mol)
        mf.max_cycle = 300
        mf.conv_tol  = 1e-9
        mf.kernel()

        mc = mcscf.CASSCF(mf, n_active_orb, n_active_elec)
        mc.max_cycle_macro = 300
        mc.conv_tol        = 1e-8
        mc.verbose         = verbose
        mc.kernel(mo_coeff_seed) if mo_coeff_seed is not None else mc.kernel()

        result['converged'] = mc.converged
        result['e_t1']      = float(mc.e_tot)
        try:
            nelec = mc.nelecas
            if not isinstance(nelec, tuple):
                nelec = (nelec // 2 + nelec % 2, nelec // 2)
            dm1  = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, nelec)
            result['no_occ_t1'] = np.sort(np.linalg.eigvalsh(dm1))[::-1]
        except Exception:
            pass

    except Exception as exc:
        result['error'] = str(exc)
    return result


# ── NM coordinate projection ──────────────────────────────────────────────────

def coords_to_q(coords_ang, eq_coords_ang, eigvecs_mw, sqrt_mass):
    """Project Cartesian geometry to NM coordinates. Returns (n_vib,) array."""
    delta_bohr = (coords_ang - eq_coords_ang).flatten() * ANGSTROM_TO_BOHR
    return eigvecs_mw.T @ (delta_bohr * sqrt_mass)


# ── KRR training with LOO-CV ──────────────────────────────────────────────────

def _rbf(X1, X2, gamma):
    A2 = np.sum(X1 ** 2, axis=1, keepdims=True)
    B2 = np.sum(X2 ** 2, axis=1, keepdims=True)
    return np.exp(-gamma * (A2 + B2.T - 2.0 * X1 @ X2.T))


def train_nm_krr(X_q, y_ha, label,
                 gamma_values=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
                 alpha_values=(1e-6, 1e-5, 1e-4, 1e-3)):
    """
    Train NM-coordinate KRR via LOO-CV grid search.

    X_q  : (M, n_vib)   NM coordinate training set
    y_ha : (M,)          target values in Hartree
    label: str           for printing

    Returns (best_gamma, best_alpha, best_loo_rmse_kcal, alpha_vec, K_best).
    """
    M = len(X_q)
    print(f"\n  [{label}] LOO-CV grid search  (M={M} frames):")

    best_rmse = np.inf
    best_g = best_a = None

    for gamma in gamma_values:
        for alpha in alpha_values:
            K  = _rbf(X_q, X_q, gamma)
            Kr = K.copy(); Kr[np.diag_indices_from(Kr)] += alpha
            try:
                av = np.linalg.solve(Kr, y_ha)
            except np.linalg.LinAlgError:
                continue
            # Exact LOO
            errs = []
            for i in range(M):
                mask = np.ones(M, bool); mask[i] = False
                K_tr = _rbf(X_q[mask], X_q[mask], gamma)
                K_tr[np.diag_indices_from(K_tr)] += alpha
                try:
                    av_i = np.linalg.solve(K_tr, y_ha[mask])
                except np.linalg.LinAlgError:
                    errs.append(abs(y_ha[i])); continue
                k_pred = _rbf(X_q[[i]], X_q[mask], gamma)[0]
                errs.append(float(y_ha[i] - k_pred @ av_i))
            rmse = float(np.sqrt(np.mean(np.array(errs) ** 2))) * HARTREE_TO_KCAL
            tag  = ' ← best' if rmse < best_rmse else ''
            print(f"    γ={gamma:.1e}  α={alpha:.1e}  LOO-CV={rmse:.3f} kcal/mol{tag}")
            if rmse < best_rmse:
                best_rmse = rmse; best_g = gamma; best_a = alpha

    print(f"  [{label}] Best: γ={best_g:.1e}  α={best_a:.1e}  "
          f"LOO-CV={best_rmse:.3f} kcal/mol")

    K_best  = _rbf(X_q, X_q, best_g)
    K_best[np.diag_indices_from(K_best)] += best_a
    alpha_vec = np.linalg.solve(K_best, y_ha)
    train_rmse = float(np.sqrt(np.mean((K_best @ alpha_vec - y_ha - best_a * alpha_vec) ** 2
                                        if False else   # skip; use full predict
                               (_rbf(X_q, X_q, best_g) @ alpha_vec - y_ha) ** 2
                               ))) * HARTREE_TO_KCAL

    return best_g, best_a, best_rmse, alpha_vec


def build_nm_krr_model(X_q, y_ha, label, eq_coords_ang, eigvecs_mw, sqrt_mass,
                        frequencies, symbols, gamma_values, alpha_values,
                        e_ref_ha, e_cas_ref_ha):
    """Train KRR and return a NMKRRDeltaModel (from casscf_nm_delta.py)."""
    import importlib.util as ilu
    spec = ilu.spec_from_file_location('casscf_nm_delta', REPO_ROOT / 'casscf_nm_delta.py')
    mod  = ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    NMKRRDeltaModel = mod.NMKRRDeltaModel

    best_g, best_a, best_loo, alpha_vec = train_nm_krr(
        X_q, y_ha, label, gamma_values, alpha_values)

    model = NMKRRDeltaModel(
        eq_coords_ang = eq_coords_ang,
        U_vib         = eigvecs_mw,
        sqrt_mass     = sqrt_mass,
        freqs_vib     = frequencies,
        symbols       = symbols,
        gamma         = best_g,
        alpha_reg     = best_a,
        X_train_q     = X_q,
        y_train_ha    = y_ha,
        e_b3lyp_ref_ha = e_ref_ha,
        e_cas_ref_ha   = e_cas_ref_ha,
        cv_rmse_kcal   = best_loo,
    )
    return model, best_loo


# ── Diagnostics plot ──────────────────────────────────────────────────────────

def make_diagnostics_plot(frame_list, delta_s0_kcal, gap_s1_kcal, gap_t1_kcal,
                           out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        q_norms = np.array([np.linalg.norm(f['q_nm']) for f in frame_list])
        freqs   = np.array([f['freq_cm1'] for f in frame_list])

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax = axes[0]
        sc = ax.scatter(q_norms, delta_s0_kcal, c=freqs, cmap='viridis', s=15, alpha=0.7)
        plt.colorbar(sc, ax=ax, label='Mode freq (cm⁻¹)')
        ax.axhline(0, ls='--', lw=0.8, color='gray')
        ax.set_xlabel('||q|| (√amu·Bohr)')
        ax.set_ylabel('δ_S0 (kcal/mol)')
        ax.set_title('Ground-state correction δ_S0')

        ax = axes[1]
        ax.scatter(q_norms, gap_s1_kcal, c=freqs, cmap='viridis', s=15, alpha=0.7)
        ax.axhline(GAP_S1_EQ_HA * HARTREE_TO_KCAL, ls='--', lw=0.8, color='gray',
                   label='eq gap')
        ax.set_xlabel('||q|| (√amu·Bohr)')
        ax.set_ylabel('Δgap_S1 (kcal/mol)')
        ax.set_title('S0→S1 adiabatic gap')
        ax.legend(fontsize=8)

        ax = axes[2]
        ax.scatter(q_norms, gap_t1_kcal, c=freqs, cmap='viridis', s=15, alpha=0.7)
        ax.axhline(GAP_T1_EQ_HA * HARTREE_TO_KCAL, ls='--', lw=0.8, color='gray',
                   label='eq gap')
        ax.set_xlabel('||q|| (√amu·Bohr)')
        ax.set_ylabel('Δgap_T1 (kcal/mol)')
        ax.set_title('S0→T1 adiabatic gap')
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Diagnostics plot: {out_path}")
    except Exception as exc:
        print(f"  Diagnostics plot skipped: {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Systematic SA-2-CASSCF(4,4) + triplet NM grid '
                    'with wB97X-D reference for delta-ML PES family')
    parser.add_argument('--eq-coords', default=None,
                        help='Equilibrium geometry .npy (Angstrom). '
                             'Default: training data frame 502.')
    parser.add_argument('--hessian', required=True,
                        help='B3LYP/6-31G* Hessian .npy (3N×3N, '
                             'Ha/(Bohr²·amu)) — e.g. '
                             'outputs/casscf_nm_delta_20260401_110049/hessian_used.npy')
    parser.add_argument('--amplitudes', default='0.5,1.0,1.5,2.0',
                        help='Thermal amplitude factors (default: 0.5,1.0,1.5,2.0)')
    parser.add_argument('--T-nm', type=float, default=300.0,
                        help='Temperature for thermal amplitudes (K, default: 300)')
    parser.add_argument('--n-states', type=int, default=2,
                        help='Singlet states for SA-CASSCF (default: 2)')
    parser.add_argument('--basis', default='6-31g*')
    parser.add_argument('--n-active-orb',  type=int, default=4)
    parser.add_argument('--n-active-elec', type=int, default=4)
    parser.add_argument('--no-occ-threshold', type=float,
                        default=NO_OCC_SWITCH_THRESHOLD)
    parser.add_argument('--skip-triplet', action='store_true',
                        help='Skip triplet CASSCF (saves ~25%% time)')
    parser.add_argument('--psi4-timeout', type=int, default=600,
                        help='Per-frame PSI4 timeout in seconds')
    parser.add_argument('--gamma-values', default='0.1,0.3,1.0,3.0,10.0,30.0',
                        help='KRR gamma grid for LOO-CV')
    parser.add_argument('--alpha-values', default='1e-6,1e-5,1e-4,1e-3',
                        help='KRR alpha grid for LOO-CV')
    parser.add_argument('--resume', default=None,
                        help='Output directory from a partial run to resume')
    parser.add_argument('--retrain-only', action='store_true',
                        help='Skip calculations, reload results.json and retrain KRR')
    parser.add_argument('--eq-ref-json', default=None,
                        help='JSON from test_casscf_equilibrium.py for this conformer. '
                             'Overrides the hardcoded E_CASSCF_S0_EQ_HA / E_WB97X_EQ_HA '
                             'constants so δ_S0=0 at the correct conformer equilibrium. '
                             'Keys used: e_casscf_s0_eq_ha, e_wb97x_eq_ha, '
                             'singlet.e_s1, triplet.e_t1, singlet.no_occ_s0')
    args = parser.parse_args()

    # Override module-level equilibrium constants for conformer-specific runs.
    # Without this, δ_S0 is only guaranteed to be 0 at the anti-cis equilibrium.
    if args.eq_ref_json:
        global E_CASSCF_S0_EQ_HA, E_WB97X_EQ_HA, GAP_S1_EQ_HA, GAP_T1_EQ_HA, EQ_NO_OCC_REF
        with open(args.eq_ref_json) as _fh:
            _ref = json.load(_fh)
        E_CASSCF_S0_EQ_HA = _ref['e_casscf_s0_eq_ha']
        E_WB97X_EQ_HA     = _ref['e_wb97x_eq_ha']
        _sing = _ref.get('singlet', {})
        _trip = _ref.get('triplet', {})
        _e_s1 = _sing.get('e_s1')
        _e_t1 = _trip.get('e_t1')
        if _e_s1 is not None:
            GAP_S1_EQ_HA = _e_s1 - E_CASSCF_S0_EQ_HA
        if _e_t1 is not None:
            GAP_T1_EQ_HA = _e_t1 - E_CASSCF_S0_EQ_HA
        _no_occ = _sing.get('no_occ_s0')
        if _no_occ is not None:
            EQ_NO_OCC_REF = np.array(_no_occ)
        print(f"Equilibrium reference loaded from: {args.eq_ref_json}")
        print(f"  E_CASSCF_S0(eq) = {E_CASSCF_S0_EQ_HA:.10f} Ha")
        print(f"  E_wB97X(eq)     = {E_WB97X_EQ_HA:.10f} Ha")
        print(f"  Gap_S1(eq)      = {GAP_S1_EQ_HA * HARTREE_TO_KCAL:.3f} kcal/mol")
        print(f"  Gap_T1(eq)      = {GAP_T1_EQ_HA * HARTREE_TO_KCAL:.3f} kcal/mol")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.resume) if args.resume else \
              Path(f'outputs/casscf_wB97X_nm_grid_{ts}')
    out_dir.mkdir(parents=True, exist_ok=True)

    amplitudes   = [float(x) for x in args.amplitudes.split(',')]
    gamma_values = [float(x) for x in args.gamma_values.split(',')]
    alpha_values = [float(x) for x in args.alpha_values.split(',')]

    # ── Load equilibrium geometry ─────────────────────────────────────────────
    if args.eq_coords:
        eq_coords = np.load(args.eq_coords)
    else:
        traj = load_trajectory(
            str(REPO_ROOT / 'outputs/mvko_20260319_081314/combined_training_data.npz'))
        eq_coords = traj.coordinates[502]
        print("Loaded equilibrium: training data frame 502 (wB97X minimum)")

    symbols = MVKOO_SYMBOLS

    # ── Load Hessian and compute NM modes ─────────────────────────────────────
    hessian = np.load(args.hessian)
    print(f"Hessian: {args.hessian}  (shape {hessian.shape})")
    frequencies, eigvecs_mw, eigenvalues, mass_vec = compute_normal_modes(
        symbols, hessian)
    sqrt_mass = np.sqrt(mass_vec)
    n_vib = len(frequencies)

    n_neg = int((frequencies < 0).sum())
    print(f"NM modes: {n_vib}  ({n_neg} imaginary)")
    print(f"Freq range: {frequencies[frequencies>0].min():.1f} – "
          f"{frequencies.max():.1f} cm⁻¹")

    # ── Generate grid ─────────────────────────────────────────────────────────
    grid_frames = make_nm_grid(
        eq_coords, frequencies, eigenvalues, eigvecs_mw, sqrt_mass,
        symbols, amplitudes=amplitudes, T=args.T_nm)
    print(f"\nNM grid: {len(grid_frames)} frames  "
          f"({n_vib - n_neg} modes × {len(amplitudes)} amplitudes × 2 signs)\n")

    # Processing order for MO chaining: within each (mode_idx, sign) direction,
    # sort by increasing factor so MOs chain along a smooth displacement path.
    # Each (mode, sign) chain restarts from equilibrium MOs.
    proc_order = []
    for mode_k in range(n_vib):
        for sign in (+1, -1):
            chain = [(i, f) for i, f in enumerate(grid_frames)
                     if f['mode_idx'] == mode_k and f['sign'] == sign]
            chain.sort(key=lambda x: x[1]['factor'])
            proc_order.extend(chain)

    # ── Resume: load checkpoint ───────────────────────────────────────────────
    results_path = out_dir / 'results.json'
    completed    = {}
    if results_path.exists():
        with open(results_path) as f:
            completed = json.load(f)
        n_done = sum(1 for v in completed.values() if v.get('status') == 'ok')
        print(f"  Resuming: {len(completed)} frames in checkpoint "
              f"({n_done} ok, {len(completed)-n_done} failed/skipped)")

    # ── Save equilibrium reference ────────────────────────────────────────────
    eq_ref = {
        'e_casscf_s0_eq_ha': E_CASSCF_S0_EQ_HA,
        'e_wb97x_eq_ha':     E_WB97X_EQ_HA,
        'gap_s1_eq_ha':      GAP_S1_EQ_HA,
        'gap_t1_eq_ha':      GAP_T1_EQ_HA,
        'gap_s1_eq_kcal':    GAP_S1_EQ_HA * HARTREE_TO_KCAL,
        'gap_t1_eq_kcal':    GAP_T1_EQ_HA * HARTREE_TO_KCAL,
        'no_occ_ref':        EQ_NO_OCC_REF.tolist(),
        'no_occ_threshold':  args.no_occ_threshold,
        'source':            (f'test_casscf_equilibrium.py via {args.eq_ref_json}'
                              if args.eq_ref_json else
                              'test_casscf_equilibrium.py (2026-04-07)'),
    }
    with open(out_dir / 'eq_reference.json', 'w') as f:
        json.dump(eq_ref, f, indent=2)

    # ── MO cache for chaining ─────────────────────────────────────────────────
    # Keyed by (mode_idx, sign).  Seeded with None → RHF orbitals at first frame.
    mo_cache = {}  # (mode_idx, sign) → mo_coeff or None

    # ── Run calculations ───────────────────────────────────────────────────────
    if not args.retrain_only:
        n_ok = n_fail = 0
        t_start = time.time()

        for rank, (grid_idx, frame) in enumerate(proc_order):
            key = frame_key(frame)

            # Skip if already computed successfully
            if key in completed and completed[key].get('status') == 'ok':
                n_ok += 1
                # Restore MO cache for chaining continuity
                mc_key = (frame['mode_idx'], frame['sign'])
                if completed[key].get('mo_coeff') is not None:
                    mo_cache[mc_key] = np.array(completed[key]['mo_coeff'])
                continue

            print(f"  [{rank+1:3d}/{len(proc_order)}] {key}  "
                  f"(mode {frame['mode_idx']:2d}, {frame['freq_cm1']:6.0f} cm⁻¹, "
                  f"f={frame['factor']:.1f}{'+'if frame['sign']==1 else '-'})")

            rec = {
                'status':    'failed',
                'mode_idx':  frame['mode_idx'],
                'sign':      frame['sign'],
                'factor':    frame['factor'],
                'freq_cm1':  frame['freq_cm1'],
                'q_nm':      frame['q_nm'].tolist(),
                'q_norm2':   float(np.sum(frame['q_nm'] ** 2)),
            }
            coords = frame['coords_ang']
            mc_key = (frame['mode_idx'], frame['sign'])

            # ── PSI4 wB97X-D ──────────────────────────────────────────────────
            t0 = time.time()
            e_wb, forces_wb, err_wb = psi4_wb97x_point(
                symbols, coords, basis=args.basis, timeout_s=args.psi4_timeout)
            t_wb = time.time() - t0

            if e_wb is None:
                rec['psi4_error'] = err_wb
                print(f"    PSI4 FAILED: {err_wb}")
                completed[key] = rec
                with open(results_path, 'w') as f:
                    json.dump(completed, f)
                n_fail += 1
                continue

            rec['e_wb97x_ha'] = float(e_wb)
            rec['t_psi4_s']   = float(t_wb)
            print(f"    wB97X:   {e_wb:.8f} Ha  ({t_wb:.0f}s)")

            # ── SA-2-CASSCF singlet ───────────────────────────────────────────
            t0 = time.time()
            sing = run_singlet_casscf(
                symbols, coords,
                basis           = args.basis,
                n_active_orb    = args.n_active_orb,
                n_active_elec   = args.n_active_elec,
                n_states        = args.n_states,
                mo_coeff_init   = mo_cache.get(mc_key),
            )
            t_sing = time.time() - t0

            if sing['error'] or not sing['converged'] or sing['e_s0'] is None:
                rec['casscf_error'] = sing.get('error', 'not converged')
                print(f"    CASSCF FAILED: {rec['casscf_error']}")
                completed[key] = rec
                with open(results_path, 'w') as f:
                    json.dump(completed, f)
                n_fail += 1
                continue

            # Update MO cache for next frame in this chain
            mo_cache[mc_key] = sing['mo_coeff']

            delta_s0 = (sing['e_s0'] - E_CASSCF_S0_EQ_HA) - (e_wb - E_WB97X_EQ_HA)
            gap_s1   = sing['e_s1'] - sing['e_s0'] if sing['e_s1'] is not None else None

            rec.update({
                'e_casscf_s0_ha':   float(sing['e_s0']),
                'e_casscf_s1_ha':   float(sing['e_s1']) if sing['e_s1'] else None,
                'delta_s0_ha':      float(delta_s0),
                'delta_s0_kcal':    float(delta_s0 * HARTREE_TO_KCAL),
                'gap_s1_ha':        float(gap_s1) if gap_s1 is not None else None,
                'gap_s1_kcal':      float(gap_s1 * HARTREE_TO_KCAL) if gap_s1 else None,
                'no_occ_s0':        sing['no_occ_s0'].tolist() if sing['no_occ_s0'] is not None else None,
                'no_occ_s1':        sing['no_occ_s1'].tolist() if sing['no_occ_s1'] is not None else None,
                'state_switched':   sing['state_switched'],
                'no_occ_max_dev':   sing['no_occ_max_dev'],
                'mo_coeff':         sing['mo_coeff'].tolist(),
                't_casscf_s':       float(t_sing),
            })
            print(f"    CASSCF S0: {sing['e_s0']:.8f} Ha  "
                  f"δ_S0={delta_s0*HARTREE_TO_KCAL:+.3f} kcal/mol  "
                  f"gap_S1={gap_s1*HARTREE_TO_KCAL:.1f} kcal/mol  "
                  f"{'STATE-SWITCH!' if sing['state_switched'] else 'OK'}  "
                  f"({t_sing:.0f}s)")

            # ── SS-CASSCF triplet ─────────────────────────────────────────────
            if not args.skip_triplet:
                t0 = time.time()
                trip = run_triplet_casscf(
                    symbols, coords,
                    mo_coeff_seed = sing['mo_coeff'],
                    basis         = args.basis,
                    n_active_orb  = args.n_active_orb,
                    n_active_elec = args.n_active_elec,
                )
                t_trip = time.time() - t0

                if trip['error'] or not trip['converged'] or trip['e_t1'] is None:
                    rec['triplet_error'] = trip.get('error', 'not converged')
                    print(f"    Triplet FAILED: {rec['triplet_error']}")
                else:
                    gap_t1 = trip['e_t1'] - sing['e_s0']
                    rec.update({
                        'e_casscf_t1_ha': float(trip['e_t1']),
                        'gap_t1_ha':      float(gap_t1),
                        'gap_t1_kcal':    float(gap_t1 * HARTREE_TO_KCAL),
                        'no_occ_t1':      trip['no_occ_t1'].tolist()
                                          if trip['no_occ_t1'] is not None else None,
                        't_triplet_s':    float(t_trip),
                    })
                    print(f"    Triplet T1:  {trip['e_t1']:.8f} Ha  "
                          f"gap_T1={gap_t1*HARTREE_TO_KCAL:.1f} kcal/mol  ({t_trip:.0f}s)")

            rec['status'] = 'ok'
            n_ok += 1
            completed[key] = rec

            elapsed = time.time() - t_start
            rate    = elapsed / n_ok
            remain  = (len(proc_order) - n_ok) * rate / 3600
            print(f"    [{n_ok} ok / {n_fail} fail | ETA {remain:.1f} h]")

            with open(results_path, 'w') as f:
                json.dump(completed, f)

        print(f"\nCalculations complete: {n_ok} ok, {n_fail} failed")

    # ── Assemble training arrays ──────────────────────────────────────────────
    print("\nAssembling training arrays ...")

    ok_frames    = []
    delta_s0_ha  = []
    gap_s1_ha    = []
    gap_t1_ha    = []
    n_switched   = 0
    n_no_triplet = 0

    for frame in grid_frames:
        key = frame_key(frame)
        rec = completed.get(key, {})
        if rec.get('status') != 'ok':
            continue
        if rec.get('state_switched', False):
            n_switched += 1
            print(f"    Excluded (state-switched): {key}  "
                  f"max_dev={rec.get('no_occ_max_dev', '?'):.3f}")
            continue

        delta_s0_ha.append(rec['delta_s0_ha'])
        gap_s1_ha.append(rec.get('gap_s1_ha'))
        gap_t1_ha.append(rec.get('gap_t1_ha'))
        ok_frames.append(frame)

    M = len(ok_frames)
    print(f"\n  Clean frames: {M}  "
          f"(excluded: {n_switched} state-switched, "
          f"{len(grid_frames) - len(completed)} not computed)")

    if M < 10:
        print("ERROR: too few clean frames for KRR training.  "
              "Inspect results.json and re-run or lower threshold.")
        return

    X_q = np.array([f['q_nm'] for f in ok_frames])               # (M, n_vib)
    delta_s0_arr = np.array(delta_s0_ha, dtype=float)             # (M,)
    gap_s1_arr   = np.array([g if g is not None else np.nan
                             for g in gap_s1_ha])                 # (M,)
    gap_t1_arr   = np.array([g if g is not None else np.nan
                             for g in gap_t1_ha])                 # (M,)

    np.save(out_dir / 'X_q_train.npy',      X_q)
    np.save(out_dir / 'delta_s0_ha.npy',    delta_s0_arr)
    np.save(out_dir / 'gap_s1_ha.npy',      gap_s1_arr)
    np.save(out_dir / 'gap_t1_ha.npy',      gap_t1_arr)

    print(f"\n  δ_S0 stats (kcal/mol):  "
          f"mean={delta_s0_arr.mean()*HARTREE_TO_KCAL:+.3f}  "
          f"std={delta_s0_arr.std()*HARTREE_TO_KCAL:.3f}  "
          f"range=[{delta_s0_arr.min()*HARTREE_TO_KCAL:+.3f}, "
          f"{delta_s0_arr.max()*HARTREE_TO_KCAL:+.3f}]")

    mask_s1 = ~np.isnan(gap_s1_arr)
    if mask_s1.sum() > 5:
        g = gap_s1_arr[mask_s1]
        print(f"  Δgap_S1 stats (kcal/mol): "
              f"mean={g.mean()*HARTREE_TO_KCAL:.2f}  "
              f"std={g.std()*HARTREE_TO_KCAL:.3f}  "
              f"range=[{g.min()*HARTREE_TO_KCAL:.2f}, {g.max()*HARTREE_TO_KCAL:.2f}]")

    mask_t1 = ~np.isnan(gap_t1_arr)
    if mask_t1.sum() > 5:
        g = gap_t1_arr[mask_t1]
        print(f"  Δgap_T1 stats (kcal/mol): "
              f"mean={g.mean()*HARTREE_TO_KCAL:.2f}  "
              f"std={g.std()*HARTREE_TO_KCAL:.3f}  "
              f"range=[{g.min()*HARTREE_TO_KCAL:.2f}, {g.max()*HARTREE_TO_KCAL:.2f}]")

    # ── Train KRR models ──────────────────────────────────────────────────────
    print("\nTraining KRR models ...")
    summary = {
        'n_grid_frames':   len(grid_frames),
        'n_computed_ok':   sum(1 for v in completed.values() if v.get('status')=='ok'),
        'n_state_switched': n_switched,
        'n_clean':          M,
        'eq_reference':     eq_ref,
        'amplitudes':       amplitudes,
        'T_nm_K':           args.T_nm,
        'basis':            args.basis,
    }

    # δ_S0 model
    model_s0, loo_s0 = build_nm_krr_model(
        X_q, delta_s0_arr, 'δ_S0',
        eq_coords, eigvecs_mw, sqrt_mass, frequencies, symbols,
        gamma_values, alpha_values,
        e_ref_ha  = E_WB97X_EQ_HA,
        e_cas_ref_ha = E_CASSCF_S0_EQ_HA,
    )
    model_s0.save(str(out_dir / 'nm_delta_s0_model.pkl'))
    summary['s0_loo_cv_kcal'] = loo_s0
    summary['s0_gamma'] = model_s0.gamma
    summary['s0_alpha'] = model_s0.alpha_reg

    # Δgap_S1 model (only where S1 available)
    if mask_s1.sum() >= 10:
        model_s1, loo_s1 = build_nm_krr_model(
            X_q[mask_s1], gap_s1_arr[mask_s1], 'Δgap_S1',
            eq_coords, eigvecs_mw, sqrt_mass, frequencies, symbols,
            gamma_values, alpha_values,
            e_ref_ha  = E_WB97X_EQ_HA,
            e_cas_ref_ha = E_CASSCF_S0_EQ_HA,
        )
        model_s1.save(str(out_dir / 'nm_gap_s1_model.pkl'))
        summary['s1_loo_cv_kcal'] = loo_s1
        summary['s1_n_frames']    = int(mask_s1.sum())
    else:
        print("  Δgap_S1: too few frames, skipping model")

    # Δgap_T1 model
    if not args.skip_triplet and mask_t1.sum() >= 10:
        model_t1, loo_t1 = build_nm_krr_model(
            X_q[mask_t1], gap_t1_arr[mask_t1], 'Δgap_T1',
            eq_coords, eigvecs_mw, sqrt_mass, frequencies, symbols,
            gamma_values, alpha_values,
            e_ref_ha  = E_WB97X_EQ_HA,
            e_cas_ref_ha = E_CASSCF_S0_EQ_HA,
        )
        model_t1.save(str(out_dir / 'nm_gap_t1_model.pkl'))
        summary['t1_loo_cv_kcal'] = loo_t1
        summary['t1_n_frames']    = int(mask_t1.sum())
    else:
        print("  Δgap_T1: skipped")

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Diagnostics plot ──────────────────────────────────────────────────────
    make_diagnostics_plot(
        ok_frames,
        delta_s0_arr * HARTREE_TO_KCAL,
        gap_s1_arr   * HARTREE_TO_KCAL,
        gap_t1_arr   * HARTREE_TO_KCAL,
        out_dir / 'diagnostics.png',
    )

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"COMPLETE — outputs in {out_dir}")
    print(f"{'='*60}")
    print(f"  Clean frames:     {M} / {len(grid_frames)}")
    print(f"  δ_S0  LOO-CV:     {loo_s0:.3f} kcal/mol"
          f"  {'✓ < 0.5' if loo_s0 < 0.5 else '✗ > target 0.5'}")
    if 's1_loo_cv_kcal' in summary:
        print(f"  Δgap_S1 LOO-CV:   {summary['s1_loo_cv_kcal']:.3f} kcal/mol")
    if 't1_loo_cv_kcal' in summary:
        print(f"  Δgap_T1 LOO-CV:   {summary['t1_loo_cv_kcal']:.3f} kcal/mol")
    print()
    print("Models:")
    print(f"  nm_delta_s0_model.pkl  → use with --nm-delta-model in ir_md_spectrum.py")
    print(f"  nm_gap_s1_model.pkl    → S0→S1 gap surface")
    print(f"  nm_gap_t1_model.pkl    → S0→T1 gap surface (ISC)")
    print()
    print("Next steps:")
    print("  1. Check diagnostics.png — δ_S0 should be smooth, range ±5 kcal/mol near eq")
    print("  2. Run IR spectrum with δ_S0 correction:")
    print(f"     python3 ir_md_spectrum.py --nm-delta-model {out_dir}/nm_delta_s0_model.pkl \\")
    print( "       --model outputs/wB97X_surface_20260406_223155/mlpes_wB97X.pkl \\")
    print( "       --training-data outputs/wB97X_surface_20260406_223155/training_data_wB97X.npz \\")
    print( "       --steps 30000 --temp 300 --preminimize --zpe-min-freq 50 --zpe-max-freq 4000")


if __name__ == '__main__':
    main()
