#!/usr/bin/env python3
"""
casscf_irc_correction.py — CASSCF(4,4) single points on filtered IRC frames,
delta-ML correction, and diabatic coupling profile for syn-MVKO → VHP.

Pipeline
--------
  1. Load the 9 IRC frames with ΔE ≤ 100 kcal/mol (outputs from mvko_syn_oh_path.py)
  2. For each frame (ordered MVKO → VHP by IRC coordinate s):
       a. RHF/6-31G*  — initial orbital guess
       b. SS-CASSCF(4,4)/6-31G*  — ground-state energy + natural orbital occupations
       c. SA-2-CASSCF(4,4)/6-31G*  — two adiabatic states, diabatic coupling H₁₂ ≈ ΔE₀₁/2
       d. DMRG-CASPT2 (optional)  — dynamic correlation; skipped if CheMPS2 unavailable
  3. Compare B3LYP vs CASSCF barriers; flag if CASSCF(6,6) is needed
  4. Train a delta-ML KRR on  ΔE = E_CASSCF_root0 − E_B3LYP  (9 frames)
  5. Assemble corrected PESFamily: reactant + rxn_path + delta-ML
  6. Write diagnostic figure: 4-panel IRC comparison

Active space  CASSCF(4,4)  for the 1,4-H shift (C3–H7 → O2):
  σ (C3–H7)   breaking bond
  σ*(C3–H7)   breaking antibond
  σ (O2–H7)   forming bond
  σ*(O2–H7)   forming antibond

MVKO  C₄H₆O₂  46 electrons:
  frozen_docc    = 6   (1s cores: 4×C + 2×O)
  restricted_docc = 15  ((46 − 12_core − 4_active) / 2)
  active          = 4

Active space check: if any active-space NO occupation lies between 0.10 and 1.90
at a frame near the MVKO minimum (s < −2.0), the active space is probably
capturing the wrong orbitals — a warning is printed and orbital rotation may
be needed. At the TS (s ≈ 0), occupations close to (1, 1, 1, 1) confirm
correct biradical description. If the outermost occupations remain near
(2, 2, 0, 0) everywhere, extend to CASSCF(6,6) with the O1–O2 π/π* pair.

Multi-reference warning
-----------------------
B3LYP is single-reference. The delta-ML correction trained here accounts for
*static* correlation recovered by CASSCF. Dynamic correlation (NEVPT2 / CASPT2)
typically adds 5–10 kcal/mol to the barrier. To include dynamic correlation:
  - Install forte:  conda install forte -c conda-forge/label/forte
  - Or use DMRG-CASPT2 (dmrg_scf + CheMPS2 link) — see commented block below.

Usage
-----
  # All 9 frames, full pipeline:
  python3 casscf_irc_correction.py \\
      --irc-data outputs/mvko_rxn_path_20260330_181025/irc_training_data.npz \\
      --reactant-model outputs/mvko_20260319_081314/mlpes_initial.pkl \\
      --rxn-path-family outputs/mvko_rxn_path_20260330_202029/pes_family.pkl

  # Skip PSI4 and reload saved results (re-plot / re-train delta-ML):
  python3 casscf_irc_correction.py \\
      --load-results outputs/casscf_irc_<ts>/casscf_results.json

  # Test with only the TS frame (s ≈ 0):
  python3 casscf_irc_correction.py --irc-data ... --ts-only
"""

import argparse
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

HARTREE_TO_KCAL = 627.509474
E_FILTER_KCAL   = 100.0   # keep IRC frames within this of the minimum

# ── MVKO active space constants ───────────────────────────────────────────────
N_ELEC_ACTIVE      = 4    # CASSCF(4,4)
N_ORBS_ACTIVE      = 4
N_FROZEN_CORE      = 6    # 1s: 4×C + 2×O
N_RESTRICTED_DOCC  = 15   # (46 − 12 − 4) / 2


# ── Geometry helpers ──────────────────────────────────────────────────────────

def geometry_string(symbols, coords, charge=0, mult=1):
    """Build a PSI4-ready geometry block (Angstrom, no_reorient, no_com)."""
    lines = [f"{charge} {mult}"]
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"  {sym}  {x:.10f}  {y:.10f}  {z:.10f}")
    lines += ["units angstrom", "no_reorient", "no_com", "symmetry c1"]
    return "\n".join(lines)


# ── PSI4 output parsing ───────────────────────────────────────────────────────

def _parse_no_occupations(output_text: str) -> list[float] | None:
    """
    Extract active-space natural orbital occupations from PSI4 CASSCF output.
    Returns a list of floats in descending order, or None if not found.
    """
    # Pattern: "Occupation Numbers:" block (PSI4 DETCI CASSCF format)
    m = re.search(
        r'Occupation Numbers:\s*\n((?:\s+\d+\s+[\d.]+\s*\n)+)',
        output_text)
    if m:
        nums = re.findall(r'\d+\s+([\d.]+)', m.group(1))
        if nums:
            return sorted([float(x) for x in nums], reverse=True)
    # Fallback: look for "Natural Orbital Occupations" table
    m2 = re.search(
        r'Natural Orbital Occup[^\n]*\n[-\s]+\n((?:\s+\S+\s+\d+\s+[\d.]+\s*\n)+)',
        output_text)
    if m2:
        nums = re.findall(r'([\d.]+)\s*\n', m2.group(1))
        if nums:
            return sorted([float(x) for x in nums], reverse=True)
    return None


def _parse_ci_root_energies(output_text: str, n_roots: int = 2) -> list[float]:
    """Extract per-root energies from SA-CASSCF output."""
    energies = []
    for i in range(n_roots):
        m = re.search(
            rf'CI Root\s+{i+1}\s+energy\s*=\s*(-[\d.]+)',
            output_text, re.IGNORECASE)
        if not m:
            m = re.search(
                rf'ROOT\s+{i}\s+ECI\s*=\s*(-[\d.]+)',
                output_text, re.IGNORECASE)
        if m:
            energies.append(float(m.group(1)))
    return energies


# ── Per-frame PSI4 calculations ───────────────────────────────────────────────

def run_frame(symbols, coords, irc_s, e_b3lyp, frame_idx,
              out_dir: Path, run_caspt2: bool = False,
              n_threads: int = 4, memory: str = '6 GB') -> dict:
    """
    Run RHF → SS-CASSCF(4,4) → SA-2-CASSCF(4,4) [→ DMRG-CASPT2] for one geometry.
    Returns a result dict.
    """
    try:
        import psi4
    except ImportError:
        print("  PSI4 not available — cannot run CASSCF.")
        return {'frame_idx': frame_idx, 'irc_s': float(irc_s), 'error': 'no_psi4'}

    psi4.core.clean()
    frame_outfile = str(out_dir / f'psi4_frame{frame_idx:02d}_s{irc_s:+.2f}.dat')
    psi4.core.set_output_file(frame_outfile, False)

    geom = psi4.geometry(geometry_string(symbols, coords))
    psi4.set_memory(memory)
    psi4.set_num_threads(n_threads)

    result = {
        'frame_idx':    int(frame_idx),
        'irc_s':        float(irc_s),
        'e_b3lyp':      float(e_b3lyp),
        'e_b3lyp_rel':  float((e_b3lyp - 0.0) * HARTREE_TO_KCAL),  # filled later
        'e_rhf':        None,
        'e_casscf_ss':  None,
        'e_sa_root0':   None,
        'e_sa_root1':   None,
        'gap_kcal':     None,
        'H12_kcal':     None,
        'no_occs':      None,
        'e_caspt2':     None,
        'delta_kcal':   None,
        'output_file':  frame_outfile,
    }

    base_opts = {
        'basis':         '6-31G*',
        'scf_type':      'df',
        'reference':     'rhf',
        'e_convergence': 1e-8,
        'd_convergence': 1e-8,
        'maxiter':       200,
    }

    # ── Step 1: RHF ──────────────────────────────────────────────────────────
    print(f"\n  Frame {frame_idx:02d}  s = {irc_s:+.2f}  ΔE_B3LYP = "
          f"{(e_b3lyp)*HARTREE_TO_KCAL:.1f} kcal/mol (raw)")
    psi4.set_options(base_opts)
    try:
        E_rhf, wfn_rhf = psi4.energy('hf', return_wfn=True)
        result['e_rhf'] = float(E_rhf)
        print(f"    RHF:      {E_rhf:.8f} Ha")
    except Exception as exc:
        print(f"    RHF FAILED: {exc}")
        result['error'] = f'rhf: {exc}'
        return result

    # ── Step 2: SS-CASSCF(4,4) ───────────────────────────────────────────────
    casscf_opts = {
        **base_opts,
        'frozen_docc':     [N_FROZEN_CORE],
        'restricted_docc': [N_RESTRICTED_DOCC],
        'active':          [N_ORBS_ACTIVE],
        'num_roots':       1,
        'mcscf_algorithm': 'ah',       # augmented Hessian — more robust
        'mcscf_maxiter':   200,
        'mcscf_diis_start': 3,
        'mcscf_r_convergence': 1e-6,
        'mcscf_e_convergence': 1e-9,
    }
    psi4.set_options(casscf_opts)
    try:
        E_cas, wfn_cas = psi4.energy('casscf', return_wfn=True, ref_wfn=wfn_rhf)
        result['e_casscf_ss'] = float(E_cas)
        print(f"    CASSCF:   {E_cas:.8f} Ha")

        # Extract natural orbital occupations from output file
        try:
            with open(frame_outfile) as f:
                out_text = f.read()
            no_occs = _parse_no_occupations(out_text)
            if no_occs:
                result['no_occs'] = no_occs[:N_ORBS_ACTIVE]
                print(f"    NO occs:  {[f'{x:.4f}' for x in result['no_occs']]}")
                # Warn if active space may be wrong
                inner_occs = result['no_occs'][1:-1]   # two middle orbitals
                if irc_s > -1.5 and all(abs(o - 1.0) < 0.3 for o in inner_occs):
                    print(f"    ✓ Biradical character confirmed at s={irc_s:+.2f}")
                elif irc_s < -2.0 and any(0.1 < o < 1.9 for o in result['no_occs']):
                    print(f"    ⚠ Unexpected fractional occ at MVKO minimum — "
                          f"check active space orbital order")
        except Exception:
            pass

    except Exception as exc:
        print(f"    CASSCF FAILED: {exc}")
        result['error'] = f'casscf_ss: {exc}'
        return result

    # ── Step 3: SA-2-CASSCF(4,4) for diabatic coupling ───────────────────────
    sa_opts = {
        **casscf_opts,
        'num_roots':       2,
        'avg_states':      [0, 1],
        'avg_weights':     [0.5, 0.5],
        'mcscf_maxiter':   250,
    }
    psi4.set_options(sa_opts)
    try:
        E_sa, wfn_sa = psi4.energy('casscf', return_wfn=True, ref_wfn=wfn_cas)

        # Extract per-root energies from PSI4 variables
        E0 = psi4.variable('CI ROOT 0 TOTAL ENERGY')
        E1 = psi4.variable('CI ROOT 1 TOTAL ENERGY')
        if E0 == 0.0 and E1 == 0.0:
            # fallback: parse from output file
            with open(frame_outfile) as f:
                out_text = f.read()
            roots = _parse_ci_root_energies(out_text, n_roots=2)
            if len(roots) == 2:
                E0, E1 = roots[0], roots[1]

        gap = (E1 - E0) * HARTREE_TO_KCAL
        H12 = gap / 2.0   # minimum-gap estimate: valid near the crossing

        result['e_sa_root0'] = float(E0)
        result['e_sa_root1'] = float(E1)
        result['gap_kcal']   = float(gap)
        result['H12_kcal']   = float(H12)
        print(f"    SA root0: {E0:.8f} Ha")
        print(f"    SA root1: {E1:.8f} Ha")
        print(f"    gap:      {gap:.2f} kcal/mol   H₁₂ ≈ {H12:.2f} kcal/mol")

    except Exception as exc:
        print(f"    SA-CASSCF FAILED: {exc}")
        # Non-fatal: continue with SS result

    # ── Step 4 (optional): DMRG-CASPT2 for dynamic correlation ───────────────
    if run_caspt2:
        try:
            caspt2_opts = {
                **sa_opts,
                'dmrg_sweep_states':     [500],
                'dmrg_sweep_energy_conv': [1e-8],
                'dmrg_sweep_max_sweeps': [20],
            }
            psi4.set_options(caspt2_opts)
            E_caspt2 = psi4.energy('dmrg-caspt2', ref_wfn=wfn_sa)
            result['e_caspt2'] = float(E_caspt2)
            print(f"    DMRG-CASPT2: {E_caspt2:.8f} Ha")
        except Exception as exc:
            print(f"    DMRG-CASPT2 skipped ({exc.__class__.__name__}): "
                  f"install CheMPS2-linked PSI4 or forte for dynamic correlation")

    return result


# ── Delta-ML training ─────────────────────────────────────────────────────────

def train_delta_ml(results: list[dict], symbols: list[str],
                   coords_sorted: np.ndarray, out_dir: Path,
                   use_caspt2: bool = False):
    """
    Train a KRR delta-ML model on the RELATIVE correction:
        delta(i) = ΔE_CASSCF(i) - ΔE_B3LYP(i)
    where both ΔE are referenced to the minimum-energy frame.

    This correctly captures how the CASSCF barrier SHAPE differs from B3LYP,
    avoiding the ~1.8 Ha absolute offset due to dynamic correlation in DFT.

    Uses the SA-CASSCF root-0 energy (ground adiabatic state) as the
    high-level reference; falls back to SS-CASSCF if SA failed.
    Very wide kernel (gamma=0.00005) for smooth interpolation over 9 sparse frames.

    Returns (trainer, training_npz_path).
    """
    from modules.ml_pes import MLPESTrainer, MLPESConfig
    from modules.data_formats import TrajectoryData

    # Collect frames with both CASSCF and B3LYP energies
    raw = []
    for r, c in zip(results, coords_sorted):
        e_hi = r.get('e_caspt2') or r.get('e_sa_root0') or r.get('e_casscf_ss')
        e_lo = r.get('e_b3lyp')
        if e_hi is not None and e_lo is not None:
            raw.append({'coords': c, 'e_cas': e_hi, 'e_b3lyp': e_lo,
                        'irc_s': r['irc_s']})

    if len(raw) < 3:
        print(f"  Only {len(raw)} usable frames for delta-ML — skipping")
        return None, None

    # Compute RELATIVE correction: delta_rel(i) = ΔE_CASSCF(i) - ΔE_B3LYP(i)
    # Both ΔE are referenced to the minimum-energy frame of each method
    e_cas_ref   = min(g['e_cas']    for g in raw)   # CASSCF min frame (Ha)
    e_b3lyp_ref = min(g['e_b3lyp'] for g in raw)    # B3LYP min frame (Ha)
    good = []
    for g in raw:
        dE_cas   = g['e_cas']    - e_cas_ref     # Ha
        dE_b3lyp = g['e_b3lyp'] - e_b3lyp_ref   # Ha
        delta_rel = dE_cas - dE_b3lyp            # Ha  (correction to relative energy)
        good.append({**g, 'delta_ha': delta_rel,
                     'delta_kcal': delta_rel * HARTREE_TO_KCAL})

    coords_arr   = np.array([g['coords']   for g in good])
    energies_arr = np.array([g['delta_ha'] for g in good])
    forces_arr   = np.zeros_like(coords_arr)

    print(f"\n  Delta-ML training: {len(good)} frames")
    print(f"  CASSCF barrier (relative): "
          f"{(max(g['e_cas'] for g in raw) - e_cas_ref)*HARTREE_TO_KCAL:.1f} kcal/mol")
    print(f"  B3LYP  barrier (relative): "
          f"{(max(g['e_b3lyp'] for g in raw) - e_b3lyp_ref)*HARTREE_TO_KCAL:.1f} kcal/mol")
    print(f"  delta_rel range: {energies_arr.min()*HARTREE_TO_KCAL:.2f} to "
          f"{energies_arr.max()*HARTREE_TO_KCAL:.2f} kcal/mol")

    traj = TrajectoryData(
        symbols=symbols,
        coordinates=coords_arr,
        energies=energies_arr,
        forces=forces_arr,
        dipoles=None,
        metadata={'source': 'casscf_delta_correction', 'n_frames': len(good)},
    )

    # Wide kernel: small gamma so correction smoothly interpolates across sparse IRC
    config = MLPESConfig()
    config.gamma                = 0.00005
    config.alpha                = 1e-10
    config.tune_hyperparameters = False

    trainer = MLPESTrainer(config)
    trainer.train(traj)

    delta_pkl = out_dir / 'delta_ml_casscf.pkl'
    trainer.save(str(delta_pkl))

    # Save training data
    npz_path = out_dir / 'delta_ml_training.npz'
    np.savez(str(npz_path),
             symbols=np.array(symbols),
             coordinates=coords_arr,
             delta_energies_ha=energies_arr,
             irc_s=np.array([g['irc_s'] for g in good]))
    print(f"  Delta-ML model saved: {delta_pkl}")
    print(f"  Training data saved : {npz_path}")
    return trainer, delta_pkl


# ── Corrected PESFamily ───────────────────────────────────────────────────────

def save_corrected_manifest(rxn_family_pkl: str, delta_pkl: str,
                            out_dir: Path) -> Path:
    """
    Write a manifest JSON that downstream scripts can use to apply the
    delta-ML correction on top of the PESFamily.

    The correction is applied as:
        E_corrected(R) = E_PESFamily(R) + taper(R) * E_delta_ML(R)

    where taper(R) is a Gaussian in IRC coordinate, centred at the TS energy
    (~37 kcal/mol above minimum), with σ = 25 kcal/mol, so it activates only
    in the TS region and decays to zero at the MVKO equilibrium and the VHP
    product well.
    """
    manifest = {
        'pes_family_pkl':    str(rxn_family_pkl),
        'delta_ml_pkl':      str(delta_pkl),
        'taper': {
            'type':          'gaussian_energy',
            'centre_kcal':   37.0,
            'sigma_kcal':    25.0,
            'description':   'Gaussian taper in ΔE above MVKO minimum; '
                             'correction active in TS region (15–80 kcal/mol)'
        },
        'level':             'SA-2-CASSCF(4,4)/6-31G* − B3LYP/6-31G*',
        'note':              'Dynamic correlation not included (NEVPT2/CASPT2 pending). '
                             'Install forte for NEVPT2 correction.',
    }
    manifest_path = out_dir / 'corrected_family_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Corrected manifest: {manifest_path}")
    return manifest_path


# ── Diagnostic figure ─────────────────────────────────────────────────────────

def plot_results(results: list[dict], e_b3lyp_min: float, out_dir: Path):
    """4-panel diagnostic figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        s_vals    = np.array([r['irc_s']            for r in results])
        e_b3lyp   = np.array([r['e_b3lyp']          for r in results])
        e_cas_ss  = np.array([r.get('e_casscf_ss') or np.nan for r in results])
        e_sa_r0   = np.array([r.get('e_sa_root0')  or np.nan for r in results])
        e_sa_r1   = np.array([r.get('e_sa_root1')  or np.nan for r in results])
        gap       = np.array([r.get('gap_kcal')    or np.nan for r in results])
        H12       = np.array([r.get('H12_kcal')    or np.nan for r in results])
        e_caspt2  = np.array([r.get('e_caspt2')    or np.nan for r in results])

        # Align B3LYP to its own minimum
        e_ref      = e_b3lyp_min
        dE_b3lyp   = (e_b3lyp  - e_ref) * HARTREE_TO_KCAL

        # Align CASSCF energies to CASSCF minimum (may differ from B3LYP min frame)
        cas_valid = e_cas_ss[~np.isnan(e_cas_ss)]
        e_cas_ref  = np.nanmin(e_cas_ss) if len(cas_valid) else e_ref
        e_sa_ref   = np.nanmin(e_sa_r0)  if not np.all(np.isnan(e_sa_r0)) else e_ref
        e_pt2_ref  = np.nanmin(e_caspt2) if not np.all(np.isnan(e_caspt2)) else e_ref

        dE_cas    = (e_cas_ss - e_cas_ref) * HARTREE_TO_KCAL
        dE_sa_r0  = (e_sa_r0  - e_sa_ref)  * HARTREE_TO_KCAL
        dE_sa_r1  = (e_sa_r1  - e_sa_ref)  * HARTREE_TO_KCAL
        dE_caspt2 = (e_caspt2 - e_pt2_ref) * HARTREE_TO_KCAL

        # Relative delta correction: ΔE_CASSCF(i) - ΔE_B3LYP(i) (both relative to min)
        dE_sa_r0_b3lyp_ref = (e_sa_r0 - e_sa_ref) * HARTREE_TO_KCAL
        dE_b3lyp_for_delta = (e_b3lyp  - e_ref)    * HARTREE_TO_KCAL
        delta_kcal = dE_sa_r0_b3lyp_ref - dE_b3lyp_for_delta   # relative correction

        fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                                 gridspec_kw={'hspace': 0.38, 'wspace': 0.32})
        ax1, ax2, ax3, ax4 = axes.flat

        # ── Panel 1: Energy comparison ────────────────────────────────────────
        ax1.plot(s_vals, dE_b3lyp, 'o-', color='steelblue', lw=1.8,
                 ms=7, label='B3LYP/6-31G*')
        ax1.plot(s_vals, dE_cas,   's--', color='darkorange', lw=1.5,
                 ms=7, label='SS-CASSCF(4,4)')
        ax1.plot(s_vals, dE_sa_r0, '^-', color='crimson', lw=1.5,
                 ms=7, label='SA-CASSCF root 0')
        ax1.plot(s_vals, dE_sa_r1, 'v:', color='salmon', lw=1.2,
                 ms=5, label='SA-CASSCF root 1')
        if not np.all(np.isnan(dE_caspt2)):
            ax1.plot(s_vals, dE_caspt2, 'D-', color='purple', lw=1.5,
                     ms=6, label='DMRG-CASPT2')
        ax1.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax1.set_xlabel('IRC coordinate (Å·√amu)', fontsize=11)
        ax1.set_ylabel('ΔE (kcal/mol)', fontsize=11)
        ax1.set_title('Energy comparison: B3LYP vs CASSCF(4,4)', fontsize=11)
        ax1.legend(fontsize=8)

        # ── Panel 2: Delta correction ΔE = E_CASSCF − E_B3LYP ───────────────
        ax2.plot(s_vals, delta_kcal, 'o-', color='seagreen', lw=1.8, ms=7)
        ax2.axhline(0, color='gray', lw=0.8, ls='--')
        ax2.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax2.fill_between(s_vals, 0, delta_kcal, alpha=0.15, color='seagreen')
        ax2.set_xlabel('IRC coordinate (Å·√amu)', fontsize=11)
        ax2.set_ylabel('ΔE_corr (kcal/mol)', fontsize=11)
        ax2.set_title('Delta correction: E_CASSCF_root0 − E_B3LYP', fontsize=11)
        ax2.text(0.05, 0.92,
                 f'Max correction: {np.nanmax(np.abs(delta_kcal)):.1f} kcal/mol',
                 transform=ax2.transAxes, fontsize=9)

        # ── Panel 3: Diabatic coupling H₁₂ ───────────────────────────────────
        valid = ~np.isnan(H12)
        if valid.any():
            ax3.plot(s_vals[valid], H12[valid], 'o-', color='firebrick',
                     lw=1.8, ms=7, label='H₁₂ ≈ ΔE₀₁/2')
            ax3.fill_between(s_vals[valid], 0, H12[valid],
                             alpha=0.15, color='firebrick')
            ax3.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
            ax3.set_xlabel('IRC coordinate (Å·√amu)', fontsize=11)
            ax3.set_ylabel('H₁₂ (kcal/mol)', fontsize=11)
            ax3.set_title('Diabatic coupling  H₁₂ ≈ ΔE₀₁/2\n'
                          '(exact at crossing; upper bound elsewhere)', fontsize=10)
            ax3.legend(fontsize=9)
            # Annotate max coupling
            i_max = np.nanargmax(H12)
            ax3.annotate(f'{H12[i_max]:.1f} kcal/mol',
                         xy=(s_vals[i_max], H12[i_max]),
                         xytext=(s_vals[i_max] + 0.3, H12[i_max] * 0.9),
                         fontsize=9, color='firebrick',
                         arrowprops=dict(arrowstyle='->', color='firebrick', lw=1))
        else:
            ax3.text(0.5, 0.5, 'SA-CASSCF did not converge\nfor any frame',
                     transform=ax3.transAxes, ha='center', va='center',
                     fontsize=11, color='gray')
            ax3.set_title('Diabatic coupling H₁₂', fontsize=11)

        # ── Panel 4: Natural orbital occupations (biradical character) ────────
        no_data = [r.get('no_occs') for r in results]
        has_no  = [o for o in no_data if o is not None]
        if has_no:
            n_active = min(N_ORBS_ACTIVE, min(len(o) for o in has_no))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            labels = ['σ(C3–H7)', 'HOMO−1', 'LUMO', 'σ*(O2–H7)']
            for k in range(n_active):
                s_k   = [s_vals[i] for i, o in enumerate(no_data) if o is not None]
                occ_k = [o[k]      for o in no_data         if o is not None]
                ax4.plot(s_k, occ_k, 'o-', color=colors[k], lw=1.5,
                         ms=6, label=labels[k])
            ax4.axhline(1.0, color='gray', lw=0.7, ls='--', alpha=0.5,
                        label='n=1 (biradical)')
            ax4.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
            ax4.set_ylim(-0.05, 2.05)
            ax4.set_xlabel('IRC coordinate (Å·√amu)', fontsize=11)
            ax4.set_ylabel('NO occupation', fontsize=11)
            ax4.set_title('Active-space natural orbital occupations\n'
                          '(n≈1 for both inner orbitals → biradical at TS)',
                          fontsize=10)
            ax4.legend(fontsize=8)
            # Mark if any occupation is in the suspicious range near MVKO min
            for i, o in enumerate(no_data):
                if o and s_vals[i] < -2.0:
                    if any(0.10 < x < 1.90 for x in o):
                        ax4.axvline(s_vals[i], color='orange', lw=1, ls=':',
                                    alpha=0.7)
        else:
            ax4.text(0.5, 0.5, 'Natural orbital occupations\nnot parsed',
                     transform=ax4.transAxes, ha='center', va='center',
                     fontsize=11, color='gray')
            ax4.set_title('Active-space NO occupations', fontsize=11)

        fig.suptitle(
            'CASSCF(4,4)/6-31G* IRC analysis: syn-MVKO → VHP\n'
            'Static correlation correction and diabatic coupling profile',
            fontsize=12, fontweight='bold')

        fig_path = out_dir / 'casscf_irc_analysis.png'
        fig.savefig(str(fig_path), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  Figure saved: {fig_path}")

    except Exception as exc:
        print(f"  (Plot failed: {exc})")
        traceback.print_exc()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--irc-data',
                        default='outputs/mvko_rxn_path_20260330_181025/irc_training_data.npz',
                        help='Filtered IRC training data (.npz)')
    parser.add_argument('--reactant-model',
                        default='outputs/mvko_20260319_081314/mlpes_initial.pkl',
                        help='Near-eq MVKO ML-PES for PESFamily assembly')
    parser.add_argument('--rxn-path-family',
                        default='outputs/mvko_rxn_path_20260330_202029/pes_family.pkl',
                        help='Existing PESFamily pkl to annotate with delta correction')
    parser.add_argument('--load-results', default=None,
                        help='Path to casscf_results.json — skip PSI4, go straight to '
                             'delta-ML training and plotting')
    parser.add_argument('--run-caspt2', action='store_true',
                        help='Attempt DMRG-CASPT2 (requires CheMPS2-linked PSI4)')
    parser.add_argument('--ts-only', action='store_true',
                        help='Process only the TS frame (smallest |s|) — quick test')
    parser.add_argument('--n-threads', type=int, default=4,
                        help='PSI4 threads per calculation [default 4]')
    parser.add_argument('--memory', default='6 GB',
                        help='PSI4 memory per calculation [default "6 GB"]')
    args = parser.parse_args()

    ts_str  = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(f'outputs/casscf_irc_{ts_str}')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Output directory: {out_dir}")

    # ── Load IRC data ─────────────────────────────────────────────────────────
    if args.load_results is None:
        data     = np.load(args.irc_data, allow_pickle=True)
        symbols  = data['symbols'].tolist()
        coords   = data['coordinates']
        energies = data['energies']
        irc_s    = data['irc_s']

        e_min    = energies.min()
        mask     = (energies - e_min) * HARTREE_TO_KCAL < E_FILTER_KCAL
        coords   = coords[mask]
        energies = energies[mask]
        irc_s    = irc_s[mask]

        # Sort by IRC coordinate (MVKO → VHP)
        order    = np.argsort(irc_s)
        coords   = coords[order]
        energies = energies[order]
        irc_s    = irc_s[order]

        n_frames = len(coords)
        print(f"\n  IRC frames loaded: {n_frames} (ΔE ≤ {E_FILTER_KCAL:.0f} kcal/mol)")
        print(f"  IRC s range: {irc_s.min():+.2f} to {irc_s.max():+.2f} Å·√amu")
        print(f"  B3LYP energy range: 0 to "
              f"{(energies.max()-e_min)*HARTREE_TO_KCAL:.1f} kcal/mol")
        print(f"\n  Active space: CASSCF({N_ELEC_ACTIVE},{N_ORBS_ACTIVE})/6-31G*")
        print(f"  frozen_docc = {N_FROZEN_CORE}  |  "
              f"restricted_docc = {N_RESTRICTED_DOCC}  |  "
              f"active = {N_ORBS_ACTIVE}")
        print(f"  Electron check: 2×{N_FROZEN_CORE} + 2×{N_RESTRICTED_DOCC} + "
              f"{N_ELEC_ACTIVE} = "
              f"{2*N_FROZEN_CORE + 2*N_RESTRICTED_DOCC + N_ELEC_ACTIVE} "
              f"(should be 46)")

        if args.ts_only:
            ts_idx = int(np.argmin(np.abs(irc_s)))
            print(f"\n  --ts-only: processing frame {ts_idx} (s={irc_s[ts_idx]:+.2f})")
            run_indices = [ts_idx]
        else:
            run_indices = list(range(n_frames))

        # ── Run PSI4 for each frame ───────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  Running PSI4 on {len(run_indices)} frame(s)")
        print(f"{'='*60}")

        results = []
        e_b3lyp_min = energies.min()
        for i, fi in enumerate(run_indices):
            res = run_frame(
                symbols, coords[fi], irc_s[fi], energies[fi],
                frame_idx=fi, out_dir=out_dir,
                run_caspt2=args.run_caspt2,
                n_threads=args.n_threads, memory=args.memory)
            res['e_b3lyp_rel'] = (energies[fi] - e_b3lyp_min) * HARTREE_TO_KCAL
            results.append(res)

        # Save raw results
        results_path = out_dir / 'casscf_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Raw results saved: {results_path}")

    else:
        # Reload previous PSI4 results
        print(f"\n  Loading saved results from: {args.load_results}")
        with open(args.load_results) as f:
            results = json.load(f)
        # Need IRC data for delta-ML training coords
        data     = np.load(args.irc_data, allow_pickle=True)
        symbols  = data['symbols'].tolist()
        coords   = data['coordinates']
        energies = data['energies']
        irc_s    = data['irc_s']
        e_min    = energies.min()
        mask     = (energies - e_min) * HARTREE_TO_KCAL < E_FILTER_KCAL
        order    = np.argsort(irc_s[mask])
        coords   = coords[mask][order]
        energies = energies[mask][order]
        e_b3lyp_min = energies.min()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  {'Frame':>5}  {'s':>6}  {'ΔE_B3LYP':>10}  {'ΔE_CASSCF':>10}  "
          f"{'Δcorr':>8}  {'H₁₂':>8}  {'NO occs'}")
    print(f"  {'':>5}  {'Å√amu':>6}  {'kcal/mol':>10}  {'kcal/mol':>10}  "
          f"{'kcal/mol':>8}  {'kcal/mol':>8}")
    print(f"  {'-'*75}")
    # Use CASSCF minimum for relative CASSCF energies
    e_cas_vals = [r.get('e_sa_root0') or r.get('e_casscf_ss')
                  for r in results if (r.get('e_sa_root0') or r.get('e_casscf_ss'))]
    e_cas_min = min(e_cas_vals) if e_cas_vals else e_b3lyp_min
    for r in results:
        s_     = r.get('irc_s', float('nan'))
        dE_b3  = r.get('e_b3lyp_rel', float('nan'))
        e_hi   = r.get('e_sa_root0') or r.get('e_casscf_ss')
        dE_cas = (e_hi - e_cas_min) * HARTREE_TO_KCAL if e_hi else float('nan')
        delta  = (dE_cas - dE_b3) if not (np.isnan(dE_cas) or np.isnan(dE_b3)) else float('nan')
        H12    = r.get('H12_kcal', float('nan')) or float('nan')
        occs   = r.get('no_occs')
        occ_str = ' '.join(f'{x:.3f}' for x in occs) if occs else '—'
        print(f"  {r['frame_idx']:>5}  {s_:>+6.2f}  {dE_b3:>10.2f}  "
              f"{dE_cas:>10.2f}  {delta:>+8.2f}  {H12:>8.2f}  {occ_str}")
    print(f"{'='*75}")

    # ── Delta-ML training ─────────────────────────────────────────────────────
    # Match results to coords by frame index
    idx_map = {r['frame_idx']: i for i, r in enumerate(results)}
    coords_for_delta = []
    results_ordered  = []
    for fi in sorted(idx_map.keys()):
        res_i   = results[idx_map[fi]]
        frame_i = res_i['frame_idx']
        if frame_i < len(coords):
            coords_for_delta.append(coords[frame_i])
            results_ordered.append(res_i)

    delta_trainer, delta_pkl = train_delta_ml(
        results_ordered, symbols,
        np.array(coords_for_delta), out_dir)

    # ── Corrected PESFamily manifest ──────────────────────────────────────────
    if delta_pkl and Path(args.rxn_path_family).exists():
        save_corrected_manifest(args.rxn_path_family, str(delta_pkl), out_dir)

    # ── Diagnostic figure ─────────────────────────────────────────────────────
    plot_results(results, e_b3lyp_min, out_dir)

    # ── Final summary ─────────────────────────────────────────────────────────
    good_cas   = [r for r in results if r.get('e_casscf_ss') is not None]
    good_sa    = [r for r in results if r.get('e_sa_root0')  is not None]
    dE_b3lyp_barrier  = max(
        (r.get('e_b3lyp_rel', 0) for r in results), default=0)
    cas_vals_all = [r.get('e_sa_root0') or r.get('e_casscf_ss')
                    for r in results if (r.get('e_sa_root0') or r.get('e_casscf_ss'))]
    if cas_vals_all:
        e_cas_min_all = min(cas_vals_all)
        cas_rel = [(v - e_cas_min_all) * HARTREE_TO_KCAL for v in cas_vals_all]
        dE_cas_barrier = max(cas_rel)
    else:
        dE_cas_barrier = float('nan')
    max_H12 = max(
        (r.get('H12_kcal') or 0 for r in results), default=0)

    print(f"\n  ── Summary ──────────────────────────────────────────────────")
    print(f"  Frames computed:    {len(good_cas)}/{len(results)}")
    print(f"  SA-CASSCF success:  {len(good_sa)}/{len(results)}")
    print(f"  B3LYP barrier:      {dE_b3lyp_barrier:.1f} kcal/mol")
    print(f"  CASSCF barrier:     {dE_cas_barrier:.1f} kcal/mol" if not (dE_cas_barrier != dE_cas_barrier) else "  CASSCF barrier:     n/a (no converged frames)")
    shift = dE_cas_barrier - dE_b3lyp_barrier
    print(f"  Barrier shift:      {shift:+.1f} kcal/mol" if not (shift != shift) else "  Barrier shift:      n/a")
    print(f"  Max H₁₂:            {max_H12:.1f} kcal/mol")
    print(f"\n  Note: dynamic correlation (NEVPT2/CASPT2) not included.")
    print(f"  Install 'forte' (conda install forte -c conda-forge/label/forte)")
    print(f"  and re-run with psi4.energy('nevpt2') for further ~5–10 kcal/mol")
    print(f"  correction to the barrier.")
    print(f"\n  All outputs in: {out_dir}")

    # Save summary
    summary = {
        'timestamp':          ts_str,
        'n_frames':           len(results),
        'n_casscf_converged': len(good_cas),
        'n_sa_converged':     len(good_sa),
        'barrier_b3lyp_kcal': dE_b3lyp_barrier,
        'barrier_casscf_kcal': dE_cas_barrier,
        'barrier_shift_kcal': (dE_cas_barrier - dE_b3lyp_barrier) if dE_cas_barrier == dE_cas_barrier else None,
        'max_H12_kcal':       max_H12,
        'irc_data':           args.irc_data,
        'active_space':       f'CASSCF({N_ELEC_ACTIVE},{N_ORBS_ACTIVE})',
        'basis':              '6-31G*',
        'dynamic_corr':       'not_computed',
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
