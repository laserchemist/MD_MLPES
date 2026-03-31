#!/usr/bin/env python3
"""
casscf_surface_correction.py — CASSCF(4,4) single points on a sparse sample
of the full B3LYP MVKO training set; trains a smooth delta-ML KRR correction
that can be applied across the entire ML-PES surface.

Purpose
-------
The IRC-only CASSCF correction (casscf_irc_correction.py) covers only the
syn-MVKO → VHP reaction path. This script computes the delta correction on
a representative sample of the *equilibrium* training data, giving:

    E_corrected(R) = E_B3LYP_ML(R) + E_delta_ML(R)

where E_delta_ML is a smooth KRR model trained on
    delta(R_i) = [E_CASSCF(R_i) - E_CASSCF_ref] - [E_B3LYP(R_i) - E_B3LYP_ref]

The corrected surface is more accurate than pure B3LYP for MD at 300 K and
provides better thermal sampling near equilibrium.

Frame selection
---------------
Stratified energy sampling over the existing B3LYP training set:
  bin 0:  [0,  5) kcal/mol  — equilibrium region
  bin 1:  [5, 15) kcal/mol  — small distortions
  bin 2: [15, 30) kcal/mol  — NM thermal distortions at 1000 K
  bin 3: [30, 60) kcal/mol  — large distortions / high-T MD
  bin 4: [60,100) kcal/mol  — high-energy tail

Frames are drawn by farthest-point sampling (FPS) in Coulomb-matrix descriptor
space within each bin — maximises coverage of the descriptor landscape.

Optionally merges with IRC CASSCF results to also cover the reaction path.

Active space
------------
CASSCF(4,4)/6-31G* — same as casscf_irc_correction.py
  frozen_docc = 6  (1s: 4×C + 2×O)
  restricted_docc = 15  ((46 − 12 − 4) / 2)
  active = 4  (σ/σ*(C3-H7) + σ/σ*(O2-H7); near equilibrium captures
               the dominant single-reference correction)

Note: at the MVKO minimum (closed-shell), the 4 active orbitals will have
NO occupations ≈ (2, 2, 0, 0), so the correction is small (~0–3 kcal/mol).
At higher energies the active space captures more distorted geometries where
correlation effects grow. This is expected and physical.

Usage
-----
  # Full run (select 30 frames, stratified, merge with IRC):
  python3 casscf_surface_correction.py \\
      --training-data outputs/mvko_20260319_081314/combined_training_data.npz \\
      --n-frames 30 \\
      --irc-results outputs/casscf_irc_20260330_230145/casscf_results.json \\
      --b3lyp-model outputs/mvko_20260319_081314/mlpes_initial.pkl

  # Custom energy cap, FPS within each bin:
  python3 casscf_surface_correction.py \\
      --training-data outputs/mvko_20260319_081314/combined_training_data.npz \\
      --n-frames 50 --max-energy 80

  # Reload saved CASSCF results without re-running PSI4:
  python3 casscf_surface_correction.py \\
      --load-results outputs/casscf_surface_<ts>/surface_results.json \\
      --irc-results outputs/casscf_irc_20260330_230145/casscf_results.json

  # Quick test: 5 frames only
  python3 casscf_surface_correction.py \\
      --training-data ... --n-frames 5 --max-energy 20
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

# MVKO active space: CASSCF(4,4) — same as casscf_irc_correction.py
N_ELEC_ACTIVE     = 4
N_ORBS_ACTIVE     = 4
N_FROZEN_CORE     = 6    # 1s: 4C + 2O
N_RESTRICTED_DOCC = 15   # (46 - 12 - 4) / 2

# Energy bins for stratified sampling (kcal/mol)
ENERGY_BINS = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 100)]


# ── Geometry helpers (shared with casscf_irc_correction.py) ───────────────────

def geometry_string(symbols, coords, charge=0, mult=1):
    """Build a PSI4-ready geometry block (Angstrom, no_reorient, no_com)."""
    lines = [f"{charge} {mult}"]
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"  {sym}  {x:.10f}  {y:.10f}  {z:.10f}")
    lines += ["units angstrom", "no_reorient", "no_com", "symmetry c1"]
    return "\n".join(lines)


# ── PSI4 output parsing ───────────────────────────────────────────────────────

def _parse_no_occupations(output_text: str) -> list[float] | None:
    """Parse active-space natural orbital occupations from PSI4 1.10 CASSCF output."""
    matches = list(re.finditer(
        r'Active Space Natural occupation numbers:\s*\n\s*\n([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n|\Z)',
        output_text))
    if matches:
        block = matches[0].group(1)
        nums = re.findall(r'[A-Za-z]+\s+([\d.]+)', block)
        if nums:
            return sorted([float(x) for x in nums], reverse=True)
    return None


# ── Frame selection ───────────────────────────────────────────────────────────

def _coulomb_descriptors(symbols, coords_batch):
    """
    Compute upper-triangle Coulomb matrix descriptors for a batch of geometries.
    Same formula as modules/ml_pes.py CoulombMatrixDescriptor.
    Returns array of shape (N, n_desc).
    """
    atomic_numbers = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16}
    charges = np.array([atomic_numbers.get(s, 1) for s in symbols], dtype=float)
    n_atoms = len(symbols)
    n_desc = n_atoms * (n_atoms + 1) // 2
    descs = np.zeros((len(coords_batch), n_desc))
    for b, coords in enumerate(coords_batch):
        k = 0
        for i in range(n_atoms):
            for j in range(i, n_atoms):
                if i == j:
                    descs[b, k] = 0.5 * charges[i] ** 2.4
                else:
                    r = np.linalg.norm(coords[i] - coords[j])
                    descs[b, k] = charges[i] * charges[j] / r if r > 1e-8 else 0.0
                k += 1
    return descs


def fps_select(descs, n_select, seed_idx=0):
    """
    Farthest-point sampling in descriptor space.
    Returns indices of n_select maximally spread points.
    """
    n = len(descs)
    if n_select >= n:
        return list(range(n))
    selected = [seed_idx]
    min_dists = np.full(n, np.inf)
    for _ in range(n_select - 1):
        last = descs[selected[-1]]
        dists = np.sum((descs - last) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -np.inf   # exclude already selected
        selected.append(int(np.argmax(min_dists)))
    return selected


def select_frames(coords, energies, symbols, n_total, max_energy_kcal,
                  bins=ENERGY_BINS, rng_seed=42):
    """
    Stratified + FPS frame selection.

    Distributes n_total frames across energy bins (proportional to bin width),
    then uses FPS within each bin to maximise descriptor-space coverage.

    Returns selected indices into the (filtered) coords/energies arrays.
    """
    e_min = energies.min()
    e_rel = (energies - e_min) * HARTREE_TO_KCAL

    # Apply max-energy filter first
    mask = e_rel < max_energy_kcal
    avail_idx = np.where(mask)[0]
    if len(avail_idx) == 0:
        raise ValueError(f"No frames below {max_energy_kcal} kcal/mol")

    print(f"  Frames within {max_energy_kcal:.0f} kcal/mol: {len(avail_idx)}/{len(energies)}")

    # Compute descriptors once for available frames
    print("  Computing Coulomb descriptors for frame selection...")
    descs = _coulomb_descriptors(symbols, coords[avail_idx])

    # Bin allocation: proportional to sqrt(bin_width) to slightly favour wide bins
    bin_widths = np.array([b[1] - b[0] for b in bins], dtype=float)
    alloc = np.maximum(1, np.round(
        n_total * np.sqrt(bin_widths) / np.sqrt(bin_widths).sum()
    ).astype(int))
    # Trim to n_total
    while alloc.sum() > n_total:
        alloc[np.argmax(alloc)] -= 1
    while alloc.sum() < n_total:
        alloc[np.argmin(alloc)] += 1

    selected_global = []
    for b_idx, (lo, hi) in enumerate(bins):
        bin_mask = (e_rel[avail_idx] >= lo) & (e_rel[avail_idx] < hi)
        bin_local = np.where(bin_mask)[0]   # indices into avail_idx/descs
        n_bin = min(alloc[b_idx], len(bin_local))
        if n_bin == 0:
            print(f"  Bin [{lo:3d},{hi:3d}) kcal/mol: 0 frames available — skipping")
            continue
        if n_bin >= len(bin_local):
            chosen = bin_local
        else:
            chosen = fps_select(descs[bin_local], n_bin,
                                seed_idx=int(np.argmin(e_rel[avail_idx][bin_local])))
            chosen = bin_local[chosen]
        global_idx = avail_idx[chosen]
        selected_global.extend(global_idx.tolist())
        print(f"  Bin [{lo:3d},{hi:3d}) kcal/mol: {len(bin_local):4d} available "
              f"→ {len(chosen)} selected")

    selected_global = sorted(set(selected_global))
    print(f"  Total selected: {len(selected_global)} frames")
    return selected_global


# ── Per-frame PSI4 SS-CASSCF ─────────────────────────────────────────────────

def run_frame_ss(symbols, coords, e_b3lyp, frame_idx,
                 out_dir: Path, n_threads: int = 4, memory: str = '6 GB') -> dict:
    """
    Run RHF → SS-CASSCF(4,4) for one geometry (no SA-CASSCF needed for surface).
    Returns a result dict.
    """
    try:
        import psi4
    except ImportError:
        print("  PSI4 not available — cannot run CASSCF.")
        return {'frame_idx': frame_idx, 'e_b3lyp': float(e_b3lyp), 'error': 'no_psi4'}

    psi4.core.clean()
    try:
        psi4.core.clean_options()
    except AttributeError:
        pass

    frame_outfile = str(out_dir / f'psi4_frame{frame_idx:04d}.dat')
    psi4.core.set_output_file(frame_outfile, False)

    psi4.geometry(geometry_string(symbols, coords))
    psi4.set_memory(memory)
    psi4.set_num_threads(n_threads)

    result = {
        'frame_idx':   int(frame_idx),
        'e_b3lyp':     float(e_b3lyp),
        'e_b3lyp_rel': None,   # filled after all frames computed
        'e_rhf':       None,
        'e_casscf_ss': None,
        'no_occs':     None,
        'delta_kcal':  None,   # filled after all frames computed
        'output_file': frame_outfile,
        'error':       None,
    }

    base_opts = {
        'basis':         '6-31G*',
        'scf_type':      'df',
        'reference':     'rhf',
        'e_convergence': 1e-8,
        'd_convergence': 1e-8,
        'maxiter':       200,
    }

    # ── RHF ──────────────────────────────────────────────────────────────────
    psi4.set_options(base_opts)
    try:
        E_rhf, wfn_rhf = psi4.energy('hf', return_wfn=True)
        result['e_rhf'] = float(E_rhf)
        print(f"    RHF: {E_rhf:.8f} Ha")
    except Exception as exc:
        print(f"    RHF FAILED: {exc}")
        result['error'] = f'rhf: {exc}'
        return result

    # ── SS-CASSCF(4,4) ───────────────────────────────────────────────────────
    casscf_opts = {
        **base_opts,
        'frozen_docc':         [N_FROZEN_CORE],
        'restricted_docc':     [N_RESTRICTED_DOCC],
        'active':              [N_ORBS_ACTIVE],
        'num_roots':           1,
        'avg_states':          [0],
        'avg_weights':         [1.0],
        'mcscf_algorithm':     'ah',
        'mcscf_maxiter':       200,
        'mcscf_diis_start':    3,
        'mcscf_r_convergence': 1e-5,
        'mcscf_e_convergence': 1e-8,
    }
    psi4.set_options(casscf_opts)
    try:
        E_cas, _ = psi4.energy('casscf', return_wfn=True, ref_wfn=wfn_rhf)
        result['e_casscf_ss'] = float(E_cas)
        print(f"    CASSCF: {E_cas:.8f} Ha")

        # Parse NO occupations
        try:
            with open(frame_outfile) as f:
                out_text = f.read()
            no_occs = _parse_no_occupations(out_text)
            if no_occs:
                result['no_occs'] = no_occs[:N_ORBS_ACTIVE]
                print(f"    NO occs: {[f'{x:.4f}' for x in result['no_occs']]}")
        except Exception:
            pass

    except Exception as exc:
        print(f"    CASSCF FAILED: {exc}")
        result['error'] = f'casscf_ss: {exc}'

    return result


# ── Delta-ML training ─────────────────────────────────────────────────────────

def train_delta_ml(results_all, symbols, coords_all, out_dir: Path,
                   gamma: float = 0.00005, alpha: float = 1e-10):
    """
    Train a KRR delta-ML on the relative correction:
        delta(i) = ΔE_CASSCF(i) - ΔE_B3LYP(i)
    (both ΔE referenced to their respective method minima)

    results_all: list of result dicts with 'e_b3lyp' and 'e_casscf_ss' (Ha)
    coords_all:  matching array of geometries (Angstrom)

    Returns (trainer, delta_pkl_path) or (None, None) if too few points.
    """
    from modules.ml_pes import MLPESTrainer, MLPESConfig
    from modules.data_formats import TrajectoryData

    good = [(r, c) for r, c in zip(results_all, coords_all)
            if r.get('e_casscf_ss') is not None and r.get('e_b3lyp') is not None]

    if len(good) < 3:
        print(f"  Only {len(good)} good frames — skipping delta-ML")
        return None, None

    e_cas   = np.array([r['e_casscf_ss'] for r, _ in good])
    e_b3lyp = np.array([r['e_b3lyp']     for r, _ in good])
    coords  = np.array([c                for _, c in good])

    e_cas_ref   = e_cas.min()
    e_b3lyp_ref = e_b3lyp.min()

    dE_cas   = e_cas   - e_cas_ref     # Ha
    dE_b3lyp = e_b3lyp - e_b3lyp_ref  # Ha
    delta_rel = dE_cas - dE_b3lyp      # Ha relative correction

    print(f"\n  Delta-ML training: {len(good)} frames")
    print(f"  CASSCF barrier span: {dE_cas.max()*HARTREE_TO_KCAL:.1f} kcal/mol")
    print(f"  B3LYP  barrier span: {dE_b3lyp.max()*HARTREE_TO_KCAL:.1f} kcal/mol")
    print(f"  delta_rel range: {delta_rel.min()*HARTREE_TO_KCAL:.2f} to "
          f"{delta_rel.max()*HARTREE_TO_KCAL:.2f} kcal/mol")

    traj = TrajectoryData(
        symbols=symbols,
        coordinates=coords,
        energies=delta_rel,
        forces=np.zeros_like(coords),
        dipoles=None,
        metadata={'source': 'casscf_surface_delta', 'n_frames': len(good)},
    )

    config = MLPESConfig()
    config.gamma                = gamma
    config.alpha                = alpha
    config.tune_hyperparameters = False

    trainer = MLPESTrainer(config)
    trainer.train(traj)

    delta_pkl = out_dir / 'delta_ml_surface.pkl'
    trainer.save(str(delta_pkl))

    np.savez(str(out_dir / 'delta_ml_surface_training.npz'),
             symbols=np.array(symbols),
             coordinates=coords,
             delta_energies_ha=delta_rel,
             e_b3lyp_ha=e_b3lyp,
             e_casscf_ha=e_cas,
             e_b3lyp_ref_ha=np.array([e_b3lyp_ref]),
             e_cas_ref_ha=np.array([e_cas_ref]))

    print(f"  Delta-ML model: {delta_pkl}")
    return trainer, delta_pkl


# ── Diagnostic figure ─────────────────────────────────────────────────────────

def plot_results(results, coords_all, symbols, out_dir: Path):
    """3-panel figure: ΔE_B3LYP vs ΔE_CASSCF, delta correction, NO occupations."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from modules.ml_pes import CoulombMatrixDescriptor

        good = [(r, c) for r, c in zip(results, coords_all)
                if r.get('e_casscf_ss') is not None]
        if not good:
            return

        e_b3lyp  = np.array([r['e_b3lyp']     for r, _ in good])
        e_cas    = np.array([r['e_casscf_ss']  for r, _ in good])
        e_b3_ref = e_b3lyp.min()
        e_cas_ref = e_cas.min()
        dE_b3    = (e_b3lyp - e_b3_ref) * HARTREE_TO_KCAL
        dE_cas   = (e_cas   - e_cas_ref) * HARTREE_TO_KCAL
        delta    = dE_cas - dE_b3

        # Sort by B3LYP energy for cleaner plots
        order = np.argsort(dE_b3)
        dE_b3  = dE_b3[order]
        dE_cas = dE_cas[order]
        delta  = delta[order]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                                 gridspec_kw={'wspace': 0.35})
        ax1, ax2, ax3 = axes

        # Panel 1: energy correlation scatter
        ax1.scatter(dE_b3, dE_cas, c=dE_b3, cmap='viridis', s=60, zorder=3)
        lim = max(dE_b3.max(), dE_cas.max()) * 1.05
        ax1.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.4, label='y=x')
        ax1.set_xlabel('ΔE B3LYP (kcal/mol)', fontsize=11)
        ax1.set_ylabel('ΔE CASSCF(4,4) (kcal/mol)', fontsize=11)
        ax1.set_title('Energy correlation (each vs own ref)', fontsize=11)
        ax1.legend(fontsize=9)

        # Panel 2: delta correction vs B3LYP energy
        sc = ax2.scatter(dE_b3, delta, c=dE_b3, cmap='plasma', s=60, zorder=3)
        ax2.axhline(0, color='gray', lw=0.8, ls='--')
        ax2.set_xlabel('ΔE B3LYP (kcal/mol)', fontsize=11)
        ax2.set_ylabel('Δ correction (kcal/mol)', fontsize=11)
        ax2.set_title('Delta correction = ΔE_CASSCF − ΔE_B3LYP', fontsize=11)
        plt.colorbar(sc, ax=ax2, label='ΔE B3LYP (kcal/mol)')

        # Panel 3: NO occupations (innermost two orbitals = biradical indicators)
        occ_data = [(r['no_occs'], dE_b3[i])
                    for i, (r, _) in enumerate([(good[j]) for j in order])
                    if r.get('no_occs')]
        if occ_data:
            occs_arr = np.array([o for o, _ in occ_data])
            ens_arr  = np.array([e for _, e in occ_data])
            for col, lbl in enumerate(['NO 1', 'NO 2', 'NO 3', 'NO 4']):
                if col < occs_arr.shape[1]:
                    ax3.plot(ens_arr, occs_arr[:, col], 'o-',
                             ms=5, lw=1.2, label=lbl)
            ax3.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
            ax3.set_xlabel('ΔE B3LYP (kcal/mol)', fontsize=11)
            ax3.set_ylabel('Occupation number', fontsize=11)
            ax3.set_title('Active-space NO occupations', fontsize=11)
            ax3.legend(fontsize=8)
            ax3.set_ylim(-0.05, 2.15)

        fig.suptitle(f'CASSCF(4,4)/6-31G* surface correction — {len(good)} frames',
                     fontsize=12, fontweight='bold')
        fig_path = out_dir / 'casscf_surface_analysis.png'
        fig.savefig(str(fig_path), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Figure: {fig_path}")
    except Exception as exc:
        print(f"  Figure failed: {exc}")
        traceback.print_exc()


# ── Manifest for corrected ML-PES ─────────────────────────────────────────────

def save_corrected_manifest(b3lyp_model_path: str, delta_pkl: str,
                            out_dir: Path, n_frames: int,
                            e_b3lyp_ref: float, e_cas_ref: float) -> Path:
    """
    Write a JSON manifest for the delta-corrected ML-PES.

    Downstream usage:
        E_corrected(R) = E_B3LYP_ML(R) + delta_ML(R)
    where delta_ML is trained on relative corrections (each method vs its own ref).

    The absolute offset between CASSCF and B3LYP is stored so it can be applied
    if needed, but for MD (forces only), it cancels.
    """
    manifest = {
        'b3lyp_model':        b3lyp_model_path,
        'delta_ml_model':     str(delta_pkl),
        'active_space':       f'CASSCF({N_ELEC_ACTIVE},{N_ORBS_ACTIVE})',
        'basis':              '6-31G*',
        'n_frames':           n_frames,
        'e_b3lyp_ref_ha':     e_b3lyp_ref,
        'e_cas_ref_ha':       e_cas_ref,
        'abs_offset_ha':      e_cas_ref - e_b3lyp_ref,  # informational only
        'abs_offset_kcal':    (e_cas_ref - e_b3lyp_ref) * HARTREE_TO_KCAL,
        'note': (
            'Apply as E_corr(R) = E_B3LYP_ML(R) + delta_ML(R). '
            'The delta_ML model predicts the relative correction '
            'ΔE_CASSCF(R) - ΔE_B3LYP(R) in Hartree. '
            'Absolute energy offset (CASSCF vs B3LYP) does not affect forces.'
        ),
    }
    path = out_dir / 'corrected_surface_manifest.json'
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Corrected manifest: {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--training-data', default=None,
                    help='B3LYP training set .npz (combined_training_data.npz)')
    ap.add_argument('--n-frames', type=int, default=30,
                    help='Total frames to compute CASSCF on [default 30]')
    ap.add_argument('--max-energy', type=float, default=100.0,
                    help='Max B3LYP energy (kcal/mol above minimum) to include [default 100]')
    ap.add_argument('--irc-results', default=None,
                    help='Path to casscf_irc_correction.py results JSON — merged into delta-ML')
    ap.add_argument('--b3lyp-model', default=None,
                    help='B3LYP ML-PES model .pkl — written into corrected manifest')
    ap.add_argument('--load-results', default=None,
                    help='Reload saved surface_results.json — skip PSI4')
    ap.add_argument('--gamma', type=float, default=0.00005,
                    help='KRR gamma for delta-ML [default 0.00005]')
    ap.add_argument('--alpha', type=float, default=1e-10,
                    help='KRR alpha for delta-ML [default 1e-10]')
    ap.add_argument('--n-threads', type=int, default=4,
                    help='PSI4 threads per calculation [default 4]')
    ap.add_argument('--memory', default='6 GB',
                    help='PSI4 memory [default "6 GB"]')
    args = ap.parse_args()

    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(f'outputs/casscf_surface_{ts_str}')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Output directory: {out_dir}")

    # ── Load training data ────────────────────────────────────────────────────
    if args.load_results:
        print(f"\n  Reloading saved results: {args.load_results}")
        with open(args.load_results) as f:
            results = json.load(f)
        # Re-read geometry from the training data
        if args.training_data is None:
            print("  WARNING: --training-data not given; coords unavailable for re-plot")
            coords_selected = None
            symbols = None
        else:
            d = np.load(args.training_data, allow_pickle=True)
            symbols = list(d['symbols'])
            all_coords   = d['coordinates']
            all_energies = d['energies']
            e_min = all_energies.min()
            e_rel = (all_energies - e_min) * HARTREE_TO_KCAL
            # Recover selected frame indices from results
            sel_idx = [r['frame_idx'] for r in results]
            coords_selected = all_coords[sel_idx]
    else:
        if args.training_data is None:
            ap.error("--training-data required unless --load-results is given")

        d = np.load(args.training_data, allow_pickle=True)
        symbols      = list(d['symbols'])
        all_coords   = d['coordinates']
        all_energies = d['energies']

        print(f"\n  Training data: {args.training_data}")
        print(f"  Frames: {len(all_energies)}")
        e_min = all_energies.min()
        e_rel = (all_energies - e_min) * HARTREE_TO_KCAL
        print(f"  B3LYP energy range: {e_rel.min():.1f} – {e_rel.max():.1f} kcal/mol")

        # ── Frame selection ───────────────────────────────────────────────────
        print(f"\n  Selecting {args.n_frames} frames (max ΔE = {args.max_energy} kcal/mol):")
        sel_idx = select_frames(all_coords, all_energies, symbols,
                                n_total=args.n_frames,
                                max_energy_kcal=args.max_energy)

        coords_selected   = all_coords[sel_idx]
        energies_selected = all_energies[sel_idx]

        print(f"\n  Selected frames (sorted by energy):")
        order = np.argsort(energies_selected)
        e_sel_min = energies_selected.min()
        for rank, oi in enumerate(order):
            dE = (energies_selected[oi] - e_sel_min) * HARTREE_TO_KCAL
            print(f"    [{rank+1:2d}] global idx {sel_idx[oi]:4d}  "
                  f"ΔE = {dE:7.2f} kcal/mol")

        # ── PSI4 CASSCF single points ─────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  Running CASSCF(4,4) on {len(sel_idx)} frames")
        print(f"{'='*60}")
        results = []
        for rank, fi in enumerate(sel_idx):
            dE_b3 = (all_energies[fi] - e_min) * HARTREE_TO_KCAL
            print(f"\n  [{rank+1}/{len(sel_idx)}] Frame {fi}  "
                  f"ΔE_B3LYP = {dE_b3:.1f} kcal/mol")
            res = run_frame_ss(
                symbols, all_coords[fi], all_energies[fi], frame_idx=fi,
                out_dir=out_dir, n_threads=args.n_threads, memory=args.memory)
            results.append(res)
            # Save incrementally after each frame
            with open(out_dir / 'surface_results.json', 'w') as f:
                json.dump(results, f, indent=2)

    # ── Fill in relative energies ─────────────────────────────────────────────
    good = [r for r in results if r.get('e_casscf_ss') is not None]
    if good:
        e_b3lyp_vals = np.array([r['e_b3lyp']     for r in good])
        e_cas_vals   = np.array([r['e_casscf_ss']  for r in good])
        e_b3_ref  = e_b3lyp_vals.min()
        e_cas_ref = e_cas_vals.min()
        for r in results:
            r['e_b3lyp_rel'] = (r['e_b3lyp'] - e_b3_ref) * HARTREE_TO_KCAL
            if r.get('e_casscf_ss') is not None:
                dE_cas   = (r['e_casscf_ss'] - e_cas_ref) * HARTREE_TO_KCAL
                dE_b3lyp = r['e_b3lyp_rel']
                r['delta_kcal'] = dE_cas - dE_b3lyp
    else:
        e_b3_ref = e_cas_ref = 0.0

    # Save final results
    with open(out_dir / 'surface_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # ── Merge with IRC results (optional) ────────────────────────────────────
    merged_results = list(results)
    merged_coords  = list(coords_selected) if coords_selected is not None else []

    if args.irc_results:
        print(f"\n  Merging IRC CASSCF results: {args.irc_results}")
        with open(args.irc_results) as f:
            irc_res = json.load(f)
        # Load IRC coordinates from delta_ml_training.npz (casscf_irc_correction output)
        irc_dir = Path(args.irc_results).parent
        candidates = (list(irc_dir.glob('delta_ml_training.npz')) +
                      list(irc_dir.rglob('irc_training_data.npz')))
        if candidates:
            irc_d = np.load(str(candidates[0]), allow_pickle=True)
            irc_coords = irc_d['coordinates']   # (N, n_atoms, 3)
            # IRC results have frame_idx 0..N-1, matching coordinate order directly
            irc_good = [r for r in irc_res if r.get('e_casscf_ss') is not None]
            for r in irc_good:
                fi = r['frame_idx']
                if fi < len(irc_coords):
                    merged_results.append(r)
                    merged_coords.append(irc_coords[fi])
            print(f"  Added {len(irc_good)} IRC frames → total {len(merged_results)}")
        else:
            print("  Could not locate IRC training data — IRC frames not merged")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {'Frame':>5}  {'ΔE_B3LYP':>10}  {'ΔE_CASSCF':>10}  {'Δcorr':>8}  "
          f"{'NO occs'}")
    print(f"  {'':>5}  {'kcal/mol':>10}  {'kcal/mol':>10}  {'kcal/mol':>8}")
    print(f"  {'-'*70}")
    for r in sorted(results, key=lambda x: x.get('e_b3lyp_rel', 0)):
        dE_b3 = r.get('e_b3lyp_rel', float('nan'))
        e_hi  = r.get('e_casscf_ss')
        dE_cas = ((e_hi - e_cas_ref) * HARTREE_TO_KCAL
                  if e_hi is not None else float('nan'))
        delta  = r.get('delta_kcal', float('nan')) or float('nan')
        occs   = r.get('no_occs') or []
        occ_str = ' '.join(f'{x:.3f}' for x in occs) if occs else '—'
        print(f"  {r['frame_idx']:>5}  {dE_b3:>10.2f}  {dE_cas:>10.2f}  "
              f"{delta:>+8.2f}  {occ_str}")
    print(f"{'='*70}")
    n_conv = sum(1 for r in results if r.get('e_casscf_ss') is not None)
    print(f"\n  Converged: {n_conv}/{len(results)}")

    # ── Delta-ML training (surface frames only, then merged) ─────────────────
    if coords_selected is not None and len(coords_selected) > 0:
        coords_arr = np.array(list(coords_selected))
    else:
        coords_arr = np.array([])

    if len(coords_arr) > 0:
        print("\n  Training delta-ML on surface frames:")
        trainer_surf, delta_pkl_surf = train_delta_ml(
            results, symbols, coords_arr, out_dir,
            gamma=args.gamma, alpha=args.alpha)
    else:
        trainer_surf = delta_pkl_surf = None

    if merged_coords and len(merged_coords) > len(coords_arr):
        print("\n  Training combined delta-ML (surface + IRC frames):")
        merged_dir = out_dir / 'merged'
        merged_dir.mkdir(exist_ok=True)
        trainer_merged, delta_pkl_merged = train_delta_ml(
            merged_results, symbols, np.array(merged_coords),
            merged_dir, gamma=args.gamma, alpha=args.alpha)
    else:
        trainer_merged = delta_pkl_merged = None

    # ── Save manifest ─────────────────────────────────────────────────────────
    if delta_pkl_surf and args.b3lyp_model:
        save_corrected_manifest(
            args.b3lyp_model, str(delta_pkl_surf), out_dir,
            n_frames=n_conv,
            e_b3lyp_ref=float(e_b3_ref),
            e_cas_ref=float(e_cas_ref))

    # ── Diagnostic figure ─────────────────────────────────────────────────────
    if coords_selected is not None and len(coords_selected) > 0:
        plot_results(results, list(coords_selected), symbols, out_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    if good:
        dE_b3_arr  = np.array([r['e_b3lyp_rel']  for r in good])
        dE_cas_arr = np.array([(r['e_casscf_ss'] - e_cas_ref)*HARTREE_TO_KCAL
                                for r in good])
        delta_arr  = dE_cas_arr - dE_b3_arr
        print(f"\n  ── Summary ──────────────────────────────────────────")
        print(f"  Frames:         {n_conv}/{len(results)} CASSCF converged")
        print(f"  B3LYP range:    {dE_b3_arr.min():.1f} – {dE_b3_arr.max():.1f} kcal/mol")
        print(f"  CASSCF range:   {dE_cas_arr.min():.1f} – {dE_cas_arr.max():.1f} kcal/mol")
        print(f"  Delta range:    {delta_arr.min():+.2f} to {delta_arr.max():+.2f} kcal/mol")
        print(f"  Mean |delta|:   {np.abs(delta_arr).mean():.2f} kcal/mol")
        print(f"  Max |delta|:    {np.abs(delta_arr).max():.2f} kcal/mol")

    if delta_pkl_surf:
        print(f"\n  Surface delta-ML: {delta_pkl_surf}")
    if delta_pkl_merged:
        print(f"  Merged  delta-ML: {delta_pkl_merged}")
    print(f"\n  All outputs in: {out_dir}")

    # Save run summary JSON
    summary = {
        'timestamp': ts_str,
        'training_data': args.training_data,
        'n_selected': len(results),
        'n_converged': n_conv,
        'max_energy_kcal': args.max_energy,
        'gamma': args.gamma,
        'alpha': args.alpha,
        'active_space': f'CASSCF({N_ELEC_ACTIVE},{N_ORBS_ACTIVE})',
        'basis': '6-31G*',
        'irc_results_merged': args.irc_results,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
