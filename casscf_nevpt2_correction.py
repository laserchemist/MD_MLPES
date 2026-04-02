"""
casscf_nevpt2_correction.py

Two-layer delta-ML correction: B3LYP/6-31G* → CASSCF(4,4)/6-31G* → SC-NEVPT2/6-31G*

Builds on the existing PSI4 CASSCF NM-displacement dataset
(outputs/casscf_nm_delta_<ts>/all_casscf_results.json) by adding PySCF
SC-NEVPT2 on top of every geometry.  Produces a single NMKRRDeltaModel
predicting δ_total = E_NEVPT2(PySCF) − E_B3LYP(PSI4) in NM-coordinate
space.  The model is drop-in compatible with the --nm-delta-model flag in
ir_md_spectrum.py.

Usage
-----
# Compute NEVPT2 corrections and train model (full run, ~80 min for 59 frames):
python3 casscf_nevpt2_correction.py \
    --casscf-dir outputs/casscf_nm_delta_20260401_110049 \
    --training-data outputs/mvko_20260319_081314/combined_training_data.npz

# Resume from a previous partial run (skip already-done frames):
python3 casscf_nevpt2_correction.py \
    --casscf-dir outputs/casscf_nm_delta_20260401_110049 \
    --training-data outputs/mvko_20260319_081314/combined_training_data.npz \
    --resume outputs/nevpt2_correction_<ts>

# Skip PySCF, reload saved results and just retrain KRR:
python3 casscf_nevpt2_correction.py \
    --casscf-dir outputs/casscf_nm_delta_20260401_110049 \
    --training-data outputs/mvko_20260319_081314/combined_training_data.npz \
    --resume outputs/nevpt2_correction_<ts> \
    --retrain-only

Outputs (in outputs/nevpt2_correction_<ts>/)
--------------------------------------------
nevpt2_correction_model.pkl     — NMKRRDeltaModel  (use with --nm-delta-model)
nevpt2_results.json             — per-frame NEVPT2 results + both deltas
summary.json                    — hyperparameters, CV RMSE, energy ranges
diagnostics.png                 — parity plot and delta decomposition
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────
HARTREE_TO_KCAL = 627.509474
ANGSTROM_TO_BOHR = 1.8897259886

# MVKOO atom ordering (from CLAUDE.md — must be consistent with Coulomb descriptor)
MVKOO_SYMBOLS = ['C', 'O', 'O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']


# ─────────────────────────────────────────────────────────────────────────────
# Geometry I/O
# ─────────────────────────────────────────────────────────────────────────────

def parse_psi4_geometry(filename):
    """
    Extract the first Cartesian geometry block (Angstrom) from a PSI4 output file.

    PSI4 format:
        Geometry (in Angstrom), charge = 0, multiplicity = 1:

           Center              X          Y          Z        Mass
        ------------   --------  --------  --------  --------
             C            x.xxx    y.yyy    z.zzz    m.mmm
             ...

    Returns np.ndarray shape (N_atoms, 3) or None if not found.
    """
    try:
        txt = Path(filename).read_text()
    except OSError:
        return None

    # Find the dashes separator line (unique to the geometry header block),
    # then capture all atom coordinate lines that immediately follow.
    # PSI4 layout:
    #   Geometry (in Angstrom), charge = ...\n
    #   \n
    #   Center   X   Y   Z   Mass\n
    #   -----   -----  -----  -----  -----\n   ← anchor on this
    #   ATOM   x   y   z   m\n
    #   ...
    pattern = (
        r'----+\s+----+\s+----+\s+----+\s+----+\s*\n'   # dashes separator
        r'((?:\s+[A-Za-z]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+[^\n]*\n)+)'
    )
    m = re.search(pattern, txt)
    if m is None:
        return None

    coords = []
    for line in m.group(1).strip().split('\n'):
        parts = line.split()
        # atom_symbol  X  Y  Z  [mass]  — need at least 4 tokens
        if len(parts) >= 4:
            try:
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
    return np.array(coords) if coords else None


def load_frame_geometries(casscf_results, training_coords, eq_coords):
    """
    Reconstruct a Cartesian coordinate array for every frame in casscf_results.

    Frames with frame_idx < len(training_coords): taken from training_coords.
    Frames with frame_idx >= len(training_coords): parsed from PSI4 output file.

    Returns
    -------
    coords_list : list of (N_atoms, 3) arrays or None (if parse failed)
    """
    n_train = len(training_coords)
    coords_list = []
    for r in casscf_results:
        idx = r['frame_idx']
        if idx < n_train:
            coords_list.append(training_coords[idx])
        else:
            coords = parse_psi4_geometry(r.get('output_file', ''))
            coords_list.append(coords)
    return coords_list


# ─────────────────────────────────────────────────────────────────────────────
# NM coordinate projection  (mirrors NMKRRDeltaModel.project)
# ─────────────────────────────────────────────────────────────────────────────

def project_to_nm(coords_ang, eq_coords_ang, U_vib, sqrt_mass):
    """
    Project one Cartesian geometry into normal-mode coordinates.

    q = U_vib^T · M^{1/2} · (R − R_ref)   [sqrt(amu)·Bohr]

    Parameters
    ----------
    coords_ang    : (N, 3) Angstrom
    eq_coords_ang : (N, 3) Angstrom reference
    U_vib         : (3N, n_vib) mass-weighted eigenvectors
    sqrt_mass     : (3N,) sqrt(amu) per Cartesian DOF

    Returns
    -------
    q : (n_vib,) array
    """
    delta_ang  = np.asarray(coords_ang) - eq_coords_ang
    delta_bohr = delta_ang.flatten() * ANGSTROM_TO_BOHR
    delta_mw   = delta_bohr * sqrt_mass
    return U_vib.T @ delta_mw


def project_batch(coords_list, eq_coords_ang, U_vib, sqrt_mass):
    """Vectorised projection for a list of geometries."""
    qs = []
    for c in coords_list:
        qs.append(project_to_nm(c, eq_coords_ang, U_vib, sqrt_mass))
    return np.array(qs)   # (M, n_vib)


# ─────────────────────────────────────────────────────────────────────────────
# KRR training
# ─────────────────────────────────────────────────────────────────────────────

def rbf_kernel(A, B, gamma):
    """RBF kernel matrix. A:(m,d), B:(n,d) → (m,n)."""
    A2 = np.sum(A ** 2, axis=1, keepdims=True)
    B2 = np.sum(B ** 2, axis=1, keepdims=True)
    return np.exp(-gamma * (A2 + B2.T - 2.0 * A @ B.T))


def loo_cv_rmse(X, y, gamma, alpha_reg):
    """Leave-one-out cross-validation RMSE for KRR in kcal/mol."""
    K = rbf_kernel(X, X, gamma)
    K[np.diag_indices_from(K)] += alpha_reg
    alpha_vec = np.linalg.solve(K, y)
    # LOO residual via hat matrix diagonal
    K_inv = np.linalg.inv(K)
    hat_diag = 1.0 / np.diag(K_inv)
    residuals = alpha_vec / np.diag(K_inv)      # y_i - y_hat_i(LOO)
    return float(np.sqrt(np.mean(residuals ** 2))) * HARTREE_TO_KCAL


def grid_search_krr(X, y, gammas, alphas, verbose=True):
    """
    Grid search over gamma and alpha using LOO-CV RMSE.

    Returns best_gamma, best_alpha, best_rmse_kcal, full results dict.
    """
    best = {'gamma': None, 'alpha': None, 'rmse': np.inf}
    grid = {}
    for g in gammas:
        for a in alphas:
            try:
                rmse = loo_cv_rmse(X, y, g, a)
            except np.linalg.LinAlgError:
                rmse = np.inf
            grid[(g, a)] = rmse
            if verbose:
                print(f"    γ={g:.1e}  α={a:.1e}  LOO-CV RMSE = {rmse:.3f} kcal/mol")
            if rmse < best['rmse']:
                best = {'gamma': g, 'alpha': a, 'rmse': rmse}
    return best['gamma'], best['alpha'], best['rmse'], grid


def train_krr(X, y, gamma, alpha_reg):
    """Solve KRR dual problem. Returns alpha_vec (M,)."""
    K = rbf_kernel(X, X, gamma)
    K[np.diag_indices_from(K)] += alpha_reg
    return np.linalg.solve(K, y)


# NEVPTKRRModel lives in casscf_nm_delta so it is always importable as a
# proper module (not __main__), ensuring pickle round-trips work correctly.
from casscf_nm_delta import NEVPTKRRModel  # noqa: F401 (re-exported)


# ─────────────────────────────────────────────────────────────────────────────
# Main workflow
# ─────────────────────────────────────────────────────────────────────────────

def run_nevpt2_batch(frames, symbols, n_active_orb, n_active_elec,
                     checkpoint_dir, resume=False):
    """
    Run PySCF CASSCF+NEVPT2 on each frame, with per-frame JSON checkpointing.

    Parameters
    ----------
    frames         : list of dicts with keys: frame_idx, e_b3lyp, coords
    symbols        : list of str
    checkpoint_dir : Path  — per-frame results saved as frame_XXXX.json
    resume         : bool  — skip frames with existing checkpoint files

    Returns
    -------
    list of result dicts (one per frame, in order)
    """
    from modules.nevpt2_pyscf import compute_casscf_nevpt2

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for i, fr in enumerate(frames):
        ckpt_path = checkpoint_dir / f"frame_{fr['frame_idx']:05d}.json"

        if resume and ckpt_path.exists():
            with open(ckpt_path) as f:
                res = json.load(f)
            print(f"  [{i+1:3d}/{len(frames)}] frame {fr['frame_idx']:5d}  "
                  f"RESUMED (δ_NEVPT2={res.get('delta_nevpt2_kcal','?'):.2f} kcal/mol)")
            all_results.append(res)
            continue

        coords = fr['coords']
        if coords is None:
            print(f"  [{i+1:3d}/{len(frames)}] frame {fr['frame_idx']:5d}  "
                  f"SKIPPED (geometry unavailable)")
            all_results.append({'frame_idx': fr['frame_idx'],
                                 'error': 'geometry_unavailable',
                                 'converged': False})
            continue

        t0 = time.time()
        r = compute_casscf_nevpt2(
            symbols, coords,
            basis='6-31g*',
            n_active_orb=n_active_orb,
            n_active_elec=n_active_elec,
            active_space_type='auto',
            frozen_core=True,
            verbose=0,
        )
        dt = time.time() - t0

        res = {
            'frame_idx':          fr['frame_idx'],
            'e_b3lyp_ha':         fr['e_b3lyp'],
            'e_b3lyp_rel_kcal':   fr.get('e_b3lyp_rel') or 0.0,
            'e_casscf_psi4_ha':   fr.get('e_casscf_ss'),          # PSI4 reference
            'e_casscf_pyscf_ha':  r['e_casscf'],
            'e_nevpt2_ha':        r['e_nevpt2'],
            'e_nevpt2_corr_ha':   r['e_nevpt2_corr'],
            'delta_casscf_ha':    (r['e_casscf'] - fr['e_b3lyp']
                                   if r['e_casscf'] is not None else None),
            'delta_nevpt2_ha':    r['delta_nevpt2'],
            'delta_total_ha':     (r['e_nevpt2'] - fr['e_b3lyp']
                                   if r['e_nevpt2'] is not None else None),
            'delta_casscf_kcal':  ((r['e_casscf'] - fr['e_b3lyp']) * HARTREE_TO_KCAL
                                   if r['e_casscf'] is not None else None),
            'delta_nevpt2_kcal':  (r['delta_nevpt2'] * HARTREE_TO_KCAL
                                   if r['delta_nevpt2'] is not None else None),
            'delta_total_kcal':   ((r['e_nevpt2'] - fr['e_b3lyp']) * HARTREE_TO_KCAL
                                   if r['e_nevpt2'] is not None else None),
            'no_occ':             (r['no_occ'].tolist()
                                   if r['no_occ'] is not None else None),
            'converged':          r['converged'],
            'error':              r['error'],
            'wall_time_s':        round(dt, 1),
        }

        # PSI4 vs PySCF CASSCF cross-check (kcal/mol)
        if res['e_casscf_psi4_ha'] and res['e_casscf_pyscf_ha']:
            res['casscf_psi4_pyscf_diff_kcal'] = (
                (res['e_casscf_pyscf_ha'] - res['e_casscf_psi4_ha']) * HARTREE_TO_KCAL
            )

        status = "OK" if r['converged'] else f"FAILED: {r['error']}"
        print(f"  [{i+1:3d}/{len(frames)}] frame {fr['frame_idx']:5d}  "
              f"δ_cas={res.get('delta_casscf_kcal',0):.2f}  "
              f"δ_dyn={res.get('delta_nevpt2_kcal',0):.2f}  "
              f"δ_tot={res.get('delta_total_kcal',0):.2f} kcal/mol  "
              f"{dt:.0f}s  {status}")

        with open(ckpt_path, 'w') as f:
            json.dump(res, f, indent=2)

        all_results.append(res)

    return all_results


def make_diagnostics(results, X_q, y_total, y_cas, y_dyn,
                     model, output_dir):
    """
    Generate a four-panel diagnostic figure:
      1. δ_CASSCF vs B3LYP relative energy (near-eq vs emission regions)
      2. δ_NEVPT2 vs B3LYP relative energy
      3. KRR parity plot: predicted vs actual δ_total
      4. NM coordinate coverage (‖q‖ histogram by energy tier)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [diagnostics] matplotlib not available — skipping")
        return

    b3lyp_rel = np.array([r.get('e_b3lyp_rel_kcal', 0.0) for r in results])
    converged  = np.array([r['converged'] for r in results])
    q_norms    = np.linalg.norm(X_q, axis=1)

    # KRR training predictions
    K = model._kernel(X_q, X_q)
    y_pred = K @ model._alpha_vec

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('NEVPT2 Delta-ML Correction Diagnostics', fontsize=13)

    # Panel 1: δ_CASSCF vs ΔE
    ax = axes[0, 0]
    sc = ax.scatter(b3lyp_rel[converged], y_cas[converged] * HARTREE_TO_KCAL,
                    c=q_norms[converged], cmap='viridis', s=40, alpha=0.8)
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    ax.axvline(15, color='red', lw=0.8, ls=':', label='15 kcal/mol cutoff')
    ax.axvline(55, color='orange', lw=0.8, ls=':', label='ozonolysis energy')
    plt.colorbar(sc, ax=ax, label='‖q‖ (√amu·Bohr)')
    ax.set_xlabel('ΔE B3LYP (kcal/mol)')
    ax.set_ylabel('δ_CASSCF (kcal/mol)')
    ax.set_title('CASSCF correction (static correlation)')
    ax.legend(fontsize=8)

    # Panel 2: δ_NEVPT2 vs ΔE
    ax = axes[0, 1]
    ax.scatter(b3lyp_rel[converged], y_dyn[converged] * HARTREE_TO_KCAL,
               c=q_norms[converged], cmap='plasma', s=40, alpha=0.8)
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    ax.axvline(15, color='red', lw=0.8, ls=':', label='15 kcal/mol cutoff')
    ax.axvline(55, color='orange', lw=0.8, ls=':')
    ax.set_xlabel('ΔE B3LYP (kcal/mol)')
    ax.set_ylabel('δ_NEVPT2 (kcal/mol)')
    ax.set_title('NEVPT2 correction (dynamic correlation)')

    # Panel 3: parity plot
    ax = axes[1, 0]
    y_tot_kcal  = y_total * HARTREE_TO_KCAL
    y_pred_kcal = y_pred * HARTREE_TO_KCAL
    ax.scatter(y_tot_kcal, y_pred_kcal, c=b3lyp_rel[converged],
               cmap='coolwarm', s=40, alpha=0.8)
    lo = min(y_tot_kcal.min(), y_pred_kcal.min())
    hi = max(y_tot_kcal.max(), y_pred_kcal.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
    rmse = float(np.sqrt(np.mean((y_tot_kcal - y_pred_kcal) ** 2)))
    ax.set_xlabel('δ_total actual (kcal/mol)')
    ax.set_ylabel('δ_total KRR predicted (kcal/mol)')
    ax.set_title(f'KRR parity (train RMSE = {rmse:.2f} kcal/mol)')
    ax.text(0.05, 0.92, f'CV RMSE = {model.cv_rmse_kcal:.2f} kcal/mol',
            transform=ax.transAxes, fontsize=9)

    # Panel 4: ‖q‖ by energy tier
    ax = axes[1, 1]
    tiers = {
        'Absorption\n(ΔE<15)': b3lyp_rel < 15,
        'Emission\n(15-55)':   (b3lyp_rel >= 15) & (b3lyp_rel < 55),
        'High\n(ΔE>55)':       b3lyp_rel >= 55,
    }
    data_tiers  = [q_norms[converged & mask] for mask in tiers.values()]
    labels_tier = list(tiers.keys())
    ax.boxplot(data_tiers, labels=labels_tier, patch_artist=True,
               boxprops=dict(facecolor='lightblue'))
    ax.set_ylabel('‖q‖ (√amu·Bohr)')
    ax.set_title('NM displacement by energy region')
    counts = [mask.sum() for mask in tiers.values()]
    for j, (lbl, cnt) in enumerate(zip(labels_tier, counts)):
        ax.text(j + 1, ax.get_ylim()[1] * 0.98, f'n={cnt}',
                ha='center', va='top', fontsize=8)

    plt.tight_layout()
    out = Path(output_dir) / 'diagnostics.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Diagnostics saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute CASSCF+NEVPT2 corrections and train NM-KRR delta model')
    parser.add_argument('--casscf-dir', required=True,
                        help='Path to outputs/casscf_nm_delta_<ts>/ directory')
    parser.add_argument('--training-data', required=True,
                        help='Path to combined_training_data.npz (for frame geometries)')
    parser.add_argument('--n-active-orb', type=int, default=4)
    parser.add_argument('--n-active-elec', type=int, default=4)
    parser.add_argument('--filter-max-energy', type=float, default=60.0,
                        help='Discard frames with ΔE_B3LYP above this (kcal/mol)')
    parser.add_argument('--gamma-values', default='0.001,0.003,0.01,0.03,0.1',
                        help='Comma-separated gamma grid for KRR')
    parser.add_argument('--alpha-values', default='1e-5,1e-4,1e-3,0.01',
                        help='Comma-separated alpha grid for KRR')
    parser.add_argument('--resume', default=None,
                        help='Path to existing output dir to resume from')
    parser.add_argument('--retrain-only', action='store_true',
                        help='Skip PySCF, reload checkpoint JSONs and retrain KRR')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: outputs/nevpt2_correction_<ts>)')
    args = parser.parse_args()

    # ── output directory ──────────────────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.resume:
        out_dir = Path(args.resume)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = Path(f'outputs/nevpt2_correction_{ts}')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== NEVPT2 Correction Training ===")
    print(f"Output dir: {out_dir}")

    # ── load existing CASSCF data ─────────────────────────────────────────────
    casscf_dir = Path(args.casscf_dir)
    casscf_results = json.load(open(casscf_dir / 'all_casscf_results.json'))
    ref_model = pickle.load(open(casscf_dir / 'nm_delta_model_fixed.pkl', 'rb'))
    summary_in = json.load(open(casscf_dir / 'summary.json'))

    # ── load training geometries ──────────────────────────────────────────────
    data        = np.load(args.training_data, allow_pickle=True)
    train_coords = data['coordinates']      # (904, 12, 3)
    symbols     = MVKOO_SYMBOLS
    eq_coords   = ref_model.eq_coords_ang   # (12, 3) from existing NM model

    # Fill in missing e_b3lyp_rel BEFORE any filtering or printing
    # (NM-displacement frames store None for this field)
    e_ref = summary_in.get('e_b3lyp_ref_ha', casscf_results[0]['e_b3lyp'])
    for r in casscf_results:
        if r.get('e_b3lyp_rel') is None and r.get('e_b3lyp') is not None:
            r['e_b3lyp_rel'] = (r['e_b3lyp'] - e_ref) * HARTREE_TO_KCAL

    print(f"\nLoaded {len(casscf_results)} CASSCF frames from {casscf_dir.name}")
    print(f"  B3LYP range: 0 – {max(r['e_b3lyp_rel'] for r in casscf_results):.1f} kcal/mol")

    # Attach geometries to each CASSCF result
    coord_list = load_frame_geometries(casscf_results, train_coords, eq_coords)
    for r, c in zip(casscf_results, coord_list):
        r['coords'] = c

    # Energy filter
    n_before = len(casscf_results)
    frames = [r for r in casscf_results
              if (r.get('e_b3lyp_rel') or 0) <= args.filter_max_energy
              and r['coords'] is not None]
    n_skipped = n_before - len(frames)
    print(f"\nAfter filter (ΔE < {args.filter_max_energy} kcal/mol): "
          f"{len(frames)} frames ({n_skipped} skipped)")

    # ── run NEVPT2 (or reload checkpoints) ───────────────────────────────────
    ckpt_dir = out_dir / 'frame_checkpoints'
    resume   = args.resume is not None or args.retrain_only

    if args.retrain_only:
        print("\nRetrain-only mode: reloading checkpoint JSONs ...")
        all_nevpt2 = []
        for fr in frames:
            p = ckpt_dir / f"frame_{fr['frame_idx']:05d}.json"
            if p.exists():
                with open(p) as f:
                    all_nevpt2.append(json.load(f))
            else:
                print(f"  WARNING: checkpoint missing for frame {fr['frame_idx']}")
                all_nevpt2.append({'frame_idx': fr['frame_idx'],
                                   'converged': False, 'error': 'missing_checkpoint'})
    else:
        print(f"\nRunning PySCF CASSCF({args.n_active_orb},{args.n_active_elec}) + "
              f"SC-NEVPT2 / 6-31G* ...")
        est_min = len(frames) * 80 / 60
        print(f"Estimated wall time: ~{est_min:.0f} min "
              f"({len(frames)} frames × ~80 s each)")
        all_nevpt2 = run_nevpt2_batch(
            frames, symbols,
            n_active_orb=args.n_active_orb,
            n_active_elec=args.n_active_elec,
            checkpoint_dir=ckpt_dir,
            resume=resume,
        )

    # ── filter to converged frames ────────────────────────────────────────────
    good_idx = [i for i, r in enumerate(all_nevpt2)
                if r.get('converged') and r.get('delta_total_ha') is not None]
    print(f"\nConverged: {len(good_idx)}/{len(frames)} frames")

    if len(good_idx) < 5:
        print("ERROR: too few converged frames for KRR training. Exiting.")
        sys.exit(1)

    good_results = [all_nevpt2[i] for i in good_idx]
    good_frames  = [frames[i]     for i in good_idx]

    # ── project to NM coordinates ─────────────────────────────────────────────
    print("\nProjecting geometries to NM coordinates ...")
    X_q = project_batch(
        [f['coords'] for f in good_frames],
        eq_coords, ref_model.U_vib, ref_model.sqrt_mass
    )

    # Raw absolute corrections: NEVPT2 and B3LYP use different Hamiltonians,
    # so E_NEVPT2 - E_B3LYP ≈ −611 kcal/mol (constant systematic offset).
    # We need RELATIVE corrections: how much does (NEVPT2 - B3LYP) CHANGE
    # across geometries relative to the equilibrium?
    #
    #   δ_total_rel(R) = [E_NEVPT2(R) - E_B3LYP(R)]
    #                  - [E_NEVPT2(R_ref) - E_B3LYP(R_ref)]
    #
    # This is zero at R_ref and captures the differential NEVPT2 correction
    # to the B3LYP PES curvature.  The absolute offset cancels because the
    # ML-PES already uses relative B3LYP energies.

    # Compute from absolute energies stored in checkpoints
    e_nevpt2_abs = np.array([r['e_nevpt2_ha']  for r in good_results])
    e_b3lyp_abs  = np.array([r['e_b3lyp_ha']   for r in good_results])
    e_casscf_abs = np.array([r['e_casscf_pyscf_ha'] for r in good_results])

    # Reference = frame closest to equilibrium (smallest ‖q‖)
    q_norms = np.linalg.norm(X_q, axis=1)
    i_ref   = int(np.argmin(q_norms))

    e_nevpt2_ref = e_nevpt2_abs[i_ref]
    e_b3lyp_ref  = e_b3lyp_abs[i_ref]
    e_casscf_ref = e_casscf_abs[i_ref]

    # Relative correction targets (Ha) — zero at equilibrium
    y_total = (e_nevpt2_abs - e_b3lyp_abs) - (e_nevpt2_ref - e_b3lyp_ref)
    y_cas   = (e_casscf_abs - e_b3lyp_abs) - (e_casscf_ref - e_b3lyp_ref)
    y_dyn   = (e_nevpt2_abs - e_casscf_abs) - (e_nevpt2_ref - e_casscf_ref)

    print(f"  Reference frame idx={good_frames[i_ref]['frame_idx']} "
          f"(‖q‖={q_norms[i_ref]:.4f} √amu·Bohr)")
    print(f"  Absolute δ at ref:  CASSCF-B3LYP={((e_casscf_ref-e_b3lyp_ref)*HARTREE_TO_KCAL):.1f}  "
          f"NEVPT2-CASSCF={((e_nevpt2_ref-e_casscf_ref)*HARTREE_TO_KCAL):.1f}  "
          f"NEVPT2-B3LYP={((e_nevpt2_ref-e_b3lyp_ref)*HARTREE_TO_KCAL):.1f} kcal/mol")
    print(f"  Relative δ_total range: "
          f"{y_total.min()*HARTREE_TO_KCAL:.3f} to {y_total.max()*HARTREE_TO_KCAL:.3f} kcal/mol")
    print(f"  Relative δ_NEVPT2 range: "
          f"{y_dyn.min()*HARTREE_TO_KCAL:.3f} to {y_dyn.max()*HARTREE_TO_KCAL:.3f} kcal/mol")

    # ── grid search for KRR hyperparameters ──────────────────────────────────
    gammas = [float(x) for x in args.gamma_values.split(',')]
    alphas = [float(x) for x in args.alpha_values.split(',')]

    print(f"\nGrid search (γ × α) for δ_total ({len(gammas)}×{len(alphas)} grid) ...")
    best_gamma, best_alpha, best_cv, _ = grid_search_krr(
        X_q, y_total, gammas, alphas, verbose=True)
    print(f"\n  Best: γ={best_gamma:.1e}  α={best_alpha:.1e}  "
          f"LOO-CV RMSE={best_cv:.3f} kcal/mol")

    # Also CV for decomposed models (diagnostic)
    _, _, cv_cas, _ = grid_search_krr(X_q, y_cas, gammas, alphas, verbose=False)
    _, _, cv_dyn, _ = grid_search_krr(X_q, y_dyn, gammas, alphas, verbose=False)
    print(f"  δ_CASSCF alone LOO-CV: {cv_cas:.3f} kcal/mol")
    print(f"  δ_NEVPT2 alone LOO-CV: {cv_dyn:.3f} kcal/mol")

    # ── train final model ─────────────────────────────────────────────────────

    model = NEVPTKRRModel(
        eq_coords_ang        = eq_coords,
        U_vib                = ref_model.U_vib,
        sqrt_mass            = ref_model.sqrt_mass,
        freqs_vib            = ref_model.freqs_vib,
        symbols              = symbols,
        gamma                = best_gamma,
        alpha_reg            = best_alpha,
        X_train_q            = X_q,
        y_train_ha           = y_total,
        y_train_casscf_ha    = y_cas,
        y_train_nevpt2_ha    = y_dyn,
        e_b3lyp_ref_ha       = float(e_b3lyp_ref),
        e_cas_ref_ha         = float(e_casscf_ref),
        e_nevpt2_ref_ha      = float(e_nevpt2_ref),
        cv_rmse_kcal         = best_cv,
        casscf_cv_rmse_kcal  = cv_cas,
        nevpt2_cv_rmse_kcal  = cv_dyn,
    )

    # ── train RMSE ────────────────────────────────────────────────────────────
    K_full = model._kernel(X_q, X_q)
    y_pred = K_full @ model._alpha_vec
    train_rmse = float(np.sqrt(np.mean((y_total - y_pred) ** 2))) * HARTREE_TO_KCAL
    print(f"\n  Train RMSE (δ_total): {train_rmse:.4f} kcal/mol")

    # ── save everything ───────────────────────────────────────────────────────
    model_path = out_dir / 'nevpt2_correction_model.pkl'
    model.save(str(model_path))

    # Full per-frame results
    with open(out_dir / 'nevpt2_results.json', 'w') as f:
        json.dump(good_results, f, indent=2)

    # Summary
    b3lyp_rels = np.array([r['e_b3lyp_rel_kcal'] for r in good_results])
    dyn_kcal   = y_dyn * HARTREE_TO_KCAL
    cas_kcal   = y_cas * HARTREE_TO_KCAL
    tot_kcal   = y_total * HARTREE_TO_KCAL
    summary_out = {
        'timestamp':              datetime.now().isoformat(),
        'casscf_dir':             str(casscf_dir),
        'training_data':          args.training_data,
        'n_frames_total':         len(frames),
        'n_frames_converged':     len(good_idx),
        'n_active_orb':           args.n_active_orb,
        'n_active_elec':          args.n_active_elec,
        'basis':                  '6-31g*',
        'method':                 'CASSCF+SC-NEVPT2 (PySCF)',
        'b3lyp_rel_min_kcal':     float(b3lyp_rels.min()),
        'b3lyp_rel_max_kcal':     float(b3lyp_rels.max()),
        'delta_casscf_range_kcal': [float(cas_kcal.min()), float(cas_kcal.max())],
        'delta_nevpt2_range_kcal': [float(dyn_kcal.min()), float(dyn_kcal.max())],
        'delta_total_range_kcal':  [float(tot_kcal.min()), float(tot_kcal.max())],
        'gamma':                  best_gamma,
        'alpha_reg':              best_alpha,
        'loo_cv_rmse_kcal':       best_cv,
        'train_rmse_kcal':        train_rmse,
        'casscf_cv_rmse_kcal':    cv_cas,
        'nevpt2_cv_rmse_kcal':    cv_dyn,
        'model_path':             str(model_path),
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary_out, f, indent=2)
    print(f"\n  Summary saved → {out_dir / 'summary.json'}")

    # Diagnostics plot
    good_mask = np.ones(len(good_results), dtype=bool)  # all converged
    make_diagnostics(good_results, X_q, y_total, y_cas, y_dyn, model, out_dir)

    # ── final report ──────────────────────────────────────────────────────────
    print(f"""
=== Summary ===
Frames trained:      {len(good_idx)}
B3LYP range:         {b3lyp_rels.min():.1f} – {b3lyp_rels.max():.1f} kcal/mol
                     [ absorption <15 | emission 15-55 | high >55 ]
δ_CASSCF range:      {cas_kcal.min():.2f} – {cas_kcal.max():.2f} kcal/mol
δ_NEVPT2 range:      {dyn_kcal.min():.2f} – {dyn_kcal.max():.2f} kcal/mol
δ_total range:       {tot_kcal.min():.2f} – {tot_kcal.max():.2f} kcal/mol
KRR γ / α:           {best_gamma:.1e} / {best_alpha:.1e}
LOO-CV RMSE:         {best_cv:.3f} kcal/mol
Train RMSE:          {train_rmse:.4f} kcal/mol
Model:               {model_path}

To use with IR spectrum:
  python3 ir_md_spectrum.py \\
      --model outputs/mvko_20260319_081314/mlpes_initial.pkl \\
      --training-data outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \\
      --nm-delta-model {model_path} \\
      --steps 30000 --temp 300 --preminimize \\
      --zpe-min-freq 50 --zpe-max-freq 4000

For IR emission (hot molecule, ~840 K effective temperature):
  python3 ir_md_spectrum.py \\
      --model outputs/mvko_20260319_081314/mlpes_initial.pkl \\
      --training-data outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \\
      --nm-delta-model {model_path} \\
      --steps 30000 --temp 840 --preminimize \\
      --zpe-min-freq 50 --zpe-max-freq 4000
""")


if __name__ == '__main__':
    main()
