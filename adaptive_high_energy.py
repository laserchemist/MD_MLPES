#!/usr/bin/env python3
"""
adaptive_high_energy.py — Adaptive high-energy training data collection
=======================================================================
Extends ML-PES training coverage to high-energy (anharmonic) regions by
iteratively identifying structures where the ensemble model has high
uncertainty, computing PSI4 energies/forces for those structures, and
retraining.

Geometry generation uses ONLY:
  - Normal-mode distortions at high T (T_nm up to 3000 K)
  - Short PSI4 MD bursts from the equilibrium geometry

It NEVER uses ML-MD to generate candidates, which caused runaway dynamics
at high temperature (5000–10000 K) in earlier experiments.

Usage
-----
    python3 adaptive_high_energy.py \\
        --model    outputs/mvko_20260319_081314/mlpes_initial.pkl \\
        --training-data outputs/mvko_20260319_081314/combined_training_data.npz \\
        --state    outputs/mvko_20260319_081314/state.json \\
        --cycles   3 \\
        --T-nm     3000 \\
        --n-amplitudes 6 \\
        --md-steps 50 \\
        --md-temps 1000,2000 \\
        --top-n    30 \\
        --output   outputs/adaptive_high_energy_YYYYMMDD/

Key design decisions
--------------------
- Candidate geometry generation: NM distortions + PSI4 MD (never ML-MD)
- Scoring: CommitteeModel prediction variance (kcal/mol)
- Stratification: candidates bucketed into energy tiers so high-energy
  regions are sampled even when most frames are near equilibrium
- Geometry screening: reject frames with implausible bond lengths or
  ML energy > E_train_max + 100 kcal/mol before running PSI4
- Calibration: committee model calibrated against existing training data
  after each cycle to keep uncertainty estimates meaningful
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Optional PSI4 ─────────────────────────────────────────────────────────────
try:
    import psi4
    PSI4_AVAILABLE = True
    print(f"PSI4 {psi4.__version__} available")
except ImportError:
    PSI4_AVAILABLE = False
    print("PSI4 not available — will use mock energies for testing")

PSI4_METHOD  = 'b3lyp'
PSI4_OPTIONS = {'basis': '6-31G*', 'scf_type': 'df', 'reference': 'rhf',
                'maxiter': 200, 'e_convergence': 1e-7, 'd_convergence': 1e-7}
PSI4_MEM_GB  = 4
PSI4_THREADS = 4
HARTREE_TO_KCAL = 627.509474
ANGSTROM_TO_BOHR = 1.88973

# Minimum contact distances (Å) for element pairs.
# Only the lower bound is enforced — atoms closer than this indicate a crash
# geometry (nuclear repulsion runaway).  We do NOT enforce upper bounds because
# non-bonded atom pairs are legitimately far apart (e.g. O···H across a ring),
# and high-temperature / anharmonic configurations stretch bonds well beyond
# equilibrium without being unphysical.
MIN_CONTACT_DIST: dict = {
    ('H', 'H'): 0.50,
    ('H', 'C'): 0.70,
    ('H', 'O'): 0.60,
    ('H', 'N'): 0.65,
    ('C', 'C'): 1.00,
    ('C', 'O'): 0.90,
    ('C', 'N'): 0.95,
    ('O', 'O'): 0.90,
    ('N', 'O'): 0.90,
}
# Keep alias for backward compat (only lo used now)
SAFE_BOND_RANGES: dict = {k: (v, 99.0) for k, v in MIN_CONTACT_DIST.items()}


# ── PSI4 helpers ──────────────────────────────────────────────────────────────

def _psi4_setup():
    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.be_quiet()
    psi4.set_memory(f'{PSI4_MEM_GB} GB')
    psi4.set_num_threads(PSI4_THREADS)
    psi4.set_options(PSI4_OPTIONS)


def _mol_str(symbols, coords):
    lines = ['0 1']
    for s, c in zip(symbols, coords):
        lines.append(f'{s}  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}')
    lines += ['units angstrom', 'no_reorient', 'no_com']
    return '\n'.join(lines)


def psi4_energy_and_forces(symbols, coords):
    """
    PSI4 energy (Hartree) and forces (Hartree/Å).
    Returns (energy, forces) or raises on PSI4 failure.
    """
    _psi4_setup()
    mol = psi4.geometry(_mol_str(symbols, coords))
    grad, wfn = psi4.driver.gradient(
        f'{PSI4_METHOD}/{PSI4_OPTIONS["basis"]}',
        molecule=mol, return_wfn=True)
    energy = float(wfn.energy())
    grad_np = np.array(grad)               # Hartree/Bohr
    forces  = -grad_np / ANGSTROM_TO_BOHR  # Hartree/Å
    return energy, forces


def mock_energy_and_forces(symbols, coords, E_ref):
    """Placeholder used when PSI4 is unavailable."""
    rng = np.random.default_rng(int(abs(coords.sum()) * 1e6) % (2**32))
    energy = E_ref + rng.normal(0, 0.005)
    forces = rng.normal(0, 0.01, size=coords.shape)
    return energy, forces


# ── Geometry screening ────────────────────────────────────────────────────────

def _pair_key(sa, sb):
    a, b = sorted([sa, sb])
    return (a, b)


def check_geometry(symbols, coords, trainer_predict_fn=None,
                   e_train_max_ha=None, e_margin_kcal=100.0) -> tuple:
    """
    Screen a geometry for physical plausibility.

    Returns (ok: bool, reason: str).
    """
    n = len(symbols)
    for i in range(n):
        for j in range(i + 1, n):
            r = float(np.linalg.norm(coords[i] - coords[j]))
            key = _pair_key(symbols[i], symbols[j])
            lo = MIN_CONTACT_DIST.get(key, 0.3)
            if r < lo:
                return False, f'{symbols[i]}-{symbols[j]} too close: {r:.3f} Å'
    if trainer_predict_fn is not None and e_train_max_ha is not None:
        e_ml = trainer_predict_fn(symbols, coords)
        if e_ml > e_train_max_ha + e_margin_kcal / HARTREE_TO_KCAL:
            return False, f'ML energy {e_ml*HARTREE_TO_KCAL:.1f} kcal/mol too high'
    return True, 'ok'


# ── Candidate generation ──────────────────────────────────────────────────────

def generate_nm_candidates(symbols, eq_coords, hessian_path,
                            T_nm=3000, n_amplitudes=6, max_factor=3.0,
                            seed=0):
    """
    Generate candidate geometries via normal-mode distortions.

    Parameters
    ----------
    hessian_path : str
        Path to a .npy file containing the (3N, 3N) Cartesian Hessian
        in Hartree/Bohr² (as produced by modules/normal_modes.py
        compute_hessian_psi4 and saved via np.save).
    T_nm : float
        Temperature scale for displacement amplitude (K).
    n_amplitudes : int
        Number of amplitude steps per mode (positive; both ± generated).
    max_factor : float
        Maximum amplitude as a multiple of thermal amplitude.

    Returns
    -------
    coords_array : (N_cand, n_atoms, 3) Angstrom
    """
    try:
        from modules.normal_modes import compute_normal_modes, generate_nm_displacements
    except ImportError:
        from normal_modes import compute_normal_modes, generate_nm_displacements

    H = np.load(hessian_path)   # (3N, 3N) Hartree/Bohr²
    freqs, eigvecs_mw, eigenvalues, mass_vec = compute_normal_modes(symbols, H)

    displacements = generate_nm_displacements(
        symbols, eq_coords, eigvecs_mw, eigenvalues, mass_vec,
        T=T_nm, n_amplitudes=n_amplitudes, max_factor=max_factor)

    coords_list = [d[0] for d in displacements]   # d = (coords, mode_idx, factor)
    return np.array(coords_list)


def generate_psi4_md_candidates(symbols, eq_coords,
                                 temps=(1000, 2000), steps=50, seed=0):
    """
    Run short PSI4 MD bursts and return sampled geometries.

    Returns (coords_array, energies_array, forces_array) or (None,)*3 on failure.
    """
    try:
        from modules.direct_md import DirectMDConfig, DirectMDRunner
        from modules.test_molecules import TestMolecule
    except ImportError:
        from direct_md import DirectMDConfig, DirectMDRunner
        from test_molecules import TestMolecule

    formula = ''.join(
        f"{s}{symbols.count(s)}" if symbols.count(s) > 1 else s
        for s in dict.fromkeys(symbols))
    mol = TestMolecule(name='custom', formula=formula, symbols=symbols,
                       coordinates=eq_coords.copy(), charge=0, multiplicity=1)

    all_coords, all_energies, all_forces = [], [], []
    for T in temps:
        cfg = DirectMDConfig(
            method='B3LYP', basis='6-31G*',
            temperature=T, timestep=0.5, n_steps=steps,
            output_frequency=1, thermostat='berendsen',
            thermostat_coupling=20.0, random_seed=seed + int(T))
        runner = DirectMDRunner(cfg)
        try:
            traj = runner.run(mol)
            if traj is None or traj.n_frames == 0:
                continue
            all_coords.append(traj.coordinates)
            all_energies.append(traj.energies)
            all_forces.append(traj.forces)
            print(f"    PSI4 MD at {T}K: {traj.n_frames} frames", flush=True)
        except Exception as exc:
            print(f"    PSI4 MD at {T}K failed: {exc}")

    if not all_coords:
        return None, None, None
    return (np.concatenate(all_coords),
            np.concatenate(all_energies),
            np.concatenate(all_forces))


# ── Energy stratification ─────────────────────────────────────────────────────

ENERGY_TIERS_KCAL = [(0, 5), (5, 15), (15, 30), (30, 60)]
TIER_FRACTION = [0.15, 0.30, 0.30, 0.25]   # fraction of top_n per tier


def stratified_select(candidate_coords, candidate_energies_ha, candidate_sigmas,
                       top_n=30, e_ref_ha=None):
    """
    Select top_n candidates distributed across energy tiers.

    Within each tier candidates are ranked by sigma (highest first).

    Parameters
    ----------
    candidate_coords  : (N, n_atoms, 3)
    candidate_energies_ha : (N,)  ML energies in Hartree
    candidate_sigmas  : (N,)  uncertainty in kcal/mol
    top_n             : int
    e_ref_ha          : float  (reference energy; default = min of training set)

    Returns
    -------
    selected_idx : np.ndarray of int
    """
    if e_ref_ha is None:
        e_ref_ha = candidate_energies_ha.min()
    de_kcal = (candidate_energies_ha - e_ref_ha) * HARTREE_TO_KCAL

    selected = []
    remaining = np.ones(len(candidate_coords), dtype=bool)

    for (lo, hi), frac in zip(ENERGY_TIERS_KCAL, TIER_FRACTION):
        n_tier = max(1, int(round(top_n * frac)))
        in_tier = remaining & (de_kcal >= lo) & (de_kcal < hi)
        tier_idx = np.where(in_tier)[0]
        if len(tier_idx) == 0:
            continue
        order = np.argsort(-candidate_sigmas[tier_idx])   # descending sigma
        chosen = tier_idx[order[:n_tier]]
        selected.extend(chosen.tolist())
        remaining[chosen] = False

    # Backfill with highest-sigma frames from any tier
    if len(selected) < top_n:
        leftover = np.where(remaining)[0]
        order = np.argsort(-candidate_sigmas[leftover])
        need = top_n - len(selected)
        selected.extend(leftover[order[:need]].tolist())

    return np.array(selected[:top_n])


# ── Committee helpers ─────────────────────────────────────────────────────────

def build_committee(symbols, training_data_path, gamma, alpha,
                    k_models=5, seed=42):
    """Load training data and build+train a CommitteeModel."""
    try:
        from modules.uncertainty import CommitteeModel
    except ImportError:
        from uncertainty import CommitteeModel

    data = np.load(training_data_path, allow_pickle=True)
    coords   = data['coordinates']
    energies = data['energies']
    committee = CommitteeModel(symbols, coords, energies,
                               k_models=k_models, gamma=gamma, alpha=alpha,
                               seed=seed)
    committee.train(verbose=False)
    return committee, coords, energies


# ── Main adaptive loop ────────────────────────────────────────────────────────

def run_adaptive_cycle(args):
    """Execute one complete adaptive high-energy training run."""
    # ── Load model and training data ─────────────────────────────────────────
    try:
        from modules.ml_pes import MLPESTrainer, MLPESConfig
        from modules.data_formats import TrajectoryData
    except ImportError:
        from ml_pes import MLPESTrainer, MLPESConfig
        from data_formats import TrajectoryData

    trainer = MLPESTrainer.load(args.model)
    symbols = trainer.symbols
    data = np.load(args.training_data, allow_pickle=True)
    coords_all   = data['coordinates']
    energies_all = data['energies']
    forces_all   = data['forces'] if 'forces' in data else np.zeros(
        (len(coords_all), len(symbols), 3))
    e_train_max = float(energies_all.max())
    e_train_min = float(energies_all.min())

    # Equilibrium geometry = lowest-energy frame
    eq_idx    = int(np.argmin(energies_all))
    eq_coords = coords_all[eq_idx]
    print(f"Training set: {len(coords_all)} frames, "
          f"E range = [{e_train_min*HARTREE_TO_KCAL:.2f}, "
          f"{e_train_max*HARTREE_TO_KCAL:.2f}] kcal/mol", flush=True)

    # ── Output directory ──────────────────────────────────────────────────────
    if args.output is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = Path(f'outputs/adaptive_high_energy_{ts}')
    else:
        out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine gamma/alpha from loaded model ───────────────────────────────
    gamma = trainer.config.gamma
    alpha = trainer.config.alpha
    print(f"Using γ={gamma}, α={alpha} from loaded model", flush=True)

    # Accumulate new frames across cycles
    new_coords   = list(coords_all)
    new_energies = list(energies_all)
    new_forces   = list(forces_all)
    cycle_log    = []

    for cycle in range(1, args.cycles + 1):
        print(f"\n{'='*60}", flush=True)
        print(f"CYCLE {cycle}/{args.cycles}", flush=True)
        print(f"{'='*60}", flush=True)
        t_cycle = time.time()

        # ── Step 1: Build committee model ─────────────────────────────────────
        print("  Building committee model …", flush=True)
        try:
            from modules.uncertainty import CommitteeModel
        except ImportError:
            from uncertainty import CommitteeModel

        n_current = len(new_coords)
        coords_np   = np.array(new_coords)
        energies_np = np.array(new_energies)
        committee = CommitteeModel(symbols, coords_np, energies_np,
                                   k_models=args.k_models,
                                   gamma=gamma, alpha=alpha, seed=42 + cycle)
        committee.train(verbose=True)

        # Calibrate on existing training data (10% holdout)
        n_cal = max(5, n_current // 10)
        rng = np.random.default_rng(cycle)
        cal_idx = rng.choice(n_current, size=n_cal, replace=False)
        committee.calibrate(symbols, coords_np[cal_idx], energies_np[cal_idx])

        # ── Step 2a: PSI4 MD bursts — energies already computed, add directly ─
        n_md_added = 0
        if PSI4_AVAILABLE:
            md_temps = [int(t) for t in args.md_temps.split(',')]
            print("  Running PSI4 MD bursts …", flush=True)
            md_coords, md_energies_arr, md_forces_arr = generate_psi4_md_candidates(
                symbols, eq_coords, temps=md_temps, steps=args.md_steps,
                seed=cycle * 200)
            if md_coords is not None:
                for c, e, f in zip(md_coords, md_energies_arr, md_forces_arr):
                    ok, _ = check_geometry(symbols, c,
                                           trainer_predict_fn=trainer.predict,
                                           e_train_max_ha=e_train_max)
                    if ok:
                        new_coords.append(c)
                        new_energies.append(float(e))
                        new_forces.append(f)
                        n_md_added += 1
                        e_train_max = max(e_train_max, float(e))
                        e_train_min = min(e_train_min, float(e))
                print(f"    Added {n_md_added} PSI4 MD frames directly", flush=True)

        # ── Step 2b: NM distortions — score by uncertainty, eval top-N via PSI4 ─
        print("  Generating NM distortion candidates …", flush=True)
        nm_cand_coords = None
        if args.hessian_data and Path(args.hessian_data).exists():
            try:
                nm_cand_coords = generate_nm_candidates(
                    symbols, eq_coords, args.hessian_data,
                    T_nm=args.T_nm, n_amplitudes=args.n_amplitudes,
                    max_factor=3.0, seed=cycle * 100)
                print(f"    NM distortions: {len(nm_cand_coords)} candidates", flush=True)
            except Exception as exc:
                print(f"    NM distortions failed: {exc}")

        if nm_cand_coords is None and n_md_added == 0:
            print("  No new data this cycle — skipping retrain", flush=True)
            continue

        n_psi4_added = 0
        if nm_cand_coords is not None and len(nm_cand_coords) > 0:
            # ── Step 3: Screen NM candidates ──────────────────────────────────
            ok_mask = np.zeros(len(nm_cand_coords), dtype=bool)
            for i, c in enumerate(nm_cand_coords):
                ok, _ = check_geometry(symbols, c,
                                       trainer_predict_fn=trainer.predict,
                                       e_train_max_ha=e_train_max)
                ok_mask[i] = ok
            nm_ok = nm_cand_coords[ok_mask]
            print(f"  NM candidates after screening: {len(nm_ok)}/{len(nm_cand_coords)}",
                  flush=True)

            if len(nm_ok) > 0:
                # ── Step 4: Score by committee uncertainty ─────────────────────
                print("  Scoring NM candidates by committee uncertainty …", flush=True)
                cand_energies, cand_sigmas = committee.batch_uncertainty(
                    symbols, nm_ok)
                print(f"  Sigma range: {cand_sigmas.min():.4f} – "
                      f"{cand_sigmas.max():.4f} kcal/mol", flush=True)

                # ── Step 5: Stratified selection ───────────────────────────────
                sel_idx = stratified_select(nm_ok, cand_energies, cand_sigmas,
                                            top_n=args.top_n, e_ref_ha=e_train_min)
                selected_coords = nm_ok[sel_idx]
                selected_sigmas = cand_sigmas[sel_idx]
                print(f"  Selected {len(selected_coords)} NM frames for PSI4 evaluation",
                      flush=True)

                # ── Step 6: PSI4 single-points on selected NM frames ───────────
                print("  Running PSI4 on selected NM frames …", flush=True)
                for i, (c, sig) in enumerate(zip(selected_coords, selected_sigmas)):
                    try:
                        if PSI4_AVAILABLE:
                            e_ha, f_ha = psi4_energy_and_forces(symbols, c)
                        else:
                            e_ha, f_ha = mock_energy_and_forces(symbols, c, e_train_min)
                        new_coords.append(c)
                        new_energies.append(float(e_ha))
                        new_forces.append(f_ha)
                        n_psi4_added += 1
                        e_train_max = max(e_train_max, float(e_ha))
                        e_train_min = min(e_train_min, float(e_ha))
                        if (i + 1) % 10 == 0 or i == len(selected_coords) - 1:
                            print(f"    {i+1}/{len(selected_coords)} done  "
                                  f"σ={sig:.3f} kcal/mol  "
                                  f"E={e_ha*HARTREE_TO_KCAL:.2f} kcal/mol", flush=True)
                    except Exception as exc:
                        print(f"    Frame {i}: PSI4 failed ({exc})")

        n_added_total = n_md_added + n_psi4_added
        print(f"  Cycle {cycle}: +{n_md_added} MD  +{n_psi4_added} NM  "
              f"= {n_added_total} new frames  (total: {len(new_coords)})", flush=True)

        # ── Step 7: Retrain ───────────────────────────────────────────────────
        print("  Retraining ML-PES …", flush=True)
        traj_new = TrajectoryData(
            symbols=symbols,
            coordinates=np.array(new_coords),
            energies=np.array(new_energies),
            forces=np.array(new_forces),
        )
        cfg = MLPESConfig(gamma=gamma, alpha=alpha,
                          tune_hyperparameters=False, validation_split=0.1)
        trainer = MLPESTrainer(cfg)
        trainer.train(traj_new)

        # Save cycle checkpoint
        model_path = out_dir / f'mlpes_cycle{cycle}.pkl'
        trainer.save(str(model_path))
        data_path  = out_dir / f'training_data_cycle{cycle}.npz'
        np.savez(data_path,
                 symbols=np.array(symbols),
                 coordinates=np.array(new_coords),
                 energies=np.array(new_energies),
                 forces=np.array(new_forces))

        rmse = trainer.training_history.get('rmse_kcal',
               trainer.training_history.get('best_rmse_kcal', float('nan')))
        cycle_log.append({
            'cycle': cycle,
            'n_frames': len(new_coords),
            'n_added': n_added_total,
            'n_md': n_md_added,
            'n_nm': n_psi4_added,
            'rmse_kcal': float(rmse),
            'time_s': round(time.time() - t_cycle, 1),
        })
        print(f"  Cycle {cycle} done — RMSE={rmse:.4f} kcal/mol  "
              f"({time.time()-t_cycle:.0f}s)", flush=True)

    # ── Final save ────────────────────────────────────────────────────────────
    final_model = out_dir / 'mlpes_adaptive_final.pkl'
    trainer.save(str(final_model))
    final_data = out_dir / 'training_data_final.npz'
    np.savez(final_data,
             symbols=np.array(symbols),
             coordinates=np.array(new_coords),
             energies=np.array(new_energies),
             forces=np.array(new_forces))

    log_path = out_dir / 'adaptive_log.json'
    with open(log_path, 'w') as f:
        json.dump(cycle_log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Adaptive training complete.")
    print(f"  Final model  : {final_model}")
    print(f"  Training data: {final_data}")
    print(f"  Cycle log    : {log_path}")
    print(f"\nSuggested next step:")
    print(f"  python3 collect_mvko_dipoles.py \\")
    print(f"    --training-data {final_data} --n-frames 200")
    print(f"  python3 ir_md_spectrum.py --model {final_model} \\")
    print(f"    --training-data <dipole_data.npz> \\")
    print(f"    --steps 30000 --temp 300 --preminimize \\")
    print(f"    --zpe-min-freq 50 --zpe-max-freq 4000")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive high-energy training data collection")
    parser.add_argument('--model',         required=True,
                        help='Trained ML-PES .pkl path')
    parser.add_argument('--training-data', required=True,
                        help='Existing training data .npz')
    parser.add_argument('--state',         default=None,
                        help='Optional mvko_workflow state.json (for restart)')
    parser.add_argument('--cycles',        type=int, default=3,
                        help='Number of adaptive cycles (default 3)')
    parser.add_argument('--T-nm',          type=int, default=3000,
                        help='Temperature for NM distortions in K (default 3000)')
    parser.add_argument('--n-amplitudes',  type=int, default=6,
                        help='NM amplitude steps per mode (default 6)')
    parser.add_argument('--md-steps',      type=int, default=50,
                        help='Steps per PSI4 MD burst (default 50)')
    parser.add_argument('--md-temps',      default='1000,2000',
                        help='Comma-separated temperatures for PSI4 MD bursts')
    parser.add_argument('--top-n',         type=int, default=30,
                        help='Frames to add per cycle (default 30)')
    parser.add_argument('--k-models',      type=int, default=5,
                        help='Committee size (default 5)')
    parser.add_argument('--hessian-data',  default=None,
                        help='PSI4 Hessian .npy file (3N×3N, Hartree/Bohr²) — enables NM distortions. '
                             'Produced by mvko_workflow.py Step 2 and saved as psi4_hessian.npy.')
    parser.add_argument('--output',        default=None,
                        help='Output directory (default: auto-timestamped)')
    args = parser.parse_args()
    run_adaptive_cycle(args)


if __name__ == '__main__':
    main()
