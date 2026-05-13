#!/usr/bin/env python3
"""
GP posterior-variance diagnostic for the ML dipole surface.

KRR and GP regression with the same RBF kernel give identical mean
predictions.  GP also gives the posterior variance:

    σ²(x*) = k(x*,x*) - k(x*,X) @ (K + α I)^{-1} @ k(X,x*)

which equals zero at training points and grows as x* leaves the
training manifold.  We use this to:

  1. Show WHERE in MD configuration space the dipole surface is
     extrapolating (coverage map across the trajectory).
  2. Rank frames by uncertainty and compute wB97X-D/6-31G* PSI4
     dipoles at the top-N (active learning round).

Usage:
    # Coverage analysis only (no PSI4)
    python3 gp_dipole_coverage.py \
        --training-data  outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \
        --traj           outputs/ir_spectrum_NM_PES_v5_300K_20260512_204433/traj_01.xyz \
        --output-dir     outputs/gp_dipole_coverage

    # Coverage + compute top-N new wB97X-D dipoles
    python3 gp_dipole_coverage.py \
        --training-data  outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \
        --traj           outputs/ir_spectrum_NM_PES_v5_300K_20260512_204433/traj_01.xyz \
        --output-dir     outputs/gp_dipole_coverage \
        --compute-top-n  50 \
        --method         wb97x-d \
        [--dry-run]

    # Merge new points into existing training set
    python3 gp_dipole_coverage.py \
        --training-data  outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \
        --merge          outputs/gp_dipole_coverage/new_dipoles.npz \
        --output-dir     outputs/mvko_dipoles_gp_round1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Descriptor ───────────────────────────────────────────────────────────────

ATOM_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'F': 9, 'Cl': 17}

def coulomb_matrix(symbols, coords):
    """Upper-triangle Coulomb matrix as a flat descriptor."""
    n = len(symbols)
    Z = np.array([ATOM_Z[s] for s in symbols], dtype=float)
    feat = []
    for i in range(n):
        for j in range(i, n):
            if i == j:
                feat.append(0.5 * Z[i] ** 2.4)
            else:
                r = np.linalg.norm(coords[i] - coords[j])
                feat.append(Z[i] * Z[j] / r)
    return np.array(feat)

def compute_descriptors(symbols, coords_arr):
    """coords_arr: (N, n_atoms, 3)  →  (N, n_desc)"""
    return np.array([coulomb_matrix(symbols, c) for c in coords_arr])


# ── GP posterior variance ────────────────────────────────────────────────────

class GPDipoleModel:
    """
    Trains three independent GP models (one per dipole component).
    Fits using the same RBF kernel as DipoleSurface (via scikit-learn
    KernelRidge), then computes the posterior variance analytically.
    """

    def __init__(self, gamma=0.001, alpha=1e-4):
        self.gamma = gamma
        self.alpha = alpha
        self._fitted = False

    def _rbf(self, X, Y):
        """RBF kernel matrix  K[i,j] = exp(-γ |X_i - Y_j|²)."""
        # (N, D) and (M, D) → (N, M)
        diff2 = (
            np.sum(X ** 2, axis=1, keepdims=True)
            + np.sum(Y ** 2, axis=1, keepdims=True).T
            - 2.0 * X @ Y.T
        )
        return np.exp(-self.gamma * diff2)

    def fit(self, X_raw, y_dipoles):
        """
        X_raw     : (N, n_desc)  raw (unscaled) descriptors
        y_dipoles : (N, 3)       dipole components in Debye
        """
        from sklearn.preprocessing import StandardScaler
        self.sx = StandardScaler().fit(X_raw)
        self.sy = StandardScaler().fit(y_dipoles)
        X = self.sx.transform(X_raw)
        y = self.sy.transform(y_dipoles)

        self._X_train = X
        K = self._rbf(X, X)
        n = len(X)
        A = K + self.alpha * np.eye(n)
        # Cholesky for numerically stable solve and for variance
        self._L = np.linalg.cholesky(A)            # L L^T = A
        # dual coefficients: α_mat = A^{-1} y  (one column per component)
        self._alpha_mat = np.linalg.solve(A, y)    # (N, 3)
        # A^{-1} stored implicitly via L for fast variance computation
        # Precompute L^{-1} for variance  (triangular, cheap)
        self._L_inv = np.linalg.inv(self._L)       # (N, N)  upper-right zero
        self._fitted = True
        print(f'  GP dipole fitted: {n} training pts, γ={self.gamma}, α={self.alpha}')

    def predict(self, X_raw):
        """Returns (N, 3) predicted dipoles in Debye."""
        X = self.sx.transform(X_raw)
        k = self._rbf(X, self._X_train)            # (N_test, N_train)
        y_sc = k @ self._alpha_mat                  # (N_test, 3)
        return self.sy.inverse_transform(y_sc)

    def posterior_variance(self, X_raw):
        """
        Returns (N_test, 3) posterior standard deviations in Debye.

        σ²(x*) = k(x*,x*) - k(x*,X) L^{-T} L^{-1} k(X,x*)
               = 1 - ||L^{-1} k(X,x*)||²        (because k(x*,x*)=1 for RBF)

        We return σ (std dev, not variance), one per dipole component.
        Because the three components share the same descriptor kernel
        (only the dual coefficients differ), σ is the same for all three.
        We return it broadcast to shape (N_test, 3) for convenience.
        """
        X = self.sx.transform(X_raw)
        k_star = self._rbf(self._X_train, X)       # (N_train, N_test)
        # v = L^{-1} k_star                         (N_train, N_test)
        v = self._L_inv @ k_star
        # σ² = k(x*,x*) - ||v||²  =  1 - sum_i v_i²
        var = np.maximum(0.0, 1.0 - np.sum(v ** 2, axis=0))  # (N_test,)
        # Scale variance back to Debye units (approximation: use mean y-std)
        y_std_mean = self.sy.scale_.mean()
        sigma_D = np.sqrt(var) * y_std_mean
        return np.column_stack([sigma_D, sigma_D, sigma_D])  # (N_test, 3)

    def uncertainty_scalar(self, X_raw):
        """Returns (N_test,) scalar uncertainty = ||σ|| in Debye."""
        sigma = self.posterior_variance(X_raw)
        return np.linalg.norm(sigma, axis=1)

    def tune_and_fit(self, X_raw, y_dipoles,
                     gamma_values=(0.0001, 0.001, 0.01, 0.1),
                     alpha_values=(1e-6, 1e-4, 1e-2)):
        """LOO-CV grid search, then final fit on all data."""
        from sklearn.preprocessing import StandardScaler
        sx = StandardScaler().fit(X_raw)
        sy = StandardScaler().fit(y_dipoles)
        X = sx.transform(X_raw)
        y = sy.transform(y_dipoles)
        n = len(X)

        best, best_rmse = (None, None), np.inf
        for g in gamma_values:
            K = np.exp(-g * (
                np.sum(X**2,1,keepdims=True)
                + np.sum(X**2,1,keepdims=True).T
                - 2*X@X.T
            ))
            for a in alpha_values:
                A = K + a * np.eye(n)
                try:
                    alpha_vec = np.linalg.solve(A, y)       # (N,3)
                    # LOO via Sherman-Morrison-Woodbury identity
                    A_inv_diag = np.sum(np.linalg.inv(A), axis=1)  # approx diagonal
                    loo_err = (alpha_vec / A_inv_diag[:, None]) ** 2
                    rmse = np.sqrt(loo_err.mean())
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best = (g, a)
                except np.linalg.LinAlgError:
                    continue
        g_best, a_best = best
        print(f'  Best hyperparams: γ={g_best}, α={a_best}, LOO-RMSE≈{best_rmse:.4f}')
        self.gamma = g_best
        self.alpha = a_best
        self.fit(X_raw, y_dipoles)


# ── XYZ parser ───────────────────────────────────────────────────────────────

def parse_xyz(path, stride=1):
    lines = Path(path).read_text().splitlines()
    n_atoms = int(lines[0].strip())
    step = n_atoms + 2
    n_frames = len(lines) // step
    symbols, times, coords = None, [], []
    for i in range(0, n_frames, stride):
        base = i * step
        comment = lines[base + 1]
        t = 0.0
        for tok in comment.split():
            if tok.startswith('time=') and tok.endswith('fs'):
                t = float(tok[5:-2])
        times.append(t)
        frame, syms = [], []
        for j in range(n_atoms):
            parts = lines[base + 2 + j].split()
            syms.append(parts[0])
            frame.append([float(parts[1]), float(parts[2]), float(parts[3])])
        coords.append(frame)
        if symbols is None:
            symbols = syms
    return symbols, np.array(times), np.array(coords)


# ── PSI4 dipole ───────────────────────────────────────────────────────────────

AU_TO_DEBYE = 2.541746

def psi4_dipole(symbols, coords_ang, method='wb97x-d', basis='6-31G*'):
    """Compute wB97X-D/6-31G* dipole (default) via PSI4."""
    try:
        import psi4
        psi4.core.clean_options()
        psi4.core.clean()
        psi4.core.be_quiet()
        psi4.set_memory('4 GB')
        psi4.set_num_threads(4)
        psi4.set_options({'basis': basis, 'scf_type': 'df', 'reference': 'rhf',
                          'maxiter': 200, 'e_convergence': 1e-7, 'd_convergence': 1e-7})
        geom = '0 1\n'
        for s, c in zip(symbols, coords_ang):
            geom += f'{s}  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}\n'
        geom += 'units angstrom\nno_reorient\nno_com\n'
        mol = psi4.geometry(geom)
        _, wfn = psi4.energy(f'{method}/{basis}', molecule=mol,
                             return_wfn=True, properties=['dipole'])
        try:
            dip = np.array(psi4.variable('SCF DIPOLE'))
            if np.linalg.norm(dip) < 1e-10:
                raise ValueError('zero dipole from variable')
            return dip * AU_TO_DEBYE
        except Exception:
            psi4.oeprop(wfn, 'DIPOLE')
            return np.array([wfn.variable(f'DIPOLE {ax}')
                             for ax in ['X', 'Y', 'Z']])
    except ImportError:
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--training-data', required=True,
                    help='Existing dipole npz (training set)')
    ap.add_argument('--traj', default=None, nargs='+',
                    help='MD trajectory XYZ file(s) to evaluate coverage on '
                         '(multiple files are pooled for selection)')
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--stride', type=int, default=10,
                    help='Stride for trajectory loading (default 10 → 3000 pts from 30k)')
    ap.add_argument('--gamma', type=float, default=None,
                    help='RBF gamma (default: auto grid-search)')
    ap.add_argument('--alpha', type=float, default=None,
                    help='Regularisation (default: auto grid-search)')
    ap.add_argument('--compute-top-n', type=int, default=0,
                    help='Compute PSI4 dipoles for the N highest-uncertainty traj frames')
    ap.add_argument('--method', default='wb97x-d',
                    help='PSI4 DFT method for new dipoles (default: wb97x-d)')
    ap.add_argument('--basis', default='6-31G*')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print candidate geometries without running PSI4')
    ap.add_argument('--diversity-radius', type=float, default=0.0,
                    help='Minimum Euclidean distance in *scaled* descriptor space between '
                         'any two selected frames (0 = greedy top-N without diversity; '
                         'typical useful range 3–15 for MVKO Coulomb matrix). '
                         'Enforces diverse spatial sampling so all N frames are not '
                         'drawn from the same conformational excursion.')
    ap.add_argument('--merge', default=None,
                    help='Path to new_dipoles.npz from a previous --compute-top-n run; '
                         'merges with --training-data and saves combined npz to --output-dir')
    args = ap.parse_args()

    ts = time.strftime('%Y%m%d_%H%M%S')
    out = Path(args.output_dir) if args.output_dir else Path(f'outputs/gp_dipole_{ts}')
    out.mkdir(parents=True, exist_ok=True)

    # ── Load training data ────────────────────────────────────────────────
    print(f'\nLoading training dipoles: {args.training_data}')
    d = np.load(args.training_data, allow_pickle=True)
    symbols    = list(d['symbols'])
    tr_coords  = d['coordinates']   # (N, n_atoms, 3)
    tr_dipoles = d['dipoles']       # (N, 3)  Debye
    tr_energies = d.get('energies', np.zeros(len(tr_coords)))
    tr_forces   = d.get('forces',   np.zeros_like(tr_coords))
    print(f'  {len(tr_coords)} frames, {len(symbols)} atoms')
    print(f'  |μ| range: {np.linalg.norm(tr_dipoles,axis=1).min():.3f}–'
          f'{np.linalg.norm(tr_dipoles,axis=1).max():.3f} D')

    # ── Merge mode ────────────────────────────────────────────────────────
    if args.merge:
        print(f'\nMerge mode: loading new dipoles from {args.merge}')
        nd = np.load(args.merge, allow_pickle=True)
        new_coords  = nd['coordinates']
        new_dipoles = nd['dipoles']
        print(f'  {len(new_coords)} new frames')
        all_coords  = np.concatenate([tr_coords, new_coords], axis=0)
        all_dipoles = np.concatenate([tr_dipoles, new_dipoles], axis=0)
        all_energies = np.concatenate([
            tr_energies,
            nd.get('energies', np.zeros(len(new_coords)))
        ])
        all_forces = np.concatenate([
            tr_forces,
            nd.get('forces', np.zeros_like(new_coords))
        ])
        out_path = out / 'training_with_dipoles.npz'
        np.savez(out_path,
                 symbols=np.array(symbols),
                 coordinates=all_coords,
                 energies=all_energies,
                 forces=all_forces,
                 dipoles=all_dipoles,
                 metadata=np.array(json.dumps({'merged': True, 'n_orig': len(tr_coords),
                                               'n_new': len(new_coords), 'ts': ts})))
        print(f'  Combined: {len(all_coords)} frames → {out_path}')
        return

    # ── Compute descriptors for training set ──────────────────────────────
    print('\nComputing Coulomb-matrix descriptors for training set …')
    X_train = compute_descriptors(symbols, tr_coords)
    print(f'  descriptor shape: {X_train.shape}')

    # ── Fit GP dipole model ───────────────────────────────────────────────
    print('\nFitting GP dipole model …')
    gp = GPDipoleModel(
        gamma=args.gamma if args.gamma else 0.001,
        alpha=args.alpha if args.alpha else 1e-4,
    )
    if args.gamma is None or args.alpha is None:
        gp.tune_and_fit(X_train, tr_dipoles)
    else:
        gp.fit(X_train, tr_dipoles)

    # Training-set residuals
    mu_pred = gp.predict(X_train)
    rmse = np.sqrt(np.mean((mu_pred - tr_dipoles) ** 2))
    print(f'  Training RMSE: {rmse:.4f} D')

    # Uncertainty at training points (should be ~0)
    sig_train = gp.uncertainty_scalar(X_train)
    print(f'  σ at training pts: mean={sig_train.mean():.4f} max={sig_train.max():.4f} D')

    if args.traj is None:
        print('\nNo --traj provided; exiting after training.')
        return

    # ── Load trajectory/trajectories ─────────────────────────────────────
    traj_files = args.traj if isinstance(args.traj, list) else [args.traj]
    all_times, all_coords, traj_labels = [], [], []
    for tpath in traj_files:
        print(f'\nLoading trajectory: {tpath}  (stride={args.stride})')
        syms_traj, times_fs_i, coords_i = parse_xyz(tpath, stride=args.stride)
        assert syms_traj == symbols, f"Atom ordering mismatch in {tpath}"
        print(f'  {len(times_fs_i)} frames  '
              f'({times_fs_i[0]/1000:.2f}–{times_fs_i[-1]/1000:.2f} ps)')
        all_times.append(times_fs_i)
        all_coords.append(coords_i)
        traj_labels.extend([Path(tpath).name] * len(times_fs_i))
    times_fs   = np.concatenate(all_times)
    coords_traj = np.concatenate(all_coords, axis=0)
    print(f'\nPooled: {len(times_fs)} frames from {len(traj_files)} trajectory/trajectories')

    # ── Compute descriptors for trajectory ────────────────────────────────
    print('Computing descriptors for trajectory …')
    X_traj = compute_descriptors(symbols, coords_traj)

    # ── GP uncertainty across trajectory ─────────────────────────────────
    print('Computing GP posterior variance …')
    sigma = gp.uncertainty_scalar(X_traj)   # (N_traj,)
    mu_traj = gp.predict(X_traj)            # (N_traj, 3)

    print(f'\n=== GP Dipole Coverage Report ===')
    print(f'  Traj frames evaluated : {len(sigma)}')
    print(f'  σ  mean   : {sigma.mean():.4f} D')
    print(f'  σ  median : {np.median(sigma):.4f} D')
    print(f'  σ  95th%  : {np.percentile(sigma, 95):.4f} D')
    print(f'  σ  max    : {sigma.max():.4f} D')
    print(f'  Frames with σ > 0.10 D : {(sigma > 0.10).sum()} '
          f'({100*(sigma>0.10).mean():.1f}%)')
    print(f'  Frames with σ > 0.05 D : {(sigma > 0.05).sum()} '
          f'({100*(sigma>0.05).mean():.1f}%)')
    print(f'  Frames with σ > 0.02 D : {(sigma > 0.02).sum()} '
          f'({100*(sigma>0.02).mean():.1f}%)')

    # ── Save coverage data ────────────────────────────────────────────────
    np.savez(out / 'gp_coverage.npz',
             times_fs=times_fs,
             sigma_D=sigma,
             mu_predicted_D=mu_traj,
             coords=coords_traj)
    print(f'\nSaved coverage data → {out}/gp_coverage.npz')

    # ── Plot ──────────────────────────────────────────────────────────────
    _plot_coverage(times_fs, sigma, out / 'gp_coverage.png')

    # ── Active learning: select and compute top-N ─────────────────────────
    if args.compute_top_n > 0:
        _run_active_learning(
            gp, sigma, times_fs, coords_traj, symbols,
            args.compute_top_n, args.method, args.basis,
            args.dry_run, out, traj_labels=traj_labels,
            X_traj_scaled=gp.sx.transform(X_traj),
            diversity_radius=args.diversity_radius,
        )

    # JSON summary
    summary = {
        'training_data': args.training_data,
        'traj': args.traj,
        'n_train': int(len(tr_coords)),
        'n_traj_frames': int(len(sigma)),
        'gp_gamma': float(gp.gamma),
        'gp_alpha': float(gp.alpha),
        'train_rmse_D': float(rmse),
        'sigma_mean_D': float(sigma.mean()),
        'sigma_max_D': float(sigma.max()),
        'sigma_p95_D': float(np.percentile(sigma, 95)),
        'frac_above_0p05D': float((sigma > 0.05).mean()),
        'frac_above_0p10D': float((sigma > 0.10).mean()),
    }
    (out / 'gp_coverage_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Summary → {out}/gp_coverage_summary.json')


def _plot_coverage(times_fs, sigma, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    times_ps = times_fs / 1000.0
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax = axes[0]
    ax.fill_between(times_ps, 0, sigma, alpha=0.4, color='steelblue')
    ax.plot(times_ps, sigma, lw=0.5, color='steelblue')
    for thresh, col, ls in [(0.10, 'red', '--'), (0.05, 'orange', ':'), (0.02, 'green', '-.')]:
        ax.axhline(thresh, color=col, lw=1.0, ls=ls, label=f'σ={thresh} D')
    ax.set_ylabel('GP uncertainty σ (D)', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title('GP Dipole Surface — Posterior Uncertainty along MD Trajectory', fontsize=11)

    # Running-maximum to show worst-case
    ax2 = axes[1]
    # Histogram of σ values
    ax2.hist(sigma, bins=80, color='steelblue', alpha=0.7, density=True)
    ax2.axvline(0.05, color='orange', lw=1.5, ls=':', label='0.05 D')
    ax2.axvline(0.10, color='red',    lw=1.5, ls='--', label='0.10 D')
    ax2.set_xlabel('σ (D)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_title('Distribution of GP uncertainty across trajectory frames')

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Coverage plot → {path}')


def _greedy_diverse_selection(sigma, X_scaled, top_n, diversity_radius):
    """
    Greedy σ-maximising selection with diversity constraint.

    Iteratively selects the highest-σ candidate that is at least
    `diversity_radius` away (Euclidean, scaled descriptor space) from
    all already-selected frames.  Falls back to pure top-N if
    diversity_radius <= 0.

    Returns: array of selected frame indices (length ≤ top_n).
    """
    if diversity_radius <= 0.0:
        return np.argsort(sigma)[::-1][:top_n]

    # Work in σ-descending order; skip candidates too close to any selected pt
    order = np.argsort(sigma)[::-1]
    selected = []
    selected_X = []

    for idx in order:
        if len(selected) >= top_n:
            break
        x = X_scaled[idx]
        if selected_X:
            dists = np.linalg.norm(np.array(selected_X) - x[None, :], axis=1)
            if dists.min() < diversity_radius:
                continue
        selected.append(idx)
        selected_X.append(x)

    if len(selected) < top_n:
        print(f'  [diversity] Only found {len(selected)}/{top_n} frames '
              f'satisfying radius={diversity_radius:.1f}; '
              f'consider reducing --diversity-radius')
    return np.array(selected)


def _run_active_learning(gp, sigma, times_fs, coords_traj, symbols,
                         top_n, method, basis, dry_run, out, traj_labels=None,
                         X_traj_scaled=None, diversity_radius=0.0):
    """Compute PSI4 dipoles for the top_n highest-uncertainty frames."""
    if X_traj_scaled is None:
        X_traj_scaled = np.zeros((len(sigma), 1))   # fallback: no diversity

    ranked = _greedy_diverse_selection(sigma, X_traj_scaled, top_n, diversity_radius)

    print(f'\n=== Active Learning: computing dipoles for top {top_n} frames ===')
    print(f'  Method: {method}/{basis}')
    if diversity_radius > 0:
        print(f'  Diversity radius: {diversity_radius:.1f} (scaled descriptor units)')
    print(f'  σ range in selection: {sigma[ranked].min():.4f}–{sigma[ranked].max():.4f} D')

    results = []
    if traj_labels is None:
        traj_labels = ['?'] * len(sigma)
    for rank, frame_idx in enumerate(ranked):
        t_ps = times_fs[frame_idx] / 1000.0
        s = sigma[frame_idx]
        coords = coords_traj[frame_idx]
        src = traj_labels[frame_idx]
        print(f'  [{rank+1:3d}/{top_n}]  {src}  frame={frame_idx:5d}  t={t_ps:.3f} ps  '
              f'σ={s:.4f} D  …', end=' ', flush=True)

        if dry_run:
            print('SKIP (--dry-run)')
            results.append({'frame_idx': int(frame_idx), 'traj': src, 'sigma': float(s),
                            'time_ps': float(t_ps), 'dipole_D': None,
                            'coords': coords.tolist()})
            continue

        t0 = time.perf_counter()
        dip = psi4_dipole(symbols, coords, method=method, basis=basis)
        dt = time.perf_counter() - t0

        if dip is None:
            print('PSI4 unavailable')
            continue

        print(f'μ=({dip[0]:+.3f},{dip[1]:+.3f},{dip[2]:+.3f}) D  |μ|={np.linalg.norm(dip):.3f} D  '
              f'({dt:.1f}s)')
        results.append({'frame_idx': int(frame_idx), 'traj': src, 'sigma': float(s),
                        'time_ps': float(t_ps), 'dipole_D': dip.tolist(),
                        'coords': coords.tolist()})

    # Save results
    finished = [r for r in results if r['dipole_D'] is not None]
    if finished:
        new_coords  = np.array([r['coords']    for r in finished])
        new_dipoles = np.array([r['dipole_D']  for r in finished])
        np.savez(out / 'new_dipoles.npz',
                 symbols=np.array(symbols),
                 coordinates=new_coords,
                 energies=np.zeros(len(new_coords)),
                 forces=np.zeros_like(new_coords),
                 dipoles=new_dipoles,
                 metadata=np.array(json.dumps({
                     'method': method, 'basis': basis,
                     'source': 'gp_active_learning', 'ts': time.strftime('%Y%m%d_%H%M%S')
                 })))
        print(f'\n  Saved {len(finished)} new dipoles → {out}/new_dipoles.npz')
        print(f'  Merge command:')
        print(f'    python3 gp_dipole_coverage.py \\')
        print(f'      --training-data <current.npz> \\')
        print(f'      --merge {out}/new_dipoles.npz \\')
        print(f'      --output-dir outputs/mvko_dipoles_gp_round1')

    (out / 'active_learning_log.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
