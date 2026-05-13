#!/usr/bin/env python3
"""
Plot NM-coordinate time series from a saved trajectory XYZ to diagnose
mode energy injection and ZPE leakage.

For each of the 30 vibrational modes this shows:
  - q_i(t): instantaneous NM coordinate vs time
  - running-RMS envelope (200-step window)
  - reference lines: ±a_ZPE and ±a_T (classical thermal at target T)

Usage:
    python3 plot_nm_mode_sampling.py \
        --traj outputs/ir_spectrum_NM_PES_v5_300K_20260512_204433/traj_01.xyz \
        --model outputs/wB97X_nm_model_v5/mlpes_wB97X_nm.pkl \
        --output outputs/nm_mode_sampling.png \
        [--temp 300] [--stride 10]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent / 'modules'))

FREQ_CONV     = 5140.48        # cm-1 / sqrt(Ha/(Bohr^2 amu))
KB_HARTREE    = 3.1668114e-6   # Ha/K
BOHR_TO_ANG   = 0.529177210903
CM1_TO_HA     = 1.0 / 219474.63  # cm-1 to Hartree (hc factor)


# ── 1. Load NM model ─────────────────────────────────────────────────────────

def load_nm_model(path):
    """Return (eq_coords_ang, U_vib, sqrt_mass, freqs_cm1, symbols)."""
    import pickle
    with open(path, 'rb') as fh:
        state = pickle.load(fh)
    if isinstance(state, dict):
        eq  = state['eq_coords_ang']
        U   = state['U_vib']
        sm  = state['sqrt_mass']
        fr  = state['freqs_vib']
        sym = list(state['symbols'])
    else:  # legacy object pickle
        eq  = state.eq_coords_ang
        U   = state.U_vib
        sm  = state.sqrt_mass
        fr  = state.freqs_vib
        sym = list(state.symbols)
    return eq, U, sm, fr, sym


# ── 2. Parse XYZ trajectory ───────────────────────────────────────────────────

def parse_xyz(path, stride=1):
    """Return (times_fs, coords_ang) arrays."""
    lines = Path(path).read_text().splitlines()
    n_atoms = int(lines[0].strip())
    step = n_atoms + 2
    n_frames = len(lines) // step

    times, coords = [], []
    for i in range(0, n_frames, stride):
        base = i * step
        comment = lines[base + 1]
        # Parse time from comment: "Frame=0  time=0.500fs  ..."
        t = 0.0
        for tok in comment.split():
            if tok.startswith('time=') and tok.endswith('fs'):
                t = float(tok[5:-2])
        times.append(t)
        frame = []
        for j in range(n_atoms):
            parts = lines[base + 2 + j].split()
            frame.append([float(parts[1]), float(parts[2]), float(parts[3])])
        coords.append(frame)

    return np.array(times), np.array(coords)


# ── 3. Project onto NM coordinates ───────────────────────────────────────────

def project_nm(coords_ang, eq_ang, U_vib, sqrt_mass):
    """
    coords_ang : (N_frames, N_atoms, 3)
    returns    : (N_frames, n_vib)  in sqrt(amu)*Bohr units
    """
    dR_ang  = coords_ang - eq_ang[None, :, :]          # (N, Natom, 3)
    dR_bohr = dR_ang / BOHR_TO_ANG
    dR_mw   = dR_bohr.reshape(len(coords_ang), -1) * sqrt_mass[None, :]  # (N, 3Natom)
    return dR_mw @ U_vib                               # (N, n_vib)


# ── 4. Amplitude references ───────────────────────────────────────────────────

def mode_amplitudes(freqs_cm1, T_K):
    """
    Return (a_zpe, a_thermal) in sqrt(amu)*Bohr.

    From ½ freq_nu² q² = E at the classical turning point:
      a_ZPE : E = ZPE = ½ℏω = ½ freq_cm1 * CM1_TO_HA
              → a_ZPE = FREQ_CONV / sqrt(219474.63 * freq_cm1)
      a_T   : E = ½ k_B T  (equipartition, one quadratic DOF)
              → a_T = sqrt(KB * T) / freq_nu
    """
    freq_nu = freqs_cm1 / FREQ_CONV
    # ZPE turning-point amplitude (correct physical formula)
    a_zpe   = FREQ_CONV / np.sqrt(219474.63 * freqs_cm1)
    # Classical thermal amplitude at T_K
    a_T     = np.sqrt(KB_HARTREE * T_K) / freq_nu
    return a_zpe, a_T


# ── 5. Running RMS ────────────────────────────────────────────────────────────

def running_rms(x, window):
    """Centred running RMS with given window size."""
    n = len(x)
    half = window // 2
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.sqrt(np.mean(x[lo:hi] ** 2))
    return out


# ── 6. Main plot ──────────────────────────────────────────────────────────────

def plot_mode_sampling(q_traj, times_ps, freqs_cm1, a_zpe, a_T, output_path,
                       rms_window=200, T_K=300):
    """
    q_traj   : (N_frames, n_vib)
    times_ps : (N_frames,)
    """
    n_vib = q_traj.shape[1]
    ncols = 6
    nrows = 5   # 30 modes in 5×6 grid

    fig = plt.figure(figsize=(22, 16))
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.35)

    cmap   = plt.cm.coolwarm
    t_max  = times_ps[-1]

    for i in range(n_vib):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])

        q   = q_traj[:, i]
        rms = running_rms(q, rms_window)

        # Thin trace
        ax.plot(times_ps, q,   color='#aaaaaa', lw=0.3, alpha=0.6, zorder=1)
        # RMS envelope
        ax.plot(times_ps, rms, color='#1f77b4', lw=1.0, zorder=2, label='RMS')

        # Reference amplitudes
        a_hot = np.sqrt(KB_HARTREE * 3000.0) / (freqs_cm1[i] / FREQ_CONV)
        for amp, col_ref, ls, lbl in [
            (a_zpe[i], '#d62728', '--', 'ZPE'),
            (a_T[i],   '#2ca02c', ':',  f'{T_K}K'),
            (a_hot,    '#ff7f0e', '-.',  '3000K'),
        ]:
            ax.axhline( amp, color=col_ref, lw=0.9, ls=ls)
            ax.axhline(-amp, color=col_ref, lw=0.9, ls=ls)

        # Shade above ZPE (good sampling)
        ax.axhspan( a_zpe[i],  4,  alpha=0.06, color='#d62728')
        ax.axhspan(-4, -a_zpe[i],  alpha=0.06, color='#d62728')

        # Mean RMS (text)
        mean_rms = np.mean(rms)
        frac = mean_rms / a_zpe[i]
        color_txt = '#d62728' if frac >= 0.8 else ('#ff7f0e' if frac >= 0.4 else '#888888')
        ax.set_title(f'Mode {i+1}  {freqs_cm1[i]:.0f} cm⁻¹\n'
                     f'⟨rms⟩/a_ZPE={frac:.2f}',
                     fontsize=6.5, color=color_txt, pad=2)

        ax.set_xlim(0, t_max)
        ax.set_ylim(-max(a_zpe[i] * 3.0, q.max() * 1.1),
                     max(a_zpe[i] * 3.0, q.max() * 1.1))
        ax.tick_params(labelsize=5)
        if row == nrows - 1:
            ax.set_xlabel('t (ps)', fontsize=5)
        if col == 0:
            ax.set_ylabel('q (√amu·Bohr)', fontsize=5)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='#1f77b4', lw=1.2,           label='running RMS'),
        Line2D([0], [0], color='#d62728', lw=0.9, ls='--',  label='± a_ZPE (½ℏω turning pt)'),
        Line2D([0], [0], color='#2ca02c', lw=0.9, ls=':',   label=f'± a_{T_K}K classical'),
        Line2D([0], [0], color='#ff7f0e', lw=0.9, ls='-.',  label='± a_3000K (ozonolysis)'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=3,
               fontsize=8, framealpha=0.85,
               bbox_to_anchor=(0.5, 1.01))

    fig.suptitle('NM-coordinate sampling: v5 trajectory 1   (ZPE-initialized, τ=200 fs, T=300 K)',
                 fontsize=10, y=1.03)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')


# ── 7. Summary heatmap ────────────────────────────────────────────────────────

def plot_heatmap(q_traj, times_ps, freqs_cm1, a_zpe, a_T, output_path,
                 T_K=300, block_ps=0.5):
    """
    Block-averaged heatmap: colour = ⟨|q_i|⟩ / a_ZPE(i) over block_ps-ps windows.
    """
    dt_ps  = times_ps[1] - times_ps[0] if len(times_ps) > 1 else 0.5e-3
    blk    = max(1, int(block_ps / dt_ps))
    n_blk  = len(times_ps) // blk
    n_vib  = q_traj.shape[1]

    heat = np.zeros((n_vib, n_blk))
    t_blk = np.empty(n_blk)
    for b in range(n_blk):
        chunk = q_traj[b * blk : (b + 1) * blk, :]
        heat[:, b] = np.sqrt(np.mean(chunk ** 2, axis=0)) / a_zpe
        t_blk[b]   = times_ps[b * blk]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                             gridspec_kw={'width_ratios': [4, 1]})
    ax, ax2 = axes

    im = ax.imshow(heat, aspect='auto', origin='lower',
                   extent=[t_blk[0], t_blk[-1], 0.5, n_vib + 0.5],
                   vmin=0, vmax=2.0, cmap='inferno')
    ax.set_xlabel('Time (ps)', fontsize=11)
    ax.set_ylabel('Mode index', fontsize=11)
    ax.set_yticks(range(1, n_vib + 1))
    ax.set_yticklabels([f'{i+1}: {f:.0f}' for i, f in enumerate(freqs_cm1)],
                       fontsize=6)
    ax.axhline(n_vib - 5.5, color='white', lw=0.8, ls='--', alpha=0.6)  # C-H boundary
    ax.text(t_blk[-1] * 0.98, n_vib - 4.8, 'C-H', color='white',
            fontsize=7, ha='right', va='bottom')
    cb = fig.colorbar(im, ax=ax, label='⟨rms(q)⟩ / a_ZPE')

    # Right panel: time-averaged ⟨rms⟩/a_ZPE per mode as a bar chart
    mean_frac = heat.mean(axis=1)
    colors = ['#d62728' if f >= 0.8 else ('#ff7f0e' if f >= 0.4 else '#aaaaaa')
              for f in mean_frac]
    ax2.barh(range(1, n_vib + 1), mean_frac, color=colors, height=0.7)
    ax2.axvline(1.0, color='#d62728', lw=1.2, ls='--', label='ZPE level')
    ax2.axvline(np.sqrt(KB_HARTREE * T_K / (a_zpe * freqs_cm1 / FREQ_CONV) ** 2).mean(),
                color='#2ca02c', lw=1.0, ls=':', label=f'{T_K}K classical')
    ax2.set_xlabel('⟨rms⟩ / a_ZPE', fontsize=9)
    ax2.set_yticks(range(1, n_vib + 1))
    ax2.set_yticklabels([f'{i+1}' for i in range(n_vib)], fontsize=6)
    ax2.legend(fontsize=7)
    ax2.set_xlim(0, max(2.0, mean_frac.max() * 1.1))

    fig.suptitle(f'NM mode energy injection   v5 traj-1   (ZPE-init, τ=200 fs, T={T_K} K)\n'
                 f'Red bars ≥ ZPE level · Orange 40–80% ZPE · Grey < 40% ZPE',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj',   required=True)
    ap.add_argument('--model',  required=True)
    ap.add_argument('--output', default='outputs/nm_mode_sampling.png')
    ap.add_argument('--temp',   type=float, default=300.0)
    ap.add_argument('--stride', type=int,   default=5,
                    help='Read every N-th frame (default 5 → 6000 pts from 30000)')
    ap.add_argument('--rms-window', type=int, default=100,
                    help='Running-RMS window in (strided) frames')
    args = ap.parse_args()

    print(f'Loading NM model: {args.model}')
    eq, U, sm, freqs, syms = load_nm_model(args.model)
    print(f'  n_vib={len(freqs)}, freqs {freqs[0]:.1f}–{freqs[-1]:.1f} cm⁻¹')

    print(f'Parsing trajectory: {args.traj}  (stride={args.stride})')
    times_fs, coords = parse_xyz(args.traj, stride=args.stride)
    times_ps = times_fs / 1000.0
    print(f'  {len(times_ps)} frames  ({times_ps[0]:.3f}–{times_ps[-1]:.3f} ps)')

    print('Projecting onto NM coordinates …')
    q = project_nm(coords, eq, U, sm)
    print(f'  q shape: {q.shape}  |q|_max={np.abs(q).max():.3f}')

    a_zpe, a_T = mode_amplitudes(freqs, args.temp)

    print('\nMode-by-mode summary:')
    print(f'{"Mode":>5} {"freq":>8} {"a_ZPE":>8} {"a_T":>8} {"rms(q)":>8} {"rms/a_ZPE":>10}')
    for i in range(len(freqs)):
        rms_i = np.sqrt(np.mean(q[:, i] ** 2))
        print(f'{i+1:5d} {freqs[i]:8.1f} {a_zpe[i]:8.4f} {a_T[i]:8.4f} '
              f'{rms_i:8.4f} {rms_i/a_zpe[i]:10.3f}')

    base = Path(args.output).with_suffix('')
    plot_mode_sampling(q, times_ps, freqs, a_zpe, a_T,
                       str(base) + '_timeseries.png',
                       rms_window=args.rms_window, T_K=args.temp)
    plot_heatmap(q, times_ps, freqs, a_zpe, a_T,
                 str(base) + '_heatmap.png',
                 T_K=args.temp)


if __name__ == '__main__':
    main()
