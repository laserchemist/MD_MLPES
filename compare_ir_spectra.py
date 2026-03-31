#!/usr/bin/env python3
"""
compare_ir_spectra.py — Overlay and compare multiple ML-MD IR spectra.

Usage
-----
  # Two spectra with auto labels from directory names:
  python3 compare_ir_spectra.py \\
      outputs/ir_spectrum_20260331_B3LYP/ \\
      outputs/ir_spectrum_20260331_CASSCF_delta/

  # Explicit labels:
  python3 compare_ir_spectra.py \\
      outputs/ir_spectrum_20260331_B3LYP/:B3LYP \\
      outputs/ir_spectrum_20260331_CASSCF_delta/:"B3LYP+CASSCF_delta" \\
      --title "MVKO 300 K: B3LYP vs CASSCF-corrected"

  # Three spectra + save to specific path:
  python3 compare_ir_spectra.py spec1/ spec2/ spec3/ \\
      --out comparison.png --max-freq 2000

Input format
------------
Each argument is either:
  - A directory containing ir_spectrum.csv (standard ir_md_spectrum.py output)
  - A path directly to a .csv file
  - Either form may be followed by :label  (e.g.  dir/:My label)

The CSV must have columns: frequency_cm-1, intensity (arbitrary units).
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# ── Spectrum loading ──────────────────────────────────────────────────────────

def load_spectrum_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs, intensities) from a spectrum CSV."""
    freqs, intens = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            try:
                freqs.append(float(parts[0]))
                intens.append(float(parts[1]))
            except ValueError:
                continue   # header row
    return np.array(freqs), np.array(intens)


def find_csv(spec_path: str) -> Path:
    """Locate the spectrum CSV given a directory or file path."""
    p = Path(spec_path)
    if p.is_file() and p.suffix == '.csv':
        return p
    if p.is_dir():
        candidates = list(p.glob('ir_spectrum*.csv')) + list(p.glob('spectrum*.csv'))
        if candidates:
            return sorted(candidates)[-1]   # newest
        raise FileNotFoundError(f"No spectrum CSV found in {p}")
    raise FileNotFoundError(f"Not a file or directory: {p}")


def auto_label(spec_path: str) -> str:
    """Make a short label from directory name."""
    p = Path(spec_path)
    name = p.name if p.is_dir() else p.parent.name
    # Remove timestamp suffix like _20260331_133413
    import re
    name = re.sub(r'_\d{8}_\d{6}$', '', name)
    name = name.replace('ir_spectrum_', '').replace('_', ' ').strip()
    return name or str(p)


def parse_spec_arg(arg: str) -> tuple[str, str]:
    """Parse 'path:label' or 'path' → (path, label)."""
    if ':' in arg:
        # split on last colon to allow Windows paths like C:\...
        idx = arg.rfind(':')
        path, label = arg[:idx], arg[idx+1:]
        if label:
            return path, label
    return arg, auto_label(arg)


# ── Smoothing ─────────────────────────────────────────────────────────────────

def smooth_spectrum(freqs, intens, sigma_cm=8.0):
    """Gaussian broadening of a stick/line spectrum."""
    f_out = np.linspace(freqs.min(), freqs.max(), 4000)
    i_out = np.zeros_like(f_out)
    for f, i in zip(freqs, intens):
        i_out += i * np.exp(-0.5 * ((f_out - f) / sigma_cm) ** 2)
    return f_out, i_out


# ── Peak finding ──────────────────────────────────────────────────────────────

def find_peaks(freqs, intens, min_freq=200, threshold_frac=0.05):
    """Return (peak_freqs, peak_intens) above threshold_frac × max."""
    from scipy.signal import find_peaks as _fp
    thresh = intens.max() * threshold_frac
    mask = freqs >= min_freq
    f, i = freqs[mask], intens[mask]
    peaks, _ = _fp(i, height=thresh, distance=20)
    return f[peaks], i[peaks]


# ── Comparison figure ─────────────────────────────────────────────────────────

def plot_comparison(spectra: list[tuple[str, np.ndarray, np.ndarray]],
                    out_path: Path,
                    max_freq: float = 4000.0,
                    min_freq: float = 100.0,
                    sigma_cm: float = 8.0,
                    title: str = 'IR Spectrum Comparison',
                    show_peaks: bool = True,
                    normalise: bool = True):
    """
    spectra: list of (label, freqs, intens)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Colour palette
    COLORS = ['#2166ac', '#d6604d', '#1a9641', '#762a83',
              '#f4a582', '#74add1', '#fdae61', '#a6d96a']

    n = len(spectra)
    has_diff = n == 2   # show difference panel for exactly 2 spectra

    if has_diff:
        fig = plt.figure(figsize=(14, 10))
        ax_main = fig.add_subplot(3, 1, (1, 2))
        ax_diff = fig.add_subplot(3, 1, 3)
    else:
        fig, ax_main = plt.subplots(figsize=(14, 6))
        ax_diff = None

    smoothed = []
    for (label, freqs, intens), col in zip(spectra, COLORS):
        # Restrict to [min_freq, max_freq]
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        f, i = freqs[mask], intens[mask]
        if len(f) == 0:
            smoothed.append((label, np.array([]), np.array([]), col))
            continue

        # Smooth
        f_s, i_s = smooth_spectrum(f, i, sigma_cm=sigma_cm)

        # Normalise to peak = 1
        if normalise and i_s.max() > 0:
            i_s = i_s / i_s.max()

        smoothed.append((label, f_s, i_s, col))
        ax_main.plot(f_s, i_s, lw=1.8, color=col, label=label)

        # Mark peaks
        if show_peaks and len(f_s) > 10:
            try:
                pk_f, pk_i = find_peaks(f_s, i_s, min_freq=min_freq,
                                         threshold_frac=0.08)
                for pf, pi in zip(pk_f, pk_i):
                    ax_main.annotate(f'{pf:.0f}',
                                     xy=(pf, pi), xytext=(0, 6),
                                     textcoords='offset points',
                                     ha='center', fontsize=7, color=col,
                                     arrowprops=dict(arrowstyle='-',
                                                     color=col, lw=0.5))
            except Exception:
                pass

    ax_main.set_xlim(min_freq, max_freq)
    ax_main.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    ax_main.set_ylabel('Intensity (normalised)' if normalise else 'Intensity (arb.)',
                        fontsize=12)
    ax_main.set_title(title, fontsize=13, fontweight='bold')
    ax_main.legend(fontsize=10, loc='upper right')
    ax_main.axhline(0, color='gray', lw=0.6)
    ax_main.set_ylim(bottom=-0.03)

    # ── Difference panel (2-spectrum case) ────────────────────────────────────
    if ax_diff is not None and len(smoothed) == 2:
        lbl0, f0, i0, c0 = smoothed[0]
        lbl1, f1, i1, c1 = smoothed[1]
        if len(f0) > 0 and len(f1) > 0:
            # Interpolate onto common grid
            f_common = np.linspace(max(f0.min(), f1.min()),
                                   min(f0.max(), f1.max()), 3000)
            i0c = np.interp(f_common, f0, i0)
            i1c = np.interp(f_common, f1, i1)
            diff = i1c - i0c
            ax_diff.plot(f_common, diff, lw=1.4, color='#636363')
            ax_diff.fill_between(f_common, 0, diff,
                                 where=diff >= 0, alpha=0.25, color=c1,
                                 label=f'{lbl1} stronger')
            ax_diff.fill_between(f_common, 0, diff,
                                 where=diff < 0,  alpha=0.25, color=c0,
                                 label=f'{lbl0} stronger')
            ax_diff.axhline(0, color='gray', lw=0.8, ls='--')
            ax_diff.set_xlim(min_freq, max_freq)
            ax_diff.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
            ax_diff.set_ylabel(f'Δ ({lbl1} − {lbl0})', fontsize=10)
            ax_diff.set_title('Difference spectrum', fontsize=11)
            ax_diff.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Comparison figure: {out_path}")


# ── CSV summary ───────────────────────────────────────────────────────────────

def print_peak_table(spectra, min_freq=200, max_freq=4000, sigma_cm=8):
    """Print aligned peak-position table for all spectra."""
    print(f"\n  Peak positions (cm⁻¹)  [threshold: 8% of max, σ={sigma_cm} cm⁻¹]")
    print(f"  {'─'*60}")
    for label, freqs, intens in spectra:
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        f, i = freqs[mask], intens[mask]
        if len(f) == 0:
            print(f"  {label}: (no data)")
            continue
        f_s, i_s = smooth_spectrum(f, i, sigma_cm)
        try:
            pk_f, _ = find_peaks(f_s, i_s, min_freq=min_freq, threshold_frac=0.08)
            peaks_str = '  '.join(f'{p:.0f}' for p in sorted(pk_f))
        except Exception:
            peaks_str = '(scipy not available)'
        print(f"  {label}: {peaks_str}")
    print(f"  {'─'*60}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('spectra', nargs='+',
                    help='Spectrum directories or CSV files, optionally with :label suffix')
    ap.add_argument('--out', default=None,
                    help='Output PNG path [default: comparison_<ts>.png in first spectrum dir]')
    ap.add_argument('--title', default='IR Spectrum Comparison',
                    help='Figure title')
    ap.add_argument('--max-freq', type=float, default=4000.0,
                    help='Upper frequency limit cm⁻¹ [default 4000]')
    ap.add_argument('--min-freq', type=float, default=100.0,
                    help='Lower frequency limit cm⁻¹ [default 100]')
    ap.add_argument('--sigma', type=float, default=8.0,
                    help='Gaussian broadening σ in cm⁻¹ [default 8]')
    ap.add_argument('--no-normalise', action='store_true',
                    help='Do not normalise each spectrum to peak = 1')
    ap.add_argument('--no-peaks', action='store_true',
                    help='Suppress peak annotations')
    args = ap.parse_args()

    spectra = []
    first_dir = None
    for arg in args.spectra:
        path_str, label = parse_spec_arg(arg)
        try:
            csv_path = find_csv(path_str)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} — skipping")
            continue
        freqs, intens = load_spectrum_csv(csv_path)
        spectra.append((label, freqs, intens))
        if first_dir is None:
            first_dir = Path(path_str) if Path(path_str).is_dir() else Path(path_str).parent
        print(f"  Loaded: {csv_path}  ({len(freqs)} points)  label='{label}'")

    if not spectra:
        print("No spectra loaded — nothing to compare.")
        sys.exit(1)

    # Output path
    if args.out:
        out_path = Path(args.out)
    else:
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = (first_dir or Path('.')) / f'comparison_{ts}.png'

    plot_comparison(
        spectra,
        out_path,
        max_freq=args.max_freq,
        min_freq=args.min_freq,
        sigma_cm=args.sigma,
        title=args.title,
        show_peaks=not args.no_peaks,
        normalise=not args.no_normalise,
    )

    print_peak_table([(l, f, i) for l, f, i in spectra],
                     min_freq=args.min_freq, max_freq=args.max_freq,
                     sigma_cm=args.sigma)


if __name__ == '__main__':
    main()
