"""
Hot-molecule IR emission spectrum via trajectory-averaged local harmonic intensities.

For each sampled geometry from a high-T MD trajectory:
  1. Compute MACE Hessian → local (anharmonic) NM frequencies
  2. Compute NMDipoleSurface.ir_intensities(coords) → local dipole derivatives
  3. Accumulate Lorentzian peaks at the local frequencies

This captures anharmonic frequency shifts and mode-coupling effects without
the permanent-dipole reorientation problem of the lab-frame ACF approach.

Usage:
  python3 hot_emission_spectrum.py \
      --traj-dir  outputs/ir_mace_anti_cis_2000K \
      --mace-model outputs/mace_wB97X_20260417/mace_model.pt \
      --dipole-model outputs/ir_nm_pes_anti_cis_300K/dipole_surface.pkl \
      --nm-delta-model outputs/casscf_wB97X_nm_grid_20260407_184904/nm_delta_s0_model.pkl \
      --stride 300 \
      --fwhm 30.0 \
      --output-dir outputs/hot_emission_anti_cis_2000K
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# XYZ reader
# ---------------------------------------------------------------------------

def read_xyz_frames(xyz_path: str, n_atoms: int, stride: int = 1) -> list:
    """Read every `stride`-th frame from an XYZ file."""
    frames = []
    lines_per_frame = n_atoms + 2
    with open(xyz_path) as fh:
        raw = fh.readlines()
    n_frames = len(raw) // lines_per_frame
    for i in range(0, n_frames, stride):
        start = i * lines_per_frame
        coords = np.array(
            [[float(x) for x in raw[start + 2 + j].split()[1:4]]
             for j in range(n_atoms)]
        )
        frames.append(coords)
    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Hot-molecule IR emission spectrum from trajectory-averaged local Hessians',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--traj-dir',      required=True,
                        help='Directory containing traj_01.xyz … traj_NN.xyz')
    parser.add_argument('--mace-model',    required=True,
                        help='MACE model .pt for Hessian computation')
    parser.add_argument('--dipole-model',  required=True,
                        help='NMDipoleSurface .pkl for local IR intensities')
    parser.add_argument('--nm-delta-model', default=None,
                        help='Optional NMKRRDeltaModel .pkl (CASSCF δ_S0 correction)')
    parser.add_argument('--stride',        type=int, default=300,
                        help='Sample every STRIDE frames')
    parser.add_argument('--fwhm',          type=float, default=30.0,
                        help='Lorentzian FWHM in cm-1 (use ≥20 for hot broadening)')
    parser.add_argument('--max-freq',      type=float, default=4000.0)
    parser.add_argument('--min-freq',      type=float, default=50.0,
                        help='Ignore imaginary/very-low modes below this threshold')
    parser.add_argument('--output-dir',    default=None)
    parser.add_argument('--ref-harmonic',  default=None,
                        help='Equilibrium geometry .npy for overlaying harmonic spectrum')
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else \
          Path('outputs') / f'hot_emission_{Path(args.traj_dir).name}'
    out.mkdir(parents=True, exist_ok=True)

    # ── Load models ────────────────────────────────────────────────────────
    print('Loading MACE model...')
    from modules.mace_pes import MACEDriver
    driver = MACEDriver(args.mace_model)
    # Keep base driver reference for Hessian (NMDeltaDriver does not expose it)
    hess_driver = driver
    if args.nm_delta_model:
        from ir_md_spectrum import NMDeltaDriver
        driver = NMDeltaDriver(driver, args.nm_delta_model)
        hess_driver = driver._base   # MACE base — delta correction negligible for curvature
        print(f'  + CASSCF delta: {args.nm_delta_model}')

    print('Loading NMDipoleSurface...')
    from modules.nm_pes import load_dipole_surface
    ds = load_dipole_surface(args.dipole_model)
    print(f'  R²={ds.r2_test:.4f}  γ={ds.gamma:.5f}')

    from modules.normal_modes import compute_normal_modes

    n_atoms = len(driver.symbols)

    # ── Find trajectory files ───────────────────────────────────────────────
    traj_dir = Path(args.traj_dir)
    traj_files = sorted(traj_dir.glob('traj_*.xyz'))
    if not traj_files:
        sys.exit(f'No traj_*.xyz files found in {traj_dir}')
    print(f'\nFound {len(traj_files)} trajectories in {traj_dir}')

    # ── Spectrum grid ───────────────────────────────────────────────────────
    nu_grid = np.arange(0.0, args.max_freq + 1.0, 1.0)
    gamma_lor = args.fwhm / 2.0
    spectrum = np.zeros_like(nu_grid)
    n_samples = 0

    # Also track per-mode statistics
    freq_records = []   # (freq_k,) for each sample
    int_records  = []   # (I_k,) for each sample

    t_start = time.time()

    for traj_path in traj_files:
        print(f'\n  {traj_path.name} — reading (stride={args.stride})...')
        frames = read_xyz_frames(str(traj_path), n_atoms, stride=args.stride)
        print(f'    {len(frames)} sampled geometries')

        for i, coords in enumerate(frames):
            # Local NM frequencies via MACE Hessian (base driver, delta negligible)
            try:
                H = hess_driver.analytic_hessian(coords)
                freqs_local, _, _, _ = compute_normal_modes(driver.symbols, H)
            except Exception as e:
                print(f'    Hessian failed at sample {i}: {e}')
                continue

            # Local IR intensities via NMDipoleSurface
            try:
                ints_local = ds.ir_intensities(coords)
            except Exception as e:
                print(f'    ir_intensities failed at sample {i}: {e}')
                continue

            # Skip imaginary/very-low modes
            valid = freqs_local >= args.min_freq
            f_v = freqs_local[valid]
            I_v = ints_local[valid]

            freq_records.append(f_v.copy())
            int_records.append(I_v.copy())

            # Accumulate Lorentzian peaks
            for nu_k, I_k in zip(f_v, I_v):
                spectrum += I_k * (gamma_lor**2 / ((nu_grid - nu_k)**2 + gamma_lor**2))

            n_samples += 1
            if n_samples % 20 == 0:
                elapsed = time.time() - t_start
                rate = elapsed / n_samples
                print(f'    sample {n_samples}  ({rate:.1f} s/sample, '
                      f'{rate * (len(frames) * len(traj_files) // args.stride - n_samples):.0f} s remaining)')

    if n_samples == 0:
        sys.exit('No samples processed.')

    print(f'\nTotal samples: {n_samples}  ({time.time()-t_start:.1f} s)')

    # Normalise
    if spectrum.max() > 0:
        spectrum /= spectrum.max()

    # ── Save spectrum ───────────────────────────────────────────────────────
    spec_path = out / 'ir_hot_emission.csv'
    header = (f'Hot-molecule IR emission spectrum  (trajectory-averaged local harmonic)\n'
              f'# Traj dir: {args.traj_dir}  MACE: {args.mace_model}\n'
              f'# Dipole: {args.dipole_model}\n'
              f'# Stride: {args.stride}  N_samples: {n_samples}  FWHM: {args.fwhm} cm-1\n'
              f'# frequency_cm-1,intensity_normalized')
    np.savetxt(spec_path,
               np.column_stack([nu_grid, spectrum]),
               delimiter=',', header=header, comments='# ')
    print(f'Spectrum saved: {spec_path}')

    # ── Summary statistics ──────────────────────────────────────────────────
    # Per-mode statistics — pad to equal length with NaN so np.array works
    n_vib_all = ds.freqs_vib.shape[0]
    freq_mat = np.full((n_samples, n_vib_all), np.nan)
    int_mat  = np.full((n_samples, n_vib_all), np.nan)
    for s, (fv, iv) in enumerate(zip(freq_records, int_records)):
        freq_mat[s, :len(fv)] = fv
        int_mat[s,  :len(iv)] = iv

    freq_mean = np.nanmean(freq_mat, axis=0)
    freq_std  = np.nanstd(freq_mat,  axis=0)
    int_mean  = np.nanmean(int_mat,  axis=0)
    int_n     = int_mean / np.nanmax(int_mean)

    print('\n--- Local mode statistics (hot trajectory) ---')
    print(f'{"Mode":>4}  {"<ν> (cm-1)":>12}  {"σ_ν":>8}  {"I_rel":>7}')
    print('-' * 40)
    for k in range(n_vib_all):
        if not np.isnan(freq_mean[k]):
            print(f'  {k+1:2d}   {freq_mean[k]:8.1f}     {freq_std[k]:6.1f}    {int_n[k]:.4f}')

    # Optional: overlay equilibrium harmonic spectrum for comparison
    if args.ref_harmonic:
        print(f'\nOverlaying equilibrium harmonic spectrum from {args.ref_harmonic}...')
        eq_coords = np.load(args.ref_harmonic)
        freqs_eq = ds.freqs_vib
        ints_eq  = ds.ir_intensities(eq_coords)
        ints_eq_n = ints_eq / ints_eq.max()
        gamma_eq  = 10.0   # narrower FWHM for harmonic reference
        spectrum_eq = np.zeros_like(nu_grid)
        for nu_k, I_k in zip(freqs_eq, ints_eq_n):
            spectrum_eq += I_k * (gamma_eq**2 / ((nu_grid - nu_k)**2 + gamma_eq**2))
        spectrum_eq /= spectrum_eq.max()
        eq_path = out / 'ir_harmonic_reference.csv'
        np.savetxt(eq_path,
                   np.column_stack([nu_grid, spectrum_eq]),
                   delimiter=',',
                   header='Equilibrium harmonic reference (FWHM=10 cm-1)\n# frequency_cm-1,intensity_normalized',
                   comments='# ')
        print(f'Harmonic reference saved: {eq_path}')

    # Peak table for hot spectrum
    print('\n--- Hot emission spectrum peaks ---')
    for lo, hi in [(100,500),(500,700),(700,900),(900,1200),(1200,1500),(1500,1800),(2800,3400)]:
        mask = (nu_grid >= lo) & (nu_grid < hi)
        if spectrum[mask].max() > 0.02:
            pk = nu_grid[mask][np.argmax(spectrum[mask])]
            mx = spectrum[mask].max()
            print(f'  {lo:4d}-{hi:4d}: max={mx:.4f}  @ {pk:.0f} cm-1')


if __name__ == '__main__':
    main()
