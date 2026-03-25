#!/usr/bin/env python3
"""
MVKO VPT2 Anharmonic Frequency Analysis
========================================
Computes anharmonic vibrational frequencies and IR intensities for MVKO
(methyl vinyl ketone oxide, C₄H₆O₂) using PSI4 second-order vibrational
perturbation theory (VPT2) at B3LYP/6-31G*.

VPT2 gives:
  - Harmonic fundamentals ωᵢ
  - Anharmonic fundamentals νᵢ
  - Overtones 2νᵢ (important for C-H stretch overtone region 5750–6300 cm⁻¹)
  - Combination bands νᵢ + νⱼ
  - Anharmonicity constants Xᵢⱼ

Cost: ~36 Hessian evaluations (one per displaced coordinate) × ~108 s each
       ≈ 1–2 hours at B3LYP/6-31G* for MVKO (12 atoms).

Usage
-----
    # Using PSI4-optimised geometry from state file
    python3 mvko_vpt2.py --state outputs/mvko_20260319_081314/state.json

    # Using an XYZ file
    python3 mvko_vpt2.py --xyz optimized_mvko.xyz

    # Restart from PSI4 output file (VPT2 already ran)
    python3 mvko_vpt2.py --state ... --output-dir outputs/mvko_vpt2_<ts>/

References
----------
  Kurtén et al. JPCL 2020 / Lester group papers on MVKO IR overtones.
  PSI4 VPT2: King et al. JCTC 2022 (psi4/psi4#2398).
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── PSI4 ──────────────────────────────────────────────────────────────────────
try:
    import psi4
    PSI4_AVAILABLE = True
    print(f"PSI4 {psi4.__version__} available")
except ImportError:
    PSI4_AVAILABLE = False
    print("PSI4 not available — cannot run VPT2")

PSI4_METHOD  = 'b3lyp'
PSI4_BASIS   = '6-31G*'
PSI4_OPTIONS = {
    'basis':         PSI4_BASIS,
    'scf_type':      'df',
    'reference':     'rhf',
    'maxiter':       200,
    'e_convergence': 1e-7,
    'd_convergence': 1e-7,
}
PSI4_MEM_GB  = 8    # VPT2 is memory-intensive; bump above normal
PSI4_THREADS = 4

# C-H stretch overtone region of interest
CH_OVERTONE_LOW  = 5500.0   # cm⁻¹
CH_OVERTONE_HIGH = 6500.0   # cm⁻¹
CH_FUNDAMENTAL_LOW  = 2700.0
CH_FUNDAMENTAL_HIGH = 3300.0


def _mol_str(symbols, coords, charge=0, mult=1):
    """Build a PSI4 geometry block (Angstrom, fixed orientation)."""
    lines = [f'{charge} {mult}']
    for s, c in zip(symbols, coords):
        lines.append(f'{s}  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}')
    lines += ['units angstrom', 'no_reorient', 'no_com']
    return '\n'.join(lines)


def load_geometry_from_state(state_path):
    """Load optimised geometry from mvko_workflow state.json."""
    with open(state_path) as f:
        state = json.load(f)
    coords = np.array(state['opt_coords'])
    # MVKO atom ordering: C O O C C C H H H H H H
    symbols = ['C', 'O', 'O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    print(f"  Loaded PSI4-optimised geometry from {state_path}")
    print(f"  Energy: {state.get('opt_energy', 'unknown')} Ha")
    return symbols, coords


def load_geometry_from_xyz(xyz_path):
    """Load geometry from a plain XYZ file (first frame)."""
    with open(xyz_path) as f:
        n = int(f.readline())
        f.readline()  # comment
        symbols, coords = [], []
        for _ in range(n):
            parts = f.readline().split()
            symbols.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
    print(f"  Loaded geometry from {xyz_path}  ({n} atoms)")
    return symbols, np.array(coords)


def run_vpt2(symbols, coords, out_dir):
    """
    Run PSI4 VPT2 anharmonic frequency analysis.

    PSI4 1.10 computes VPT2 via finite differences of the analytical Hessian:
      - Hessian at equilibrium geometry
      - Hessian at ±δ along each mass-weighted normal coordinate
        (3N-6 = 30 displaced geometries for MVKO × 2 = 60 Hessians)
    This gives cubic and quartic force constants → VPT2 corrections.

    Returns dict with harmonic and anharmonic frequencies, overtones, combos.
    """
    psi4.core.clean_options()
    psi4.core.clean()
    # VPT2 output must go to a real file (not /dev/null) so we can parse it
    psi4_out = str(out_dir / 'psi4_vpt2_output.dat')
    psi4.core.set_output_file(psi4_out, False)
    psi4.set_memory(f'{PSI4_MEM_GB} GB')
    psi4.set_num_threads(PSI4_THREADS)
    psi4.set_options(PSI4_OPTIONS)
    # VPT2-specific options
    psi4.set_options({
        'normal_modes_write': True,
    })

    mol = psi4.geometry(_mol_str(symbols, coords))
    mol.update_geometry()

    print(f"  Running PSI4 VPT2 ({PSI4_METHOD}/{PSI4_BASIS}) ...")
    print(f"  PSI4 output → {psi4_out}")
    print(f"  Estimated cost: ~60 Hessian evals × ~2 min each ≈ 2 hrs")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    t0 = time.time()
    # psi4.frequency with anharmonic=True triggers VPT2
    freq_wfn = psi4.frequency(f'{PSI4_METHOD}/{PSI4_BASIS}',
                               molecule=mol,
                               return_wfn=True,
                               anharmonic=True)
    elapsed = time.time() - t0
    print(f"  VPT2 completed in {elapsed:.0f} s ({elapsed/3600:.2f} h)",
          flush=True)

    # freq_wfn is (freq_result, wfn) when return_wfn=True
    if isinstance(freq_wfn, tuple):
        _, wfn = freq_wfn
    else:
        wfn = freq_wfn

    return wfn, psi4_out, elapsed


def parse_vpt2_output(psi4_out_path):
    """
    Parse PSI4 VPT2 output file to extract:
      - Harmonic fundamentals
      - Anharmonic fundamentals
      - Overtones and combination bands
      - IR intensities

    PSI4 VPT2 output format (psi4 >= 1.5):
      'Fundamental Bands'  table: mode, harmonic ω, VPT2 ν, intensity
      'Overtones and Combination Bands' table
    """
    with open(psi4_out_path) as f:
        text = f.read()

    results = {
        'harmonics':     [],   # (mode, freq_cm1, intensity_kmmol)
        'fundamentals':  [],   # (mode, harm_cm1, anharm_cm1, intensity)
        'overtones':     [],   # (mode, 2*harm, vpt2_freq, intensity)
        'combinations':  [],   # (mode_i, mode_j, harm_sum, vpt2_freq, intensity)
    }

    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]

        # --- Harmonic section ---
        if 'Harmonic Vibrational Analysis' in line or 'HARMONIC' in line.upper():
            # Look for frequency table entries like:
            # Freq [cm^-1]   IR activ [km/mol]
            pass

        # --- Fundamental bands (VPT2) ---
        if 'Fundamental Bands' in line and '=' not in line:
            i += 1
            while i < len(lines) and lines[i].strip():
                parts = lines[i].split()
                if len(parts) >= 4:
                    try:
                        mode = int(parts[0])
                        harm = float(parts[1])
                        anharm = float(parts[2])
                        intens = float(parts[3]) if len(parts) > 3 else 0.0
                        results['fundamentals'].append((mode, harm, anharm, intens))
                    except (ValueError, IndexError):
                        pass
                i += 1
            continue

        # --- Overtones (2νᵢ) ---
        if 'Overtones' in line and 'Combination' not in line:
            i += 1
            while i < len(lines) and lines[i].strip():
                parts = lines[i].split()
                if len(parts) >= 3:
                    try:
                        mode = int(parts[0])
                        harm2 = float(parts[1])   # 2×harmonic
                        vpt2  = float(parts[2])   # VPT2 overtone
                        intens = float(parts[3]) if len(parts) > 3 else 0.0
                        results['overtones'].append((mode, harm2, vpt2, intens))
                    except (ValueError, IndexError):
                        pass
                i += 1
            continue

        # --- Combination bands (νᵢ + νⱼ) ---
        if 'Combination Bands' in line:
            i += 1
            while i < len(lines) and lines[i].strip():
                parts = lines[i].split()
                if len(parts) >= 4:
                    try:
                        mi = int(parts[0])
                        mj = int(parts[1])
                        harm_sum = float(parts[2])
                        vpt2 = float(parts[3])
                        intens = float(parts[4]) if len(parts) > 4 else 0.0
                        results['combinations'].append((mi, mj, harm_sum, vpt2, intens))
                    except (ValueError, IndexError):
                        pass
                i += 1
            continue

        i += 1

    return results


def report_ch_overtone_region(results, output_path):
    """Print and save all VPT2 transitions in the 2νCH region (5500-6500 cm⁻¹)."""

    all_transitions = []

    # Fundamentals
    for mode, harm, anharm, intens in results['fundamentals']:
        if CH_FUNDAMENTAL_LOW <= anharm <= CH_FUNDAMENTAL_HIGH:
            all_transitions.append({
                'type': f'ν{mode}', 'harmonic_cm1': harm,
                'vpt2_cm1': anharm, 'intensity': intens, 'ch_stretch': True
            })

    # Overtones
    for mode, harm2, vpt2, intens in results['overtones']:
        if CH_OVERTONE_LOW <= vpt2 <= CH_OVERTONE_HIGH:
            all_transitions.append({
                'type': f'2ν{mode}', 'harmonic_cm1': harm2,
                'vpt2_cm1': vpt2, 'intensity': intens, 'ch_stretch': True
            })

    # Combination bands
    for mi, mj, harm_sum, vpt2, intens in results['combinations']:
        if CH_OVERTONE_LOW <= vpt2 <= CH_OVERTONE_HIGH:
            all_transitions.append({
                'type': f'ν{mi}+ν{mj}', 'harmonic_cm1': harm_sum,
                'vpt2_cm1': vpt2, 'intensity': intens, 'ch_stretch': True
            })

    all_transitions.sort(key=lambda x: x['vpt2_cm1'])

    print(f"\n{'='*70}")
    print(f"  C-H STRETCH OVERTONE REGION  ({CH_OVERTONE_LOW:.0f}–{CH_OVERTONE_HIGH:.0f} cm⁻¹)")
    print(f"{'='*70}")
    print(f"  {'Transition':>12}  {'Harmonic':>10}  {'VPT2 (cm⁻¹)':>12}  "
          f"{'Intensity':>12}")
    print(f"  {'-'*55}")
    for t in all_transitions:
        print(f"  {t['type']:>12}  {t['harmonic_cm1']:>10.1f}  "
              f"{t['vpt2_cm1']:>12.1f}  {t['intensity']:>12.4f}")
    if not all_transitions:
        print("  (no transitions found — check PSI4 output for parsing errors)")

    print(f"\n  Total transitions in region: {len(all_transitions)}")
    print(f"  (Paper expects 21 per conformer: 6 overtones + 15 combinations)")

    # Full summary of all fundamentals
    print(f"\n{'─'*70}")
    print(f"  ALL VPT2 ANHARMONIC FUNDAMENTALS")
    print(f"{'─'*70}")
    print(f"  {'Mode':>5}  {'Harmonic':>10}  {'Anharmonic':>12}  "
          f"{'ΔX (cm⁻¹)':>10}  {'Intensity':>12}")
    print(f"  {'-'*55}")
    for mode, harm, anharm, intens in results['fundamentals']:
        delta = anharm - harm
        marker = ' ←CH' if CH_FUNDAMENTAL_LOW <= anharm <= CH_FUNDAMENTAL_HIGH else ''
        print(f"  {mode:>5}  {harm:>10.1f}  {anharm:>12.1f}  "
              f"{delta:>10.1f}  {intens:>12.4f}{marker}")

    # Save JSON
    import json as _json
    out_data = {
        'method': f'{PSI4_METHOD}/{PSI4_BASIS} VPT2',
        'molecule': 'MVKO (C4H6O2)',
        'date': datetime.now().isoformat(),
        'ch_overtone_region': all_transitions,
        'all_fundamentals': [
            {'mode': m, 'harmonic_cm1': h, 'vpt2_cm1': a, 'intensity': i}
            for m, h, a, i in results['fundamentals']
        ],
        'all_overtones': [
            {'mode': m, 'harmonic_cm1': h2, 'vpt2_cm1': v, 'intensity': i}
            for m, h2, v, i in results['overtones']
        ],
        'n_combinations': len(results['combinations']),
    }
    with open(output_path, 'w') as f:
        _json.dump(out_data, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    return all_transitions


def main():
    parser = argparse.ArgumentParser(
        description='MVKO VPT2 anharmonic frequencies (B3LYP/6-31G*)')
    geo_group = parser.add_mutually_exclusive_group(required=True)
    geo_group.add_argument('--state', help='Path to mvko_workflow state.json')
    geo_group.add_argument('--xyz',   help='Path to XYZ geometry file')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: outputs/mvko_vpt2_<ts>)')
    parser.add_argument('--parse-only', default=None,
                        help='Skip PSI4, parse existing output file at this path')
    args = parser.parse_args()

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = Path(f'outputs/mvko_vpt2_{ts}')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  MVKO VPT2 ANHARMONIC FREQUENCY ANALYSIS")
    print(f"{'='*70}")
    print(f"  Method  : {PSI4_METHOD}/{PSI4_BASIS} VPT2")
    print(f"  Output  : {out_dir}")
    print(f"  Focus   : C-H overtone region {CH_OVERTONE_LOW:.0f}–{CH_OVERTONE_HIGH:.0f} cm⁻¹")

    # Load geometry
    if args.state:
        symbols, coords = load_geometry_from_state(args.state)
    else:
        symbols, coords = load_geometry_from_xyz(args.xyz)

    # Parse-only mode (PSI4 already ran)
    if args.parse_only:
        print(f"\n  Parsing existing PSI4 output: {args.parse_only}")
        results = parse_vpt2_output(args.parse_only)
        report_ch_overtone_region(results, out_dir / 'vpt2_results.json')
        return

    if not PSI4_AVAILABLE:
        raise RuntimeError("PSI4 required for VPT2 calculation")

    # Run VPT2
    wfn, psi4_out, elapsed = run_vpt2(symbols, coords, out_dir)

    # Parse output
    print(f"\n  Parsing VPT2 output ...", flush=True)
    results = parse_vpt2_output(psi4_out)

    if not results['fundamentals']:
        print(f"\n  WARNING: No fundamentals parsed from PSI4 output.")
        print(f"  Check {psi4_out} for VPT2 tables.")
        print(f"  Try: python3 mvko_vpt2.py --state {args.state} "
              f"--parse-only {psi4_out}")
        return

    # Report
    transitions = report_ch_overtone_region(results, out_dir / 'vpt2_results.json')

    print(f"\n{'='*70}")
    print(f"  VPT2 COMPLETE")
    print(f"{'='*70}")
    print(f"  PSI4 output  : {psi4_out}")
    print(f"  Results JSON : {out_dir}/vpt2_results.json")
    print(f"  Wall time    : {elapsed/3600:.2f} h")
    print(f"  C-H overtone transitions: {len(transitions)}")


if __name__ == '__main__':
    main()
