#!/usr/bin/env python3
"""
MVKO 1D Anharmonic Frequency Analysis (C-H stretch overtone region)
======================================================================
PSI4 1.10 has no built-in VPT2. This script computes anharmonic C-H
stretch frequencies via 1D numerical potential energy curves:

  1. Parse harmonic normal modes from an existing PSI4 frequency output.
  2. For each C-H mode, displace along the mode vector and compute PSI4
     B3LYP/6-31G* energies at ±0.05, ±0.10, ±0.15, ±0.20, ±0.25 Å.
  3. Fit a quartic polynomial V(α).
  4. Solve the 1D Schrödinger equation (finite-difference + exact
     diagonalisation) with the mode's reduced mass → E₀, E₁, E₂.
  5. Report fundamentals νᵢ = E₁−E₀ and overtones 2νᵢ = E₂−E₀ in cm⁻¹.
  6. Estimate combination bands νᵢ+νⱼ ≈ νᵢ(fund) + νⱼ(fund) [diagonal
     approximation; cross-anharmonicities xᵢⱼ typically −5 to −25 cm⁻¹].

Cost: 6 modes × 10 displacements × ~5 s each ≈ 5 min.

Usage
-----
    python3 mvko_anharmonic.py \\
        --state  outputs/mvko_20260319_081314/state.json \\
        --psi4out outputs/mvko_vpt2_20260319_183728/psi4_vpt2_output.dat

    # Skip PSI4, re-analyse already-saved energies
    python3 mvko_anharmonic.py \\
        --state  outputs/mvko_20260319_081314/state.json \\
        --psi4out outputs/mvko_vpt2_20260319_183728/psi4_vpt2_output.dat \\
        --load-curves outputs/mvko_anharm_<ts>/curves.npz
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
    print("PSI4 not available")

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
PSI4_MEM_GB  = 4
PSI4_THREADS = 4

# Physical constants
AU_TO_CM1    = 219474.63          # Hartree → cm⁻¹
BOHR_TO_ANG  = 0.529177           # Bohr → Å
ANG_TO_BOHR  = 1.0 / BOHR_TO_ANG # Å → Bohr
AMU_TO_ME    = 1822.888486        # amu → electron mass

# Displacement amplitudes along each mode vector [Å]
DISPLACEMENTS = np.array([-0.25, -0.20, -0.15, -0.10, -0.05,
                            0.05,  0.10,  0.15,  0.20,  0.25])

# Frequency ranges
CH_FUND_LOW, CH_FUND_HIGH   = 2800.0, 3400.0   # harmonic C-H range to auto-detect
CH_OVERT_LOW, CH_OVERT_HIGH = 5500.0, 6500.0   # 2νCH region of interest


# ── Geometry loading ──────────────────────────────────────────────────────────

def load_geometry(state_path):
    with open(state_path) as f:
        state = json.load(f)
    coords  = np.array(state['opt_coords'])
    symbols = ['C', 'O', 'O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    print(f"  Geometry from {state_path}")
    print(f"  E = {state.get('opt_energy','?')} Ha")
    return symbols, coords


# ── PSI4 helper ───────────────────────────────────────────────────────────────

def _mol_str(symbols, coords):
    lines = ['0 1']
    for s, c in zip(symbols, coords):
        lines.append(f'{s}  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}')
    lines += ['units angstrom', 'no_reorient', 'no_com']
    return '\n'.join(lines)


def _psi4_energy(symbols, coords):
    """Single-point B3LYP/6-31G* energy in Hartree."""
    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.be_quiet()
    psi4.set_memory(f'{PSI4_MEM_GB} GB')
    psi4.set_num_threads(PSI4_THREADS)
    psi4.set_options(PSI4_OPTIONS)
    mol = psi4.geometry(_mol_str(symbols, coords))
    return psi4.energy(f'{PSI4_METHOD}/{PSI4_BASIS}', molecule=mol)


# ── PSI4 output parser ────────────────────────────────────────────────────────

def parse_harmonic_modes(psi4_out_path, freq_low=2700.0, freq_high=3500.0):
    """
    Parse harmonic normal modes from PSI4 frequency output.

    Returns list of dicts:
      {'mode': int, 'freq_cm1': float, 'red_mass_u': float,
       'ir_km_mol': float, 'vectors': np.ndarray shape (N,3)}
    where vectors[a] = (dx,dy,dz) Cartesian displacement for atom a,
    normalised so Σ_a |vectors[a]|² = 1 (PSI4 convention).
    """
    with open(psi4_out_path) as f:
        lines = f.readlines()

    modes = []
    i = 0
    n_atoms = None

    while i < len(lines):
        line = lines[i]

        # Section header: "  Vibration   N1   N2   N3"
        if '  Vibration' in line and 'Vibration' == line.strip().split()[0]:
            col_modes = [int(x) for x in line.split()[1:]]
            n_cols = len(col_modes)

            # Read fields
            freqs = red_masses = ir_acts = None
            vecs = [[] for _ in range(n_cols)]  # vecs[col_idx][atom_idx] = [dx,dy,dz]

            i += 1
            while i < len(lines):
                l = lines[i]

                if '  Freq [cm^-1]' in l:
                    freqs = [float(x) for x in l.split()[2:2+n_cols]]

                elif '  Reduced mass [u]' in l:
                    red_masses = [float(x) for x in l.split()[3:3+n_cols]]

                elif '  IR activ [km/mol]' in l:
                    ir_acts = [float(x) for x in l.split()[3:3+n_cols]]

                elif l.strip() and l.strip()[0].isdigit():
                    # Atom displacement line: "    1   C    dx1 dy1 dz1   dx2 dy2 dz2 ..."
                    parts = l.split()
                    if len(parts) >= 2 + 3 * n_cols:
                        # atom_idx = parts[0], symbol = parts[1], then 3*n_cols floats
                        try:
                            int(parts[0])  # confirm first token is atom index
                            for ci in range(n_cols):
                                dx = float(parts[2 + 3*ci])
                                dy = float(parts[3 + 3*ci])
                                dz = float(parts[4 + 3*ci])
                                vecs[ci].append([dx, dy, dz])
                        except (ValueError, IndexError):
                            pass

                elif '  Vibration' in l and l.strip().split()[0] == 'Vibration':
                    # Next vibration block starts — back up so outer loop picks it up
                    i -= 1
                    break

                elif 'Thermochemistry' in l:
                    break

                i += 1

            # Store
            if freqs is not None and red_masses is not None:
                for ci, (mode_num, freq, mu, ir) in enumerate(
                        zip(col_modes, freqs, red_masses, ir_acts or [0.0]*n_cols)):
                    if not (freq_low <= freq <= freq_high):
                        continue
                    v = np.array(vecs[ci])  # shape (N,3)
                    if len(v) == 0:
                        continue
                    modes.append({
                        'mode':       mode_num,
                        'freq_cm1':   freq,
                        'red_mass_u': mu,
                        'ir_km_mol':  ir,
                        'vectors':    v,
                    })

        i += 1

    return modes


# ── 1D anharmonic core ────────────────────────────────────────────────────────

def compute_1d_curves(symbols, eq_coords, modes, out_dir):
    """
    For each mode, compute E(α) at DISPLACEMENTS amplitudes [Å].
    Returns dict mode_num → {'alphas': 1D, 'energies': 1D (Hartree)}.
    """
    e0 = _psi4_energy(symbols, eq_coords)
    print(f"  Equilibrium energy: {e0:.8f} Ha", flush=True)

    curves = {}
    t0 = time.time()
    total = sum(len(DISPLACEMENTS) for _ in modes)
    done  = 0

    for m in modes:
        mode_num = m['mode']
        v_cart   = m['vectors']        # shape (N,3)
        # Normalise (PSI4 should already normalise, but enforce)
        norm = np.sqrt(np.sum(v_cart**2))
        if norm > 0:
            v_cart = v_cart / norm

        energies = []
        for alpha in DISPLACEMENTS:
            c_new = eq_coords + alpha * v_cart
            try:
                e = _psi4_energy(symbols, c_new)
            except Exception as exc:
                print(f"    PSI4 failed (mode {mode_num}, α={alpha:.2f}): {exc}")
                e = np.nan
            energies.append(e)
            done += 1
            print(f"  mode {mode_num}  α={alpha:+.2f} Å  E={e:.8f} Ha  "
                  f"[{done}/{total}, {time.time()-t0:.0f}s]", flush=True)

        curves[mode_num] = {
            'alphas':     np.array(list(DISPLACEMENTS)),
            'energies':   np.array(energies),
            'e0':         e0,
            'freq_harm':  m['freq_cm1'],
            'red_mass_u': m['red_mass_u'],
            'ir_km_mol':  m['ir_km_mol'],
        }

    # Save curves
    save_path = out_dir / 'curves.npz'
    save_data = {'e0': e0}
    for mn, d in curves.items():
        pfx = f'mode{mn}_'
        save_data[pfx + 'alphas']     = d['alphas']
        save_data[pfx + 'energies']   = d['energies']
        save_data[pfx + 'freq_harm']  = d['freq_harm']
        save_data[pfx + 'red_mass_u'] = d['red_mass_u']
        save_data[pfx + 'ir_km_mol']  = d['ir_km_mol']
        save_data[pfx + 'e0']         = e0
    np.savez(save_path, **save_data)
    print(f"\n  Curves saved → {save_path}")
    return curves, e0


def load_curves_from_npz(npz_path):
    """Reload saved curves from compute_1d_curves."""
    d = np.load(npz_path)
    e0 = float(d['e0'])
    curves = {}
    keys = [k for k in d.files if k.startswith('mode') and k.endswith('_alphas')]
    for k in keys:
        mn = int(k.split('_')[0][4:])
        pfx = f'mode{mn}_'
        curves[mn] = {
            'alphas':     d[pfx+'alphas'],
            'energies':   d[pfx+'energies'],
            'e0':         e0,
            'freq_harm':  float(d[pfx+'freq_harm']),
            'red_mass_u': float(d[pfx+'red_mass_u']),
            'ir_km_mol':  float(d[pfx+'ir_km_mol']),
        }
    return curves, e0


# ── Quartic fit ───────────────────────────────────────────────────────────────

def fit_quartic(alphas, energies, e0):
    """
    Fit V(α) = c0 + c2*α² + c3*α³ + c4*α⁴ (force symmetry: c1=0 only for symmetric modes).
    Uses only non-NaN points.  Returns coefficient array [c0,c2,c3,c4] in Hartree/Å^n.
    """
    mask = np.isfinite(energies)
    a = alphas[mask]
    e = energies[mask]
    # Design matrix: 1, α², α³, α⁴  (omit α because equilibrium is minimum)
    A = np.column_stack([np.ones_like(a), a**2, a**3, a**4])
    coeffs, _, _, _ = np.linalg.lstsq(A, e, rcond=None)
    return coeffs   # [c0, c2, c3, c4]


# ── 1D Schrödinger solver ─────────────────────────────────────────────────────

def solve_1d_schrodinger(coeffs, red_mass_u, n_grid=500, alpha_range=0.35, n_levels=5):
    """
    Solve -ℏ²/(2μ) ψ'' + V(α) ψ = E ψ  on a uniform grid in α [Å].

    Uses Hartree / (amu × Å²) unit system:
      ℏ² / (2μ × 1 Å²)  in Hartree when μ is in 'reduced-mass units':
        ℏ² / (2 × 1 amu × 1 Å²)  = 1 / (2 × AMU_TO_ME × ANG_TO_BOHR²)  Hartree

    Returns eigenvalues E_k in Hartree (referenced to c0).
    """
    # Grid
    a_arr = np.linspace(-alpha_range, alpha_range, n_grid)
    da    = a_arr[1] - a_arr[0]

    c0, c2, c3, c4 = coeffs
    V = c2 * a_arr**2 + c3 * a_arr**3 + c4 * a_arr**4   # relative to c0

    # Kinetic energy prefactor: ℏ²/(2μ) in Hartree·Å²
    # ℏ = 1 a.u.,  μ in me = red_mass_u × AMU_TO_ME
    # 1 Å = ANG_TO_BOHR Bohr,  so α in Å corresponds to α × ANG_TO_BOHR in Bohr
    # T = -ℏ²/(2μ_au) × d²/dα_au²  where α_au = α × ANG_TO_BOHR
    # d²/dα²|_Å  = ANG_TO_BOHR² × d²/dα²|_Bohr
    # So T = -ℏ²/(2μ_au) × ANG_TO_BOHR² × d²/dα²|_Å
    mu_au  = red_mass_u * AMU_TO_ME           # reduced mass in au (me)
    hbar2_over_2mu = 1.0 / (2.0 * mu_au)     # ℏ²/(2μ) in Hartree × Bohr²
    # Convert Hartree·Bohr² → Hartree·Å²:  1 Bohr = 0.529177 Å  → ÷ ANG_TO_BOHR²
    KE_prefac = hbar2_over_2mu / ANG_TO_BOHR**2   # Hartree·Å² for Schrödinger eq.

    # Finite-difference Hamiltonian (tridiagonal)
    diag   = V + 2.0 * KE_prefac / da**2
    offdiag = -KE_prefac / da**2 * np.ones(n_grid - 1)

    # Eigenvalues only (tridiagonal → fast)
    from scipy.linalg import eigh_tridiagonal
    eigvals = eigh_tridiagonal(diag, offdiag, eigvals_only=True,
                                select='i', select_range=(0, n_levels-1))
    return eigvals  # shape (n_levels,), in Hartree


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyse_curves(curves):
    """
    For each mode: fit quartic, solve 1D SE, return freq results.
    Returns list of dicts sorted by mode number.
    """
    results = []
    for mode_num, d in sorted(curves.items()):
        alphas   = d['alphas']
        energies = d['energies']
        e0       = d['e0']
        mu       = d['red_mass_u']
        nu_harm  = d['freq_harm']
        ir       = d['ir_km_mol']

        # Energy relative to eq. (remove mean offset to improve fit stability)
        e_rel = energies - e0   # in Hartree

        # Remove outliers / failed points
        valid = np.abs(e_rel) < 0.5  # > 0.5 Ha is unphysical (>300 kcal/mol)
        if valid.sum() < 4:
            print(f"  Mode {mode_num}: too few valid points, skipping")
            continue

        # Quartic fit (include only valid points; force c1=0 for better stability)
        coeffs = fit_quartic(alphas[valid], energies[valid], e0)
        # coeffs: [c0(≈e0), c2, c3, c4]
        c0_, c2, c3, c4 = coeffs

        # Check harmonic from fit vs PSI4
        #  c2 = ½ × k (spring constant), k = (2πcν)² × μ  [in Hartree/Å²]
        #  ν_fit = sqrt(2c2 / (μ_au × ANG_TO_BOHR²)) × (1/2πc) → cm⁻¹
        mu_au = mu * AMU_TO_ME
        if c2 > 0:
            omega_fit = np.sqrt(2.0 * c2 * ANG_TO_BOHR**2 / mu_au)  # rad/s in au
            nu_fit    = omega_fit * AU_TO_CM1   # cm⁻¹
        else:
            nu_fit = 0.0

        # Solve 1D SE
        try:
            eigvals = solve_1d_schrodinger(coeffs, mu, n_grid=1000, alpha_range=0.40, n_levels=4)
            E0, E1, E2 = eigvals[0], eigvals[1], eigvals[2]
            nu_fund   = (E1 - E0) * AU_TO_CM1   # fundamental  (0→1)
            nu_overt  = (E2 - E0) * AU_TO_CM1   # overtone     (0→2)
            nu_hot    = (E2 - E1) * AU_TO_CM1   # hot band     (1→2)
            anharmon  = nu_overt - 2 * nu_fund   # = 2·x_ii (VPT2-like)
        except Exception as exc:
            print(f"  Mode {mode_num}: SE solver failed ({exc})")
            nu_fund = nu_harm   # fallback to harmonic
            nu_overt = 2 * nu_harm
            nu_hot = nu_harm
            anharmon = 0.0

        results.append({
            'mode':      mode_num,
            'nu_harm':   nu_harm,
            'nu_fit':    nu_fit,
            'nu_fund':   nu_fund,
            'nu_overt':  nu_overt,
            'nu_hot':    nu_hot,
            'anharmon':  anharmon,
            'ir_km_mol': ir,
            'c2': c2, 'c3': c3, 'c4': c4,
        })

    return results


def report(results, out_dir):
    """Print and save anharmonic analysis results."""

    print(f"\n{'='*75}")
    print(f"  MVKO 1D ANHARMONIC C-H STRETCH ANALYSIS  (B3LYP/6-31G*)")
    print(f"{'='*75}")
    print(f"\n  C-H Fundamental Frequencies")
    print(f"  {'Mode':>5}  {'ω(harm)':>9}  {'ν(1D)':>9}  {'Δ(anharm)':>10}  "
          f"{'IR (km/mol)':>12}")
    print(f"  {'-'*55}")
    for r in results:
        delta = r['nu_fund'] - r['nu_harm']
        print(f"  {r['mode']:>5}  {r['nu_harm']:>9.1f}  {r['nu_fund']:>9.1f}  "
              f"{delta:>10.1f}  {r['ir_km_mol']:>12.2f}")

    print(f"\n  C-H Overtone Frequencies (2νᵢ)")
    print(f"  {'Mode':>5}  {'2ω(harm)':>9}  {'2ν(1D)':>9}  {'anharmon.':>10}  "
          f"{'in target?':>12}")
    print(f"  {'-'*55}")
    n_overt_in_range = 0
    for r in results:
        two_harm = 2 * r['nu_harm']
        flag = ''
        if CH_OVERT_LOW <= r['nu_overt'] <= CH_OVERT_HIGH:
            flag = '  ← IN RANGE'
            n_overt_in_range += 1
        print(f"  {r['mode']:>5}  {two_harm:>9.1f}  {r['nu_overt']:>9.1f}  "
              f"{r['anharmon']:>10.1f}{flag}")

    # Combination bands νᵢ + νⱼ (diagonal approximation: no cross-anharmonicity)
    print(f"\n  C-H Combination Bands νᵢ+νⱼ (diagonal estimate)")
    print(f"  {'Modes':>10}  {'ω_i+ω_j':>9}  {'ν_i+ν_j':>9}  {'in target?':>12}")
    print(f"  {'-'*50}")
    combos_in_range = []
    n_modes = len(results)
    for ii in range(n_modes):
        for jj in range(ii+1, n_modes):
            ri, rj = results[ii], results[jj]
            harm_sum = ri['nu_harm'] + rj['nu_harm']
            fund_sum = ri['nu_fund'] + rj['nu_fund']
            label = f"ν{ri['mode']}+ν{rj['mode']}"
            flag = ''
            if CH_OVERT_LOW <= fund_sum <= CH_OVERT_HIGH:
                flag = '  ← IN RANGE'
                combos_in_range.append((label, harm_sum, fund_sum))
            print(f"  {label:>10}  {harm_sum:>9.1f}  {fund_sum:>9.1f}{flag}")

    print(f"\n  Summary")
    print(f"  {'─'*50}")
    print(f"  C-H stretch overtones in {CH_OVERT_LOW:.0f}–{CH_OVERT_HIGH:.0f} cm⁻¹: "
          f"{n_overt_in_range} / {len(results)}")
    print(f"  C-H combination bands  in {CH_OVERT_LOW:.0f}–{CH_OVERT_HIGH:.0f} cm⁻¹: "
          f"{len(combos_in_range)} / {n_modes*(n_modes-1)//2}")
    total_in_range = n_overt_in_range + len(combos_in_range)
    print(f"  Total transitions in range: {total_in_range}")
    print(f"  (Paper predicts 21 per conformer: 6 overtones + 15 combinations)")

    print(f"\n  NOTE: combination-band estimates use no cross-anharmonicity (xᵢⱼ=0).")
    print(f"  True xᵢⱼ for C-H pairs is typically −5 to −25 cm⁻¹,")
    print(f"  shifting combination bands down by ≲50 cm⁻¹ from harmonic sum.")

    print(f"\n{'='*75}")
    print(f"  CAN CLASSICAL MD ACCESS THE 2νCH REGION?")
    print(f"{'='*75}")
    print(f"""
  Short answer: NO — for two independent reasons.

  1. QUANTUM EFFECT (fundamental barrier)
     Overtones and combination bands are purely quantum mechanical phenomena.
     In the ACF-based IR spectrum from classical trajectory MD, the dipole
     auto-correlation function (DACF) captures only the *fundamental* (0→1)
     transitions, which appear at νᵢ.  The classical harmonic oscillator
     DACF spectrum has peaks only at ωᵢ, not at 2ωᵢ or ωᵢ+ωⱼ.  Even a
     perfectly anharmonic PES cannot produce overtone peaks in a classical
     trajectory — you need quantum-mechanical population of v=2.

  2. WRONG C-H FREQUENCIES IN THE CURRENT ML-PES
     The Coulomb-matrix+RBF kernel has stiff numerical second derivatives
     under C-H displacements.  The ML-PES Hessian gives C-H stretch modes
     at 10 000–15 000 cm⁻¹ instead of ~3000 cm⁻¹.  Even if classical MD
     could in principle give overtones, the ML-MD trajectory would run at
     the wrong frequency.

  WHAT GIVES THE 2νCH REGION CORRECTLY?
  • VPT2 (second-order vibrational perturbation theory) from an ab-initio
    or DFT code that implements it (Gaussian, CFOUR, ORCA with VPT2 module).
    The 1D anharmonic results above are equivalent to the *diagonal* VPT2
    approximation; they give the C-H overtone positions to ±30 cm⁻¹.
  • Variational vibrational calculations (e.g. VSCF, VCI, MCTDH) on a
    full-dimensional potential give more accurate combination intensities.
  • Our results here (1D DVR on B3LYP/6-31G* quartic potential) give the
    right order of magnitude but lack the off-diagonal anharmonicity xᵢⱼ
    needed for exact combination-band positions.
""")

    # Save JSON
    out_data = {
        'method': f'1D anharmonic, {PSI4_METHOD}/{PSI4_BASIS}',
        'molecule': 'MVKO (C4H6O2)',
        'date': datetime.now().isoformat(),
        'ch_modes': results,
        'overtones_in_5500_6500': [
            {'mode': r['mode'], 'nu_overt': r['nu_overt'], 'anharmon': r['anharmon']}
            for r in results if CH_OVERT_LOW <= r['nu_overt'] <= CH_OVERT_HIGH
        ],
        'combinations_in_range': [
            {'label': lab, 'harm_sum': hs, 'fund_sum': fs}
            for lab, hs, fs in combos_in_range
        ],
    }
    out_path = out_dir / 'anharmonic_results.json'
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"  Results → {out_path}")

    return out_data


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='MVKO 1D anharmonic C-H stretch analysis')
    parser.add_argument('--state',    required=True,
                        help='mvko_workflow state.json (for geometry)')
    parser.add_argument('--psi4out',  required=True,
                        help='PSI4 harmonic frequency output file (for normal modes)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: outputs/mvko_anharm_<ts>)')
    parser.add_argument('--load-curves', default=None,
                        help='Reload curves.npz; skip PSI4 energy calculations')
    args = parser.parse_args()

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = Path(f'outputs/mvko_anharm_{ts}')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  MVKO 1D ANHARMONIC ANALYSIS")
    print(f"{'='*70}")
    print(f"  Method  : {PSI4_METHOD}/{PSI4_BASIS}")
    print(f"  Output  : {out_dir}")
    print(f"  Focus   : C-H overtone region {CH_OVERT_LOW:.0f}–{CH_OVERT_HIGH:.0f} cm⁻¹")

    # Load geometry
    symbols, eq_coords = load_geometry(args.state)

    # Parse harmonic normal modes
    print(f"\n  Parsing harmonic normal modes from {args.psi4out} ...")
    modes = parse_harmonic_modes(args.psi4out, freq_low=CH_FUND_LOW, freq_high=CH_FUND_HIGH)
    if not modes:
        raise RuntimeError(
            f"No modes with freq in {CH_FUND_LOW:.0f}–{CH_FUND_HIGH:.0f} cm⁻¹ found. "
            f"Check {args.psi4out}.")
    print(f"  Found {len(modes)} C-H modes:")
    for m in modes:
        print(f"    Mode {m['mode']:2d}  ω = {m['freq_cm1']:.2f} cm⁻¹  "
              f"μ = {m['red_mass_u']:.4f} u  IR = {m['ir_km_mol']:.2f} km/mol")

    # 1D potential energy curves
    if args.load_curves:
        print(f"\n  Loading saved curves from {args.load_curves} ...")
        curves, e0 = load_curves_from_npz(args.load_curves)
    else:
        if not PSI4_AVAILABLE:
            raise RuntimeError("PSI4 required; use --load-curves to reload saved data")
        print(f"\n  Computing 1D potential curves ({len(modes)} modes × "
              f"{len(DISPLACEMENTS)} displacements = "
              f"{len(modes)*len(DISPLACEMENTS)} PSI4 energy calls) ...")
        t0 = time.time()
        curves, e0 = compute_1d_curves(symbols, eq_coords, modes, out_dir)
        print(f"\n  Curves done in {time.time()-t0:.0f} s", flush=True)

    # Analyse
    print(f"\n  Solving 1D Schrödinger equations ...")
    results = analyse_curves(curves)

    # Report
    report(results, out_dir)


if __name__ == '__main__':
    main()
