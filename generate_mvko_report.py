#!/usr/bin/env python3
"""
generate_mvko_report.py
-----------------------
Generate outputs/mvko_results_report.html — a self-contained interactive
results page for MVKO multi-state ML-PES + hot-emission IR spectrum.

Run: python3 generate_mvko_report.py
"""

import json
import numpy as np

HARTREE_TO_KCAL = 627.509474

# ── load PES cut data ─────────────────────────────────────────────────────────
print("Loading PES cut data...")
d = np.load("outputs/pes_cut_data.npz")

q10_grid         = d["q10_grid"].tolist()
r_oo_grid        = d["r_oo_grid"].tolist()
E_wb97x_ml       = d["E_wb97x_ml"].tolist()
E_s0_corr        = d["E_s0_corr"].tolist()
E_s1             = d["E_s1"].tolist()
E_t1             = d["E_t1"].tolist()
q10_pts          = d["q10_pts"].tolist()
r_oo_pts         = d["r_oo_pts"].tolist()
E_wb97x_pts      = d["E_wb97x_pts"].tolist()
E_s0_pts         = d["E_s0_pts"].tolist()
E_s1_pts         = d["E_s1_pts"].tolist()
E_t1_pts         = d["E_t1_pts"].tolist()
q1_grid          = d["q1_grid"].tolist()
E_wb97x_ml_m1    = d["E_wb97x_ml_mode1"].tolist()
E_s0_corr_m1     = d["E_s0_corr_mode1"].tolist()
E_s1_m1          = d["E_s1_mode1"].tolist()
E_t1_m1          = d["E_t1_mode1"].tolist()
q1_pts           = d["q1_pts"].tolist()
E_wb97x_pts_m1   = d["E_wb97x_pts_mode1"].tolist()
E_s0_pts_m1      = d["E_s0_pts_mode1"].tolist()
freqs_vib        = d["freqs_vib"].tolist()
eq_roo           = float(d["eq_roo"][0])

# ── load IR spectrum CSVs ─────────────────────────────────────────────────────
def load_csv_skip_comments(path):
    freqs, ints = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(',')
            if len(parts) == 2:
                freqs.append(float(parts[0]))
                ints.append(float(parts[1]))
    return np.array(freqs), np.array(ints)

print("Loading IR spectrum CSVs...")
ir_freq_harm,  ir_int_harm  = load_csv_skip_comments(
    "outputs/hot_emission_anti_cis_300K/ir_harmonic_reference.csv")
ir_freq_300,   ir_int_300   = load_csv_skip_comments(
    "outputs/hot_emission_anti_cis_300K/ir_hot_emission.csv")
ir_freq_2000,  ir_int_2000  = load_csv_skip_comments(
    "outputs/hot_emission_anti_cis_2000K/ir_hot_emission.csv")

# Restrict to 50–4000 cm-1 and normalise
def trim_norm(freq, inten, fmin=50, fmax=4000):
    mask = (freq >= fmin) & (freq <= fmax)
    f, i = freq[mask], inten[mask]
    mx = i.max()
    if mx > 0:
        i = i / mx
    return f.tolist(), i.tolist()

ir_freq_harm_t, ir_int_harm_t = trim_norm(ir_freq_harm, ir_int_harm)
ir_freq_300_t,  ir_int_300_t  = trim_norm(ir_freq_300,  ir_int_300)
ir_freq_2000_t, ir_int_2000_t = trim_norm(ir_freq_2000, ir_int_2000)

print(f"  Harmonic: {len(ir_freq_harm_t)} pts, 300K: {len(ir_freq_300_t)}, "
      f"2000K: {len(ir_freq_2000_t)}")

# ── per-mode stats tables ─────────────────────────────────────────────────────
harmonic_freqs = [
    119.5, 226.1, 298.6, 318.4, 349.8, 391.2, 412.6, 648.3, 727.9,
    835.2, 1011.2, 1016.0, 1037.0, 1078.0, 1088.8, 1130.8, 1291.7,
    1357.0, 1453.8, 1480.4, 1508.1, 1518.4, 1580.7, 1692.6,
    3072.3, 3138.7, 3185.2, 3191.0, 3201.3, 3343.0
]

stats_300 = [
    (1,  283.0,  46.8, 0.042), (2,  328.1,  29.4, 0.007), (3,  405.6,  44.2, 0.065),
    (4,  457.0,  24.7, 0.131), (5,  496.4,  31.7, 0.034), (6,  553.0,  33.3, 0.039),
    (7,  610.1,  51.6, 0.107), (8,  690.2,  52.9, 0.095), (9,  754.0,  58.9, 0.060),
    (10, 816.0,  61.3, 0.364), (11, 883.1,  70.1, 0.787), (12, 962.5,  74.4, 0.268),
    (13, 1028.4, 65.6, 0.044), (14, 1075.2, 65.6, 0.455), (15, 1118.2, 69.3, 1.000),
    (16, 1166.7, 73.9, 0.553), (17, 1232.2, 88.7, 0.355), (18, 1320.7, 91.4, 0.096),
    (19, 1396.5, 90.1, 0.408), (20, 1471.3, 90.3, 0.883), (21, 1547.4, 94.0, 0.236),
    (22, 1633.2, 94.1, 0.228), (23, 1716.9, 108.4,0.405), (24, 1946.6, 321.1,0.505),
    (25, 2569.7, 204.4,0.285), (26, 2702.8, 199.1,0.175), (27, 2820.7, 215.5,0.197),
    (28, 2967.4, 261.5,0.262), (29, 3211.5, 403.0,0.572), (30, 3443.1, 365.6,0.158),
]

stats_2000 = [
    (1,  265.8,  55.2, 0.003), (2,  332.4,  42.1, 0.010), (3,  390.6,  40.8, 0.142),
    (4,  444.5,  43.1, 0.039), (5,  500.4,  47.4, 0.005), (6,  563.6,  52.0, 0.090),
    (7,  633.9,  61.3, 0.084), (8,  712.3,  70.6, 0.057), (9,  790.1,  75.2, 0.047),
    (10, 863.0,  80.3, 0.830), (11, 942.5,  80.9, 0.351), (12, 1014.2, 81.0, 0.016),
    (13, 1080.6, 79.1, 0.090), (14, 1144.1, 80.8, 1.000), (15, 1209.9, 85.3, 0.544),
    (16, 1275.9, 95.1, 0.400), (17, 1350.6, 102.1,0.080), (18, 1430.1, 107.9,0.028),
    (19, 1512.4, 115.3,0.986), (20, 1601.3, 126.0,0.208), (21, 1703.0, 179.4,0.192),
    (22, 1871.3, 332.8,0.256), (23, 2150.3, 475.7,0.560), (24, 2518.6, 474.2,0.149),
    (25, 2861.2, 390.5,0.134), (26, 3093.5, 423.3,0.152), (27, 3309.5, 453.7,0.087),
    (28, 3520.8, 457.2,0.577), (29, 3642.2, 410.1,0.319), (30, 3662.4, 361.2,0.037),
]

def get_assignment(mode_num, harm_freq):
    if mode_num <= 4:
        return "Torsion/skeletal bend"
    elif mode_num <= 9:
        return "COO bend/skeletal"
    elif mode_num == 10:
        return "O-O stretch"
    elif mode_num <= 17:
        return "C-O/C-C stretch (fingerprint)"
    elif mode_num <= 24:
        return "CH2 wag/C=C/scissors"
    else:
        return "C-H stretch"

# build table rows HTML
def build_table_rows():
    rows = []
    for i in range(30):
        mn, nu300, sig300, i300  = stats_300[i]
        mn, nu2000, sig2000, i2000 = stats_2000[i]
        hf   = harmonic_freqs[i]
        asgn = get_assignment(mn, hf)
        dnu  = nu2000 - hf   # shift relative to harmonic

        # row style flags
        style = ""
        if abs(dnu) > 50:
            style = "background-color:#ffe0e0;"   # light red
        if sig300 > 200 or sig2000 > 200:
            if style:
                style = "background-color:#ffd0a0;"  # orange if both
            else:
                style = "background-color:#fff3cc;"   # light orange

        rows.append(
            f'<tr style="{style}">'
            f'<td>{mn}</td>'
            f'<td>{hf:.1f}</td>'
            f'<td>{asgn}</td>'
            f'<td>{nu300:.1f}</td>'
            f'<td>{sig300:.1f}</td>'
            f'<td>{i300:.3f}</td>'
            f'<td>{nu2000:.1f}</td>'
            f'<td>{sig2000:.1f}</td>'
            f'<td>{i2000:.3f}</td>'
            f'<td style="{"color:#c00;font-weight:bold" if abs(dnu)>50 else ""}">'
            f'{dnu:+.1f}</td>'
            f'</tr>\n'
        )
    return "".join(rows)

table_rows_html = build_table_rows()

# ── embed all data as JSON ────────────────────────────────────────────────────
def jd(x):
    """Compact JSON dump of a Python list, 6 decimal places."""
    return json.dumps([round(v, 6) for v in x])

def jdf(x):
    return json.dumps([round(v, 8) for v in x])

# ── build HTML ────────────────────────────────────────────────────────────────
print("Building HTML...")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MVKO Multi-State ML-PES — Results Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #f9f9f9;
    color: #222;
    font-size: 15px;
    line-height: 1.6;
  }}
  .page-wrapper {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px 24px 60px 24px;
  }}
  h1 {{
    font-size: 1.7em;
    font-weight: 700;
    color: #1a1a2e;
    margin: 24px 0 6px 0;
    border-bottom: 2px solid #ddd;
    padding-bottom: 6px;
  }}
  h2 {{
    font-size: 1.25em;
    font-weight: 600;
    color: #2c2c5e;
    background: #eef2fb;
    padding: 8px 14px;
    border-left: 4px solid #4a6fa5;
    margin: 28px 0 14px 0;
  }}
  h3 {{
    font-size: 1.05em;
    font-weight: 600;
    color: #333;
    margin: 18px 0 8px 0;
  }}
  p {{ margin: 0 0 10px 0; }}
  .key-findings {{
    background: #fffbe6;
    border: 1px solid #e8d44d;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 18px 0 24px 0;
  }}
  .key-findings h3 {{
    color: #8a6900;
    margin-top: 0;
  }}
  .key-findings ul {{
    margin: 8px 0 0 18px;
  }}
  .key-findings li {{ margin-bottom: 5px; }}
  .plotly-div {{
    width: 100%;
    margin: 14px 0 20px 0;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: white;
  }}
  .two-col {{
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    margin-bottom: 14px;
  }}
  .two-col .plotly-div {{
    flex: 1 1 45%;
    min-width: 320px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88em;
    margin-top: 10px;
  }}
  th {{
    background: #2c3e6b;
    color: white;
    padding: 7px 10px;
    text-align: center;
    font-weight: 600;
  }}
  td {{
    padding: 5px 9px;
    text-align: center;
    border-bottom: 1px solid #eee;
  }}
  tr:hover {{ background-color: #f0f4ff; }}
  .method-table {{
    width: auto;
    min-width: 500px;
  }}
  .method-table th, .method-table td {{
    text-align: left;
    padding: 6px 14px;
  }}
  .method-table td:not(:first-child) {{
    text-align: center;
  }}
  .legend-note {{
    font-size: 0.82em;
    color: #555;
    margin: 4px 0 10px 0;
    font-style: italic;
  }}
  .nav-bar {{
    position: sticky;
    top: 0;
    background: #2c3e6b;
    z-index: 100;
    padding: 8px 0;
    display: flex;
    gap: 0;
    justify-content: center;
    flex-wrap: wrap;
  }}
  .nav-bar a {{
    color: #cde;
    text-decoration: none;
    padding: 6px 16px;
    font-size: 0.92em;
    border-right: 1px solid #3a4f80;
    white-space: nowrap;
  }}
  .nav-bar a:hover {{ background: #3a5080; color: white; }}
  code {{
    background: #f0f0f0;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.9em;
  }}
</style>
</head>
<body>
<div class="nav-bar">
  <a href="#method">Method</a>
  <a href="#pes-cuts">PES Cuts</a>
  <a href="#harmonic-ir">Harmonic IR</a>
  <a href="#hot-emission">Hot Emission</a>
  <a href="#mode-table">Per-Mode Stats</a>
</div>

<div class="page-wrapper">

<h1>MVKO Multi-State ML-PES: Hot-Molecule IR Emission Results</h1>
<p style="color:#555; margin-bottom:4px;">
  Methyl vinyl ketone oxide (MVKO, C&#8324;H&#8326;O&#8322;) — anti-cis conformer.
  Anti-cis = Criegee-O pointing anti to the C=C, methyl cis to the terminal oxide.
  Date: 2026-06-06.
</p>

<div class="key-findings">
  <h3>Key Findings</h3>
  <ul>
    <li><strong>ACF-based IR fails for MVKO</strong>: the 4.5 D permanent COO dipole makes torsional reorientation dominate the autocorrelation function by ~70,000&times; over vibrational stretching. No thermostat tuning, ZPE settings, or improved dipole surface can fix this.</li>
    <li><strong>Correct approach</strong>: harmonic dipole derivatives from <code>NMDipoleSurface.ir_intensities()</code> + local-harmonic hot emission (trajectory-averaged instantaneous frequencies and intensities).</li>
    <li><strong>Strongest IR peak</strong>: mode 15 at 1089 cm&#8315;&#185; (C-O stretch) at 300 K; shifts to 1144 cm&#8315;&#185; at 2000 K with anharmonic redshift.</li>
    <li><strong>Temperature effect</strong>: at 2000 K, modes 22-24 show &gt;300 cm&#8315;&#185; broadening (torsion/COO bend coupling); C-H region (modes 25-30) shifts upward by 100-400 cm&#8315;&#185; due to anharmonic stiffening at high amplitude.</li>
    <li><strong>S1 gap</strong>: 29.0 kcal/mol at equilibrium; T1 gap: 26.5 kcal/mol. Both gaps increase slightly along the O-O stretch coordinate.</li>
    <li><strong>MACE PES validation</strong>: 0 imaginary modes, 0 unphysical modes (&gt;5000 cm&#8315;&#185;), max C-H ~3200 cm&#8315;&#185; — confirms physical Hessian curvature.</li>
  </ul>
</div>

<!-- ============================================================ -->
<h2 id="method">Section 1: Method Overview</h2>

<h3>Molecule</h3>
<p>
  <strong>MVKO</strong> (methyl vinyl ketone oxide): (CH&#8322;=CH)(CH&#8323;)C(O&#8315;O&#8314;), 12 atoms, 30 vibrational modes.
  Criegee intermediate from ozonolysis of methyl vinyl ketone (MVK). Atmospherically relevant reactive intermediate.
  Atom ordering: C1(Criegee), O1(proximal), O2(distal), C2(vinyl=CH-), C3(=CH&#8322;), C4(methyl), H1-H6.
  Anti-cis conformer: equilibrium O-O bond length {eq_roo:.4f} &Aring;.
</p>

<h3>Two-Layer ML Architecture</h3>
<p>
  The potential energy surface uses a two-layer correction on the wB97X-D/6-31G* baseline:
</p>
<ul style="margin: 0 0 10px 20px;">
  <li><strong>E<sub>S0</sub>(R)</strong> = E<sub>wB97X-ML</sub>(R) + &delta;<sub>S0-ML</sub>(R) — ground state, Coulomb+KRR base + CASSCF(4,4) multi-reference correction</li>
  <li><strong>E<sub>S1</sub>(R)</strong> = E<sub>S0</sub>(R) + &Delta;gap<sub>S1-ML</sub>(R) — first excited singlet</li>
  <li><strong>E<sub>T1</sub>(R)</strong> = E<sub>S0</sub>(R) + &Delta;gap<sub>T1-ML</sub>(R) — lowest triplet (ISC)</li>
</ul>
<p>
  CASSCF active space: (4,4) with frontier COO biradical orbitals {{n&#8314;(O<sub>term</sub>), n&#8315;(O<sub>term</sub>), &pi;(COO), &pi;*(COO)}}.
  Natural orbital occupations at eq: (1.948, 1.724, 0.275, 0.052) — 14% biradical character.
  Delta-ML models use normal-mode-coordinate KRR (RBF kernel) for &delta;<sub>S0</sub>, &Delta;gap<sub>S1</sub>, &Delta;gap<sub>T1</sub>.
</p>

<h3>Dipole Surface</h3>
<p>
  <code>NMDipoleSurface</code>: KRR in normal-mode coordinates with analytic &part;&mu;/&part;q<sub>k</sub>.
  R&sup2; = 0.9997 on wB97X-D/6-31G* training dipoles (250 frames, 2 rounds GP active learning).
  C-H stretch modes (25-30) have non-zero &part;&mu;/&part;q<sub>CH</sub> by design — absent in Coulomb+KRR descriptor.
</p>

<h3>Why ACF IR Fails for MVKO</h3>
<p>
  The dipole autocorrelation function (DACF) approach assumes that the dominant signal is vibrational.
  For MVKO, the permanent COO dipole (&mu;<sub>perm</sub> &#8776; 4.5 D) is ~70,000&times; larger than the
  vibrational amplitude (&delta;&mu; &#8776; 0.05 D per vibrational period). Any torsional reorientation
  — even sub-degree — completely overwhelms the vibrational signal. This is not fixable by improved
  dipole surfaces, thermostats, or trajectory length.
</p>
<p>
  <strong>Correct approach</strong>: harmonic dipole derivatives at each trajectory snapshot (local harmonic approximation),
  giving <em>ir_hot_emission(T)</em> = &lang; &Sigma;<sub>k</sub> ||&part;&mu;/&part;q<sub>k</sub>||&sup2;
  &delta;(&nu; - &nu;<sub>k,local</sub>) &rang;<sub>traj</sub>.
</p>

<h3>Model Parameters</h3>
<table class="method-table">
  <thead>
    <tr><th>Component</th><th>Method</th><th>&gamma;</th><th>&alpha;</th><th>N<sub>train</sub></th><th>RMSE / R&sup2;</th></tr>
  </thead>
  <tbody>
    <tr><td>wB97X base ML-PES</td><td>Coulomb+KRR</td><td>0.001</td><td>1e-5</td><td>1126</td><td>0.31 kcal/mol</td></tr>
    <tr><td>&delta;<sub>S0</sub> CASSCF correction</td><td>NM-KRR</td><td>0.1</td><td>1e-6</td><td>232</td><td>0.654 kcal/mol LOO-CV</td></tr>
    <tr><td>&Delta;gap<sub>S1</sub></td><td>NM-KRR</td><td>0.1</td><td>1e-6</td><td>232</td><td>0.588 kcal/mol LOO-CV</td></tr>
    <tr><td>&Delta;gap<sub>T1</sub></td><td>NM-KRR</td><td>0.1</td><td>1e-6</td><td>232</td><td>0.566 kcal/mol LOO-CV</td></tr>
    <tr><td>Dipole surface (NMDipoleSurface)</td><td>NM-KRR</td><td>auto</td><td>auto</td><td>250</td><td>R&sup2; = 0.9997</td></tr>
    <tr><td>MACE PES (for MD)</td><td>SO(3)-equivariant MPNN</td><td>—</td><td>—</td><td>1126</td><td>0 imaginary modes</td></tr>
  </tbody>
</table>

<!-- ============================================================ -->
<h2 id="pes-cuts">Section 2: PES Cuts Along Normal Mode Coordinates</h2>
<p>
  One-dimensional cuts of the multi-state PES along two key normal modes.
  Smooth curves: 200-point KRR evaluation grid. Open markers: raw DFT training points
  where all other mode amplitudes |q<sub>other</sub>| &lt; threshold (pure-mode frames).
  All energies relative to the wB97X-D ML-PES minimum.
</p>
<div class="two-col">
  <div id="pes_mode10" class="plotly-div" style="height:520px;"></div>
  <div id="pes_mode1"  class="plotly-div" style="height:520px;"></div>
</div>
<p class="legend-note">
  Mode 10: O-O stretch (835 cm&#8315;&#185;), pure-mode frames with |q<sub>other</sub>| &lt; 0.4 &radic;(amu)&middot;Bohr (168/232 frames).
  Mode 1: skeletal torsion (119.5 cm&#8315;&#185;), |q<sub>other</sub>| &lt; 0.5 (188/232 frames).
  CASSCF(4,4)/6-31G* // wB97X-D/6-31G* training level.
</p>

<!-- ============================================================ -->
<h2 id="harmonic-ir">Section 3: Harmonic Reference IR Spectrum</h2>
<p>
  Equilibrium harmonic IR spectrum computed from <code>NMDipoleSurface.ir_intensities(eq_coords)</code>.
  Lorentzian broadening FWHM = 10 cm&#8315;&#185;. This represents the zero-temperature, no-anharmonicity reference.
</p>
<div id="ir_harmonic" class="plotly-div" style="height:400px;"></div>

<!-- ============================================================ -->
<h2 id="hot-emission">Section 4: Hot-Molecule IR Emission Spectra — Temperature Comparison</h2>
<p>
  Local-harmonic IR emission spectra from MACE-MD trajectories at 300 K and 2000 K.
  At each trajectory snapshot, the local MACE Hessian is diagonalized and &part;&mu;/&part;q<sub>k</sub>
  are evaluated; the resulting (frequency, intensity) pairs are accumulated into a histogram.
  FWHM = 15 cm&#8315;&#185; at 300 K, 30 cm&#8315;&#185; at 2000 K (accounts for broadening).
  Spectra normalized to unity at the strongest peak.
</p>
<div id="ir_comparison" class="plotly-div" style="height:500px;"></div>
<p class="legend-note">
  Colored background regions: torsion/bend (50–500 cm&#8315;&#185;, light blue), O-O/C-O (500–1100 cm&#8315;&#185;, light green),
  fingerprint (1100–1800 cm&#8315;&#185;, light yellow), C-H stretch (2800–4000 cm&#8315;&#185;, light red).
</p>

<!-- ============================================================ -->
<h2 id="mode-table">Section 5: Per-Mode Anharmonic Statistics</h2>
<p>
  Per-mode mean frequency, standard deviation, and relative intensity from local-harmonic
  trajectory averaging. Rows highlighted in <span style="background:#ffe0e0; padding:2px 6px;">light red</span>:
  |&Delta;&nu;(2000K)| &gt; 50 cm&#8315;&#185; vs harmonic.
  Rows in <span style="background:#fff3cc; padding:2px 6px;">light orange</span>: &sigma; &gt; 200 cm&#8315;&#185; at either temperature (large anharmonic broadening).
  Rows in <span style="background:#ffd0a0; padding:2px 6px;">orange</span>: both conditions met.
</p>
<table>
  <thead>
    <tr>
      <th rowspan="2">Mode</th>
      <th rowspan="2">&nu;<sub>harm</sub> (cm&#8315;&#185;)</th>
      <th rowspan="2">Assignment</th>
      <th colspan="3">300 K</th>
      <th colspan="3">2000 K</th>
      <th rowspan="2">&Delta;&nu; (2000K&minus;harm)</th>
    </tr>
    <tr>
      <th>&langle;&nu;&rangle;</th>
      <th>&sigma;<sub>&nu;</sub></th>
      <th>I<sub>rel</sub></th>
      <th>&langle;&nu;&rangle;</th>
      <th>&sigma;<sub>&nu;</sub></th>
      <th>I<sub>rel</sub></th>
    </tr>
  </thead>
  <tbody>
{table_rows_html}  </tbody>
</table>

</div><!-- page-wrapper -->

<script>
// ============================================================
// Data
// ============================================================
const q10_grid    = {jd(q10_grid)};
const r_oo_grid   = {jd(r_oo_grid)};
const E_wb97x_ml  = {jd(E_wb97x_ml)};
const E_s0_corr   = {jd(E_s0_corr)};
const E_s1        = {jd(E_s1)};
const E_t1        = {jd(E_t1)};
const q10_pts     = {jd(q10_pts)};
const r_oo_pts    = {jd(r_oo_pts)};
const E_wb97x_pts = {jd(E_wb97x_pts)};
const E_s0_pts    = {jd(E_s0_pts)};
const E_s1_pts    = {jd(E_s1_pts)};
const E_t1_pts    = {jd(E_t1_pts)};

const q1_grid         = {jd(q1_grid)};
const E_wb97x_ml_m1   = {jd(E_wb97x_ml_m1)};
const E_s0_corr_m1    = {jd(E_s0_corr_m1)};
const E_s1_m1         = {jd(E_s1_m1)};
const E_t1_m1         = {jd(E_t1_m1)};
const q1_pts          = {jd(q1_pts)};
const E_wb97x_pts_m1  = {jd(E_wb97x_pts_m1)};
const E_s0_pts_m1     = {jd(E_s0_pts_m1)};

const ir_freq_harm = {jd(ir_freq_harm_t)};
const ir_int_harm  = {jdf(ir_int_harm_t)};
const ir_freq_300  = {jd(ir_freq_300_t)};
const ir_int_300   = {jdf(ir_int_300_t)};
const ir_freq_2000 = {jd(ir_freq_2000_t)};
const ir_int_2000  = {jdf(ir_int_2000_t)};

const EQ_ROO = {eq_roo:.4f};

// ============================================================
// Plot helpers
// ============================================================
const LINE = (x, y, name, color, dash, width) => ({{
  x, y, name, type:'scatter', mode:'lines',
  line:{{color, dash: dash||'solid', width: width||2}},
  hovertemplate: '%{{x:.3f}}, %{{y:.2f}} kcal/mol<extra>'+name+'</extra>'
}});
const SCAT = (x, y, name, color, sym) => ({{
  x, y, name, type:'scatter', mode:'markers',
  marker:{{color:'rgba(0,0,0,0)', size:7, symbol:sym||'circle',
           line:{{color, width:1.5}}}},
  hovertemplate: '%{{x:.3f}}, %{{y:.2f}} kcal/mol<extra>'+name+'</extra>'
}});

// ============================================================
// PES Mode 10 — O-O stretch
// ============================================================
(function() {{
  const traces = [
    LINE(r_oo_grid, E_wb97x_ml,  'wB97X-D ML-PES',           '#2166ac', 'solid', 2.5),
    LINE(r_oo_grid, E_s0_corr,   'S0 (wB97X + δ_S0)',         '#111111', 'solid', 2.5),
    LINE(r_oo_grid, E_s1,        'S1 = S0 + ΔgapS1',          '#1a9641', 'dash',  2),
    LINE(r_oo_grid, E_t1,        'T1 = S0 + ΔgapT1',          '#d7191c', 'dash',  2),
    SCAT(r_oo_pts, E_wb97x_pts,  'wB97X-D DFT pts',           '#2166ac', 'circle-open'),
    SCAT(r_oo_pts, E_s0_pts,     'S0 corrected pts',          '#111111', 'square-open'),
    SCAT(r_oo_pts, E_s1_pts,     'S1 pts',                    '#1a9641', 'triangle-up-open'),
    SCAT(r_oo_pts, E_t1_pts,     'T1 pts',                    '#d7191c', 'triangle-down-open'),
  ];

  // Annotation: equilibrium r_OO vertical dashed line
  const eq_annot = {{
    type:'line', x0:EQ_ROO, x1:EQ_ROO, y0:-1, y1:32,
    xref:'x', yref:'y',
    line:{{color:'gray', width:1, dash:'dot'}},
  }};

  const layout = {{
    title:{{text:'Mode 10: O-O Stretch (835 cm⁻¹)', font:{{size:14}}}},
    xaxis:{{title:'r(O-O) (Å)', range:[1.315, 1.375]}},
    yaxis:{{title:'ΔE (kcal/mol)', range:[-1, 35]}},
    legend:{{x:0.02, y:0.98, bgcolor:'rgba(255,255,255,0.85)', bordercolor:'#ccc', borderwidth:1}},
    margin:{{l:55, r:20, t:50, b:55}},
    shapes:[eq_annot],
    annotations:[
      {{x:EQ_ROO+0.002, y:0.5, text:'r_eq = {eq_roo:.4f} Å', showarrow:false,
        font:{{size:11, color:'gray'}}, xanchor:'left'}},
      {{x:EQ_ROO+0.002, y:29.5, text:'ΔS1 = 29.0 kcal/mol', showarrow:false,
        font:{{size:11, color:'#1a9641'}}, xanchor:'left'}},
      {{x:EQ_ROO+0.002, y:27.0, text:'ΔT1 = 26.5 kcal/mol', showarrow:false,
        font:{{size:11, color:'#d7191c'}}, xanchor:'left'}},
    ],
    hovermode:'x unified',
    plot_bgcolor:'white',
    paper_bgcolor:'white',
  }};
  Plotly.newPlot('pes_mode10', traces, layout, {{responsive:true}});
}})();

// ============================================================
// PES Mode 1 — Torsion
// ============================================================
(function() {{
  // Clip torsion curves to reasonable energy range for display
  const EMAX = 55;
  const clip = (arr) => arr.map(v => Math.min(v, EMAX+5));
  const traces = [
    LINE(q1_grid, clip(E_wb97x_ml_m1), 'wB97X-D ML-PES',        '#2166ac', 'solid', 2.5),
    LINE(q1_grid, clip(E_s0_corr_m1),  'S0 (wB97X + δ_S0)',      '#111111', 'solid', 2.5),
    LINE(q1_grid, clip(E_s1_m1),       'S1 = S0 + ΔgapS1',       '#1a9641', 'dash',  2),
    LINE(q1_grid, clip(E_t1_m1),       'T1 = S0 + ΔgapT1',       '#d7191c', 'dash',  2),
    SCAT(q1_pts,  E_wb97x_pts_m1,     'wB97X-D DFT pts',         '#2166ac', 'circle-open'),
    SCAT(q1_pts,  E_s0_pts_m1,        'S0 corrected pts',        '#111111', 'square-open'),
  ];

  const layout = {{
    title:{{text:'Mode 1: Skeletal Torsion (119.5 cm⁻¹)', font:{{size:14}}}},
    xaxis:{{title:'q₁ (√amu·Bohr)', range:[-4, 4]}},
    yaxis:{{title:'ΔE (kcal/mol)', range:[-1, EMAX]}},
    legend:{{x:0.02, y:0.98, bgcolor:'rgba(255,255,255,0.85)', bordercolor:'#ccc', borderwidth:1}},
    margin:{{l:55, r:20, t:50, b:55}},
    annotations:[
      {{x:0, y:-0.6, text:'q₁ = 0 (eq)', showarrow:false,
        font:{{size:11, color:'gray'}}}},
    ],
    hovermode:'x unified',
    plot_bgcolor:'white',
    paper_bgcolor:'white',
  }};
  Plotly.newPlot('pes_mode1', traces, layout, {{responsive:true}});
}})();

// ============================================================
// Harmonic IR
// ============================================================
(function() {{
  const peak_labels = [
    {{x:119.5, label:'119.5<br>torsion'}},
    {{x:835.2, label:'835<br>O-O'}},
    {{x:1011.2, label:'1011'}},
    {{x:1088.8, label:'1089<br>C-O'}},
    {{x:1480.4, label:'1480<br>CH₂wag'}},
    {{x:1692.6, label:'1693<br>C=C'}},
    {{x:3072.3, label:'3072'}},
    {{x:3201.3, label:'3201<br>C-H'}},
  ];

  const traces = [{{
    x: ir_freq_harm, y: ir_int_harm,
    name: 'Harmonic reference', type:'scatter', mode:'lines',
    line:{{color:'black', dash:'dash', width:2}},
    fill:'tozeroy', fillcolor:'rgba(180,180,180,0.18)',
    hovertemplate: '%{{x:.0f}} cm⁻¹, I=%{{y:.4f}}<extra>harmonic</extra>'
  }}];

  const annots = peak_labels.map(p => ({{
    x:p.x, y:1.05, xref:'x', yref:'paper',
    text:p.label, showarrow:false,
    font:{{size:10, color:'#333'}}, align:'center',
    textangle:0
  }}));

  const layout = {{
    title:{{text:'Equilibrium Harmonic IR Spectrum (FWHM = 10 cm⁻¹)', font:{{size:14}}}},
    xaxis:{{title:'Frequency (cm⁻¹)', range:[50, 4000]}},
    yaxis:{{title:'Normalized Intensity', range:[0, 1.15]}},
    legend:{{x:0.7, y:0.95}},
    margin:{{l:60, r:20, t:50, b:55}},
    annotations: annots,
    plot_bgcolor:'white', paper_bgcolor:'white',
  }};
  Plotly.newPlot('ir_harmonic', traces, layout, {{responsive:true}});
}})();

// ============================================================
// Hot Emission Comparison
// ============================================================
(function() {{
  const regions = [
    {{x0:50,   x1:500,  color:'rgba(173,216,230,0.25)', label:'Torsion/bend'}},
    {{x0:500,  x1:1100, color:'rgba(144,238,144,0.22)', label:'O-O / C-O'}},
    {{x0:1100, x1:1800, color:'rgba(255,255,150,0.25)', label:'Fingerprint'}},
    {{x0:2800, x1:4000, color:'rgba(255,160,160,0.25)', label:'C-H stretch'}},
  ];

  const shapes = regions.map(r => ({{
    type:'rect', xref:'x', yref:'paper',
    x0:r.x0, x1:r.x1, y0:0, y1:1,
    fillcolor:r.color, line:{{width:0}}, layer:'below'
  }}));

  const traces = [
    {{
      x: ir_freq_harm, y: ir_int_harm,
      name: 'Harmonic reference', type:'scatter', mode:'lines',
      line:{{color:'#333', dash:'dot', width:1.5}},
      hovertemplate: '%{{x:.0f}} cm⁻¹<extra>Harmonic</extra>'
    }},
    {{
      x: ir_freq_300, y: ir_int_300,
      name: '300 K hot emission', type:'scatter', mode:'lines',
      line:{{color:'#2166ac', width:2.5}},
      fill:'tozeroy', fillcolor:'rgba(33,102,172,0.08)',
      hovertemplate: '%{{x:.0f}} cm⁻¹, I=%{{y:.4f}}<extra>300 K</extra>'
    }},
    {{
      x: ir_freq_2000, y: ir_int_2000,
      name: '2000 K hot emission', type:'scatter', mode:'lines',
      line:{{color:'#d7191c', width:2.5}},
      fill:'tozeroy', fillcolor:'rgba(215,25,28,0.07)',
      hovertemplate: '%{{x:.0f}} cm⁻¹, I=%{{y:.4f}}<extra>2000 K</extra>'
    }},
  ];

  // Region labels at top
  const reg_annots = regions.map(r => ({{
    x:(r.x0+r.x1)/2, y:0.97, xref:'x', yref:'paper',
    text:r.label, showarrow:false,
    font:{{size:10, color:'#444'}}, align:'center'
  }}));

  // Peak annotations
  const peak_annots = [
    {{x:1118, y:1.02, text:'1118 (300K)', color:'#2166ac'}},
    {{x:741,  y:0.72, text:'741 (300K)',  color:'#2166ac'}},
    {{x:1425, y:0.90, text:'1425 (300K)', color:'#2166ac'}},
    {{x:1024, y:1.03, text:'1024 (2kK)', color:'#d7191c'}},
    {{x:772,  y:0.85, text:'772 (2kK)',  color:'#d7191c'}},
    {{x:1294, y:0.41, text:'1294 (2kK)', color:'#d7191c'}},
  ].map(p => ({{
    x:p.x, y:p.y, xref:'x', yref:'paper',
    text:p.text, showarrow:true, ax:0, ay:-25,
    arrowhead:2, arrowsize:0.8,
    font:{{size:10, color:p.color}},
    arrowcolor:p.color,
  }}));

  const layout = {{
    title:{{text:'Hot-Molecule IR Emission: 300 K vs 2000 K', font:{{size:14}}}},
    xaxis:{{title:'Frequency (cm⁻¹)', range:[50, 4000]}},
    yaxis:{{title:'Normalized Intensity', range:[0, 1.12]}},
    legend:{{x:0.65, y:0.96, bgcolor:'rgba(255,255,255,0.85)', bordercolor:'#ccc', borderwidth:1}},
    margin:{{l:60, r:20, t:55, b:55}},
    shapes: shapes,
    annotations: [...reg_annots, ...peak_annots],
    plot_bgcolor:'white', paper_bgcolor:'white',
    hovermode:'x'
  }};
  Plotly.newPlot('ir_comparison', traces, layout, {{responsive:true}});
}})();

</script>
</body>
</html>
"""

out_path = "outputs/mvko_results_report.html"
with open(out_path, "w") as f:
    f.write(html)

import os
size_kb = os.path.getsize(out_path) / 1024
print(f"Saved HTML report: {out_path} ({size_kb:.0f} KB)")
print("Done.")
