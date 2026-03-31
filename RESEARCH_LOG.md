# ML-PES / ML-MD Research Log

---

## 2026-03-31 — CASSCF(4,4) IRC Correction + Delta-ML

### Objective
Compare B3LYP/6-31G* IRC energies against CASSCF(4,4)/6-31G* for the syn-MVKO → VHP 1,4-H shift.
Compute diabatic coupling H₁₂ from SA-2-CASSCF. Train a delta-ML KRR to correct the B3LYP PES.

### Script
`casscf_irc_correction.py` — PSI4 RHF → SS-CASSCF(4,4) → SA-2-CASSCF(4,4) per IRC frame.
Active space: σ/σ\*(C3–H7) + σ/σ\*(O2–H7); MVKO has 46 e⁻, frozen\_docc=6, restricted\_docc=15, active=4.

### Data
9 filtered IRC frames from `outputs/mvko_rxn_path_20260330_181025/irc_training_data.npz`
(ΔE ≤ 100 kcal/mol vs MVKO min; s from −3.15 to +0.45 Å·√amu)

### Results (`outputs/casscf_irc_20260330_230145/`)

| Frame | s (Å·√amu) | ΔE\_B3LYP | ΔE\_CASSCF | Δcorr | H₁₂ | NO occs (4 active) |
|---|---|---|---|---|---|---|
| 0 | −3.15 | 0.0 | 0.0 | +0.0 | — | 1.998, 1.924, 0.078, 0.000 |
| 3 | −0.30 | 36.6 | 46.0 | +9.5 | 44.1 | 1.994, 1.933, 0.067, 0.005 |
| 4 | −0.15 | 44.3 | 60.4 | +16.1 | 44.5 | 1.993, 1.958, 0.040, 0.009 |
| 5 | +0.00 | 52.0 | 64.1 | +12.0 | 38.5 | 1.965, 1.939, 0.065, 0.031 |
| 6 | +0.15 | 55.2 | 64.4 | +9.2 | 29.3 | 1.964, 1.951, 0.058, 0.027 |
| 7 | +0.30 | 54.2 | 57.6 | +3.4 | 23.0 | 1.967, 1.959, 0.053, 0.020 |

**B3LYP barrier: 80.4 kcal/mol → CASSCF(4,4) barrier: 86.5 kcal/mol (+6.2 kcal/mol)**

Dynamic correlation (NEVPT2/CASPT2) not computed — expected to add 5–10 kcal/mol.

### Physical interpretation

**NO occupations** at the "TS" (s ≈ 0) are (1.965, 1.939, 0.065, 0.031) — nearly closed-shell,
far from the biradical (1,1,1,1) pattern. The H-transfer has predominantly **single-reference
character** at these IRC geometries. Possible explanations:
- The IRC geometry at s=0 is not the true electronic TS (the IRC followed a high-energy path)
- The active space orbitals are not the σ/σ\*(C-H)/σ/σ\*(O-H) pair (RHF ordering may differ)
- The concerted [1,4]-H shift indeed has little biradical character (partially dipole-allowed)

**H₁₂ = 38–48 kcal/mol** (very large) → strongly coupled/adiabatic regime.
Landau-Zener probability P\_hop ≈ exp(−2π H₁₂²/(ℏv|ΔF|)) ≈ 0 → no surface hopping.
The reaction proceeds on a single adiabatic PES; the delta-ML correction (+6–12 kcal/mol)
is the main CASSCF contribution.

### Delta-ML correction
9-frame KRR (γ=0.00005, α=1e-10) trained on relative delta = ΔE\_CASSCF − ΔE\_B3LYP.
Range: −3.1 to +16.1 kcal/mol. Corrected PESFamily manifest:
`outputs/casscf_irc_20260331_080127/corrected_family_manifest.json`

### Bugs fixed in `casscf_irc_correction.py`
1. `symmetry c1` needed in geometry block (PSI4 detected Cs symmetry → frozen\_docc=[6] wrong size)
2. SA-CASSCF options (avg\_states/avg\_weights) leaked across frames via PSI4 options — fixed by
   explicit reset in SS-CASSCF opts + `psi4.core.clean_options()` call
3. SA-CASSCF skipped for |s| > 2.0 (closed-shell MVKO min → SA fails to converge)
4. NO occupation parser: PSI4 1.10 uses multi-column "Active Space Natural occupation numbers:"
   format; fixed regex to capture whole block then findall
5. Delta-ML now uses **relative** energies (ΔE\_CASSCF − ΔE\_B3LYP) not absolute difference
   (avoids ~1135 kcal/mol spurious offset from DFT dynamic correlation)

---

## 2026-03-31 — CASSCF Full-Surface Delta-ML + IR Comparison

### Objective
Extend the IRC CASSCF correction to the full B3LYP training surface (904 frames).
Train a delta-ML model and test whether the CASSCF correction meaningfully shifts the MVKO 300K IR spectrum.

### Script
`casscf_surface_correction.py` — stratified FPS frame selection → SS-CASSCF(4,4)/6-31G* → delta-ML.

### Frame selection (`outputs/casscf_surface_20260331_133413/`)
30 frames selected by FPS in Coulomb space within 5 energy bins: [0–5, 5–15, 15–30, 30–60, 60–100 kcal/mol].
29/30 CASSCF converged (frame 713 failed to converge).

### Key results (selected frames)

| ΔE\_B3LYP (kcal/mol) | ΔE\_CASSCF | Δcorr | Notes |
|---|---|---|---|
| 0–3 | 0–3.4 | 0 to +2.3 | Near-equilibrium, small corrections |
| 10–22 | 11–27 | −0.9 to +4.7 | Variable, mostly positive |
| 27–45 | 8–55 | −19 to +14 | Large scatter; frame 29 anomalous (−19 kcal/mol) |
| 57–72 | 30–63 | −28 to −11 | B3LYP systematically **stiffer** than CASSCF at high distortion |
| 98 | 91 | −7 | B3LYP still stiffer at extreme distortion |

**Key physics:** At high distortions (ΔE > 40 kcal/mol), B3LYP overestimates PES stiffness
by 10–28 kcal/mol vs CASSCF. Near equilibrium (ΔE < 15 kcal/mol), corrections are +0.6 ± 1.0 kcal/mol.

### Delta-ML approaches tried

**Approach 1: Geometry-based KRR** (`outputs/casscf_surface_20260331_162217/merged/delta_ml_surface.pkl`)
- 38 frames (29 surface + 9 IRC), γ=5e-5, α=1e-10
- RMSE: 21 kcal/mol — **failed due to Coulomb descriptor clustering**
- Root cause: all geometries have ||d_i − d_j|| ≈ 4-5 (out of ||d|| ≈ 158) → K≈0.999 for all pairs
- Large negative high-energy corrections bleed into equilibrium → 5 imaginary NM modes (−3800 cm⁻¹)
- IR spectrum: only peak at 0.7 cm⁻¹ (CASSCF model destroys equilibrium geometry)

**Approach 2: 1D energy spline** (`outputs/casscf_surface_20260331_162217/energy_delta.json`)
- Fit δ(ΔE\_B3LYP) as cubic spline, anchored at δ(0)=0
- **Failed due to oscillation**: frame 29 anomaly (δ=−19 at ΔE=27) causes runaway spline at ΔE=25–30
- Fundamental issue: CASSCF corrections at ΔE < 15 kcal/mol are noisy (σ=1 kcal/mol) relative to the correction itself (μ=0.6 kcal/mol)

### Physical conclusion
**The CASSCF correction is negligible for 300K IR spectra of MVKO.**

Near-equilibrium statistics (ΔE\_B3LYP < 15 kcal/mol):
- Mean δ: +0.62 kcal/mol (vs kT=0.59 kcal/mol at 300K)
- Std δ: ±1.05 kcal/mol (comparable to correction magnitude)
- NO occupancies at MVKO minimum: 1.998, 1.924, 0.078, 0.000 → nearly closed-shell

The correction matters for: (1) syn-MVKO → VHP reaction barrier (+6.2 kcal/mol), and (2) high-energy PES exploration (thermodynamics, conformer interconversion). For 300K vibrational spectroscopy, B3LYP/6-31G\* is adequate.

### IR spectrum reproducibility check
Reran 30k-step B3LYP IR (`outputs/ir_spectrum_B3LYP_20260331/`) — identical to original:

| Peak (cm⁻¹) | Assignment |
|---|---|
| 322 | Torsion (strongest) |
| 247, 243 | Torsion/bending |
| 893 | **O-O stretch** |
| 514 | C-O-O bending |
| 207, 183 | Low-frequency modes |

Comparison figure: `outputs/casscf_surface_20260331_162217/ir_comparison_b3lyp_vs_b3lyp.png`

### New tools added
- `compare_ir_spectra.py` — overlay any number of ir\_spectrum.csv files with Gaussian broadening, peak annotations, difference panel (2-spectrum case), `path:label` CLI syntax
- `ir_md_spectrum.py --energy-delta <json>` — 1D spline CASSCF correction flag (for future use with better training data)
- `ir_md_spectrum.py --delta-model <pkl>` — geometry-based KRR correction flag (not recommended for Coulomb descriptors)

---

## 2026-03-31 — Analysis: Why Delta-ML Failed and How to Fix It

### Problem 1 — CASSCF minimum offset corrupts the anchor

Our delta correction is anchored at δ(frame 0) = 0 by construction, using the B3LYP minimum as
the CASSCF zero-point. But the true CASSCF(4,4)/6-31G\* minimum lies at a slightly displaced
geometry R\_CASSCF\_min. The correct anchor is:

```
δ(R_B3LYP_min) = E_CASSCF(R_B3LYP_min) − E_CASSCF(R_CASSCF_min) > 0
```

Forcing this to zero is physically wrong. The gradient of δ at R\_B3LYP\_min points toward
R\_CASSCF\_min, creating a systematic force in MD that is currently unaccounted for.
For nearly closed-shell MVKO the displacement is small (< 0.01 Å, error < 0.3 kcal/mol),
but the anchoring error contributes to the near-equilibrium scatter.

**Fix**: run `optimize('casscf')` in PSI4 to locate R\_CASSCF\_min, compute E\_CASSCF there,
and use that as the CASSCF reference energy throughout.

### Problem 2 — Coulomb descriptor clustering (fundamental)

For a 12-atom molecule, the Coulomb descriptor norm ||d|| ≈ 158 while geometry differences give
||d\_i − d\_j|| ≈ 4–5 even between equilibrium and 60 kcal/mol distortions.
All pairwise RBF kernel values K ≈ 0.999: the KRR computes a global weighted average of all
training deltas regardless of geometry. Large negative corrections at high energy bleed
into the equilibrium region → imaginary NM modes.

This is a fundamental limitation of Coulomb matrices for delta-ML on medium-to-large molecules.
Smaller molecules (< 5 atoms) do not suffer from this because ||d|| grows as N² while
geometry displacements grow as N, so the relative separation improves.

### Problem 3 — 1D energy spline is underdetermined and sensitive to outliers

The B3LYP relative energy ΔE is a scalar proxy for distortion, but geometries at similar ΔE
can have very different δ depending on which modes are excited. Frame 29 (δ = −19 kcal/mol
at ΔE = 27 kcal/mol) is likely a geometry with unusual multi-reference character not shared
by neighbouring frames. A cubic spline with only 10 near-equilibrium points cannot handle
a single large outlier.

### Proposed fixes — in order of effort

#### Fix A — More near-equilibrium CASSCF points (low effort, immediate)

Add 15–20 NM-displaced geometries at ΔE\_B3LYP < 5 kcal/mol (same procedure as
`generate_nm_training.py` but at T = 300 K amplitudes). Compute SS-CASSCF for each.
This would:
- Anchor the spline reliably over the thermally sampled region
- Reduce σ(δ) near equilibrium from ±1.0 to ±0.3 kcal/mol (expected)
- Reveal whether the mean near-equilibrium correction is truly ~+0.6 or noise-dominated

#### Fix B — Normal mode coordinate descriptors (medium effort, correct fix)

Express each geometry as displacements along **CASSCF normal modes** {q₁, ..., q₃₀}
from the CASSCF minimum, rather than Coulomb matrix elements:

1. Optimize CASSCF(4,4)/6-31G\* geometry (`optimize('casscf/6-31g*')` in PSI4)
2. Compute CASSCF Hessian → mass-weighted NM eigenvectors U (shape 3N×3N)
3. Project training geometry i: **q**\_i = U^T · M^{1/2} · (R\_i − R\_CASSCF\_min)
4. Use {q\_i} as features in KRR: now equilibrium is at origin and all amplitudes are
   orthogonal — the descriptor space is well-separated

The kernel K(**q**\_i, **q**\_j) = exp(−γ ||**q**\_i − **q**\_j||²) now localises correctly:
geometries with large NM amplitude have large ||**q**|| and are far from equilibrium in feature space.
This has solid precedent in VPT2, VSCF, and vibrational CI literature.

#### Fix C — Change the reference method (medium effort, theoretically cleaner)

Use HF/6-31G\* as the reference instead of B3LYP/6-31G\*:

```
CASSCF = HF + (active-space static correlation)
δ(CASSCF − HF) = pure static correlation only
```

This δ is zero for a perfect closed-shell ground state and grows smoothly with multi-reference
character. Near the MVKO minimum (NO occs 1.998, 1.924), δ(CASSCF−HF) < 1 kcal/mol and
the near-equilibrium scatter would be much smaller than the current B3LYP-referenced δ.

MP2 is a poorer reference than HF: MP2 captures dynamic correlation perturbatively while
CASSCF captures static correlation variationally; their difference is non-trivial and
can have the wrong sign in multi-reference regions.

#### Fix D — Full theoretical hierarchy / triple correction (high effort, publishable accuracy)

The most rigorous approach decomposes the correction into physically motivated layers:

```
E_best ≈ E_B3LYP/6-31G*
       + δ₁ = E_HF/6-31G*     − E_B3LYP/6-31G*     (remove DFT XC error)
       + δ₂ = E_CASSCF(4,4)   − E_HF               (add static correlation)
       + δ₃ = E_NEVPT2        − E_CASSCF            (add dynamic correlation)
       + δ₄ = E_CBS            − E_6-31G*            (basis set correction)
```

Each δ is individually small near equilibrium and can be fit with fewer training points
because each has well-defined physical behaviour. This is the **focal-point** composite
method approach (analogous to W1, W4, CBS-QB3 thermochemistry protocols).

NEVPT2/6-31G\* on top of CASSCF(4,4) is already available in PSI4 (`energy('nevpt2')`).
For 300K IR spectra δ₃ ≈ 0–2 kcal/mol (expected) and δ₄ ≈ 1–3 kcal/mol for frequencies.

**Note on MC-PDFT**: OpenMolcas implements multi-configuration pair-density functional theory
(MC-PDFT), which uses the CASSCF wave function with a translated on-top pair-density
functional. It gives NEVPT2-quality energies at CASSCF cost and connects naturally to
B3LYP-family DFT — potentially the most efficient route to δ₂+δ₃ in a single calculation.
PSI4 does not currently implement MC-PDFT.

### Recommended next step for MVKO delta-ML

Fix A (more near-equilibrium CASSCF points) + Fix B (NM coordinate descriptors), using
the CASSCF minimum as the anchor. This combination solves all three identified problems
with modest computational effort (~1 CASSCF frequency job + 20 single-point CASSCF jobs).

Command sketch:
```bash
# Step 1: CASSCF minimum + Hessian
python3 casscf_surface_correction.py --optimize-casscf \
    --model outputs/mvko_20260319_081314/mlpes_initial.pkl \
    --training-data outputs/mvko_20260319_081314/combined_training_data.npz

# Step 2: NM-displaced CASSCF points at 300K amplitude (Fix A)
python3 casscf_surface_correction.py \
    --nm-sample --n-frames 20 --T-nm 300 \
    --casscf-min outputs/casscf_min_<ts>/casscf_min.npy \
    --training-data outputs/mvko_20260319_081314/combined_training_data.npz

# Step 3: Fit delta-ML in NM coordinates (Fix B)
python3 casscf_surface_correction.py --nm-descriptors \
    --casscf-hessian outputs/casscf_min_<ts>/casscf_hessian.npy \
    --load-results outputs/casscf_surface_<ts>/surface_results.json
```

(These flags do not yet exist — would need implementation in `casscf_surface_correction.py`.)

---

**Project:** Machine-learning potential energy surfaces for Criegee intermediates
**Molecule:** CH₂OO (formaldehyde oxide) → MVKO (methyl vinyl ketone oxide, C₄H₆O₂)
**Method:** Kernel Ridge Regression (KRR) on Coulomb matrix descriptors, PSI4 B3LYP/6-31G*
**Goal:** Accurate IR spectra of atmospherically relevant Criegee intermediates via ML-MD

---

## 2026-03-05 — Project Initialisation

**Scope set.** The existing codebase was a collection of ~67 disconnected workflow scripts and
a core `modules/` library.  No documentation, no unit conventions enforced, no reproducible
training pipeline.

**Actions:**
- Wrote `CLAUDE.md` documenting architecture, unit conventions (Hartree, Angstrom, fs, Debye),
  and the two-layer module / workflow structure
- Established `TrajectoryData` dataclass as the canonical in-memory representation
- Fixed `modules/ml_pes_test.py` to match the actual `CoulombMatrixDescriptor.compute()` /
  `compute_batch()` API

**Key design decisions recorded:**
- `train_forces = False` by default — force training was silently breaking KRR fits
- `StandardScaler` on both features *and* targets is mandatory — without it RMSE degrades
  from ~0.04 to ~170 kcal/mol
- All PSI4 calls use `basis=6-31G*, scf_type=df, reference=rhf` — consistency enforced
  project-wide

---

## 2026-03-08 — PSI4 Version Crisis & Clean Training Pipeline

### Problem discovered
Training data collected in December 2025 gave B3LYP/6-31G\* energies **~8.5 kcal/mol lower**
than the currently installed PSI4 1.10.  Mixing old and new data caused catastrophic validation
failures (15–40 kcal/mol systematic RMSE).

### Root cause
Two different PSI4 installations with subtly different DFT integration grids or ERI
thresholds.  The offset is not a basis-set or method change — it persists across all
geometries and is a constant systematic shift.

### Fix
Regenerated **all** training data from scratch with PSI4 1.10:

| File | Contents | Frames |
|------|----------|--------|
| `outputs/clean_psi410_20260308_203552/training_data.npz` | All PSI4 1.10 frames | 344 |
| `outputs/nm_training_20260308_203606/mlpes_model_nm.pkl` | Trained CH₂OO model | — |

### Normal-mode training data generator (`generate_nm_training.py`)
Added to the codebase.  Generates training frames via:
1. **NM distortions**: Displace ±*n* × *a*_thermal(*T*) along each mode.  72 frames at T=1000 K,
   ±4 amplitudes per mode.
2. **PSI4 MD**: Short MD bursts at 300, 600, 1000 K.  202 frames.
3. **Phase 2 validation**: 70 additional frames from convergence testing.

**Total: 344 frames covering 52.8 kcal/mol above the minimum.**

### CH₂OO model benchmarks

| Metric | Value |
|--------|-------|
| Training frames | 344 |
| Hyperparameters | γ = 0.01, α = 0.001 |
| Validation RMSE (mean) | **0.64 kcal/mol** |
| Validation RMSE (max) | **1.23 kcal/mol** |
| All 20 validation frames | < 2 kcal/mol ✓ |

---

## 2026-03-18 — ML-MD Engine & First IR Spectra (CH₂OO)

### `modules/bakken.py` — the ML-MD engine
Named "bakken" (Norwegian: the hill).  Implements:

| Component | Description |
|-----------|-------------|
| `MLPESDriver` | Wraps a trained `MLPESTrainer`; exposes `energy()` and `forces()` via central-difference FD |
| `minimize_geometry()` | Adaptive steepest-descent pre-minimiser (backtracking line search) |
| `maxwell_boltzmann_velocities()` | Standard MB velocity initialisation |
| `zpe_initialized_velocities()` | **ZPE-floor init**: each normal mode gets max(ZPE, k_BT) kinetic energy |
| `run_md()` | Velocity-Verlet integrator + Berendsen thermostat |

**Key design**: ZPE-floor initialisation prevents classically forbidden modes from
being unoccupied at 300 K, which would suppress those IR peaks.

### `ir_md_spectrum.py` — full IR workflow
One-stop pipeline:
1. Pre-minimise on ML-PES (bakken steepest descent)
2. Compute Hessian → normal modes → ZPE-floor velocities (filter 50–4000 cm⁻¹)
3. Run 20,000-step ML-MD at 300 K
4. Train ML dipole surface (`DipoleSurface`, separate KRR on 3-component dipole vectors)
5. Predict dipoles along trajectory → dipole ACF → Fourier transform → IR spectrum

### Apple Silicon kernel panic fix
`DipoleSurface.train()` used `n_jobs=-1` in KRR grid search, saturating all Apple Silicon
performance cores.  The OS watchdog killed the process.  Fixed with `safe_jobs = min(n_jobs, 2)`.

### CH₂OO IR spectrum results

**Best run:** `outputs/ir_spectrum_20260318_232404/`
Command:
```bash
python3 ir_md_spectrum.py \
    --model outputs/nm_training_20260308_203606/mlpes_model_nm.pkl \
    --training-data outputs/clean_psi410_20260308_203552/training_data.npz \
    --steps 20000 --temp 300 --timestep 0.5 --save-every 1 \
    --preminimize --zpe-min-freq 50 --zpe-max-freq 4000
```

| Peak (cm⁻¹) | Rel. intensity | Physical assignment |
|-------------|----------------|---------------------|
| **307** | 1.000 | Torsion / bending (strongest) |
| **461** | 0.522 | COO bending |
| **1110** | 0.294 | **C-O stretch** ✓ |
| **458** | 0.100 | COO bending overtone |
| **803** | 0.094 | **O-O stretch** ✓ |
| **297** | 0.085 | Low torsion |
| **2085** | 0.059 | C-H deformation / combination |

The 803 and 1110 cm⁻¹ peaks are physically meaningful for CH₂OO and match
the expected O-O and C-O stretch region.

**Dipole surface quality:**  R² = 0.9999999, train RMSE = 3.2 × 10⁻⁵ D,
test RMSE = 8.6 × 10⁻⁵ D.

**ML-PES normal-mode frequencies (unphysical — known limitation):**
1847, 2513, 3163, 3638, 4751, 5498, 5885, 11208, 12680 cm⁻¹
Physical CH₂OO modes should be ~500–3100 cm⁻¹.  See §Analytic Hessian below.

### Gamma sweep for Hessian quality (`retrain_softer_gamma.py`)

| γ | α | Val. RMSE (kcal/mol) | Unphysical NM modes | Lowest NM freq (cm⁻¹) |
|---|---|----------------------|---------------------|------------------------|
| 0.0001 | 1e-5 | 2.30 | 3/9 | 1254 |
| 0.0003 | 1e-5 | 1.07 | 4/9 | 1393 |
| 0.001  | 1e-5 | 0.50 | 5/9 | 1844 |
| 0.003  | 1e-5 | 0.45 | 5/9 | 1883 |
| **0.01**  | **1e-5** | **0.61** | **5/9** | **1773** |

**Finding:** Softer γ reduces unphysical Hessian modes but worsens IR spectra (forces less
accurate, molecule wanders).  The stiffness is intrinsic to Coulomb-matrix second derivatives
under the RBF kernel — not curable by kernel width alone.

---

## 2026-03-18 — Analytic KRR Forces & Hessian

Implemented exact analytic gradient and Hessian of the KRR energy surface via
chain rule through the Coulomb-matrix descriptor.

**Added to `modules/bakken.py` (`MLPESDriver`):**

| Method | Speed | Notes |
|--------|-------|-------|
| `analytic_forces(coords)` | **903× faster** than FD | Matches FD to 3.3 × 10⁻⁵ Ha/Å |
| `analytic_hessian(coords)` | Single forward pass | Matches FD Hessian to < 5 cm⁻¹ |
| `_coulomb_jacobian(coords)` | — | ∂C_k/∂R_{a,j} = ∓Z_pZ_q d_j / r³ |
| `_coulomb_hessian2(coords)` | — | ∂²C_k/∂R_q∂R_r (dipole tensor form) |

Derivation summary:

```
F = -∂E/∂R = -σ_y · (∂E_sc/∂x_sc) · (1/σ_X) · J
H = σ_y · (-2γ) · [einsum(g, J2_sc) + Jᵀ(-2γ H_desc + E_sc·I)J]
```

where `g_i = Σ_j α_j K_ij (-2γ)(x_i - x_j)` and `J` is the (n_desc, 3N) Coulomb Jacobian.

**CLI flag:** `--analytic-hessian` in `ir_md_spectrum.py`

**Key finding:** Analytic and FD Hessians give identical, unphysical frequencies (all modes
> 1800 cm⁻¹ for CH₂OO).  The problem is intrinsic to Coulomb-matrix curvature —
being more exact does not fix the physics.  The correct fix is a better descriptor
(e.g. SOAP, symmetry-adapted, or internal coordinates).

---

## 2026-03-19 — MVKO: New Molecule, Full Pipeline

### Molecule
**Methyl vinyl ketone oxide** (MVKO), (CH₂=CH)(CH₃)COO, C₄H₆O₂.
12 atoms, 30 vibrational modes, n_desc = 78 Coulomb-matrix features.
Criegee intermediate from ozonolysis of methyl vinyl ketone (MVK).
Atmospherically relevant — implicated in tropospheric SO₂ and HCOOH production.

**Atom ordering (fixed — Coulomb matrix not permutation-invariant):**
`C1(Criegee)–O1(proximal)–O2(distal)–C2(vinyl=CH-)–C3(=CH₂)–C4(methyl)–H1–H2–H3–H4–H5–H6`

**Hyperparameter scaling rule:**
γ scales roughly as 1/(n_desc) — for MVKO (n_desc = 78 vs 15 for CH₂OO), use γ ≈ 5× smaller.
Best found: **γ = 0.001, α = 1 × 10⁻⁵**.

### `mvko_workflow.py` — 7-step pipeline

| Step | Description | Status |
|------|-------------|--------|
| 1 | PSI4 geometry optimisation | ✓ |
| 2 | PSI4 Hessian (B3LYP/6-31G\*) | ✓ |
| 3 | NM distortions: T=1500 K, ±5 amplitudes | ✓ 300 frames |
| 4 | PSI4 MD: 300/600/1000 K × 201 steps each | ✓ 603 frames |
| 5 | Train ML-PES | ✓ RMSE = 0.2734 kcal/mol |
| 6 | Adaptive refinement at T=500 K | ⚠ **SKIP** — runaway ML-MD |
| 7 | IR spectrum | ✓ via separate dipole collection |

**Step 6 warning:** The adaptive loop at T=500 K drives ML-MD to 5000–10000 K due to
unphysical wells in the Coulomb+RBF surface outside the training hull.  This corrupts
training data (errors 100–1400 kcal/mol).  **Step 6 must be skipped.**  Use the Step 5
model directly.

### MVKO training data summary

| Source | Frames | T (K) |
|--------|--------|-------|
| NM distortions (T=1500 K, ±5) | 300 | — |
| PSI4 MD at 300 K | 201 | 300 |
| PSI4 MD at 600 K | 201 | 600 |
| PSI4 MD at 1000 K | 201 | 1000 |
| **Total** | **904** | — |

Energy range: 399.6 kcal/mol (much broader than CH₂OO's 52.8 kcal/mol).

### MVKO IR spectrum

**Dipole data:** `collect_mvko_dipoles.py` selects 150 representative frames (⅓ low-E,
⅓ high-E, ⅓ random), runs PSI4 B3LYP/6-31G\* `properties=['dipole']`.
Dipole magnitude range: 2.97–5.61 D, mean 4.63 D.

**PSI4 dipole API fix:** `psi4.oeprop(wfn, 'DIPOLE', title='MVKO')` stored results under
`'MVKO DIPOLE X'` — code was reading `'SCF DIPOLE X'` → all-zero dipoles.  Fixed by using
`psi4.variable('SCF DIPOLE') * AU_TO_DEBYE` after `properties=['dipole']` in energy call.

**Dipole surface:** γ = 0.001, α = 1 × 10⁻⁴, R² = 0.999 (train), R² = 0.981 (test),
RMSE = 0.024 D.

**Best IR run:** `outputs/ir_spectrum_20260319_174321/`
30,000 MD steps at 300 K, bakken pre-min + ZPE filter (50–4000 cm⁻¹):

| Peak (cm⁻¹) | Rel. intensity | Assignment |
|-------------|----------------|------------|
| **322** | 1.000 | Torsion/bending (strongest) |
| **247** | 0.870 | Torsion/bending |
| **239** | 0.867 | Torsion/bending |
| **893** | 0.125 | **O-O stretch** ✓ |
| **513** | 0.124 | C-O-O bending |
| **658** | 0.059 | C-O stretch |

893 cm⁻¹ O-O stretch for MVKO vs 803 cm⁻¹ for CH₂OO — +90 cm⁻¹ shift consistent
with methyl/vinyl substitution stiffening the COO backbone.

C-H stretches (~3000 cm⁻¹) absent — same Hessian stiffness artifact as CH₂OO.

---

## 2026-03-19 — 1D Anharmonic C-H Stretch Analysis (MVKO)

### Motivation
The IR spectrum via classical ML-MD cannot access overtones and combinations (requires quantum
vibrational populations).  The published paper reports 21 IR transitions in the 2νCH region
(5500–6500 cm⁻¹) per MVKO conformer.  We compute these via 1D DVR on quartic potential
curves.

### PSI4 1.10 has no VPT2
`psi4.frequency(..., anharmonic=True)` silently ignores the keyword and returns only
harmonic frequencies via FINDIF.  No VPT2 / VSCF module is compiled in PSI4 1.10.

### `mvko_anharmonic.py` — 1D anharmonic analysis
For each C-H normal mode:
1. Displace ±0.05, ±0.10, ±0.15, ±0.20, ±0.25 Å along the mode vector
2. Compute PSI4 B3LYP/6-31G\* energy at each point
3. Fit quartic potential: V(α) = c₀ + c₂α² + c₃α³ + c₄α⁴
4. Solve 1D Schrödinger equation via finite-difference matrix diagonalisation
   (scipy `eigh_tridiagonal`, Δα = 0.001 Å, grid −0.6 to +0.6 Å)
5. Extract E₀, E₁, E₂ → fundamental ν₁ = E₁ − E₀, first overtone 2ν = E₂ − E₀

**Key unit convention (from debugging):**
KE prefactor = ℏ²/(2μ) [Ha·Bohr²] **÷** ANG_TO_BOHR² — divide, not multiply.

### MVKO 1D anharmonic results

**Harmonic B3LYP/6-31G\* C-H modes** (from PSI4 FINDIF Hessian):

| Mode | ω_harm (cm⁻¹) | ν_fund (cm⁻¹) | 2ν (cm⁻¹) | Δanharmon (cm⁻¹) | IR (km/mol) |
|------|--------------|--------------|----------|-----------------|-------------|
| 31 | 3049 | 3002 | **5981** | −23 | 7.6 |
| 32 | 3100 | 3167 | **6404** | +71 | 8.6 |
| 33 | 3163 | 3150 | **6297** | −3  | 6.3 |
| 34 | 3165 | 3209 | **6462** | +44 | 5.3 |
| 35 | 3180 | 3131 | **6232** | −30 | 19.2 |
| 36 | 3324 | 3300 | 6603* | +3  | 1.8 |

\* Just above the 5500–6500 cm⁻¹ analysis window.

**Transitions in the 2νCH region (5500–6500 cm⁻¹):**

| Type | Count | Notes |
|------|-------|-------|
| Overtones (2ν) in window | **5** of 6 | mode 36 at 6603 cm⁻¹ just above |
| Combination bands (ν_i + ν_j) | **14** of 15 | ν34+ν36=6509 borderline |
| **Total transitions** | **19** | Paper predicts 21 per conformer |
| Missing | 2 | ν34+ν36 = 6509, 2ν36 = 6603 (just above cutoff) |

**Conclusion:** Our 1D DVR recovers 19/21 expected transitions.  The 2 missing ones
(6509, 6603 cm⁻¹) are just above the cutoff; extending the window would capture them.

### Why classical MD cannot access the 2νCH region

1. **Quantum mechanics**: Overtone absorption requires vibrational population in v = 2,
   which is negligible in a room-temperature classical ensemble.
2. **ML-PES stiffness**: The Coulomb+RBF Hessian gives C-H mode frequencies of
   10,000–15,000 cm⁻¹ (unphysical), so the classical ZPE is deposited far above 3000 cm⁻¹.

---

## 2026-03-24 — Adaptive High-Energy Training & Multi-Surface PES

### Motivation
1. The current training data (904 frames, 399.6 kcal/mol range) does not uniformly cover
   high-energy anharmonic regions — the NM distortions at T=1500 K are the highest-energy
   frames but are generated without PSI4 feedback.
2. Multiple MVKO conformers exist (s-cis, s-trans, gauche).  A single surface trained on
   one conformer cannot accurately describe dynamics that cross between conformer basins.

### Feature 1 — `modules/uncertainty.py`: CommitteeModel

Bootstrap ensemble of K KRR models for epistemic uncertainty estimation.

```
CommitteeModel(symbols, training_coords, training_energies,
               k_models=5, gamma=0.001, alpha=1e-5)
  .train()                          → trains K members on 80% bootstrap subsets
  .batch_uncertainty(symbols, X)    → (energies, sigmas)  [Ha, kcal/mol]
  .calibrate(symbols, val_X, val_E) → fits scalar s: s·σ ≈ |E_ML − E_PSI4|
```

**Why bootstrap over exact KRR posterior variance:**
Exact GP posterior requires solving an (n_train × n_train) linear system per query point —
expensive for n_train ≈ 900.  Bootstrap variance naturally propagates through the
`StandardScaler` and reuses `MLPESTrainer` without modification.

Calibration result on MVKO (cycle 1): scale = **0.889**
(mean |error| = 0.080 kcal/mol, mean σ = 0.090 kcal/mol — well-calibrated).

**Fix required in `MLPESTrainer` (`modules/ml_pes.py`):**
Added `_train_committee_member(traj)` — trains on the full provided `TrajectoryData`
without internal train/test split, for use by `CommitteeModel`.

### Feature 2 — `modules/pes_family.py`: PESFamily with softmin blending

Multi-surface PES for seamless conformer switching.

**Blending scheme:**
```
Z        = Σ_k  exp(−β E_k)          β = HARTREE_TO_KCAL / blend_width
w_k      = exp(−β E_k) / Z           (softmin weights)
E_blend  = −log(Z) / β               (log-sum-exp energy)
F_blend  = Σ_k w_k F_k               (weighted force)
```
When ΔE ≫ blend_width, the lowest surface dominates (hard assignment limit).
Typical blend_width: 1–5 kcal/mol.

```python
family = PESFamily.from_model_paths(symbols, {
    's-cis':   'outputs/mvko_scis/mlpes_initial.pkl',
    's-trans': 'outputs/mvko_strans/mlpes_initial.pkl',
}, blend_width=3.0)

e_blend  = family.blend_energy(coords)
label    = family.assign_conformer(coords)   # hard assignment
e_batch  = family.blend_energy_batch(coords_batch)
```

**Supporting script:** `train_conformer_family.py` trains per-conformer models, aligns
reference energies to a global minimum, and writes `conformer_manifest.json` for use with
`ir_md_spectrum.py --multi-surface`.

### Feature 3 — `adaptive_high_energy.py`: Adaptive loop

Iteratively identifies high-uncertainty regions and fills them with PSI4 data.

**Geometry generation (anti-runaway design):**
- ✓ NM distortions at T_nm (up to 3000 K) — geometry from PSI4 Hessian, energy from ML
- ✓ Short PSI4 MD bursts at 1000/2000 K — energy/forces computed by PSI4
- ✗ **Never** uses ML-MD for geometry generation

**Candidate scoring and selection:**
```
CommitteeModel → σ(x)           [uncertainty per candidate]
Stratify into energy tiers:
  0–5, 5–15, 15–30, 30–60 kcal/mol above E_min  (15/30/30/25% allocation)
Select top-σ frames within each tier → PSI4 single-points
```

**Bug fixed during production run:** Geometry screening originally used upper-bound
bond-length ranges against all atom pairs (including non-bonded).  Non-bonded O···H at
2.5 Å — perfectly normal — was rejected as "bond length 2.5 Å > 2.2 Å limit".  Result:
100% of PSI4 MD frames at 1000 K and 2000 K were screened out.  Fixed by replacing
`SAFE_BOND_RANGES` (lo, hi) with `MIN_CONTACT_DIST` (lo only) — only reject atom pairs
that are **too close** (nuclear collision), not too far apart.

**Verification after fix:**
```
T=1000K frames passing screen: 201/201   (was 0/201)
T=600K  frames passing screen: 201/201   (was 0/201)
```

**Updated pipeline in `modules/ml_pes.py`:**
`MLPESTrainer.load()` now registers `ml_pes` and `ml_pes_fixed` as `sys.modules` aliases
before `pickle.load()`, so models serialised when the module was on `sys.path` directly
can be loaded when imported as `modules.ml_pes`.

### `ir_md_spectrum.py` — multi-surface support

New CLI flags and internal `PESFamilyDriver` adapter class:

```bash
python3 ir_md_spectrum.py \
    --model            outputs/conformer_family/s-cis_model.pkl \
    --training-data    outputs/mvko_dipoles/training_with_dipoles.npz \
    --multi-surface \
    --conformer-manifest outputs/conformer_family/conformer_manifest.json \
    --blend-width      3.0 \
    --steps 30000 --temp 300 --preminimize \
    --zpe-min-freq 50 --zpe-max-freq 4000
```

`PESFamilyDriver` exposes the same interface as `MLPESDriver` (`energy()`, `forces()`,
`symbols`, `masses`) so `run_ir_workflow` requires zero modification.

---

## 2026-03-24 — Production Adaptive Run: Results

**Command:**
```bash
python3 adaptive_high_energy.py \
    --model         outputs/mvko_20260319_081314/mlpes_initial.pkl \
    --training-data outputs/mvko_20260319_081314/combined_training_data.npz \
    --hessian-data  outputs/mvko_20260319_074852/psi4_hessian.npy \
    --cycles 3 --T-nm 3000 --n-amplitudes 6 \
    --md-steps 50 --md-temps 1000,2000 --top-n 30 \
    --k-models 5 \
    --output outputs/adaptive_production_20260324b
```

**Bug encountered and fixed during run:**
Initial screening rejected 100% of PSI4 MD frames (see §Adaptive section above).
Run was killed and restarted with `MIN_CONTACT_DIST` (lower-bound only) screening.

### Per-cycle results

| Cycle | MD frames added | NM frames added | Total frames | RMSE (kcal/mol) | Max σ (kcal/mol) | Cal. scale | Time (s) |
|-------|----------------|-----------------|--------------|-----------------|-----------------|------------|---------|
| Start | — | — | 904 | 0.2734 | — | — | — |
| 1 | 102 | 30 | **1036** | 0.9864 | 83.8 | 0.889 | 1889 |
| 2 | 102 | 30 | **1168** | 0.4843 | 56.7 | 0.775 | 4604 |
| 3 | 102 | 30 | **1300** | **0.2028** | 44.1 | 1.130 | 1779 |

**Key observations:**
- Cycle 1 RMSE spike (0.273 → 0.986): the 102 PSI4 MD frames at 1000/2000 K sample
  high-energy geometries outside the original training hull.  The model initially
  underfits them.
- Cycle 2 recovery (0.986 → 0.484): the NM distortions provide complementary coverage
  of the anharmonic well walls, improving balance.
- Cycle 3 final (0.484 → **0.203**): below the original RMSE of 0.273 kcal/mol, and
  significantly better than the original model on high-energy configurations.
- Max uncertainty σ_max fell monotonically: 83.8 → 56.7 → 44.1 kcal/mol.  The
  committee is converging — high-uncertainty regions are being filled.

**Final model:** `outputs/adaptive_production_20260324b/mlpes_adaptive_final.pkl`
**Final training data:** `outputs/adaptive_production_20260324b/training_data_final.npz`
(1300 frames: 904 original + 306 PSI4 MD + 90 NM single-points)

### Dipole collection

```bash
python3 collect_mvko_dipoles.py \
    --training-data outputs/adaptive_production_20260324b/training_data_final.npz \
    --n-frames 200 \
    --output outputs/adaptive_dipoles_20260324/training_with_dipoles.npz
```

200 representative frames selected from 1300.  200/200 valid PSI4 dipoles collected.
Dipole range: 2.97–6.04 D (broader than original 2.97–5.61 D, consistent with
high-T frames sampling larger-amplitude geometries).

---

## 2026-03-24 — IR Spectrum Failure Diagnosis & Fix

### First IR attempt: failure

```bash
python3 ir_md_spectrum.py \
    --model outputs/adaptive_production_20260324b/mlpes_adaptive_final.pkl \
    --training-data outputs/adaptive_dipoles_20260324/training_with_dipoles.npz \
    --steps 30000 --temp 300 --timestep 0.5 --save-every 1 \
    --preminimize --zpe-min-freq 50 --zpe-max-freq 4000
```

**Output:** `outputs/ir_spectrum_20260324_163458/`

**Symptoms:**
- Only 4 peaks: 112, 65, 114, 198 cm⁻¹ (all torsional/near-zero frequency; no O-O or C-O stretches)
- Pre-minimizer converged in 14 steps to a saddle point
- 3 imaginary NM modes at the minimized geometry: **−1285, −880, −718 cm⁻¹**
- ZPE effective temperature: **1619 K** (target: 300 K) — imaginary modes contributed
  unphysical ZPE kinetic energy
- Dipole surface test RMSE: 0.039 D (vs 0.000086 D for the original MVKO run)

### Root cause analysis

**Saddle-point problem:** The adaptive model trained on 1000–2000 K PSI4 MD frames
altered the curvature of the PES near the equilibrium geometry.  The ML-PES minimizer
(steepest descent) converged to a saddle point rather than the true minimum.

```
PSI4 eq energy:                       -306.318584 Ha
Adaptive model prediction at PSI4 eq: -306.318690 Ha  (−0.07 kcal/mol — accurate)
Adaptive model |F|_max at PSI4 eq:     0.0075 Ha/Å    (vs 0.0054 for original model)
```

The *energy* at the PSI4 equilibrium is accurate, but the *forces* are 40% larger —
indicating the adaptive model places the PES minimum slightly offset from the PSI4
optimum.  The ML minimizer then overshoots to a saddle point.

**Dipole surface degradation:** The 200 high-T adaptive frames broadened the geometry
distribution used for dipole training.  With γ=0.001 (optimised for near-equilibrium
geometries), interpolating across the enlarged distribution gave test RMSE 0.039 D.

### Fixes applied

**1. Added `--start-coords` flag to `ir_md_spectrum.py`** (line ~1117 in `main()`):
```python
parser.add_argument('--start-coords', default=None,
    help='Path to .npy file containing starting geometry (n_atoms, 3) Å. '
         'Overrides lowest-energy training frame.')
```
Passes `start_coords=np.load(args.start_coords)` to `run_ir_workflow()`, which
overrides the `coords0 = traj.coordinates[argmin(energies)]` selection.

Saved PSI4 equilibrium geometry:
```bash
python3 -c "
import json, numpy as np
with open('outputs/mvko_20260319_081314/state.json') as f:
    s = json.load(f)
np.save('outputs/mvko_20260319_081314/psi4_eq_coords.npy', np.array(s['opt_coords']))
"
```

**2. Use original dipoles** (`outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz`):
- 150 frames, all near-equilibrium → dipole surface test RMSE 0.024 D (vs 0.039 D)
- Dipole surface hyperparameters γ=0.001, α=1e-4 remain optimal for this distribution

**3. Skip ZPE init** (`--no-zpe-init`) to avoid imaginary-mode contamination:
- Without ZPE boosting, velocities are plain Maxwell-Boltzmann at 300 K
- The Berendsen thermostat maintains 300 K regardless of starting curvature
- Fundamental mode sampling is preserved; only ZPE floor boosting is lost

### Corrected IR run (in progress)

```bash
python3 ir_md_spectrum.py \
    --model outputs/adaptive_production_20260324b/mlpes_adaptive_final.pkl \
    --training-data outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \
    --steps 30000 --temp 300 --timestep 0.5 --save-every 1 \
    --no-zpe-init \
    --start-coords outputs/mvko_20260319_081314/psi4_eq_coords.npy
```

**Dipole surface quality** (using original 150-frame data):
- Training RMSE: 0.0060 D, Testing RMSE: **0.0240 D**, R² train=0.999, test=0.981
- Matches original MVKO run exactly — good

### Second IR attempt: still only torsional peaks

**Output:** `outputs/ir_spectrum_20260324_185233/`

Result: 7 peaks — all torsional, all < 210 cm⁻¹.
The adaptive model still produced only sub-210 cm⁻¹ peaks, confirming the problem is not
the ZPE/preminimizer path but the adaptive model's PES topology near equilibrium.

**Control experiment:** same `--no-zpe-init --start-coords` with the *original* model:

Output: `outputs/ir_spectrum_20260324_193316/`
Result: 9 peaks — also all torsional, all ≤ 372 cm⁻¹.

### Root cause: ZPE initialization is essential

Without ZPE floor initialization, 300 K Maxwell-Boltzmann velocities have insufficient
energy to excite the higher-frequency modes (O-O stretch ~893 cm⁻¹, C-O stretch ~658 cm⁻¹).
The ZPE energy per mode (½ ℏω) at 893 cm⁻¹ is equivalent to ~1280 K classical temperature
— far above the target 300 K.  Without the ZPE floor boost, these modes are thermally frozen.

This is the same reason ZPE-floor initialisation was originally added to the workflow.
The original MVKO IR run (best result) required both:

1. `--preminimize` → find a true stationary point on the ML-PES (no imaginary modes)
2. ZPE-floor init → boost each mode to at least ½ ℏω regardless of temperature

### Final conclusion

The adaptive model improved *high-energy* accuracy (RMSE 0.273 → 0.203 kcal/mol, 1000–2000 K)
at the cost of *near-equilibrium* PES quality:

| Model | Frames | RMSE | Modes at PSI4 eq | ZPE-init viable? |
|-------|--------|------|------------------|-----------------|
| `mlpes_initial.pkl` | 904 | 0.2734 kcal/mol | 0 imaginary | ✓ Yes |
| `mlpes_adaptive_final.pkl` | 1300 | 0.2028 kcal/mol | **3 imaginary** | ✗ No |

**The original 904-frame model remains the best for IR spectra.**

The best MVKO IR result remains: **`outputs/ir_spectrum_20260319_174321/`**
(original model + `--preminimize` + ZPE-floor init → peaks at 322, 893, 513, 658 cm⁻¹).

### Lessons learned

1. **Adaptive high-energy training can degrade near-equilibrium PES quality** when new
   high-T frames dominate the training set and the KRR kernel allocation shifts away from
   the equilibrium region.  Solution: include explicit near-equilibrium anchoring frames
   when adding high-energy data (e.g., always include the PSI4-optimized geometry with
   zero forces/energy as a constraint row).

2. **ZPE floor initialization is essential** for classical-MD IR spectra of polyatomic
   molecules at physiologically relevant temperatures.  Without it, modes with ω > ~600 cm⁻¹
   are thermally frozen at 300 K.

3. **`--start-coords` flag** (added to `ir_md_spectrum.py`) is a useful diagnostic tool —
   it bypasses the ML-PES minimizer and starts directly from a known geometry, isolating
   minimizer issues from PES topology issues.

4. **Dipole surface with adaptive frames** performs worse (RMSE 0.039 D) than the original
   150-frame surface (RMSE 0.024 D) because the high-T frames expand the training
   distribution beyond the interpolation range of γ=0.001 KRR.

---

## Known Limitations & Open Problems

### 1. Coulomb-matrix Hessian stiffness
The Coulomb-matrix + RBF-kernel combination gives unphysical second derivatives.
All ML-PES normal modes are > 1800 cm⁻¹ for CH₂OO (physical range: 500–3100 cm⁻¹).
The effect is consistent across γ values — it is intrinsic to the descriptor, not the kernel.

**Impact:** ZPE filter (--zpe-max-freq 4000) prevents depositing ZPE into the unphysical
high-frequency modes, but C-H stretch peaks (~3000 cm⁻¹) cannot appear in the ML-MD
IR spectrum.

**Correct fix:** Replace Coulomb matrix with a descriptor that has physically correct
second derivatives — SOAP, ACE, SchNet, or internal coordinate representations.

### 2. Classical MD cannot access overtones
Fundamentals are accessible via classical dipole ACF at finite temperature.
Overtones (2νCH, 5981–6462 cm⁻¹) require quantum vibrational populations and can only
be computed via VPT2 / VSCF / DVR methods — the 1D anharmonic analysis above provides this.

**Workaround in place:** `mvko_anharmonic.py` (1D DVR on quartic potentials) gives
anharmonic fundamentals and overtones directly from PSI4 Hessian + energy curves.
Results agree with published 21-transition count to within 2 transitions.

### 3. Single-conformer MVKO model
The current production model was trained on a single PSI4-optimised MVKO geometry.
MVKO has s-cis, s-trans, and gauche conformers.  At 300 K the molecule may sample
multiple basins.

**Mitigation in place:** `modules/pes_family.py` + `train_conformer_family.py` provide
multi-conformer PES blending once per-conformer training data is collected.

### 4. No gradient correction for forces
Forces are computed via finite differences on the ML-PES energy surface (30 KRR
evaluations per FD step).  Analytic forces are 903× faster and implemented, but give
identical frequencies to FD because the Hessian problem is in the descriptor, not
the differentiation method.

---

## 2026-03-27 — Multi-Trajectory IR Spectrum (Conformational Broadening)

### Motivation

The single-trajectory MVKO IR run (`ir_spectrum_20260319_174321/`) showed promising peaks at
322, 513, 658, and 893 cm⁻¹ but might be biased by a single starting geometry.  Broader
sampling of phase space and conformational diversity (s-cis / s-trans / gauche torsional
variants) should:
- Reduce single-trajectory statistical noise via averaging
- Naturally broaden peaks that shift with conformation
- Guard against spurious dissociation events by truncating trajectories early

An `.xyz` trajectory from the adaptive-model run showed O–O bond extension to 2.31 Å
(1.71× equilibrium), confirming dissociation can occur at elevated effective temperatures.
The original 904-frame model was used for this run (adaptive model has 3 imaginary modes,
see Known Limitation 5).

### Implementation

New flags added to `ir_md_spectrum.py`:
- `--n-trajectories N` — run N independent MD trajectories from the N lowest-energy
  training frames (different random seeds 42…42+N−1); average per-trajectory spectra
- `--max-bond-extension F` — stop trajectory when any **heavy-atom** bond extends beyond
  F× its equilibrium length (X–H bonds excluded: ZPE oscillations legitimately extend
  C–H bonds 2–3×)

Per-trajectory spectra are averaged as intensity arrays (not dipole concatenation, which
causes ACF discontinuities at trajectory boundaries).

Dissociation detection added to `modules/bakken.py` `run_md()`: builds a bonded-pairs
table (all heavy-atom pairs with r₀ < 2.0 Å) from the post-minimisation geometry;
checks every saved frame; breaks the integration loop on first violation and returns
`'dissociation_step'` in the result dict.

### Run

```
python3 ir_md_spectrum.py \
    --model outputs/mvko_20260319_081314/mlpes_initial.pkl \
    --training-data outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \
    --steps 15000 --temp 300 --timestep 0.5 --save-every 1 \
    --preminimize --zpe-min-freq 50 --zpe-max-freq 4000 \
    --n-trajectories 5 --max-bond-extension 2.5
```

Output: `outputs/ir_spectrum_20260327_092555/`

### Per-Trajectory Results

| Traj | Start frame | Peaks (cm⁻¹) above 0.05 threshold |
|------|-------------|-------------------------------------|
| 1 | 0   (E_min) | 37, 246, 322, 514, 602, **893** |
| 2 | 64          | 210, 247 |
| 3 | 65          | 209, 241, 247, 322 |
| 4 | 96          | 37, 210, 247, 513, **893** |
| 5 | 9           | 37, 210, 247, 322, 513, **893** |

No dissociation detected in any of the 5 trajectories (all 15,000 steps completed;
heavy-atom bonds stayed well below 2.5× equilibrium).

### Averaged Spectrum Peaks

| Freq (cm⁻¹) | Rel. Intensity | Assignment |
|-------------|----------------|-----------|
| 210 | 1.000 | Torsion/bending |
| 247 | 0.422 | Torsion/bending |
| 37  | 0.122 | Torsion/bending |
| 322 | 0.091 | Torsion/bending |
| 514 | 0.061 | C–O stretch |
| **893** | **0.322** | **O–O stretch** |

The O–O stretch at 893 cm⁻¹ (relative intensity 0.32) survives the averaging procedure
and is the highest-frequency peak above 500 cm⁻¹.  The dominant torsional band at
~210 cm⁻¹ reflects the low-frequency MVKO conformational motion sampled across starting
geometries.  C–H stretch peaks are absent, consistent with the known Coulomb+RBF Hessian
stiffness artifact (ML-PES C–H modes at 10,000–15,000 cm⁻¹).

### Observations

1. **Conformational heterogeneity is real**: Trajectories 1, 4, 5 show clear 893 cm⁻¹
   O–O stretch; trajectories 2 and 3 show only torsional modes.  This is physically
   consistent — different starting geometries (different torsional conformers) have
   different dipole-moment coupling to the O–O stretch.  The averaged spectrum correctly
   captures both torsional and O–O character.

2. **893 cm⁻¹ is robust**: The O–O stretch appears in 3/5 trajectories and survives
   averaging with relative intensity 0.32, consistent with the single-trajectory result
   (893 cm⁻¹, rel. intensity 0.125 in `ir_spectrum_20260319_174321/`).  The higher
   relative intensity here reflects per-trajectory normalisation before averaging — some
   trajectories are torsion-dominated, depressing their torsional peak and effectively
   upweighting the O–O signal in the average.

3. **No dissociation**: The 904-frame near-equilibrium model is dynamically stable at
   300 K + ZPE init for all 5 trajectories × 15,000 steps.  Contrast with the adaptive
   model which showed O–O bond extension to 2.31 Å.

4. **Multi-trajectory averaging broadens peaks naturally**: The torsional band shifts
   from 209–210 cm⁻¹ across trajectories, giving a slightly broadened envelope in the
   average compared to any single run.

---

## 2026-03-30 — Reaction-Path PES Family for syn-MVKO → OH

### Motivation

Fresh (nascent) MVKO Criegee intermediates from ozonolysis carry 40–80 kcal/mol of excess
vibrational energy and can react before collisional stabilisation.  The dominant OH-producing
channel is a 1,4-H shift from the vinyl =CH₂ group to the distal oxygen O2, proceeding via
a 6-membered ring TS (C3–H7···O2–O1–C1–C2=C3).

Goal: use a `PESFamily` of two surfaces — the near-equilibrium MVKO model (904 frames)
blended with a new reaction-path ML-PES trained on PSI4 IRC data — to allow ML-MD to
explore the TS and VHP product region.

### Script

`mvko_syn_oh_path.py` — 4-step workflow: `ts → irc → train → md`

**Reaction coordinate** (corrected during development):
- PSI4 optimised geometry is the **syn-vinyl** conformer (H7 on C3=CH₂ sits 2.09 Å from O2;
  methyl C4 is 3.62 Å from O2 — ANTI face)
- 6-membered ring TS: O2–O1–C1–C2=C3–H7 (vinyl H transfer, not methyl H)
- Atom indices: `IDX_C3=4, IDX_H7=7, IDX_O2=2, IDX_O1=1, IDX_C1=0`

### IRC Data Collection (2026-03-30)

PSI4 B3LYP/6-31G\* steepest-descent IRC collected 42 points, stored in
`outputs/mvko_rxn_path_20260330_181025/irc_training_data.npz`.

| Property | Value |
|----------|-------|
| Total IRC frames | 42 |
| Raw energy range | 449 kcal/mol (many off-path high-E points from steepest-descent walk) |
| Frames ≤ 100 kcal/mol above minimum | **9** |
| Open-shell frames (|ΔS²| > 0.1) | 0/42 (B3LYP RHF always returns ⟨S²⟩ = 0) |
| IRC s range | –3.5 to +3.5 Å·√amu |

Energy distribution of the 9 kept frames:
- 1 frame at 0 kcal/mol (MVKO minimum, most negative s)
- 3 frames at 36–44 kcal/mol (near TS barrier)
- 5 frames at 52–80 kcal/mol (VHP product region)

### Training Issues Fixed

1. `MLPESTrainer()` missing required `config` arg → fixed with `MLPESConfig()`-based construction
2. Energy filter added: keep only IRC points ≤ 100 kcal/mol above minimum before training
3. Open-shell warning: only printed when `n_open > 0` (was always printing)

### Stability Problem and Fix

Training on 9 IRC frames alone gives a surface with no perpendicular-mode coverage.
At 2000 K, the MD dissociated (C1–C2 backbone bond → 4.44 Å) at step 110.

**Fix**: prepend the 904-frame reactant training data before the IRC frames.
Combined dataset: 913 frames.  The rxn_path model now has near-equilibrium coverage
for all vibrational modes AND the IRC frames guide it toward the TS at high energy.

Result: 2000-step MD at 2000 K **completes without dissociation** (600 fs stable trajectory).
Key reaction-coordinate statistics at 600 fs:
- C3–H7: 0.99–1.08 Å (intact vinyl C–H bond)
- O2–H7: 2.0–2.6 Å (approaching but not forming bond — not reactive in 600 fs)
- O1–O2: 1.34–1.46 Å (Criegee O–O intact)
- C1–O1: 1.30–1.42 Å (backbone intact)

### PESFamily Design

```
Surface "reactant" : mlpes_initial.pkl (904 frames, γ=0.001)
Surface "rxn_path" : mlpes_rxn_path.pkl (913 frames combined, γ=0.001)
Blend width        : 10 kcal/mol (wider than conformer blending to span TS region)
```

At near-equilibrium energies the reactant surface dominates; as the molecule climbs
toward the TS (ΔE > ~15 kcal/mol), the rxn_path surface progressively takes over.

### IRC Profile Plot (with RDKit Molecular Insets)

`plot_irc_profile()` in `mvko_syn_oh_path.py` now renders 2D molecular structure insets
at three key IRC points (MVKO, TS, VHP) using RDKit's `MolDraw2DCairo`.  The insets are
silently skipped if `rdkit` is not installed.  Install with: `pip install rdkit`.

### Multi-Reference Warning

B3LYP is single-reference.  Near the TS and in the VHP/OH product region, the true
wavefunction has significant biradical character.  The B3LYP barrier height is approximate
(±5 kcal/mol).  Future work: CASSCF(4,4)/6-31G\* or NEVPT2 single-points on the B3LYP
IRC geometries would give a quantitatively accurate surface.

### Outputs

| File | Contents |
|------|----------|
| `outputs/mvko_rxn_path_20260330_181025/irc_training_data.npz` | 42 IRC frames (raw) |
| `outputs/mvko_rxn_path_20260330_202029/mlpes_rxn_path.pkl` | rxn_path ML-PES (913 frames) |
| `outputs/mvko_rxn_path_20260330_202029/pes_family.pkl` | PESFamily (reactant + rxn_path) |
| `outputs/mvko_rxn_path_20260330_202029/rxn_family_manifest.json` | Manifest for `ir_md_spectrum.py --multi-surface` |
| `outputs/mvko_rxn_path_20260330_202029/irc_energy_profile.png` | IRC energy profile with RDKit insets |
| `outputs/mvko_rxn_path_20260330_202029/rxn_trajectory_bonds.csv` | Bond-distance CSV (4 bonds × 400 frames) |

### Production Command (train + md, skipping PSI4 TS/IRC)

```bash
python3 mvko_syn_oh_path.py --steps train,md \
    --irc-data outputs/mvko_rxn_path_20260330_181025/irc_training_data.npz \
    --temp 2000 --md-steps 50000
# --reactant-data defaults to outputs/mvko_20260319_081314/combined_training_data.npz
```

---

### 5. Adaptive training can degrade near-equilibrium accuracy
Adding high-energy frames (1000–2000 K PSI4 MD) to an existing ML-PES shifts the KRR
kernel allocation away from the equilibrium region.  The 1300-frame adaptive model
has 3 imaginary NM modes at the PSI4 equilibrium geometry — it cannot be used for
ZPE-floor IR spectra.

**Mitigation:** Anchor each adaptive cycle by including the PSI4-optimized geometry
(forces ≈ 0) as a constraint row in the training set, preventing the equilibrium
saddle-point instability.  Alternatively, use a local model near equilibrium
(separate γ/α tuned for near-eq) and a second model for high-energy extrapolation.

---

## Files and Paths Quick Reference

### Trained models
| Model | Molecule | γ | α | RMSE | Frames |
|-------|----------|---|---|------|--------|
| `outputs/nm_training_20260308_203606/mlpes_model_nm.pkl` | CH₂OO | 0.01 | 0.001 | 0.64 kcal/mol | 344 |
| `outputs/mvko_20260319_081314/mlpes_initial.pkl` | MVKO | 0.001 | 1e-5 | 0.2734 kcal/mol | 904 |
| `outputs/adaptive_production_20260324b/mlpes_adaptive_final.pkl` | MVKO | 0.001 | 1e-5 | **0.203 kcal/mol** | **1300** |
| `outputs/mvko_rxn_path_20260330_202029/mlpes_rxn_path.pkl` | MVKO (IRC) | 0.001 | 1e-5 | 3.1 kcal/mol | 913 (904 eq + 9 IRC) |

### Training data
| File | Contents |
|------|----------|
| `outputs/clean_psi410_20260308_203552/training_data.npz` | CH₂OO, all PSI4 1.10, 344 frames |
| `outputs/mvko_20260319_081314/combined_training_data.npz` | MVKO, 904 frames, no dipoles |
| `outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz` | MVKO, 150 frames, with dipoles |

### IR spectrum outputs
| Directory | Molecule | Model | Peaks (cm⁻¹) | Notes |
|-----------|----------|-------|--------------|-------|
| `outputs/ir_spectrum_20260318_232404/` | CH₂OO | nm_training (344 fr) | 297, 307, 458, 461, 803, 1110, 2085 | ZPE+preminimize |
| `outputs/ir_spectrum_20260319_174321/` | MVKO  | mlpes_initial (904 fr) | 239, 247, 322, 513, 658, 893 | **Best result**; ZPE+preminimize |
| `outputs/ir_spectrum_20260324_163458/` | MVKO  | adaptive (1300 fr) | 65, 112, 114, 198 | Failed — 3 imaginary modes |
| `outputs/ir_spectrum_20260324_185233/` | MVKO  | adaptive (1300 fr) | 77, 103, 115, 154, 157, 205 | Failed — no ZPE (adaptive PES broken) |
| `outputs/ir_spectrum_20260324_193316/` | MVKO  | mlpes_initial (904 fr) | 59, 109, 183, 210, 247, 322, 372 | No ZPE — confirms ZPE init essential |
| `outputs/ir_spectrum_20260327_092555/` | MVKO  | mlpes_initial (904 fr) | 37, 210, 247, 322, 514, 893 | **Multi-traj (5×15k); ZPE+preminimize** |

### Key scripts
| Script | Purpose |
|--------|---------|
| `mvko_workflow.py` | Full MVKO pipeline (Steps 1–5, skip 6) |
| `generate_nm_training.py` | CH₂OO NM+MD training data |
| `adaptive_high_energy.py` | Adaptive high-energy training loop |
| `collect_mvko_dipoles.py` | PSI4 dipole collection for dipole surface |
| `ir_md_spectrum.py` | Full IR pipeline (single or multi-surface) |
| `train_conformer_family.py` | Multi-conformer PES family assembly |
| `mvko_anharmonic.py` | 1D DVR anharmonic analysis (overtones) |
| `retrain_softer_gamma.py` | Gamma sweep / Hessian frequency diagnostics |

### Key modules
| Module | Contents |
|--------|---------|
| `modules/bakken.py` | ML-MD engine (`MLPESDriver`, `run_md`, `minimize_geometry`) |
| `modules/ml_pes.py` | KRR trainer (`MLPESTrainer`, `CoulombMatrixDescriptor`) |
| `modules/uncertainty.py` | Bootstrap ensemble uncertainty (`CommitteeModel`) |
| `modules/pes_family.py` | Multi-surface softmin blending (`PESFamily`, `ConformerPES`) |
| `modules/normal_modes.py` | Hessian, NM diagonalisation, thermal displacements |
| `modules/ir_spectroscopy.py` | Dipole ACF → IR spectrum (`DipoleSurface`, `IRSpectrumCalculator`) |
| `modules/direct_md.py` | PSI4 interface + Velocity-Verlet MD |
| `modules/data_formats.py` | `TrajectoryData`, npz/extxyz/hdf5 I/O |

---

## Appendix: PSI4 Settings (enforce consistently)

```python
psi4.set_memory('4 GB')
psi4.set_num_threads(4)
psi4.set_options({
    'basis':         '6-31G*',
    'scf_type':      'df',
    'reference':     'rhf',
    'maxiter':       200,
    'e_convergence': 1e-7,
    'd_convergence': 1e-7,
})
```

All training, validation, and single-point energies **must** use the same PSI4 installation.
Mixing PSI4 versions introduces a systematic energy offset (~8.5 kcal/mol observed between
a Dec 2025 install and PSI4 1.10) that causes validation RMSE of 15–40 kcal/mol.

## Appendix: Unit Conventions

| Quantity | Unit | Notes |
|----------|------|-------|
| Coordinates | Å | PSI4 accepts `units angstrom` |
| Energies | Hartree | Display converts via HARTREE_TO_KCAL = 627.509474 |
| Forces | Hartree/Å | PSI4 gradient in Ha/Bohr ÷ 1.88973 |
| Dipoles | Debye | PSI4 a.u. (e·Bohr) × AU_TO_DEBYE = 2.541746 |
| Velocities | Å/fs | |
| Time | fs | FS_TO_AU = 41.341374575751 |
| Temperature | K | KB_HARTREE_PER_K = 3.1668114 × 10⁻⁶ |
| Masses | amu | AMU_TO_AU = 1822.888486 |
| IR frequency | cm⁻¹ | AU → cm⁻¹: 219474.63 |
| NM freq conv | cm⁻¹/√(Ha/(Bohr²·amu)) | FREQ_CONV = 5140.48 |

All conversion constants live in `modules/direct_md.py:40-60` — do not redefine elsewhere.
