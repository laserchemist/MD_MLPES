# Research Log — MD_MLPES Project

**PI:** Jonathan M. Smith, Temple University  
**Topic:** Machine-learning potential energy surfaces for Criegee intermediate dynamics and IR spectroscopy

---

## Overview

This project develops machine-learning potential energy surfaces (ML-PES) for molecular dynamics simulations of Criegee intermediates, targeting IR spectra and multi-state electronic structure corrections. The two primary molecules are **CH₂OO** (5 atoms, carbonyl oxide) and **MVKO** (methyl vinyl ketone oxide, C₄H₆O₂, 12 atoms). The general pipeline is: PSI4 *ab initio* reference data → Coulomb-matrix descriptor → Kernel Ridge Regression ML-PES → ML-MD → IR spectrum via dipole autocorrelation or harmonic dipole derivatives.

---

## Phase 1 — Initial CH₂OO ML-PES Framework (December 2025)

**Commit:** `3215e6c` Initial commit: existing codebase

Established the two-layer architecture:
- `modules/` — core library (PSI4 interface, KRR trainer, data formats, visualization)
- Root directory — workflow orchestration scripts

Key design choices locked in:
- **Coulomb matrix descriptor**: upper-triangle of Z_i Z_j / r_ij (off-diagonal) and 0.5 Z_i^2.4 (diagonal). Simple, robust; atom ordering must be fixed (not permutation-invariant).
- **KRR with StandardScaler**: StandardScaler on both features and targets is critical. Without it RMSE degrades from ~0.04 to ~170 kcal/mol.
- **Force training disabled** (`MLPESConfig.train_forces = False`): enabling KRR force training broke the model. Forces obtained via finite differences on the energy surface.
- **PSI4 level of theory**: B3LYP/6-31G*, `scf_type=df`, `reference=rhf`, tight convergence (e: 1e-7, d: 1e-7).

Initial training data (December 2025): NM displacements + PSI4 MD at 300 K and 600 K.

---

## Phase 2 — PSI4 Version Crisis & Fix (February–March 2026)

**Commits:** `2978d1a`, `c0525a4`

**Problem**: The original December 2025 training data (B3LYP/6-31G*) gave energies ~8.5 kcal/mol lower than newly installed PSI4 1.10. Mixing old and new data caused catastrophic ML-PES errors (15–40 kcal/mol systematic offset in validation).

**Resolution**:
- Discarded all old training data.
- Generated clean training set under PSI4 1.10: `outputs/clean_psi410_20260308_203552/training_data.npz`
- **Rule enforced permanently**: never mix training data from different PSI4 installations.

Added `generate_nm_training.py` for systematic normal-mode displacement training data generation (72 displaced geometries covering all modes at multiple amplitudes).

---

## Phase 3 — CH₂OO Production Model (March 2026)

**Commits:** `c0525a4`, `c566687`, `f6bde5d`, `d7ffe38`, `1acc493`

**Best model**: `outputs/nm_training_20260308_203606/mlpes_model_nm.pkl`
- **344 frames**: 72 NM displacements + 202 PSI4 MD (300 K + 600 K) + 70 Phase 2 validation frames
- **Validation**: 0.64 kcal/mol mean error, 1.23 kcal/mol max
- **Hyperparameters**: γ = 0.01, α = 0.001 (RBF kernel)
- **CH₂OO true equilibrium** (PSI4 1.10): −189.5768 Ha; stored "eq" geometry was −189.5752 Ha (8.5 kcal/mol above true minimum)

**IR spectrum pipeline** (`ir_md_spectrum.py`):
- Bakken steepest-descent pre-minimization → ZPE-floor velocity initialization → ML-MD → dipole ACF → FFT → spectrum
- ZPE frequency filter [50, 4000] cm⁻¹ prevents unphysical mode boosting
- **Best CH₂OO IR spectrum**: `outputs/ir_spectrum_20260318_232404/` — 7 peaks at 297, 307, 458, 461, 803, 1110, 2085 cm⁻¹ (803 cm⁻¹ ≈ O-O stretch, 1110 cm⁻¹ ≈ C-O stretch)

**Bug fixed**: `DipoleSurface.train()` with `n_jobs=-1` caused kernel watchdog panic on Apple Silicon. Fixed with hard cap `safe_jobs = min(n_jobs, 2)`.

**Gamma sweep** (`retrain_softer_gamma.py`, `outputs/softer_gamma_20260318_234300/`): softer γ reduces unphysical Hessian modes (5→3 of 9) but harms IR spectrum quality. Hessian stiffness is intrinsic to Coulomb+RBF second derivatives — not fixable by hyperparameter tuning.

---

## Phase 4 — MVKO: New Molecule (March 2026)

**Commits:** `8f4b731`, `5cffad0`, `3d73e44`

Introduced **methyl vinyl ketone oxide** (MVKO): (CH₂=CH)(CH₃)COO, C₄H₆O₂, 12 atoms, 30 vibrational modes. Atmospherically relevant Criegee intermediate from ozonolysis of methyl vinyl ketone (MVK).

**Atom ordering** (fixed; Coulomb matrix not permutation-invariant):  
`C1(Criegee), O1(proximal), O2(distal), C2(vinyl=CH-), C3(=CH2), C4(methyl), H1(on C2), H2(on C3), H3(on C3), H4(CH3), H5(CH3), H6(CH3)`

**Descriptor dimension**: n_desc = 78 (Coulomb upper triangle for 12 atoms, vs 15 for CH₂OO)

**Hyperparameter guidance**: γ ~ 5× smaller than CH₂OO because n_desc is 5× larger.

**Base energy model**: `outputs/mvko_20260319_081314/mlpes_initial.pkl`  
- γ = 0.001, α = 1e-5, 904 frames, RMSE = 0.27 kcal/mol

**Dipole surface**: `outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz`  
- 150 frames; dipoles 2.97–5.61 D (mean 4.63 D); R² = 0.999, RMSE = 0.024 D

**Workflow**: `mvko_workflow.py` — 7 steps: PSI4 optimize → Hessian → NM data → MD data → train → adaptive → IR

**PSI4 dipole bug (fixed)**: `psi4.oeprop(wfn, 'DIPOLE', title='MVKO')` stored results under `'MVKO DIPOLE X'` but code queried `'SCF DIPOLE X'` → all-zero dipoles. Fix: use `properties=['dipole']` in gradient call + `psi4.variable('SCF DIPOLE') * AU_TO_DEBYE`, or `psi4.oeprop(wfn, 'DIPOLE')` (no title) + `wfn.variable('DIPOLE X')` (PSI4 1.10 returns Debye directly).

---

## Phase 5 — Adaptive Sampling & Lessons Learned (March 2026)

**Commits:** `9627c91`, `2978d1a`

**Adaptive sampling workflow** (`adaptive_high_energy.py`): NM distortions + short PSI4 MD bursts (never ML-MD) scored by CommitteeModel uncertainty, stratified by energy tier, top-N frames run through PSI4, retrained.

**Critical failure**: `outputs/adaptive_production_20260324b/mlpes_adaptive_final.pkl` (1300 frames, RMSE 0.203 kcal/mol) has **3 imaginary NM modes** at the PSI4 equilibrium (−1285, −880, −718 cm⁻¹). Pre-minimizer lands on a saddle point; ZPE initialization amplifies imaginary-mode velocities to ~1619 K. Without ZPE init, only torsional modes (<400 cm⁻¹) are sampled at 300 K.

**Root cause**: Adding 1000–2000 K PSI4 MD frames shifted the KRR kernel allocation away from equilibrium, displacing the ML-PES minimum from the PSI4 equilibrium.

**Fix for future adaptive runs**: Always include the PSI4-optimized geometry (forces ≈ 0) as an anchor row when adding high-energy frames.

**Rule**: For ZPE-floor IR spectra, always use the near-equilibrium model (`mlpes_initial.pkl`, 904 frames) with `--preminimize` + ZPE-filter [50, 4000] cm⁻¹.

Step 6 (adaptive loop at 500 K) of `mvko_workflow.py` causes runaway ML-MD (5000–10000 K) due to ML-PES unphysical wells → skip Step 6.

---

## Phase 6 — Analytic KRR Forces & Hessian (March 2026)

**Commit:** `f6bde5d` (bakken module)

Implemented exact analytic forces and Hessian in `modules/bakken.py` (`MLPESDriver`):
- `analytic_forces(coords)`: F = 2γ σ_y J_scaled^T g → **903× faster** than finite differences, matches FD to 3.3×10⁻⁵ Ha/Å
- `analytic_hessian(coords)`: exact H = σ_y(−2γ)[einsum(g, J2_sc) + J^T(−2γH_desc + E_sc·I)J], matches FD to <5 cm⁻¹
- Cached at load time: `_alpha_vec`, `_X_train_sc`, `_gamma_krr`, `_sx_mean`, `_sx_std`, `_sy_std`

**Key finding**: The analytic and FD Hessians give identical (unphysical) C-H frequencies. The problem is intrinsic to Coulomb+RBF curvature — not fixable by being more exact.

---

## Phase 7 — CASSCF Delta-ML Corrections (April 2026)

### 7a. CASSCF Full-Surface (March 2026) — Failed

`casscf_surface_correction.py`, 29 converged frames. Near-eq: δ = +0.62 ± 1.05 kcal/mol — negligible for 300 K IR. Coulomb-descriptor KRR failed (K≈0.999 clustering of all near-eq geometries); 1D spline failed (frame-29 anomaly δ = −19 kcal/mol). Superseded.

### 7b. Normal-Mode Coordinate Delta-ML (April 2026)

**Problem with Coulomb space**: All MVKO geometries at 300 K have K≈0.999 in Coulomb space (heavy-atom charges dominate). High-energy CASSCF corrections bleed into equilibrium because they are not far away in Coulomb space.

**Fix (`casscf_nm_delta.py`)**: KRR in normal-mode coordinate space q = U_vib^T · M^{1/2} · (R − R_ref). q = 0 at equilibrium; ||q||² grows monotonically with distortion energy. High-energy outliers have large ||q|| → no bleeding to equilibrium.

**Class**: `NMKRRDeltaModel`; integrated into `ir_md_spectrum.py` via `NMDeltaDriver` and `--nm-delta-model` flag.

### 7c. NEVPT2 Correction (April 2026) — Marginal

`casscf_nevpt2_correction.py`; PySCF CASSCF(4,4)/6-31G* + SC-NEVPT2; ~80 s/frame.  
- 14/50 training frames corrupted by CASSCF state-switching (δ = −15 to +15 kcal/mol at tiny displacements). Removed → 36 clean frames.
- **Clean model LOO-CV**: 1.60 kcal/mol (γ = 0.01, α = 1e-5)
- Physical significance: ozonolysis-energized MVKO (~50 kcal/mol internal energy, T_eff ≈ 840 K) accesses geometries where NEVPT2 corrects B3LYP by −9 to −21 kcal/mol (expected red-shift of O-O and C-O stretches)
- **Pickle bug fixed**: `NEVPTKRRModel` defined in `__main__` → unpicklable. Moved to `casscf_nm_delta.py` module scope.

---

## Phase 8 — C-H Stretch Retrain (April 2026) — Failed

Attempted to fix unphysical C-H ML-PES wall by adding 86 NM C-H stretch frames (filtered to 48 by ΔE < 500 kcal/mol threshold).

**Result**: `outputs/mvko_ch_retrain_20260403_101429/mlpes_ch_retrained.pkl` (952 frames) — **corrupts near-equilibrium PES** (3 imaginary modes, loses 893 cm⁻¹ O-O peak, near-eq RMSE increases). Adding high-energy frames always shifts KRR kernel away from equilibrium.

**Conclusion**: Address H-wandering at MD level (bond guard), not by adding high-energy data. Best IR model remains `outputs/mvko_20260319_081314/mlpes_initial.pkl` (904 frames).

---

## Phase 9 — sGDML Investigation (April 2026) — Failed

Evaluated sGDML (symmetric Gradient Domain Machine Learning) as an alternative to Coulomb+KRR.

**Results**: FAILED — same 1/r descriptor stiffness as Coulomb+KRR (38,342 cm⁻¹ max C-H mode; physical: ~3,000 cm⁻¹). Additional problems:
1. Multi-T+NM training data violates sGDML path-integral coherence assumption (89× force-integral mismatch)
2. Kernel memory: 23.5 GB at n_train = 900, must cap at 300
3. With `use_E_cstr` + dE < 5 filter: near-eq RMSE 0.69 kcal/mol but 192 kcal/mol on full set, 8 imaginary modes, MD dissociates at step 40

**Root cause diagnosis**: Both Coulomb+KRR and sGDML use 1/r-based descriptors whose second derivatives are intrinsically stiff under RBF/Matérn kernels. This cannot be fixed by hyperparameter tuning.

---

## Phase 10 — MACE Backend (April 2026)

**Commits:** `921bf44`, subsequent

**Solution to IR frequency problem**: MACE (Multi-Atomic Cluster Expansion) — SO(3)-equivariant message-passing neural network with local atomic energy decomposition + force-weighted training → physical Hessian by design.

**Infrastructure implemented**:
- `modules/mace_pes.py`: `MACEDriver` (bakken-compatible), `npz_to_extxyz()` (Ha→eV conversion, extxyz format with `REF_energy`/`REF_forces` keys, 20 Å cube cell, energy filter)
- `train_mace_model.py`: npz→extxyz + `mace_run_train` subprocess + checkpoint discovery, saves `mace_model.pt` + `mace_model.symbols.pkl`
- `validate_pes_frequencies.py`: generalized validation (`--mace-model`/`--sgdml-model`/`--model`), PASS/FAIL/WARN output
- `ir_md_spectrum.py`: `--mace-model` flag added

**Architecture**: `hidden_irreps='64x0e + 64x1o'`, `r_max = 5.0 Å`, 2 interaction layers, `forces_weight = 100`, float64, SWA

**MACE fw10 retrain (June 2026)** — concluded: `forces_weight=10`, 500 epochs; loss completely plateaued at epoch 100. Not validated — moot since ACF IR approach is fundamentally unsuitable for MVKO (see Phase 12).

**Key notes**:
- Requires `mace-torch >= 0.3.4`, `ase`, PyTorch with MPS backend
- `MACEDriver` requires `.symbols.pkl` companion file alongside `.pt`
- MACE composable with `NMDeltaDriver` for CASSCF δ_S0 correction (same interface)

---

## Phase 11 — Multi-Conformer CASSCF Grid (April–May 2026)

**Commits:** `709a4b9`, `be0c094`, `3418ee4`, `c3719c9`

MVKO has four stable conformers (syn/anti × trans/cis of the vinyl–COO dihedral). Computed CASSCF NM-displacement grids for all four to build multi-state delta-ML corrections.

### Active Space Choices

| Conformer | Active Space | Rationale |
|-----------|-------------|-----------|
| syn-trans | CASSCF(2,2) | 4th MO occupation = 0.0007; (4,4) non-convergent |
| syn-cis   | CASSCF(2,2) | same |
| anti-trans | CASSCF(4,4) | NO occ [1.952, 1.717, 0.282, 0.049] — genuine 4-orbital space |
| anti-cis   | CASSCF(4,4) | NO occ [1.948, 1.724, 0.275, 0.052] |

**MO seeding NOT safe across conformers**: passing 96×96 MO coefficient matrix from anti-cis to syn-trans gave E_S0 = −3258 Ha (catastrophic). Always let PSI4 build fresh MOs.

### Grid Results (May 2026) — All Complete

| Conformer | Output Directory | LOO-CV δ_S0 | Status |
|-----------|-----------------|-------------|--------|
| syn-trans | `outputs/casscf_wB97X_nm_grid_20260514_224806/` | 0.064 kcal/mol | ✓ |
| syn-cis   | `outputs/casscf_wB97X_nm_grid_20260519_132602/` | 0.170 kcal/mol | ✓ |
| anti-trans | `outputs/casscf_wB97X_nm_grid_20260519_132839/` | 2.23 kcal/mol | ✗ (outliers) |
| anti-cis   | `outputs/casscf_wB97X_nm_grid_20260407_184904/` | 0.654 kcal/mol | ~ |

**Anti-trans outlier fix**: Initial LOO-CV 2.23 kcal/mol due to 7 outlier frames (mode-0 torsion at f=1.5/2.0 giving duplicate geometries with δ_S0 up to 16 kcal/mol; mode-7 orbital root-flip; mode-13 duplicate geometry bug). Retrained with `--outlier-delta-s0-max 5.0 --outlier-gap-t1-min 15.0` → δ_S0 = 0.101 kcal/mol (22× improvement).

**wB97X base ML-PES**: `outputs/wB97X_surface_20260406_223155/mlpes_wB97X_aug.pkl`  
- 1126 frames, γ = 0.001, α = 1e-5, RMSE = 0.31 kcal/mol

### S1/T1 Electronic Gaps (at conformer equilibria)

| Conformer | S1 gap (kcal/mol) | T1 gap (kcal/mol) |
|-----------|------------------|------------------|
| syn-trans  | 23.4 | 20.7 |
| syn-cis    | 21.6 | 19.0 |
| anti-trans | 27.3 | 24.8 |
| anti-cis   | ~29.0 | ~26.5 |

---

## Phase 12 — ACF IR Spectrum Failure: Root Cause Identified (June 2026)

**Commits:** `ab58255`, `921bf44`

**Problem**: All ACF-based IR runs for MVKO produce spectra dominated by ~150 cm⁻¹ torsional features with zero fingerprint-region signal. This persisted across:
- Coulomb+KRR, CH-retrained model, MACE PES
- B3LYP and wB97X dipole surfaces
- NMDipoleSurface (R² = 0.9997), full dipole surface (R² = 0.999)
- ZPE leakage time constants τ = 200 vs 50,000
- 5 × 30,000-step trajectories

**Root cause**: MVKO has a ~4.5 D permanent dipole from the COO zwitterion. Torsional motion **reorients** the permanent dipole → dominates the ACF signal.

The permanent dipole rotation contributes I_ACF(ω_tors) ∝ |μ₀|² × θ_max². The stretching contribution I(ω_stretch) ∝ (∂|μ|/∂q_stretch)² is ~70,000× smaller. This is a fundamental physical property of the molecule, **not a bug** — not fixable by thermostat tuning, ZPE settings, or better dipole surface.

---

## Phase 13 — Harmonic Dipole Derivatives: Correct IR Approach (June 2026)

**Commits:** `921bf44`, `ab58255`

**Solution**: Compute IR intensities from harmonic dipole derivatives — `NMDipoleSurface.ir_intensities(eq_coords)` = ||∂μ/∂q_k||² (Debye²/amu·Bohr²). No ACF, no MD required for the spectrum.

**NMDipoleSurface** trained on normal-mode displacement geometries with wB97X-D/6-31G* dipoles, producing a KRR dipole surface in NM-coordinate space. The analytic gradient ∂μ/∂q_k is computed from the KRR kernel gradient.

### Anti-cis MVKO Harmonic IR Spectrum (June 2026)

NMDipoleSurface (R² = 0.9997), MACE April 17 PES + CASSCF δ_S0 anti-cis grid. Saved: `outputs/ir_nm_dipole_harmonic_anti_cis/ir_harmonic_spectrum.csv` (Lorentzian FWHM = 10 cm⁻¹)

| Mode | Freq (cm⁻¹) | I_rel | Assignment |
|------|------------|-------|------------|
| 15 | **1088.8** | 1.000 | C-O stretch (strongest) |
| 20 | **1480.4** | 0.985 | CH₂ wag / C=C |
| 11 | **1011.2** | 0.827 | C-O stretch |
| 24 | **1692.6** | 0.550 | C=C stretch |
| 16 | **1130.8** | 0.513 | C-O stretch |
| 17 | **1291.7** | 0.399 | C=C stretch |
| 29 | **3201.3** | 0.283 | C-H stretch |
| 10 | 835.2 | 0.041 | O-O stretch |

Torsional region (100–500 cm⁻¹): I_rel = 0.14 (physically correct, vs 1.0 in ACF).

**NMDipoleSurface files (anti-cis)**:
- Model pkl: `outputs/ir_nm_pes_anti_cis_300K/dipole_surface.pkl` (γ = 0.01186, α = 1e-5, 300 frames)
- NM-PES model: `outputs/anti_cis_nm_pes_20260513/mlpes_nm.pkl`
- Equilibrium coords: `outputs/anti_cis_nm_pes_20260513/eq_coords.npy`

---

## Phase 14 — Hot-Molecule IR Emission & Experiment Comparison (July 2026)

**Commits:** `e48a08b`, `34a69bb`, `2454d9c`, `cbc0351`, `c3719c9`, `1a2e6be`

Extended from equilibrium IR to **hot-molecule emission** relevant to ozonolysis-energized MVKO (~50 kcal/mol internal energy above zero-point, T_eff ≈ 840 K).

**Workflow**: Multi-conformer population weighting (Boltzmann over conformer energies) + CASSCF-corrected MACE PES + harmonic dipole derivatives at each conformer geometry. Compared to digitized Chung & Lee (2021) experimental MVKO spectrum.

**Key results**:
- Hot-emission spectrum for all four conformers computed and combined
- Conformer figure (`outputs/mvko_conformers_figure.png`) shows wB97X-D/6-31G* and Barber CCSD(T) conformer energies + CASSCF S1/T1 gaps
- Anti-cis IR spectrum (300 K and hot) compared against experiment

**Local harmonic approximation (LHA) conceptual section added** (commit `2454d9c`, `cbc0351`): 4-panel figure explaining why LHA (harmonic dipole derivatives at ML-PES-minimized geometry) is the correct approach for MVKO, vs why ACF IR fails.

**Syn-trans hot emission pipeline**: `outputs/syn_trans_...` — full CASSCF δ_S0 + S1/T1 gap + MACE PES pipeline, compared against anti-cis result.

---

## Descriptor Benchmark Summary (April 2026)

| Descriptor | Dimension | RMSE (kcal/mol) | Notes |
|------------|-----------|-----------------|-------|
| Coulomb matrix | 78 | **0.152** | Best; global KRR compatible |
| Pairwise distances | 66 | 0.235 | |
| ACSF | 1188 | 2.49 | Incompatible with global KRR |

Stiffness fix requires sGDML or per-atom neural networks (superseded by MACE).

---

## Concluded / Superseded Approaches

| Approach | Status | Reason |
|----------|--------|--------|
| ACF-based IR for MVKO | **Dead end** | 4.5 D permanent dipole causes 70,000× torsional dominance |
| sGDML | **Dead end** | Same 1/r stiffness as Coulomb+KRR; memory 23.5 GB; dissociates |
| Adaptive KRR (high-energy frames) | **Do not use for IR** | Shifts kernel from equilibrium → imaginary modes |
| CASSCF full-surface delta-ML (Coulomb space) | **Dead end** | K≈0.999 clustering; replaced by NM-coordinate KRR |
| NEVPT2 correction | Marginal | LOO-CV 1.60 kcal/mol; useful for 840 K prediction only |
| 1D anharmonic DVR (C-H overtones) | Reference only | 2νCH at 5500–6500 cm⁻¹; classical MD cannot access |
| MACE fw10 retrain | Not validated | Loss plateaued; ACF approach moot for MVKO |

---

## Key File Inventory

### CH₂OO

| File | Description |
|------|-------------|
| `outputs/clean_psi410_20260308_203552/training_data.npz` | Clean PSI4 1.10 training data (344 frames) |
| `outputs/nm_training_20260308_203606/mlpes_model_nm.pkl` | Best CH₂OO model (γ=0.01, α=0.001) |
| `outputs/ir_spectrum_20260318_232404/` | Best CH₂OO IR spectrum |

### MVKO

| File | Description |
|------|-------------|
| `outputs/mvko_20260319_081314/mlpes_initial.pkl` | Best MVKO energy model (904 frames, γ=0.001) |
| `outputs/mvko_20260319_081314/psi4_eq_coords.npy` | PSI4 equilibrium coordinates |
| `outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz` | Dipole training data (150 frames) |
| `outputs/wB97X_surface_20260406_223155/mlpes_wB97X_aug.pkl` | wB97X base ML-PES (1126 frames) |
| `outputs/anti_cis_nm_pes_20260513/mlpes_nm.pkl` | Anti-cis NM-PES model |
| `outputs/anti_cis_nm_pes_20260513/eq_coords.npy` | Anti-cis equilibrium coords |
| `outputs/ir_nm_pes_anti_cis_300K/dipole_surface.pkl` | NMDipoleSurface (anti-cis) |
| `outputs/ir_nm_dipole_harmonic_anti_cis/ir_harmonic_spectrum.csv` | Harmonic IR spectrum |
| `outputs/casscf_wB97X_nm_grid_20260407_184904/` | Anti-cis CASSCF grid (complete) |
| `outputs/casscf_wB97X_nm_grid_20260514_224806/` | Syn-trans CASSCF grid (complete) |
| `outputs/casscf_wB97X_nm_grid_20260519_132602/` | Syn-cis CASSCF grid (complete) |
| `outputs/casscf_wB97X_nm_grid_20260519_132839/` | Anti-trans CASSCF grid (complete, outlier-corrected) |
| `outputs/mvko_conformers_figure.png` | Conformer energies + CASSCF gaps figure |

---

## PSI4 Settings (Standard — Use Everywhere)

```python
psi4.set_options({
    'basis': '6-31G*',
    'scf_type': 'df',
    'reference': 'rhf',
    'maxiter': 200,
    'e_convergence': 1e-7,
    'd_convergence': 1e-7
})
```

---

## Unit Conventions

| Quantity | Unit |
|----------|------|
| Coordinates | Angstrom |
| Energies | Hartree (display: kcal/mol) |
| Forces | Hartree/Angstrom |
| Dipoles | Debye |
| Velocities | Angstrom/fs |
| Time | femtoseconds |
| IR frequency | cm⁻¹ |

Constants defined in `modules/direct_md.py:40-60` — do not redefine elsewhere.
