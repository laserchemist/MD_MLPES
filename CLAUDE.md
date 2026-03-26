# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Verify installation and imports:**
```bash
python3 test_install.py
```

**Run module unit tests:**
```bash
python3 -m pytest modules/data_formats_test.py modules/ml_pes_test.py -v
# Or run individually:
python3 modules/data_formats_test.py
python3 modules/ml_pes_test.py
```

**Quick PSI4 sanity check (requires PSI4):**
```bash
python3 quick_test_psi4.py
```

**Run a workflow end-to-end:**
```bash
python3 complete_workflow_v2.2.py     # Full pipeline (PSI4 required)
python3 two_phase_workflow.py         # ML-PES phase only (no PSI4 needed)
python3 simple_production_workflow.py # Streamlined pipeline
```

**Generate training data via normal mode distortions + high-T PSI4 MD:**
```bash
python3 generate_nm_training.py \
    --training-data outputs/.../augmented_training_data.npz \
    --T-nm 1000 --n-amplitudes 4 --max-factor 3 \
    --md-temps 300,600,1000 --md-steps 50
# Skip MD, NM only:
python3 generate_nm_training.py --training-data <data.npz> --no-md
# Skip NM, MD only:
python3 generate_nm_training.py --training-data <data.npz> --no-nm
```

**Compute IR spectrum from existing trajectory:**
```bash
python3 compute_ir_workflow.py
```

**Run ML-MD IR spectrum (production command):**
```bash
python3 ir_md_spectrum.py \
    --model outputs/nm_training_20260308_203606/mlpes_model_nm.pkl \
    --training-data outputs/clean_psi410_20260308_203552/training_data.npz \
    --steps 20000 --temp 300 --timestep 0.5 --save-every 1 \
    --preminimize --zpe-min-freq 50 --zpe-max-freq 4000
```

**Run multi-trajectory averaged IR spectrum (conformational broadening):**
```bash
python3 ir_md_spectrum.py \
    --model outputs/mvko_20260319_081314/mlpes_initial.pkl \
    --training-data outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \
    --steps 15000 --temp 300 --timestep 0.5 --save-every 1 \
    --preminimize --zpe-min-freq 50 --zpe-max-freq 4000 \
    --n-trajectories 5 \
    --max-bond-extension 2.5
# Runs 5 independent MD trajectories from the 5 lowest-energy training frames.
# Dipoles are per-trajectory centred and concatenated for ACF averaging.
# Dissociation guard: stops trajectory when any heavy-atom bond extends >2.5× eq.
# Total effective sampling: 5 × 15000 = 75000 frames.
```

**Retrain with softer gamma (to improve Hessian frequencies):**
```bash
python3 retrain_softer_gamma.py \
    --training-data outputs/clean_psi410_20260308_203552/training_data.npz \
    --gamma-values 0.0001,0.0003,0.001,0.003,0.01
```

**Adaptive high-energy training (extend coverage to anharmonic regions):**
```bash
python3 adaptive_high_energy.py \
    --model    outputs/mvko_20260319_081314/mlpes_initial.pkl \
    --training-data outputs/mvko_20260319_081314/combined_training_data.npz \
    --hessian-data outputs/mvko_<ts>/nm_displacements.npz \
    --cycles 3 --T-nm 3000 --n-amplitudes 6 \
    --md-steps 50 --md-temps 1000,2000 --top-n 30
# Outputs: outputs/adaptive_high_energy_<ts>/mlpes_adaptive_final.pkl + training_data_final.npz
```
# WARNING: adaptive model degrades near-equilibrium PES (3 imaginary modes at PSI4 eq).
# The adaptive model is UNSUITABLE for ZPE-floor IR spectra.
# Use mlpes_initial.pkl (904 frames) for IR spectra; adaptive model for high-energy analysis only.

**Train multi-conformer PES family:**
```bash
python3 train_conformer_family.py \
    --conformers "s-cis:outputs/mvko_scis/combined_training_data.npz" \
                 "s-trans:outputs/mvko_strans/combined_training_data.npz" \
    --gamma 0.001 --alpha 1e-5 --align-energies --blend-width 3.0
# Outputs: outputs/conformer_family_<ts>/family.pkl + conformer_manifest.json
```

**Run IR spectrum with multi-surface PES family:**
```bash
python3 ir_md_spectrum.py \
    --model outputs/conformer_family_<ts>/s-cis_model.pkl \
    --training-data outputs/mvko_dipoles_<ts>/training_with_dipoles.npz \
    --multi-surface \
    --conformer-manifest outputs/conformer_family_<ts>/conformer_manifest.json \
    --blend-width 3.0 \
    --steps 30000 --temp 300 --preminimize \
    --zpe-min-freq 50 --zpe-max-freq 4000
```

**MVKO full workflow (optimize → train → IR):**
```bash
python3 mvko_workflow.py --steps 1,2       # PSI4 optimize + Hessian
python3 mvko_workflow.py --restart outputs/mvko_<ts>/state.json --steps 3,4,5 \
    --T-nm 1500 --n-amplitudes 5 --md-temps 300,600,1000 --md-steps 200
# SKIP Step 6 (adaptive loop causes runaway ML-MD → corrupts model)
# Collect dipoles separately, then run IR:
python3 collect_mvko_dipoles.py \
    --training-data outputs/mvko_<ts>/combined_training_data.npz --n-frames 150
python3 ir_md_spectrum.py \
    --model outputs/mvko_<ts>/mlpes_initial.pkl \
    --training-data outputs/mvko_dipoles_<ts>/training_with_dipoles.npz \
    --steps 30000 --temp 300 --timestep 0.5 --save-every 1 \
    --preminimize --zpe-min-freq 50 --zpe-max-freq 4000
```

## Architecture

### Two-Layer Structure

**`modules/`** — Core library (stable, reusable):
- `direct_md.py` — PSI4 interface + Velocity-Verlet MD engine
- `ml_pes.py` — Kernel Ridge Regression ML-PES trainer (use `ml_pes_fixed.py` as the authoritative version)
- `data_formats.py` — `TrajectoryData` dataclass + multi-format I/O (xyz, extxyz, npz, hdf5)
- `test_molecules.py` — Pre-optimized test molecule library (B3LYP/6-31G*)
- `visualization.py` — Matplotlib-based trajectory and training plots
- `ir_spectroscopy.py` — `DipoleSurface` and `IRSpectrumCalculator` (dipole ACF → IR spectrum)
- `normal_modes.py` — Hessian via PSI4, normal mode diagonalization, NM-displaced geometry generation
- `bakken.py` — **ML-MD engine** (Norwegian: "the hill"). `MLPESDriver` (energy + FD forces), `minimize_geometry` (adaptive steepest descent), `maxwell_boltzmann_velocities`, `zpe_initialized_velocities` (with frequency filter), `kinetic_temperature`, `run_md` (Velocity-Verlet + Berendsen). The canonical ML-MD engine for all IR spectrum workflows.
- `uncertainty.py` — `CommitteeModel`: K=5 bootstrap KRR ensemble for epistemic uncertainty. `.train()`, `.batch_uncertainty()`, `.calibrate()`. Used by `adaptive_high_energy.py` to score candidate geometries.
- `pes_family.py` — `PESFamily` + `ConformerPES`: multi-surface softmin blending. `.blend_energy()`, `.assign_conformer()`, `.from_model_paths()`. Used by `ir_md_spectrum.py --multi-surface`.

**Root directory** — Workflow orchestration scripts (~67 scripts). Key ones:
- `master_workflow.py` — Menu-driven interface with JSON state tracking
- `complete_workflow_v2.2.py` — Full end-to-end pipeline (latest complete version)
- `generate_nm_training.py` — **Primary training data generator**: NM distortions + multi-T PSI4 MD
- `adaptive_sampling_workflow.py` — Adaptive data collection loop
- `on_the_fly_validation.py` — Validation during training

### Data Flow

```
PSI4 (B3LYP/6-31G*)
  → DirectMDRunner._calculate_energy_gradient()
  → TrajectoryData (coords/energies/forces/dipoles)
  → CoulombMatrixDescriptor (upper-triangle Coulomb matrix)
  → MLPESTrainer (KRR, StandardScaler critical)
  → Trained model .pkl
  → Fast ML-MD
  → DipoleSurface predictions or stored dipoles
  → IRSpectrumCalculator (dipole ACF via FFT)
  → IR spectrum (cm⁻¹)
```

### PSI4 Interface
`DirectMDRunner._calculate_energy_gradient()` in `modules/direct_md.py:286-418` captures:
- Energy: `psi4.energy()` — Hartree
- Forces: `-gradient / ANGSTROM_TO_BOHR` — Hartree/Angstrom
- Dipole: `psi4.oeprop(wfn, 'DIPOLE')` with 5 fallback methods — Debye

All code has a mock-calculation fallback for testing without PSI4.

## Unit Conventions (Enforce Strictly)

| Quantity | Unit | Notes |
|----------|------|-------|
| Coordinates | Angstrom | PSI4 accepts Angstrom geometry input |
| Energies | Hartree | Display/reporting converts to kcal/mol |
| Forces | Hartree/Angstrom | PSI4 gradient in Hartree/Bohr, divided by 1.88973 |
| Dipoles | Debye | PSI4 oeprop in a.u. (e·Bohr), converted via AU_TO_DEBYE=2.541746 |
| Velocities | Angstrom/fs | |
| Time | femtoseconds | FS_TO_AU = 41.341374575751 for internal dynamics |
| Temperature | Kelvin | KB_HARTREE_PER_K = 3.1668114e-6 |
| Masses | amu | AMU_TO_AU = 1822.888486 |
| IR frequency | cm⁻¹ | AU → cm⁻¹ conversion factor: 219474.63 |

**Conversion constants are defined in `modules/direct_md.py:40-60` — do not redefine elsewhere.**

## Key Design Decisions

- **Force training disabled**: `MLPESConfig.train_forces = False` by default. Enabling it breaks the KRR model. Forces should be predicted via finite differences on the ML-PES energy surface or via a separate force model.
- **StandardScaler is critical**: The ML-PES uses `StandardScaler` on both features and targets. Without it, RMSE degrades from ~0.04 kcal/mol to ~170 kcal/mol. See `modules/ml_pes_fixed.py` for the authoritative implementation.
- **Coulomb matrix descriptor**: Simple upper-triangle of `Z_i*Z_j/r_ij` (off-diagonal) and `0.5*Z_i^2.4` (diagonal). Robust but not permutation-invariant — atom ordering must be consistent.
- **Dipole surface separate from energy surface**: `DipoleSurface` in `ir_spectroscopy.py` is a separate KRR model predicting 3-component dipole vectors. The energy `MLPESTrainer` only predicts scalar energy.
- **Mock PSI4 fallback**: All direct MD functions detect PSI4 absence and return physically plausible mock values for testing.
- **CRITICAL — PSI4 version consistency**: ALL training data, NM displacement energies, and validation energies must be computed with the SAME PSI4 installation. The Dec 2025 training data gave energies ~8.5 kcal/mol lower than PSI4 1.10 (current). Mixing data from different PSI4 versions causes systematic offsets that make ML-PES validation fail catastrophically (15–40 kcal/mol errors). When starting fresh or after PSI4 upgrades, regenerate ALL training data with the current PSI4. The safe pipeline is: `generate_nm_training.py` (NM + PSI4 MD, all fresh) → `two_phase_workflow.py` (validation).

## Planned Features (In Progress / Next Steps)

The following extensions were scoped by the user and should be built consistently with existing conventions:

1. **Normal mode distortions for adaptive sampling**: ✅ **Implemented** in `modules/normal_modes.py` + `generate_nm_training.py`. Computes PSI4 Hessian, diagonalises mass-weighted Hessian, displaces ±n×a_thermal(T) along each mode (FREQ_CONV = 5140.48 cm⁻¹/√(Hartree/(Bohr²·amu))). Run: `python3 generate_nm_training.py --training-data <data.npz> --T-nm 1000 --n-amplitudes 4 --max-factor 3`.

2. **ML-PES quality testing via normal modes and MD**: After training, compute normal mode frequencies on the ML-PES and compare to PSI4 frequencies. Run short ML-MD and flag frames where ML energy/force error exceeds threshold; add those frames to training set.

3. **Analytic KRR forces and Hessian** (next priority): The numerical FD Hessian gives unphysical frequencies (lowest mode ~1800 cm⁻¹ vs expected ~500 cm⁻¹) because the Coulomb matrix descriptor has stiff numerical second derivatives under the RBF kernel. Gamma-sweeping from 0.01 → 0.0001 reduces unphysical modes (5→3/9) but makes dynamics worse. The correct fix is analytic forces via the KRR kernel gradient + Coulomb matrix Jacobian. Design in `modules/bakken.py`:
   - `_coulomb_jacobian(symbols, coords, charges)` → `(n_desc, 3N)` Jacobian `∂C/∂R`
   - `_coulomb_hessian_2nd(symbols, coords, charges)` → `(n_desc, 3N, 3N)` second derivative `∂²C/∂R²`
   - `MLPESDriver.analytic_forces(coords)` = `-(∂E_scaled/∂x_scaled · 1/σ_X) · J · σ_y`
   - `MLPESDriver.analytic_hessian(coords)` = `J^T · H_KRR · J + Σ_k G_k · H2_k`
   - where `H_KRR[k,l] = Σ_i α_i K_i [4γ²Δx_k Δx_l − 2γδ_{kl}]` and `G_k = Σ_i α_i K_i (−2γ)Δx_k`
   - **Key benefit**: replaces 30 KRR evaluations per FD step with 1 evaluation + Jacobian; gives exact frequencies.

4. **IR spectra via dipole ACF**: ✅ **Implemented** in `ir_md_spectrum.py`. Best result to date: γ=0.01 model with bakken pre-min + ZPE filter (50–4000 cm⁻¹) → 7 peaks at 297, 307, 458, 461, 803, 1110, 2085 cm⁻¹. The 803 and 1110 cm⁻¹ peaks correspond to O-O and C-O stretch region of CH₂OO. ACF-based peaks are more physically meaningful than Hessian frequencies.

5. **Multi-state PES family**: Future design — maintain a family of `MLPESTrainer` models (one per electronic state or molecular species). Mix energies/forces with scalar coefficients or transition between states with a hopping probability. Design: `PESFamily` container holding `{label: MLPESTrainer}` with a `mix(coefficients)` or `hop(probability_matrix)` interface. Units and descriptor conventions must be identical across all family members.

## Output Conventions

- Training data: `outputs/nm_training_YYYYMMDD_HHMMSS/combined_training_data.npz`
- NM displacements: `outputs/nm_training_.../nm_displacements.npz`
- PSI4 MD data: `outputs/nm_training_.../multi_T_md.npz`
- Clean PSI4 1.10 training data: `outputs/clean_psi410_*/training_data.npz`
- Trained models: `outputs/nm_training_.../mlpes_model_nm.pkl`
- Phase 1 diagnostics: `outputs/diagnostic_phase1_*/phase1_snapshots.pkl`
- Phase 2 validation: `outputs/diagnostic_phase2_*/validation_results.pkl`
- MD trajectories: `md_output/`
- IR spectra: `ir_spectrum_output/`
- Workflow state: JSON files tracking completed steps and file paths

## Current Best Model (March 2026)

- **Model**: `outputs/nm_training_20260308_203606/mlpes_model_nm.pkl`
- **Training data**: `outputs/clean_psi410_20260308_203552/training_data.npz`
- **344 frames**: 72 NM displacements + 202 PSI4 MD + 70 Phase 2 validation frames
- **Validation RMSE**: 0.64 kcal/mol mean, 1.23 kcal/mol max (all 20 frames < 2 kcal/mol)
- **Hyperparameters**: γ=0.01, α=0.001 (RBF kernel)
- **Note**: All data computed with PSI4 1.10; older training data (pre-2026-03-08) is incompatible

## IR Spectrum Results (March 2026)

**Best IR run**: `outputs/ir_spectrum_20260318_232404/` — γ=0.01 model, bakken pre-min + ZPE filter
- Command: `python3 ir_md_spectrum.py --model ... --preminimize --zpe-min-freq 50 --zpe-max-freq 4000`
- **7 IR peaks**: 297, 307, 458, 461, 803, 1110, 2085 cm⁻¹
- 803 cm⁻¹ ≈ O-O stretch, 1110 cm⁻¹ ≈ C-O stretch (physically meaningful for CH₂OO)
- Dipole surface R²=0.9999999, train RMSE=3e-5 D

**Gamma sweep findings** (`retrain_softer_gamma.py`, `outputs/softer_gamma_20260318_234300/`):
| γ | α | RMSE | Unphysical NM modes | Lowest NM freq |
|---|---|------|---------------------|---------------|
| 0.0001 | 1e-5 | 2.30 kcal/mol | 3/9 | 1254 cm⁻¹ |
| 0.0003 | 1e-5 | 1.07 kcal/mol | 4/9 | 1393 cm⁻¹ |
| 0.001  | 1e-5 | 0.50 kcal/mol | 5/9 | 1844 cm⁻¹ |
| 0.003  | 1e-5 | 0.45 kcal/mol | 5/9 | 1883 cm⁻¹ |
| 0.01   | 1e-5 | 0.61 kcal/mol | 5/9 | 1773 cm⁻¹ |

Key finding: softer gamma reduces unphysical Hessian modes but causes worse IR spectra (forces less accurate, molecule wanders). The Hessian stiffness is intrinsic to Coulomb matrix second derivatives — not curable by kernel width tuning alone. **Analytic KRR forces+Hessian is the correct fix.**

**Kernel panic fix** (March 2026): `ir_spectroscopy.py` `DipoleSurface.train()` previously used `n_jobs=-1`, which saturated all CPU cores on Apple Silicon (watchdog timeout panic). Fixed with `safe_jobs = min(n_jobs, 2)` hard cap.

## MVKO IR Spectrum Results (March 2026)

**Training**: `outputs/mvko_20260319_081314/` — Steps 1-5 completed
- 904 frames: 300 NM displacements (T=1500K, ±5 amplitudes) + 603 PSI4 MD (300/600/1000K, 201 each)
- Best model: γ=0.001, α=1e-5, RMSE=0.2734 kcal/mol
- NOTE: Step 6 (adaptive refinement) was SKIPPED — 500K ML-MD causes runaway to 5000-10000K, corrupting training

**Dipole collection**: `outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz`
- 150 representative frames with PSI4 B3LYP/6-31G* dipoles
- Collected via `collect_mvko_dipoles.py` using `properties=['dipole']` API
- Dipole range: 2.97–5.61 D, mean 4.63 D (physically reasonable for Criegee intermediate)

**Dipole surface**: γ=0.001, α=1e-4, R²=0.999 train / R²=0.981 test, RMSE=0.024 D

**IR spectrum**: `outputs/ir_spectrum_20260319_174321/` — 30,000 steps at 300 K
| Peak (cm⁻¹) | Rel. Intensity | Assignment |
|---|---|---|
| 322 | 1.000 | Torsion/bending (strongest) |
| 247, 239 | 0.87 | Torsion/bending |
| 893 | 0.125 | **O-O stretch** (physically meaningful) |
| 513 | 0.124 | C-O-O bending |
| 658 | 0.059 | C-O stretch |

Key: 893 cm⁻¹ O-O stretch for MVKO vs 803 cm⁻¹ for CH₂OO — consistent with substitution effect. C-H stretches (~3000 cm⁻¹) absent due to Coulomb+RBF Hessian stiffness artifact (same as CH₂OO).

## MVKO 2νCH Overtone Region — 1D Anharmonic Analysis (March 2026)

**Script**: `mvko_anharmonic.py` — 1D DVR on PSI4 B3LYP/6-31G* quartic potential curves
**Usage**:
```bash
python3 mvko_anharmonic.py \
    --state outputs/mvko_20260319_081314/state.json \
    --psi4out outputs/mvko_vpt2_20260319_183728/psi4_vpt2_output.dat
# To reload saved energy curves (skip PSI4):
python3 mvko_anharmonic.py ... --load-curves outputs/mvko_anharm_<ts>/curves.npz
```

**PSI4 1.10 has NO VPT2**: `psi4.frequency(..., anharmonic=True)` silently ignores the keyword and computes only harmonic frequencies. Use `mvko_anharmonic.py` for anharmonic corrections.

**B3LYP/6-31G* harmonic C-H modes** (computed via FINDIF from gradient):
3049, 3100, 3163, 3165, 3180, 3324 cm⁻¹ (modes 31–36, IRR = App/App/Ap...)

**1D anharmonic results** (`outputs/mvko_anharm_20260319_202143/`):
| Mode | ω(harm) | ν(fund,1D) | 2ν(DVR) | Δanharmon |
|------|---------|-----------|---------|-----------|
| 31 | 3049 | 3002 | 5981 | −23 cm⁻¹ |
| 32 | 3100 | 3167 | 6404 | +71 cm⁻¹ |
| 33 | 3163 | 3150 | 6297 | −3 cm⁻¹ |
| 34 | 3165 | 3209 | 6462 | +44 cm⁻¹ |
| 35 | 3180 | 3131 | 6232 | −30 cm⁻¹ |
| 36 | 3324 | 3300 | 6603 | +3 cm⁻¹ |

**2νCH region (5500–6500 cm⁻¹)**: 5/6 overtones + 14/15 combinations = **19 transitions** (paper: 21/conformer). Missing 2 just above 6500 cm⁻¹ cutoff (2ν36=6603, ν34+ν36=6509).

**Can classical MD access 2νCH?** **NO** — two reasons:
1. Overtones are quantum-mechanical (need v=2 population, absent from classical DACF)
2. ML-PES C-H modes at 10,000–15,000 cm⁻¹ (Coulomb+RBF stiffness artifact)
