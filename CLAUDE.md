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

**Train MACE model (equivariant MPNN — preferred backend, physical frequencies):**
```bash
# Standard training (MPS/CUDA auto-selected):
python3 train_mace_model.py \
    --training-data outputs/wB97X_surface_20260406_223155/training_data_wB97X_aug.npz \
    --output-dir    outputs/mace_wB97X_20260417 \
    --n-train 900 --n-valid 80 --epochs 500

# Quick test (small architecture, few epochs):
python3 train_mace_model.py \
    --training-data <data.npz> --output-dir outputs/mace_test \
    --n-train 200 --n-valid 30 --epochs 100 \
    --hidden-irreps '32x0e + 32x1o'
```
Outputs: `mace_model.pt` + `mace_model.symbols.pkl` (companion atom-order file required by MACEDriver).

**Validate any ML-PES frequencies and MD stability (works with all driver types):**
```bash
# MACE (preferred):
python3 validate_pes_frequencies.py \
    --mace-model    outputs/mace_wB97X_20260417/mace_model.pt \
    --training-data outputs/wB97X_surface_20260406_223155/training_data_wB97X_aug.npz \
    --psi4-hessian  outputs/mvko_20260319_081314/nm_displacements.npz

# sGDML:
python3 validate_pes_frequencies.py \
    --sgdml-model   outputs/sgdml_wB97X_20260417/sgdml_model.pkl \
    --training-data outputs/.../training_data.npz

# Coulomb+KRR (legacy):
python3 validate_pes_frequencies.py \
    --model         outputs/mvko_20260319_081314/mlpes_initial.pkl \
    --training-data outputs/mvko_20260319_081314/combined_training_data.npz
# Outputs: validation_report.txt, frequencies.npy; prints PASS/FAIL/WARN assessment
# MACE expected: 0 imaginary, 0 unphysical (>5000 cm⁻¹), max ~3200 cm⁻¹
# Coulomb+KRR: typically 5/30 unphysical at 10000-38000 cm⁻¹ (intrinsic artifact)
```

**Run IR spectrum with MACE model:**
```bash
python3 ir_md_spectrum.py \
    --mace-model    outputs/mace_wB97X_20260417/mace_model.pt \
    --training-data outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \
    --steps 30000 --temp 300 --preminimize \
    --zpe-min-freq 50 --zpe-max-freq 4000 \
    --n-trajectories 5 --max-bond-extension 2.0
```

**Train sGDML model (DEPRECATED — same 1/r descriptor stiffness as Coulomb+KRR, use MACE):**
```bash
# Sweep sig hyperparameter:
python3 train_sgdml_model.py \
    --training-data outputs/wB97X_surface_20260406_223155/training_data_wB97X_aug.npz \
    --output-dir    outputs/sgdml_wB97X_20260417 \
    --sig-values    0.05,0.1,0.2,0.5,1,2,5 \
    --n-train 300 --n-valid 80   # max n_train=300 to avoid OOM (kernel = (n×3N)²)
```

**Run IR spectrum with sGDML model (legacy):**
```bash
python3 ir_md_spectrum.py \
    --sgdml-model   outputs/sgdml_wB97X_20260417/sgdml_model.pkl \
    --training-data outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \
    --steps 30000 --temp 300 --preminimize \
    --zpe-min-freq 50 --zpe-max-freq 4000 \
    --n-trajectories 5 --max-bond-extension 2.0
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
- `sgdml_pes.py` — sGDML backend (DEPRECATED — same unphysical frequency artifact as Coulomb+KRR; use MACE instead). `SGDMLModel`, `SGDMLDriver` (bakken-compatible), `train_sgdml()`, `train_sgdml_sweep()`. Use `--sgdml-model` in `ir_md_spectrum.py`.
- `mace_pes.py` — **MACE backend (preferred)**. `MACEDriver` (bakken-compatible: `.energy()`, `.forces()`, `.analytic_forces()`, `.analytic_hessian()` via FD on MACE analytic forces; auto-selects MPS/CUDA/CPU). `npz_to_extxyz()` converts training npz to MACE extxyz (eV/eV/Å, REF_energy/REF_forces keys). Loads companion `.symbols.pkl` for atom ordering. Unit conventions: MACE uses eV internally; all public methods use Ha/Å (pipeline units). Why MACE: local atomic energy decomposition + SO(3) equivariance + force training eliminates the 1/r descriptor stiffness that causes unphysical C-H frequencies in KRR-based models.

**Root directory** — Workflow orchestration scripts (~67 scripts). Key ones:
- `master_workflow.py` — Menu-driven interface with JSON state tracking
- `complete_workflow_v2.2.py` — Full end-to-end pipeline (latest complete version)
- `generate_nm_training.py` — **Primary training data generator**: NM distortions + multi-T PSI4 MD
- `adaptive_sampling_workflow.py` — Adaptive data collection loop
- `on_the_fly_validation.py` — Validation during training

### Data Flow

**Coulomb+KRR path (legacy):**
```
PSI4 (wB97X-D/6-31G*)
  → TrajectoryData (coords/energies/forces/dipoles)
  → CoulombMatrixDescriptor (upper-triangle Coulomb matrix)
  → MLPESTrainer (KRR, StandardScaler critical)
  → Trained model .pkl  [forces via FD only]
  → Fast ML-MD (bakken MLPESDriver)
  → DipoleSurface predictions or stored dipoles
  → IRSpectrumCalculator (dipole ACF via FFT)
  → IR spectrum (cm⁻¹)
```

**MACE path (PREFERRED — physical frequencies, equivariant, no descriptor stiffness):**
```
PSI4 (wB97X-D/6-31G*)
  → TrajectoryData (coords/energies/forces/dipoles) [npz format]
  → train_mace_model.py  [npz→extxyz, mace_run_train subprocess, saves .pt + .symbols.pkl]
  → MACEDriver .pt  [SO(3)-equivariant MPNN; analytic forces; FD Hessian on analytic forces]
  → validate_pes_frequencies.py  [verify C-H modes at ~3000 cm⁻¹, 0 imaginary, 0 unphysical]
  → Fast ML-MD (bakken via MACEDriver)
  → DipoleSurface predictions or stored dipoles
  → IRSpectrumCalculator (dipole ACF via FFT)
  → IR spectrum (cm⁻¹) with physical C-H stretch region
```

**sGDML path (DEPRECATED — same unphysical frequency artifact as Coulomb+KRR):**
```
PSI4 (wB97X-D/6-31G*)
  → TrajectoryData (coords/energies/forces/dipoles)
  → train_sgdml_model.py  [trains on forces; 1/r descriptor stiffness remains]
  → SGDMLModel .pkl  [38000 cm⁻¹ C-H modes — not physical]
  → ir_md_spectrum.py --sgdml-model (missing C-H stretch region in IR)
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

- **MACE is the preferred backend**: Both Coulomb+KRR and sGDML share the same root pathology — 1/r-based descriptors have intrinsically stiff second derivatives under RBF/Matérn kernels, producing C-H modes at 10,000–38,000 cm⁻¹ instead of ~3,000 cm⁻¹. This cannot be fixed by hyperparameter tuning (confirmed: analytic and FD Hessians give identical wrong answers; sig sweeps in sGDML make no difference). MACE (`modules/mace_pes.py`) uses local atomic energy decomposition + SO(3) equivariance + force-weighted training, which gives physically correct Hessian curvature by design. Train with `train_mace_model.py`; validate with `validate_pes_frequencies.py --mace-model`.
- **MACE architecture for MVKO**: `hidden_irreps='64x0e + 64x1o'`, `r_max=5.0 Å`, 2 interaction layers, `forces_weight=100`, float64, SWA. Requires: `mace-torch >= 0.3.4`, `ase`. Training runs `mace_run_train` via subprocess (MACE CLI). MPS backend (Apple Silicon) auto-selected. Best checkpoint: `mace_model_run-0_stagetwo.pt` (SWA) > `_run-0.pt` (best epoch).
- **MACE companion file**: `MACEDriver` requires a `.symbols.pkl` file alongside the `.pt` model (saves atom ordering). `train_mace_model.py` writes `mace_model.symbols.pkl` automatically. Without it, pass `symbols=` explicitly to `MACEDriver(model_path, symbols=[...])`.
- **MACE delta-ML composability**: `MACEDriver` implements the same bakken interface as `MLPESDriver` and `SGDMLDriver`. It can be wrapped with `NMDeltaDriver` for the CASSCF δ_S0 correction without any changes: `NMDeltaDriver(MACEDriver(mace_pt), nm_delta_model_path)`. Use `--mace-model` + `--nm-delta-model` in `ir_md_spectrum.py`.
- **sGDML failure modes (documented for reference)**: (1) Training data incoherence — sGDML assumes single-trajectory MD17-style data; multi-T + NM-displacement data violates this, giving 89× force-integral mismatch. (2) Kernel memory — (n_train × 3N)²: n_train=900, 3N=36 → 23.5 GB; must cap at ~300 frames. (3) Descriptor stiffness — same 1/r second derivatives as Coulomb+KRR appear regardless of training setup.
- **sGDML vs Coulomb+KRR**: Both produce C-H modes at 10,000–38,000 cm⁻¹. sGDML uses Matérn 5/2 on inverse-distances (66 elements for 12-atom molecule); Coulomb uses RBF on upper-triangle Coulomb matrix (78 elements). Neither can produce physical Hessian — use MACE.
- **sGDML delta-ML (legacy)**: `SGDMLDriver` implements the same interface as `MLPESDriver` and can be wrapped with delta-ML drivers. Use `--sgdml-model` + `--nm-delta-model` in `ir_md_spectrum.py`.
- **Force training disabled**: `MLPESConfig.train_forces = False` by default. Enabling it breaks the KRR model. Forces should be predicted via finite differences on the ML-PES energy surface or via a separate force model.
- **StandardScaler is critical**: The ML-PES uses `StandardScaler` on both features and targets. Without it, RMSE degrades from ~0.04 kcal/mol to ~170 kcal/mol. See `modules/ml_pes_fixed.py` for the authoritative implementation.
- **Coulomb matrix descriptor**: Simple upper-triangle of `Z_i*Z_j/r_ij` (off-diagonal) and `0.5*Z_i^2.4` (diagonal). Robust but not permutation-invariant — atom ordering must be consistent.
- **Dipole surface separate from energy surface**: `DipoleSurface` in `ir_spectroscopy.py` is a separate KRR model predicting 3-component dipole vectors. The energy `MLPESTrainer` only predicts scalar energy.
- **Mock PSI4 fallback**: All direct MD functions detect PSI4 absence and return physically plausible mock values for testing.
- **CRITICAL — PSI4 version consistency**: ALL training data, NM displacement energies, and validation energies must be computed with the SAME PSI4 installation. The Dec 2025 training data gave energies ~8.5 kcal/mol lower than PSI4 1.10 (current). Mixing data from different PSI4 versions causes systematic offsets that make ML-PES validation fail catastrophically (15–40 kcal/mol errors). When starting fresh or after PSI4 upgrades, regenerate ALL training data with the current PSI4. The safe pipeline is: `generate_nm_training.py` (NM + PSI4 MD, all fresh) → `two_phase_workflow.py` (validation).

## Environment Patches Required (MACE 0.3.15 + PyTorch 2.2.2 + macOS)

Three bugs must be patched every time `mace-torch`, `torch`, or `numpy` are reinstalled.
These affect the conda environment `jupyterenv` on this machine.

### 1 — NumPy mixed 1.x/2.x installation

**Symptom**: `ImportError: cannot import name 'ERR_IGNORE' from 'numpy.core.umath'`
when running `mace_run_train`.

**Cause**: Conda installed numpy 1.x files in `numpy/core/` (notably `_ufunc_config.py`
and `_methods.py`); `pip install numpy==2.2.6` updated `numpy/_core/` but left the 1.x
shim files in place. The stale `_ufunc_config.py` imports `ERR_IGNORE` which was removed
from the public API in numpy 2.0.

**Fix**:
```bash
pip install "numpy<2.0"   # installs 1.26.4 — internally consistent
```

### 2 — `torch.compiler.is_compiling` absent in PyTorch 2.2.2

**Symptom**: `AttributeError: module 'torch.compiler' has no attribute 'is_compiling'`
at first validation step during `mace_run_train`.

**Cause**: `torch.compiler.is_compiling()` was added in PyTorch 2.3. PyTorch 2.2 has
the equivalent at `torch._dynamo.is_compiling()`.

**Fix** — patch `mace/modules/utils.py` line 604:
```python
# Replace:
if not torch.compiler.is_compiling():

# With:
_is_compiling = getattr(getattr(torch, 'compiler', None), 'is_compiling', None) \
    or getattr(getattr(torch, '_dynamo', None), 'is_compiling', lambda: False)
if not _is_compiling():
```

Location: `$(python3 -c "import mace.modules.utils as m; print(m.__file__)")`

### 3 — MPS backend rejects float64 in MACE forward pass

**Symptom**: `TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS
framework doesn't support float64.` in `mace/modules/models.py:580`.

**Cause**: MACE hardcodes `.double()` in `MACE.forward()` to accumulate node energies
with float64 precision. Apple Silicon MPS has no float64 support at all.

**Fix** — patch `mace/modules/models.py` line 580:
```python
# Replace:
node_energy = node_e0.clone().double() + node_inter_es.clone().double()

# With (dtype-preserving — float32 on MPS, float64 on CPU):
node_energy = node_e0.clone().to(node_inter_es.dtype) + node_inter_es.clone()
```

Location: `$(python3 -c "import mace.modules.models as m; print(m.__file__)")`

**Note on MPS vs CPU performance**: Despite the float32 patch, CPU is faster than MPS
for this workload (~47 s/epoch vs ~75 s/epoch) because MACE scatter operations and
small batch sizes (10 molecules) do not amortise MPS dispatch overhead. `train_mace_model.py`
defaults to `--device cpu` until a larger batch size or PyTorch 2.3+ resolves this.

### Resolving all three permanently

Upgrading to `torch>=2.3` fixes patches 2 and 3. Upgrading to a clean conda/pip numpy
(not a mixed install) fixes patch 1. Until then, these three patches are applied to the
installed `mace-torch` package files.

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

5. **Multi-state CASSCF(4,4) delta-ML PES family** (✅ grid complete, IR run in progress, April 2026): Three-surface delta-ML architecture on top of the wB97X-D/6-31G* base ML-PES.

   **Energy architecture**:
   ```
   E_S0(R) = E_wB97X_ML(R)  + δ_S0_ML(R)          ← ground state, always used
   E_S1(R) = E_S0(R)         + Δgap_S1_ML(R)        ← first excited singlet
   E_T1(R) = E_S0(R)         + Δgap_T1_ML(R)        ← lowest triplet (ISC)
   ```

   **Active space**: CASSCF(4,4) with {n⁺(O_terminal), n⁻(O_terminal), π(COO), π*(COO)} — the COO biradical frontier orbitals. Actual NO occupations at equilibrium: (1.948, 1.724, 0.275, 0.052) — 14% biradical character. `NO_OCC_SWITCH_THRESHOLD = 0.20`.

   **Why no NEVPT2**: wB97X-D already captures dynamic correlation. Adding NEVPT2 would double-count it. δ_S0 captures only the static (multi-reference/biradical) correction.

   **Completed training data**: 232/240 frames clean (8 failures, all isolated mid-chain). Output: `outputs/casscf_wB97X_nm_grid_20260407_184904/`.

   **KRR models** (γ=0.1, α=1e-6, LOO-CV): δ_S0=0.654, Δgap_S1=0.588, Δgap_T1=0.566 kcal/mol.
   - `nm_delta_s0_model.pkl`, `nm_gap_s1_model.pkl`, `nm_gap_t1_model.pkl`

   **wB97X base ML-PES** (`outputs/wB97X_surface_20260406_223155/`):
   - 894 training frames (all dE_B3LYP < 100 kcal/mol), γ=0.001, α=1e-5, RMSE=0.31 kcal/mol (augmented)
   - **CRITICAL**: Use `train_wB97X_model.py` with `tune_hyperparameters=False` — the MLPESTrainer grid search only sweeps α ∈ {0.01, 0.1, 1.0}, missing the optimal 1e-5, and picks γ=0.01 which gives only 2 IR peaks.
   - **CRITICAL**: Use `mlpes_wB97X_aug.pkl` (1126 frames = 894 base + 232 NM-grid), NOT `mlpes_wB97X.pkl` (894 only). The augmented model includes C-H stretch geometries from the CASSCF grid that provide the repulsive wall. Without it, C-H bonds elongate to > 2.0 Å during 300K MD (imaginary modes −2300, −953 cm⁻¹).
   - Script `augment_wB97X_with_nm_grid.py` rebuilds augmented model from `results.json`.

   **IR run command** (v3, in progress):
   ```bash
   python3 ir_md_spectrum.py \
       --model outputs/wB97X_surface_20260406_223155/mlpes_wB97X_aug.pkl \
       --training-data outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \
       --nm-delta-model outputs/casscf_wB97X_nm_grid_20260407_184904/nm_delta_s0_model.pkl \
       --steps 30000 --temp 300 --preminimize --zpe-min-freq 50 --zpe-max-freq 4000 \
       --n-trajectories 5 --max-bond-extension 2.0 \
       --output-dir outputs/ir_spectrum_wB97X_delta_v3_300K
   ```
   Note: dipoles from B3LYP training data (wB97X dipoles all zero due to PSI4 oeprop bug in `recompute_wB97X_surface.py`).

   **Scripts**:
   - `test_casscf_equilibrium.py` — SA-2-CASSCF + triplet at eq; validates active space, reports gaps
   - `casscf_wB97X_nm_grid.py` — 240-frame NM grid; resume with `--resume <outdir> --hessian <path>`; retrain with `--retrain-only --resume <outdir> --hessian <path>`
   - `train_wB97X_model.py` — assemble `training_data_wB97X.npz` + train `mlpes_wB97X.pkl` from existing `results.json` (no PSI4); use `--gamma 0.001 --alpha 1e-5`
   - `augment_wB97X_with_nm_grid.py` — merge base + CASSCF-grid wB97X frames; saves `mlpes_wB97X_aug.pkl`
   - `plot_delta_surfaces.py` — 3D plot of δ_S0, Δgap_S1, Δgap_T1 surfaces
   - `plot_wB97X_surface.py` — 3D plot of ΔE_wB97X with harmonic reference overlay

   **Bug fixed** (April 2026): `casscf_nm_delta.py` `NMKRRDeltaModel.save()` failed with `PicklingError` when called from a `__main__` script. Fixed by saving a plain state dict and reconstructing on load (backward-compatible with old pickles).

   **Surface hopping for MD** (planned): `MultiStateMDDriver` in `modules/bakken.py` will wrap the three-surface model. Gap-based hopping probability per step; tracks `current_state`; dipole ACF on occupied surface.

   **Three target applications**:
   1. **IR absorption (300 K)**: S0 surface only; δ_S0 correction improves equilibrium description
   2. **IR emission from hot ozonolysis MVKOO**: high-T init (2000–5000 K); surface hopping S0↔T1
   3. **Unimolecular dissociation**: long trajectories; branching to dioxirane, VHP+OH, or CH₃CHO+O

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
