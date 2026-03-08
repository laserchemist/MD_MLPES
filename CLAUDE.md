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

## Planned Features (In Progress / Next Steps)

The following extensions were scoped by the user and should be built consistently with existing conventions:

1. **Normal mode distortions for adaptive sampling**: ✅ **Implemented** in `modules/normal_modes.py` + `generate_nm_training.py`. Computes PSI4 Hessian, diagonalises mass-weighted Hessian, displaces ±n×a_thermal(T) along each mode (FREQ_CONV = 5140.48 cm⁻¹/√(Hartree/(Bohr²·amu))). Run: `python3 generate_nm_training.py --training-data <data.npz> --T-nm 1000 --n-amplitudes 4 --max-factor 3`.

2. **ML-PES quality testing via normal modes and MD**: After training, compute normal mode frequencies on the ML-PES and compare to PSI4 frequencies. Run short ML-MD and flag frames where ML energy/force error exceeds threshold; add those frames to training set.

3. **Energy + force consistency via Hessian**: Compute forces as `-dE/dR` analytically where possible. For KRR, gradient of the energy prediction w.r.t. coordinates is analytically available via the kernel gradient.

4. **IR spectra via dipole ACF**: Framework exists in `ir_spectroscopy.py`. Requires dipole moments along ML-MD trajectories — use `DipoleSurface` model or recalculate with PSI4 for key frames.

5. **Multi-state PES family**: Future design — maintain a family of `MLPESTrainer` models (one per electronic state or molecular species). Mix energies/forces with scalar coefficients or transition between states with a hopping probability. Design: `PESFamily` container holding `{label: MLPESTrainer}` with a `mix(coefficients)` or `hop(probability_matrix)` interface. Units and descriptor conventions must be identical across all family members.

## Output Conventions

- Training data: `outputs/training_with_dipoles_*/`
- Trained models: `outputs/refined_*/mlpes_model_*.pkl`
- MD trajectories: `md_output/`
- IR spectra: `ir_spectrum_output/`
- Workflow state: JSON files tracking completed steps and file paths
