# MD-MLPES: Machine Learning Potential Energy Surface for Molecular Dynamics

A Python framework for training and running ML-accelerated molecular dynamics using quantum chemistry (PSI4) as the reference method. Targets the **CH₂OO Criegee intermediate** at B3LYP/6-31G* but is easily extended to other molecules.

---

## Overview

Direct *ab initio* MD (AIMD) is accurate but slow — every step calls PSI4. This framework uses **Kernel Ridge Regression (KRR)** with a Coulomb matrix descriptor to learn the potential energy surface from a compact set of PSI4 calculations, then runs MD ~100× faster on the learned surface.

An **adaptive refinement loop** automatically identifies geometries where the ML-PES is inaccurate, adds them to the training set with correct PSI4 energies/forces, and retrains — with no manual intervention.

```
PSI4 (B3LYP/6-31G*)           ~1–2 s/point
  → Training data              NM distortions + multi-T MD
    → KRR ML-PES               ~90 steps/s
      → ML-MD trajectory       100× faster
        → PSI4 validation      spot-check every Nth frame
          → Adaptive refine    add high-error frames, retrain
            → Production MD    converged, accurate
```

---

## Requirements

```bash
conda install -c psi4 psi4          # PSI4 1.10
pip install scikit-learn numpy matplotlib tqdm
```

---

## Quick Start

### 1. Generate training data (NM distortions + PSI4 MD)

```bash
python3 generate_nm_training.py \
    --training-data outputs/clean_psi410_20260308_203552/training_data.npz \
    --T-nm 1000 --n-amplitudes 4 --max-factor 3 \
    --md-temps 300,600 --md-steps 100
```

This computes the PSI4 Hessian, generates displaced geometries along each normal mode at amplitudes up to 3×*a*_thermal(1000 K), runs PSI4 MD at 300 K and 600 K, combines everything, and trains a KRR model with hyperparameter search.

### 2. Validate with two-phase diagnostic

```bash
python3 two_phase_workflow.py \
    --model outputs/nm_training_TIMESTAMP/mlpes_model_nm.pkl \
    --training-data outputs/nm_training_TIMESTAMP/combined_training_data.npz \
    --steps 200 --temp 300
```

Phase 1 runs 200-step ML-MD. Phase 2 validates every snapshot against PSI4.
**Current result: 0.64 kcal/mol mean error, 1.15 kcal/mol max.**

### 3. Production adaptive MD

```bash
python3 production_adaptive_md.py \
    --model outputs/nm_training_20260308_203606/mlpes_model_nm.pkl \
    --training-data outputs/nm_training_20260308_203606/combined_training_data.npz \
    --steps 2000 --temp 300 --max-cycles 5 --error-threshold 2.0
```

Runs 2000-step (1 ps) ML-MD, validates 100 snapshots with PSI4, and refines automatically if any error exceeds 2 kcal/mol. Saves a diagnostic figure after every run.

---

## Repository Structure

```
modules/                     Core library
  direct_md.py               PSI4 interface + Velocity-Verlet MD engine
  ml_pes.py                  KRR trainer (StandardScaler, force training disabled)
  normal_modes.py            Hessian → normal modes → displaced geometries
  data_formats.py            TrajectoryData I/O (npz, xyz, extxyz, hdf5)
  test_molecules.py          Pre-defined molecules (CH2OO, water, methane …)
  ir_spectroscopy.py         Dipole ACF → IR spectrum via FFT
  visualization.py           Matplotlib helpers

generate_nm_training.py      Generate training data (NM + PSI4 MD) + retrain
two_phase_workflow.py        Phase 1: ML-MD  |  Phase 2: PSI4 validation
production_adaptive_md.py    Production MD with automatic adaptive refinement

outputs/
  nm_training_TIMESTAMP/     Training data + trained model
  adaptive_production_*/     Per-cycle snapshots, validation, figure
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **KRR + Coulomb matrix** | Simple, robust, no permutation issue for fixed atom ordering |
| **StandardScaler on X and y** | Critical — without it RMSE degrades ~1000× |
| **Force training disabled** | Adding forces to KRR objective breaks convergence |
| **PSI4 version consistency** | ALL training data must come from the same PSI4 build. Mixing versions causes 5–10 kcal/mol systematic offsets. |
| **NM displacements at 1000 K** | Covers anharmonic regions missed by equilibrium-only sampling |

---

## Adaptive Refinement Figure

After each production run `production_adaptive_md.py` saves `production_run_figure.png`:

| Panel | Content |
|-------|---------|
| Trajectory | ML & PSI4 energies along MD time |
| Parity plot | ML vs PSI4 energy, coloured by error |
| Error vs time | Per-snapshot \|ML − PSI4\| with threshold line |
| Histogram | Error distribution (all cycles overlaid) |
| Forces | PSI4 force magnitude distribution |
| Summary | Convergence table or run metadata |

---

## Unit Conventions

| Quantity | Unit |
|----------|------|
| Coordinates | Ångström |
| Energies | Hartree (display: kcal/mol) |
| Forces | Hartree/Ångström |
| Dipoles | Debye |
| Time | femtoseconds |
| Temperature | Kelvin |

Conversion constants live in `modules/direct_md.py` — do not redefine elsewhere.

---

## Planned Extensions

- **ML-PES quality via normal mode comparison** — compute frequencies on ML surface, compare to PSI4
- **Analytical forces** — KRR gradient w.r.t. coordinates via kernel gradient
- **IR spectra** — dipole ACF along ML-MD trajectories using `ir_spectroscopy.py`
- **Multi-state PES family** — `PESFamily` container for multiple electronic states

---

## Citation / Acknowledgements

Quantum chemistry calculations use [PSI4](https://psicode.org/).
ML training uses [scikit-learn](https://scikit-learn.org/).
