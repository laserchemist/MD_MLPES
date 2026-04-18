# ML-PES / ML-MD Research Log

---

## 2026-04-17 — Root Cause of IR Failures Confirmed; MACE Backend Implemented; Training Running

### Root cause: 1/r descriptor stiffness is intrinsic, not fixable by tuning

All prior IR spectrum failures (missing C-H region 2800–3200 cm⁻¹, imaginary modes,
MD dissociation) share a single root cause: **Coulomb-matrix and sGDML descriptors
both encode geometry through inverse pairwise distances (1/r), whose second derivatives
are intrinsically stiff under RBF/Matérn kernels**.

Evidence:
- Coulomb+KRR (B3LYP base, γ=0.001): max C-H NM freq = 15,000–38,000 cm⁻¹ (physical: ~3,100 cm⁻¹)
- Analytic KRR Hessian (exact chain rule, implemented March 2026): same unphysical frequencies — rules out FD step-size as cause
- sGDML Matérn-5/2 on inverse distances (sig sweep 0.05–5.0): max C-H freq = **38,342 cm⁻¹**
- γ sweeps, α sweeps, energy cutoffs, use_E_cstr, symmetry enforcement: none change the Hessian eigenvalue structure

The stiffness arises because ∂²(1/r)/∂R² ~ 1/r³, which diverges near the bond
length regardless of kernel width. This is not a hyperparameter problem.

### sGDML: investigated and deprecated

`modules/sgdml_pes.py`, `train_sgdml_model.py`, `validate_sgdml_frequencies.py`
implemented and run. Additional failure modes beyond the descriptor stiffness:

| Failure | Root cause |
|---------|-----------|
| 89× force-integral mismatch | Multi-T + NM data violates sGDML path-integral coherence assumption (designed for MD17-style single-trajectory data) |
| Kernel OOM at n_train=900 | Kernel matrix = (n_train × 3N)² = (900×36)² = 23.5 GB; must cap at n_train≤300 |
| 8 imaginary NM modes at n_train=300 | Insufficient data density near equilibrium when capped |
| MD dissociation at step 40 | Imaginary modes amplified by ZPE init → immediate trajectory explosion |
| Near-eq RMSE 0.69 kcal/mol but 192 kcal/mol on full validation set | use_E_cstr + dE<5 filter creates a model that only works in a tiny near-eq bubble |

**Conclusion**: sGDML deprecated. Same Hessian pathology as Coulomb+KRR; additionally
incompatible with our multi-temperature + NM-displacement training data strategy.

### MACE: infrastructure complete, training running

**Why MACE solves the problem**: MACE (Multi-Atomic Cluster Expansion) uses:
1. **Local atomic energy decomposition** — total energy = Σ E_i(local environment); no global 1/r descriptor
2. **SO(3) equivariance by construction** — correct symmetry without symmetry enforcement overhead
3. **Force-weighted training** (forces_weight=100) — PES curvature constrained by forces at every training point

These design choices mean the Hessian is physically determined by the training forces,
not by descriptor second derivatives. Expected result: C-H modes at ~3,100 cm⁻¹ (physical).

**New files**:
- `modules/mace_pes.py` — `MACEDriver` (bakken-compatible; `.energy()`, `.forces()`, `.analytic_forces()`, `.analytic_hessian()` via FD on MACE analytic forces; auto MPS/CUDA/CPU). `npz_to_extxyz()` converts training npz to MACE extxyz (eV / eV/Å, `REF_energy`/`REF_forces` keys, 20 Å cell).
- `train_mace_model.py` — Full training pipeline: npz→extxyz, train/valid split, `mace_run_train` subprocess, checkpoint discovery, saves `mace_model.pt` + `mace_model.symbols.pkl`.
- `validate_pes_frequencies.py` — **Generalized validation** replacing `validate_sgdml_frequencies.py`. Works with all three backends via `--mace-model`/`--sgdml-model`/`--model`. Prints PASS/FAIL/WARN with unphysical-mode count and redirect to MACE if KRR artifact detected.
- `ir_md_spectrum.py` — `--mace-model` flag added; wired through `run_ir_workflow()`.

**MACE composability**: `MACEDriver` implements the identical bakken interface as `MLPESDriver`
and `SGDMLDriver`. The CASSCF delta-ML correction (`NMDeltaDriver`) wraps it without any
changes: `NMDeltaDriver(MACEDriver(mace_pt), nm_delta_model_path)`.

### Environment patches required for MACE 0.3.15 + PyTorch 2.2.2 + macOS

Three bugs encountered and fixed (patching installed packages):

**1. NumPy mixed 1.x/2.x installation** (`numpy 2.2.6`):

Conda env had `numpy/core/_ufunc_config.py` from numpy 1.x alongside the rest of numpy
2.2.6. The 1.x file imports `ERR_IGNORE` from `numpy.core.umath`, which was removed in
numpy 2.0. The `pip install --force-reinstall numpy==2.2.6` did not overwrite the stale
conda-installed file.

Fix: `pip install "numpy<2.0"` → numpy 1.26.4 (internally consistent).

**2. `torch.compiler.is_compiling` absent in PyTorch 2.2.2** (`mace/modules/utils.py:604`):

`torch.compiler.is_compiling()` was added in PyTorch 2.3. PyTorch 2.2 has the equivalent
at `torch._dynamo.is_compiling()`.

Patch applied to `mace/modules/utils.py:604`:
```python
# Before:
if not torch.compiler.is_compiling():

# After:
_is_compiling = getattr(getattr(torch, 'compiler', None), 'is_compiling', None) \
    or getattr(getattr(torch, '_dynamo', None), 'is_compiling', lambda: False)
if not _is_compiling():
```

**3. MPS backend rejects float64** (`mace/modules/models.py:580`):

MACE hardcodes `.double()` in the forward pass to accumulate node energies with float64
precision. Apple Silicon MPS does not support float64 tensors at all (not even as a
precision override). CPU is also faster than MPS for this workload (~23 s/epoch vs ~75 s/epoch)
because MACE's scatter operations and small batch sizes (10) don't amortise MPS dispatch overhead.

Patch applied to `mace/modules/models.py:580`:
```python
# Before:
node_energy = node_e0.clone().double() + node_inter_es.clone().double()

# After (dtype-preserving; float32 on MPS, float64 on CPU):
node_energy = node_e0.clone().to(node_inter_es.dtype) + node_inter_es.clone()
```

Training runs on CPU with float64. MPS is set as default in `train_mace_model.py` but
automatically overridden to CPU when training small-molecule batches.

**Note for future installs**: These three patches are required every time `mace-torch` is
upgraded or `torch`/`numpy` are reinstalled. Consider pinning:
- `numpy<2.0` (or wait for mace-torch to declare numpy 2.x support)
- `torch>=2.3` once available (resolves patches 2 and 3)

### MACE training: in progress

**Command**:
```bash
python3 train_mace_model.py \
    --training-data outputs/wB97X_surface_20260406_223155/training_data_wB97X_aug.npz \
    --output-dir outputs/mace_wB97X_20260417 \
    --n-train 900 --n-valid 80 --epochs 500 --energy-cutoff 50 --device cpu
```

**Data**: 999/1126 frames (dE < 50 kcal/mol), 900 train / 80 valid / 19 dropped as test.
Energy cutoff widened from 15→50 kcal/mol vs KRR workflow — MACE handles high-energy frames
without corrupting equilibrium (unlike KRR where adding 484 kcal/mol frames killed the model).

**Architecture**: `hidden_irreps='64x0e + 64x1o'`, r_max=5.0 Å, 2 interaction layers,
forces_weight=100, energy_weight=1 (stage 1) → 1000 (stage 2 SWA after epoch 450),
float64, Adam lr=0.01 with cosine scheduler.

**Early training progress** (CPU, ~47 s/epoch):

| Epoch | Force RMSE | Energy RMSE/atom |
|-------|-----------|-----------------|
| Initial | 303.7 meV/Å | 54.7 meV |
| 0  | 79.8 meV/Å  | 167.8 meV |
| 4  | 56.4 meV/Å  | 72.2 meV  |
| 10 | 48.2 meV/Å  | 60.4 meV  |
| 22 | 44.6 meV/Å  | 60.4 meV  |

Force RMSE at epoch 22 = 44.6 meV/Å = 1.03 kcal/mol/Å, converging well.
Estimated completion: ~3.5 h (epoch 500 ≈ 02:45 on 2026-04-18).
Log: `outputs/mace_wB97X_20260417/train_log.txt`

### Next steps

1. Check training convergence at completion; target F-RMSE < 10 meV/Å
2. `python3 validate_pes_frequencies.py --mace-model outputs/mace_wB97X_20260417/mace_model.pt ...`
   — verify 0 imaginary, 0 unphysical (>5000 cm⁻¹) modes, C-H stretch at ~3100 cm⁻¹
3. If PASS: run IR spectrum with MACE + δ_S0 NM-delta correction:
   ```bash
   python3 ir_md_spectrum.py \
       --mace-model    outputs/mace_wB97X_20260417/mace_model.pt \
       --nm-delta-model outputs/casscf_wB97X_nm_grid_20260407_184904/nm_delta_s0_model.pkl \
       --training-data outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \
       --steps 30000 --temp 300 --preminimize --zpe-min-freq 50 --zpe-max-freq 4000 \
       --n-trajectories 5 --max-bond-extension 2.0
   ```
4. Compare IR spectrum to B3LYP+KRR result — specifically check for C-H stretch peak ~3100 cm⁻¹

---

## 2026-04-09 — CASSCF NM Grid Complete + wB97X+δ_S0 IR Run in Progress

### CASSCF(4,4) NM grid: completed

`casscf_wB97X_nm_grid.py` finished (PID 5719, started 2026-04-07):
- **232/240 frames computed, 8 failed** (isolated mid-chain frames in modes 7, 14, 22, 23)
- All 30 modes covered; failures do not break KRR (surrounding factors present)
- Runtime: ~36 h total; MO chaining suppressed state-switching throughout
- Output: `outputs/casscf_wB97X_nm_grid_20260407_184904/results.json`

**KRR training results** (`--retrain-only`):

| Model | Best γ | Best α | LOO-CV | Target |
|-------|--------|--------|--------|--------|
| δ_S0  | 0.1    | 1e-6   | 0.654 kcal/mol | < 0.5 ✗ |
| Δgap_S1 | 0.1  | 1e-6   | 0.588 kcal/mol | — |
| Δgap_T1 | 0.1  | 1e-6   | 0.566 kcal/mol | — |

All three models converge to γ=0.1, α=1e-6 — the smallest gamma in the grid, indicating
the correction surface is broad and smooth in NM space. The δ_S0 LOO-CV (0.654) just misses
the 0.5 target but is driven by high-amplitude frames (δ_S0 up to +13.9 kcal/mol); near
equilibrium (f ≤ 1.0, 300 K sampling region) corrections are ≤ 1 kcal/mol.

δ_S0 range: −1.23 to +13.94 kcal/mol; Δgap_S1: 19.21–34.53 kcal/mol; Δgap_T1: 17.24–31.37 kcal/mol.

**Bug fix**: `casscf_nm_delta.py` `NMKRRDeltaModel.save()` originally used `pickle.dump(self)`
which failed with `PicklingError: not the same object` when called from a script running as
`__main__`. Fixed by saving a plain state dict and reconstructing on load:
```python
def save(self, path):
    state = {field: getattr(self, field) for field in [...]}
    pickle.dump(state, f)

@classmethod
def load(cls, path):
    obj = pickle.load(f)
    return obj if isinstance(obj, cls) else cls(**obj)  # backward compat
```

### wB97X-D training set: expanded to 894 frames

`recompute_wB97X_surface.py` completed the remaining 400 frames (495–894) overnight.
Final: **894 frames, 0 failures**. Energy range 0–101 kcal/mol.

`train_wB97X_model.py` (new helper script, bypasses PSI4) assembles `training_data_wB97X.npz`
and trains `mlpes_wB97X.pkl` with fixed γ/α (no grid search). Key flag:
`MLPESConfig(tune_hyperparameters=False)` required to bypass MLPESTrainer's built-in
grid search (which only sweeps α ∈ {0.01, 0.1, 1.0}, missing the optimal 1e-5).

### wB97X ML-PES: C-H elongation problem and fix

**v1** (γ=0.01, 894 frames): Only 2 IR peaks (171, 322 cm⁻¹). Root cause: MLPESTrainer
grid search picked γ=0.01 which gives stiffer, less accurate near-equilibrium forces.

**v2** (γ=0.001, 894 frames): C-H bonds elongated to > 2.0 Å during MD. Root cause:
wB97X ML-PES has **two imaginary modes** (−2300, −953 cm⁻¹) from Coulomb+RBF artifact —
worse than B3LYP (−1888 cm⁻¹). No training data at stretched C-H geometries → no
repulsive wall → molecule slides along imaginary directions.

**v3 (augmented, current)**: Fixed by merging 894 base frames with 232 CASSCF NM-grid frames.
The NM-grid includes C-H stretch modes (24–29) up to 2.98 Å — providing the repulsive wall.
Augmented model (γ=0.001, α=1e-5, 1126 frames): **train RMSE = 0.31 kcal/mol**. IR run
launched as `outputs/ir_spectrum_wB97X_delta_v3_300K/` (PID 46412, in progress).

Script: `augment_wB97X_with_nm_grid.py` — reconstructs NM-displaced coords from grid
`q_nm` vectors + Hessian eigenvectors, merges with base training set, retrains.

### Visualisation scripts

Two new 3D surface plots for the NM-grid results (X=amplitude, Y=mode freq, Z=correction):
- `plot_delta_surfaces.py` — 3-panel: δ_S0, Δgap_S1, Δgap_T1 (saved to `surfaces_3d.png`)
- `plot_wB97X_surface.py` — single panel: ΔE_wB97X with harmonic reference overlay; color capped at 20 kcal/mol by default (saved to `wB97X_surface_3d.png`)

Both work with partial results (mid-run) and update on each call.

---

## 2026-04-07 — Multi-State CASSCF(4,4) Delta-ML Design + wB97X Surface Complete

### wB97X-D/6-31G* surface: completed

`recompute_wB97X_surface.py` finished overnight:
- **494 frames computed, 0 failed** (all frames with dE_B3LYP < 100 kcal/mol)
- Energy range: 0–101 kcal/mol (wB97X relative)
- δ(wB97X − B3LYP) near equilibrium (dE < 5 kcal/mol): 0.06 ± 0.28 kcal/mol — confirming wB97X and B3LYP give nearly identical relative energies for this molecule at near-eq geometries
- Equilibrium frame: index 502 in `combined_training_data.npz` (same frame as B3LYP minimum), E = −306.209985 Ha
- Output: `outputs/wB97X_surface_20260406_223155/results.json` (738 KB)
- `training_data_wB97X.npz` and `mlpes_wB97X.pkl` to be assembled after retrain step

### Multi-state CASSCF delta-ML: design finalised

**Three target applications** drive the architecture requirements:
1. **IR absorption (300 K thermal)**: 300 K ML-MD on corrected S0 surface; dipole ACF → spectrum. δ_S0 correction expected to be small (< 3 kcal/mol near eq).
2. **IR emission from hot ozonolysis-born MVKOO**: Criegee intermediates form with ~50–80 kcal/mol internal energy from MVK + O₃. Need high-T ML-MD (2000–5000 K effective) with surface hopping S0↔T1 (ISC relevant). Dipole ACF on occupied surface gives emission spectrum.
3. **Unimolecular dissociation dynamics**: Long trajectories crossing barriers to dioxirane, vinyl hydroperoxide + OH, CH₃CHO + O channels. Requires surface hopping near conical intersections.

**Energy architecture** (three-surface family):
```
E_S0(R) = E_wB97X_ML(R)  + δ_S0_ML(R)       ← always used; primary dynamics surface
E_S1(R) = E_S0(R)         + Δgap_S1_ML(R)    ← first excited singlet; gap ~2–3 eV at eq
E_T1(R) = E_S0(R)         + Δgap_T1_ML(R)    ← lowest triplet; gap ~1.5–2 eV at eq; ISC
```

Learning gaps (Δgap) rather than absolute deltas from wB97X is numerically preferred: gaps are smaller quantities (~35–70 kcal/mol, smooth) vs absolute offsets (~300 kcal/mol). The KRR fits smoother targets.

**Active space**: CASSCF(4,4) with COO biradical orbitals {n⁺(O_terminal), n⁻(O_terminal), π(COO), π*(COO)}. This replaces the original TS active space {σ/σ*(C-H), σ/σ*(O-H)} which was wrong for the equilibrium region. The COO active space morphs continuously: near-closed-shell at equilibrium (NO ~ 1.98, 1.90, 0.10, 0.02) → full biradical at the TS (NO ~ 1.5, 1.5, 0.5, 0.5).

**Why NEVPT2 is skipped with wB97X base**: wB97X-D captures dynamic correlation through the range-separated exchange-correlation functional. CASSCF(4,4) adds only the static (multi-reference/biradical) correction. Adding NEVPT2 on top of CASSCF would double-count the dynamic correlation already in wB97X — the δ_S0 would be non-physical. With B3LYP as base, NEVPT2 helped because B3LYP underestimates dynamic correlation more severely. Conclusion: CASSCF only, no NEVPT2, when base = wB97X.

**Training data (Option B — NM grid)**:
- 30 modes × [0.5, 1.0, 1.5, 2.0] × ±1 = **240 frames**
- Generated around wB97X equilibrium geometry using B3LYP Hessian (NM directions nearly identical between functionals)
- Amplitudes capped at 2.0× thermal (300 K) to stay within wB97X ML-PES reliable region
- Per-frame: PSI4 wB97X-D single-point (~90 s) + PySCF SA-2-CASSCF(4,4)/6-31G* singlets (~120 s) + SS-CASSCF triplet (~60 s). MO chaining along each mode direction.
- Estimated runtime: ~18 h total; checkpoint/resume after every frame
- Three KRR outputs: `nm_delta_s0_model.pkl`, `nm_gap_s1_model.pkl`, `nm_gap_t1_model.pkl`

**Surface hopping design** (planned, for `modules/bakken.py`):
- `MultiStateMDDriver` wraps the three-surface model
- Gap-based hopping probability per step: P_hop(S0→S1) ∝ exp(−B × Δgap_S1) × dt
- ISC (S0→T1): same form with smaller spin-orbit coupling prefactor
- Tracks `current_state` in trajectory; dipole ACF recorded on occupied surface
- For dissociation: bond-extension detection for C-O, O-O bonds → terminate + record products

### Equilibrium test results (test_casscf_equilibrium.py)

Both SA-2-CASSCF singlet and SS-CASSCF triplet converged at the wB97X equilibrium geometry:

| Quantity | Value | Note |
|---|---|---|
| CASSCF S0 | −304.5422196022 Ha | reference for grid |
| wB97X (eq) | −306.2099845838 Ha | reference for grid |
| δ_S0 at eq | 0.000 kcal/mol | 0 by construction |
| NO occ S0 | [1.948, 1.724, 0.275, 0.052] | **new COO reference** |
| NO occ S1 | [1.948, 1.000, 1.000, 0.052] | pure open-shell singlet |
| NO occ T1 | [1.949, 1.000, 1.000, 0.051] | triplet counterpart |
| Δgap_S1 | 29.0 kcal/mol (1.26 eV) | CASSCF level; true gap higher with NEVPT2 |
| Δgap_T1 | 26.5 kcal/mol (1.15 eV) | T1 < S1 by 2.5 kcal/mol — Hund's rule confirmed |

Key observation: S0 NO occ [1.948, **1.724**, **0.275**, 0.052] shows ~14% biradical character
at equilibrium — real physical COO biradical character, not a state-switching artifact.
The old EQ_NO_OCC_REF [1.998, 1.924, 0.077, 0.000] was for the C-H bond active space and
does not apply here.  New threshold: 0.20 for the COO active space.

S1 and T1 both have two singly-occupied orbitals [1.0, 1.0] — the biradical open-shell
states.  T1 lies 2.5 kcal/mol below S1 (Hund's rule).  Both are thermally inaccessible
at 300 K (29/26.5 >> kT = 0.59 kcal/mol) but relevant for hot ozonolysis MVKOO
(~50–80 kcal/mol internal energy).

### Scripts produced this session

```
test_casscf_equilibrium.py     ← equilibrium validation (completed, results above)
casscf_wB97X_nm_grid.py        ← 240-frame NM grid; ready to run
```

### Immediate next steps

1. Run NM grid overnight (~18 h):
   ```bash
   python3 casscf_wB97X_nm_grid.py \
       --eq-coords outputs/mvko_20260319_081314/psi4_eq_coords.npy \
       --hessian   outputs/casscf_nm_delta_20260401_110049/hessian_used.npy
   ```
2. If interrupted: `--resume outputs/casscf_wB97X_nm_grid_<ts>`
3. After completion: inspect `diagnostics.png`, check LOO-CV for δ_S0 < 0.5 kcal/mol
4. Run IR spectrum with corrected wB97X model using `--nm-delta-model nm_delta_s0_model.pkl`

---

## 2026-04-06 — ωB97X-D3 Surface Recompute + Systematic CASSCF/NEVPT2 NM Grid

### Motivation

Four attempts at CASSCF(4,4) delta-ML on the B3LYP surface have all failed or been
marginal, sharing three root causes:

1. **Wrong active space near equilibrium**: the {σ/σ*(C-H), σ/σ*(O-H)} active space was
   designed for the H-transfer IRC transition state.  At the MVKO equilibrium the active
   NOs are nearly doubly occupied (1.998, 1.924, 0.077, 0.000); CASSCF(4,4) finds a
   lower minimum 17 kcal/mol below the B3LYP equilibrium → correction is not smooth near
   equilibrium.

2. **State-switching contamination**: PySCF CASSCF converges to excited states at some
   geometries → bimodal δ distribution (δ = 1 kcal/mol for normal frames, 11–31 kcal/mol
   for switched frames).  NEVPT2 clean model (36 frames) improved LOO-CV from 9.3 to
   1.60 kcal/mol but still slightly above kT = 0.59 kcal/mol at 300 K.

3. **Sparse, non-uniform sampling**: existing CASSCF single-points come from the MD/NM
   training frames (non-uniform in NM space), making the KRR training set poorly conditioned.

### Strategy: two parallel surfaces for benchmarking

Rather than continuing to patch the CASSCF correction on a B3LYP base, we pursue a
parallel track:

**Track 1 — Better DFT base (ωB97X-D3/6-31G*)**

`ωB97X-D3` is the preferred functional for Criegee intermediates:
- Range-separated hybrid → better charge-transfer / zwitterion (C⁺)(O-O⁻) character
- D3BJ dispersion → more physical O-O and C-O-O stretch curves beyond equilibrium
- Reference level of ANI-2x (ωB97X/6-31G*) → enables later ANI-2x zero-shot comparison
- Smaller multireference correction needed → CASSCF δ shrinks → NM-KRR more reliable

Plan: recompute all ~700 frames (dE_B3LYP < 100 kcal/mol) at ωB97X-D3/6-31G* overnight.
PSI4 call: `energy('wb97x-d', return_wfn=True)` (PSI4 1.10 uses Grimme D2-style suffix).
Keep B3LYP surface in parallel for direct comparison of IR spectra and Hessian frequencies.

Script: `recompute_wB97X_surface.py`

**Track 2 — Systematic CASSCF/NEVPT2 NM grid with state tracking**

Replace the non-uniform training set with a systematic NM displacement grid:
- 30 modes × 5 amplitudes × 2 signs = 300 candidate frames
- Amplitudes: ±[0.5, 1.0, 1.5, 2.0, 3.0] × a_thermal(300 K)  [uniform q-space coverage]
- For each frame: SA-2-CASSCF(4,4) with `state_weights=[0.5, 0.5]`
- State tracking: extract NO occupancies, compare against reference at equilibrium
  (flag switched if |ΔNO_occ_k| > 0.15 for any active orbital)
- Chain MO guess: `mc.kernel(mo_coeff_prev)` along each displacement direction
- SC-NEVPT2 on the ground state
- Store per-state energies for S0 and S1 → enables multi-state PES family

The multi-state family: at MD time, the `NEVPTDeltaDriver` selects:
  - `δ = δ_S0(q)` when gap E_S1 − E_S0 > threshold (all thermally accessible 300 K geometries)
  - `δ = min(δ_S0, δ_S1)` near barrier crossings (840 K / H-transfer TS region)

Script: `casscf_nm_systematic.py`
Post-processing: `train_nm_delta_clean.py` (filter by NO consistency, NM-KRR grid search)

### Active space for equilibrium vs IRC

| Region | Appropriate active space | Notes |
|---|---|---|
| Equilibrium + IR (300 K) | CASSCF(2,2): HOMO π / LUMO π* of COO block | Captures zwitterion ↔ biradical balance |
| H-transfer IRC / barrier | CASSCF(4,4): {σ/σ*(C-H), σ/σ*(O-H)} | Designed for bond-breaking reaction |
| Full-surface delta-ML | SA-2-CASSCF(4,4) with state tracking | Compromise: (4,4) with strict filtering |

The equilibrium active space mismatch is the root cause of failure.  For the systematic
grid, SA-2-CASSCF(4,4) is retained (for continuity with existing data) but with strict
NO-occupancy filtering to reject state-switched frames.  Future work should compare
CASSCF(2,2) vs CASSCF(4,4) near equilibrium.

### Expected outcome

With 200+ clean frames on a uniform NM grid:
- LOO-CV RMSE target: < 0.5 kcal/mol (was 1.60 with 36 random frames)
- Near-equilibrium σ(δ) target: < 0.3 kcal/mol (was 1.05 with mixed frames)
- Benchmark: compare B3LYP IR vs ωB97X-D3 IR vs (B3LYP + NEVPT2 correction) IR

### Scripts produced this session

```
recompute_wB97X_surface.py       ← overnight ωB97X-D3 recompute of ~700 frames
casscf_nm_systematic.py          ← systematic NM grid + SA-CASSCF + NEVPT2 + state tracking
```

---

## 2026-04-03 — Descriptor Benchmark: ACSF, Pairwise Distances, and Coulomb Matrix with Global KRR

### Objective

Test whether replacing the Coulomb matrix with ACSF (Atom-Centered Symmetry Functions)
in the existing KRR framework improves the C-H Hessian stiffness artifact while
maintaining energy prediction accuracy.  Run in parallel with the ongoing
CH-retrained Coulomb+KRR model (952 frames) to preserve benchmarking continuity.

### Setup

Training data: 783 near-equilibrium frames (dE < 50 kcal/mol) from the
952-frame combined dataset.  5-fold cross-validation RMSE as the metric.

DScribe ACSF parameters (default Behler 2011 organic):
- R_cut = 6.0 Å
- G2: 8 (η, Rs) pairs, η = [0.5, 1.0, 2.0, 3.5, 6.0, 10.0, 18.0, 36.0] Å⁻², Rs=0
- G4: 12 (η, λ, ζ) triples, η=[0.01, 0.1], λ=[1,-1], ζ=[1,2,4]
- Per-atom features: 99 (24 G2 + 72 G4 + 3 G1 cutoff-sum terms)

Aggregation strategies tested:
1. **concatenate**: stack all 12 × 99 = 1188-dim global descriptor
2. **sum_by_species**: sum per element, concatenate → 3 × 99 = 297-dim
3. **pairwise distances**: sGDML-style, r_ij for i<j → 66-dim (no ACSF)
4. **Coulomb matrix**: Z_iZ_j/r_ij → 78-dim (existing baseline)

### Results

| Descriptor | Dim | Best γ | Best α | 5-fold CV RMSE |
|---|---|---|---|---|
| ACSF concatenate | 1188 | 3e-5 | 1e-6 | 2.49 kcal/mol |
| ACSF sum_by_species | 297 | 1e-3 | 1e-3 | 3.26 kcal/mol |
| Pairwise distances | 66 | 1e-2 | 1e-5 | **0.235 kcal/mol** |
| Coulomb matrix | 78 | 3e-3 | 1e-5 | **0.152 kcal/mol** |

The ACSF descriptors perform 10–20× worse than Coulomb matrix with global KRR.
The pairwise distance descriptor is close to Coulomb (0.24 vs 0.15 kcal/mol) but
still worse.  The Coulomb matrix remains the best descriptor for global KRR on
this dataset.

### Interpretation

**ACSF is incompatible with global KRR for a fixed single molecule.**

ACSF was designed for neural networks where each atom has its own network and
the total energy is a sum of atomic contributions:
```
E = Σ_i NN_i(G_i)
```
The local per-atom descriptor G_i captures the chemical environment of each atom
independently.  When aggregated to a global descriptor for a single KRR model:
- **Concatenation** (12 × 99 = 1188 features) creates a descriptor space too large
  for KRR with ~800 training points.  The RBF kernel cannot meaningfully interpolate
  in 1188D with only 800 examples — the kernel matrix is nearly singular and
  regularisation dominates.
- **Summing** (99 or 297 features) destroys the atomic identity information that
  ACSF was designed to preserve.  Two very different geometries can have similar sum
  ACSF vectors if the per-atom environment contributions happen to cancel, making
  the descriptor non-injective.

The Coulomb matrix works well because it is directly the full pairwise interaction
information in a compact 78D form.  Every off-diagonal element Z_iZ_j/r_ij encodes
a specific pairwise distance weighted by chemical identity.  For a fixed-atom-ordering
single molecule with ~1000 training frames, this is a near-optimal global descriptor
for energy-label KRR.

**The Hessian stiffness is not a descriptor problem alone.**

Even the pairwise distance descriptor (which eliminates Z_iZ_j/r³ stiffness) requires
a larger γ to achieve comparable RMSE.  The second derivative of the KRR energy:
```
∂²E_KRR/∂R² = Σ_i α_i K_i [4γ²(...)² - 2γ Σ_k ∂²d_k/∂R² - ...]
```
has a 2γ × ∂²r_ij/∂R² ~ 2γ/r term (1/r from the distance chain rule).  With
γ_dist ≈ 0.01 vs γ_Coulomb ≈ 0.001, and r_CH ≈ 1.1 Å:
- Coulomb: 2 × 0.001 × Z_CZ_H / r³ ≈ 0.009 per C-H pair
- Distance: 2 × 0.01 × 1 / r ≈ 0.018 per C-H pair

The pairwise distance descriptor actually has MORE Hessian stiffness contribution
per C-H pair because γ must be 10× larger to achieve comparable energy RMSE.
The Z_iZ_j weighting in Coulomb is partially offset by the smaller optimal γ.

**Fixing the Hessian requires force learning, not descriptor substitution.**

The core issue is that KRR learns an energy surface and computes forces by
differentiating it.  Any global energy descriptor + RBF kernel will have Hessian
curvature that reflects descriptor second derivatives, not molecular physics.

The principled solutions are:
1. **sGDML**: learns the force field directly; energy is the integral; Hessian is
   the first derivative of the learned forces (well-conditioned).
2. **ANI-2x / per-atom NN**: per-atom networks with local descriptors; energy is
   a sum of properly learned atomic energies; Hessian follows physical curvature.

### Conclusion

The ACSF+KRR parallel model experiment fails to improve on Coulomb+KRR for this
use case.  The key finding for the methods comparison story:

- Coulomb+KRR: best energy accuracy (0.15 kcal/mol), stiff C-H Hessian artifact
- Pairwise dist+KRR: slightly worse accuracy (0.24 kcal/mol), same stiffness issue
- ACSF+KRR: much worse accuracy (2.5 kcal/mol), even worse Hessian

The natural next step in building a physically sound parallel model is **sGDML**,
which uses the same pairwise distance descriptor but learns forces rather than
energies.  The existing PSI4 training data includes forces for all 904 base frames,
satisfying sGDML's training requirement.

### Scripts produced

```
modules/acsf_descriptor.py     ← ACSFDescriptor (3 aggregation modes)
modules/pairwise_descriptor.py ← PairwiseDistanceDescriptor (sGDML-style)
train_acsf_model.py            ← grid search + CV + Hessian check
outputs/acsf_model_20260403_153824/  ← ACSF concatenate run (γ=3e-5, RMSE 2.38)
outputs/acsf_model_20260403_154057/  ← ACSF sum_by_species run (γ=1e-3, RMSE 3.18)
```

### IR Spectrum of CH-retrained Model — Near-Equilibrium Degradation

The IR run completed (5 × 30000 steps, 300 K, PID 30559).

**CH-retrained model peaks** (`outputs/ir_spectrum_CH_retrain_300K/`):

| Peak (cm⁻¹) | Rel. intensity | Notes |
|---|---|---|
| 275 | 1.000 | Torsion/bending (spurious dominant) |
| 86 | 0.953 | Spurious low-frequency |
| 232 | 0.907 | Torsion/bending |
| 363 | 0.369 | |
| 520 | 0.175 | C-O-O bending (shifted from 513) |

**Baseline B3LYP model** (`outputs/ir_spectrum_20260319_174321/`):
322 (1.000), 893 (O-O stretch), 513 (C-O-O bend), 658 (C-O stretch)

**Finding**: The O-O stretch peak (893 cm⁻¹) has disappeared and the spectrum is
dominated by spurious sub-400 cm⁻¹ activity.  This is the same near-equilibrium PES
corruption as the adaptive model (March 2026): adding high-energy C-H stretch frames
(up to 484 kcal/mol) shifts the KRR kernel weight away from the equilibrium region.

Evidence: the analytic Hessian showed 3 imaginary NM modes (vs 5 in the original model
but 0 in the target PSI4 B3LYP Hessian).  The pre-minimiser converged in 1 step with
|F|_max = 0.004 Ha/Å (barely below the 0.005 threshold), indicating the ML-PES minimum
has shifted away from the correct equilibrium.

**Root cause (recurrent pattern):**

| Model | Imaginary NM modes | O-O peak | C-O-O bend |
|---|---|---|---|
| Base B3LYP 904 frames | 5 | 893 cm⁻¹ ✓ | 513 cm⁻¹ ✓ |
| Adaptive 1300 frames | 3 | absent ✗ | absent ✗ |
| CH-retrained 952 frames | 3 | absent ✗ | 520 cm⁻¹ (shifted) |

Whenever high-energy frames (> ~100 kcal/mol above minimum) are added to the training
set, the RBF kernel is re-partitioned to cover the new energy range, reducing the
effective density of kernel support near the equilibrium.  The ML-PES minimum shifts.

**Fix (documented in CLAUDE.md)**: Always include an explicit anchor frame at the
PSI4-optimised geometry (forces ≈ 0) when adding high-energy frames.  Without this
anchor, the equilibrium can become a saddle point.

**Conclusion**: The baseline B3LYP model (904 frames, `mlpes_initial.pkl`) remains the
best model for 300 K IR spectra.  H-wandering can be guarded at the MD level
(bond extension guard) rather than by corrupting the near-equilibrium PES.

---

## 2026-04-03 — ML Descriptor Landscape: ACSF, ANI-2x, and sGDML vs Coulomb Matrix

### Motivation

The persistent C-H Hessian stiffness artifact (modes at 5700–8000 cm⁻¹ vs true ~3100 cm⁻¹)
and the H-wandering artifact in ML-MD trajectories both originate from the Coulomb matrix
descriptor, not from insufficient training data or hyperparameter choices.  This note
documents three alternative approaches — ACSF, ANI-2x, and sGDML — and their relative
strengths and weaknesses for the MVKO IR spectroscopy problem.

### Root Cause of Coulomb Matrix Stiffness

The Coulomb matrix off-diagonal element is `C_ij = Z_i Z_j / r_ij`.  Its second derivative
with respect to atomic positions is `∂²C_ij/∂R² ∝ Z_i Z_j / r³`, which is largest for the
lightest, closest atom pairs — exactly C-H bonds.  When the RBF KRR energy surface is
differentiated twice for the Hessian, this amplification makes all C-H modes appear 2–3×
too stiff, regardless of training data volume, kernel width, or regularisation.  It is a
descriptor design flaw, not a data or hyperparameter problem.

The flat-PES H-wandering artifact is a separate consequence: the RBF kernel decays to zero
beyond the training boundary, making the ML-PES asymptote to a constant (the mean energy)
rather than a repulsive wall.  Adding targeted C-H stretch frames (2026-04-03) addresses
this dynamical artifact but does not fix the Hessian stiffness.

---

### ACSF — Atom-Centered Symmetry Functions (Behler-Parrinello)

**Principles**

Energy is decomposed as a sum of atomic contributions:

```
E = Σ_i ε_i(G_i)
```

Each atomic energy `ε_i` depends only on the local chemical environment within a cutoff
radius `R_c` (~4–6 Å), encoded via two families of symmetry functions:

Radial (G²) — pair interactions:
```
G²_i = Σ_{j≠i} exp(-η(R_ij - R_s)²) · f_c(R_ij)
```

Angular (G⁴) — three-body interactions:
```
G⁴_i = 2^{1-ζ} Σ_{j,k≠i} (1 + λ cos θ_ijk)^ζ · exp(-η(R²_ij + R²_ik + R²_jk)) · f_c·f_c·f_c
```

The cutoff function `f_c(R) = 0.5[cos(πR/R_c) + 1]` is smooth and goes to zero at `R_c`,
so the descriptor and all its derivatives are continuous — no 1/r singularity and no
stiffness artifact.  The Gaussian basis in G² is centred on interatomic *distances* rather
than inverse distances, so the second derivative of the descriptor w.r.t. atomic positions
is well-behaved at typical C-H bond lengths.

ACSF functions are invariant to translation, rotation, and permutation of equivalent atoms
by construction.

**Key citations**

1. Behler, J.; Parrinello, M. *Phys. Rev. Lett.* **2007**, *98*, 146401 — original method
2. Behler, J. *J. Chem. Phys.* **2011**, *134*, 074106 — full G1–G5 catalog, parameter guidelines
3. Behler, J. *Int. J. Quantum Chem.* **2015**, *115*, 1032 — tutorial review
4. Himanen et al. *Comput. Phys. Commun.* **2020**, *247*, 106949 — DScribe library (`pip install dscribe`)

**Feasibility for this project**

High.  DScribe provides `dscribe.descriptors.ACSF` as a drop-in numpy-compatible
implementation.  The plan is to write an `ACSFDescriptor` wrapper matching the existing
`compute(symbols, coords)` / `compute_batch()` API, then retrain with the same 952 frames
and a γ grid search (descriptor size grows from 78 to ~12 × 50 = 600 features, so γ
will need re-tuning, likely 0.0001–0.001).  No additional PSI4 calculations needed.

Expected gain: C-H Hessian modes at ~3050–3300 cm⁻¹ (currently 5700–8000 cm⁻¹), correct
ZPE initialization of C-H modes, and a physically meaningful IR C-H stretch region.

---

### ANI-2x

**Principles**

ANI (Smith et al.) builds on Behler-Parrinello with three key additions:

1. **Tuned AEV descriptor** (Atomic Environment Vector): species-pair-specific radial terms
   and species-triple-specific angular terms, so C-H and H-H environments are described by
   different basis functions.

2. **Element-specific sub-networks**: separate neural network weight sets for each element
   (H, C, N, O, S, F, Cl), allowing the model to learn element-specific bonding physics
   rather than relying on atomic-number scaling.

3. **Active learning on chemical space**: ANI-1x used uncertainty-guided sampling across
   conformers derived from SMILES strings; ANI-2x extended coverage to S and halogens.
   Reference level: **ωB97X/6-31G*** — handles non-covalent interactions and partial
   charge-transfer states better than B3LYP.

For MVKO (C₄H₆O₂): all elements (C, H, O) are in the original ANI-1 training set — no
extrapolation for element coverage.

**Key citations**

1. Smith et al. *Chem. Sci.* **2017**, *8*, 3192 — ANI-1
2. Smith et al. *Nat. Commun.* **2019**, *10*, 2903 — ANI-1ccx (transfer learning to CCSD(T))
3. Devereux et al. *J. Chem. Theory Comput.* **2020**, *16*, 4192 — ANI-2x (adds S, F, Cl)
4. Gao et al. *J. Chem. Inf. Model.* **2020**, *60*, 3408 — TorchANI implementation

**Feasibility for this project**

Two sub-options:

*Zero-shot evaluation (half day)*: evaluate pre-trained ANI-2x on the 952 MVKO training
geometries.  If RMSE vs B3LYP < ~3 kcal/mol and Hessian frequencies are correct, ANI-2x
can be used directly for MD without any retraining.

*Delta-ML correction (2–3 days)*: train a KRR model on the residual
`δ = E_B3LYP - E_ANI2x`.  Because ANI-2x already captures most of the C-H physics, the
delta is small (< 5 kcal/mol range vs the current 0–484 kcal/mol span), making KRR
interpolation much more accurate for the same training set size.

---

### sGDML — Symmetric Gradient Domain Machine Learning

**Principles**

sGDML (Chmiela, Tkatchenko, Müller) inverts the standard energy-first paradigm.  All other
approaches learn energy then differentiate for forces.  sGDML learns the **force field
directly**, recovering energy by integration.  This is a fundamental architectural choice:

The kernel for forces is derived from the energy kernel by double differentiation:
```
K_F(x, x') = ∇_x ⊗ ∇_{x'} K_E(x, x')
```

Because the model *is* the derivative of a scalar potential, the learned force field is
always curl-free — energy conservation is guaranteed by construction, not enforced
approximately.  Energy drift in long MD runs is eliminated at the mathematical level.

**Descriptor**: pairwise distances {r_ij} — global like the Coulomb matrix, but without
the 1/r weighting.  Permutation symmetry of equivalent atoms is handled analytically by
projecting the kernel onto the symmetric subspace of the permutation group of the molecule.

**Why the Hessian is better**: In Coulomb+KRR, the Hessian is the second derivative of
the learned energy surface — amplifying descriptor curvature.  In sGDML, the Hessian is
the *first* derivative of the learned force field.  With Matérn-class kernels on pairwise
distances, this first derivative tracks true molecular curvature rather than amplifying
1/r³ behaviour.  In practice, sGDML gives Hessian frequencies that match DFT reference
values accurately with 200–500 training geometries for 7–20 atom molecules.

**Data efficiency**: Because each force label constrains the PES in `3N` independent
directions (vs 1 for an energy label), sGDML is extremely data-efficient.  The 2017/2018
papers demonstrate near-chemical-accuracy (~0.1–0.3 kcal/mol) for 7–10 atom molecules
with only 200–500 force evaluations.  For MVKO (12 atoms), the 904 base force-labelled
frames would be comfortably over-determined.

**Key citations**

1. Chmiela et al. *Sci. Adv.* **2017**, *3*, e1603015 — original GDML
2. Chmiela et al. *Nat. Commun.* **2018**, *9*, 3887 — sGDML (symmetry adaptation)
3. Sauceda et al. *J. Chem. Phys.* **2019**, *150*, 114102 — coupled-cluster application, MD
4. Chmiela et al. *Comput. Phys. Commun.* **2023**, *290*, 108711 — sGDML software (sgdml.org)

**Key weakness for this project**

sGDML was designed for near-equilibrium MD of single-conformer small molecules (ethanol,
malonaldehyde, aspirin; 7–21 atoms; sampling within ~10–15 kcal/mol of minimum).

The MVKO training set has a fundamentally different character: 48 C-H stretch frames at up
to 484 kcal/mol above minimum, with C-H bonds extended to 2.60 Å.  The pairwise-distance
descriptor + RBF/Matérn kernel extrapolates poorly to this dissociative regime — the same
flat-PES problem as the Coulomb model, just at a different length scale.  The energy
conservation guarantee holds only within the interpolation regime.

Additionally, sGDML's O(N²) scaling in training set size makes prediction slow for
> ~2000 training frames, limiting the total data that can be incorporated.

---

### Full Comparison

| Feature | Coulomb + KRR | ACSF + KRR | ANI-2x | sGDML |
|---|---|---|---|---|
| Descriptor type | Global Coulomb | Local atom-centered | Local AEV | Global pairwise distances |
| Primary training target | Energy | Energy | Energy | **Forces** |
| Permutation invariant | No (fixed atom order) | Yes | Yes | Yes (symmetry adapt) |
| C-H Hessian stiffness | Severe | Fixed | Fixed | Likely fixed |
| Data efficiency | Moderate | Moderate | Pre-trained | **Very high** |
| Force labels required | No | No | No | **Yes (all frames)** |
| Energy conservation in MD | No | Approximate | Yes | **Yes by construction** |
| Reactive / dissociative PES | Partial (wall data) | Better | Partial | **Problematic** |
| Scales beyond ~30 atoms | Yes (degrades) | Yes | Yes | No (O(N²) kernel) |
| Implementation effort | Done | ~1–2 days | ~half day | ~1–2 days |

---

### Recommended Strategy for MVKO

**Immediate** (this week): let the CH-retrained Coulomb+KRR model (952 frames) complete
the IR run.  This addresses H-wandering dynamically.

**Next experiment**: zero-shot ANI-2x evaluation on the 952 training geometries.  Near-zero
implementation cost; immediately reveals whether the stiffness artifact is eliminated and
whether the PES shape is accurate for MVKO.

**Targeted methods comparison** (publishable): ACSF+KRR vs sGDML vs Coulomb+KRR, using
the 904 near-equilibrium frames only (dE < 50 kcal/mol).  The comparison narrative:
- Coulomb+KRR: simple baseline, wrong C-H frequencies, H-wandering artifact
- sGDML: correct frequencies, energy-conserving MD, but restricted to near-eq regime
- ACSF+KRR: correct frequencies, handles large-amplitude C-H stretch, physically
  motivated local descriptor, no dissociation issues

This three-way comparison is a coherent methods contribution in the context of MVKO
IR emission spectroscopy.

---

## 2026-04-03 — C-H Stretch Training Data and ML-PES Retraining

### Objective
MVKO ML-MD trajectories show H atoms drifting 2.5+ Å from their bonded carbon atoms
(visible as -CH biradical + H₂ fragments in rendered trajectories).  The root cause is
that the Coulomb-matrix ML-PES has no repulsive wall for C-H dissociation beyond the
NM training data (~1.2 Å), so H atoms freely drift once ZPE-initialized velocities
push them toward 1.5–2 Å.

The `--max-bond-extension` guard in `bakken.py` deliberately skips X-H bonds (by design,
to avoid false positives from quantum ZPE oscillations), so no safety net catches the drift.

### Root cause: ML-PES flat beyond C-H training boundary

At the equilibrium Hessian level, the Coulomb+RBF kernel predicts C-H modes at
9825–15005 cm⁻¹ (old model) vs the true PSI4 B3LYP values of 3049–3324 cm⁻¹.
This stiffness artifact (factor ~3× too high) is intrinsic to the Coulomb matrix
second derivatives under the RBF kernel and cannot be cured by kernel width tuning.

The MD H-wandering is a distinct consequence: beyond the training boundary (~1.3 Å),
the RBF kernel decays to zero and the ML-PES flattens to a constant (the mean training
energy), creating an artificial "zero-force plateau" where H atoms can roam freely.

### Solution: targeted C-H stretch NM displacement training data

Rather than running high-T PSI4 MD (8000 K), which hangs on soft bending modes with
a_thermal >> 1 Å, only the 6 C-H stretch modes (≥ 2500 cm⁻¹) are displaced at large
amplitudes.

**Script**: `generate_ch_stretch_training.py`
- Loads pre-computed Hessian (`outputs/casscf_nm_delta_20260401_110049/hessian_used.npy`)
- Selects 6 modes with ω ≥ 2500 cm⁻¹ (modes 24–29: 3049–3324 cm⁻¹)
- Generates ±1 to ±8 × a_thermal(T=8000K) displacements for each mode
- a_thermal for C-H modes at 8000K: 0.14–0.20 Å (×10 factor = 1.4–2.0 Å max stretch)
- Runs PSI4 B3LYP/6-31G* single-points; 10/96 failed (extreme compressions, atoms too close)

**Results**: `outputs/nm_ch_stretch_20260402_225350/nm_displacements.npz`
- 86 successful frames (96 attempted − 10 PSI4 failures)
- Energy range: 8–13,390 kcal/mol above minimum
- 48 frames survive dE < 500 kcal/mol filter (used for training)
- C-H bond coverage (kept frames): 222 normal (<1.3 Å), 51 stretch (1.3–2.0 Å),
  15 dissociated (>2.0 Å), max C-H = 2.60 Å

### Retraining

**Script**: `retrain_with_ch_stretch.py`

Combined 904 base frames + 48 filtered CH-stretch frames = **952 total frames**.
Retrained with same hyperparameters: γ=0.001, α=1e-5.

**Accuracy**:
| Energy range | OLD RMSE | NEW RMSE |
|---|---|---|
| Near-eq (dE < 5 kcal/mol), training data | 0.072 kcal/mol | 0.080 kcal/mol |
| Mid-energy (5–30 kcal/mol), training data | 0.161 kcal/mol | **0.105 kcal/mol** |
| Overall test set (10% split, 0–484 kcal/mol) | 0.27 kcal/mol | 1.43 kcal/mol |

Near-eq accuracy essentially unchanged. Mid-energy accuracy improved 37%.  The
overall test RMSE increase is expected: the test set now includes extreme CH-stretch
frames (100–484 kcal/mol) which are hard to interpolate accurately with γ=0.001.

**Normal mode Hessian check** (analytic KRR Hessian at PSI4 eq):

| Mode range | OLD | NEW |
|---|---|---|
| Imaginary modes | 5 | 3 |
| Lowest real mode | 195 cm⁻¹ | 215 cm⁻¹ |
| Bottom C-H cluster (modes 19–23) | 2572–3648 cm⁻¹ | 2694–3779 cm⁻¹ |
| Upper C-H cluster (modes 24–29) | 5200–7940 cm⁻¹ | 5734–6583 cm⁻¹ |
| PSI4 B3LYP reference | 3049–3324 cm⁻¹ | (same) |

The bottom 5 C-H modes improved toward the physical range.  The upper 6 modes remain
2–3× too high — this is the Coulomb+RBF curvature artifact, not cured by training data
(requires analytic descriptor redesign, e.g. ACSF or ANI-2x).

The repulsive wall is now present in the training data up to 2.60 Å.  H-wandering
should be suppressed dynamically even without a perfect Hessian.

### New model and IR run

- **New model**: `outputs/mvko_ch_retrain_20260403_101429/mlpes_ch_retrained.pkl`
- **Combined dataset**: `outputs/mvko_ch_retrain_20260403_101429/combined_training_data.npz`
- **IR run (in progress)**: `outputs/ir_spectrum_CH_retrain_300K/` (PID 30559, started 2026-04-03)

```bash
python3 ir_md_spectrum.py \
    --model outputs/mvko_ch_retrain_20260403_101429/mlpes_ch_retrained.pkl \
    --training-data outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \
    --steps 30000 --temp 300 --preminimize --zpe-min-freq 50 --zpe-max-freq 4000 \
    --n-trajectories 5 --max-bond-extension 2.5 \
    --output-dir outputs/ir_spectrum_CH_retrain_300K
```

### Files produced
```
generate_ch_stretch_training.py            ← new script (targeted C-H NM displacements)
retrain_with_ch_stretch.py                 ← new script (combine + retrain)
outputs/nm_ch_stretch_20260402_225350/
  nm_displacements.npz                     ← 86 C-H stretch frames (use with dE<500 filter)
outputs/mvko_ch_retrain_20260403_101429/
  combined_training_data.npz               ← 952 frames (904 base + 48 CH-stretch)
  mlpes_ch_retrained.pkl                   ← retrained model (γ=0.001, α=1e-5)
```

---

## 2026-04-02 — NEVPT2 Delta-ML: State-Switching Discovery and Clean Model

### Objective
Following the CASSCF(4,4) NM-coordinate delta-ML framework developed 2026-04-01,
extend to CASSCF+NEVPT2 (dynamic correlation) via PySCF to test whether NEVPT2
can correct the B3LYP ML-PES for the O-O and C-O stretch modes that dominate the
MVKO IR emission spectrum at 840 K.

### New scripts
- `casscf_nevpt2_correction.py` — full pipeline: load PSI4 CASSCF frames from
  `outputs/casscf_nm_delta_*/all_casscf_results.json` → PySCF SC-NEVPT2 single
  points → relative corrections → NM-KRR grid search → `NEVPTKRRModel` save.
  CLI: `--casscf-dir`, `--training-data`, `--resume`, `--retrain-only`,
  `--gamma-values`, `--alpha-values`, `--output-dir`.
- `modules/nevpt2_pyscf.py` — PySCF interface:
  `compute_casscf_nevpt2(symbols, coords, ...)` returns `e_hf, e_casscf,
  e_nevpt2_corr, e_nevpt2, delta_nevpt2, dipole_casscf, no_occ, converged, error`.
  Active space: CASSCF(4,4)/6-31G*, SC-NEVPT2; ~80 s/frame on M-series Mac.

### Original model results (`outputs/nevpt2_correction_20260401_194425/`)
- 50/51 frames converged (1 PySCF CASSCF failure)
- LOO-CV RMSE: **9.3 kcal/mol** (δ_total = δ_CASSCF + δ_NEVPT2)
- Best hyperparameters: γ=1e-4, α=1e-4
- Near-equilibrium correction: ~0.6 kcal/mol (small, reasonable)
- Bond-stretch frames: δ_total = −9 to −21 kcal/mol (B3LYP too stiff)

However, at the equilibrium geometry the model predicted:
- Correction = −1.9 kcal/mol (should be ~0 since eq frame has δ=0 by construction)
- Max |∂δ/∂q_k| at eq = 10.2 kcal/mol/NM-unit → 37 kcal/mol/Å in Cartesian

These unphysical gradients corrupted ML-MD dynamics and the runs had to be killed.

### Root cause: CASSCF state-switching artifacts

Inspecting the 50 training frames by plotting δ vs RMS NM displacement:

| Pattern | Count | Example |
|---------|-------|---------|
| Tiny displacement (rms_q < 0.04), δ ≈ −15 to −17 kcal/mol | 6 | frames 44–49 |
| Small displacement (rms_q ≈ 0.1–0.3), δ ≈ −16 to −18 kcal/mol | 4 | frames 13, 14, 33 |
| Moderate displacement (rms_q ≈ 0.4–0.8), δ ≈ −9 to −16 kcal/mol | 4 | frames 2, 5, 8, 17 |
| Positive outliers (rms_q ≈ 0.33), δ ≈ +15 kcal/mol | 2 | frames 20, 21 |

In all cases, neighbouring frames at **identical** RMS displacement have δ < 1 kcal/mol.
The only consistent explanation is PySCF CASSCF converging to an excited state
(different orbital occupation) rather than the ground state.  Evidence:
- Frames 44–49 are displaced by rms_q ≈ 0.04 from equilibrium (< ZPE amplitude)
  yet carry δ ≈ −16 kcal/mol — physically impossible if the ground state PES is smooth
- The pattern is identical to the PSI4 CASSCF Fix A bimodal distribution (2026-04-01),
  where the IRC active space reorganised along torsional modes

Identification criterion: **rms_q < 1.0 AND |δ| > 5 kcal/mol** → 14 suspicious frames.

### Clean model (`nevpt2_clean_model.pkl`)

Removed the 14 suspicious frames (indices 2, 5, 8, 13, 14, 17, 20, 21, 33, 44–49).
Retrained on 36 clean frames with grid search:

| γ | 1e-5 | 1e-4 | 1e-3 | 1e-2 | 1e-1 |
|---|------|------|------|------|------|
| 1e-4 | 1.81 | 2.03 | 2.04 | 2.52 | 3.91 |
| 3e-4 | 1.83 | 1.88 | 2.05 | 2.21 | 3.17 |
| 1e-3 | 1.91 | 1.85 | 1.97 | 2.05 | 2.55 |
| 3e-3 | 1.83 | 1.92 | 1.90 | 2.02 | 2.28 |
| 1e-2 | **1.60** | 1.77 | 1.96 | 2.01 | 2.19 |

Best: γ=0.01, α=1e-5, **LOO-CV = 1.60 kcal/mol** (9.3 → 1.6, factor 5.8 improvement).

Clean model properties at equilibrium:
- Prediction = +0.59 kcal/mol (vs training value 0.0; driven by nearby clean frames
  with δ ≈ 0.3–0.9 kcal/mol — physically reasonable constant near-eq offset)
- Max |∂δ/∂q_k| at eq = **1.13 kcal/mol/NM-unit** (vs 10.2 before; factor 9 reduction)
- Train RMSE = 0.12 kcal/mol (good interpolation)
- Near-eq frames (rms_q < 0.15): δ range [−0.28, 0.90] kcal/mol ← consistent with
  the earlier geometry-KRR finding (mean +0.62 ± 1.05 kcal/mol, 2026-03-31)

The clean model is physically sensible: a slow, smooth ramp from ~0.6 kcal/mol
at equilibrium toward larger corrections at bond-stretch geometries.

### Limitation: LOO-CV still exceeds kT at 300 K

LOO-CV = 1.60 kcal/mol > kT = 0.59 kcal/mol at 300 K.  The model cannot reliably
predict the per-frame correction better than thermal noise.  This is driven by the
large-displacement frames (rms_q > 0.7) where δ varies widely (−5 to −21 kcal/mol)
within the 36 clean training points.  At 300 K MD, the molecule rarely reaches
rms_q > 0.5, so the dominant near-eq regime (δ ≈ 0.6 kcal/mol ± 0.4) is
well-captured.  At 840 K, larger amplitudes become accessible but the LOO-CV is
comparable to kT(840 K) ≈ 1.67 kcal/mol — the model is near its reliability limit.

**Expected IR effect**: a small, approximately constant correction near equilibrium
should shift all modes by < 1 cm⁻¹.  Any observed red-shift in O-O/C-O peaks
between the 840K NEVPT2 and 840K B3LYP spectra will be physically informative but
should be interpreted as an upper bound on the NEVPT2 effect.

### IR spectrum runs (in progress at time of writing)

Using the clean model with `--nm-delta-model nevpt2_clean_model.pkl`:

```bash
python3 ir_md_spectrum.py \
    --model outputs/mvko_20260319_081314/mlpes_initial.pkl \
    --training-data outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \
    --nm-delta-model outputs/nevpt2_correction_20260401_194425/nevpt2_clean_model.pkl \
    --steps 30000 --temp 300 --preminimize --zpe-min-freq 50 --zpe-max-freq 4000 \
    --n-trajectories 5 --max-bond-extension 2.5 \
    --output-dir outputs/ir_spectrum_NEVPT2clean_300K
```

- 300K absorption: `outputs/ir_spectrum_NEVPT2clean_300K/`
- 840K emission: `outputs/ir_spectrum_NEVPT2clean_840K/`
- Baseline B3LYP: `outputs/ir_spectrum_20260319_174321/` (893, 513, 658, 322 cm⁻¹)
- Estimated run time: ~2.3 hours each (0.056 s/step × 5 traj × 30000 steps)

### Files produced
```
casscf_nevpt2_correction.py            ← new pipeline script
modules/nevpt2_pyscf.py                ← PySCF CASSCF+NEVPT2 interface
outputs/nevpt2_correction_20260401_194425/
  nevpt2_correction_model.pkl          ← original 50-frame model (corrupted, LOO 9.3)
  nevpt2_clean_model.pkl               ← clean 36-frame model (LOO 1.6)  ← USE THIS
  nevpt2_results.json                  ← per-frame energies and diagnostics
  summary.json                         ← grid search and model summary
  diagnostics.png                      ← scatter plot δ_CASSCF vs δ_NEVPT2
```

---

## 2026-04-01: NM-Coordinate Delta-ML Implementation

### Why Coulomb matrix fails for delta-ML

The existing `casscf_surface_correction.py` uses Coulomb matrix descriptors for the
CASSCF−B3LYP delta-ML model. This has a fundamental flaw for delta-ML near equilibrium:

- **All MVKO geometries at 300 K have K ≈ 0.999** — the Coulomb matrix is dominated by
  heavy-atom nuclear charges (Z_C=6, Z_O=8) which are invariant. Small thermal distortions
  (0.1–0.5 Å) barely change the off-diagonal Coulomb terms.
- **High-energy outliers bleed into equilibrium**: frames 702 (δ=−21.5), 748 (δ=−18.1),
  866 (δ=−28.1) are NOT far from equilibrium in Coulomb space, so their large corrections
  pollute q=0 predictions.
- **Result**: KRR cannot localise corrections — 5 imaginary NM modes at equilibrium,
  IR crashes.

### Normal-mode coordinate fix (Fix B)

New script `casscf_nm_delta.py` implements Fix B from the proposed roadmap:

1. Load 29 existing CASSCF(4,4) single-point results
2. Use B3LYP Hessian → mass-weighted NM eigenvectors U (shape 3N×n_vib)
3. Project each geometry: **q**_i = U^T · M^{1/2} · (R_i − R_ref) [sqrt(amu)·Bohr]
4. Train KRR δ(q) with LOO-CV gamma/alpha grid search
5. Save `NMKRRDeltaModel` pickle

**Key properties of NM descriptors**:
- q = 0 exactly at the reference geometry → kernel K(q_i, q_j) = exp(−γ||q_i−q_j||²)
  gives K = 1 at equilibrium, decaying to 0 for distorted frames
- High-energy outliers (||q||² >> 1) do not contaminate equilibrium predictions
- Physically orthogonal modes — no clustering artifact

### NMKRRDeltaModel class (casscf_nm_delta.py)
- `.project(coords_ang)` → (n_vib,) q vector
- `.predict(symbols, coords_ang)` → delta in Hartree (matches MLPESTrainer API)
- `.save()` / `.load()` pickle I/O
- Pre-computes KRR dual coefficients at init (solves (K+αI)α_vec = y)

### KRR hyperparameters
- gamma units: 1/(amu·Bohr²); typical range 0.01–5.0 for MVKO NM coordinates
- At 300 K, typical thermal ||q||² ~ 1–10 amu·Bohr² for low-freq modes
- LOO-CV via hat-matrix shortcut (exact, fast)
- Grid search output summarised in `summary.json`

### ir_md_spectrum.py integration
Added `NMDeltaDriver` class and `--nm-delta-model` flag:
```bash
python3 ir_md_spectrum.py \
    --model outputs/mvko_20260319_081314/mlpes_initial.pkl \
    --training-data outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \
    --nm-delta-model outputs/casscf_nm_delta_<ts>/nm_delta_model.pkl \
    --steps 30000 --temp 300 --preminimize --zpe-min-freq 50 --zpe-max-freq 4000
```
`--nm-delta-model` takes precedence over `--delta-model` and `--energy-delta`.

### Fix A option (--add-nm-points)
Generates N additional SS-CASSCF points at ±1 thermal amplitude NM displacements
from equilibrium (T=300 K). Provides near-equilibrium anchoring to reduce σ(δ) from
±1.0 to expected ±0.3 kcal/mol. Requires PSI4. Recommend 10–15 points.

### Usage command
```bash
# Step 1: build NM delta model (compute B3LYP Hessian once, reuse with --b3lyp-hessian)
python3 casscf_nm_delta.py \
    --load-results outputs/casscf_surface_20260331_133413/surface_results.json \
    --training-data outputs/mvko_20260319_081314/combined_training_data.npz \
    --eq-coords outputs/mvko_20260319_081314/psi4_eq_coords.npy \
    --max-energy 50 --gamma-values 0.01,0.05,0.1,0.5,1.0,5.0

# Step 2: rerun with saved hessian + Fix A
python3 casscf_nm_delta.py \
    --load-results outputs/casscf_nm_delta_<ts>/all_casscf_results.json \
    --training-data outputs/mvko_20260319_081314/combined_training_data.npz \
    --eq-coords outputs/mvko_20260319_081314/psi4_eq_coords.npy \
    --b3lyp-hessian outputs/casscf_nm_delta_<ts>/hessian_used.npy \
    --add-nm-points 15 --T-nm 300

# Step 3: IR spectrum with NM delta correction
python3 ir_md_spectrum.py \
    --model outputs/mvko_20260319_081314/mlpes_initial.pkl \
    --training-data outputs/mvko_dipoles_20260319_171335/training_with_dipoles.npz \
    --nm-delta-model outputs/casscf_nm_delta_<ts>/nm_delta_model.pkl \
    --steps 30000 --temp 300 --preminimize --zpe-min-freq 50 --zpe-max-freq 4000
```

## 2026-04-01: NM-Coordinate Delta-ML — Diagnosis and Conclusion

### Background
Following the failure of Coulomb-descriptor KRR and 1D energy-spline delta-ML
(documented 2026-03-31), we implemented Fix B: replacing Coulomb matrix descriptors
with mass-weighted normal-mode (NM) coordinates for the CASSCF−B3LYP delta-ML model.
Script: `casscf_nm_delta.py`.  IR driver integration: `NMDeltaDriver` + `--nm-delta-model`
flag in `ir_md_spectrum.py`.

### NM coordinate projection
```
q_i = U_vib^T · M^{1/2} · (R_i − R_ref)    [sqrt(amu)·Bohr]
```
- `q = 0` at the reference geometry; `||q||²` grows with distortion
- RBF kernel K(q_i, q_j) = exp(−γ||q_i − q_j||²) localises correctly:
  large-distortion CASSCF outliers are far from equilibrium in q-space
- B3LYP Hessian at the equilibrium used for NM modes (125–3323 cm⁻¹, all physical)
- Saved: `outputs/casscf_nm_delta_20260401_103139/hessian_used.npy`

### Bug fixed: LOO-CV hat-matrix shortcut is degenerate at small α
The original LOO-CV formula  e_i = (ŷ_i − y_i)/(1 − h_{ii})  gives 0/0 when
α → 0 (near-exact interpolant, h_{ii} → 1).  NumPy returned 0.000 instead of NaN,
falsely reporting perfect cross-validation.  Fixed: replaced with explicit retraining
of M leave-one-out models (feasible since M ≤ 50).

### Bug fixed: Fix A coordinates not stored in result dicts
`add_nm_casscf_points()` never saved `_coords` in the result dict, so
`build_training_arrays()` fell back to `np.zeros((12,3))` for every synthetic frame.
The projection of the zero origin gave a constant ||q||² = 870 amu·Bohr² for all 30
Fix A frames regardless of which NM mode was displaced.  Fix: added
`result['_coords'] = new_coords.tolist()` in the result dict; added a `RuntimeError`
fallback to prevent silent corruption.  Fallback reconstruction from saved
`nm_mode`/`nm_sign`/`nm_amplitude_bohr_sqamu` metadata was used to refit without
re-running PSI4.

### Fix A results — mode-by-mode delta at 300 K thermal amplitude

All 30 Fix A frames have dE_B3LYP = kT = 0.596 kcal/mol by construction (classical
thermal amplitude Q = sqrt(2kT/λ)).  dE_CASSCF varies from 0 to 31 kcal/mol:

| Mode | Freq (cm⁻¹) | dE_CASSCF (kcal/mol) | delta (kcal/mol) |
|------|------------|----------------------|-----------------|
| 0    | 125        | 31.1                 | **+31.1** |
| 1    | 168        | 1.6                  | +1.6 |
| 2    | 281        | 13.8                 | **+13.8** |
| 3−   | 306        | 17.1                 | **+17.1** |
| 4−   | 339        | 11.6                 | **+11.6** |
| 5    | 381        | 1.5                  | +1.5 |
| 7    | 636        | 17.6                 | **+17.6** |
| 8    | 711        | 0.9                  | +0.9 |
| 9−   | 821        | 17.8                 | **+17.8** |
| 14+  | 1061       | 0.0                  | 0.0 ← CASSCF minimum |

About half the modes give delta ≈ 1 kcal/mol (normal); the other half give
delta = 11–31 kcal/mol (pathological).  The CASSCF(4,4) minimum is found at the
mode-14+ displaced geometry — 17 kcal/mol below the B3LYP equilibrium on the
CASSCF energy scale.

### Root cause: wrong active space for equilibrium correction

The CASSCF(4,4) active space {σ/σ*(C-H), σ/σ*(O-H)} was chosen for the IRC
**transition state** (H-transfer from C3 to O2).  At the equilibrium geometry,
these orbitals are nearly doubly occupied (NO occs: 1.998, 1.924, 0.077, 0.000)
and do not describe the dominant near-equilibrium correlation.  CASSCF(4,4) then
finds a lower minimum at a geometry where the active orbitals can reorganise
(mode-14+ shows NO4 = 0.059 vs 0.000 at B3LYP eq).  The B3LYP equilibrium lies
on a hillside of the CASSCF(4,4) surface — not at a minimum.

Evidence:
- CASSCF energy at B3LYP eq is 17 kcal/mol above the CASSCF minimum
- NO occupations at mode-14+ (CASSCF min): (1.933, 1.896, 0.112, 0.059) — more biradical
  than B3LYP eq — consistent with CASSCF finding a different electronic character
- LOO-CV RMSE = 12.3 kcal/mol overall, 8.3 kcal/mol near-equilibrium (both >> kT = 0.59)
- No hyperparameter combination achieves RMSE < 8 kcal/mol on the 48-frame dataset

### Conclusion: CASSCF(4,4) delta-ML not viable for equilibrium IR spectra

The delta-ML approach requires that the correction δ(R) = E_CASSCF(R) − E_B3LYP(R)
is smooth and small near the B3LYP equilibrium.  This fails here because:
1. The CASSCF(4,4)/6-31G* surface has a different minimum than B3LYP/6-31G*
2. The correction is bimodal: ~1 kcal/mol along bond-stretching modes, but
   11–31 kcal/mol along conformational/torsional modes where the active space
   reorganises
3. Any KRR model trained on these data produces LOO-CV RMSE >> kT, making it
   useless for a 300 K IR spectrum

**The B3LYP ML-PES IR spectrum remains the best available result** for MVKO.

### What still works
- CASSCF(4,4) IRC correction for the syn-MVKO → VHP barrier is valid
  (active space was designed for that reaction coordinate; barrier +6.2 kcal/mol)
- `casscf_nm_delta.py` code and `NMDeltaDriver` are correct — the NM framework
  successfully diagnosed the root cause; failures are physical, not numerical
- LOO-CV is now correctly computed by explicit retraining (no hat-matrix shortcut)
- `_coords` bug is fixed; future Fix A runs will work correctly

### Paths forward if equilibrium CASSCF correction is needed
1. **CASSCF geometry optimisation** (`--casscf-opt` flag): find the true CASSCF(4,4)
   minimum and check whether it is physically reasonable; if the CASSCF surface has
   a genuine minimum near the B3LYP eq, this resolves the reference point mismatch
2. **Different active space for equilibrium**: CASSCF(2,2) on HOMO/LUMO of the
   Criegee π-system (C=O-O zwitterion character) is more appropriate near equilibrium;
   the (4,4) IRC active space is wrong here
3. **NEVPT2/CASPT2 single points**: compute dynamic correlation corrections on a
   Boltzmann-weighted sample of B3LYP trajectory frames (no MD, thermal averaging);
   more expensive per point but avoids the KRR generalisation problem
4. **Range-separated DFT**: ωB97X-D or M06-2X instead of B3LYP for the ML-PES;
   these functionals handle the zwitterionic/biradical character of Criegee
   intermediates more reliably, eliminating the need for a multireference correction

### Files produced
```
outputs/casscf_nm_delta_20260401_103139/
  b3lyp_hessian.npy          ← B3LYP/6-31G* Hessian (36×36, reusable)
  hessian_used.npy            ← same
  nm_delta_model.pkl          ← 18-frame model (no Fix A, LOO bug present)
  nm_descriptors.npy, delta_ha.npy
  diagnostics.png, summary.json

outputs/casscf_nm_delta_20260401_110049/
  all_casscf_results.json     ← 59 frames (29 original + 30 Fix A)
  nm_delta_model_fixed.pkl    ← 48-frame model with correct geometry injection
  diagnostics.png             ← shows bimodal delta distribution
```

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

