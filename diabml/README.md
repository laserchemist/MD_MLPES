# diabml

**Physics-framed learned matrices for coupled potential energy surfaces.**

Energized molecules spend their time between "bonded" and "broken", where states cross, ionic and covalent
structures mix and spin couplings change. A single learned energy surface handles this badly. Following
Shu, Varga, Zhang, Meng, Parameswaran, Zhou & Truhlar (*J. Chem. Theory Comput.* **21**, 3342, 2025), `diabml` learns a
small **matrix** whose eigenvalues are the adiabatic energies (CDNN). It adds one idea: build that matrix
from **physics frames** (fragment states, valence-bond structures, charged and excited fragments) and let networks
learn only corrections that keep the frame's symmetry and dissociation limits.

```
refs/       reference engines (PySCF): FCI, SA-CASSCF scans with orbital continuity and state-character diagnostics,
            ladder fragments (IE, EA) for ionic structures
refset.py   RefSet: one .npz per data set (coordinates, multistate energies, diagnostics, metadata)
frames/     LondonFrame (H3, 2x2 VB), DIMWaterFrame (H2O, covalent diatomics-in-molecules), ChannelFrame (A + B
            dissociation channels with covalent, biradical and ionic structures; CH2OO)
models.py   CDNN (the paper's model), IndependentNN (negative control), FramedModel (frame + learned slots), KRR,
            and one shared training loop (Adam, cosine schedule, restarts, validation selection)
metrics.py  per-state errors, seam gaps, permutation errors
plotting.py standard figures;   ui.py  optional ipywidgets cut explorer
notebooks/  01 H3 (London frame vs CDNN), 02 H2O (DIM frame), 03 CH2OO (fragment-channel frame, first look)
workflow/   config-driven CH2OO workflow: run folders, stages, figures, tables, report
dynamics/   Cartesian surfaces (forces and couplings of every state by autograd) and an ASE calculator
examples/   ch2oo_nve.py: constant-energy test trajectories on a trained model
tests/      physics guarantees of every frame (degeneracies, permutation invariance, exact limits) and plumbing
data/       cached reference data, so notebooks need no quantum chemistry
```

## Install and run

```bash
pip install numpy scipy matplotlib torch jupyter ipywidgets   # pyscf only to generate new references
python -m pytest -q tests                                     # 29 tests, about a minute
cd notebooks && jupyter lab                                   # or: python build_notebooks.py --execute
```

## CH2OO workflow (run folders and config files)

`diabml.workflow.ch2oo` runs the CH2OO study end to end from one `config.toml` per run folder:
reference scans (SA-CASSCF, optional SC-NEVPT2; resumable), ladder anchors (delta-CCSD(T) IE and EA), ab initio
derivative couplings along r_OO (SA-CASSCF, resumable), training (framed CDNN vs CDNN: random split, extrapolation,
learning curve, all-state charge restraint, fitting the couplings), figures, tables and an HTML report.

```bash
python -m diabml.workflow.ch2oo init runs/00_quick --template quick     # templates: quick baseline nevpt2_2d cas_3d restraint active explore
python -m diabml.workflow.ch2oo all  runs/00_quick                      # refs, ladder, train, figures, tables, report
python -m diabml.workflow.ch2oo status runs/02_nevpt2_2d                # progress of a running scan
```

Each run folder gets `data/ models/ results/ figures/ tables/ logs/` and `report.html`. Stages can be rerun one at a
time; the scan writes a checkpoint after every geometry and continues where it stopped. See `RUNNING_CH2OO.html` for
setup on macOS (Apple silicon), the recommended sequence of runs, and what every figure and table shows.

### Active learning (`--template active`, `diabml/workflow/active.py`)

One run folder per round. A committee of framed models (bootstrap samples, different seeds) is evaluated on random
candidate geometries; new references are computed where they disagree most, at exactly the level of the base data
(same CASSpec, same energy zero, orbitals started from the nearest computed geometry), merged, and the model retrained.

```bash
python -m diabml.workflow.ch2oo init runs/08_al1 --template active     # refs.file = the refs.npz to extend
python -m diabml.workflow.ch2oo all  runs/08_al1     # refs, ladder, propose, compute, merge, train, figures, tables, report
# next round: a new folder whose refs.file is ../08_al1/data/refs.npz
```

Outputs: `data/proposed.npz` and `results/proposed.csv` (the chosen geometries), `data/new_refs.npz`, the merged
`data/refs.npz` (extra `al_round` records when each point was added; grid figures use round 0 only), `A1` (committee
spread and proposals), `A2` (committee error at the new points before training on them) and table `T9`.

### Production surface (`production` stage, `diabml/workflow/production.py`)

```bash
python -m diabml.workflow.ch2oo production runs/12_al5   # -> models/production.pt, T10, P1, P2
```

Trains `members` framed models (charge restraint + coupling fit) with different seeds, aligns their structure signs
and averages their matrices into one `EnsembleModel`; it loads with `load_model` and drives the ASE calculator like any
model. A fixed held-out test set gives the errors by region (T10): the dynamics region (S0 within `max_S0_eV` of
CH2O + O(1D), outside the low-angle corner), high energy, and the corner; then the members are retrained on all data.

Watched dynamics (`diabml/dynamics/monitor.py`, `report.py`): every step records all state energies, energy
conservation, the ensemble members' spread, the distance to the reference data, gaps and |d.v| to neighbouring
states. A trajectory stops (after a short look-ahead) where the surface becomes uncertain, and every run writes an
HTML log (`examples/ch2oo_nve.py ... --refs data/refs.npz` -> `<out>.html`). An optional extrapolation guard
(`[frame] guard = true`: networks frozen outside the data plus a repulsive wall) exists but is off by default; the
data should come from active learning from dynamics instead.

### Reference audit with ORCA (`audit` stage, `diabml/workflow/audit.py`)

```bash
python -m diabml.workflow.ch2oo audit runs/19_md6 --orca /path/to/orca_folder --jobs 6   # -> runs/19_md6/audit.html
```

Recomputes every reference geometry with ORCA at the same level (SA-CASSCF, plus SC-NEVPT2 and QD-NEVPT2), in
parallel, and classifies each point: same SA-CASSCF solution, ours on a higher solution, ORCA on a higher solution
(retried from a good neighbour's orbitals) or failed. Results: results/audit.npz and audit.csv, and audit.html.

### Active learning from dynamics (`--template explore`, `diabml/workflow/explore.py`)

```bash
python -m diabml.workflow.ch2oo init runs/14_md1 --template explore     # refs.file = the refs.npz to extend
python -m diabml.workflow.ch2oo all  runs/14_md1     # refs, ladder, explore, compute, merge, train, ... + dynamics.html
```

A committee runs watched trajectories from the starts in `[explore]`; the geometries where each trajectory found the
surface uncertain are computed at the level of the base data and merged. Repeat (refs.file = previous round) until
every trajectory completes.

## Models on disk and dynamics

Trained models are saved as `name.pt` (a `state_dict` of weights plus a plain-data `spec` that rebuilds the model:
class, layer sizes, scalers, the frame's structures, asymptotes and walls) and `name.json` (the same spec, readable).
`diabml.models.load_model(path)` rebuilds a model with PyTorch's safe loader (no pickled code).

```python
from diabml.models import load_model
from diabml.dynamics import CoordinateSurface, DiabmlCalculator, ch2oo_atoms, ch2oo_coordinates, reduced_restraints
model = load_model('runs/01_baseline/models/random_framed.pt')
atoms = ch2oo_atoms()
names = model.meta['coord_names']
atoms.calc = DiabmlCalculator(reduced_restraints(CoordinateSurface(model, ch2oo_coordinates(names)),
                                                 atoms.positions, free=names), state=0)
atoms.get_forces(); atoms.calc.results['energies'], atoms.calc.results['couplings']
```

`DistanceSurface` (all interatomic distances, optional permutation averaging) is the entry point for a
full-dimensional model. `python examples/ch2oo_nve.py <model.pt>` runs a constant-energy test trajectory.

## The one interface to know: `Frame`

A frame returns a physics matrix Ω(geometry) and declares **slots**, the named functions a network may learn.

```python
class MyFrame(Frame):
    n_diabats = 3
    slots = {'coupling': (n_inputs, n_outputs)}          # learned functions, evaluated k times per geometry
    def prepare(self, coords):                           # geometry-dependent, parameter-free tensors
        return {..., 'slot_in': {'coupling': x}}         # x: (N, k, n_inputs)
    def assemble(self, cache, out):                      # out['coupling']: (N, k, n_outputs)
        return Omega                                     # (N, 3, 3), differentiable in out
```

`FramedModel(frame, n_states)` attaches a small MLP to every slot with a zero-initialised last layer, so an
untrained model *is* the physics prior. Training changes only what physics misses. Guarantees that hold for any slot
values (exact degeneracies, permutation invariance, dissociation limits) are tested in `tests/`.

## Adding a system

1. Write a reference generator in `refs/` that saves a `RefSet` (see `refs/ch2oo.py`, which runs SA-CASSCF scans
   carrying orbitals from the nearest finished geometry).
2. Choose or write a frame: for a single dissociation channel, `ChannelFrame` with one `Diabat` per structure
   (asymptote from fragment calculations, an optional Coulomb tail for ion pairs).
3. Train `FramedModel` and `CDNN` side by side with `models.train`, and compare with `metrics`.

## Reference data shipped in `data/`

| file | content |
|---|---|
| `h2_curves.npz`, `h3_cut.npz`, `h3_global.npz` | H2 singlet/triplet and two lowest H3 doublets, FCI/cc-pVDZ |
| `oh_curves.npz`, `h2_631g.npz`, `h2o.npz` | eight OH states, H2, and 149 H2O geometries, frozen-core FCI/6-31G |
| `ch2oo_casscf.npz` | CH2OO, SA4-CASSCF(8,6)/6-31G*, four singlet A' states, 120 planar geometries (r_OO, angle COO), with per-state terminal-O charge and effective unpaired electrons |

The CH2OO data come from CASSCF without dynamic correlation and a rigid CH2O fragment: a methods testbed, not
quantitative energetics (the O–O bond is underbound relative to accurate theory and experiment).

## Key references

- Y. Shu, Z. Varga, D. Zhang, Q. Meng, A. M. Parameswaran, J.-G. Zhou, D. G. Truhlar, *J. Chem. Theory Comput.* **21**, 3342 (2025): CDNN.
- Y. Shu, D. G. Truhlar, *J. Chem. Theory Comput.* **16**, 6456 (2020): diabatization by deep neural network (DDNN).
- C. A. Mead, D. G. Truhlar, *J. Chem. Phys.* **77**, 6090 (1982): strictly diabatic bases do not exist in general.
- F. O. Ellison, *J. Am. Chem. Soc.* **85**, 3540 (1963): diatomics in molecules.
- Z. Yin, B. J. Braams, B. Fu, D. H. Zhang, *J. Chem. Theory Comput.* **17**, 1678 (2021): VB-based symmetry-adapted NN diabatic Hamiltonians (H3+).
- K. Samanta, J. M. Beames, M. I. Lester, J. E. Subotnik, *J. Chem. Phys.* **141**, 134303 (2014): CH2OO O–O photodissociation, quasi-diabatic states by charge separation.
