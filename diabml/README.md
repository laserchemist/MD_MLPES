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
tests/      physics guarantees of every frame (degeneracies, permutation invariance, exact limits) and plumbing
data/       cached reference data, so notebooks need no quantum chemistry
```

## Install and run

```bash
pip install numpy scipy matplotlib torch jupyter ipywidgets   # pyscf only to generate new references
python -m pytest -q tests                                     # 14 tests, under a minute
cd notebooks && jupyter lab                                   # or: python build_notebooks.py --execute
```

## CH2OO workflow (run folders and config files)

`diabml.workflow.ch2oo` runs the CH2OO study end to end from one `config.toml` per run folder:
reference scans (SA-CASSCF, optional SC-NEVPT2; resumable), ladder anchors (delta-CCSD(T) IE and EA), training
(framed CDNN vs CDNN: random split, extrapolation, learning curve, optional charge-based diabatic restraint), figures,
tables and an HTML report.

```bash
python -m diabml.workflow.ch2oo init runs/00_quick --template quick     # templates: quick baseline nevpt2_2d cas_3d restraint
python -m diabml.workflow.ch2oo all  runs/00_quick                      # refs, ladder, train, figures, tables, report
python -m diabml.workflow.ch2oo status runs/02_nevpt2_2d                # progress of a running scan
```

Each run folder gets `data/ models/ results/ figures/ tables/ logs/` and `report.html`. Stages can be rerun one at a
time; the scan writes a checkpoint after every geometry and continues where it stopped. See `RUNNING_CH2OO.html` for
setup on macOS (Apple silicon), the recommended sequence of runs, and what every figure and table shows.

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
