"""Builds the three example notebooks (and optionally executes them).

    python build_notebooks.py            # write .ipynb files
    python build_notebooks.py --execute  # write and execute in place
"""
import sys
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
HERE = Path(__file__).resolve().parent

SETUP = r"""import sys, time, numpy as np, torch, matplotlib.pyplot as plt
sys.path.insert(0, '..')                      # the diabml package lives one level up
from diabml import RefSet
from diabml.units import EV_KCAL
from diabml import plotting as P, metrics as M
from diabml.models import CDNN, IndependentNN, FramedModel, KRR, train
P.style()
torch.set_num_threads(2)
DATA = '../data/'"""

# ===================================================================================================== 01 H3
h3 = [md(r"""# 01 · H₃: a learned matrix with and without a physics frame

Two coupled states of H₃ (FCI/cc-pVDZ). This notebook reproduces, through the `diabml` backend, the comparison of
**CDNN** (Shu, Varga, Zhang, Meng, Parameswaran, Zhou & Truhlar, *JCTC* **21**, 3342, 2025) with a **London-framed**
model, in which the network learns only corrections to the 2 × 2 valence-bond matrix built from exact H₂ curves.

| Model | What builds the matrix |
|---|---|
| `physics prior` | London matrix from H₂ singlet/triplet curves, no H₃ data |
| `CDNN` | 256-128-64 GELU network of (r₁₂, r₁₃, r₂₃), as in the paper |
| `CDNN + permutations` | CDNN trained on all six atom permutations of every geometry |
| `independent NN` | the same network predicting the two energies directly (no matrix) |
| `framed CDNN` | London matrix + learned Sato function and three-body corrections (exact symmetry and limits) |

Change the control cell and re-run. Defaults keep the run to roughly 10 minutes on a laptop."""),
code(r"""# ---------------- controls ----------------
MODELS     = ['CDNN', 'CDNN + permutations', 'independent NN', 'framed CDNN']
EPOCHS     = 3000          # Adam steps per restart (the paper trained to convergence; 3000 is enough here)
RESTARTS   = 2             # random restarts per model, best kept on validation data (paper: 300)
N_TRAIN    = [25, 100]     # training-set sizes for the global-surface learning curve
SPLITS     = 1             # random splits per size
SEED       = 0"""),
code(SETUP + r"""
from diabml.frames import LondonFrame
from diabml.geometry import H3_PERMUTATIONS
h2, cut, glob = RefSet.load(DATA + 'h2_curves.npz'), RefSet.load(DATA + 'h3_cut.npz'), RefSet.load(DATA + 'h3_global.npz')
frame = LondonFrame(h2)
print(cut, glob, sep='\n')"""),
md(r"""## Models

`make(name)` returns a fresh model and the keyword arguments its training needs. The permutation variant augments
training data; the framed model gets a larger learning rate because it starts from the physics and only needs small corrections."""),
code(r"""def permute(coords, energies):
    return np.concatenate([coords[:, p] for p in H3_PERMUTATIONS]), np.concatenate([energies] * 6)

def make(name):
    if name == 'CDNN':                return CDNN(3, 2), {}
    if name == 'CDNN + permutations': return CDNN(3, 2), {'augment': permute}
    if name == 'independent NN':      return IndependentNN(3, 2), {}
    if name == 'framed CDNN':         return FramedModel(frame, 2), {'lr': 3e-3}
    raise ValueError(name)

def fit(name, data, tr, va):
    m, kw = make(name)
    kw = {'lr': 1e-3, **kw}
    return train(m, data.coords[tr], data.energies[tr], data.coords[va], data.energies[va], epochs=EPOCHS,
                 restarts=RESTARTS, **kw)"""),
md(r"""## Part A · The paper's cut through the conical intersection

r₂₃ = 1.08 Å; r₁₂, r₁₃ ∈ [0.88, 1.40] Å. The states are degenerate at r₁₂ = r₁₃ = 1.08 Å. Random 60/20/20 split."""),
code(r"""tr, va, te = cut.split(265, 88, seed=SEED)
A, t0 = {}, time.time()
for name in MODELS:
    A[name] = fit(name, cut, tr, va)
    print(f'{name} trained ({time.time() - t0:.0f} s)')
predictors = {'physics prior': frame.prior_energies, **{k: m.predict for k, m in A.items()}}
rows = {}
for k, f in predictors.items():
    mue = M.mue_mev(f(cut.coords[te]), cut.energies[te])
    # on the 2D cut only the r12 <-> r13 swap stays inside the slice the models were trained on
    rows[k] = [mue[0], mue[1], M.gap_mev(f, [[1.08, 1.08, 1.08]]), M.permutation_error_mev(f, cut.coords, [[0, 1, 2], [1, 0, 2]])]
print(M.summary_table(rows, ['MUE S0 meV', 'MUE S1 meV', 'gap@CI meV', 'swap meV']))"""),
code(r"""d = np.linspace(-0.26, 0.26, 121)
C = np.c_[1.08 + d, 1.08 - d, np.full_like(d, 1.08)]
near = np.abs(cut.coords[:, 0] + cut.coords[:, 1] - 2.16) < 1e-6
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
P.cut_plot(d, cut.energies[near], {k: f(C) for k, f in predictors.items()}, 'r₁₂ − 1.08 Å   (r₁₃ = 2.16 Å − r₁₂)',
           title='Across the seam', ax=axes[0], ref_x=cut.coords[near, 0] - 1.08)
for k, f in predictors.items():
    E = f(C)
    axes[1].semilogy(d, np.maximum((E[:, 1] - E[:, 0]) * 1000, 1e-3), color=P.color(k), label=k)
axes[1].set(xlim=(-0.05, 0.05), xlabel='r₁₂ − 1.08 Å', ylabel='E₁ − E₀ (meV)', title='Gap near the intersection')
axes[1].legend(fontsize=8); plt.tight_layout(); plt.show()"""),
md(r"""### What the matrices look like

A conical intersection needs Ω₁₁ = Ω₂₂ **and** Ω₁₂ = 0 at once: the two zero lines must cross at the star. The framed
matrix does this by symmetry; CDNN has to learn it."""),
code(r"""g = np.linspace(0.88, 1.40, 61); G1, G2 = np.meshgrid(g, g, indexing='ij')
Rg = np.c_[G1.ravel(), G2.ravel(), np.full(G1.size, 1.08)]
for k in [m for m in ('CDNN', 'framed CDNN') if m in A]:
    Mx = A[k].matrix(Rg).detach().numpy().reshape(*G1.shape, 2, 2)
    fig = P.matrix_maps(G1, G2, Mx, 'r₁₂ (Å)', 'r₁₃ (Å)', marker=(1.08, 1.08)); fig.suptitle(k); plt.show()"""),
md(r"""## Part B · The global surface and its symmetry

Same geometry window as the earlier exploratory notebooks (all H–H ≥ 0.6 Å, ground state within 5.4 eV of 3 H).
Reported: test RMSE per state, the mean gap at nine equilateral geometries (exactly 0 in FCI), and the change on
relabelling identical atoms (exactly 0 physically)."""),
code(r"""W = np.where((glob.coords.min(1) >= 0.6) & (glob.energies[:, 0] <= 5.44))[0]
bend = glob.extra['bend_geometry']
seam = glob.coords[(bend[:, 2] == 60) & (bend[:, 0] == bend[:, 1])]
B = {k: {n: [] for n in N_TRAIN} for k in MODELS}
t0 = time.time()
for s in range(SPLITS):
    p = np.random.default_rng(10 + s).permutation(W); va, pool = p[:60], p[60:]
    for n in N_TRAIN:
        tr, te = pool[:n], pool[n:]
        for k in MODELS:
            m = fit(k, glob, tr, va)
            B[k][n].append(list(M.rmse_kcal(m.predict(glob.coords[te]), glob.energies[te]))
                           + [M.gap_mev(m, seam), M.permutation_error_mev(m, glob.coords[te], H3_PERMUTATIONS)])
        print(f'split {s}, n = {n} done ({time.time() - t0:.0f} s)')
prior = M.rmse_kcal(frame.prior_energies(glob.coords[te]), glob.energies[te])
rows = {f'{k}, n={n}': list(np.mean(B[k][n], 0)) for k in MODELS for n in N_TRAIN}
rows['physics prior (0 H3 points)'] = list(prior) + [M.gap_mev(frame.prior_energies, seam), 0.0]
print(M.summary_table(rows, ['S0 kcal', 'S1 kcal', 'seam meV', 'swap meV']))"""),
code(r"""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for s, ax in enumerate(axes):
    P.learning_curves(N_TRAIN, {k: [[x[s] for x in B[k][n]] for n in N_TRAIN] for k in MODELS}, baseline=prior[s],
                      title=['ground state S₀', 'excited state S₁'][s], ax=ax)
plt.tight_layout(); plt.show()"""),
md(r"""## Explore

Move the third distance and watch every model along a cut through r₁₂ = r₁₃ (needs `ipywidgets`; otherwise a static plot)."""),
code(r"""from diabml.ui import cut_explorer
cut_explorer(lambda x, r23: np.c_[x, x, np.full_like(x, r23)],
             {'physics prior': frame.prior_energies, **{k: m for k, m in A.items() if k != 'independent NN'}},
             x_range=(0.8, 2.5), fixed={'r23': (1.08, 0.8, 2.5)}, xlabel='r₁₂ = r₁₃ (Å)')"""),
]

# ===================================================================================================== 02 H2O
h2o = [md(r"""# 02 · H₂O: a fragment frame with a heavy atom

The covalent diatomics-in-molecules (DIM) frame builds water from eight OH states and the H₂ singlet/triplet,
with **no H₂O data**. It gets the limits exactly right and fails at equilibrium, because ionic O⁻H⁺ structures
are missing from the frame. The notebook measures how much learned corrections recover, and where they live:

| Model | Learned part |
|---|---|
| `physics prior` | none (covalent DIM) |
| `framed (mixing)` | OH term-mixing angles (³P/¹D, ¹D/¹S) as functions of r: a fragment-level correction |
| `framed CDNN` | the above plus a dense 9 × 9 correction inside the DIM frame, switched off when either H leaves |
| `framed + KRR` | DIM prior + kernel ridge on the residual |
| `KRR` | kernel ridge on the energy, no physics |

All energies: frozen-core FCI/6-31G, eV relative to O(³P) + 2H."""),
code(r"""# ---------------- controls ----------------
MODELS   = ['framed (mixing)', 'framed CDNN', 'framed + KRR', 'KRR']
N_TRAIN  = [10, 40, 80]
SPLITS   = 2
EPOCHS   = 2000
RESTARTS = 2"""),
code(SETUP + r"""
from diabml.frames.dim_oh import DIMWaterFrame, bond_additivity, h2o_invariants
oh, h2, w = RefSet.load(DATA + 'oh_curves.npz'), RefSet.load(DATA + 'h2_631g.npz'), RefSet.load(DATA + 'h2o.npz')
frame = DIMWaterFrame(oh, h2)
frame_g = DIMWaterFrame(oh, h2, generic=True)
t0 = time.time(); E_dim = frame.prior_energies(w.coords)[:, 0]; print(f'DIM prior for {len(w)} geometries: {time.time() - t0:.0f} s')
E_add = bond_additivity(w.coords, frame)
for k, E in [('covalent DIM', E_dim), ('bond additivity', E_add)]:
    print(f'{k:16s} RMSE {M.rmse_kcal(E[:, None], w.energies)[0]:.1f} kcal/mol')"""),
code(r"""tag = w.extra['tag']
def scan(mask, col):
    i = np.where(mask)[0]; return i[np.argsort(w.coords[i, col])]
scans = [(scan((tag == 'grid') & (w.coords[:, 0] == w.coords[:, 1]), 0), 0, 'r₁ = r₂ (Å)', 'Symmetric stretch'),
         (scan((tag == 'grid') & (w.coords[:, 0] == 0.96), 1), 1, 'r₂ (Å), r₁ = 0.96 Å', 'H₂O → H + OH'),
         (scan(tag == 'bend096', 2), 2, 'angle (°)', 'Bend, r = 0.96 Å')]
def show(preds, title=''):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (i, col, xl, t) in zip(axes, scans):
        P.cut_plot(w.coords[i, col], w.energies[i], {k: E[i][:, None] if E.ndim == 1 else E[i] for k, E in preds.items()},
                   xl, 'E − [E(O) + 2E(H)] (eV)', title=f'{t}{title}', ax=ax)
    plt.tight_layout(); plt.show()
show({'physics prior': E_dim})"""),
md(r"""## Learning the rest of water

For each training size, random splits of all 149 geometries: train on *n*, validate on 20, test on the rest."""),
code(r"""FEAT = h2o_invariants
res = {k: {n: [] for n in N_TRAIN} for k in MODELS}
keep = {}
t0 = time.time()
for s in range(SPLITS):
    for n in N_TRAIN:
        tr, va, te = w.split(n, 20, seed=s)
        for k in MODELS:
            if k == 'framed (mixing)':
                m = train(FramedModel(frame, 1, hidden=(16, 16)), w.coords[tr], w.energies[tr], w.coords[va], w.energies[va],
                          epochs=EPOCHS, lr=3e-3, restarts=RESTARTS); pred = m.predict(w.coords)
            elif k == 'framed CDNN':
                m = train(FramedModel(frame_g, 1, hidden=(32, 32)), w.coords[tr], w.energies[tr], w.coords[va], w.energies[va],
                          epochs=EPOCHS, lr=3e-3, restarts=RESTARTS); pred = m.predict(w.coords)
            elif k == 'framed + KRR':
                kr = KRR(FEAT).fit(w.coords[tr], w.energies[tr, 0] - E_dim[tr]); pred = (E_dim + kr.predict(w.coords))[:, None]
            else:
                kr = KRR(FEAT).fit(w.coords[tr], w.energies[tr, 0]); pred = kr.predict(w.coords)[:, None]
            res[k][n].append(M.rmse_kcal(pred[te], w.energies[te])[0])
            if s == 0 and n == N_TRAIN[-1]:
                keep[k] = pred
    print(f'split {s} done ({time.time() - t0:.0f} s)')
for k in MODELS:
    print(f'{k:16s}', '  '.join(f'n={n}: {np.mean(res[k][n]):5.1f}' for n in N_TRAIN))
P.learning_curves(N_TRAIN, {k: [res[k][n] for n in N_TRAIN] for k in MODELS},
                  baseline=M.rmse_kcal(E_dim[:, None], w.energies)[0], title='H₂O test RMSE'); plt.show()"""),
code(r"""show({'physics prior': E_dim, **keep}, f', n = {N_TRAIN[-1]}')"""),
md(r"""**Reading the results.** The frame gets the H + OH channel right with no water data, and mixing angles alone barely
help, because the missing physics (ionic O⁻H⁺ structures) is outside the covalent space. Corrections that act on the
whole matrix or the energy recover much more, and starting from the DIM prior beats learning from scratch.
The natural next step in the backend is an **ionic DIM frame**: add O⁻ + H⁺ structures, with OH⁻ and H₂⁺ fragment
curves supplying their energies (the electron ladder in fragment form)."""),
]

# ===================================================================================================== 03 CH2OO
ch2oo = [md(r"""# 03 · CH₂OO: first steps with a fragment-channel frame

The simplest Criegee intermediate, H₂C=O⁺–O⁻ ↔ ·H₂C–O–O·, is the case the whole approach is aimed at. Its ground
state mixes **zwitterionic** and **biradical** character; its O–O bond breaks to CH₂O + O(¹D) on singlet surfaces,
with the biradical channel CH₂O(T₁) + O(³P) and the ion pair CH₂O⁺ + O⁻ above.

**Reference data**: SA4-CASSCF(8,6)/6-31G\*, the four lowest singlet A′ states on a planar grid of r(O–O) and ∠COO, with
the CH₂O fragment held rigid (`diabml.refs.ch2oo`, ~20 min). Each state also carries two diagnostics: the
Mulliken charge on the terminal O (zwitterion character) and the effective number of unpaired electrons
(biradical character).

**The frame** (`diabml.frames.ChannelFrame`): five structures labelled by the fragment states they dissociate to.

| Structure | Asymptote from | Long-range form |
|---|---|---|
| CH₂O(S₀) + O(¹D), three A′ components | the three lowest states at 8 Å | constant |
| CH₂O(T₁) + O(³P), singlet-coupled (biradical) | the fourth state at 8 Å | constant |
| CH₂O⁺ + O⁻ (zwitterion) | **the ladder**: IE(CH₂O) − EA(O) − ΔE(¹D−³P) | −14.40 / r Coulomb |

Every learned part is multiplied by an O–O overlap envelope that decays as the O leaves, and the network sees
overlap features that saturate at long range, so the asymptotes and the Coulomb tail are exact by construction and
the network cannot extrapolate wildly beyond its data (the descriptor lesson of the first exploratory notebook). This is a *minimal* frame: correct structures, limits and long-range physics, with all
short-range shape learned."""),
code(r"""# ---------------- controls ----------------
N_STATES  = 4            # adiabatic states fitted (lowest N of the 5-structure matrix)
EPOCHS    = 6000
RESTARTS  = 2
LADDER    = 'experiment' # 'experiment' (IE 10.88, EA 1.461, 1D-3P 1.967 eV) or 'compute' (delta-CCSD(T)/aug-cc-pVTZ, minutes)
R_TRAIN_MAX = 2.3        # extrapolation test: train only on r(O-O) <= this (Angstrom)"""),
code(SETUP + r"""
from diabml.frames import ChannelFrame, Diabat
ref = RefSet.load(DATA + 'ch2oo_casscf.npz')
print(ref); print(ref.meta['method'])
print('all converged:', bool(ref.extra['converged'].all()), '| active space consistent:', bool(ref.extra['active_space_consistent'].all()))
r, a = ref.coords[:, 0], ref.coords[:, 1]"""),
md(r"""## What the states are

Along the O–O stretch at the equilibrium angle: energies, charge separation and open-shell character of each state."""),
code(r"""eq = np.where(np.isclose(a, 117.7))[0]; eq = eq[np.argsort(r[eq])]
fig, ax = plt.subplots(figsize=(7, 4))
for s in range(ref.n_states):
    ax.plot(r[eq], ref.energies[eq, s], marker='o', ms=3, label=f'S{s}')
ax.set(xlabel='r(O–O) (Å)', ylabel='E (eV) rel. CH₂O + O(¹D)', title='SA4-CASSCF singlet A′ states, ∠COO = 117.7°', xlim=(1.1, 5.2))
ax.legend(); plt.show()
P.character_plot(r[eq], ref.extra['q_O_terminal'][eq], ref.extra['n_unpaired'][eq]); plt.show()"""),
md(r"""## The ladder: anchoring the ionic structure

The zwitterion H₂C=O⁺–O⁻ dissociates diabatically to CH₂O⁺ + O⁻. Its asymptote, relative to CH₂O + O(¹D), needs
exactly the charge- and spin-state data the electron ladder is about: the ionisation energy of the fragment, the
electron affinity of O, and the O(¹D)–O(³P) splitting. Where it crosses the covalent structures is then fixed by
Coulomb's law, r* ≈ 14.40 eV·Å / ΔE."""),
code(r"""if LADDER == 'compute':
    from diabml.refs.ladder import compute
    lad = compute(); dE_ion = lad['dE_ion']; print(lad)
else:
    from diabml.refs.ladder import EXPERIMENT as X
    dE_ion = X['IE_CH2O_adiabatic'] - X['EA_O'] - X['O_1D_3P']
far = np.where(r == r.max())[0]
asym = ref.energies[far].mean(0)
print(f'asymptotes at 8 Å (eV): {np.round(asym, 3)}  |  ion pair: {dE_ion:.2f} eV  ->  Coulomb crossing near {14.40 / dE_ion:.2f} Å')
diabats = [Diabat('CH₂O + O(¹D) a', asym[0]), Diabat('CH₂O + O(¹D) b', asym[1]), Diabat('CH₂O + O(¹D) c', asym[2]),
           Diabat('CH₂O(T₁) + O(³P)', asym[3], kind='biradical'),
           Diabat('CH₂O⁺ + O⁻', asym[:3].mean() + dE_ion, charge_product=-1.0, kind='ionic')]
def new_frame():
    return ChannelFrame(diabats, n_coords=2, r_index=0)"""),
md(r"""## Test 1 · Interpolation (random split)"""),
code(r"""tr, va, te = ref.split(70, 15, seed=0)
t0 = time.time()
fr = new_frame()
framed = train(FramedModel(fr, N_STATES, hidden=(128, 64)), ref.coords[tr], ref.energies[tr], ref.coords[va], ref.energies[va],
               epochs=EPOCHS, lr=2e-3, restarts=RESTARTS)
cdnn = train(CDNN(2, N_STATES, n_matrix=5, hidden=(128, 64)), ref.coords[tr], ref.energies[tr], ref.coords[va], ref.energies[va],
             epochs=EPOCHS, lr=1e-3, restarts=RESTARTS)
print(f'trained in {time.time() - t0:.0f} s')
for k, m in [('framed CDNN', framed), ('CDNN', cdnn)]:
    print(f'{k:12s} test RMSE per state (kcal/mol):', np.round(M.rmse_kcal(m.predict(ref.coords[te]), ref.energies[te]), 2))"""),
md(r"""## Test 2 · Extrapolation to dissociation

Train only on r(O–O) ≤ `R_TRAIN_MAX`, then predict the whole stretch. This is the situation of a data set built from
trajectories that rarely reach full dissociation. The frame knows the asymptotes; plain CDNN has to guess."""),
code(r"""inner = np.where(r <= R_TRAIN_MAX)[0]; outer = np.where(r > R_TRAIN_MAX)[0]
p = np.random.default_rng(1).permutation(inner); tr2, va2 = p[:int(0.8 * len(p))], p[int(0.8 * len(p)):]
fr2 = new_frame()
framed2 = train(FramedModel(fr2, N_STATES, hidden=(128, 64)), ref.coords[tr2], ref.energies[tr2], ref.coords[va2], ref.energies[va2],
                epochs=EPOCHS, lr=2e-3, restarts=RESTARTS)
cdnn2 = train(CDNN(2, N_STATES, n_matrix=5, hidden=(128, 64)), ref.coords[tr2], ref.energies[tr2], ref.coords[va2], ref.energies[va2],
              epochs=EPOCHS, lr=1e-3, restarts=RESTARTS)
for k, m in [('framed CDNN', framed2), ('CDNN', cdnn2)]:
    print(f'{k:12s} RMSE beyond {R_TRAIN_MAX} Å (kcal/mol):', np.round(M.rmse_kcal(m.predict(ref.coords[outer]), ref.energies[outer]), 1))
x = np.linspace(1.15, 8.0, 300); C = np.c_[x, np.full_like(x, 117.7)]
ax = P.cut_plot(x, ref.energies[eq], {'framed CDNN': framed2.predict(C), 'CDNN': cdnn2.predict(C)}, 'r(O–O) (Å)', 'E (eV)',
                title=f'Trained on r ≤ {R_TRAIN_MAX} Å only, ∠COO = 117.7°', ref_x=r[eq]).axes[0]
ax.axvspan(R_TRAIN_MAX, 8.2, color='#eda100', alpha=0.12, lw=0); ax.set_ylim(-3, 6); plt.show()"""),
md(r"""## The learned diabats

The diagonal of the framed matrix along the stretch, coloured by structure type, with the adiabatic states on top.
The ionic (zwitterion) structure plunges at short range under its Coulomb attraction and crosses the covalent ones;
the adiabatic ground state is the avoided crossing between them. These diabats are *learned*, constrained only by
their asymptotes and long-range form, so read them as a hypothesis to test (e.g. against charge-separation
diabatisation, Samanta et al. 2014), not a result."""),
code(r"""Md = framed.matrix(C).detach().numpy()
P.diabat_plot(x, np.diagonal(Md, axis1=1, axis2=2), framed.predict(C), fr.diabat_labels, [d.kind for d in diabats],
              ref_x=r[eq], ref=ref.energies[eq], xlabel='r(O–O) (Å)', ylabel='E (eV)', ylim=(-4, 10)); plt.show()"""),
code(r"""from diabml.ui import cut_explorer
cut_explorer(lambda x, angle: np.c_[x, np.full_like(x, angle)], {'framed CDNN': framed, 'CDNN': cdnn},
             x_range=(1.15, 6.0), fixed={'angle': (117.7, 65.0, 155.0)}, xlabel='r(O–O) (Å)')"""),
md(r"""## Where to go next (all supported by the backend's structure)

- **Diabatic restraints from the diagnostics.** The charge on the terminal O is a natural label for the ionic
  structure; adding a loss term that ties the ionic diabat's population to it moves the model from 'compatible'
  towards 'diabatic' (the DDNN idea of Shu & Truhlar, 2020).
- **More coordinates.** C–O stretch, torsion and the H motions; the frame's switch only needs the O–O distance.
- **More channels.** Isomerisation to dioxirane (small ∠COO) and the triplet manifold are separate frames or more structures.
- **Better references.** XMS-CASPT2 or MRCI on top of the same active space."""),
]

NOTEBOOKS = {'01_H3_london_vs_cdnn.ipynb': h3, '02_H2O_fragment_frame.ipynb': h2o, '03_CH2OO_first_look.ipynb': ch2oo}

if __name__ == '__main__':
    only = sys.argv[2] if len(sys.argv) > 2 else None        # e.g. "03": build (and execute) only that notebook
    for name, cells in NOTEBOOKS.items():
        if only and only not in name:
            continue
        nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
        if '--execute' in sys.argv:
            from nbclient import NotebookClient
            NotebookClient(nb, timeout=7200, kernel_name='python3', resources={'metadata': {'path': str(HERE)}}).execute()
        nbf.write(nb, HERE / name)
        print('wrote', name)
