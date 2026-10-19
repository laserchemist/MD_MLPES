"""Large-type figures for the slide deck. Run from the repository root:

    python slides/make_figures.py [schematic|h3|h2o|ch2oo|all]

Schematics are pure math; h3, h2o and ch2oo train models (about 30, 1 and 3 minutes on 2 cores) and save both the
figures and a results JSON so numbers quoted on slides are traceable.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'slides' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / 'data'
torch.set_num_threads(2)

INK, MUTED, GRID, BG = '#172033', '#5b6577', '#e3e1da', '#F7F5EF'
BLUE, ORANGE, GREEN, PINK, GOLD = '#2a78d6', '#eb6834', '#1baf7a', '#e87ba4', '#eda100'


def big():
    plt.rcParams.update({'figure.dpi': 200, 'savefig.dpi': 200, 'font.size': 15, 'axes.titlesize': 16,
                         'axes.labelsize': 15, 'legend.fontsize': 13, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'grid.color': GRID, 'grid.linewidth': 0.7, 'lines.linewidth': 2.6,
                         'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'legend.frameon': False,
                         'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


# ------------------------------------------------------------------------------------------------ schematics
def schematics():
    big()
    r = np.linspace(1.0, 12, 900)
    # three structures: bound covalent (Morse), repulsive covalent, ion pair (Coulomb); couplings decay with r
    Vb = 4.0 * (1 - np.exp(-1.3 * (r - 1.5))) ** 2 - 4.0
    Vr = 2.5 * np.exp(-(r - 1.5) / 0.55)
    Vi = 2.6 - 14.4 / r + 1363.0 * np.exp(-r / 0.30)          # Born-Mayer wall: minimum near 2.2 A
    H = lambda c, l: c * np.exp(-(r - 3.0) / l)
    Hm = np.zeros((len(r), 3, 3))
    Hm[:, 0, 0], Hm[:, 1, 1], Hm[:, 2, 2] = Vb, Vr, Vi
    Hm[:, 0, 2] = Hm[:, 2, 0] = H(0.35, 1.6)
    Hm[:, 1, 2] = Hm[:, 2, 1] = H(0.25, 1.6)
    E = np.linalg.eigvalsh(Hm)
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.axvspan(1.0, 2.4, color=BLUE, alpha=0.07, lw=0)
    ax.axvspan(2.4, 7.5, color=GOLD, alpha=0.16, lw=0)
    ax.axvspan(7.5, 12, color=GREEN, alpha=0.07, lw=0)
    for c, lab in [(Vb, None), (Vr, None), (Vi, None)]:
        ax.plot(r, c, color=MUTED, lw=1.2, ls=':')
    ax.plot(r, E[:, 0], color=INK, lw=3.2, label='ground state')
    ax.plot(r, E[:, 1], color=ORANGE, lw=2.4, label='first excited state')
    ax.plot(r, E[:, 2], color=BLUE, lw=2.0, label='second excited state')
    for x, t in [(1.7, 'bonded'), (4.95, 'the middle zone'), (9.75, 'separated')]:
        ax.text(x, 6.35, t, ha='center', va='top', fontsize=16, color=INK, fontweight='bold')
    for x, t in [(4.95, 'states cross, charge moves, spins recouple'), (9.75, 'known from fragments')]:
        ax.text(x, 5.6, t, ha='center', va='top', fontsize=13.5, color=MUTED)
    ax.text(11.8, 1.85, 'ion pair A⁺ + B⁻', ha='right', fontsize=13, color=MUTED)
    ax.text(11.8, 0.25, 'A + B', ha='right', fontsize=13, color=MUTED)
    ax.set(xlabel='bond length  (Å)', ylabel='energy  (eV)', ylim=(-4.4, 6.5), xlim=(1.0, 12), yticks=[-4, -2, 0, 2, 4])
    ax.legend(loc='lower right')
    save(fig, 'zone.png')

    # diabatic vs adiabatic, zoomed on the crossing
    rr = np.linspace(3.2, 7.0, 600)
    Vc2 = 4.4 * (1 - np.exp(-1.2 * (rr - 1.6))) ** 2 - 4.4             # covalent (Morse tail)
    Vc2 = Vc2 - (4.4 * (1 - np.exp(-1.2 * (40 - 1.6))) ** 2 - 4.4)
    Vi2 = 3.0 - 14.4 / rr                                               # ion pair
    H2 = 0.25 * np.exp(-0.45 * (rr - 4.8))                              # coupling
    m2, d2 = (Vc2 + Vi2) / 2, (Vc2 - Vi2) / 2
    s2 = np.sqrt(d2 * d2 + H2 ** 2)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(rr, Vc2, color=BLUE, label='covalent  A + B')
    axes[0].plot(rr, Vi2, color=ORANGE, label='ionic  A⁺ + B⁻')
    axes[0].plot(rr, H2, color=GREEN, ls=':', label='coupling H₁₂')
    axes[0].set(title='Diabatic: smooth, they simply cross', xlabel='bond length (Å)', ylabel='energy (eV)')
    axes[0].legend(loc='lower right')
    axes[1].plot(rr, m2 - s2, color=INK, label='ground')
    axes[1].plot(rr, m2 + s2, color=INK, ls='--', label='excited')
    # character of the ground state: ionic weight
    th = 0.5 * np.arctan2(2 * H2, Vc2 - Vi2)
    w_ion = np.sin(th) ** 2
    ax2 = axes[1].twinx()
    ax2.plot(rr, 1 - w_ion, color=ORANGE, lw=1.8, alpha=0.8)
    ax2.set_ylabel('ionic weight of ground state', color=ORANGE)
    ax2.tick_params(colors=ORANGE); ax2.grid(False); ax2.spines['right'].set_visible(True)
    ax2.set_ylim(-0.05, 1.05)
    axes[1].set(title='Adiabatic: avoided, character switches', xlabel='bond length (Å)')
    axes[1].legend(loc='lower right')
    fig.tight_layout()
    save(fig, 'diabatic_adiabatic.png')

    # crossing topology: independent smooth surfaces vs eigenvalues of a smooth matrix
    x = np.linspace(-1, 1, 400)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    a = x; b = 0.0 * x
    e = np.sqrt(a ** 2 + b ** 2)
    axes[0].plot(x, -np.sqrt(x ** 2 + 0.06), color=PINK); axes[0].plot(x, np.sqrt(x ** 2 + 0.06), color=PINK)
    axes[0].plot(x, -e, color=INK, ls=':', lw=1.6); axes[0].plot(x, e, color=INK, ls=':', lw=1.6)
    axes[0].set(title='Two smooth functions fitted separately', xlabel='coordinate through the crossing', ylabel='energy', yticks=[], xticks=[])
    axes[0].text(0, -0.55, 'artificial gap', ha='center', color=PINK, fontsize=15)
    axes[0].text(0.62, 0.35, 'exact', color=INK, fontsize=13)
    axes[1].plot(x, -e, color=GREEN); axes[1].plot(x, e, color=GREEN)
    axes[1].set(title='Eigenvalues of a smooth 2 × 2 matrix', xlabel='coordinate through the crossing', yticks=[], xticks=[])
    axes[1].text(0, -0.55, 'exact touching (cone)', ha='center', color=GREEN, fontsize=15)
    fig.tight_layout()
    save(fig, 'topology.png')


# ------------------------------------------------------------------------------------------------ H3
def h3():
    from diabml import RefSet, metrics as M
    from diabml.frames import LondonFrame
    from diabml.geometry import H3_PERMUTATIONS
    from diabml.models import CDNN, FramedModel, train
    big()
    h2, cut, glob = (RefSet.load(DATA / f) for f in ('h2_curves.npz', 'h3_cut.npz', 'h3_global.npz'))
    frame = LondonFrame(h2)
    perm = lambda c, e: (np.concatenate([c[:, p] for p in H3_PERMUTATIONS]), np.concatenate([e] * 6))
    makers = {'CDNN': (lambda: CDNN(3, 2), {'lr': 1e-3}),
              'CDNN + permutations': (lambda: CDNN(3, 2), {'lr': 1e-3, 'augment': perm}),
              'framed CDNN': (lambda: FramedModel(frame, 2), {'lr': 3e-3})}
    res = {'cut': {}, 'global': {}}
    # cut
    tr, va, te = cut.split(265, 88, seed=0)
    models = {}
    for k, (mk, kw) in makers.items():
        models[k] = train(mk(), cut.coords[tr], cut.energies[tr], cut.coords[va], cut.energies[va], epochs=3000, restarts=2, **kw)
    preds = {'London (no H₃ data)': frame.prior_energies, **{k: m.predict for k, m in models.items()}}
    for k, f in preds.items():
        res['cut'][k] = {'mue_meV': list(M.mue_mev(f(cut.coords[te]), cut.energies[te])),
                         'gap_CI_meV': M.gap_mev(f, [[1.08, 1.08, 1.08]]),
                         'swap_meV': M.permutation_error_mev(f, cut.coords, [[0, 1, 2], [1, 0, 2]])}
    d = np.linspace(-0.05, 0.05, 201)
    C = np.c_[1.08 + d, 1.08 - d, np.full_like(d, 1.08)]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    cols = {'London (no H₃ data)': BLUE, 'CDNN': ORANGE, 'CDNN + permutations': GOLD, 'framed CDNN': GREEN}
    for k, f in preds.items():
        E = f(C)
        ax.semilogy(d, np.maximum((E[:, 1] - E[:, 0]) * 1000, 1e-3), color=cols[k], label=k)
    ax.set(xlabel='r₁₂ − 1.08 Å  (across the seam)', ylabel='E₁ − E₀  (meV)', title='Gap near the conical intersection (exact: |x|-shaped cone, 0 at the seam)')
    ax.legend(loc='upper right')
    save(fig, 'h3_gap.png')
    # global learning curves
    W = np.where((glob.coords.min(1) >= 0.6) & (glob.energies[:, 0] <= 5.44))[0]
    bend = glob.extra['bend_geometry']
    seam = glob.coords[(bend[:, 2] == 60) & (bend[:, 0] == bend[:, 1])]
    ns = [25, 50, 100, 200]
    G = {k: {n: [] for n in ns} for k in makers}
    lon = []
    t0 = time.time()
    for s in range(2):
        p = np.random.default_rng(10 + s).permutation(W)
        va, pool = p[:60], p[60:]
        for n in ns:
            tr, te = pool[:n], pool[n:]
            if n == ns[0]:
                lon.append(list(M.rmse_kcal(frame.prior_energies(glob.coords[te]), glob.energies[te])))
            for k, (mk, kw) in makers.items():
                m = train(mk(), glob.coords[tr], glob.energies[tr], glob.coords[va], glob.energies[va], epochs=3000, restarts=2, **kw)
                G[k][n].append(list(M.rmse_kcal(m.predict(glob.coords[te]), glob.energies[te]))
                               + [M.gap_mev(m, seam), M.permutation_error_mev(m, glob.coords[te], H3_PERMUTATIONS)])
            print(f'  h3 split {s} n {n} ({time.time() - t0:.0f} s)', flush=True)
    res['global'] = {k: {str(n): np.mean(v[n], 0).tolist() for n in ns} for k, v in G.items()}
    res['global']['London'] = np.mean(lon, 0).tolist() + [M.gap_mev(frame.prior_energies, seam), 0.0]
    res['global_columns'] = ['S0 RMSE kcal', 'S1 RMSE kcal', 'seam gap meV', 'swap meV']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for st, ax in enumerate(axes):
        for k in makers:
            v = np.array([[x[st] for x in G[k][n]] for n in ns])
            ax.errorbar(ns, v.mean(1), yerr=[v.mean(1) - v.min(1), v.max(1) - v.mean(1)], color=cols[k], marker='o', ms=7, capsize=3, label=k)
        ax.axhline(np.mean(lon, 0)[st], color=BLUE, ls=':', lw=2.2, label='London, 0 H₃ points')
        ax.set(xscale='log', yscale='log', xlabel='H₃ training points', ylabel='test RMSE (kcal/mol)',
               title=['Ground state', 'Excited state'][st])
        ax.set_xticks(ns); ax.set_xticklabels(ns); ax.xaxis.set_minor_formatter(plt.NullFormatter())
        lo, hi = ax.get_ylim()
        yt = [t for t in [0.5, 1, 2, 5, 10, 20, 50] if lo <= t <= hi]
        ax.set_yticks(yt); ax.set_yticklabels([f'{t:g}' for t in yt]); ax.yaxis.set_minor_formatter(plt.NullFormatter())
    axes[0].legend(loc='upper right')
    fig.tight_layout()
    save(fig, 'h3_learning.png')
    json.dump(res, open(OUT / 'h3_results.json', 'w'), indent=1)


# ------------------------------------------------------------------------------------------------ H2O
def h2o():
    from diabml import RefSet
    from diabml.frames.dim_oh import DIMWaterFrame, bond_additivity
    from diabml.units import EV_KCAL
    big()
    oh, h2, w = RefSet.load(DATA / 'oh_curves.npz'), RefSet.load(DATA / 'h2_631g.npz'), RefSet.load(DATA / 'h2o.npz')
    fr = DIMWaterFrame(oh, h2)
    tag = w.extra['tag']
    def scan(mask, col):
        i = np.where(mask)[0]; return i[np.argsort(w.coords[i, col])]
    sc = [(scan((tag == 'grid') & (w.coords[:, 0] == 0.96), 1), 1, 'r₂ (Å), r₁ = 0.96 Å', 'Breaking one bond: H₂O → H + OH'),
          (scan(tag == 'bend096', 2), 2, 'H–O–H angle (°)', 'Bending, r = 0.96 Å')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (i, col, xl, t) in zip(axes, sc):
        x = w.coords[i, col]
        ax.plot(x, w.energies[i, 0] * EV_KCAL, 'o', color=INK, ms=6, label='exact (FCI)')
        ax.plot(x, fr.prior_energies(w.coords[i])[:, 0] * EV_KCAL, color=BLUE, label='covalent DIM frame, no H₂O data')
        ax.plot(x, bond_additivity(w.coords[i], fr) * EV_KCAL, color=ORANGE, ls='--', lw=2, label='two OH bonds + H–H')
        ax.set(xlabel=xl, ylabel='E − [O + 2H]  (kcal/mol)', title=t)
    axes[0].legend(loc='lower right')
    fig.tight_layout()
    save(fig, 'h2o_dim.png')


# ------------------------------------------------------------------------------------------------ CH2OO
def ch2oo():
    from diabml import RefSet, metrics as M
    from diabml.frames import ChannelFrame, Diabat
    from diabml.models import CDNN, FramedModel, train
    from diabml.refs.ladder import EXPERIMENT as X
    big()
    ref = RefSet.load(DATA / 'ch2oo_casscf.npz')
    r, a = ref.coords[:, 0], ref.coords[:, 1]
    eq = np.where(np.isclose(a, 117.7))[0]; eq = eq[np.argsort(r[eq])]
    cols = [INK, BLUE, ORANGE, GREEN]
    # character
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for s in range(4):
        axes[0].plot(r[eq], ref.energies[eq, s], color=cols[s], marker='o', ms=4, label=f'S{s}')
        axes[1].plot(r[eq], ref.extra['q_O_terminal'][eq, s], color=cols[s], marker='o', ms=4)
        axes[2].plot(r[eq], ref.extra['n_unpaired'][eq, s], color=cols[s], marker='o', ms=4)
    axes[0].set(xlabel='r(O–O) (Å)', ylabel='E (eV) rel. CH₂O + O(¹D)', title='Four singlet states', xlim=(1.1, 5.2))
    axes[1].set(xlabel='r(O–O) (Å)', ylabel='charge on terminal O (e)', title='Charge separation', xlim=(1.1, 5.2))
    axes[2].set(xlabel='r(O–O) (Å)', ylabel='effective unpaired electrons', title='Open-shell character', xlim=(1.1, 5.2))
    axes[0].legend(loc='upper right', ncol=2)
    fig.tight_layout()
    save(fig, 'ch2oo_character.png')
    # frame
    asym = ref.energies[r == r.max()].mean(0)
    dE = X['IE_CH2O_adiabatic'] - X['EA_O'] - X['O_1D_3P']
    diabats = [Diabat('CH₂O + O(¹D) a', asym[0]), Diabat('CH₂O + O(¹D) b', asym[1]), Diabat('CH₂O + O(¹D) c', asym[2]),
               Diabat('CH₂O(T₁) + O(³P)', asym[3], kind='biradical'),
               Diabat('CH₂O⁺ + O⁻', asym[:3].mean() + dE, charge_product=-1.0, kind='ionic')]
    res = {'dE_ion_eV': dE, 'r_cross_A': 14.399645 / dE, 'asymptotes_eV': asym.tolist()}
    inner, outer = np.where(r <= 2.3)[0], np.where(r > 2.3)[0]
    out = {'framed CDNN': [], 'CDNN': []}
    inr = {'framed CDNN': [], 'CDNN': []}
    keep = {}
    for seed in range(3):
        p = np.random.default_rng(seed).permutation(inner)
        tr, va = p[:int(0.8 * len(p))], p[int(0.8 * len(p)):]
        for k in out:
            m = (FramedModel(ChannelFrame(diabats, 2), 4, hidden=(128, 64)) if k == 'framed CDNN' else CDNN(2, 4, n_matrix=5, hidden=(128, 64)))
            m = train(m, ref.coords[tr], ref.energies[tr], ref.coords[va], ref.energies[va], epochs=6000,
                      lr=2e-3 if k == 'framed CDNN' else 1e-3, restarts=2)
            out[k].append(M.rmse_kcal(m.predict(ref.coords[outer]), ref.energies[outer]).tolist())
            inr[k].append(M.rmse_kcal(m.predict(ref.coords[va]), ref.energies[va]).tolist())
            if seed == 0:
                keep[k] = m
        print('  ch2oo seed', seed, flush=True)
    res['beyond_2.3A_rmse_kcal'] = {k: np.mean(v, 0).tolist() for k, v in out.items()}
    res['validation_rmse_kcal'] = {k: np.mean(v, 0).tolist() for k, v in inr.items()}
    x = np.linspace(1.15, 8.0, 400)
    C = np.c_[x, np.full_like(x, 117.7)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, k, c in [(axes[0], 'CDNN', ORANGE), (axes[1], 'framed CDNN', GREEN)]:
        E = keep[k].predict(C)
        ax.axvspan(2.3, 8.2, color=GOLD, alpha=0.13, lw=0)
        for s in range(4):
            ax.plot(r[eq], ref.energies[eq, s], 'o', color=INK, ms=5, mfc='white' if s else INK, label='CASSCF' if s == 0 else None)
            ax.plot(x, E[:, s], color=c, lw=2.2, label=k if s == 0 else None)
        ax.set(xlabel='r(O–O) (Å)', ylabel='E (eV)', ylim=(-2.5, 6), xlim=(1.15, 8.0),
               title=f'{k}: trained on r ≤ 2.3 Å only')
        ax.text(5.2, 5.3, 'no training data here', ha='center', color='#7a5a00', fontsize=14)
        ax.legend(loc='lower right')
    fig.tight_layout()
    save(fig, 'ch2oo_extrapolation.png')
    # learned diabats (framed model trained on everything in range of the random split)
    tr, va, te = ref.split(90, 15, seed=0)
    m = train(FramedModel(ChannelFrame(diabats, 2), 4, hidden=(128, 64)), ref.coords[tr], ref.energies[tr],
              ref.coords[va], ref.energies[va], epochs=6000, lr=2e-3, restarts=2)
    res['random_split_test_rmse_kcal'] = M.rmse_kcal(m.predict(ref.coords[te]), ref.energies[te]).tolist()
    x = np.linspace(1.15, 5.0, 400); C = np.c_[x, np.full_like(x, 117.7)]
    D = np.diagonal(m.matrix(C).detach().numpy(), axis1=1, axis2=2)
    E = m.predict(C)
    fig, ax = plt.subplots(figsize=(11, 5.4))
    kc = {'covalent': BLUE, 'biradical': GREEN, 'ionic': ORANGE}
    for i, dbt in enumerate(diabats):
        ax.plot(x, D[:, i], ls='--', lw=3.0, color=kc[dbt.kind], label=(dbt.label if i != 0 else 'CH₂O + O(¹D), 3 components') if i in (0, 3, 4) else None)
    for s in range(4):
        ax.plot(x, E[:, s], color='#8d96a6', lw=1.6, label='adiabatic states (model)' if s == 0 else None)
        ax.plot(r[eq], ref.energies[eq, s], 'o', color=INK, ms=4, mfc='white', label='CASSCF' if s == 0 else None)
    ax.set(xlabel='r(O–O) (Å)', ylabel='E (eV)', ylim=(-3, 11), xlim=(1.15, 5.0),
           title='Learned structures (dashed) behind the adiabatic states')
    ax.legend(loc='upper right', ncol=2)
    save(fig, 'ch2oo_diabats.png')
    json.dump(res, open(OUT / 'ch2oo_results.json', 'w'), indent=1)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for name, f in [('schematic', schematics), ('h2o', h2o), ('ch2oo', ch2oo), ('h3', h3)]:
        if which in (name, 'all'):
            f()
