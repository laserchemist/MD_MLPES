"""Active learning for the CH2OO workflow: each round is one run folder that adds new reference points where a
committee of models disagrees.

    python -m diabml.workflow.ch2oo init    runs/08_al1 --template active   # set refs.file to the data to extend
    python -m diabml.workflow.ch2oo all     runs/08_al1     # refs, ladder, propose, compute, merge, train, figures,
                                                            # tables, report
or stage by stage:

    refs      copy the data to extend (data/base_refs.npz), its couplings and its orbital library
    propose   train a committee of framed models on the base data (bootstrap + different seeds), evaluate them on
              random candidate geometries inside [active] bounds, and pick the n_new geometries where they disagree
              most, keeping a minimum distance from existing points and from each other  -> data/proposed.npz, A1
    compute   SA-CASSCF (+ SC-NEVPT2, + couplings along r_OO) at the proposed geometries, at exactly the level of
              the base data, each started from the orbitals of the nearest point already computed  -> data/new_refs.npz
              (resumable: data/new_checkpoint.npz after every geometry)
    merge     base + new points -> data/refs.npz (and nacs.npz, orbitals.npz); base rows stay first, every row keeps
              the round it was added in (extra 'al_round')

The next round is a new folder whose refs.file points at this round's data/refs.npz; its refs stage picks up the
merged couplings and orbitals from the same folder.

How to read a round
  committee spread   standard deviation of the committee's energies over the candidates (A1, T9): where the data does
                     not yet pin the surface down. The loop has converged when its 95th percentile is about 1 kcal/mol.
  error at proposed  the committee mean against the new reference energies at the proposed points (A2, T9), measured
                     before those points were trained on: the honest error of the surface where it is weakest.
"""
from __future__ import annotations

import json
import shutil
import time

import numpy as np

from . import ch2oo as W
from . import config as C

EV_KCAL = W.EV_KCAL


# ------------------------------------------------------------------------------------------------ helpers
def _ac(run):
    return run.cfg['active']


def _scales(names, ac):
    ls = ac.get('length_scales', {})
    return np.array([float(ls.get(n, 0.1 if n.startswith('r') else 10.0)) for n in names])


def _round_of(ref):
    return int(np.max(ref.extra['al_round'])) + 1 if 'al_round' in ref.extra else 1


def _with_round(ref):
    if 'al_round' not in ref.extra:
        ref.extra['al_round'] = np.zeros(len(ref), int)
    return ref


def _spec(ref):
    from ..refs.casscf import CASSpec
    return CASSpec(**ref.meta['spec'])


def _library_from_scan(path, ref):
    """Orbitals of the reference scan (scan_checkpoint.npz), matched to ref rows: (coords, mo, ref_irr) or None."""
    z = np.load(path)
    if 'mo' not in z.files:
        return None
    axes = ref.meta.get('grid_axes', ref.coord_names)
    cols = [list(axes).index(a) for a in ref.coord_names]
    key = {tuple(np.round(p[cols], 6)): i for i, p in enumerate(z['points'])}
    rows, idx = [], []
    for r_, c in enumerate(ref.coords):
        i = key.get(tuple(np.round(c, 6)))
        if i is not None and z['done'][i]:
            rows.append(r_); idx.append(i)
    if not rows:
        return None
    irr = json.loads(str(z['ref_irr'])) if 'ref_irr' in z.files else None
    return ref.coords[rows], z['mo'][idx], irr


def _save_library(path, coords, mo, irr):
    np.savez(path, coords=coords, mo=mo, ref_irr=np.array(json.dumps(irr)))


def _load_library(path):
    if not path.exists():
        return None
    z = np.load(path)
    return z['coords'], z['mo'], json.loads(str(z['ref_irr']))


# ------------------------------------------------------------------------------------------------ refs
def setup_refs(run):
    """refs stage of an active run: copy the base data, its couplings and its orbitals; merge new points if any."""
    from ..refset import RefSet
    src = C.resolve(run.cfg['refs']['file'], run.path)
    if not src.exists():
        raise SystemExit(f'refs.file in {run.p("config.toml")} points to {src}, which does not exist: set it to the '
                         f'refs.npz to extend (e.g. "../03_cas_3d/data/refs.npz" or the previous round\'s).')
    base = _with_round(RefSet.load(src))
    if 'spec' not in base.meta or 'zero_hartree' not in base.meta:
        raise SystemExit(f'{src} has no record of how it was computed (meta spec / zero_hartree), so new points could '
                         f'not be made consistent with it. Extend data made by the refs stage of this workflow.')
    base.meta = dict(base.meta, copied_from=str(src))
    base.save(run.p('data', 'base_refs.npz'))
    run.log(f'base data {src}: {base}; round {_round_of(base)}')
    nac = src.parent / 'nacs.npz'
    if nac.exists():
        shutil.copy(nac, run.p('data', 'nacs_base.npz'))
        run.log(f'copied {nac} (ab initio couplings)')
    olib = src.parent / 'orca_orbitals.json'
    if olib.exists():
        shutil.copy(olib, run.p('data', 'orca_orbitals_base.json'))
        run.log(f'copied ORCA orbital library {olib}')
    lib = src.parent / 'orbitals.npz'
    if lib.exists():
        shutil.copy(lib, run.p('data', 'orbitals_base.npz'))
        run.log(f'copied orbital library {lib}')
    elif (src.parent / 'scan_checkpoint.npz').exists():
        out = _library_from_scan(src.parent / 'scan_checkpoint.npz', base)
        if out is None:
            run.log('scan checkpoint has no orbitals: new points will start from scratch (check them in A2)')
        else:
            _save_library(run.p('data', 'orbitals_base.npz'), *out)
            run.log(f'orbital library from {src.parent / "scan_checkpoint.npz"}: {len(out[0])} geometries')
    else:
        run.log('no orbitals next to the base data: new points will start from scratch (check them in A2)')
    return stage_merge(run)


# ------------------------------------------------------------------------------------------------ propose
def _candidates(ref, U, ac, rng):
    names = ref.coord_names
    lo, hi = ref.coords[U].min(0), ref.coords[U].max(0)
    b = ac.get('bounds', {})
    lo = np.array([float(b.get(n, [l, h])[0]) for n, l, h in zip(names, lo, hi)])
    hi = np.array([float(b.get(n, [l, h])[1]) for n, l, h in zip(names, lo, hi)])
    hi[0] = min(hi[0], float(ac['r_max']))
    X = lo + (hi - lo) * rng.random((int(ac['n_candidates']), len(names)))
    keep = np.ones(len(X), bool)
    for box in ac.get('exclude', []):
        inside = np.ones(len(X), bool)
        for n, (a, c) in box.items():
            inside &= (X[:, names.index(n)] >= a) & (X[:, names.index(n)] <= c)
        keep &= ~inside
    return X[keep], lo, hi


def train_committee(run, ref, U):
    """K framed models (different seeds, bootstrap samples of the usable points U): the committee whose
    disagreement guides active learning. Saved as models/committee_<k>.pt. Returns (models, held-out RMSE)."""
    import torch
    from ..models import train, save_model
    ac, tc = _ac(run), run.cfg['train']
    E = ref.energies[:, :4]
    K, frac = int(ac['committee']), float(ac['bootstrap'])
    run.log(f'committee of {K} framed models on {int(frac * 100)}% bootstrap samples of {len(U)} usable points, '
            f'{ac["epochs"]} epochs')
    models, val = [], []
    for c in range(K):
        p = np.random.default_rng(1000 + c).permutation(U)
        n = int(frac * len(p))
        tr, va = p[:n], p[n:]
        torch.manual_seed(c)
        t0 = time.time()
        m = train(W.make_model(run, ref, 'framed'), ref.coords[tr], E[tr], ref.coords[va], E[va],
                  epochs=int(ac['epochs']), restarts=1, lr=float(tc['lr_framed']), alpha2=float(tc['alpha2']))
        rm = np.sqrt(np.mean((m.predict(ref.coords[va]) - E[va]) ** 2, 0)) * EV_KCAL
        val.append(rm.tolist())
        models.append(m)
        save_model(m, run.p('models', f'committee_{c}.pt'),
                   meta={'experiment': f'active-learning committee member {c}', 'coord_names': ref.coord_names,
                         'fixed_coordinates': ref.meta.get('fixed', {}), 'n_train': int(len(tr))})
        run.log(f'  member {c}: held-out RMSE {np.round(rm, 2)} kcal/mol ({time.time() - t0:.0f} s)')
    return models, val


def stage_propose(run, force=False):
    import torch
    from ..models import train, save_model
    from ..refset import RefSet
    out = run.p('data', 'proposed.npz')
    if out.exists() and not force:
        run.log(f'{out.name} exists (use --force to propose again); nothing to do')
        return
    run.threads()
    ac, tc = _ac(run), run.cfg['train']
    ref = RefSet.load(run.p('data', 'base_refs.npz'))
    if tc.get('sort_states', True):
        ref = W.adiabatic_order(ref)
    U = W.usable_points(run, ref)
    models, val = train_committee(run, ref, U)
    K = len(models)
    rng = np.random.default_rng(int(ac.get('seed', 0)) + _round_of(ref))
    X, lo, hi = _candidates(ref, U, ac, rng)
    P = np.stack([m.predict(X) for m in models])                    # (K, N, 4)
    mean, std = P.mean(0), P.std(0)
    states = [int(s) for s in ac['states']]
    w = np.array([float(x) for x in ac['state_weights']])[:len(states)]
    score = np.max(std[:, states] * w, 1)
    ok = mean[:, 0] <= float(ac['max_S0_eV'])
    sc = _scales(ref.coord_names, ac)
    existing = ref.coords / sc
    Z = X / sc
    dmin = float(ac['min_distance'])
    chosen = []
    for i in np.argsort(-np.where(ok, score, -np.inf)):
        if not ok[i] or len(chosen) >= int(ac['n_new']):
            break
        if np.min(np.sum((existing - Z[i]) ** 2, 1)) < dmin ** 2:
            continue
        if chosen and np.min(np.sum((Z[chosen] - Z[i]) ** 2, 1)) < dmin ** 2:
            continue
        chosen.append(int(i))
    chosen = np.array(chosen, int)
    pct = {f'p{q}': (np.percentile(std[ok], q, axis=0) * EV_KCAL).tolist() for q in (50, 95)}
    stats = {'round': _round_of(ref), 'n_base': len(ref), 'n_usable': int(len(U)), 'committee': K,
             'committee_val_rmse_kcal': val, 'n_candidates': int(len(X)), 'n_candidates_in_window': int(ok.sum()),
             'bounds': {n: [float(a), float(b)] for n, a, b in zip(ref.coord_names, lo, hi)},
             'spread_kcal': pct, 'spread_max_kcal': (std[ok].max(0) * EV_KCAL).tolist(),
             'n_proposed': int(len(chosen)),
             'score_proposed_kcal': [float(score[chosen].min() * EV_KCAL), float(score[chosen].max() * EV_KCAL)]
             if len(chosen) else None}
    np.savez(out, coords=X[chosen], score=score[chosen], mean=mean[chosen], std=std[chosen],
             cand=X[ok], cand_score=score[ok], round=stats['round'])
    _write_json(run, stats, 'propose')
    run.log(f'committee spread over {int(ok.sum())} candidates (S0 mean ≤ {ac["max_S0_eV"]} eV): median '
            f'{np.round(pct["p50"], 2)}, 95th percentile {np.round(pct["p95"], 2)} kcal/mol (S0..S3)')
    run.log(f'proposed {len(chosen)} geometries, scores {stats["score_proposed_kcal"]} kcal/mol -> {out}')
    with open(run.p('results', 'proposed.csv'), 'w') as fh:
        fh.write(','.join(ref.coord_names + ['score_kcal'] + [f'std_S{s}_kcal' for s in range(4)]) + '\n')
        for i in chosen:
            fh.write(','.join(f'{v:.4f}' for v in list(X[i]) + [score[i] * EV_KCAL] + list(std[i] * EV_KCAL)) + '\n')
    _figure_proposal(run, ref, X[ok], score[ok], X[chosen])


def _write_json(run, d, key):
    path = run.p('results', 'active.json')
    allr = json.loads(path.read_text()) if path.exists() else {}
    allr[key] = d
    path.write_text(json.dumps(allr, indent=1))


def _active_caption(run, name, text):
    path = run.p('figures', 'active_captions.json')
    caps = json.loads(path.read_text()) if path.exists() else {}
    caps[name] = text
    path.write_text(json.dumps(caps, indent=1))


def _figure_proposal(run, ref, X, score, chosen):
    plt = W._mpl()
    names = ref.coord_names
    pairs = [(0, j) for j in range(1, len(names))] or [(0, 0)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(6.4 * len(pairs), 4.6), squeeze=False)
    rounds = ref.extra.get('al_round', np.zeros(len(ref), int))
    for ax, (i, j) in zip(axes.flat, pairs):
        o = np.argsort(score)
        pc = ax.scatter(X[o, i], X[o, j], c=score[o] * EV_KCAL, s=4, cmap='viridis', alpha=0.6)
        ax.scatter(ref.coords[rounds == 0, i], ref.coords[rounds == 0, j], marker='+', color='0.45', s=22,
                   label='grid points')
        if (rounds > 0).any():
            ax.scatter(ref.coords[rounds > 0, i], ref.coords[rounds > 0, j], marker='x', color='C1', s=22,
                       label='earlier rounds')
        ax.scatter(chosen[:, i], chosen[:, j], facecolor='none', edgecolor='red', s=60, lw=1.5, label='proposed')
        ax.set(xlabel=f'{W.LABEL[names[i]]} ({W.UNITS[names[i]]})', ylabel=f'{W.LABEL[names[j]]} ({W.UNITS[names[j]]})')
        fig.colorbar(pc, ax=ax, label='committee spread (kcal/mol)')
    axes.flat[0].legend(fontsize=9, loc='upper right')
    fig.tight_layout()
    fig.savefig(run.p('figures', 'A1_proposal.png'), bbox_inches='tight')
    plt.close(fig)
    _active_caption(run, 'A1_proposal.png', 'Active learning: committee spread (largest weighted standard deviation '
                    'over the states) at random candidate geometries, projected on pairs of coordinates; existing '
                    'points and the geometries proposed for new reference calculations (red).')
    run.log('figures/A1_proposal.png')


# ------------------------------------------------------------------------------------------------ compute
def stage_compute(run):
    from ..geometry import ch2oo_xyz
    from ..refs.casscf import _mol, active_irreps, nevpt2_states, run_sa_casscf, state_diagnostics
    from ..refset import RefSet
    from ..units import BOHR_ANG, HARTREE_EV
    prop = run.p('data', 'proposed.npz')
    if not prop.exists():
        raise SystemExit('no data/proposed.npz: run the propose stage first')
    n_thr = run.threads()
    ac = _ac(run)
    base = RefSet.load(run.p('data', 'base_refs.npz'))
    z = np.load(prop)
    X = z['coords']
    N = len(X)
    if N == 0:
        for f in ('new_refs.npz', 'new_orbitals.npz', 'new_nacs.npz', 'new_checkpoint.npz'):
            run.p('data', f).unlink(missing_ok=True)
        _write_json(run, {'n_new': 0, 'n_usable_new': 0, 'hours': 0.0}, 'compute')
        run.log('no geometries proposed: nothing to compute')
        return
    if run.cfg['compute'].get('engine', 'pyscf') == 'orca':
        from .orca_compute import stage_compute_orca
        return stage_compute_orca(run)
    if base.meta.get('energy_level', 'casscf') not in ('casscf', 'nevpt2'):
        raise SystemExit(f'the base data are at the {base.meta["energy_level"]} level, which PySCF cannot make: set '
                         f'[compute] engine = "orca" in {run.p("config.toml")}')
    spec = _spec(base)
    zero = base.meta['zero_hartree']
    dynamic = 'nevpt2' in zero
    pairs = None
    if ac.get('nacs', True) and run.p('data', 'nacs_base.npz').exists():
        pairs = [tuple(int(i) for i in p) for p in np.load(run.p('data', 'nacs_base.npz'))['pairs']]
    r_nac = float(run.cfg['nacs'].get('r_max', 4.0))
    sc = _scales(base.coord_names, ac)
    lib = _load_library(run.p('data', 'orbitals_base.npz'))
    L_coords, L_mo, ref_irr = lib if lib is not None else (np.zeros((0, X.shape[1])), None, None)
    R = dict(E=np.full((N, spec.nroots), np.nan), E_pt=np.full((N, spec.nroots), np.nan),
             q=np.full((N, spec.nroots), np.nan), nu=np.full((N, spec.nroots), np.nan), conv=np.zeros(N, bool),
             cons=np.zeros(N, bool), mism=np.full(N, np.nan), sec=np.zeros(N), done=np.zeros(N, bool),
             tau=np.full((N, len(pairs or [])), np.nan), vec=np.full((N, len(pairs or []), 5, 3), np.nan))
    mo_new = None
    ck = run.p('data', 'new_checkpoint.npz')
    if ck.exists():
        c = np.load(ck)
        if c['coords'].shape == X.shape and np.allclose(c['coords'], X):
            R = {k: c[k].copy() for k in R}
            mo_new = c['mo'].copy() if 'mo' in c.files else None
            ref_irr = json.loads(str(c['ref_irr'])) if 'ref_irr' in c.files else ref_irr
            run.log(f'  resuming: {int(R["done"].sum())}/{N} geometries already done')
    run.log(f'{N} new geometries; {spec}; NEVPT2 {dynamic}; couplings {pairs if pairs else "no"}; '
            f'orbital library {len(L_coords)} geometries; {n_thr} threads')

    def kw(x):
        return W._geometry_kwargs(base, x)

    def nearest(i, exclude=()):
        """Nearest geometry with known orbitals: ('lib', j) or ('new', j)."""
        cands = []
        if len(L_coords):
            d = np.sum(((L_coords - X[i]) / sc) ** 2, 1)
            cands += [(float(d[j]), 'lib', int(j)) for j in np.argsort(d)[:3]]
        for j in np.where(R['done'])[0]:
            cands.append((float(np.sum(((X[j] - X[i]) / sc) ** 2)), 'new', int(j)))
        cands = [c for c in sorted(cands) if (c[1], c[2]) not in exclude]
        return cands[0] if cands else None

    def guess(src):
        if src is None:
            return None
        _, kind, j = src
        if kind == 'lib':
            return L_mo[j], _mol(ch2oo_xyz(**kw(L_coords[j])), spec)
        return mo_new[j], _mol(ch2oo_xyz(**kw(X[j])), spec)

    def save():
        tmp = str(ck) + '.tmp.npz'
        np.savez(tmp, coords=X, ref_irr=np.array(json.dumps(ref_irr)), **R, **({} if mo_new is None else {'mo': mo_new}))
        import os
        os.replace(tmp, ck)

    t0 = time.time()
    while not R['done'].all():
        left = np.where(~R['done'])[0]
        near = [nearest(i) for i in left]
        k_ = int(np.argmin([n_[0] if n_ else 0.0 for n_ in near]))    # always grow from what is already known
        i, src = int(left[k_]), near[k_]
        t1 = time.time()
        atoms = ch2oo_xyz(**kw(X[i]))
        mol, mc = run_sa_casscf(atoms, spec, guess(src))
        if not mc.converged and src is not None:
            alt = nearest(i, exclude={(src[1], src[2])})
            if alt is not None:
                mol, mc = run_sa_casscf(atoms, spec, guess(alt))
        R['E'][i], R['conv'][i] = mc.e_states, mc.converged
        R['q'][i], R['nu'][i] = state_diagnostics(mol, mc, (2,))
        irr = sorted(active_irreps(mol, mc))
        ref_irr = irr if ref_irr is None else ref_irr
        R['cons'][i] = irr == sorted(ref_irr)
        if dynamic:
            e_pt, e_ci = nevpt2_states(mc, spec)
            R['E_pt'][i] = e_pt
            R['mism'][i] = float(np.max(np.abs(e_ci - np.asarray(mc.e_states))))
        if pairs and X[i, 0] <= r_nac and mc.converged:
            from pyscf.nac import sacasscf as NAC
            xyz = np.array([x for _, x in atoms])
            e = (xyz[2] - xyz[1]) / np.linalg.norm(xyz[2] - xyz[1])
            nac = NAC.NonAdiabaticCouplings(mc)
            nac.verbose = 0
            for k, pr in enumerate(pairs):
                v = np.asarray(nac.kernel(state=pr)) / BOHR_ANG
                R['vec'][i, k] = v
                R['tau'][i, k] = abs(float(v[2] @ e))
        if mo_new is None:
            mo_new = np.full((N,) + mc.mo_coeff.shape, np.nan)
        mo_new[i] = mc.mo_coeff
        R['done'][i] = True
        R['sec'][i] = time.time() - t1
        save()
        k = int(R['done'].sum())
        if k % 5 == 1 or k == N:
            run.log(f'  {k}/{N}  {np.round(X[i], 3)}  converged={mc.converged}  start: '
                    f'{"scratch" if src is None else src[1]}  {R["sec"][i]:.1f} s  ({time.time() - t0:.0f} s)')
    # energies on the base data's zero, at the level the base data trains on
    ev = {'casscf': (R['E'] - zero['casscf']) * HARTREE_EV}
    if dynamic:
        ev['nevpt2'] = (R['E_pt'] - zero['nevpt2']) * HARTREE_EV
    level = base.meta.get('energy_level', 'casscf')
    rnd = int(z['round'])
    extra = {'q_O_terminal': R['q'], 'n_unpaired': R['nu'], 'converged': R['conv'], 'active_space_consistent': R['cons'],
             'seconds': R['sec'], 'E_casscf_eV': ev['casscf'], 'al_round': np.full(N, rnd, int)}
    if dynamic:
        extra['E_nevpt2_eV'] = ev['nevpt2']
        extra['casci_mismatch_hartree'] = R['mism']
    new = RefSet(base.name, X, ev[level], list(base.coord_names), extra=extra,
                 meta=dict(base.meta, active_round=rnd))
    new.save(run.p('data', 'new_refs.npz'))
    np.savez(run.p('data', 'new_orbitals.npz'), coords=X, mo=mo_new)
    if pairs:
        np.savez(run.p('data', 'new_nacs.npz'), tau=R['tau'], vectors=R['vec'], pairs=np.array(pairs),
                 E_eV=ev['casscf'])
    # how wrong was the surface where it was least sure?
    good = R['conv'] & R['cons']
    if dynamic:
        good &= np.nan_to_num(R['mism']) <= float(run.cfg['train'].get('max_casci_mismatch_hartree', 1e-3))
    E_new = np.sort(ev[level], 1) if run.cfg['train'].get('sort_states', True) else ev[level]
    err = (z['mean'] - E_new[:, :4])[good] * EV_KCAL
    stats = {'n_new': N, 'n_usable_new': int(good.sum()), 'hours': float(R['sec'].sum() / 3600),
             'error_at_proposed_rmse_kcal': np.sqrt(np.mean(err ** 2, 0)).tolist() if len(err) else None,
             'error_at_proposed_max_kcal': np.abs(err).max(0).tolist() if len(err) else None,
             'spread_at_proposed_kcal': (z['std'][good].mean(0) * EV_KCAL).tolist() if good.any() else None,
             'started_from_scratch': int(lib is None)}
    _write_json(run, stats, 'compute')
    run.log(f'{int(good.sum())}/{N} new points usable (converged, same active space'
            f'{", CASCI reproduces SA-CASSCF" if dynamic else ""}); {stats["hours"]:.2f} h of compute')
    if len(err):
        run.log(f'committee error at the proposed points, before training on them: RMSE '
                f'{np.round(stats["error_at_proposed_rmse_kcal"], 2)} kcal/mol (spread there '
                f'{np.round(stats["spread_at_proposed_kcal"], 2)})')
    _figure_new_points(run, z['mean'], z['std'], E_new, good)
    ck.unlink(missing_ok=True)


def _figure_new_points(run, mean, std, E, good):
    plt = W._mpl()
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for s, ax in enumerate(axes):
        y, p, e = E[good, s], mean[good, s], std[good, s]
        ax.errorbar(y, p, yerr=e, fmt='o', ms=4, color=W.SCOL[s], ecolor='0.6', capsize=2)
        if (~good).any():
            ax.plot(E[~good, s], mean[~good, s], 'x', color='red', label='not usable')
        lim = [np.nanmin(np.r_[y, p]) - 0.2, np.nanmax(np.r_[y, p]) + 0.2] if len(y) else [0, 1]
        ax.plot(lim, lim, 'k--', lw=0.8)
        rm = np.sqrt(np.mean((p - y) ** 2)) * EV_KCAL if len(y) else np.nan
        ax.set(xlabel='new reference (eV)', ylabel='committee mean (eV)', title=f'S{s}: RMSE {rm:.2f} kcal/mol')
    fig.tight_layout()
    fig.savefig(run.p('figures', 'A2_new_points.png'), bbox_inches='tight')
    plt.close(fig)
    _active_caption(run, 'A2_new_points.png', 'Active learning: the committee\'s prediction (bars: committee spread) '
                    'against the new reference energies at the proposed geometries, before the model was trained on '
                    'them. Red crosses: unconverged points, a changed active space or a NEVPT2 state mismatch.')
    run.log('figures/A2_new_points.png')


# ------------------------------------------------------------------------------------------------ merge
def _fill(v, n):
    v = np.asarray(v)
    if v.dtype == bool:
        return np.zeros((n,) + v.shape[1:], bool)
    if np.issubdtype(v.dtype, np.integer):
        return np.zeros((n,) + v.shape[1:], v.dtype)
    return np.full((n,) + v.shape[1:], np.nan)


def stage_merge(run):
    from ..refset import RefSet
    base = _with_round(RefSet.load(run.p('data', 'base_refs.npz')))
    newp = run.p('data', 'new_refs.npz')
    nb = len(base)
    lib_base = run.p('data', 'orbitals_base.npz')
    olib_b, olib_n = run.p('data', 'orca_orbitals_base.json'), run.p('data', 'new_orca_orbitals.json')
    if not newp.exists():
        if olib_b.exists():
            shutil.copy(olib_b, run.p('data', 'orca_orbitals.json'))
        base.save(run.p('data', 'refs.npz'))
        if run.p('data', 'nacs_base.npz').exists():
            shutil.copy(run.p('data', 'nacs_base.npz'), run.p('data', 'nacs.npz'))
        if lib_base.exists():
            shutil.copy(lib_base, run.p('data', 'orbitals.npz'))
        run.log(f'data/refs.npz = base data ({nb} points); no new points yet')
        return base
    new = RefSet.load(newp)
    if list(new.coord_names) != list(base.coord_names) or new.meta.get('spec') != base.meta.get('spec') or \
            new.meta.get('zero_hartree') != base.meta.get('zero_hartree'):
        raise SystemExit('new points were not computed at the level of the base data (coordinates, method or energy '
                         'zero differ): refusing to merge')
    sc = _scales(base.coord_names, _ac(run))
    keep = np.ones(len(new), bool)
    for i, x in enumerate(new.coords):
        d = np.sqrt(np.sum(((base.coords - x) / sc) ** 2, 1))
        j = int(np.argmin(d))
        if d[j] < 1e-6:
            keep[i] = False
            run.log(f'  {np.round(x, 4)} is already in the base data (energies differ by '
                    f'{np.nanmax(np.abs(new.energies[i] - base.energies[j])) * 1000:.1f} meV): keeping the base point')
    new = new.subset(np.where(keep)[0])
    if olib_b.exists() or olib_n.exists():
        from .orca_compute import load_library, save_library
        bx, bg = load_library(olib_b)
        nx, ng = load_library(olib_n)
        kept = {tuple(np.round(c, 6)) for c in new.coords}
        sel = [i for i, c in enumerate(nx) if tuple(np.round(c, 6)) in kept]
        allx = np.vstack([bx.reshape(-1, base.coords.shape[1]), nx[sel].reshape(-1, base.coords.shape[1])])
        save_library(run.p('data', 'orca_orbitals.json'), base.coord_names, allx, list(bg) + [ng[i] for i in sel])
        run.log(f'ORCA orbital library: {len(bg)} + {len(sel)} = {len(allx)} points')
    extra = {}
    for k in sorted(set(base.extra) | set(new.extra)):
        a = base.extra.get(k)
        b = new.extra.get(k)
        if a is None:
            a = _fill(b, nb)
        if b is None:
            b = _fill(a, len(new))
        extra[k] = np.concatenate([np.asarray(a), np.asarray(b).astype(np.asarray(a).dtype)])
    rounds = sorted(set(int(x) for x in extra['al_round']))
    merged = RefSet(base.name, np.vstack([base.coords, new.coords]), np.vstack([base.energies, new.energies]),
                    list(base.coord_names), extra=extra,
                    meta=dict(base.meta, active_rounds=rounds, n_grid=int((extra['al_round'] == 0).sum())))
    merged.save(run.p('data', 'refs.npz'))
    # couplings: base rows keep their indices; new rows are appended after them
    nb_path = run.p('data', 'nacs_base.npz')
    nn_path = run.p('data', 'new_nacs.npz')
    if nb_path.exists():
        zb = dict(np.load(nb_path))
        if nn_path.exists():
            zn = np.load(nn_path)
            sel = np.where(keep & np.isfinite(zn['tau']).any(1))[0]
            pos = np.cumsum(keep) - 1                           # row of each kept new point among the new rows
            idx_new = nb + pos[sel]
            Xn = new.coords[pos[sel]] if len(sel) else np.zeros((0, base.coords.shape[1]))
            add = {'r': Xn[:, 0], 'idx': idx_new, 'coords': Xn, 'tau': zn['tau'][sel], 'vectors': zn['vectors'][sel],
                   'E_eV': zn['E_eV'][sel], 'converged': np.ones(len(sel), bool),
                   'energy_mismatch_eV': np.zeros(len(sel))}
            for k, v in add.items():
                zb[k] = np.concatenate([zb[k], v.astype(zb[k].dtype) if len(v) else v.reshape((0,) + zb[k].shape[1:])])
            if 'tau_all' in zb:
                zb['tau_all'] = np.concatenate([zb['tau_all'], zn['tau'][sel]])
            run.log(f'couplings: {len(sel)} new geometries added to nacs.npz')
        np.savez(run.p('data', 'nacs.npz'), **zb)
    if lib_base.exists():
        lb = np.load(lib_base)
        if run.p('data', 'new_orbitals.npz').exists():
            ln = np.load(run.p('data', 'new_orbitals.npz'))
            ok = keep & np.isfinite(ln['mo']).all(axis=(1, 2))
            _save_library(run.p('data', 'orbitals.npz'), np.vstack([lb['coords'], ln['coords'][ok]]),
                          np.concatenate([lb['mo'], ln['mo'][ok]]), json.loads(str(lb['ref_irr'])))
        else:
            shutil.copy(lib_base, run.p('data', 'orbitals.npz'))
    U = W.usable_points(run, merged, log=False)
    run.log(f'merged: {nb} base + {len(new)} new = {len(merged)} points ({int((U >= nb).sum())} new usable); '
            f'rounds {rounds} -> data/refs.npz')
    return merged


def table_rows(run):
    """Rows for T9 (active learning) or None."""
    path = run.p('results', 'active.json')
    if not path.exists():
        return None
    a = json.loads(path.read_text())
    p, c = a.get('propose', {}), a.get('compute', {})
    if not p and not c.get('n_new'):
        return None
    f = lambda v: ' / '.join(f'{x:.2f}' for x in v) if v else ''
    rows = [] if not p else [['round', p.get('round', '')], ['base points (usable)', f'{p.get("n_base", "")} ({p.get("n_usable", "")})'],
            ['committee members', p.get('committee', '')],
            ['candidates (in the energy window)', f'{p.get("n_candidates", "")} ({p.get("n_candidates_in_window", "")})'],
            ['committee spread, median S0 / S1 / S2 / S3 (kcal/mol)', f(p.get('spread_kcal', {}).get('p50'))],
            ['committee spread, 95th percentile (kcal/mol)', f(p.get('spread_kcal', {}).get('p95'))],
            ['proposed geometries', p.get('n_proposed', '')]]
    if c:
        rows += [['new points usable', f'{c["n_usable_new"]} / {c["n_new"]}'],
                 ['committee error at the proposed points, RMSE (kcal/mol)', f(c.get('error_at_proposed_rmse_kcal'))],
                 ['committee error at the proposed points, max (kcal/mol)', f(c.get('error_at_proposed_max_kcal'))],
                 ['committee spread at the proposed points (kcal/mol)', f(c.get('spread_at_proposed_kcal'))],
                 ['compute (h)', f'{c["hours"]:.2f}']]
    return rows
