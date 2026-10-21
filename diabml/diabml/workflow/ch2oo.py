"""CH2OO workflow: reference scans -> ladder -> training -> figures -> tables -> report, one run folder per study.

    python -m diabml.workflow.ch2oo init     RUN --template nevpt2_2d     # make RUN/ and RUN/config.toml
    python -m diabml.workflow.ch2oo refs     RUN     # SA-CASSCF (+ NEVPT2) scan; resumable, checkpoint every point
    python -m diabml.workflow.ch2oo status   RUN     # progress and time per point of a running scan
    python -m diabml.workflow.ch2oo ladder   RUN     # fragment IE / EA (delta-CCSD(T)) for the ion-pair asymptote
    python -m diabml.workflow.ch2oo train    RUN     # framed CDNN vs CDNN: random split, extrapolation, learning curve,
                                                     # optional diabatic restraint
    python -m diabml.workflow.ch2oo figures  RUN
    python -m diabml.workflow.ch2oo tables   RUN
    python -m diabml.workflow.ch2oo report   RUN     # RUN/report.html
    python -m diabml.workflow.ch2oo all      RUN     # refs, ladder, train, figures, tables, report

Everything a stage needs is read from RUN/config.toml and from files earlier stages wrote, so stages can be rerun
one at a time (e.g. change training settings and rerun train, figures, tables, report without touching the scan).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import config as C

AXES = ['r_OO', 'a_COO', 'r_CO', 'r_CH', 'a_HCO']                   # allowed grid coordinates, in this order
KW = {'r_OO': 'r_oo', 'a_COO': 'a_coo', 'r_CO': 'r_co', 'r_CH': 'r_ch', 'a_HCO': 'a_hco'}
UNITS = {'r_OO': 'Å', 'a_COO': '°', 'r_CO': 'Å', 'r_CH': 'Å', 'a_HCO': '°'}
LABEL = {'r_OO': 'r(O–O)', 'a_COO': '∠C–O–O', 'r_CO': 'r(C–O)', 'r_CH': 'r(C–H)', 'a_HCO': '∠H–C–O'}
ION = 4                                                             # index of the ionic structure in the frame
EV_KCAL = 23.060548


# ------------------------------------------------------------------------------------------------ plumbing
class Run:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        for d in ('data', 'models', 'results', 'figures', 'tables', 'logs'):
            (self.path / d).mkdir(parents=True, exist_ok=True)
        self.cfg = C.load(self.path)
        self._log = None

    def p(self, *parts):
        return self.path.joinpath(*parts)

    def open_log(self, stage):
        self._log = open(self.p('logs', f'{stage}.log'), 'a')
        self.log(f'==== {stage}  ({time.strftime("%Y-%m-%d %H:%M:%S")})  run: {self.path}')

    def log(self, msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        if self._log:
            self._log.write(line + '\n'); self._log.flush()

    def threads(self):
        n = int(self.cfg['run'].get('threads', 8))
        os.environ.setdefault('OMP_NUM_THREADS', str(n))
        try:
            import torch
            torch.set_num_threads(n)
        except Exception:
            pass
        try:
            from pyscf import lib
            lib.num_threads(n)
        except Exception:
            pass
        return n


def _reference_values():
    from ..geometry import CH2OO_REF
    return {'r_OO': CH2OO_REF['r_oo'], 'a_COO': CH2OO_REF['a_coo'], 'r_CO': CH2OO_REF['r_co'],
            'r_CH': CH2OO_REF['r_ch'], 'a_HCO': CH2OO_REF['a_hco']}


def load_refs(run):
    from ..refset import RefSet
    path = run.p('data', 'refs.npz')
    if not path.exists():
        sys.exit(f'{path} not found: run the refs stage first')
    return RefSet.load(path)


def slice_values(ref):
    """For every coordinate except r_OO: the grid value closest to the CH2OO equilibrium (where cuts are drawn)."""
    rv = _reference_values()
    out = {}
    for j, name in enumerate(ref.coord_names):
        if name == 'r_OO':
            continue
        vals = np.unique(ref.coords[:, j])
        out[name] = float(vals[np.argmin(np.abs(vals - rv.get(name, vals.mean())))])
    return out


def cut_mask(ref, values=None):
    values = slice_values(ref) if values is None else values
    m = np.ones(len(ref), bool)
    for j, name in enumerate(ref.coord_names):
        if name in values:
            m &= np.isclose(ref.coords[:, j], values[name])
    return m


def cut_coords(ref, x, values=None):
    values = slice_values(ref) if values is None else values
    cols = [x if name == 'r_OO' else np.full_like(x, values[name]) for name in ref.coord_names]
    return np.stack(cols, 1)


# ------------------------------------------------------------------------------------------------ init
def stage_init(path, template):
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    cfg = path / 'config.toml'
    if cfg.exists():
        print(f'{cfg} exists; leaving it unchanged')
    else:
        if template not in C.TEMPLATES:
            sys.exit(f'unknown template {template}; choose from {sorted(C.TEMPLATES)}')
        cfg.write_text(C.TEMPLATES[template])
        print(f'wrote {cfg} (template "{template}")')
    Run(path)
    print('folders: data/ models/ results/ figures/ tables/ logs/')


# ------------------------------------------------------------------------------------------------ refs
def _grid(cfg):
    grid = cfg['refs']['grid']
    unknown = set(grid) - set(AXES)
    if unknown:
        sys.exit(f'unknown grid coordinates {sorted(unknown)}; allowed: {AXES}')
    if 'r_OO' not in grid:
        sys.exit('the grid needs r_OO (the dissociation coordinate)')
    rv = _reference_values()
    axes = [a for a in AXES if a in grid]
    values = [sorted(float(v) for v in np.atleast_1d(grid[a])) for a in axes]
    fixed = {a: rv[a] for a in AXES if a not in grid}
    fixed.update({a: v[0] for a, v in zip(axes, values) if len(v) == 1})
    varying = [a for a, v in zip(axes, values) if len(v) > 1]
    pts = np.array(list(itertools.product(*values)), float)
    return axes, pts, varying, fixed


def stage_refs(run, force=False):
    from ..refset import RefSet
    cfg = run.cfg['refs']
    out = run.p('data', 'refs.npz')
    if cfg['source'] == 'file':
        src = C.resolve(cfg['file'], run.path)
        ref = RefSet.load(src)
        ref.meta = dict(ref.meta, copied_from=str(src))
        ref.save(out)
        run.log(f'copied {src} -> {out}: {ref}')
        return ref
    if out.exists() and not force:
        run.log(f'{out} exists (use --force to recompute); nothing to do')
        return RefSet.load(out)
    from ..geometry import ch2oo_xyz
    from ..refs.casscf import CASSpec, scan
    from ..units import HARTREE_EV
    n = run.threads()
    axes, pts, varying, fixed = _grid(run.cfg)
    spin_ss = {'singlet': 0.0, 'triplet': 2.0}[cfg['spin']]
    spec = CASSpec(ncas=int(cfg['ncas']), nelecas=int(cfg['nelecas']), nroots=int(cfg['nroots']), basis=cfg['basis'],
                   symmetry='Cs', wfnsym="A'", spin_ss=spin_ss, max_cycle=int(cfg['max_cycle']))
    dynamic = None if cfg['dynamic'] in (None, '', 'none') else cfg['dynamic']
    run.log(f'{len(pts)} geometries over {axes} (varying {varying}; fixed {fixed}); {spec}; dynamic={dynamic}; '
            f'{n} threads')
    # start at the point nearest equilibrium so the first orbitals are the well-behaved ones
    rv = _reference_values()
    scale = np.array([1.0 if a.startswith('r') else 50.0 for a in axes])
    i0 = int(np.argmin(np.sum(((pts - [rv[a] for a in axes]) / scale) ** 2, 1)))
    perm = np.r_[i0, np.delete(np.arange(len(pts)), i0)]

    def atoms_fn(*p):
        return ch2oo_xyz(**{KW[a]: v for a, v in zip(axes, p)})

    R = scan(pts[perm], atoms_fn, spec, fragment_atoms=(2,), log=run.log,
             checkpoint=str(run.p('data', 'scan_checkpoint.npz')), dynamic=dynamic)
    inv = np.argsort(perm)
    R = {k: v[inv] for k, v in R.items()}
    # zero of energy: lowest state at the largest r_OO, other coordinates nearest equilibrium (CH2O + O(1D))
    far = np.isclose(pts[:, 0], pts[:, 0].max())
    for j, a in enumerate(axes[1:], 1):
        vals = np.unique(pts[:, j])
        far &= np.isclose(pts[:, j], vals[np.argmin(np.abs(vals - rv[a]))])
    levels = {'casscf': R['E_hartree']}
    if dynamic == 'nevpt2':
        levels['nevpt2'] = R['E_nevpt2_hartree']
    zeros = {k: float(np.nanmin(v[far, 0])) for k, v in levels.items()}
    ev = {k: (v - zeros[k]) * HARTREE_EV for k, v in levels.items()}
    use = cfg['energy'] if cfg['energy'] in ev else 'casscf'
    cols = [axes.index(a) for a in varying]
    extra = {'q_O_terminal': R['charge'], 'n_unpaired': R['n_unpaired'], 'converged': R['converged'],
             'active_space_consistent': R['active_space_consistent'], 'seconds': R['seconds'],
             'E_casscf_eV': ev['casscf']}
    if 'nevpt2' in ev:
        extra['E_nevpt2_eV'] = ev['nevpt2']
        extra['casci_mismatch_hartree'] = R['casci_mismatch_hartree']
    method = f"SA{spec.nroots}-CASSCF({spec.nelecas},{spec.ncas})/{spec.basis}" + (' + SC-NEVPT2' if dynamic else '')
    ref = RefSet(f'CH2OO {method} ({cfg["spin"]}s)', pts[:, cols], ev[use], varying, extra=extra,
                 meta={'method': method, 'energy_level': use, 'spin': cfg['spin'], 'fixed': fixed,
                       'zero': 'lowest state at the largest r_OO, other coordinates nearest equilibrium',
                       'zero_hartree': zeros, 'grid_axes': axes, 'spec': spec.__dict__})
    ref.save(out)
    run.log(f'saved {out}: {ref}; converged {int(R["converged"].sum())}/{len(pts)}, '
            f'consistent active space {int(R["active_space_consistent"].sum())}/{len(pts)}, '
            f'{R["seconds"].sum() / 3600:.2f} h of compute')
    return ref


def stage_status(run):
    ck = run.p('data', 'scan_checkpoint.npz')
    if not ck.exists():
        print('no scan checkpoint yet'); return
    z = np.load(ck)
    done, sec = z['done'], z['seconds']
    n, k = len(done), int(done.sum())
    per = sec[done].mean() if k else float('nan')
    print(f'{k}/{n} points done; {per:.1f} s per point; about {(n - k) * per / 60:.0f} min left; '
          f'{int((~z["converged"][done]).sum())} unconverged so far')


# ------------------------------------------------------------------------------------------------ ladder
def stage_ladder(run):
    from ..refs import ladder as L
    lc = run.cfg['ladder']
    X = L.EXPERIMENT
    out = {'experiment': dict(X), 'dE_ion_experiment': X['IE_CH2O_adiabatic'] - X['EA_O'] - X['O_1D_3P']}
    old = run.p('data', 'ladder.json')
    if old.exists():
        prev = json.loads(old.read_text())
        if 'computed' in prev and prev['computed'].get('basis') == lc['basis']:
            out['computed'] = prev['computed']
    if lc['compute'] and 'computed' not in out:
        run.threads()
        run.log(f'delta-CCSD(T)/{lc["basis"]}: CH2O, CH2O+, O(3P), O-')
        out['computed'] = L.compute(basis=lc['basis'], log=run.log)
    out['use'] = lc['use'] if (lc['use'] == 'experiment' or 'computed' in out) else 'experiment'
    out['dE_ion_used'] = out['computed']['dE_ion'] if out['use'] == 'computed' else out['dE_ion_experiment']
    out['r_cross_A'] = 14.399645 / out['dE_ion_used']
    old.write_text(json.dumps(out, indent=1))
    run.log(f'ion-pair asymptote {out["dE_ion_used"]:.3f} eV ({out["use"]}); Coulomb crossing r* = '
            f'{out["r_cross_A"]:.2f} Å')
    return out


def _dE_ion(run):
    path = run.p('data', 'ladder.json')
    if not path.exists():
        run.log('no ladder.json: using experimental IE, EA and O(1D)-O(3P) (run the ladder stage to record them)')
        return stage_ladder(run)['dE_ion_used']
    return json.loads(path.read_text())['dE_ion_used']


# ------------------------------------------------------------------------------------------------ frame
class Asymptotes:
    """Asymptotic energies of the five structures as a function of the coordinates that survive dissociation.

    States 0-2 (CH2O + O(1D)) and 3 (CH2O(T1) + O(3P)) come from the reference data at the largest r_OO, averaged
    over the angle; if r_CO varies they are interpolated in r_CO (the CH2O fragment energy depends on it). The ion
    pair sits dE_ion (ladder) above the mean of the three O(1D) components."""

    def __init__(self, ref, dE_ion):
        r = ref.coords[:, 0]
        far = np.isclose(r, r.max())
        self.j = ref.coord_names.index('r_CO') if 'r_CO' in ref.coord_names else None
        self.dE = float(dE_ion)
        if self.j is None:
            self.knots = np.array([0.0])
            self.vals = np.nanmean(ref.energies[far, :4], 0)[None, :]
        else:
            self.knots = np.unique(ref.coords[far, self.j])
            self.vals = np.array([np.nanmean(ref.energies[far & np.isclose(ref.coords[:, self.j], k), :4], 0)
                                  for k in self.knots])

    def __call__(self, coords):
        coords = np.atleast_2d(coords)
        if self.j is None:
            a = np.repeat(self.vals, len(coords), 0)
        else:
            x = coords[:, self.j]
            a = np.stack([np.interp(x, self.knots, self.vals[:, s]) for s in range(4)], 1)
        return np.c_[a, a[:, :3].mean(1) + self.dE]


def build_frame(run, ref):
    from ..frames import ChannelFrame, Diabat
    fc = run.cfg['frame']
    asy = Asymptotes(ref, _dE_ion(run))
    c = asy.vals.mean(0)
    diabats = [Diabat('CH₂O + O(¹D) a', c[0]), Diabat('CH₂O + O(¹D) b', c[1]), Diabat('CH₂O + O(¹D) c', c[2]),
               Diabat('CH₂O(T₁) + O(³P)', c[3], kind='biradical'),
               Diabat('CH₂O⁺ + O⁻', c[:3].mean() + asy.dE, charge_product=-1.0, kind='ionic')]
    return ChannelFrame(diabats, n_coords=len(ref.coord_names), r_index=0, zeta_env=fc['zeta_env'],
                        r_ref=fc['r_ref'], feature_zetas=tuple(fc['feature_zetas']), asymptote_fn=asy)


def make_model(run, ref, kind):
    from ..models import CDNN, FramedModel
    tc = run.cfg['train']
    if kind == 'framed':
        # couplings start at coupling_init (eV), not 0: with every coupling exactly 0 the matrix is diagonal and the
        # energy gradient with respect to a coupling vanishes, so the coupling network would never learn
        return FramedModel(build_frame(run, ref), 4, hidden=tuple(tc['hidden_framed']),
                           slot_bias={'coup': float(tc.get('coupling_init', 0.1))})
    if kind == 'cdnn':
        return CDNN(len(ref.coord_names), 4, n_matrix=5, hidden=tuple(tc['hidden_cdnn']))
    raise ValueError(kind)


def ionic_weight(model, coords):
    """Weight of the ionic (zwitterion) structure in the model's adiabatic ground state."""
    M = model.matrix(coords).detach().numpy()
    w, V = np.linalg.eigh(M)
    return V[:, ION, 0] ** 2


def charge_target(ref, idx, q_ref, min_gap):
    q = ref.extra['q_O_terminal'][idx, 0]
    t = np.clip(q / q_ref, 0.0, 1.0)
    mask = (ref.energies[idx, 1] - ref.energies[idx, 0]) > min_gap
    return t, mask


def restraint_penalty(weight, target, mask):
    import torch
    t = torch.tensor(target[mask])
    m = torch.tensor(mask)

    def pen(model, cache):
        if weight == 0 or not m.any():
            return torch.zeros(())
        M = model._matrix(cache)[m]
        _, V = torch.linalg.eigh(M)
        return weight * ((V[:, ION, 0] ** 2 - t) ** 2).mean()
    return pen


# ------------------------------------------------------------------------------------------------ train
def stage_train(run):
    import torch
    from ..models import train
    from .. import metrics as Mx
    run.threads()
    ref = load_refs(run)
    tc = run.cfg['train']
    usable = np.all(np.isfinite(ref.energies[:, :4]), 1)
    if tc['exclude_flagged']:
        for k in ('converged', 'active_space_consistent'):
            if k in ref.extra:
                usable &= ref.extra[k].astype(bool)
    U = np.where(usable)[0]
    run.log(f'{ref}; {len(U)} usable points; models {tc["models"]}')
    r = ref.coords[:, 0]
    E = ref.energies[:, :4]
    kw = dict(epochs=int(tc['epochs']), restarts=int(tc['restarts']), alpha2=float(tc['alpha2']))
    lr = {'framed': float(tc['lr_framed']), 'cdnn': float(tc['lr_cdnn'])}
    metrics = {'n_points': int(len(ref)), 'n_usable': int(len(U)), 'coord_names': ref.coord_names,
               'method': ref.meta.get('method', ref.name), 'config': tc}
    x = np.linspace(max(1.15, r.min() - 0.05), r.max(), 400)
    X = cut_coords(ref, x)
    curves = {'x': x, 'cut_values': json.dumps(slice_values(ref))}

    def fit(kind, tr, va, seed, penalty=None):
        torch.manual_seed(seed)
        m = make_model(run, ref, kind)
        t0 = time.time()
        m = train(m, ref.coords[tr], E[tr], ref.coords[va], E[va], lr=lr[kind], penalty=penalty, **kw)
        return m, time.time() - t0

    # ---- random split (interpolation)
    rs = tc['random_split']
    if rs['enabled']:
        res = {k: [] for k in tc['models']}
        for seed in tc['seeds']:
            p = np.random.default_rng(seed).permutation(U)
            ntr, nva = int(rs['train_fraction'] * len(p)), int(rs['val_fraction'] * len(p))
            tr, va, te = p[:ntr], p[ntr:ntr + nva], p[ntr + nva:]
            for kind in tc['models']:
                m, dt = fit(kind, tr, va, seed)
                res[kind].append({'seed': seed, 'test_rmse_kcal': Mx.rmse_kcal(m.predict(ref.coords[te]), E[te]).tolist(),
                                  'val_rmse_kcal': Mx.rmse_kcal(m.predict(ref.coords[va]), E[va]).tolist(),
                                  'n_train': int(len(tr)), 'seconds': dt})
                run.log(f'random split seed {seed} {kind}: test RMSE {np.round(res[kind][-1]["test_rmse_kcal"], 2)} '
                        f'kcal/mol ({dt:.0f} s)')
                if seed == tc['seeds'][0]:
                    torch.save(m, run.p('models', f'random_{kind}.pt'))
                    curves[f'random_{kind}_E'] = m.predict(X)
                    curves[f'random_{kind}_test_pred'] = m.predict(ref.coords[te])
                    curves['random_test_idx'] = te
                    if kind == 'framed':
                        M = m.matrix(X).detach().numpy()
                        curves['random_framed_diag'] = np.diagonal(M, axis1=1, axis2=2)
                        iu = np.triu_indices(5, 1)
                        curves['random_framed_coup'] = M[:, iu[0], iu[1]]
                        curves['random_framed_ionic_weight'] = ionic_weight(m, X)
        metrics['random_split'] = res
    # ---- extrapolation
    ex = tc['extrapolation']
    if ex['enabled']:
        inner = U[r[U] <= ex['r_train_max']]
        outer = U[r[U] > ex['r_train_max']]
        res = {k: [] for k in tc['models']}
        for seed in tc['seeds']:
            p = np.random.default_rng(seed).permutation(inner)
            tr, va = p[:int(0.8 * len(p))], p[int(0.8 * len(p)):]
            for kind in tc['models']:
                m, dt = fit(kind, tr, va, seed)
                res[kind].append({'seed': seed, 'outer_rmse_kcal': Mx.rmse_kcal(m.predict(ref.coords[outer]), E[outer]).tolist(),
                                  'inner_val_rmse_kcal': Mx.rmse_kcal(m.predict(ref.coords[va]), E[va]).tolist(),
                                  'seconds': dt})
                run.log(f'extrapolation seed {seed} {kind}: beyond {ex["r_train_max"]} Å RMSE '
                        f'{np.round(res[kind][-1]["outer_rmse_kcal"], 1)} kcal/mol')
                if seed == tc['seeds'][0]:
                    torch.save(m, run.p('models', f'extrap_{kind}.pt'))
                    curves[f'extrap_{kind}_E'] = m.predict(X)
        metrics['extrapolation'] = dict(res, r_train_max=ex['r_train_max'], n_inner=int(len(inner)),
                                        n_outer=int(len(outer)))
    # ---- learning curve
    lc = tc['learning_curve']
    if lc['enabled']:
        p = np.random.default_rng(12345).permutation(U)
        nte = int(lc['test_fraction'] * len(p))
        te, pool = p[:nte], p[nte:]
        res = {k: {} for k in tc['models']}
        for n in lc['n_train']:
            nva = max(5, min(n // 5, len(pool) - n))
            if n + nva > len(pool):
                run.log(f'learning curve: skipping n={n} (only {len(pool)} points outside the test set)')
                continue
            for kind in tc['models']:
                res[kind][str(n)] = []
                for seed in tc['seeds']:
                    q = np.random.default_rng(1000 + seed).permutation(pool)
                    m, _ = fit(kind, q[:n], q[n:n + nva], seed)
                    res[kind][str(n)].append(Mx.rmse_kcal(m.predict(ref.coords[te]), E[te]).tolist())
                run.log(f'learning curve n={n} {kind}: mean test RMSE '
                        f'{np.mean(res[kind][str(n)]):.2f} kcal/mol')
        metrics['learning_curve'] = dict(res, n_test=int(nte))
    # ---- diabatic restraint
    rc = tc['restraint']
    if rc['enabled'] and 'framed' in tc['models']:
        if 'q_O_terminal' not in ref.extra:
            run.log('restraint skipped: the reference data have no q_O_terminal')
        else:
            res = {}
            eqv = cut_mask(ref) & usable
            for wgt in rc['weights']:
                res[str(wgt)] = []
                for seed in tc['seeds']:
                    p = np.random.default_rng(seed).permutation(U)
                    ntr = int(tc['random_split']['train_fraction'] * len(p))
                    nva = int(tc['random_split']['val_fraction'] * len(p))
                    tr, va, te = p[:ntr], p[ntr:ntr + nva], p[ntr + nva:]
                    tt, mt = charge_target(ref, tr, rc['q_ref'], rc['min_gap_eV'])
                    m, dt = fit('framed', tr, va, seed, penalty=restraint_penalty(float(wgt), tt, mt))
                    te_t, te_m = charge_target(ref, te, rc['q_ref'], rc['min_gap_eV'])
                    iw = ionic_weight(m, ref.coords[te])
                    res[str(wgt)].append({
                        'seed': seed, 'test_rmse_kcal': Mx.rmse_kcal(m.predict(ref.coords[te]), E[te]).tolist(),
                        'ionic_weight_mae': float(np.mean(np.abs(iw[te_m] - te_t[te_m]))) if te_m.any() else None,
                        'ionic_weight_at_eq': float(ionic_weight(m, cut_coords(ref, np.array([1.35])))[0])})
                    if seed == tc['seeds'][0]:
                        curves[f'restraint_{wgt}_ionic_weight'] = ionic_weight(m, X)
                        torch.save(m, run.p('models', f'restraint_w{wgt}.pt'))
                run.log(f'restraint weight {wgt}: energy RMSE '
                        f'{np.round(np.mean([v["test_rmse_kcal"] for v in res[str(wgt)]], 0), 2)} kcal/mol, '
                        f'ionic-weight MAE {np.mean([v["ionic_weight_mae"] or np.nan for v in res[str(wgt)]]):.3f}')
            metrics['restraint'] = dict(res, q_ref=rc['q_ref'], min_gap_eV=rc['min_gap_eV'],
                                        target_cut=np.clip(ref.extra['q_O_terminal'][eqv, 0] / rc['q_ref'], 0, 1).tolist(),
                                        target_cut_r=r[eqv].tolist())
    run.p('results', 'metrics.json').write_text(json.dumps(metrics, indent=1))
    np.savez(run.p('results', 'curves.npz'), **curves)
    run.log(f'wrote results/metrics.json, results/curves.npz and models/*.pt')


# ------------------------------------------------------------------------------------------------ figures
INK, BLUE, ORANGE, GREEN, PLUM, COPPER, GREY = '#2E2A36', '#2a78d6', '#eb6834', '#1baf7a', '#4E3A6E', '#C8873A', '#9A94A0'
SCOL = [INK, BLUE, ORANGE, GREEN]
MCOL = {'framed': GREEN, 'cdnn': ORANGE}
MNAME = {'framed': 'framed CDNN', 'cdnn': 'CDNN'}


def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 200, 'font.size': 12, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'legend.frameon': False})
    return plt


def stage_figures(run):
    plt = _mpl()
    ref = load_refs(run)
    r = ref.coords[:, 0]
    E = ref.energies
    sv = slice_values(ref)
    cm = cut_mask(ref)
    cap = {}
    note = ', '.join(f'{LABEL[k]} = {v:g}{UNITS[k]}' for k, v in sv.items())
    method = ref.meta.get('method', ref.name)
    flag = ~(ref.extra.get('converged', np.ones(len(ref), bool)).astype(bool)
             & ref.extra.get('active_space_consistent', np.ones(len(ref), bool)).astype(bool))

    def save(fig, name, caption):
        fig.tight_layout()
        fig.savefig(run.p('figures', name), bbox_inches='tight')
        plt.close(fig)
        cap[name] = caption
        run.log(f'figures/{name}')

    # F1 cuts at every angle
    ja = ref.coord_names.index('a_COO') if 'a_COO' in ref.coord_names else None
    angles = np.unique(ref.coords[:, ja]) if ja is not None else [None]
    base = {k: v for k, v in sv.items() if k != 'a_COO'}
    ncol = min(3, len(angles)); nrow = int(np.ceil(len(angles) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.8 * nrow), squeeze=False)
    for ax, a in zip(axes.flat, angles):
        vals = dict(base, **({'a_COO': a} if a is not None else {}))
        m = cut_mask(ref, vals)
        o = np.argsort(r[m])
        for s in range(min(4, ref.n_states)):
            ax.plot(r[m][o], E[m][o, s], '-o', ms=3, color=SCOL[s], label=f'S{s}')
        bad = m & flag
        if bad.any():
            ax.plot(r[bad], E[bad, 0], 'x', color='red', ms=8, label='flagged')
        ax.set(xlabel='r(O–O) (Å)', ylabel='E (eV)', title=f'∠COO = {a:g}°' if a is not None else '', xlim=(1.1, 5.2))
    for ax in list(axes.flat)[len(angles):]:
        ax.axis('off')
    axes.flat[0].legend(ncol=2, fontsize=10)
    save(fig, 'F1_ref_cuts.png', f'Reference energies of the four singlet A′ states along r(O–O) at each C–O–O angle '
         f'({method}; energies relative to CH₂O + O(¹D)). Red crosses mark unconverged points or points whose active '
         f'space changed character.')

    # F2 character
    o = np.argsort(r[cm])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for s in range(min(4, ref.n_states)):
        axes[0].plot(r[cm][o], E[cm][o, s], '-o', ms=3, color=SCOL[s], label=f'S{s}')
        if 'q_O_terminal' in ref.extra:
            axes[1].plot(r[cm][o], ref.extra['q_O_terminal'][cm][o, s], '-o', ms=3, color=SCOL[s])
            axes[2].plot(r[cm][o], ref.extra['n_unpaired'][cm][o, s], '-o', ms=3, color=SCOL[s])
    axes[0].set(xlabel='r(O–O) (Å)', ylabel='E (eV)', title='Energies', xlim=(1.1, 5.2))
    axes[1].set(xlabel='r(O–O) (Å)', ylabel='Mulliken charge, terminal O (e)', title='Charge separation', xlim=(1.1, 5.2))
    axes[2].set(xlabel='r(O–O) (Å)', ylabel='effective unpaired electrons', title='Open-shell character', xlim=(1.1, 5.2))
    axes[0].legend(ncol=2)
    save(fig, 'F2_ref_character.png', f'State character along r(O–O) at {note}: energy, terminal-O charge (zwitterion '
         f'proxy) and effective unpaired electrons (biradical proxy), each from the state\'s own density.')

    # F3 maps over (r_OO, angle)
    if ja is not None and len(angles) > 2:
        m = cut_mask(ref, base) if base else np.ones(len(ref), bool)
        rr, aa = np.unique(r[m]), np.unique(ref.coords[m, ja])
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
        for s, ax in enumerate(axes):
            Z = np.full((len(aa), len(rr)), np.nan)
            for i in np.where(m)[0]:
                Z[np.searchsorted(aa, ref.coords[i, ja]), np.searchsorted(rr, r[i])] = E[i, s]
            pc = ax.pcolormesh(rr, aa, Z, shading='nearest', cmap='viridis')
            ax.contour(rr, aa, Z, levels=10, colors='white', linewidths=0.5)
            fig.colorbar(pc, ax=ax, label='E (eV)')
            ax.set(xlabel='r(O–O) (Å)', title=f'S{s}', xscale='log')
            ax.grid(False)
        axes[0].set_ylabel('∠C–O–O (°)')
        save(fig, 'F3_ref_maps.png', 'Each state over the planar (r(O–O), ∠C–O–O) grid' +
             (f' at {", ".join(f"{LABEL[k]} = {v:g}{UNITS[k]}" for k, v in base.items())}' if base else '') +
             '; log scale in r to show the short-range region.')

    # F3b r_CO dependence
    if 'r_CO' in ref.coord_names:
        jc = ref.coord_names.index('r_CO')
        rcs = np.unique(ref.coords[:, jc])
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
        cmap = plt.get_cmap('plasma')
        for k, rc in enumerate(rcs):
            vals = dict(sv, r_CO=rc)
            m = cut_mask(ref, vals); o = np.argsort(r[m])
            for ax, s in zip(axes, (0, 1)):
                ax.plot(r[m][o], E[m][o, s], '-o', ms=3, color=cmap(k / max(1, len(rcs) - 1)), label=f'r(C–O) = {rc:g} Å')
        for ax, s in zip(axes, (0, 1)):
            ax.set(xlabel='r(O–O) (Å)', ylabel='E (eV)', title=f'S{s}', xlim=(1.1, 5.2))
        axes[0].legend()
        save(fig, 'F3b_ref_rCO.png', 'How the C–O1 bond length changes the two lowest states along r(O–O) '
             '(at the equilibrium C–O–O angle): the zwitterion and biradical structures prefer different C–O lengths.')

    # F4 dynamic correlation
    if 'E_nevpt2_eV' in ref.extra:
        Ec, En = ref.extra['E_casscf_eV'][cm][o], ref.extra['E_nevpt2_eV'][cm][o]
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
        for s in range(4):
            axes[0].plot(r[cm][o], Ec[:, s], '--', color=SCOL[s], lw=1.5)
            axes[0].plot(r[cm][o], En[:, s], '-o', ms=3, color=SCOL[s], label=f'S{s}')
            axes[1].plot(r[cm][o], En[:, s] - Ec[:, s], '-o', ms=3, color=SCOL[s], label=f'S{s}')
        axes[0].set(xlabel='r(O–O) (Å)', ylabel='E (eV)', title='SC-NEVPT2 (solid) vs SA-CASSCF (dashed)', xlim=(1.1, 5.2))
        axes[1].set(xlabel='r(O–O) (Å)', ylabel='E(NEVPT2) − E(CASSCF) (eV)', title='Dynamic-correlation shift', xlim=(1.1, 5.2))
        axes[0].legend(ncol=2)
        save(fig, 'F4_dynamic_correlation.png', f'Effect of dynamic correlation along r(O–O) at {note}. Each level is '
             'zeroed at its own CH₂O + O(¹D) asymptote, so the right panel shows how binding changes. State-specific '
             'NEVPT2 can be unreliable where states nearly cross; look for kinks there.')

    mpath = run.p('results', 'metrics.json')
    if not mpath.exists():
        run.p('figures', 'captions.json').write_text(json.dumps(cap, indent=1, ensure_ascii=False))
        run.log('no results/metrics.json yet: model figures skipped (run train)'); return
    M = json.loads(mpath.read_text())
    cv = np.load(run.p('results', 'curves.npz'))
    x = cv['x']
    models = M['config']['models']

    # F5 parity
    if 'random_test_idx' in cv.files:
        te = cv['random_test_idx']
        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5.4), squeeze=False)
        for ax, k in zip(axes[0], models):
            P = cv[f'random_{k}_test_pred']
            for s in range(4):
                ax.plot(E[te, s], P[:, s], 'o', ms=4, color=SCOL[s], alpha=0.8, label=f'S{s}')
            lo, hi = np.nanmin(E[te, :4]), np.nanmax(E[te, :4])
            ax.plot([lo, hi], [lo, hi], color=GREY, lw=1)
            rm = np.mean([v['test_rmse_kcal'] for v in M['random_split'][k]], 0)
            ax.set(xlabel='reference E (eV)', ylabel='model E (eV)',
                   title=f'{MNAME[k]}: test RMSE ' + ' / '.join(f'{v:.1f}' for v in rm) + ' kcal/mol')
            ax.legend()
        save(fig, 'F5_parity.png', 'Held-out test points (random split, first seed): model against reference for each '
             'state. Titles give the test RMSE per state, averaged over seeds.')

    # F6 extrapolation
    if 'extrapolation' in M:
        rmax = M['extrapolation']['r_train_max']
        fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5), squeeze=False)
        o = np.argsort(r[cm])
        for ax, k in zip(axes[0], models):
            Pk = cv[f'extrap_{k}_E']
            ax.axvspan(rmax, x.max(), color=COPPER, alpha=0.12, lw=0)
            for s in range(4):
                ax.plot(r[cm][o], E[cm][o, s], 'o', color=INK, ms=4, mfc='white' if s else INK,
                        label='reference' if s == 0 else None)
                ax.plot(x, Pk[:, s], color=MCOL[k], lw=2, label=MNAME[k] if s == 0 else None)
            rm = np.mean([v['outer_rmse_kcal'] for v in M['extrapolation'][k]], 0)
            ax.set(xlabel='r(O–O) (Å)', ylabel='E (eV)', ylim=(np.nanmin(E[:, :4]) - 1, np.nanmax(E[cm, :4]) + 1),
                   title=f'{MNAME[k]}: beyond {rmax} Å RMSE ' + ' / '.join(f'{v:.0f}' for v in rm) + ' kcal/mol')
            ax.legend(loc='lower right')
        save(fig, 'F6_extrapolation.png', f'Trained only on r(O–O) ≤ {rmax} Å (shaded: no training data), drawn at '
             f'{note}. The frame fixes the asymptotes and the ion-pair Coulomb tail; plain CDNN has to guess them.')

    # F7 learned matrix
    if 'random_framed_diag' in cv.files:
        D, U, Ead = cv['random_framed_diag'], cv['random_framed_coup'], cv['random_framed_E']
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        kc = {0: BLUE, 1: BLUE, 2: BLUE, 3: GREEN, 4: ORANGE}
        lab = {0: 'CH₂O + O(¹D) (3 components)', 3: 'CH₂O(T₁) + O(³P)', 4: 'CH₂O⁺ + O⁻ (zwitterion)'}
        for i in range(5):
            axes[0].plot(x, D[:, i], '--', lw=2.4, color=kc[i], label=lab.get(i))
        for s in range(4):
            axes[0].plot(x, Ead[:, s], color=GREY, lw=1.4, label='adiabatic (model)' if s == 0 else None)
        o = np.argsort(r[cm])
        axes[0].plot(r[cm][o], E[cm][o, 0], 'o', color=INK, ms=3, mfc='white', label='reference S0')
        axes[0].set(xlabel='r(O–O) (Å)', ylabel='E (eV)', xlim=(x.min(), 5), ylim=(np.nanmin(E[:, :4]) - 1.5, 11),
                    title='Diagonal elements (structures)')
        axes[0].legend(fontsize=10)
        iu = np.triu_indices(5, 1)
        big = np.max(np.abs(U), 0)
        for c in np.argsort(-big)[:6]:
            i, j = iu[0][c], iu[1][c]
            axes[1].plot(x, np.abs(U[:, c]) * 1000, lw=1.6, label=f'|Ω{i}{j}|' + (' (ionic)' if j == ION else ''))
        axes[1].set_ylabel('|coupling| (meV)')
        axes[1].set(xlabel='r(O–O) (Å)', xlim=(x.min(), 5), title='Six largest off-diagonal elements')
        axes[1].legend(fontsize=9, ncol=2)
        save(fig, 'F7_learned_matrix.png', f'The framed model\'s matrix along r(O–O) at {note}: structures (dashed) '
             'behind the adiabatic states, and the learned couplings. Energies alone do not fix how much the '
             'structures mix (see F9).')

    # F8 learning curve
    if 'learning_curve' in M:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for k in models:
            ns = sorted(int(n) for n in M['learning_curve'][k])
            if not ns:
                continue
            mean = [np.mean(M['learning_curve'][k][str(n)]) for n in ns]
            per_seed = [np.mean(M['learning_curve'][k][str(n)], 1) for n in ns]
            lo = [m - np.min(p) for m, p in zip(mean, per_seed)]
            hi = [np.max(p) - m for m, p in zip(mean, per_seed)]
            ax.errorbar(ns, mean, yerr=[lo, hi], marker='o', color=MCOL[k], label=MNAME[k], capsize=4, lw=2)
        ax.set(xlabel='training points', ylabel='test RMSE, mean of 4 states (kcal/mol)', xscale='log', yscale='log',
               title=f'Learning curve ({M["learning_curve"]["n_test"]} fixed test points)')
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        save(fig, 'F8_learning_curve.png', 'Test error against the number of training points; bars span the seeds.')

    # F9 restraint
    if 'restraint' in M:
        R = M['restraint']
        ws = [w for w in R if w not in ('q_ref', 'min_gap_eV', 'target_cut', 'target_cut_r')]
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        cmap = plt.get_cmap('viridis')
        for k, w in enumerate(ws):
            key = f'restraint_{w}_ionic_weight'
            if key in cv.files:
                axes[0].plot(x, cv[key], color=cmap(k / max(1, len(ws) - 1)), lw=2, label=f'weight {w}')
        axes[0].plot(R['target_cut_r'], R['target_cut'], 'o', color=INK, ms=5, label=f'target: q / ({R["q_ref"]} e)')
        axes[0].set(xlabel='r(O–O) (Å)', ylabel='ionic weight in ground state', xlim=(x.min(), 5), ylim=(-0.05, 1.05),
                    title='Zwitterion character of S0')
        axes[0].legend(fontsize=10)
        e = [np.mean([np.mean(v['test_rmse_kcal']) for v in R[w]]) for w in ws]
        iw = [np.mean([v['ionic_weight_mae'] for v in R[w] if v['ionic_weight_mae'] is not None]) for w in ws]
        axes[1].plot(iw, e, '-o', color=PLUM)
        for w, a, b in zip(ws, iw, e):
            axes[1].annotate(f' {w}', (a, b))
        axes[1].set(xlabel='ionic-weight error vs charge target (MAE)', ylabel='energy test RMSE, mean of states (kcal/mol)',
                    title='Trade-off as the restraint weight grows')
        save(fig, 'F9_restraint.png', 'Diabatic restraint: the ground state\'s weight on the CH₂O⁺ + O⁻ structure is '
             f'pulled toward the CASSCF terminal-O charge divided by q_ref = {R["q_ref"]} e (an assumption to vary). '
             'Left: along r(O–O); right: energy accuracy against agreement with the charge target.')
    run.p('figures', 'captions.json').write_text(json.dumps(cap, indent=1, ensure_ascii=False))


# ------------------------------------------------------------------------------------------------ tables
def _write_table(run, name, header, rows, title):
    with open(run.p('tables', f'{name}.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    md = [f'**{title}**', '', '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    md += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    run.p('tables', f'{name}.md').write_text('\n'.join(md) + '\n')
    run.log(f'tables/{name}.csv')
    return {'name': name, 'title': title, 'header': header, 'rows': rows}


def _f(x, d=2):
    return '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


def stage_tables(run):
    ref = load_refs(run)
    r = ref.coords[:, 0]
    tables = []
    ex = ref.extra
    conv = ex.get('converged', np.ones(len(ref), bool)).astype(bool)
    cons = ex.get('active_space_consistent', np.ones(len(ref), bool)).astype(bool)
    far = np.isclose(r, r.max()) & cut_mask(ref, {k: v for k, v in slice_values(ref).items() if k != 'a_COO'})
    rows = [['method', ref.meta.get('method', ref.name)], ['energies used for training', ref.meta.get('energy_level', 'casscf')],
            ['geometries', len(ref)],
            ['coordinates', ', '.join(f'{n} {np.min(ref.coords[:, j]):g}–{np.max(ref.coords[:, j]):g}' for j, n in enumerate(ref.coord_names))],
            ['fixed', ', '.join(f'{k} = {v:g}' for k, v in ref.meta.get('fixed', {}).items())],
            ['converged', f'{conv.sum()} / {len(ref)}'], ['consistent active space', f'{cons.sum()} / {len(ref)}'],
            ['O(¹D) components at largest r_OO: spread (meV)', _f(1000 * float(np.nanmax(np.ptp(ref.energies[far, :3], 1))) if far.any() else None, 1)],
            ['S3 asymptote (eV)', _f(float(np.nanmean(ref.energies[far, 3])) if far.any() else None, 3)]]
    if 'seconds' in ex:
        rows += [['mean time per geometry (s)', _f(float(np.mean(ex['seconds'])), 1)],
                 ['total compute (h)', _f(float(np.sum(ex['seconds'])) / 3600, 2)]]
    if 'casci_mismatch_hartree' in ex:
        rows += [['largest CASCI vs SA-CASSCF energy difference before NEVPT2 (µEh)', _f(1e6 * float(np.nanmax(ex['casci_mismatch_hartree'])), 2)]]
    cm = cut_mask(ref); o = np.argsort(r[cm])
    s0 = ref.energies[cm][o, 0]
    rows += [['S0 well depth along the cut (eV)', _f(float(np.nanmax(s0[-1:]) - np.nanmin(s0)), 3)],
             ['S0 minimum at r_OO (Å)', _f(float(r[cm][o][np.nanargmin(s0)]), 3)]]
    tables.append(_write_table(run, 'T1_reference_summary', ['quantity', 'value'], rows, 'T1. Reference data'))

    lp = run.p('data', 'ladder.json')
    if lp.exists():
        L = json.loads(lp.read_text())
        X, Cc = L['experiment'], L.get('computed', {})
        rows = [['IE(CH₂O) (eV)', _f(X['IE_CH2O_adiabatic'], 3) + ' (adiabatic)', _f(Cc.get('IE_CH2O_vertical'), 3) + (' (vertical)' if Cc else '')],
                ['EA(O) (eV)', _f(X['EA_O'], 3), _f(Cc.get('EA_O'), 3)],
                ['O ¹D – ³P (eV)', _f(X['O_1D_3P'], 3), _f(Cc.get('O_1D_3P'), 3) + (' (experiment)' if Cc else '')],
                ['ΔE(ion pair) = IE − EA − ΔE(¹D) (eV)', _f(L['dE_ion_experiment'], 3), _f(Cc.get('dE_ion'), 3)],
                ['Coulomb crossing r* = 14.40 / ΔE (Å)', _f(14.399645 / L['dE_ion_experiment'], 2), _f(14.399645 / Cc['dE_ion'], 2) if Cc else '']]
        tables.append(_write_table(run, 'T2_ladder', ['quantity', 'experiment', f'computed ({Cc.get("basis", "not run")})'], rows,
                                   f'T2. Ladder anchors for the ion-pair structure (frame uses: {L["use"]})'))

    mp = run.p('results', 'metrics.json')
    if mp.exists():
        M = json.loads(mp.read_text())
        models = M['config']['models']
        rows = []
        def add(test, k, vals):
            a = np.array(vals)
            rows.append([test, MNAME[k]] + [f'{m:.2f} ± {s:.2f}' for m, s in zip(a.mean(0), a.std(0))] + [len(a)])
        if 'random_split' in M:
            for k in models:
                add('random split: test', k, [v['test_rmse_kcal'] for v in M['random_split'][k]])
        if 'extrapolation' in M:
            rt = M['extrapolation']['r_train_max']
            for k in models:
                add(f'trained r ≤ {rt} Å: validation', k, [v['inner_val_rmse_kcal'] for v in M['extrapolation'][k]])
            for k in models:
                add(f'trained r ≤ {rt} Å: beyond {rt} Å', k, [v['outer_rmse_kcal'] for v in M['extrapolation'][k]])
        if rows:
            tables.append(_write_table(run, 'T3_model_errors', ['test', 'model', 'S0', 'S1', 'S2', 'S3', 'seeds'], rows,
                                       'T3. RMSE (kcal/mol), mean ± s.d. over seeds'))
        if 'learning_curve' in M:
            rows = []
            for k in models:
                for n in sorted(M['learning_curve'][k], key=int):
                    a = np.mean(M['learning_curve'][k][n], 1)
                    rows.append([n, MNAME[k], f'{a.mean():.2f} ± {a.std():.2f}'] +
                                [f'{v:.2f}' for v in np.mean(M['learning_curve'][k][n], 0)])
            tables.append(_write_table(run, 'T4_learning_curve', ['n train', 'model', 'mean RMSE', 'S0', 'S1', 'S2', 'S3'],
                                       rows, f'T4. Learning curve, test RMSE (kcal/mol) on {M["learning_curve"]["n_test"]} fixed points'))
        if 'restraint' in M:
            R = M['restraint']
            rows = []
            for w in [w for w in R if w not in ('q_ref', 'min_gap_eV', 'target_cut', 'target_cut_r')]:
                e = np.mean([v['test_rmse_kcal'] for v in R[w]], 0)
                iw = [v['ionic_weight_mae'] for v in R[w] if v['ionic_weight_mae'] is not None]
                rows.append([w] + [f'{v:.2f}' for v in e] + [_f(float(np.mean(iw)), 3) if iw else '',
                                                             _f(float(np.mean([v['ionic_weight_at_eq'] for v in R[w]])), 2)])
            tables.append(_write_table(run, 'T5_restraint', ['weight', 'S0', 'S1', 'S2', 'S3', 'ionic-weight MAE',
                                                             'ionic weight at 1.35 Å'], rows,
                                       f'T5. Diabatic restraint (q_ref = {R["q_ref"]} e): test RMSE (kcal/mol) and agreement with the charge target'))
    if 'E_nevpt2_eV' in ex:
        rows = []
        Ec, En = ex['E_casscf_eV'][cm][o], ex['E_nevpt2_eV'][cm][o]
        i_eq = int(np.argmin(np.abs(r[cm][o] - 1.35)))
        for s in range(4):
            rows.append([f'S{s}', _f(float(Ec[i_eq, s]), 3), _f(float(En[i_eq, s]), 3), _f(float(En[i_eq, s] - Ec[i_eq, s]), 3)])
        rows.append(['S0 well depth', _f(float(Ec[-1, 0] - np.nanmin(Ec[:, 0])), 3), _f(float(En[-1, 0] - np.nanmin(En[:, 0])), 3),
                     _f(float((En[-1, 0] - np.nanmin(En[:, 0])) - (Ec[-1, 0] - np.nanmin(Ec[:, 0]))), 3)])
        tables.append(_write_table(run, 'T6_dynamic_correlation', ['state (near r_OO = 1.35 Å)', 'SA-CASSCF (eV)',
                                                                   'SC-NEVPT2 (eV)', 'shift (eV)'], rows,
                                   'T6. Dynamic correlation along the equilibrium cut (each level zeroed at its own asymptote)'))
    run.p('tables', 'tables.json').write_text(json.dumps(tables, indent=1, ensure_ascii=False))


# ------------------------------------------------------------------------------------------------ report
def stage_report(run):
    import html
    cfg_text = run.p('config.toml').read_text()
    caps = json.loads(run.p('figures', 'captions.json').read_text()) if run.p('figures', 'captions.json').exists() else {}
    tabs = json.loads(run.p('tables', 'tables.json').read_text()) if run.p('tables', 'tables.json').exists() else []
    esc = html.escape
    parts = []
    for t in tabs:
        head = ''.join(f'<th>{esc(str(h))}</th>' for h in t['header'])
        body = ''.join('<tr>' + ''.join(f'<td>{esc(str(c))}</td>' for c in row) + '</tr>' for row in t['rows'])
        parts.append(f'<h3>{esc(t["title"])}</h3><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
                     f'<p class="src">tables/{t["name"]}.csv</p>')
    figs = ''.join(f'<figure><img src="figures/{esc(n)}" alt="{esc(c)}"><figcaption><b>{esc(n.split("_")[0])}.</b> '
                   f'{esc(c)}</figcaption></figure>' for n, c in caps.items())
    name = run.cfg['run'].get('name') or run.path.name
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CH2OO run: {esc(name)}</title>
<style>
:root{{--paper:#F6F4F0;--ink:#2E2A36;--muted:#7A7480;--rule:#DAD3C8;--plum:#3D2C58;--copper:#C8873A;--panel:#EFEBF3}}
@media (prefers-color-scheme:dark){{:root{{--paper:#17131f;--ink:#ece8f2;--muted:#a69cb6;--rule:#3d3250;--panel:#241b33}}}}
body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,-apple-system,sans-serif}}
header{{background:var(--plum);color:#fff;border-bottom:5px solid var(--copper);padding:28px 24px}}
header h1{{margin:0;font:600 30px Georgia,serif}} header p{{margin:6px 0 0;color:#E9B872}}
main{{max-width:1180px;margin:0 auto;padding:16px}}
h2{{font:600 24px Georgia,serif;border-bottom:1px solid var(--rule);padding-bottom:4px;margin-top:36px}}
table{{border-collapse:collapse;width:100%;margin:8px 0;font-size:14px}} th,td{{border-bottom:1px solid var(--rule);padding:6px 8px;text-align:left}}
th{{background:var(--panel)}} figure{{margin:22px 0}} img{{max-width:100%;background:#fff;border:1px solid var(--rule)}}
figcaption,.src{{color:var(--muted);font-size:14px}} pre{{background:var(--panel);padding:12px;overflow:auto;font-size:13px}}
</style></head><body>
<header><h1>CH₂OO workflow run: {esc(name)}</h1><p>{esc(run.cfg['run'].get('description', ''))} · {time.strftime('%Y-%m-%d %H:%M')}</p></header>
<main><h2>Tables</h2>{''.join(parts) or '<p>No tables yet.</p>'}
<h2>Figures</h2>{figs or '<p>No figures yet.</p>'}
<h2>Configuration</h2><pre>{esc(cfg_text)}</pre></main></body></html>'''
    run.p('report.html').write_text(page)
    run.log(f'wrote {run.p("report.html")}')


# ------------------------------------------------------------------------------------------------ cli
STAGES = ['init', 'refs', 'status', 'ladder', 'train', 'figures', 'tables', 'report', 'all']


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m diabml.workflow.ch2oo', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=STAGES)
    ap.add_argument('run', help='run folder (holds config.toml)')
    ap.add_argument('--template', default='quick', help=f'for init: {", ".join(C.TEMPLATES)}')
    ap.add_argument('--force', action='store_true', help='refs: recompute even if data/refs.npz exists')
    a = ap.parse_args(argv)
    if a.stage == 'init':
        return stage_init(a.run, a.template)
    run = Run(a.run)
    if a.stage == 'status':
        return stage_status(run)
    todo = ['refs', 'ladder', 'train', 'figures', 'tables', 'report'] if a.stage == 'all' else [a.stage]
    for st in todo:
        run.open_log(st)
        t0 = time.time()
        {'refs': lambda: stage_refs(run, a.force), 'ladder': lambda: stage_ladder(run), 'train': lambda: stage_train(run),
         'figures': lambda: stage_figures(run), 'tables': lambda: stage_tables(run),
         'report': lambda: stage_report(run)}[st]()
        run.log(f'{st} done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
