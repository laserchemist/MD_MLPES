"""Dipoles at the reference points (finite fields, PySCF) and the dipole surface fitted to them.

    python -m diabml.workflow.ch2oo dipoles    RUN [--jobs 6]   # -> RUN/data/dipoles.npz   (~30 min, 1000 points)
    python -m diabml.workflow.ch2oo dipole_fit RUN              # -> RUN/models/dipole.pt   (~1 min)

dipoles     For every point of RUN/data/refs.npz: SA-CASSCF + SC-NEVPT2 with PySCF, field-free and with +-F along x
            and y (the molecular plane), giving the relaxed dipole of every state (diabml/refs/finite_field.py).
            The field-free SA-CASSCF energies must equal the stored ones (within tol_uEh): that proves PySCF found
            the same SA-CASSCF solution as the data (ORCA's, for repaired and ORCA-computed points). A point that
            does not match is retried from the orbitals of its nearest matched neighbour. Parallel (`jobs`
            processes, one thread each), resumable (data/dipoles_checkpoint.npz).
dipole_fit  The ground state's dipole (the state with the lowest energy at the data's level), in the molecule's
            frame, fitted with learned atomic charges (diabml/dipole.py) on every matched point where S0 is more
            than gap_mEh from S1 (finite differences of two nearly degenerate states are not meaningful).
            Writes models/dipole.pt (+ .json), results/dipole_fit.json and figures D1 (dipole of the reference
            points along r_OO) and D2 (parity of the fit on held-out points).
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import time

import numpy as np

from . import ch2oo as W

HARTREE_EV = 27.211386245988
AU_DEBYE = 2.541746
AU_EA = 0.529177210903


def _worker_init():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'


def _dip_worker(task):
    i, atoms, spec, F, guess = task
    _worker_init()
    from pyscf import gto, lib
    lib.num_threads(1)
    from ..refs.finite_field import state_dipoles
    g = None
    if guess is not None:
        mo, gatoms = guess
        gmol = gto.M(atom=[(a, tuple(x)) for a, x in gatoms], basis=spec['basis'],
                     symmetry=spec.get('symmetry') or False, verbose=0, unit='Angstrom')
        g = (mo, gmol)
    t0 = time.time()
    try:
        r = state_dipoles(atoms, spec, F=F, guess=g)
    except Exception as exc:                                  # a failed point is recorded, not fatal
        return i, {'error': f'{exc.__class__.__name__}: {exc}', 'seconds': time.time() - t0}
    r.pop('mol')
    r['seconds'] = time.time() - t0
    return i, r


def stage_dipoles(run, jobs=None):
    from ..geometry import ch2oo_xyz
    from ..refset import RefSet
    dc = run.cfg['dipoles']
    jobs = int(jobs or dc['jobs'])
    ref = RefSet.load(run.p('data', 'refs.npz'))
    spec, zero = ref.meta['spec'], ref.meta['zero_hartree']
    nr = int(spec['nroots'])
    Ec = zero['casscf'] + np.asarray(ref.extra['E_casscf_eV'], float) / HARTREE_EV
    N = len(ref)
    pts = [i for i in range(N) if np.all(np.isfinite(Ec[i, :nr]))]
    tol = float(dc['tol_uEh']) * 1e-6
    F = float(dc['field_au'])
    ck = run.p('data', 'dipoles_checkpoint.npz')
    keys = ('e_cas', 'e_pt', 'mu_cas', 'mu_pt', 'mu_sa')
    res = {}
    if ck.exists():
        z = np.load(ck, allow_pickle=False)
        for j, i in enumerate(z['done']):
            res[int(i)] = {k: z[k][j] for k in keys + ('min_gap', 'seconds', 'mo')}
            res[int(i)]['converged'] = bool(z['converged'][j])
        run.log(f'resuming: {len(res)} points from {ck.name}')
    atoms = {i: [(a, np.asarray(x, float).tolist()) for a, x in ch2oo_xyz(**W._geometry_kwargs(ref, ref.coords[i]))]
             for i in pts}

    def matches(i):
        r = res.get(i)
        return r is not None and 'e_cas' in r and np.max(np.abs(np.sort(r['e_cas']) - np.sort(Ec[i, :nr]))) <= tol

    def save():
        done = sorted(i for i in res if 'e_cas' in res[i])
        if not done:
            return
        np.savez(ck, done=np.array(done), converged=np.array([res[i]['converged'] for i in done]),
                 **{k: np.array([res[i][k] for i in done]) for k in keys + ('min_gap', 'seconds', 'mo')})

    def batch(tasks, label):
        t0, n = time.time(), 0
        with cf.ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init) as ex:
            for fut in cf.as_completed([ex.submit(_dip_worker, t) for t in tasks]):
                i, r = fut.result()
                n += 1
                if 'error' in r:
                    run.log(f'  point {i}: {r["error"]}')
                else:
                    old = res.get(i)
                    # keep the new result if the old one did not match (a retry), never replace a match
                    if old is None or not matches(i):
                        res[i] = r
                if n % 25 == 0 or n == len(tasks):
                    el = time.time() - t0
                    run.log(f'  {label}: {n}/{len(tasks)} points, {el / 60:.1f} min, about '
                            f'{el / n * (len(tasks) - n) / 60:.0f} min left')
                    save()
        save()

    run.log(f'{len(pts)} points; SA{nr}-CASSCF({spec["nelecas"]},{spec["ncas"]})/{spec["basis"]} + SC-NEVPT2 in '
            f'fields of +-{F} a.u. along x and y (PySCF); same solution if every root is within {dc["tol_uEh"]} µEh '
            f'of the data; {jobs} processes')
    todo = [i for i in pts if i not in res]
    if todo:
        batch([(i, atoms[i], spec, F, None) for i in todo], 'pass 1')
    sc = np.array([{'r_OO': 0.1, 'a_COO': 10.0, 'r_CO': 0.05}.get(n, 1.0) for n in ref.coord_names])
    for k in range(1, int(dc['retries']) + 1):
        good = [i for i in pts if matches(i)]
        bad = [i for i in pts if not matches(i)]
        if not bad or not good:
            break
        G = ref.coords[good] / sc
        tasks = []
        for i in bad:
            d = np.sum((G - ref.coords[i] / sc) ** 2, 1)
            j = good[int(np.argsort(d)[min(k - 1, len(good) - 1)])]          # k-th nearest matched neighbour
            tasks.append((i, atoms[i], spec, F, (res[j]['mo'], atoms[j])))
        run.log(f'retry {k}: {len(tasks)} points that did not reach the data\'s solution, from a matched neighbour')
        batch(tasks, f'retry {k}')
    # ------------------------------------------------------------------ results
    ok = np.array([matches(i) for i in range(N)])
    def arr(key, shape):
        a = np.full((N,) + shape, np.nan)
        for i, r in res.items():
            if key in r and r[key] is not None:
                a[i] = r[key]
        return a
    e_cas, e_pt = arr('e_cas', (nr,)), arr('e_pt', (nr,))
    mu_cas, mu_pt = arr('mu_cas', (nr, 3)), arr('mu_pt', (nr, 3))
    gap = arr('min_gap', ())
    level = 'nevpt2' if np.isfinite(e_pt).any() else 'casscf'
    E = e_pt if level == 'nevpt2' else e_cas
    MU = mu_pt if level == 'nevpt2' else mu_cas
    s0 = np.where(np.isfinite(E).all(1), np.argmin(np.nan_to_num(E, nan=np.inf), 1), 0)
    Es = np.sort(E, 1)
    gap01 = Es[:, 1] - Es[:, 0]
    mu_s0 = MU[np.arange(N), s0]
    mu_s0[~ok] = np.nan
    status = np.where(ok, 'ok', np.where(np.isfinite(e_cas).all(1), 'other solution', 'failed'))
    np.savez(run.p('data', 'dipoles.npz'), coords=ref.coords, coord_names=np.array(ref.coord_names), status=status,
             level=level, e_cas=e_cas, e_pt=e_pt, mu_cas_au=mu_cas, mu_pt_au=mu_pt, mu_S0_au=mu_s0, S0_root=s0,
             gap_S0_S1_hartree=gap01, min_gap_cas_hartree=gap, field_au=F,
             fixed=json.dumps(ref.meta.get('fixed', {})))
    secs = np.array([r.get('seconds', 0.0) for r in res.values()])
    mag = np.linalg.norm(mu_s0[ok], axis=1) * AU_DEBYE
    info = {'n': int(len(pts)), 'n_ok': int(ok.sum()), 'n_other_solution': int((status == 'other solution').sum()),
            'n_failed': int((status == 'failed').sum()), 'level': level, 'field_au': F,
            'S0_dipole_debye': {'min': float(mag.min()) if len(mag) else None,
                                'median': float(np.median(mag)) if len(mag) else None,
                                'max': float(mag.max()) if len(mag) else None},
            'cpu_hours': float(secs.sum() / 3600)}
    run.p('results', 'dipoles.json').write_text(json.dumps(info, indent=1))
    run.log(f'{int(ok.sum())}/{len(pts)} points on the data\'s solution ({info["n_other_solution"]} on another, '
            f'{info["n_failed"]} failed); S0 dipole {info["S0_dipole_debye"]["min"]:.2f}-'
            f'{info["S0_dipole_debye"]["max"]:.2f} D (median {info["S0_dipole_debye"]["median"]:.2f}); '
            f'{info["cpu_hours"]:.1f} CPU h -> data/dipoles.npz')
    _figure_dipoles(run, ref, mu_s0, ok)


def _figure_dipoles(run, ref, mu, ok):
    plt = W._mpl()
    X = ref.coords
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    m = ok & (X[:, 0] < 4.0)
    mag = np.linalg.norm(mu, axis=1) * AU_DEBYE
    sc = ax[0].scatter(X[m, 0], mag[m], c=X[m, 1], s=10, cmap='viridis')
    fig.colorbar(sc, ax=ax[0], label='∠COO (°)')
    ax[0].set(xlabel='r(O–O) (Å)', ylabel='|μ(S0)| (D)', title='Ground-state dipole at the reference points')
    try:
        cm = W.cut_mask(ref) & ok
        o = np.argsort(X[cm, 0])
        for c, lab, col in ((0, 'μx (along C→O)', W.SCOL[1]), (1, 'μy', W.SCOL[2])):
            ax[1].plot(X[cm, 0][o], mu[cm][o, c] * AU_DEBYE, 'o-', ms=4, color=col, label=lab)
        ax[1].plot(X[cm, 0][o], mag[cm][o], 'k--', lw=1, label='|μ|')
        ax[1].legend(fontsize=8)
        ax[1].set(xlabel='r(O–O) (Å)', ylabel='D', title='Equilibrium cut, molecule frame')
    except Exception:
        ax[1].axis('off')
    fig.tight_layout()
    fig.savefig(run.p('figures', 'D1_dipoles.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)
    run.log('figures/D1_dipoles.png')


def stage_dipole_fit(run):
    from ..dipole import EA_DEBYE, fit_dipole_model
    fc = run.cfg['dipole_fit']
    path = run.p('data', 'dipoles.npz')
    if not path.exists():
        raise SystemExit(f'{path} not found: run the dipoles stage first')
    z = np.load(path, allow_pickle=False)
    names = [str(n) for n in z['coord_names']]
    fixed = json.loads(str(z['fixed']))
    X, mu = z['coords'], z['mu_S0_au'] * AU_EA
    use = (z['status'] == 'ok') & np.isfinite(mu).all(1) & (z['gap_S0_S1_hartree'] > float(fc['gap_mEh']) * 1e-3)
    if fc.get('r_max'):
        use &= X[:, 0] <= float(fc['r_max'])
    run.log(f'dipole surface: {int(use.sum())} points (status ok, S1 - S0 > {fc["gap_mEh"]} mEh'
            + (f', r_OO <= {fc["r_max"]}' if fc.get('r_max') else '') + f'); charges of C, O1, O2, H, H from '
            f'{fc["members"]} networks {fc["hidden"]}')
    # a fixed held-out test set, then the members on the rest
    idx = np.where(use)[0]
    p = np.random.default_rng(0).permutation(idx)
    nte = int(float(fc['test_fraction']) * len(p))
    te, tr = p[:nte], p[nte:]
    model, val = fit_dipole_model(X[tr], mu[tr], names, fixed, int(fc['members']), tuple(fc['hidden']),
                                  int(fc['epochs']), float(fc['lr']), log=run.log)
    pred = model.body_dipole(X[te])
    err = np.sqrt(np.mean(np.sum((pred - mu[te]) ** 2, 1))) * EA_DEBYE if nte else float('nan')
    mag_err = np.sqrt(np.mean((np.linalg.norm(pred, axis=1) - np.linalg.norm(mu[te], axis=1)) ** 2)) * EA_DEBYE \
        if nte else float('nan')
    run.log(f'held-out test ({nte} points): vector RMSE {err:.3f} D, |μ| RMSE {mag_err:.3f} D')
    if fc.get('final', True):
        model, _ = fit_dipole_model(X[idx], mu[idx], names, fixed, int(fc['members']), tuple(fc['hidden']),
                                    int(fc['epochs']), float(fc['lr']), log=None)
    model.meta = {'level': str(z['level']), 'state': 'S0', 'units': 'e*Angstrom (x 4.8032 = Debye)',
                  'frame': 'ch2oo_xyz: O1 at origin, C on -x, O2 in the xy plane (y > 0)',
                  'n_points': int(len(idx)), 'test_rmse_debye': float(err), 'test_mag_rmse_debye': float(mag_err),
                  'member_val_rmse_debye': [float(v) for v in val], 'trained_on_all': bool(fc.get('final', True))}
    model.save(run.p('models', 'dipole.pt'))
    run.p('results', 'dipole_fit.json').write_text(json.dumps(model.meta, indent=1))
    run.log('wrote models/dipole.pt (+ .json) and results/dipole_fit.json')
    plt = W._mpl()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for c, lab in ((0, 'μx'), (1, 'μy')):
        ax[c].plot(mu[te, c] * EA_DEBYE, pred[:, c] * EA_DEBYE, 'o', ms=4, color=W.SCOL[c + 1])
        lim = [np.min(mu[te, c]) * EA_DEBYE - 0.3, np.max(mu[te, c]) * EA_DEBYE + 0.3] if nte else [0, 1]
        ax[c].plot(lim, lim, 'k--', lw=0.8)
        ax[c].set(xlabel=f'{lab}, finite field (D)', ylabel=f'{lab}, dipole surface (D)',
                  title=f'{lab}: held-out points')
    fig.tight_layout()
    fig.savefig(run.p('figures', 'D2_dipole_fit.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)
    run.log('figures/D2_dipole_fit.png')
