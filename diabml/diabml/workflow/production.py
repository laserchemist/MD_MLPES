"""Production surface: an averaged ensemble of framed models, and its error by region.

    python -m diabml.workflow.ch2oo production RUN      # any run folder with data/refs.npz (e.g. the last AL round)

Two phases:
  test    a fixed held-out test set (test_fraction of the usable points, never trained on). `members` framed models
          (charge restraint + coupling fit, the production setting) are trained on the rest, each with its own
          seed and train/validation split. The ensemble averages their adiabatic energies and forces (EnsembleModel,
          mode 'adiabatic'); averaging their matrices instead is reported for comparison and is worse, because the
          members represent the three degenerate CH2O + O(1D) structures differently. Errors of the members and of
          the ensemble are reported by region (T10, P1, P2).
  final   (final = true) the members are retrained on all usable points and saved as models/production.pt, the
          model to load for dynamics. With final = false the test-phase ensemble is saved instead.

Regions (each test point belongs to one):
  dynamics     S0 at most max_S0_eV above CH2O + O(1D) and outside the corner box: where trajectories go
  high energy  S0 above max_S0_eV (outside the corner)
  corner       the compressed low-angle corner (corner box), where SA-CASSCF has several solutions
and, inside the dynamics region, the rows whose NEVPT2 states had to be put in energy order are reported separately.

Settings ([production] in config.toml; defaults in diabml/workflow/config.py):
  members = 5, test_fraction = 0.1, final = true, coupling_weight = 0.1, charge_weight = 1.0,
  max_S0_eV = 1.0, corner = {r_OO = [0, 2.0], a_COO = [0, 70.0]}
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import ch2oo as W

EV_KCAL = W.EV_KCAL


def regions(ref, pc):
    """(labels, masks): dynamics / high energy / corner for every row of ref."""
    X, E0 = ref.coords, ref.energies[:, 0]
    corner = np.ones(len(ref), bool)
    for n, (a, b) in pc['corner'].items():
        if n in ref.coord_names:
            j = ref.coord_names.index(n)
            corner &= (X[:, j] >= float(a)) & (X[:, j] <= float(b))
    high = ~corner & (E0 > float(pc['max_S0_eV']))
    dyn = ~corner & ~high
    return {'dynamics': dyn, 'high energy': high, 'corner': corner}


def _reordered(ref):
    P = ref.extra.get('state_order')
    if P is None:
        return np.zeros(len(ref), bool)
    return np.any(np.asarray(P)[:, :4] != np.arange(4), 1)


def _fit_member(run, ref, tr, va, seed, pc, nac, q_ion, log):
    """One production member: framed model + charge restraint (+ coupling fit when couplings exist)."""
    import torch
    from ..models import train
    tc, rc = run.cfg['train'], run.cfg['train']['restraint']
    E = ref.energies[:, :4]
    tt, mt = W.charge_targets(ref, tr, q_ion, rc['min_gap_eV'], rc.get('states', 'all'))
    pq = W.restraint_penalty(float(pc['charge_weight']), tt, mt)
    pen = pq
    if nac is not None and float(pc['coupling_weight']) > 0:
        tau, pairs = nac
        cf = tc['coupling_fit']
        ct, cm_ = W.coupling_targets(ref, tr, tau, pairs, cf['min_gap_eV'], cf['cap'])
        pcp = W.coupling_penalty(float(pc['coupling_weight']), ref.coords[tr], ct, cm_, pairs)
        pen = lambda mod, c, pq=pq, pcp=pcp: pq(mod, c) + pcp(mod, c)
    torch.manual_seed(seed)
    t0 = time.time()
    sw = low_energy_weights(ref, pc)
    m = train(W.make_model(run, ref, 'framed'), ref.coords[tr], E[tr], ref.coords[va], E[va],
              epochs=int(tc['epochs']), restarts=int(tc['restarts']), lr=float(tc['lr_framed']),
              alpha2=float(tc['alpha2']), penalty=pen,
              sample_weights=None if sw is None else sw[tr], val_sample_weights=None if sw is None else sw[va])
    log(f'  member seed {seed}: {len(tr)} training points ({time.time() - t0:.0f} s)')
    return m


def low_energy_weights(ref, pc):
    """Per-point training weights 1 + w exp(-(E_S0 - E_min)/width) ([production] low_energy_weight, weight_width_eV),
    or None when low_energy_weight is 0. Vibrational spectra need the well to be fitted to a few meV; the rest of the
    surface keeps weight 1."""
    w = float(pc.get('low_energy_weight', 0) or 0)
    if w <= 0:
        return None
    e0 = np.asarray(ref.energies[:, 0], float)
    emin = np.nanmin(e0)
    return 1.0 + w * np.exp(-np.clip(np.nan_to_num(e0 - emin, nan=99.0), 0, None) / float(pc.get('weight_width_eV', 0.3)))


def _ensemble(run, ref, pool, seeds, pc, nac, q_ion, log):
    from ..models import EnsembleModel
    members = []
    for s in seeds:
        p = np.random.default_rng(2000 + s).permutation(pool)
        nva = max(5, int(0.1 * len(p)))
        members.append(_fit_member(run, ref, p[nva:], p[:nva], s, pc, nac, q_ion, log))
    ens = EnsembleModel(members)
    signs = ens.align_signs(ref.coords[pool])
    flips = int((signs[1:] < 0).sum())
    log(f'  sign alignment: {flips} structure sign flips across {len(seeds) - 1} members')
    return ens


def _rmse(P, E, m):
    return (np.sqrt(np.mean((P[m] - E[m]) ** 2, 0)) * EV_KCAL).tolist() if m.any() else None


def stage_production(run):
    import torch
    from ..frames.base import eigvals_sorted
    from ..models import save_model
    run.threads()
    pc = run.cfg['production']
    ref = W.load_refs(run)
    U = W.usable_points(run, ref)
    E = ref.energies[:, :4]
    seeds = list(range(int(pc['members'])))
    nac = W.load_nacs(run, ref)
    rc = run.cfg['train']['restraint']
    usable = np.zeros(len(ref), bool); usable[U] = True
    q_ion = W.q_ion_auto(ref, usable) if str(rc['q_ref']) == 'auto' else float(rc['q_ref'])
    reg = regions(ref, pc)
    reo = _reordered(ref)
    run.log(f'{ref}; {len(U)} usable points; {len(seeds)} members; charge restraint {pc["charge_weight"]}, '
            f'coupling fit {pc["coupling_weight"] if nac is not None else "(no couplings)"}; q_ion {q_ion:.3f} e')
    for k, m in reg.items():
        run.log(f'  region {k}: {int((m & usable).sum())} usable points')
    # ---- test phase
    p = np.random.default_rng(777).permutation(U)
    nte = max(10, int(float(pc['test_fraction']) * len(p)))
    te, pool = np.sort(p[:nte]), p[nte:]
    run.log(f'test phase: {len(te)} held-out points, ensemble trained on {len(pool)}')
    ens = _ensemble(run, ref, pool, seeds, pc, nac, q_ion, run.log)
    Pm = ens.member_predictions(ref.coords[te])                       # (K, n, 4)
    Pe = ens.predict(ref.coords[te])                                  # mean of the members' eigenvalues
    with torch.no_grad():                                             # for comparison: eigenvalues of the mean matrix
        Pa = eigvals_sorted(ens.matrix_average(ref.coords[te]))[:, :4].numpy()
    Et = E[te]
    rows = []
    masks = {k: v[te] for k, v in reg.items()}
    masks['dynamics, NEVPT2 states re-sorted'] = reg['dynamics'][te] & reo[te]
    masks['dynamics, states in order'] = reg['dynamics'][te] & ~reo[te]
    e0 = E[:, 0]
    masks['well (S0 within 0.5 eV of the minimum)'] = (e0[te] - np.nanmin(e0[U])) <= 0.5
    masks['all'] = np.ones(len(te), bool)
    res = {'n_test': int(len(te)), 'members': len(seeds), 'regions': {}}
    if low_energy_weights(ref, pc) is not None:
        run.log(f'  training weights 1 + {pc["low_energy_weight"]} exp(-(S0 - min)/{pc.get("weight_width_eV", 0.3)} eV)')
    for k, m in masks.items():
        mem = [_rmse(Pm[i], Et, m) for i in range(len(seeds))]
        r = {'n': int(m.sum()), 'member_rmse_kcal': mem,
             'member_mean_rmse_kcal': np.mean([x for x in mem if x is not None], 0).tolist() if m.any() else None,
             'ensemble_rmse_kcal': _rmse(Pe, Et, m), 'matrix_average_rmse_kcal': _rmse(Pa, Et, m),
             'spread_kcal': (Pm.std(0)[m].mean(0) * EV_KCAL).tolist() if m.any() else None}
        res['regions'][k] = r
        if m.any():
            run.log(f'  {k:38s} n={r["n"]:4d}  one model {np.round(r["member_mean_rmse_kcal"], 2)}  '
                    f'ensemble {np.round(r["ensemble_rmse_kcal"], 2)} kcal/mol')
    # couplings of the ensemble on held-out points
    if nac is not None:
        tau, pairs = nac
        cf = run.cfg['train']['coupling_fit']
        ct, cm_ = W.coupling_targets(ref, te, tau, pairs, cf['min_gap_eV'], cf['cap'])
        T = W.model_couplings_r(ens, ref.coords[te])
        tm = np.stack([np.abs(T[:, a, b]) for a, b in pairs], 1)
        res['coupling_rmse'] = [float(np.sqrt(np.mean((tm[cm_[:, c], c] - ct[cm_[:, c], c]) ** 2)))
                                if cm_[:, c].any() else None for c in range(len(pairs))]
        res['pairs'] = [list(x) for x in pairs]
        run.log(f'  ensemble held-out |tau| RMSE {np.round(res["coupling_rmse"], 3)} 1/Å')
    np.savez(run.p('results', 'production_test.npz'), idx=te, member_pred=Pm, ensemble_pred=Pe, E=Et,
             region=np.array([next(k for k in ('dynamics', 'high energy', 'corner') if reg[k][i]) for i in te]),
             reordered=reo[te])
    # ---- final phase
    if pc.get('final', True):
        run.log(f'final phase: retraining {len(seeds)} members on all {len(U)} usable points')
        ens = _ensemble(run, ref, U, seeds, pc, nac, q_ion, run.log)
        res['final_on_all'] = True
    meta = {'run': str(run.path), 'experiment': f'production ensemble of {len(seeds)} framed models',
            'n_train': int(len(U)) if pc.get('final', True) else int(len(pool)),
            'coord_names': ref.coord_names, 'coord_units': [W.UNITS[c] for c in ref.coord_names],
            'energy_unit': 'eV', 'energy_zero': ref.meta.get('zero', ''), 'method': ref.meta.get('method', ref.name),
            'fixed_coordinates': ref.meta.get('fixed', {}),
            'atom_order': ['C', 'O1 (central)', 'O2 (terminal)', 'H', 'H'],
            'charge_weight': float(pc['charge_weight']), 'coupling_weight': float(pc['coupling_weight']),
            'test_errors_kcal': {k: v['ensemble_rmse_kcal'] for k, v in res['regions'].items()},
            'regions': {'max_S0_eV': float(pc['max_S0_eV']), 'corner': pc['corner']}}
    hs = hole_scan(ens, ref)
    res['hole_scan'] = hs
    run.log(f'hole scan below the data ({hs["scanned"]} geometries, every member): lowest S0 {hs["lowest_S0_eV"]:.2f} eV '
            f'at {np.round(hs["at"], 2).tolist()} against the lowest reference {hs["reference_min_eV"]:.2f} eV'
            + ('  -> HOLE: do not run dynamics that can reach it' if hs['hole'] else '  (no hole)'))
    meta['hole_scan'] = {k: hs[k] for k in ('lowest_S0_eV', 'at', 'hole', 'reference_min_eV')}
    for side in edge_scans(ens, ref):
        res.setdefault('edge_scans', []).append(side)
        run.log(f'hole scan beyond the data in {side["coordinate"]} {side["range"][0]:.2f}-{side["range"][1]:.2f} Å: '
                f'worst member S0 {side["lowest_S0_eV"]:.2f} eV at {np.round(side["at"], 2).tolist()}'
                + (' (HOLE in one member)' if side['hole'] else '')
                + f'; ensemble mean (what dynamics feels) {side["lowest_mean_S0_eV"]:.2f} eV at '
                f'{np.round(side["mean_at"], 2).tolist()}' + ('  -> HOLE' if side['hole_in_mean'] else '  (no hole)')
                + f'; lowest reference {side["reference_min_eV"]:.2f} eV')
    save_model(ens, run.p('models', 'production.pt'), meta=meta)
    # harmonic check at the S0 minimum (heavy-atom modes; the rest are the CH2 restraint springs of a reduced model)
    try:
        from ..dynamics.ir import harmonic_ir
        hrows, x0 = harmonic_ir(str(run.p('models', 'production.pt')), None)
        heavy = [r for r in hrows if r['H_fraction'] <= 0.5]
        res['harmonic'] = {'minimum': x0.tolist(), 'modes': hrows}
        run.log('harmonic at the S0 minimum ' + str(dict(zip(ref.coord_names, np.round(x0, 3).tolist()))) + ': '
                + ', '.join(f'{r["cm-1"]:.0f} cm-1 (d{"/d".join(ref.coord_names)} per unit mode '
                            f'{np.round(r["dq"], 2).tolist()})' for r in heavy))
    except Exception as exc:                                       # not CH2OO, or no minimum found
        run.log(f'harmonic check skipped ({exc.__class__.__name__}: {exc})')
    run.p('results', 'production.json').write_text(json.dumps(res, indent=1))
    run.log(f'wrote models/production.pt (+ .json) and results/production.json')
    _figures(run, ref, ens, te, Pm, Pe, Et, masks)


def _figures(run, ref, ens, te, Pm, Pe, Et, masks):
    plt = W._mpl()
    caps = {}
    col = {'dynamics': W.SCOL[1], 'high energy': W.SCOL[2], 'corner': '#B03A2E'}
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    for s, ax in enumerate(axes):
        for k in ('dynamics', 'high energy', 'corner'):
            m = masks[k]
            if m.any():
                ax.plot(Et[m, s], Pe[m, s], 'o', ms=3.5, color=col[k], alpha=0.8, label=f'{k} ({int(m.sum())})')
        lo, hi = np.nanmin(Et[:, s]) - 0.2, np.nanmax(Et[:, s]) + 0.2
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
        r = masks['dynamics']
        rm = np.sqrt(np.mean((Pe[r, s] - Et[r, s]) ** 2)) * EV_KCAL if r.any() else np.nan
        ax.set(xlabel='reference (eV)', ylabel='ensemble (eV)', title=f'S{s}: dynamics region {rm:.2f} kcal/mol')
    axes[0].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(run.p('figures', 'P1_production_parity.png'), bbox_inches='tight'); plt.close(fig)
    caps['P1_production_parity.png'] = ('Production ensemble on held-out test points, coloured by region. Titles: '
                                        'RMSE in the dynamics region (S0 within the energy window, outside the '
                                        'low-angle corner).')
    # cut through equilibrium with member band
    x = np.linspace(1.15, 4.0, 300)
    X = W.cut_coords(ref, x)
    Mm = ens.member_predictions(X)
    Me = ens.predict(X)
    cm = W.cut_mask(ref)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for s in range(4):
        ax.fill_between(x, Mm[:, :, s].min(0), Mm[:, :, s].max(0), color=W.SCOL[s], alpha=0.25, lw=0)
        ax.plot(x, Me[:, s], color=W.SCOL[s], lw=1.6, label=f'S{s}')
        ax.plot(ref.coords[cm, 0], ref.energies[cm, s], 'o', ms=3, color=W.SCOL[s])
    sv = W.slice_values(ref)
    ax.set(xlabel='r(O–O) (Å)', ylabel='E (eV)', xlim=(1.15, 4.0),
           title=', '.join(f'{W.LABEL[k]} = {v:g}{W.UNITS[k]}' for k, v in sv.items()))
    ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(run.p('figures', 'P2_production_cut.png'), bbox_inches='tight'); plt.close(fig)
    caps['P2_production_cut.png'] = ('Production surface along r(O–O) through the equilibrium angle: ensemble '
                                     '(lines), range of the individual members (bands) and reference points.')
    r = json.loads(run.p('results', 'production.json').read_text()) if run.p('results', 'production.json').exists() else {}
    hs = r.get('hole_scan')
    if hs:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.plot(hs['r'], hs['lowest_S0_by_r'], '-o', ms=3, color=W.SCOL[0], label='lowest S0 of any member')
        ax.axhline(hs['reference_min_eV'], ls=':', color='k', label='lowest reference energy')
        ax.set(xlabel='r(O–O) (Å)', ylabel='E (eV)', title='Below the data: ' + ('HOLE' if hs['hole'] else 'no hole'))
        ax.legend()
        fig.tight_layout(); fig.savefig(run.p('figures', 'P3_hole_scan.png'), bbox_inches='tight'); plt.close(fig)
        caps['P3_hole_scan.png'] = ('Lowest S0 of any ensemble member at O–O distances shorter than any reference '
                                    'point, minimised over the other coordinates across their data range. A curve '
                                    'below the dotted line is a hole that trajectories can fall into.')
    run.p('figures', 'production_captions.json').write_text(json.dumps(caps, indent=1))
    for k in caps:
        run.log(f'figures/{k}')


def hole_scan(model, ref, r_lo=0.7, n_r=40, n_other=7):
    """Lowest S0 of the model below the data (r_OO from r_lo to the shortest r_OO in the data), over a grid of the
    other coordinates spanning the data. A value below the lowest reference energy is a hole."""
    X = ref.coords
    r_min = float(X[:, 0].min())
    axes = [np.linspace(r_lo, r_min, n_r)] + [np.linspace(X[:, j].min(), X[:, j].max(), n_other)
                                              for j in range(1, X.shape[1])]
    G = np.array(np.meshgrid(*axes, indexing='ij')).reshape(len(axes), -1).T
    P = model.member_predictions(G)[:, :, 0] if hasattr(model, 'member_predictions') else model.predict(G)[None, :, 0]
    worst = P.min(0)
    i = int(np.argmin(worst))
    e_min = float(np.nanmin(ref.energies[:, 0]))
    per_r = [float(worst[np.isclose(G[:, 0], rv)].min()) for rv in axes[0]]
    return {'reference_min_eV': e_min, 'lowest_S0_eV': float(worst[i]), 'at': G[i].tolist(),
            'hole': bool(worst[i] < e_min - 0.1), 'r': axes[0].tolist(), 'lowest_S0_by_r': per_r,
            'scanned': int(len(G))}


def edge_scans(model, ref, beyond=0.5, below=0.2, n_c=21, n_other=7, r_oo_max=3.0):
    """Lowest S0 of any member beyond the data in every other bond length (e.g. r_CO from its largest value to
    +beyond Angstrom, and from its smallest down by `below`), with r_OO and the angles spanning the data. The
    r_OO scan (hole_scan) cannot see a C-O bond that the surface lets stretch without limit."""
    X = ref.coords
    e_min = float(np.nanmin(ref.energies[:, 0]))
    out = []
    for j, name in enumerate(ref.coord_names):
        if j == 0 or not name.startswith('r_'):
            continue
        lo, hi = float(X[:, j].min()), float(X[:, j].max())
        for rng in ((hi, hi + beyond), (lo - below, lo)):
            axes = []
            for k in range(X.shape[1]):
                if k == j:
                    axes.append(np.linspace(*rng, n_c))
                elif k == 0:
                    axes.append(np.linspace(X[:, 0].min(), min(X[:, 0].max(), r_oo_max), n_other))
                else:
                    axes.append(np.linspace(X[:, k].min(), X[:, k].max(), n_other))
            G = np.array(np.meshgrid(*axes, indexing='ij')).reshape(len(axes), -1).T
            P = model.member_predictions(G)[:, :, 0] if hasattr(model, 'member_predictions') else model.predict(G)[None, :, 0]
            worst, mean = P.min(0), P.mean(0)
            i, k = int(np.argmin(worst)), int(np.argmin(mean))
            out.append({'coordinate': name, 'range': list(rng), 'lowest_S0_eV': float(worst[i]), 'at': G[i].tolist(),
                        'lowest_mean_S0_eV': float(mean[k]), 'mean_at': G[k].tolist(),
                        'reference_min_eV': e_min, 'hole': bool(worst[i] < e_min - 0.1),
                        'hole_in_mean': bool(mean[k] < e_min - 0.1), 'scanned': int(len(G))})
    return out


def table_rows(run):
    path = run.p('results', 'production.json')
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    f = lambda v: ' / '.join(f'{x:.2f}' for x in v) if v else ''
    rows = []
    for k, v in r['regions'].items():
        if v['n']:
            rows.append([k, v['n'], f(v['member_mean_rmse_kcal']), f(v['ensemble_rmse_kcal']), f(v['spread_kcal'])])
    return rows
