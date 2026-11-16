"""Repair the reference data with the audit's ORCA results, and make a QD-NEVPT2 version of it.

    python -m diabml.workflow.ch2oo repair RUN          # after the audit stage of RUN

The audit (diabml/workflow/audit.py) recomputed every point of RUN/data/refs.npz with ORCA and classified it. This
stage writes two new data sets, each in a folder of its own with the couplings and orbital library that go with it,
so a new run can use it directly as refs.file (its folder supplies nacs.npz and orbitals.npz, as for any run):

  RUN/data/repaired/nevpt2/refs.npz     the base data, energies at the same level (SC-NEVPT2, or CASSCF for
                                        CASSCF-only data), with every 'ours higher' point replaced by ORCA's lower
                                        SA-CASSCF solution and its SC-NEVPT2 energies
  RUN/data/repaired/qd_nevpt2/refs.npz  the same points with QD-NEVPT2 (van Vleck) energies from ORCA wherever ORCA
                                        found the right solution ('same' and 'ours higher'); the energy zero is moved
                                        by the QD shift at the zero point, so the two sets share their asymptote

What happens to each audit class:
  same          unchanged (in the QD set: ORCA's QD-NEVPT2 energies)
  ours higher   energies from ORCA. The per-state diagnostics computed by PySCF on the wrong solution (terminal-O
                charges, unpaired electrons) are set to NaN, so the charge restraint skips those points, unless the
                solutions differ by less than extras_tol_mEh (the far-field near-degenerate points). Couplings
                (nacs.npz) and orbitals (orbitals.npz) at those geometries are removed: they belong to the wrong
                solution, and orbitals from it would start new points on it again
  orca higher   ORCA could not reach our solution: our (lower) energies are kept, unverified (keep_unverified)
  failed        ORCA did not finish: kept, unverified (keep_unverified); no QD-NEVPT2 energies
Unverified points have no QD-NEVPT2 energies, so they are left out of the QD set. Same-solution points whose PySCF
NEVPT2 ran on CASCI states that did not reproduce SA-CASSCF (casci_mismatch, so far left out of training) get ORCA's
SC-NEVPT2 energies and become usable.

extra['audit'] records the class of every row (0 same, 1 repaired, 2 orca higher, 3 failed, -1 not audited) and
extra['from_orca'] which energies came from ORCA. Writes results/repair.json, results/repair.csv and
figures/R1_repair.png (how much each repaired point moved; the QD - SC shifts).

New points of later active-learning rounds are computed with PySCF from the (now cleaned) orbital library, so they can
land on a wrong solution too: audit each round's new points. The QD set cannot be extended by the compute stage
(PySCF has no QD-NEVPT2); use it for production surfaces and explore runs without compute.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from . import ch2oo as W

HARTREE_EV = 27.211386245988
KCAL = 627.509474
EV_KCAL = W.EV_KCAL
CODE = {'same': 0, 'ours higher': 1, 'orca higher': 2, 'failed': 3}


def _filter_nacs(src, dst, drop_rows):
    """Copy nacs.npz without the couplings at rows in drop_rows. Returns how many were removed."""
    if not src.exists():
        return 0
    z = dict(np.load(src))
    idx = np.asarray(z['idx'])
    keep = ~np.isin(idx, np.where(drop_rows)[0])
    n = len(idx)
    for k, v in z.items():
        if k != 'pairs' and np.ndim(v) >= 1 and len(v) == n:
            z[k] = v[keep]
    np.savez(dst, **z)
    return int((~keep).sum())


def _filter_orbitals(src, dst, coords_drop):
    """Copy orbitals.npz without the orbitals at the given geometries. Returns how many were removed."""
    if not src.exists():
        return 0
    z = dict(np.load(src))
    bad = {tuple(np.round(c, 6)) for c in coords_drop}
    keep = np.array([tuple(np.round(c, 6)) not in bad for c in z['coords']], bool)
    z['coords'], z['mo'] = z['coords'][keep], z['mo'][keep]
    np.savez(dst, **z)
    return int((~keep).sum())


def _zero_row(E_level_eV):
    """The row that defines the energy zero (state-0 energy exactly 0)."""
    e0 = np.abs(np.where(np.isfinite(E_level_eV[:, 0]), E_level_eV[:, 0], np.inf))
    z = int(np.argmin(e0))
    return z if e0[z] < 1e-6 else None


def stage_repair(run):
    from ..refset import RefSet
    rc = run.cfg['repair']
    path_a = run.p('results', 'audit.npz')
    if not path_a.exists():
        raise SystemExit(f'{path_a} not found: run the audit stage first '
                         f'(python -m diabml.workflow.ch2oo audit {run.path} --orca ...)')
    ref = RefSet.load(run.p('data', 'refs.npz'))                      # raw: SA-CASSCF root order
    a = np.load(path_a)
    if a['coords'].shape != ref.coords.shape or not np.allclose(a['coords'], ref.coords):
        raise SystemExit('data/refs.npz has changed since the audit (different points): run the audit again')
    N, ns = ref.energies.shape
    zero = dict(ref.meta['zero_hartree'])
    level = ref.meta.get('energy_level', 'casscf')
    has_pt = 'E_nevpt2_eV' in ref.extra and 'nevpt2' in zero
    st = a['status'].astype(str)
    code = np.array([CODE.get(s, -1) for s in st])
    o_cas, o_sc, o_qd = a['orca_casscf_Eh'], a['orca_sc_nevpt2_Eh'], a['orca_qd_nevpt2_Eh']
    dev = a['casscf_dev_Eh']
    fix = (code == 1) & np.isfinite(o_cas).all(1) & (np.isfinite(o_sc).all(1) if has_pt else True)
    lose = fix & (dev > float(rc['extras_tol_mEh']) * 1e-3)            # per-state diagnostics from the wrong solution
    unver = np.isin(code, [2, 3])
    keep_unver = bool(rc['keep_unverified'])
    run.log(f'audit of {N} points: {int((code == 0).sum())} same, {int(fix.sum())} to repair ({int(lose.sum())} by more '
            f'than {rc["extras_tol_mEh"]} mEh: their PySCF charges, couplings and orbitals are dropped), '
            f'{int(unver.sum())} unverified ({"kept" if keep_unver else "left out"}), {int((code < 0).sum())} not audited')

    # --------------------------------------------------------------------------- repaired set, base level
    ex = {k: np.array(v, copy=True) for k, v in ref.extra.items()}
    Ec_old = np.asarray(ex['E_casscf_eV'], float)
    Ec = Ec_old.copy()
    Ec[fix] = (o_cas[fix] - zero['casscf']) * HARTREE_EV
    ex['E_casscf_eV'] = Ec
    levels = {'casscf': Ec}
    En_old = None
    pt_fix = np.zeros(N, bool)
    if has_pt:
        En_old = np.asarray(ex['E_nevpt2_eV'], float)
        En = En_old.copy()
        En[fix] = (o_sc[fix] - zero['nevpt2']) * HARTREE_EV            # ORCA root order, like its CASSCF roots
        ex['E_nevpt2_eV'] = En
        levels['nevpt2'] = En
        # same SA-CASSCF solution, but PySCF's NEVPT2 ran on CASCI states that did not reproduce it (those points
        # are left out of training): ORCA's SC-NEVPT2 is on the right states, so use it and the point is usable again
        cm = np.nan_to_num(np.asarray(ex.get('casci_mismatch_hartree', np.zeros(N)), float), nan=0.0)
        pt_fix = (code == 0) & (cm > float(run.cfg['train'].get('max_casci_mismatch_hartree', 1e-3))) \
            & np.isfinite(o_sc).all(1)
        En[pt_fix] = (o_sc[pt_fix] - zero['nevpt2']) * HARTREE_EV
        if pt_fix.any():
            run.log(f'{int(pt_fix.sum())} same-solution points whose PySCF NEVPT2 states did not match SA-CASSCF (left '
                    f'out of training until now): SC-NEVPT2 energies from ORCA')
        if 'casci_mismatch_hartree' in ex:
            ex['casci_mismatch_hartree'] = np.where(fix | pt_fix, 0.0, ex['casci_mismatch_hartree'])
    for k in W.PER_STATE:
        if k in ex and np.ndim(ex[k]) == 2:
            v = np.array(ex[k], float)
            v[lose] = np.nan
            ex[k] = v
    for k in ('converged', 'active_space_consistent'):
        if k in ex:
            ex[k] = ex[k].astype(bool) | fix
    ex['audit'] = code
    ex['from_orca'] = fix | pt_fix
    E = np.array(ref.energies, float)
    E[fix | pt_fix] = levels.get(level, Ec)[fix | pt_fix][:, :ns]
    if not keep_unver:
        E[unver] = np.nan
    n_fix = int(fix.sum())
    base_name = 'nevpt2' if has_pt and level == 'nevpt2' else level
    meta = dict(ref.meta, method=f'{ref.meta.get("method", ref.name)} [repaired: {n_fix} points from ORCA]',
                repair={'audit_run': str(run.path), 'n_repaired': n_fix, 'n_extras_dropped': int(lose.sum()),
                        'n_unverified': int(unver.sum()), 'unverified_kept': keep_unver})
    rep = RefSet(ref.name, ref.coords.copy(), E, list(ref.coord_names), extra=ex, meta=meta)
    out = {}
    drop_coords = ref.coords[lose]
    for name in [base_name] + (['qd_nevpt2'] if has_pt and np.isfinite(o_qd).any() else []):
        d = run.p('data', 'repaired', name)
        d.mkdir(parents=True, exist_ok=True)
        out[name] = d
    rep.save(out[base_name] / 'refs.npz')

    # --------------------------------------------------------------------------- QD-NEVPT2 set
    qd_info = None
    if 'qd_nevpt2' in out:
        ok = np.isin(code, [0, 1]) & np.isfinite(o_qd).all(1)
        z = _zero_row(En_old)
        if z is not None and ok[z]:
            shift0 = float(o_qd[z, 0] - np.sort(o_sc[z])[0])
            zq = zero['nevpt2'] + shift0
            how = f'QD shift of S0 at the zero point ({shift0 * KCAL:+.3f} kcal/mol) added to the SC-NEVPT2 zero'
        else:
            shift0, zq = 0.0, zero['nevpt2']
            how = 'the zero point has no QD-NEVPT2 energies: SC-NEVPT2 zero kept'
        Eq = np.full((N, ns), np.nan)
        Eq[ok] = ((o_qd[ok] - zq) * HARTREE_EV)[:, :ns]
        exq = dict(ex, E_qd_nevpt2_eV=Eq.copy(), from_orca=ok.copy())
        if 'casci_mismatch_hartree' in exq:
            exq['casci_mismatch_hartree'] = np.zeros(N)
        method = re.sub(r'SC-NEVPT2', 'QD-NEVPT2', ref.meta.get('method', ref.name))
        if 'QD-NEVPT2' not in method:
            method += ' + QD-NEVPT2'
        mq = dict(meta, method=f'{method} [QD-NEVPT2 (van Vleck) from ORCA at {int(ok.sum())} points]',
                  energy_level='qd_nevpt2', zero_hartree=dict(zero, qd_nevpt2=zq),
                  zero_qd=how)
        RefSet(ref.name.replace('SC-NEVPT2', 'QD-NEVPT2'), ref.coords.copy(), Eq, list(ref.coord_names),
               extra=exq, meta=mq).save(out['qd_nevpt2'] / 'refs.npz')
        En_sorted = np.sort(levels['nevpt2'], 1)
        sh = (Eq - En_sorted[:, :ns]) * EV_KCAL
        qd_info = {'n_points': int(ok.sum()), 'n_without_qd': int(N - ok.sum()), 'zero': how,
                   'shift_median_abs_kcal': np.nanmedian(np.abs(sh[ok]), 0).tolist(),
                   'shift_max_abs_kcal': np.nanmax(np.abs(sh[ok]), 0).tolist(),
                   'points_shift_over_1_kcal': int((np.nanmax(np.abs(sh[ok]), 1) > 1.0).sum())}
        run.log(f'QD-NEVPT2 set: {int(ok.sum())} points ({N - int(ok.sum())} without: unverified or not audited); '
                f'{how}; |QD - SC| median S0..S3 {np.round(qd_info["shift_median_abs_kcal"], 2)} kcal/mol, max '
                f'{np.round(qd_info["shift_max_abs_kcal"], 1)}; {qd_info["points_shift_over_1_kcal"]} points move by '
                f'more than 1 kcal/mol')

    # --------------------------------------------------------------------------- couplings and orbitals
    src = run.p('data')
    n_nac = n_orb = 0
    for d in out.values():
        n_nac = _filter_nacs(src / 'nacs.npz', d / 'nacs.npz', lose)
        n_orb = _filter_orbitals(src / 'orbitals.npz', d / 'orbitals.npz', drop_coords)
    # ORCA orbital library for the compute stage (engine = "orca"): the converged orbitals of every point whose
    # energies are ORCA's solution (same or repaired), from the audit folder
    from .orca_compute import save_library
    wd = run.p('data', 'audit')
    att = a['attempt'] if 'attempt' in a.files else np.zeros(N, int)
    lib_rows, lib_g = [], []
    for i in np.where(np.isin(code, [0, 1]))[0]:
        g = wd / (f'p{i:04d}' + (f'_r{int(att[i])}' if int(att[i]) > 0 else '') + '.gbw')
        if g.exists():
            lib_rows.append(i); lib_g.append(str(g.resolve()))
    for d in out.values():
        save_library(d / 'orca_orbitals.json', ref.coord_names, ref.coords[lib_rows], lib_g)
    run.log(f'ORCA orbital library: {len(lib_g)} points (orca_orbitals.json, pointing into {wd})')
    run.log(f'couplings: {n_nac} geometries removed; orbital library: {n_orb} geometries removed '
            f'({"no nacs.npz" if not (src / "nacs.npz").exists() else "nacs.npz"} / '
            f'{"no orbitals.npz" if not (src / "orbitals.npz").exists() else "orbitals.npz"} next to refs.npz)')

    # --------------------------------------------------------------------------- what changed
    lev_old = En_old if has_pt and level == 'nevpt2' else Ec_old
    lev_new = levels['nevpt2'] if has_pt and level == 'nevpt2' else Ec
    dS = (np.sort(lev_new, 1) - np.sort(lev_old, 1))[:, :ns] * EV_KCAL      # kcal/mol, energy order
    info = {'n': N, 'n_nevpt2_states_fixed': int(pt_fix.sum()), 'counts': {k: int((code == v).sum()) for k, v in CODE.items()}, 'n_repaired': n_fix,
            'n_extras_dropped': int(lose.sum()), 'unverified_kept': keep_unver, 'level': level,
            'couplings_removed': n_nac, 'orbitals_removed': n_orb, 'folders': {k: str(v) for k, v in out.items()}}
    if n_fix:
        info['change_median_kcal'] = np.median(dS[fix], 0).tolist()
        info['change_min_kcal'] = dS[fix].min(0).tolist()
        info['points_S0_lower_by_1_kcal'] = int((dS[fix, 0] < -1).sum())
        run.log(f'repaired points: {level} energies change by median {np.round(info["change_median_kcal"], 2)} '
                f'kcal/mol (S0..S3, energy order), most {np.round(info["change_min_kcal"], 1)}; S0 drops by more than '
                f'1 kcal/mol at {info["points_S0_lower_by_1_kcal"]} of {n_fix}')
    if has_pt:
        same = code == 0
        ok = same & ~pt_fix & np.isfinite(o_sc).all(1)
        if ok.any():
            ours = zero['nevpt2'] + En_old[ok] / HARTREE_EV
            d = np.abs(np.sort(o_sc[ok], 1) - np.sort(ours, 1)) * 1e6
            info['sc_check_same_uEh'] = {'median': float(np.median(d)), 'max': float(d.max())}
            run.log(f'check at the {int(ok.sum())} other same-solution points: ORCA and PySCF SC-NEVPT2 differ by median '
                    f'{np.median(d):.1f} µEh, max {d.max():.1f} µEh')
    z = _zero_row(lev_old)
    if z is not None and code[z] == 1:
        run.log(f'note: the zero point (row {z}) itself was repaired; its new S0 is {dS[z, 0]:+.3f} kcal/mol from zero')
    info['qd'] = qd_info
    run.p('results', 'repair.json').write_text(json.dumps(info, indent=1, default=float))
    with open(run.p('results', 'repair.csv'), 'w') as fh:
        fh.write(','.join(['index'] + ref.coord_names + ['audit', 'action'] + [f'change_S{s}_kcal' for s in range(ns)])
                 + '\n')
        for i in range(N):
            act = ('repaired' if fix[i] else 'unverified, kept' if (unver[i] and keep_unver) else
                   'unverified, left out' if unver[i] else 'unchanged')
            fh.write(','.join([str(i)] + [f'{v:.4f}' for v in ref.coords[i]] + [st[i], act]
                              + [f'{v:.3f}' for v in dS[i]]) + '\n')
    _figure(run, ref, fix, dS, out, level)
    run.log('wrote ' + ', '.join(f'{v / "refs.npz"}' for v in out.values()) + ', results/repair.json, '
            'results/repair.csv and figures/R1_repair.png')
    rel = lambda name: f'../{run.path.name}/data/repaired/{name}/refs.npz'
    run.log('next: a production surface (and an explore round) on each set, e.g.')
    for name in out:
        tag = 'sc' if name == 'nevpt2' else 'qd' if name == 'qd_nevpt2' else name
        new = os.path.relpath(run.path.parent / f'{run.path.name}_{tag}')
        run.log(f'  python -m diabml.workflow.ch2oo init {new} --template explore --refs {rel(name)}')
        run.log(f'  python -m diabml.workflow.ch2oo refs {new} && python -m diabml.workflow.ch2oo merge {new} && '
                f'python -m diabml.workflow.ch2oo production {new}')
    return info


def _figure(run, ref, fix, dS, out, level):
    plt = W._mpl()
    has_qd = 'qd_nevpt2' in out
    fig, axes = plt.subplots(1, 2 if has_qd else 1, figsize=(13 if has_qd else 6.5, 4.6), squeeze=False)
    ax = axes[0, 0]
    r = ref.coords[:, 0]
    j = ref.coord_names.index('r_CO') if 'r_CO' in ref.coord_names else None
    if fix.any():
        kw = {'c': ref.coords[fix, j], 'cmap': 'viridis'} if j is not None else {'color': 'C0'}
        sc = ax.scatter(r[fix], dS[fix, 0], s=18, label='S0', **kw)
        ax.scatter(r[fix], dS[fix].mean(1), s=18, marker='x', label='mean of S0-S3', **kw)
        if j is not None:
            fig.colorbar(sc, ax=ax, label='r(C–O) (Å)')
        ax.legend(fontsize=8)
    ax.axhline(0, color='k', lw=0.6)
    ax.set(xlabel='r(O–O) (Å)', ylabel=f'repaired − old {level} (kcal/mol)',
           title=f'{int(fix.sum())} points moved to the lower SA-CASSCF solution')
    if has_qd:
        from ..refset import RefSet
        q = RefSet.load(out['qd_nevpt2'] / 'refs.npz')
        s = RefSet.load(out[next(iter(out))] / 'refs.npz')
        ok = np.isfinite(q.energies).all(1)
        sh = (q.energies[ok] - np.sort(s.extra['E_nevpt2_eV'][ok], 1)[:, :q.n_states]) * EV_KCAL
        ax = axes[0, 1]
        for k in range(min(4, sh.shape[1])):
            ax.plot(r[ok], sh[:, k], 'o', ms=3, color=W.SCOL[k], label=f'S{k}', alpha=0.7)
        ax.axhline(0, color='k', lw=0.6)
        ax.legend(fontsize=8)
        ax.set(xlabel='r(O–O) (Å)', ylabel='QD − SC-NEVPT2 (kcal/mol)', title='How much QD-NEVPT2 moves each state')
    fig.tight_layout()
    fig.savefig(run.p('figures', 'R1_repair.png'), bbox_inches='tight', dpi=130)
    plt.close(fig)
