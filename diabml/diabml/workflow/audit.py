"""Audit the reference data with an independent program: is every point on the right SA-CASSCF solution?

    python -m diabml.workflow.ch2oo audit RUN --orca /path/to/orca_folder --jobs 6

For every geometry in RUN/data/refs.npz, ORCA computes SA-CASSCF at the level the data were made with (same basis,
active space, number of roots, Cs symmetry, singlet A') plus state-specific SC-NEVPT2 and QD-NEVPT2 (van Vleck), all
electrons correlated as in PySCF. The runs are independent, so `jobs` of them go in parallel. Each point is then
classified by comparing the four SA-CASSCF root energies:

  same          ORCA reproduces our energies (largest difference below tol_uEh): same solution, so ORCA's QD-NEVPT2
                can replace our SC-NEVPT2 directly
  ours higher   ORCA found a solution with a lower state-averaged energy: our point is on a higher (wrong) solution
  orca higher   ORCA landed on a higher solution: it is retried starting from the orbitals of the nearest point where
                ORCA's solution is at least as good as ours (MORead), up to `retries` times
  failed        ORCA did not finish or did not converge

Writes results/audit.npz and results/audit.csv (energies from both programs, classification, retries) and
RUN/audit.html (counts, maps of where the bad points are, the far-field check, and the QD-NEVPT2 shifts). Resumable:
finished ORCA runs (data/audit/*.out) are reused.
"""
from __future__ import annotations

import base64
import concurrent.futures as cf
import html
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import ch2oo as W

HARTREE_EV = 27.211386245988
KCAL = 627.509474


# ------------------------------------------------------------------------------------------------ ORCA input / output
def orca_input(atoms, spec, moread=None, qd=True, maxcore=2000):
    """SA-CASSCF (+ SC-NEVPT2, QD-NEVPT2) input matching a diabml CASSpec."""
    irrep = {"A'": 0, 'A"': 1, "A''": 1}.get(spec.get('wfnsym') or "A'", 0)
    sym = ' UseSym' if spec.get('symmetry') else ''
    pt = ('  PTMethod SC_NEVPT2\n' + ('  PTSettings\n    QDType QD_VanVleck\n  end\n' if qd else '')) if qd is not None else ''
    lines = [f"# written by diabml audit: SA{spec['nroots']}-CASSCF({spec['nelecas']},{spec['ncas']})/{spec['basis']}",
             f"! {spec['basis']} NoFrozenCore TightSCF{sym}" + (' MORead' if moread else ''),
             f'%moinp "{moread}"' if moread else '',
             f'%maxcore {int(maxcore)}',
             '%casscf',
             f"  nel    {spec['nelecas']}", f"  norb   {spec['ncas']}", '  mult   1',
             f'  irrep  {irrep}' if spec.get('symmetry') else '', f"  nroots {spec['nroots']}",
             f"  maxiter {max(100, int(spec.get('max_cycle', 100)))}",
             pt.rstrip('\n'), 'end', '', '* xyz 0 1']
    lines += [f'{a:2s} {p[0]:15.9f} {p[1]:15.9f} {p[2]:15.9f}' for a, p in atoms]
    lines += ['*', '']
    return '\n'.join(l for l in lines if l is not None)


def parse_orca(path, nroots=4):
    """{'casscf', 'sc', 'qd'} (hartree; casscf and qd in energy order, sc per CASSCF root) or None if it failed."""
    p = Path(path)
    if not p.exists():
        return None
    t = p.read_text(errors='replace')
    # ORCA sometimes aborts *after* the energies, in the QD-NEVPT2 property step (UV/CD spectra): the energies are
    # complete, so accept the file when every section we read is there
    if 'CASSCF RESULTS' not in t or 'CAS-SCF GRADIENT HAS CONVERGED' not in t:
        return None
    if 'TERMINATED NORMALLY' not in t and 'QD-NEVPT2 TOTAL ENERGIES' not in t and 'NEVPT2 TOTAL ENERGIES' not in t:
        return None
    i = t.index('CASSCF RESULTS')
    cas = [float(x) for x in re.findall(r'ROOT\s+\d+:\s+E=\s+(-?\d+\.\d+)', t[i:i + 40000])[:nroots]]
    out = {'casscf': cas, 'sc': [], 'qd': []}
    if 'NEVPT2 Results' in t:
        j = t.index('NEVPT2 Results')
        k = t.index('QD-NEVPT2 Results') if 'QD-NEVPT2 Results' in t else len(t)
        out['sc'] = [float(x) for x in re.findall(r'Total Energy \(E0\+dE\)\s+:\s+E\s+=\s+(-?\d+\.\d+)', t[j:k])][:nroots]
    if 'QD-NEVPT2 TOTAL ENERGIES' in t:
        m = t.index('QD-NEVPT2 TOTAL ENERGIES')
        out['qd'] = [float(x) for x in re.findall(r"^\s*\d+:\s+\d+\s+\d+\s+\S+\s+(-?\d+\.\d+)", t[m:m + 4000], re.M)][:nroots]
    if len(cas) < nroots or (out['qd'] == [] and 'TERMINATED NORMALLY' not in t):
        return None
    out['terminated_normally'] = 'TERMINATED NORMALLY' in t
    return out


def _orca_exe(run, given=None):
    cand = given or run.cfg['audit'].get('orca') or os.environ.get('ORCA_PATH') or os.environ.get('ORCA')
    if not cand:
        raise SystemExit('where is ORCA? pass --orca /path/to/orca_folder, set [audit] orca = "..." in config.toml, or '
                         'export ORCA_PATH=/path/to/orca_folder')
    p = Path(cand).expanduser()
    exe = p / 'orca' if p.is_dir() else p
    if not exe.exists():
        raise SystemExit(f'{exe} does not exist')
    return exe.resolve()


def _run_one(exe, workdir, name, timeout):
    """Run ORCA on workdir/name.inp (full path to the executable, as ORCA's parallel mode requires)."""
    out = workdir / f'{name}.out'
    if out.exists() and parse_orca(out) is not None:
        return name, 'reused'
    env = dict(os.environ, OMP_NUM_THREADS='1')
    t0 = time.time()
    with open(out, 'w') as fo, open(workdir / f'{name}.err', 'w') as fe:
        try:
            cmd = ([sys.executable, str(exe)] if str(exe).endswith('.py') else [str(exe)]) + [f'{name}.inp']
            subprocess.run(cmd, cwd=workdir, stdout=fo, stderr=fe, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            return name, 'timeout'
    for junk in workdir.glob(f'{name}.*tmp*'):
        try:
            junk.unlink()
        except OSError:
            pass
    return name, f'{time.time() - t0:.0f} s'


# ------------------------------------------------------------------------------------------------ stage
def classify(e_orca, e_ours, tol_eh):
    """same / ours higher / orca higher, from the four SA-CASSCF energies (each sorted)."""
    a, b = np.sort(e_orca), np.sort(e_ours)
    dev = float(np.max(np.abs(a - b)))
    if dev <= tol_eh:
        return 'same', dev, float(np.mean(a - b))
    d_avg = float(np.mean(a - b))                      # the state-averaged energy is what SA-CASSCF minimises
    return ('ours higher' if d_avg < 0 else 'orca higher'), dev, d_avg


def stage_audit(run, orca=None, jobs=None):
    from ..geometry import ch2oo_xyz
    from ..refset import RefSet
    ac = run.cfg['audit']
    exe = _orca_exe(run, orca)
    jobs = int(jobs or ac['jobs'])
    ref = RefSet.load(run.p('data', 'refs.npz'))          # raw: energies per SA-CASSCF root, unsorted
    spec = ref.meta['spec']
    zero = ref.meta['zero_hartree']
    Ec = zero['casscf'] + np.asarray(ref.extra['E_casscf_eV'], float) / HARTREE_EV
    En = (zero['nevpt2'] + np.asarray(ref.extra['E_nevpt2_eV'], float) / HARTREE_EV) if 'E_nevpt2_eV' in ref.extra else None
    N, nr = Ec.shape
    tol = float(ac['tol_uEh']) * 1e-6
    wd = run.p('data', 'audit')
    wd.mkdir(exist_ok=True)
    sc = np.array([{'r_OO': 0.1, 'a_COO': 10.0, 'r_CO': 0.05}.get(n, 1.0) for n in ref.coord_names])
    pts = [i for i in range(N) if np.all(np.isfinite(Ec[i]))]
    run.log(f'{len(pts)} geometries; ORCA {exe}; {jobs} parallel jobs; SA{spec["nroots"]}-CASSCF('
            f'{spec["nelecas"]},{spec["ncas"]})/{spec["basis"]} + SC-NEVPT2 + QD-NEVPT2; same solution if every root '
            f'agrees within {ac["tol_uEh"]} µEh')
    atoms = {i: ch2oo_xyz(**W._geometry_kwargs(ref, ref.coords[i])) for i in pts}
    name = lambda i, k=0: f'p{i:04d}' + (f'_r{k}' if k else '')

    def batch(todo, label):
        t0 = time.time()
        done = 0
        with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(_run_one, exe, wd, nm, float(ac['timeout_s'])) for nm in todo]
            for f in cf.as_completed(futs):
                done += 1
                if done % 25 == 0 or done == len(todo):
                    el = time.time() - t0
                    run.log(f'  {label}: {done}/{len(todo)} done, {el / 60:.1f} min, about '
                            f'{el / done * (len(todo) - done) / 60:.0f} min left')

    # pass 1: every point from ORCA's own guess
    for i in pts:
        f = wd / f'{name(i)}.inp'
        if not f.exists():
            f.write_text(orca_input(atoms[i], spec, qd=bool(ac['qd']), maxcore=ac['maxcore']))
    batch([name(i) for i in pts], 'pass 1')
    res, attempt = {}, {}
    for i in pts:
        res[i], attempt[i] = parse_orca(wd / f'{name(i)}.out', nr), 0
    status = {}

    def judge(i):
        r = res[i]
        if r is None:
            return 'failed', np.nan, np.nan
        return classify(r['casscf'], Ec[i], tol)

    for i in pts:
        status[i] = judge(i)
    # retries: start from the orbitals of the nearest point where ORCA's solution is at least as good as ours
    for k in range(1, int(ac['retries']) + 1):
        good = [j for j in pts if status[j][0] in ('same', 'ours higher')]
        redo = [i for i in pts if status[i][0] in ('orca higher', 'failed')]
        if not redo or not good:
            break
        G = ref.coords[good] / sc
        todo = []
        for i in redo:
            d = np.sum((G - ref.coords[i] / sc) ** 2, 1)
            j = good[int(np.argmin(d))]
            src = wd / f'{name(j, attempt[j])}.gbw'
            if not src.exists():
                continue
            f = wd / f'{name(i, k)}.inp'
            if not f.exists():
                f.write_text(orca_input(atoms[i], spec, moread=str(src), qd=bool(ac['qd']), maxcore=ac['maxcore']))
            todo.append((i, name(i, k)))
        run.log(f'retry {k}: {len(todo)} points from the orbitals of their nearest good neighbour')
        batch([nm for _, nm in todo], f'retry {k}')
        for i, nm in todo:
            r = parse_orca(wd / f'{nm}.out', nr)
            if r is None:
                continue
            new = classify(r['casscf'], Ec[i], tol)
            old = status[i]
            better = old[0] == 'failed' or np.mean(r['casscf']) < np.mean(res[i]['casscf']) - tol
            if better:
                res[i], attempt[i], status[i] = r, k, new
    # tables
    labels = np.array([status[i][0] if i in status else 'skipped' for i in range(N)], dtype=object)
    dev = np.array([status[i][1] if i in status else np.nan for i in range(N)])
    davg = np.array([status[i][2] if i in status else np.nan for i in range(N)])
    def arr(key):
        a = np.full((N, nr), np.nan)
        for i in pts:
            if res[i] is not None and len(res[i][key]) == nr:
                a[i] = res[i][key] if key != 'casscf' else np.sort(res[i][key])
        return a
    o_cas, o_sc, o_qd = arr('casscf'), arr('sc'), arr('qd')
    np.savez(run.p('results', 'audit.npz'), status=labels.astype(str), casscf_dev_Eh=dev, casscf_avg_diff_Eh=davg,
             orca_casscf_Eh=o_cas, orca_sc_nevpt2_Eh=o_sc, orca_qd_nevpt2_Eh=o_qd, ours_casscf_Eh=Ec,
             ours_sc_nevpt2_Eh=En if En is not None else np.full_like(Ec, np.nan), coords=ref.coords,
             coord_names=np.array(ref.coord_names), attempt=np.array([attempt.get(i, -1) for i in range(N)]),
             al_round=ref.extra.get('al_round', np.zeros(N, int)))
    with open(run.p('results', 'audit.csv'), 'w') as fh:
        fh.write(','.join(['index'] + ref.coord_names + ['status', 'retry', 'casscf_max_dev_mEh', 'casscf_avg_diff_mEh']
                          + [f'qd_minus_sc_S{s}_kcal' for s in range(nr)]) + '\n')
        for i in range(N):
            if i not in status:
                continue
            q = (o_qd[i] - np.sort(o_sc[i])) * KCAL if np.isfinite(o_qd[i]).all() and np.isfinite(o_sc[i]).all() else [np.nan] * nr
            fh.write(','.join([str(i)] + [f'{v:.4f}' for v in ref.coords[i]] + [status[i][0], str(attempt[i]),
                              f'{dev[i] * 1e3:.3f}', f'{davg[i] * 1e3:.3f}'] + [f'{v:.3f}' for v in q]) + '\n')
    counts = {s: int(np.sum(labels == s)) for s in ('same', 'ours higher', 'orca higher', 'failed')}
    run.log('classification: ' + ', '.join(f'{v} {k}' for k, v in counts.items()))
    summary = {'counts': counts, 'n': len(pts), 'tol_uEh': float(ac['tol_uEh']), 'orca': str(exe)}
    run.p('results', 'audit.json').write_text(json.dumps(summary, indent=1))
    write_audit_html(run, ref, labels, dev, davg, o_cas, o_sc, o_qd, Ec, En, summary)
    run.log(f'wrote results/audit.npz, results/audit.csv and {run.p("audit.html")}')


# ------------------------------------------------------------------------------------------------ report
COL = {'same': '#2E9A6B', 'ours higher': '#B03A2E', 'orca higher': '#E06C2B', 'failed': '#7F8C8D'}


def _png(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def write_audit_html(run, ref, labels, dev, davg, o_cas, o_sc, o_qd, Ec, En, summary):
    plt = W._mpl()
    X, names = ref.coords, ref.coord_names
    figs = []
    # A: where the points are, by class
    pairs = [(0, j) for j in range(1, len(names))]
    fig, axes = plt.subplots(1, len(pairs), figsize=(6.6 * len(pairs), 4.6), squeeze=False)
    for ax, (i, j) in zip(axes.flat, pairs):
        for k in ('same', 'orca higher', 'failed', 'ours higher'):
            m = labels == k
            if m.any():
                ax.scatter(X[m, i], X[m, j], s=14 if k != 'same' else 7, color=COL[k], alpha=0.85,
                           label=f'{k} ({int(m.sum())})', marker='o' if k != 'ours higher' else 'X')
        ax.set(xlabel=names[i], ylabel=names[j], xscale='log')
    axes.flat[0].legend(fontsize=9)
    figs.append(('AU1_where', 'Every reference geometry, coloured by the audit. Red crosses: ORCA found a lower SA-CASSCF '
                 'solution, so our point is on a higher (wrong) one. Orange: ORCA itself stayed on a higher solution '
                 'after retries. Log scale in r_OO.', _png(fig)))
    # B: by r_CO and angle slices, fraction bad
    if 'r_CO' in names:
        jc = names.index('r_CO')
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        bins = np.array([0.95, 1.15, 1.25, 1.3, 1.375, 1.425, 1.5])
        idx = np.digitize(X[:, jc], bins)
        frac, lab = [], []
        for b in range(1, len(bins)):
            m = (idx == b) & np.isin(labels, ['same', 'ours higher', 'orca higher'])
            if m.any():
                frac.append(np.mean(labels[m] == 'ours higher')); lab.append(f'{bins[b-1]:.2f}–{bins[b]:.2f}\n(n={m.sum()})')
        ax.bar(range(len(frac)), frac, color=COL['ours higher'])
        ax.set_xticks(range(len(frac)), lab, fontsize=8)
        ax.set(xlabel='r(C–O) (Å)', ylabel='fraction of points on a higher solution')
        figs.append(('AU2_by_rCO', 'Fraction of our points that sit on a higher SA-CASSCF solution, by C–O bond length.',
                     _png(fig)))
    # C: size of the error
    m = np.isin(labels, ['ours higher'])
    if m.any():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        Eo = np.sort(Ec[m], 1)
        d = (o_cas[m] - Eo) * HARTREE_EV
        for s in range(d.shape[1]):
            axes[0].hist(d[:, s], bins=30, alpha=0.6, label=f'root {s}')
        axes[0].set(xlabel='ORCA − ours, SA-CASSCF (eV)', ylabel='points', title='How much lower the better solution is')
        axes[0].legend(fontsize=8)
        if En is not None:
            dd = (np.sort(o_sc[m], 1) - np.sort(En[m], 1)) * KCAL
            for s in range(dd.shape[1]):
                axes[1].hist(dd[:, s], bins=30, alpha=0.6, label=f'S{s}')
            axes[1].set(xlabel='ORCA − ours, SC-NEVPT2 sorted (kcal/mol)', title='Effect on the energies we train on')
            axes[1].legend(fontsize=8)
        figs.append(('AU3_size', 'At the points on a higher solution: how far the better solution lies below ours '
                     '(per root), and what that does to the sorted SC-NEVPT2 energies.', _png(fig)))
    # D: far field check (three O(1D) components should be degenerate whatever the angle)
    far = X[:, 0] >= 3.5
    if far.any() and 'r_CO' in names:
        jc = names.index('r_CO')
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, (E, title) in zip(axes, [(Ec, 'ours'), (o_cas, 'ORCA')]):
            Es = np.sort(E[far], 1)
            for s in (2, 3):
                ax.plot(X[far, jc], (Es[:, s] - Es[:, 0]) * HARTREE_EV, 'o', ms=4, alpha=0.7, label=f'S{s} − S0')
            ax.set(xlabel='r(C–O) (Å)', ylabel='SA-CASSCF gap (eV)', title=f'r_OO ≥ 3.5 Å, every angle: {title}')
            ax.legend(fontsize=8)
        figs.append(('AU4_far_field', 'Far from the molecule the states cannot depend on the C–O–O angle: at each C–O '
                     'length the gaps must be single-valued. Two values at one r(C–O) means two different solutions.',
                     _png(fig)))
    # E: QD-NEVPT2 shifts where the solutions agree
    m = (labels == 'same') & np.isfinite(o_qd).all(1) & np.isfinite(o_sc).all(1)
    if m.any():
        q = (o_qd[m] - np.sort(o_sc[m], 1)) * KCAL
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for s in range(q.shape[1]):
            axes[0].plot(X[m, 0], q[:, s], 'o', ms=3, alpha=0.6, label=f'S{s}')
        axes[0].set(xlabel=names[0], ylabel='QD − SC (sorted), kcal/mol', xscale='log',
                    title='QD-NEVPT2 against state-specific SC-NEVPT2')
        axes[0].legend(fontsize=8)
        axes[1].hist(np.abs(q).max(1), bins=40, color=COL['same'])
        axes[1].set(xlabel='largest |QD − SC| over the states (kcal/mol)', ylabel='points', yscale='log')
        figs.append(('AU5_qd_shift', 'Where ORCA and our data share the solution: how much QD-NEVPT2 moves each state '
                     'compared with the state-specific energies (sorted). Large shifts mark near-degeneracies where '
                     'state-specific NEVPT2 is unreliable.', _png(fig)))
    c = summary['counts']
    bad = np.where(labels == 'ours higher')[0]
    rows = ''.join(f'<tr><td>{i}</td>' + ''.join(f'<td>{v:.3f}</td>' for v in X[i]) +
                   f'<td>{davg[i] * 1e3:.1f}</td><td>{dev[i] * 1e3:.1f}</td>'
                   f'<td>{int(ref.extra["al_round"][i]) if "al_round" in ref.extra else 0}</td></tr>'
                   for i in bad[np.argsort(davg[bad])])
    esc = html.escape
    figs_html = ''.join(f'<figure><img src="data:image/png;base64,{b}" alt="{esc(t)}"><figcaption><b>{n}.</b> {esc(t)}'
                        f'</figcaption></figure>' for n, t, b in figs)
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reference audit</title><style>
:root{{--paper:#F6F4F0;--ink:#2E2A36;--muted:#7A7480;--rule:#DAD3C8;--plum:#3D2C58;--copper:#C8873A;--panel:#EFEBF3}}
@media (prefers-color-scheme:dark){{:root{{--paper:#17131f;--ink:#ece8f2;--muted:#a69cb6;--rule:#3d3250;--panel:#241b33}}}}
body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,-apple-system,sans-serif}}
header{{background:var(--plum);color:#fff;border-bottom:5px solid var(--copper);padding:24px}}
header h1{{margin:0;font:600 28px Georgia,serif}} header p{{margin:6px 0 0;color:#E9B872}}
main{{max-width:1180px;margin:0 auto;padding:16px}} h2{{font:600 22px Georgia,serif;border-bottom:1px solid var(--rule)}}
.tw{{overflow-x:auto;max-height:520px}} table{{border-collapse:collapse;width:100%;font-size:14px}} th,td{{border-bottom:1px solid var(--rule);padding:5px 8px;text-align:left}}
th{{background:var(--panel);position:sticky;top:0}} img{{max-width:100%;background:#fff;border:1px solid var(--rule)}} figcaption{{color:var(--muted);font-size:14px}}
.k{{display:inline-block;margin:0 18px 8px 0}} .k b{{font-size:26px;display:block}}
</style></head><body><header><h1>Reference audit: {esc(run.path.name)}</h1><p>ORCA SA-CASSCF / SC-NEVPT2 / QD-NEVPT2 against the diabml reference data · {time.strftime('%Y-%m-%d %H:%M')}</p></header><main>
<p>{summary["n"]} geometries. "Same" means all four SA-CASSCF roots agree within {summary["tol_uEh"]:.0f} µEh. "Ours higher" means ORCA found a solution with a lower state-averaged energy: our point is on a wrong (higher) solution and its energies, charges and couplings describe other states.</p>
<div><span class="k"><b style="color:{COL['same']}">{c['same']}</b>same solution</span><span class="k"><b style="color:{COL['ours higher']}">{c['ours higher']}</b>ours on a higher solution</span><span class="k"><b style="color:{COL['orca higher']}">{c['orca higher']}</b>ORCA on a higher solution</span><span class="k"><b style="color:{COL['failed']}">{c['failed']}</b>ORCA failed</span></div>
<h2>Figures</h2>{figs_html}
<h2>Points on a higher solution ({len(bad)})</h2><div class="tw"><table><thead><tr><th>index</th>{''.join(f'<th>{n}</th>' for n in names)}<th>ORCA − ours, state average (mEh)</th><th>largest root difference (mEh)</th><th>AL round</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>Full table: results/audit.csv. Energies from both programs: results/audit.npz.</p></main></body></html>'''
    run.p('audit.html').write_text(page)
