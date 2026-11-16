"""Method check along r(O-O) cuts of a ground-state grid: is a feature of the NEVPT2 surface real?

    python -m diabml.workflow.ch2oo method_cut RUN [--jobs 8]
        # needs RUN/data/gs_grid.npz (gs_grid stage) -> results/method_cut.{json,csv}, figures/MC1_method_cut.png

Why: on the 29_aug grid (state-specific CASSCF(8,6) + SC-NEVPT2 / aug-cc-pVDZ) the O-O curvature d2E/dr2 falls
smoothly at the CASSCF level, as a stretching bond's should, but the NEVPT2 one turns back up beyond r(O-O) ~ 1.42 A,
most at short C-O. The active space keeps the O-O sigma/sigma* pair along the whole cut, so a missing orbital is not
the cause. Two independent checks at the same geometries:

  CCSD(T)       single-reference coupled cluster, RHF reference, same basis; ORCA's T1 diagnostic says how far
                to trust it (T1 < ~0.02 for a closed-shell molecule: fine).
  FIC-NEVPT2    the fully internally contracted NEVPT2 on the same CASSCF (orbitals read from the grid's own
                converged .gbw at that point), instead of the strongly contracted SC-NEVPT2.

  QD-NEVPT2     (methods: qd_nevpt2) SA[qd_nroots]-CASSCF(8,6) + quasi-degenerate NEVPT2, started from the grid's
                orbitals: S0 when dynamic correlation may mix it with the next states. Also kept: SC-NEVPT2 on the
                state-averaged orbitals (separates the effect of averaging the orbitals from that of state mixing)
                and the QD S1-S0 gap.

Reading: CCSD(T) and FIC monotonic -> the SC contraction is at fault (use FIC); CCSD(T) monotonic, FIC turns up ->
the NEVPT2 correction on this reference (try a larger active space); CCSD(T) turns up too -> the feature is real
(extend the grid instead).

Cuts: every grid point with a_COO = [method_cut] a_COO (nearest grid value) and r_CO in [method_cut] r_CO (nearest
grid values), all r_OO; the SC-NEVPT2 and CASSCF energies there come from gs_grid.npz unchanged.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import re
import time
from pathlib import Path

import numpy as np

from . import audit as AU
from . import ch2oo as W
from .gs_grid import _geometry

HARTREE_EV = AU.HARTREE_EV
_FLOAT = r'(-?\d+\.\d+)'


def cc_input(atoms, basis, maxcore=2000):
    lines = ['# written by diabml method_cut: CCSD(T) check of the NEVPT2 surface',
             f'! CCSD(T) {basis} TightSCF', f'%maxcore {int(maxcore)}', '', '* xyz 0 1']
    lines += [f'{a:2s} {p[0]:15.9f} {p[1]:15.9f} {p[2]:15.9f}' for a, p in atoms]
    return '\n'.join(lines + ['*', ''])


def fic_input(atoms, spec, moread, maxcore=2000):
    """The grid's CASSCF with FIC-NEVPT2 instead of SC-NEVPT2."""
    t = AU.orca_input(atoms, spec, moread=moread, qd=False, maxcore=maxcore)
    if 'SC_NEVPT2' not in t:
        raise RuntimeError('unexpected ORCA input template (no SC_NEVPT2 line)')
    return t.replace('SC_NEVPT2', 'FIC_NEVPT2').replace('written by diabml audit', 'written by diabml method_cut')


def parse_cc(path):
    p = Path(path)
    if not p.exists():
        return None
    t = p.read_text(errors='replace')
    if 'TERMINATED NORMALLY' not in t:
        return None
    out = {}
    m = re.findall(r'FINAL SINGLE POINT ENERGY\s+' + _FLOAT, t)
    if m:
        out['E'] = float(m[-1])
    for key, pat in (('E_ccsd', r'E\(CCSD\)\s*\.*\s*' + _FLOAT), ('E_ccsd_t', r'E\(CCSD\(T\)\)\s*\.*\s*' + _FLOAT),
                     ('T1', r'T1 diagnostic\s*\.*\s*' + _FLOAT), ('E_scf', r'Total Energy\s*:\s*' + _FLOAT)):
        m = re.findall(pat, t)
        if m:
            out[key] = float(m[-1])
    if 'E_ccsd_t' in out:
        out['E'] = out['E_ccsd_t']
    return out if 'E' in out else None


def parse_nevpt2(path):
    """CASSCF root-0 and NEVPT2 total energies of a one-root run (any NEVPT2 flavour) and which line gave the PT2."""
    p = Path(path)
    if not p.exists():
        return None
    t = p.read_text(errors='replace')
    if 'TERMINATED NORMALLY' not in t or 'CASSCF RESULTS' not in t:
        return None
    i = t.index('CASSCF RESULTS')
    cas = re.findall(r'ROOT\s+0:\s+E=\s+' + _FLOAT, t[i:i + 40000])
    out = {'casscf': float(cas[0]) if cas else float('nan')}
    for how, pat in (('Total Energy (E0+dE)', r'Total Energy \(E0\+dE\)\s*:\s*E\s*=\s*' + _FLOAT),
                     ('E(NEVPT2)', r'E\(NEVPT2\)\s*[:=.]*\s*' + _FLOAT),
                     ('Total Energy Correction + E0', None)):
        if pat is None:
            d = re.findall(r'Total Energy Correction\s*:\s*dE\s*=\s*' + _FLOAT, t)
            if d and cas:
                out['nevpt2'], out['parsed_from'] = float(cas[0]) + float(d[-1]), how
            break
        m = re.findall(pat, t)
        if m:
            out['nevpt2'], out['parsed_from'] = float(m[-1]), how
            break
    return out if 'nevpt2' in out else None


def _run(exe, wd, name, timeout, check):
    if check(wd / f'{name}.out') is not None:
        return name, 'reused'
    return AU._run_one(exe, wd, name, timeout)


def _grid_gbw(gdir, k):
    """The converged orbitals the grid kept at point k: the start with the lower CASSCF energy."""
    best = None
    for out in sorted(gdir.glob(f'g{k:04d}_*.out')):
        r = AU.parse_orca(out, 1)
        g = out.with_suffix('.gbw')
        if r is not None and g.exists() and (best is None or r['casscf'][0] < best[0]):
            best = (r['casscf'][0], g)
    return None if best is None else str(best[1].resolve())


def cut_points(X, names, a_coo, r_co):
    ja, jc, jo = names.index('a_COO'), names.index('r_CO'), names.index('r_OO')
    ua = np.unique(np.round(X[:, ja], 6))
    a = ua[np.argmin(np.abs(ua - a_coo))]
    uc = np.unique(np.round(X[:, jc], 6))
    rcs = sorted({float(uc[np.argmin(np.abs(uc - r))]) for r in r_co})
    others = [j for j in range(len(names)) if j not in (ja, jc, jo)]
    cuts = {}
    for rc in rcs:
        m = np.isclose(X[:, ja], a) & np.isclose(X[:, jc], rc)
        for j in others:                                     # extra coordinates (a 4-D grid): their centre value
            u = np.unique(np.round(X[:, j], 6))
            m &= np.isclose(X[:, j], u[len(u) // 2])
        idx = np.where(m)[0]
        cuts[rc] = idx[np.argsort(X[idx, jo])]
    return float(a), cuts


def stage_method_cut(run, jobs=None, orca=None):
    from ..geometry import ch2oo_xyz
    mc, cc = run.cfg['method_cut'], run.cfg['compute']
    jobs = int(jobs or mc.get('jobs') or cc['jobs'])
    gp = run.p('data', 'gs_grid.npz')
    if not gp.exists():
        raise SystemExit(f'{gp} not found: method_cut checks the points of a gs_grid run')
    z = np.load(gp, allow_pickle=False)
    if 'E_nevpt2_hartree' not in z.files:
        raise SystemExit('method_cut compares against a NEVPT2 gs_grid (this grid was computed with another method)')
    X, names = z['coords'], [str(n) for n in z['coord_names']]
    spec = json.loads(str(z['spec']))
    fixed = json.loads(str(z['fixed']))
    basis = str(z['basis']) if 'basis' in z.files else spec['basis']
    a, cuts = cut_points(X, names, float(mc['a_COO']), [float(r) for r in mc['r_CO']])
    pts = [int(k) for idx in cuts.values() for k in idx]
    exe = AU._orca_exe(run, orca or cc.get('orca') or run.cfg['audit'].get('orca') or None)
    gdir, wd = run.p('data', 'gs_grid'), run.p('data', 'method_cut')
    wd.mkdir(exist_ok=True)
    methods = [m for m in mc['methods']]
    nqd = int(mc.get('qd_nroots', 3))
    unknown = [m for m in methods if m not in ('ccsd_t', 'fic_nevpt2', 'qd_nevpt2')]
    if unknown:
        raise SystemExit(f'[method_cut] methods: unknown {unknown}; use ccsd_t, fic_nevpt2, qd_nevpt2')
    run.log(f'{len(pts)} grid points: a_COO {a:g}, r_CO {", ".join(f"{r:g}" for r in cuts)}; r_OO '
            f'{X[pts, names.index("r_OO")].min():.3f}-{X[pts, names.index("r_OO")].max():.3f}; methods {methods} with '
            f'{basis}; ORCA {exe}, {jobs} parallel')
    tasks = []
    for k in pts:
        atoms = ch2oo_xyz(**_geometry(names, X[k], fixed))
        if 'ccsd_t' in methods:
            nm = f'cc{k:04d}'
            if not (wd / f'{nm}.inp').exists():
                (wd / f'{nm}.inp').write_text(cc_input(atoms, basis, cc['maxcore']))
            tasks.append((k, 'ccsd_t', nm))
        if 'qd_nevpt2' in methods:
            g = _grid_gbw(gdir, k)
            if g is None:
                run.log(f'  point {k}: no converged grid orbitals; QD skipped')
            else:
                nm = f'qd{k:04d}'
                if not (wd / f'{nm}.inp').exists():
                    (wd / f'{nm}.inp').write_text(qd_input(atoms, spec, g, nqd, cc['maxcore']))
                tasks.append((k, 'qd_nevpt2', nm))
        if 'fic_nevpt2' in methods:
            g = _grid_gbw(gdir, k)
            if g is None:
                run.log(f'  point {k}: no converged grid orbitals; FIC skipped')
            else:
                nm = f'fic{k:04d}'
                if not (wd / f'{nm}.inp').exists():
                    (wd / f'{nm}.inp').write_text(fic_input(atoms, spec, g, cc['maxcore']))
                tasks.append((k, 'fic_nevpt2', nm))
    check = {'ccsd_t': parse_cc, 'fic_nevpt2': parse_nevpt2, 'qd_nevpt2': lambda p: parse_qd(p, nqd)}
    jo = names.index('r_OO')
    # one point of each method first: stop early if ORCA rejects an input or the output is not understood
    mid = list(cuts)[len(cuts) // 2]
    k0 = int(cuts[mid][len(cuts[mid]) // 2])
    first = [t for t in tasks if t[0] == k0]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        list(ex.map(lambda t: _run(exe, wd, t[2], float(cc['timeout_s']), check[t[1]]), first))
    for k, m, nm in first:
        r = check[m](wd / f'{nm}.out')
        if r is None:
            raise SystemExit(f'{m} at the first point failed or its output was not understood: see {wd / nm}.out')
        e_sc, e_cas = z['E_nevpt2_hartree'][k, 0], z['E_casscf_hartree'][k, 0]
        if m == 'fic_nevpt2':
            run.log(f'  check at r_OO {X[k, jo]:.3f}: CASSCF {r["casscf"]:.6f} (grid {e_cas:.6f}, '
                    f'{(r["casscf"] - e_cas) * 1000:+.3f} mEh); FIC-NEVPT2 {r["nevpt2"]:.6f} vs SC {e_sc:.6f} '
                    f'({(r["nevpt2"] - e_sc) * 1000:+.2f} mEh; read from "{r["parsed_from"]}")')
            if abs(r['casscf'] - e_cas) > 1e-4:
                run.log('  warning: the FIC run did not reproduce the grid CASSCF: a different solution')
        elif m == 'qd_nevpt2':
            run.log(f'  check at r_OO {X[k, jo]:.3f}: SA{nqd}-CASSCF root 0 {r["casscf"][0]:.6f}; SC-NEVPT2 on the '
                    f'SA orbitals {r["sc"][0]:.6f}, QD-NEVPT2 S0 {r["qd"][0]:.6f} (state-specific SC {e_sc:.6f}); '
                    f'QD S1-S0 {(r["qd"][1] - r["qd"][0]) * HARTREE_EV:.2f} eV')
        else:
            run.log(f'  check at r_OO {X[k, jo]:.3f}: CCSD(T) {r["E"]:.6f}' +
                    (f', T1 {r["T1"]:.4f}' if 'T1' in r else ', no T1 diagnostic found'))
    rest = [t for t in tasks if t[0] != k0]
    done = 0
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(_run, exe, wd, t[2], float(cc['timeout_s']), check[t[1]]) for t in rest]
        for f in cf.as_completed(futs):
            f.result()
            done += 1
            if done % 10 == 0 or done == len(rest):
                run.log(f'  {done + len(first)}/{len(tasks)} runs ({(time.time() - t0) / 60:.1f} min)')
    rows, res = [], {}
    for k, m, nm in tasks:
        r = check[m](wd / f'{nm}.out')
        res.setdefault(k, {})[m] = r
    for rc, idx in cuts.items():
        for k in idx:
            r = res.get(int(k), {})
            rows.append({'k': int(k), 'r_OO': float(X[k, jo]), 'r_CO': rc, 'a_COO': a,
                         'casscf': float(z['E_casscf_hartree'][k, 0]), 'sc_nevpt2': float(z['E_nevpt2_hartree'][k, 0]),
                         'fic_nevpt2': (r.get('fic_nevpt2') or {}).get('nevpt2', float('nan')),
                         'fic_casscf': (r.get('fic_nevpt2') or {}).get('casscf', float('nan')),
                         'ccsd_t': (r.get('ccsd_t') or {}).get('E', float('nan')),
                         'sa_sc_nevpt2': _qd(r, 'sc', 0), 'qd_nevpt2': _qd(r, 'qd', 0),
                         'qd_gap_eV': (_qd(r, 'qd', 1) - _qd(r, 'qd', 0)) * HARTREE_EV,
                         'T1': (r.get('ccsd_t') or {}).get('T1', float('nan'))})
    summary = _summarize(rows, cuts)
    out = {'a_COO': a, 'r_CO': list(cuts), 'basis': basis, 'methods': methods, 'rows': rows, 'summary': summary,
           'minutes': (time.time() - t0) / 60}
    run.p('results', 'method_cut.json').write_text(json.dumps(out, indent=1, default=float))
    with open(run.p('results', 'method_cut.csv'), 'w') as fh:
        keys = list(rows[0])
        fh.write(','.join(keys) + '\n')
        for r in rows:
            fh.write(','.join(f'{r[c]:.8f}' if isinstance(r[c], float) else str(r[c]) for c in keys) + '\n')
    for rc, s in summary.items():
        run.log(f'r_CO {rc}: ' + '; '.join(f'{m} curvature min {v["min"]:.1f} at {v["r_at_min"]:.3f}, last '
                                           f'{v["last"]:.1f} eV/A2 (upturn {v["upturn"]:+.1f})'
                                           for m, v in s.items()))
    t1 = [r['T1'] for r in rows if np.isfinite(r['T1'])]
    if t1:
        run.log(f'CCSD(T) T1 diagnostic {min(t1):.4f}-{max(t1):.4f}')
    _figure(run, rows, cuts, basis)
    run.log(f'wrote results/method_cut.json, results/method_cut.csv, figures/MC1_method_cut.png')


def _qd(r, key, i):
    q = r.get('qd_nevpt2')
    return float(q[key][i]) if q and len(q.get(key, [])) > i else float('nan')


def qd_input(atoms, spec, moread, nroots, maxcore=2000):
    """SA(nroots)-CASSCF + SC- and QD-NEVPT2 started from the grid's state-specific orbitals."""
    t = AU.orca_input(atoms, dict(spec, nroots=int(nroots)), moread=moread, qd=True, maxcore=maxcore)
    return t.replace('written by diabml audit', 'written by diabml method_cut (QD-NEVPT2)')


def parse_qd(path, nroots):
    """{'casscf', 'sc', 'qd'} with QD energies sorted (S0 first), or None."""
    r = AU.parse_orca(path, nroots)
    if r is None or len(r.get('qd', [])) < 2:
        return None
    r['qd'] = sorted(r['qd'])
    return r


_LAB = {'casscf': 'CASSCF(8,6)', 'sc_nevpt2': 'SC-NEVPT2', 'fic_nevpt2': 'FIC-NEVPT2', 'ccsd_t': 'CCSD(T)',
        'sa_sc_nevpt2': 'SC-NEVPT2, SA orbitals', 'qd_nevpt2': 'QD-NEVPT2 S0'}


def _curv(rows, rc, m):
    rr = [r for r in rows if r['r_CO'] == rc]
    x = np.array([r['r_OO'] for r in rr])
    e = np.array([r[m] for r in rr]) * HARTREE_EV
    ok = np.isfinite(e)
    x, e = x[ok], e[ok]
    if len(x) < 3:
        return np.zeros(0), np.zeros(0)
    h = np.diff(x)
    # second derivative on a (nearly) uniform line; gaps from failed points are skipped
    d2, xc = [], []
    for i in range(1, len(x) - 1):
        if abs(h[i - 1] - h[i]) < 1e-6:
            d2.append((e[i + 1] - 2 * e[i] + e[i - 1]) / h[i] ** 2)
            xc.append(x[i])
    return np.array(xc), np.array(d2)


def _summarize(rows, cuts):
    out = {}
    for rc in cuts:
        s = {}
        for m in _LAB:
            xc, d2 = _curv(rows, rc, m)
            if len(d2) < 3:
                continue
            i = int(np.argmin(d2))
            s[m] = {'min': float(d2[i]), 'r_at_min': float(xc[i]), 'last': float(d2[-1]),
                    'upturn': float(d2[-1] - d2[i]), 'r': xc.tolist(), 'd2': d2.tolist()}
        out[rc] = s
    return out


def _figure(run, rows, cuts, basis):
    plt = W._mpl()
    sty = {'casscf': dict(color='0.45', ls='--', lw=1.4), 'sc_nevpt2': dict(color='#B03A2E', lw=2.2, marker='o', ms=4),
           'fic_nevpt2': dict(color='#2F6DB5', lw=2.2, marker='s', ms=4),
           'ccsd_t': dict(color='#2E9A6B', lw=2.2, marker='^', ms=5),
           'sa_sc_nevpt2': dict(color='#C8873A', lw=1.6, ls=':', marker='x', ms=5),
           'qd_nevpt2': dict(color='#6B4FA0', lw=2.2, marker='D', ms=4)}
    fig, axes = plt.subplots(1, len(cuts), figsize=(5.2 * len(cuts), 4.6), sharey=True, squeeze=False)
    for ax, rc in zip(axes[0], cuts):
        for m, st in sty.items():
            xc, d2 = _curv(rows, rc, m)
            if len(d2):
                ax.plot(xc, d2, label=_LAB[m], **st)
        ax.set(title=f'r(C–O) {rc:g} Å', xlabel='r(O–O) (Å)')
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel('d²E/dr(O–O)²  (eV/Å²)')
    axes[0][0].legend(fontsize=9, frameon=False)
    fig.suptitle(f'O–O curvature along r(O–O) cuts, {basis}: is the NEVPT2 upturn real?', fontsize=12)
    fig.tight_layout()
    fig.savefig(run.p('figures', 'MC1_method_cut.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
