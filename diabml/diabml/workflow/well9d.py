"""The CH2OO well in all nine internal coordinates at CCSD(T): local Hessian, sampled batches, a symmetric polynomial.

    python -m diabml.workflow.ch2oo well9d     RUN --batch 0 [-v]   # finite-difference Hessian only (163 points)
    python -m diabml.workflow.ch2oo well9d     RUN --batch 1 [-v]   # Hessian if missing, then batch 1 of the samples
    python -m diabml.workflow.ch2oo well9d     RUN [-v]             # everything ([well9d] select = "fps" only)
    python -m diabml.workflow.ch2oo well9d_fit RUN [-v]             # fit, harmonic modes, checks; nothing recomputed
    python -m diabml.workflow.ch2oo well9d_tz  RUN [-v]             # aug-cc-pVTZ at a subset: the 9-D basis correction
    python -m diabml.workflow.ch2oo well9d_vpt2 RUN [-v]            # anharmonic fundamentals (VPT2 + Fermi resonances)
    python -m diabml.workflow.ch2oo well9d_dipole RUN [-v]          # finite-field CCSD(T) dipoles -> models/dipole9d.npz

-v / --verbose prints the detail lines (mode tables, per-batch statistics, per-order fit errors) to the terminal as
well; they always go to logs/well9d.log and logs/well9d_fit.log.

Why. The 4-D surface (r_OO, angle COO, r_CO, symmetric angle HCO; C-H fixed at the grid value, CH2 held symmetric
and planar) gets the O-O and C-O bands, but five modes are restraint springs in the dynamics. A hot molecule (the
first study: IR emission from CH2OO formed with about 13 kJ/mol above its zero point after CH2I2 photolysis in O2)
shares its energy over all nine modes, emits strongly in the C-H stretch region, and its scissor, bend and
rotational constants depend on the coordinates the 4-D model freezes. A product grid in nine coordinates is out of
reach (11^9 points), so the well is sampled instead.

What it does.

1. Centre ([well9d] center, else the minimum of the frozen 4-D benchmark surface, else the lowest gs_grid point) in
   the nine coordinates of diabml.geometry.NINE: r_OO, a_COO, r_CO, r_CH1, r_CH2, a_HCO1, a_HCO2, d_H1, d_H2 (1 = the
   H syn to the terminal O; d_H* = out-of-plane dihedrals, 0 when planar). The C-H bonds start at the grid's fixed
   value, both HCO angles at the 4-D minimum, the molecule planar.

2. Hessian (batch 0). CCSD(T) at the centre and at +-h along every coordinate and the four (+-h, +-h) corners of
   every pair: 1 + 2*9 + 4*36 = 163 points, steps [well9d] fd_step (0.01 A, 1 deg, 2 deg). Central differences give
   the gradient g and force constants K (eV per A^2 / rad^2); with the Wilson G matrix at the centre they give the
   nine CCSD(T) harmonic frequencies directly (results/well9d_hessian.json). The Newton step -K^-1 g shows how far
   the centre is from the 9-D minimum. The mirror symmetry is checked: E(+d) and E(-d) must agree.

3. Samples (batches 1, 2, ...). A pool of [well9d] n_pool candidates in the normal coordinates Q of that Hessian
   (dx = L Q with L^T G^-1 L = 1, so the harmonic energy is sum_k lambda_k Q_k^2 / 2): a harmonic energy
   E = e_max * u^e_power (u uniform) and a random direction, so the pool fills the harmonic ellipsoid up to e_max
   evenly in energy. Candidates outside |d_H*| <= d_max or with unphysical bonds and angles are dropped. Then
   farthest-point sampling in the energy-scaled coordinates sqrt(lambda_k) Q_k (equal steps = equal harmonic energy)
   picks n_points, split round-robin into `batches`, so each batch spans the whole well at a coarser density. With
   select = "gp", batch 1 is the FPS batch and each later batch is where a Gaussian process of the anharmonic part
   (CCSD(T) minus the Hessian's quadratic model) of all finished points is least sure (diabml.gpdelta), one batch at
   a time. The design is saved in data/well9d_design.npz and reused, so rerunning a batch recomputes nothing.

4. Fit (well9d_fit). The CCSD(T) energies (plus the [gs_grid.delta] aug-cc-pVTZ correction evaluated on the 4-D
   projection: mean HCO angle, C-H and out-of-plane motion ignored; logged) within fit_window_eV of the lowest point
   are fitted by a polynomial in the nine coordinates that is even in (d_H1, d_H2) jointly, so the mirror symmetry
   is exact. Each order in fit_orders is scored by leaving out one sample batch at a time (the Hessian points always
   train); the order with the lowest held-out error is kept and fitted to all points: models/well9d.npz, a PolyWell
   with the same interface as the 4-D well (predict, matrix_t; diabml.wellfit). At its minimum: the nine harmonic
   frequencies, mode labels from the potential energy distribution (CH2 symmetric/antisymmetric stretch, scissor
   and rock, wag and torsion from the sign of the paired coordinates), symmetry (a' in plane, a'' out of plane),
   the equilibrium rotational constants, and a comparison with the frozen 4-D benchmark on its planar symmetric
   slice. Writes results/well9d.json, results/well9d_fit.csv, figures/W9_well9d.png.

5. Basis-set correction in nine coordinates (well9d_tz): CCSD(T) in [well9d] tz_basis (aug-cc-pVTZ) at tz_points
   (200) of the computed points within tz_e_max_eV, spread by farthest-point sampling. well9d_fit then fits the
   difference by a Cs-even polynomial (tz_orders, chosen by held-out error) and adds it everywhere instead of the
   4-D projection of [gs_grid.delta]; that reaches the C-H bonds and the out-of-plane motion.

6. Anharmonic fundamentals (well9d_vpt2): VPT2 on the fitted polynomial (diabml/vpt2.py). The minimum is refined
   by Newton steps; Cartesian Hessians come exactly from automatic differentiation through the polynomial and the
   internal coordinates; cubic and semi-diagonal quartic constants from Hessians displaced along each normal
   coordinate; Coriolis constants from the Eckart modes. Fermi resonances (Martin test) are removed from the
   anharmonic constants and diagonalised with their partners (GVPT2). Modes are numbered v1-v9 (Herzberg, Cs).
   Writes results/well9d_vpt2.json.

7. Dipole surface (well9d_dipole): CCSD(T) dipoles by finite fields (+-F along x, y, z; ORCA %scf EField, no
   symmetry, so the field stays in the input frame) at dipole_points of the computed points within
   dipole_e_max_eV, spread by farthest-point sampling. ORCA's sign convention and whether its field term includes
   the nuclei are not assumed: each candidate is compared with ORCA's own SCF dipole of the same runs and the one
   that matches (a few tenths of a debye apart, against tens for the wrong ones) is used, or the stage stops. The
   9-D dipole surface (diabml/dipole9.py: bond vectors times invariant polynomials, exact under rotation and
   reflection) is fitted with the order chosen by held-out error: models/dipole9d.npz, with harmonic intensities
   at the 9-D minimum. With [ir] surface = "well9d" the ir stage runs on models/well9d.npz and models/dipole9d.npz:
   all nine modes free, no restraint springs, outputs ir_9d.html, results/ir_9d_*; the band markers are the GVPT2
   fundamentals of well9d_vpt2.

The harmonic frequencies are not band origins: compare their anharmonic shifts (vib_levels, dynamics) before
comparing with experiment; the experimental fundamentals in the table are for orientation.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import time

import numpy as np

from . import audit as AU
from . import ch2oo as W
from ..geometry import NINE, NINE_OUT_OF_PLANE, ch2oo_positions9, ch2oo_xyz9

HARTREE_EV = AU.HARTREE_EV
NAMES = list(NINE)
SCALE = {'r': 0.05, 'a': 5.0, 'd': 15.0}            # polynomial scaling and GP length-scale guess per kind
_LAM_FLOOR_CM = 50.0                                 # sampling floor for soft or imaginary modes (cm-1)
# experimental fundamentals for orientation (results band_origins of data/reference/ch2oo_rotational.json)
EXPT = {'CH2 scissor': ('v3', 1434.1), 'C-O stretch': ('v4', 1285.928), 'CH2 rock': ('v5', 1213.3),
        'O-O stretch': ('v6', 909.26), 'CH2 wag': ('v8', 847.44), 'CH2 torsion': ('v9 ~ 2v9/2', 617.1)}


def kind(name):
    """'r' (bond, A), 'a' (angle, deg) or 'd' (dihedral, deg)."""
    return 'd' if name.startswith('d_') else 'a' if name.startswith('a_') else 'r'


def unit(names):
    """Factor from the stored unit (A, deg) to the derivative unit (A, rad) per coordinate."""
    return np.array([np.pi / 180 if kind(n) != 'r' else 1.0 for n in names])


# ------------------------------------------------------------------------------------------ polynomial basis
def cs_powers(n, order, even_idx=()):
    """Exponent tuples of all monomials of total degree <= order in n variables whose summed degree in the
    variables even_idx is even (a polynomial even under their joint sign change). Ordered by degree."""
    out = []

    def rec(prefix, left):
        if len(prefix) == n:
            out.append(tuple(prefix))
            return
        for p in range(left + 1):
            rec(prefix + [p], left - p)
    rec([], order)
    out = [p for p in out if sum(p[i] for i in even_idx) % 2 == 0]
    return sorted(out, key=lambda p: (sum(p), [-v for v in p]))


def design_matrix(Z, pw, chunk=400):
    """Monomial values (N, T) of scaled coordinates Z (N, n) for exponents pw (T, n), in chunks of rows."""
    Z = np.asarray(Z, float)
    pw = np.asarray(pw, float)
    out = np.empty((len(Z), len(pw)))
    for a in range(0, len(Z), chunk):
        out[a:a + chunk] = np.prod(Z[a:a + chunk, None, :] ** pw[None], 2)
    return out


def _scale(names):
    return np.array([SCALE[kind(n)] for n in names])


# ------------------------------------------------------------------------------------------ centre, Hessian
def center9(run, wc):
    """The 9-D centre and where it came from. [well9d] center may give all or some coordinates; the rest come from
    the 4-D well minimum (frozen benchmark, else the lowest gs_grid point) and the grid's fixed C-H."""
    x = dict(zip(NINE, [1.345, 117.7, 1.272, 1.080, 1.080, 120.0, 120.0, 0.0, 0.0]))
    src = 'defaults'
    fixed = {}
    gp = run.p('data', 'gs_grid.npz')
    if gp.exists():
        g = np.load(gp, allow_pickle=False)
        fixed = {k.lower(): float(v) for k, v in json.loads(str(g['fixed'])).items()}
    x4 = None
    try:
        from .shell import frozen_path, load_frozen
        if frozen_path(run, str(wc.get('benchmark', 'benchmark'))).exists():
            from ..vibdvr import harmonic_frequencies
            w, _ = load_frozen(run, f'frozen:{wc.get("benchmark", "benchmark")}')
            _, xm = harmonic_frequencies(w)
            x4 = dict(zip(w.names, map(float, xm)))
            fixed.update({k.lower(): float(v) for k, v in w.meta.get('fixed_coordinates', {}).items()})
            src = f'minimum of the frozen benchmark {wc.get("benchmark", "benchmark")}'
    except Exception as e:                                  # a broken frozen file must not stop the design
        run.log(f'frozen benchmark not usable for the centre ({e}); trying gs_grid')
    if x4 is None and gp.exists():
        E = (g['E_total_hartree'] if 'E_total_hartree' in g.files else g['E_nevpt2_hartree'])[:, 0]
        k = int(np.nanargmin(E))
        x4 = dict(zip([str(n) for n in g['coord_names']], map(float, g['coords'][k])))
        src = 'lowest gs_grid point'
    if x4 is not None:
        for n in ('r_OO', 'a_COO', 'r_CO'):
            if n in x4:
                x[n] = x4[n]
        if 'a_HCO' in x4 or 'a_hco' in fixed:
            x['a_HCO1'] = x['a_HCO2'] = float(x4.get('a_HCO', fixed.get('a_hco', 120.0)))
    if 'r_ch' in fixed:
        x['r_CH1'] = x['r_CH2'] = float(fixed['r_ch'])
    given = dict(wc.get('center') or {})
    bad = [n for n in given if n not in NINE]
    if bad:
        raise SystemExit(f'[well9d.center]: unknown coordinates {bad}; use {list(NINE)}')
    x.update({n: float(v) for n, v in given.items()})
    if given:
        src = (src + ' with ' if len(given) < 9 else '') + f'[well9d.center] {sorted(given)}'
    return np.array([x[n] for n in NINE]), src


def fd_steps(wc, names=NAMES):
    s = {'r': 0.01, 'a': 1.0, 'd': 2.0}
    s.update({k: float(v) for k, v in (wc.get('fd_step') or {}).items()})
    return np.array([s[kind(n)] for n in names])


def fd_design(x0, h):
    """Points for a central-difference gradient and Hessian: the centre, +-h_i along each coordinate, and the four
    corners (+-h_i, +-h_j) of each pair i < j. Returns X (M, n) and tags [(i, si, j, sj)] (-1 / 0 for unused)."""
    n = len(x0)
    X, tags = [x0.copy()], [(-1, 0, -1, 0)]
    for i in range(n):
        for si in (1, -1):
            x = x0.copy()
            x[i] += si * h[i]
            X.append(x)
            tags.append((i, si, -1, 0))
    for i in range(n):
        for j in range(i + 1, n):
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                x = x0.copy()
                x[i] += si * h[i]
                x[j] += sj * h[j]
                X.append(x)
                tags.append((i, si, j, sj))
    return np.array(X), tags


def fd_derivatives(E, tags, h, names=NAMES):
    """Gradient (n,) and Hessian (n, n) from energies at fd_design points; derivative units eV per A or rad."""
    n = len(h)
    hu = h * unit(names)
    e = {t: float(v) for t, v in zip(tags, E)}
    e0 = e[(-1, 0, -1, 0)]
    g, K = np.zeros(n), np.zeros((n, n))
    for i in range(n):
        ep, em = e[(i, 1, -1, 0)], e[(i, -1, -1, 0)]
        g[i] = (ep - em) / (2 * hu[i])
        K[i, i] = (ep - 2 * e0 + em) / hu[i] ** 2
    for i in range(n):
        for j in range(i + 1, n):
            K[i, j] = K[j, i] = (e[(i, 1, j, 1)] - e[(i, 1, j, -1)] - e[(i, -1, j, 1)] + e[(i, -1, j, -1)]) / (
                4 * hu[i] * hu[j])
    return g, K


def normal_modes(K, G):
    """Eigenvalues lambda (ascending; eV / (amu A^2)) and columns L of G K, normalised so that L^T G^-1 L = 1:
    dx = L Q (derivative units) gives V = sum_k lambda_k Q_k^2 / 2 and T = sum_k dQ_k^2 / 2. Wavenumbers
    (cm-1, negative for imaginary) from sign(lambda) sqrt(|lambda| hbar^2)."""
    from ..vibdvr import EV_CM, HBAR2
    lam, L = np.linalg.eig(G @ K)
    o = np.argsort(lam.real)
    lam, L = lam.real[o], L.real[:, o]
    Gi = np.linalg.inv(G)
    for k in range(L.shape[1]):
        L[:, k] /= np.sqrt(L[:, k] @ Gi @ L[:, k])
    nu = np.sign(lam) * np.sqrt(np.abs(lam) * HBAR2) * EV_CM
    return lam, L, nu


def mode_info(K, L, names=NAMES):
    """Per normal mode: label, symmetry (a' / a''), potential-energy distribution {coordinate: fraction}.
    Pairs of equivalent-looking coordinates are named by the relative sign of their displacements: C-H
    stretches (same sign: symmetric), HCO angles (same sign: scissor, opposite: rock), out-of-plane dihedrals
    (same sign: torsion about C-O, opposite: wag)."""
    single = {'r_OO': 'O-O stretch', 'a_COO': 'COO bend', 'r_CO': 'C-O stretch', 'r_CH1': 'C-H stretch (syn)',
              'r_CH2': 'C-H stretch (anti)', 'a_HCO1': 'HCO bend (syn)', 'a_HCO2': 'HCO bend (anti)',
              'd_H1': 'H out of plane (syn)', 'd_H2': 'H out of plane (anti)'}
    pairs = {('r_CH1', 'r_CH2'): ('CH2 sym. stretch', 'CH2 asym. stretch'),
             ('a_HCO1', 'a_HCO2'): ('CH2 scissor', 'CH2 rock'),
             ('d_H1', 'd_H2'): ('CH2 torsion', 'CH2 wag')}
    out = []
    oop = [names.index(n) for n in NINE_OUT_OF_PLANE if n in names]
    for k in range(L.shape[1]):
        ped = L[:, k] * (K @ L[:, k])
        ped = np.abs(ped) / max(np.abs(ped).sum(), 1e-30)
        frac = {n: float(p) for n, p in zip(names, ped)}
        label = single[names[int(np.argmax(ped))]]
        for (a, b), (same, opp) in pairs.items():
            if a in names and b in names:
                ia, ib = names.index(a), names.index(b)
                if ped[ia] + ped[ib] > 0.5 and min(ped[ia], ped[ib]) > 0.15:
                    label = same if L[ia, k] * L[ib, k] > 0 else opp
        sym = "a''" if ped[oop].sum() > 0.5 else "a'"
        out.append({'label': label, 'symmetry': sym,
                    'ped': {n: round(v, 3) for n, v in sorted(frac.items(), key=lambda t: -t[1]) if v >= 0.05}})
    return out


# ------------------------------------------------------------------------------------------ sampling
def _physical(X, names=NAMES, d_max=45.0):
    """Rows of X (stored units) that are sensible CH2OO geometries for sampling."""
    ok = np.ones(len(X), bool)
    for j, n in enumerate(names):
        k = kind(n)
        if k == 'r':
            lo = 0.85 if n.startswith('r_CH') else 1.0
            ok &= X[:, j] > lo
        elif k == 'a':
            ok &= (X[:, j] > 70.0) & (X[:, j] < 170.0)
        else:
            ok &= np.abs(X[:, j]) <= d_max
    return ok


def sample_pool(x0, lam, L, n, e_max, power=1.0, d_max=45.0, seed=0, names=NAMES):
    """n candidates around x0 (stored units): harmonic energy E = e_max u^power, random direction in the normal
    coordinates of (lam, L). Returns X (n, 9), the harmonic energy (eV) and the energy-scaled normal coordinates
    sqrt(lam) Q (the farthest-point metric)."""
    from ..vibdvr import EV_CM, HBAR2
    floor = (_LAM_FLOOR_CM / EV_CM) ** 2 / HBAR2
    lam_e = np.maximum(lam, floor)
    rng = np.random.default_rng(seed)
    toU = 1.0 / unit(names)
    keep_X, keep_E, keep_Z = [], [], []
    got, tries = 0, 0
    while got < n and tries < 50:
        m = 2 * (n - got) + 100
        u = rng.standard_normal((m, len(lam)))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        E = e_max * rng.random(m) ** power
        Q = np.sqrt(2 * E)[:, None] * u / np.sqrt(lam_e)[None]
        X = x0[None] + (Q @ L.T) * toU[None]
        ok = _physical(X, names, d_max)
        keep_X.append(X[ok])
        keep_E.append(E[ok])
        keep_Z.append((Q * np.sqrt(lam_e)[None])[ok])
        got += int(ok.sum())
        tries += 1
    X, E, Z = (np.concatenate(a)[:n] for a in (keep_X, keep_E, keep_Z))
    return X, E, Z


def harmonic_model(x0, g, K, names=NAMES):
    """E(x) - E(x0) of the local quadratic model g.dx + dx K dx / 2 (eV), x in stored units."""
    u = unit(names)

    def f(X):
        dx = (np.atleast_2d(X) - x0) * u
        return dx @ g + 0.5 * np.einsum('ni,ij,nj->n', dx, K, dx)
    return f


# ------------------------------------------------------------------------------------------ ORCA
def _key(x):
    return 'n_' + '_'.join(f'{v:.4f}' for v in x).replace('-', 'm').replace('.', 'p')


def _compute(run, exe, wd, X, basis, cc, jobs, label):
    """CCSD(T) at every row of X (inputs written once, finished outputs reused). Returns E (hartree) and T1."""
    from .method_cut import cc_input, parse_cc
    wd.mkdir(parents=True, exist_ok=True)
    for x in X:
        nm = _key(x)
        if not (wd / f'{nm}.inp').exists():
            (wd / f'{nm}.inp').write_text(cc_input(ch2oo_xyz9(x), basis, cc['maxcore']))

    def one(x):
        nm = _key(x)
        if parse_cc(wd / f'{nm}.out') is not None:
            return 'reused'
        return AU._run_one(exe, wd, nm, float(cc['timeout_s']))[1]
    t0, done, status = time.time(), 0, {}
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        for s in ex.map(one, X):
            done += 1
            status[s] = status.get(s, 0) + 1
            if done % max(1, len(X) // 10) == 0:
                run.log(f'  {label}: {done}/{len(X)} points ({(time.time() - t0) / 60:.1f} min)')
    run.detail(f'{label}: ' + ', '.join(f'{k} {v}' for k, v in sorted(status.items())))
    E, T1 = np.full(len(X), np.nan), np.full(len(X), np.nan)
    for k, x in enumerate(X):
        r = parse_cc(wd / f'{_key(x)}.out')
        if r is not None:
            E[k], T1[k] = r['E'], r.get('T1', np.nan)
    bad = int(np.isnan(E).sum())
    if bad:
        run.log(f'  {label}: {bad} of {len(X)} points failed (outputs in {wd}); they are left out')
    return E, T1


def _basis(run, wc):
    if wc.get('basis'):
        return str(wc['basis'])
    gp = run.p('data', 'gs_grid.npz')
    if gp.exists():
        g = np.load(gp, allow_pickle=False)
        if 'basis' in g.files:
            return str(g['basis'])
    return 'aug-cc-pVDZ'


def _log_modes(run, nu, info, head):
    run.log(head + ', '.join(f'{v:.0f}' for v in nu[::-1]) + ' cm-1')
    for k in range(len(nu) - 1, -1, -1):
        m = info[k]
        ex = EXPT.get(m['label'])
        run.detail(f'{nu[k]:8.1f} cm-1  {m["symmetry"]:4s} {m["label"]:22s} '
                   + (f'(expt. {ex[0]} {ex[1]:.0f}) ' if ex else '')
                   + ' '.join(f'{n} {p:.2f}' for n, p in m['ped'].items()))


def hessian(run, wc, exe, basis, jobs):
    """Batch 0: the finite-difference CCSD(T) Hessian at the centre (reused from data/well9d_hessian.npz)."""
    from ..vibdvr import g_matrix
    hp = run.p('data', 'well9d_hessian.npz')
    x0, src = center9(run, wc)
    h = fd_steps(wc)
    if hp.exists():
        z = np.load(hp, allow_pickle=False)
        if not (np.allclose(z['center'], x0, atol=1e-6) and np.allclose(z['step'], h)):
            run.log(f'note: data/well9d_hessian.npz was made at another centre or step than the current [well9d] '
                    f'settings; it is kept (move it away to redo the Hessian)')
        run.detail(f'Hessian reused from {hp.name}: centre ' + ', '.join(f'{n} {v:.4f}' for n, v in
                                                                          zip(NAMES, z['center'])))
        return {k: z[k] for k in z.files}
    X, tags = fd_design(x0, h)
    run.log(f'Hessian: centre from {src}: ' + ', '.join(f'{n} {v:.4f}' for n, v in zip(NAMES, x0)))
    run.log(f'  {len(X)} CCSD(T)/{basis} points, steps ' + ', '.join(f'{k} {v:g}' for k, v in
                                                                    zip(('bond', 'angle', 'dihedral'),
                                                                        [h[0], h[1], h[-1]])) + f'; {jobs} parallel')
    Eh, T1 = _compute(run, exe, run.p('data', 'well9d'), X, basis, run.cfg['compute'], jobs, 'Hessian')
    if np.isnan(Eh).any():
        raise SystemExit('the finite-difference Hessian needs every point: rerun to retry the failed ones')
    E = Eh * HARTREE_EV
    g, K = fd_derivatives(E, tags, h)
    G = g_matrix(x0[None], NAMES, {})[0]
    lam, L, nu = normal_modes(K, G)
    info = mode_info(K, L)
    # mirror check: +d and -d displacements must have the same energy
    asym = []
    for i in (NAMES.index(n) for n in NINE_OUT_OF_PLANE):
        ep = E[tags.index((i, 1, -1, 0))]
        em = E[tags.index((i, -1, -1, 0))]
        asym.append(abs(ep - em) * 1000)
    step = -np.linalg.solve(K, g)                           # Newton step, derivative units
    dE = 0.5 * float(g @ step)
    out = dict(center=x0, step=h, coords=X, E_total_hartree=Eh, T1=T1, tags=np.array(tags), g=g, K=K, G=G, lam=lam,
               L=L, nu=nu, basis=basis, source=src)
    np.savez(hp, **out)
    res = {'center': dict(zip(NAMES, x0.tolist())), 'source': src, 'basis': basis, 'n_points': len(X),
           'harmonic_cm-1': [{'cm-1': float(v), **m} for v, m in zip(nu, info)],
           'gradient': dict(zip(NAMES, g.tolist())), 'newton_step': dict(zip(NAMES, (step / unit(NAMES)).tolist())),
           'newton_energy_meV': dE * 1000, 'mirror_asymmetry_meV': max(asym), 'T1': [float(np.nanmin(T1)),
                                                                                       float(np.nanmax(T1))]}
    run.p('results', 'well9d_hessian.json').write_text(json.dumps(res, indent=1))
    _log_modes(run, nu, info, f'CCSD(T)/{basis} harmonic at the centre (high to low): ')
    run.log(f'  Newton step to the 9-D minimum lowers E by {-dE * 1000:.2f} meV; largest moves: '
            + ', '.join(f'{n} {v:+.4f}' for n, v in sorted(zip(NAMES, step / unit(NAMES)),
                                                           key=lambda t: -abs(t[1]) / SCALE[kind(t[0])])[:3])
            + f'; mirror check |E(+d) - E(-d)| <= {max(asym):.3f} meV; T1 {np.nanmin(T1):.3f}-{np.nanmax(T1):.3f}')
    if (nu < 0).any():
        run.log(f'  warning: {int((nu < 0).sum())} imaginary mode(s) at the centre: the sampling floors them at '
                f'{_LAM_FLOOR_CM:g} cm-1; consider [well9d.center] at the Newton minimum')
    run.log('  wrote data/well9d_hessian.npz, results/well9d_hessian.json')
    return {k: np.asarray(v) for k, v in out.items()}


def _fps(Z, n, start=0):
    from .shell import _fps as fps
    return fps(Z, n, start)


def design(run, hz):
    """The sample batches (saved in data/well9d_design.npz the first time and reused)."""
    wc = run.cfg['well9d']
    dp = run.p('data', 'well9d_design.npz')
    sig = json.dumps({k: wc[k] for k in ('e_max_eV', 'e_power', 'd_max', 'n_pool', 'n_points', 'batches', 'seed',
                                         'gp_pool')} | {'center': np.round(hz['center'], 6).tolist()}, sort_keys=True)
    if dp.exists():
        z = np.load(dp, allow_pickle=False)
        if str(z['signature']) != sig:
            run.log('note: data/well9d_design.npz was made with other [well9d] sampling settings; it is kept so the '
                    'batches stay the same (move it away to redesign before computing new batches)')
        return {k: z[k] for k in z.files}
    n_pts, nb = int(wc['n_points']), max(1, int(wc['batches']))
    n_pool = max(int(wc['n_pool']), n_pts, int(wc['gp_pool']))
    X, Eh, Z = sample_pool(hz['center'], hz['lam'], hz['L'], n_pool, float(wc['e_max_eV']), float(wc['e_power']),
                           float(wc['d_max']), int(wc['seed']))
    m = max(n_pts, min(int(wc['gp_pool']), len(X)))
    order = _fps(Z, m, int(np.argmin(np.sum(Z ** 2, 1))))
    batch = np.zeros(len(order), int)
    for b in range(nb):
        batch[b:n_pts:nb] = b + 1                        # round-robin over the first n_points of the FPS order
    out = dict(coords=X[order], harmonic_eV=Eh[order], batch=batch, signature=sig)
    np.savez(dp, **out)
    run.log(f'design: {len(X)} candidates (harmonic energy 0-{float(wc["e_max_eV"]):g} eV, |d| <= '
            f'{float(wc["d_max"]):g} deg) -> {n_pts} points by farthest-point sampling in {nb} batches; '
            f'{m - n_pts} more kept for GP selection; wrote data/well9d_design.npz')
    return out


def _gp_batch(run, d, b, hz, done_X, done_E):
    """GP batch b: the candidates not yet computed where a GP of (CCSD(T) - quadratic model) is least sure."""
    from ..gpdelta import GPDelta, select_uncertain
    cp = run.p('data', f'well9d_gp_batch{b}.npz')
    if cp.exists():
        return np.load(cp)['coords']
    f = harmonic_model(hz['center'], hz['g'], hz['K'])
    e0 = float(np.load(run.p('data', 'well9d_hessian.npz'))['E_total_hartree'][0]) * HARTREE_EV
    ok = np.isfinite(done_E)
    y = done_E[ok] - e0 - f(done_X[ok])
    init = {n: SCALE[kind(n)] * 4 for n in NAMES}
    gp = GPDelta.fit(done_X[ok], y, NAMES, noise_eV=1e-4, init_ls=init)
    have = {_key(x) for x in done_X}
    cand = d['coords'][np.array([_key(x) not in have for x in d['coords']])]
    nb = int(np.sum(d['batch'] == b))
    pick = select_uncertain(gp, cand, nb)
    Xb = cand[pick]
    _, sd = gp.predict(Xb)
    np.savez(cp, coords=Xb)
    run.log(f'  batch {b} chosen by GP uncertainty (GP on {int(ok.sum())} points, anharmonic part '
            f'{np.ptp(y) * 1000:.0f} meV wide): {len(Xb)} of {len(cand)} candidates, std {sd.max() * 1000:.1f}-'
            f'{sd.min() * 1000:.1f} meV; length scales ' + ', '.join(f'{k} {v:.3g}' for k, v in
                                                                    gp.meta['length_scales'].items()))
    return Xb


def _merge(run, rows_new):
    """Add rows {key: (x, E_h, T1, batch, harmonic_eV)} to data/well9d.npz (earlier rows kept)."""
    p = run.p('data', 'well9d.npz')
    rows = {}
    if p.exists():
        z = np.load(p, allow_pickle=False)
        for x, e, t, b, h in zip(z['coords'], z['E_total_hartree'][:, 0], z['T1'], z['batch'], z['harmonic_eV']):
            rows[_key(x)] = (x, e, t, int(b), h)
    rows.update(rows_new)
    R = [r for r in rows.values() if np.isfinite(r[1])]
    np.savez(p, coords=np.array([r[0] for r in R]).reshape(-1, len(NAMES)), coord_names=np.array(NAMES),
             E_total_hartree=np.array([r[1] for r in R])[:, None], T1=np.array([r[2] for r in R]),
             batch=np.array([r[3] for r in R], int), harmonic_eV=np.array([r[4] for r in R]), method='ccsd_t')
    return R


def stage_well9d(run, jobs=None, orca=None, batch=None):
    """Hessian (batch 0) and sample batches of the 9-D well; see the module docstring."""
    wc, cc, gc = run.cfg['well9d'], run.cfg['compute'], run.cfg['gs_grid']
    jobs = int(jobs or wc.get('jobs') or gc.get('jobs') or cc['jobs'])
    basis = _basis(run, wc)
    exe = AU._orca_exe(run, orca or cc.get('orca') or run.cfg['audit'].get('orca') or None)
    hz = hessian(run, wc, exe, basis, jobs)
    f = harmonic_model(hz['center'], hz['g'], hz['K'])
    e0 = float(hz['E_total_hartree'][0]) * HARTREE_EV
    rows = {_key(x): (x, e, t, 0, float(f(x[None])[0])) for x, e, t in zip(hz['coords'], hz['E_total_hartree'],
                                                                          hz['T1'])}
    _merge(run, rows)
    if batch == 0:
        run.log('batch 0 (Hessian) done; next: well9d --batch 1')
        return
    d = design(run, hz)
    nb = int(d['batch'].max())
    mode = str(wc.get('select', 'fps'))
    want = list(range(1, nb + 1)) if batch is None else [int(batch)]
    if mode == 'gp' and batch is None:
        raise SystemExit('[well9d] select = "gp": run the batches one at a time (--batch 1, then 2, ...): each GP batch '
                         'is chosen from the results of the ones before')
    if any(b < 1 or b > nb for b in want):
        raise SystemExit(f'--batch must be 0 (Hessian) or 1-{nb}')
    for b in want:
        Xb = d['coords'][d['batch'] == b]
        if mode == 'gp' and b >= 2:
            z = np.load(run.p('data', 'well9d.npz'))
            if not (z['batch'] == b - 1).any():
                raise SystemExit(f'select = "gp": compute --batch {b - 1} first')
            Xb = _gp_batch(run, d, b, hz, z['coords'], z['E_total_hartree'][:, 0] * HARTREE_EV)
        Eb_h = f(Xb)
        run.log(f'batch {b}: {len(Xb)} points, harmonic energy {Eb_h.min():.2f}-{Eb_h.max():.2f} eV '
                f'(median {np.median(Eb_h):.2f}); RHF-CCSD(T)/{basis}, {jobs} parallel')
        for j, n in enumerate(NAMES):
            run.detail(f'{n:7s} {Xb[:, j].min():9.3f} - {Xb[:, j].max():9.3f}   (centre {hz["center"][j]:.3f})')
        Eh, T1 = _compute(run, exe, run.p('data', 'well9d'), Xb, basis, cc, jobs, f'batch {b}')
        R = _merge(run, {_key(x): (x, e, t, b, h) for x, e, t, h in zip(Xb, Eh, T1, Eb_h)})
        ok = np.isfinite(Eh)
        E = Eh[ok] * HARTREE_EV - e0
        ratio = E / np.maximum(Eb_h[ok], 1e-6)
        run.log(f'  batch {b}: CCSD(T) {E.min():.2f}-{E.max():.2f} eV above the centre; CCSD(T) / harmonic median '
                f'{np.median(ratio):.2f} (10-90%: {np.percentile(ratio, 10):.2f}-{np.percentile(ratio, 90):.2f}); '
                f'T1 {np.nanmin(T1):.3f}-{np.nanmax(T1):.3f}')
        for lo, hi in ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 9)):
            m = (Eb_h[ok] >= lo) & (Eb_h[ok] < hi)
            if m.any():
                run.detail(f'harmonic {lo:.1f}-{hi:.1f} eV: {int(m.sum()):4d} points, CCSD(T) - harmonic '
                           f'{np.median(E[m] - Eb_h[ok][m]) * 1000:+7.1f} meV median')
    done = sorted({int(r[3]) for r in R})
    run.log(f'{len(R)} points in data/well9d.npz (batches {done}); nothing else changed: next  well9d_fit')


# ------------------------------------------------------------------------------------------ fit
def fit_weights(E, w_eV):
    """Least-squares weights E0 / (E0 + E) (Braams & Bowman 2009) with E0 = w_eV; all 1 when w_eV is 0."""
    E = np.maximum(np.asarray(E, float), 0.0)
    return np.ones_like(E) if not w_eV else float(w_eV) / (float(w_eV) + E)


def _fit(X, E, x0, pw, scale, ridge=1e-10, w=None):
    A = design_matrix((X - x0) / scale, pw)
    if w is not None:
        r = np.sqrt(np.asarray(w, float))
        A, E = A * r[:, None], E * r
    s = np.linalg.norm(A, axis=0)
    s[s == 0] = 1.0
    An = A / s
    if ridge:
        coef = np.linalg.solve(An.T @ An + ridge * len(X) * np.eye(len(pw)), An.T @ E)
    else:
        coef, *_ = np.linalg.lstsq(An, E, rcond=None)
    return coef / s


def _predict(X, x0, pw, scale, coef):
    return design_matrix((np.atleast_2d(X) - x0) / scale, pw) @ coef


def fit_orders(X, E, batch, x0, orders, log=None, detail=None, w=None, low_eV=None):
    """Held-out error (meV) of each polynomial order: each sample batch left out in turn (batch 0, the Hessian
    points, always trains); with fewer than two sample batches, five interleaved folds instead. With weights w the
    fits are weighted and the order is chosen by the weighted held-out RMS; low_eV adds the held-out RMS of the
    points within low_eV of the lowest (the region the spectra depend on most)."""
    scale = _scale(NAMES)
    even = [NAMES.index(n) for n in NINE_OUT_OF_PLANE]
    bs = sorted(set(int(b) for b in batch if b > 0))
    folds = [batch == b for b in bs] if len(bs) >= 2 else [np.arange(len(X)) % 5 == k for k in range(5)]
    out = {}
    for order in orders:
        pw = np.array(cs_powers(len(NAMES), int(order), even))
        if len(pw) > 0.7 * len(X):
            (log or print)(f'order {order}: {len(pw)} terms for {len(X)} points, too many: skipped')
            continue
        errs, per, ws, es = [], {}, [], []
        for k, m in enumerate(folds):
            tr = ~m
            if tr.sum() < len(pw) + 5 or not m.any():
                continue
            c = _fit(X[tr], E[tr], x0, pw, scale, w=None if w is None else w[tr])
            e = _predict(X[m], x0, pw, scale, c) - E[m]
            errs.append(e)
            ws.append(np.ones(int(m.sum())) if w is None else w[m])
            es.append(E[m])
            per[str(bs[k]) if len(bs) >= 2 else f'fold{k}'] = float(np.sqrt(np.mean(e ** 2)) * 1000)
        if not errs:
            continue
        e, wv, ev = np.concatenate(errs), np.concatenate(ws), np.concatenate(es)
        out[int(order)] = {'terms': int(len(pw)), 'heldout_rms_meV': float(np.sqrt(np.mean(e ** 2)) * 1000),
                           'heldout_max_meV': float(np.abs(e).max() * 1000), 'per_fold_meV': per}
        if w is not None:
            out[int(order)]['heldout_weighted_rms_meV'] = float(np.sqrt(np.sum(wv * e ** 2) / np.sum(wv)) * 1000)
        if low_eV is not None and np.any(ev <= low_eV):
            el = e[ev <= low_eV]
            out[int(order)]['heldout_low_rms_meV'] = float(np.sqrt(np.mean(el ** 2)) * 1000)
            out[int(order)]['low_eV'] = float(low_eV)
        if detail:
            detail(f'order {order}: {len(pw)} terms, held-out RMS {out[int(order)]["heldout_rms_meV"]:.2f} meV '
                   f'(max {out[int(order)]["heldout_max_meV"]:.1f}); per batch '
                   + ', '.join(f'{k} {v:.2f}' for k, v in per.items()))
    return out


def stage_well9d_fit(run, window_eV=None):
    """Fit the 9-D CCSD(T) well, analyse it at its minimum, compare with the frozen 4-D surface. window_eV overrides
    [well9d] fit_window_eV (well9d_extend uses it); [well9d] fit_weight_eV > 0 weights the fit by E0 / (E0 + E)."""
    from ..vibdvr import ISOTOPIC_MASS, inertia_abc, minimise
    from ..wellfit import PolyWell
    wc = run.cfg['well9d']
    p = run.p('data', 'well9d.npz')
    if not p.exists():
        raise SystemExit(f'{p} not found: run well9d first')
    z = np.load(p, allow_pickle=False)
    X, Eh, batch = z['coords'], z['E_total_hartree'][:, 0], z['batch']
    hz = np.load(run.p('data', 'well9d_hessian.npz'), allow_pickle=False)
    x0 = hz['center']
    E = Eh * HARTREE_EV
    how = 'CCSD(T)/' + str(hz['basis'])
    dinfo = None
    d9, d9info = delta9(run, wc, X) if wc.get('apply_delta', True) else (None, None)
    if d9 is not None:
        E = E + d9
        how += f' + 9-D delta ({d9info["basis"]} - {d9info["base_basis"]}, order {d9info["order"]})'
    elif wc.get('apply_delta', True) and run.p('data', 'gs_grid.npz').exists():
        from .gs_grid import delta_function
        fdel, dinfo, gnames = delta_function(run)
        if fdel is not None:
            X4 = project4(X, gnames)
            dE = fdel(X4)
            E = E + dE
            how += f' + delta ({dinfo["basis"]} - {dinfo["basis_base"]}) on the 4-D projection'
            run.log(f'delta on the 4-D projection ({", ".join(gnames)}): {np.ptp(dE) * 1000:.0f} meV range over the '
                    f'points (C-H and out-of-plane motion not corrected)')
    E = E - np.nanmin(E)
    win = float(window_eV if window_eV is not None else wc['fit_window_eV'])
    m = np.isfinite(E) & (E <= win)
    run.log(f'{len(X)} points ({how}); {int(m.sum())} within {win:g} eV of the lowest; batches '
            f'{sorted(set(batch.tolist()))}; T1 {np.nanmin(z["T1"]):.3f}-{np.nanmax(z["T1"]):.3f}')
    Xf, Ef, bf = X[m], E[m], batch[m]
    w_eV = float(wc.get('fit_weight_eV', 0) or 0)
    wf = fit_weights(Ef, w_eV) if w_eV else None
    low = float(wc.get('fit_low_eV', 0.6))
    if wf is not None:
        run.log(f'weighted fit: w = {w_eV:g} / ({w_eV:g} + E) (1 at the minimum, {fit_weights([win], w_eV)[0]:.2f} at '
                f'{win:g} eV)')
    cv = fit_orders(Xf, Ef, bf, x0, [int(o) for o in wc['fit_orders']], run.log, run.detail, w=wf, low_eV=low)
    if not cv:
        raise SystemExit('no polynomial order could be fitted: compute more batches or lower [well9d] fit_orders')
    key = 'heldout_weighted_rms_meV' if wf is not None else 'heldout_rms_meV'
    best = min(cv, key=lambda o: cv[o][key])
    run.log('held-out error by order: ' + ', '.join(
        f'{o}: {v["heldout_rms_meV"]:.2f} meV' + (f' ({v["heldout_low_rms_meV"]:.2f} within {low:g} eV)'
                                                   if 'heldout_low_rms_meV' in v else '') + f', {v["terms"]} terms'
        for o, v in cv.items()) + f' -> order {best}')
    even = [NAMES.index(n) for n in NINE_OUT_OF_PLANE]
    pw = np.array(cs_powers(len(NAMES), best, even))
    scale = _scale(NAMES)
    coef = _fit(Xf, Ef, x0, pw, scale, w=wf)
    res = _predict(Xf, x0, pw, scale, coef) - Ef
    box = {n: [float(Xf[:, j].min()), float(Xf[:, j].max())] for j, n in enumerate(NAMES)}
    model = PolyWell(NAMES, x0, scale, pw, coef, box,
                     meta={'coord_names': NAMES, 'fixed_coordinates': {}, 'order': best, 'window_eV': win,
                           'n_points': int(m.sum()), 'n_terms': int(len(pw)), 'source': how,
                           'residual_rms_meV': float(np.sqrt(np.mean(res ** 2)) * 1000),
                           'experiment': f'9-D CCSD(T) well polynomial (Cs-even), order {best}'})
    model.save(run.p('models', 'well9d.npz'))
    Vf = lambda Y: _predict(Y, x0, pw, scale, coef)
    xm = minimise(Vf, x0, [tuple(box[n]) for n in NAMES])
    xm, _, gmax = refine_minimum(model, xm)
    Em = float(Vf(xm[None])[0])
    from ..vibdvr import g_matrix
    K, G0 = exact_K(model, xm), g_matrix(xm[None], NAMES, {})[0]
    lam, L, nu = normal_modes(K, G0)
    info = mode_info(K, L)
    abc = inertia_abc(ch2oo_positions9(xm[None]), ISOTOPIC_MASS)[0]
    ref = json.loads((W.C.PACKAGE_ROOT / 'data' / 'reference' / 'ch2oo_rotational.json').read_text())
    g0 = ref['states']['ground']
    run.log(f'order {best}: {len(pw)} terms, residual {np.sqrt(np.mean(res ** 2)) * 1000:.2f} meV (max '
            f'{np.abs(res).max() * 1000:.1f}); held-out {cv[best]["heldout_rms_meV"]:.2f} meV')
    run.log('minimum: ' + ', '.join(f'{n} {v:.4f}' for n, v in zip(NAMES, xm)) + f' ({Em * 1000:+.2f} meV vs lowest point)')
    for n, a, b in zip(NAMES, x0, xm):
        run.detail(f'{n:7s} centre {a:9.4f}  minimum {b:9.4f}  shift {b - a:+.4f}')
    _log_modes(run, nu, info, 'harmonic at the minimum (high to low): ')
    run.log(f'equilibrium A, B, C {abc[0]:.4f}, {abc[1]:.4f}, {abc[2]:.4f} cm-1 (expt. v = 0: {g0["A"]:.4f}, '
            f'{g0["B"]:.4f}, {g0["C"]:.4f}; equilibrium and v = 0 differ by vibrational averaging, ~1%)')
    bench = _benchmark_check(run, wc, Vf, xm)
    out = {'source': how, 'n_points': int(len(X)), 'n_fit': int(m.sum()), 'window_eV': win, 'weight_eV': w_eV,
           'low_eV': low, 'orders': cv,
           'order': best, 'terms': int(len(pw)), 'residual_rms_meV': float(np.sqrt(np.mean(res ** 2)) * 1000),
           'residual_max_meV': float(np.abs(res).max() * 1000), 'minimum': dict(zip(NAMES, xm.tolist())),
           'minimum_energy_meV': Em * 1000, 'center': dict(zip(NAMES, x0.tolist())),
           'harmonic_cm-1': [{'cm-1': float(v), **i, 'expt_fundamental': EXPT.get(i['label'], (None, None))[1]}
                             for v, i in zip(nu, info)],
           'rotational_equilibrium_cm-1': dict(zip('ABC', abc.tolist())), 'rotational_expt_v0_cm-1': g0,
           'delta': {k: v for k, v in (dinfo or {}).items() if k != '_function'} or None, 'delta9': d9info,
           'benchmark': bench,
           'box': box}
    run.p('results', 'well9d.json').write_text(json.dumps(out, indent=1, default=float))
    with open(run.p('results', 'well9d_fit.csv'), 'w') as fh:
        fh.write(','.join(NAMES + ['batch', 'E_eV', 'fit_eV', 'T1']) + '\n')
        for i in np.where(m)[0]:
            fh.write(','.join(f'{v:.5f}' for v in X[i]) + f',{batch[i]},{E[i]:.6f},{Vf(X[i][None])[0]:.6f},'
                     f'{z["T1"][i]:.4f}\n')
    _figure(run, Ef, Ef + res, bf, nu, info)
    run.log('wrote models/well9d.npz, results/well9d.json, results/well9d_fit.csv, figures/W9_well9d.png')


def project4(X9, names4):
    """The 4-D grid coordinates (names4, e.g. r_OO, a_COO, r_CO, a_HCO) of 9-D points: a_HCO = mean HCO angle,
    r_CH = mean C-H; columns not derivable are an error."""
    cols = []
    for n in names4:
        if n in NAMES:
            cols.append(X9[:, NAMES.index(n)])
        elif n == 'a_HCO':
            cols.append(0.5 * (X9[:, NAMES.index('a_HCO1')] + X9[:, NAMES.index('a_HCO2')]))
        elif n == 'r_CH':
            cols.append(0.5 * (X9[:, NAMES.index('r_CH1')] + X9[:, NAMES.index('r_CH2')]))
        else:
            raise SystemExit(f'cannot project the 9-D points onto the grid coordinate {n}')
    return np.stack(cols, 1)


def lift4(X4, names4, fixed):
    """Planar, symmetric 9-D points from 4-D grid points: both C-H at the fixed value, both HCO angles equal."""
    low = {k.lower(): float(v) for k, v in fixed.items()}
    out = np.zeros((len(X4), len(NAMES)))
    v = dict(zip(names4, X4.T))
    r_ch = v.get('r_CH', np.full(len(X4), low.get('r_ch', 1.08)))
    a_hco = v.get('a_HCO', np.full(len(X4), low.get('a_hco', 120.0)))
    for j, n in enumerate(NAMES):
        out[:, j] = {'r_OO': v.get('r_OO'), 'a_COO': v.get('a_COO'), 'r_CO': v.get('r_CO'), 'r_CH1': r_ch,
                     'r_CH2': r_ch, 'a_HCO1': a_hco, 'a_HCO2': a_hco, 'd_H1': 0.0, 'd_H2': 0.0}[n]
    return out


def _benchmark_check(run, wc, Vf, xm):
    """The 9-D fit on the planar symmetric slice of the frozen 4-D benchmark: RMS of the energy difference (both
    measured from their own value at the benchmark minimum) at the gs_grid points within 0.3 eV."""
    from .shell import frozen_path, load_frozen
    name = str(wc.get('benchmark', 'benchmark'))
    if not frozen_path(run, name).exists() or not run.p('data', 'gs_grid.npz').exists():
        run.log(f'no frozen benchmark "{name}" (or no gs_grid): consistency check skipped')
        return None
    from ..vibdvr import harmonic_frequencies
    w, _ = load_frozen(run, f'frozen:{name}')
    g = np.load(run.p('data', 'gs_grid.npz'), allow_pickle=False)
    fixed = json.loads(str(g['fixed']))
    names4 = list(w.names)
    Xg = g['coords'][:, [list(map(str, g['coord_names'])).index(n) for n in names4]]
    _, x4m = harmonic_frequencies(w)
    Eb = np.asarray(w.predict(Xg))[:, 0] - float(np.asarray(w.predict(x4m[None]))[0, 0])
    m = Eb <= 0.3
    X9 = lift4(Xg[m], names4, fixed)
    E9 = Vf(X9) - Vf(lift4(x4m[None], names4, fixed))[0]
    d = (E9 - Eb[m]) * 1000
    out = {'name': name, 'n': int(m.sum()), 'rms_meV': float(np.sqrt(np.mean(d ** 2))),
           'max_meV': float(np.abs(d).max()), 'mean_meV': float(d.mean())}
    run.log(f'planar symmetric slice vs frozen {name} at {out["n"]} gs_grid points within 0.3 eV: RMS '
            f'{out["rms_meV"]:.2f} meV, max {out["max_meV"]:.1f} meV (the slice holds C-H at '
            f'{float({k.lower(): v for k, v in fixed.items()}.get("r_ch", 1.08)):.3f} A, the 9-D minimum relaxes it)')
    for lo, hi in ((0, 0.1), (0.1, 0.2), (0.2, 0.3)):
        k = (Eb[m] >= lo) & (Eb[m] < hi)
        if k.any():
            run.detail(f'benchmark {lo:.1f}-{hi:.1f} eV: {int(k.sum())} points, 9-D - 4-D {d[k].mean():+.2f} '
                       f'+- {d[k].std():.2f} meV')
    return out


def _figure(run, E, Efit, batch, nu, info):
    plt = W._mpl()
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for b in sorted(set(batch.tolist())):
        k = batch == b
        ax[0].scatter(E[k], Efit[k], s=6, label='Hessian' if b == 0 else f'batch {b}')
    lim = [0, float(max(E.max(), Efit.max()))]
    ax[0].plot(lim, lim, color='0.4', lw=0.8)
    ax[0].set(xlabel='CCSD(T) (eV above the lowest point)', ylabel='fit (eV)', title='9-D polynomial vs CCSD(T)')
    ax[0].legend(frameon=False, fontsize=8, markerscale=2)
    ax[1].scatter(E, (Efit - E) * 1000, s=6, c=batch, cmap='viridis')
    ax[1].axhline(0, color='0.5', lw=0.8)
    ax[1].set(xlabel='CCSD(T) (eV)', ylabel='fit - CCSD(T) (meV)', title='Residuals')
    lab = [i['label'] for i in info]
    y = np.arange(len(nu))
    ax[2].barh(y, nu, color=['#C8873A' if i['symmetry'] == "a''" else '#2F6DB5' for i in info])
    for k, l in enumerate(lab):
        ex = EXPT.get(l)
        if ex:
            ax[2].plot([ex[1]], [k], 'k|', ms=14, mew=2)
    ax[2].set_yticks(y, [f'{l} ({i["symmetry"]})' for l, i in zip(lab, info)], fontsize=8)
    ax[2].set(xlabel='harmonic wavenumber (cm-1); | experimental fundamental', title='Harmonic modes at the minimum')
    fig.tight_layout()
    fig.savefig(run.p('figures', 'W9_well9d.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------------------------------ torch: exact derivatives
def internal9_t(P):
    """Torch version of geometry.ch2oo_internal9 (same conventions): P (..., 5, 3) -> (..., 9), degrees."""
    import torch
    C, O1, O2, H1, H2 = (P[..., k, :] for k in range(5))

    def ang(a, b, c):
        u, w = a - b, c - b
        x = (u * w).sum(-1)
        y = torch.linalg.norm(torch.cross(u, w, dim=-1), dim=-1)
        return torch.rad2deg(torch.atan2(y, x))           # atan2: well behaved near 0 and 180 degrees

    def dih(a, b, c, d):
        b0, b1, b2 = a - b, c - b, d - c
        b1n = b1 / torch.linalg.norm(b1, dim=-1, keepdim=True)
        v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
        w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
        return torch.rad2deg(torch.atan2((torch.cross(b1n, v, dim=-1) * w).sum(-1), (v * w).sum(-1)))
    d2 = dih(H2, C, O1, O2)
    return torch.stack([torch.linalg.norm(O2 - O1, dim=-1), ang(C, O1, O2), torch.linalg.norm(C - O1, dim=-1),
                        torch.linalg.norm(H1 - C, dim=-1), torch.linalg.norm(H2 - C, dim=-1), ang(H1, C, O1),
                        ang(H2, C, O1), -dih(H1, C, O1, O2), -(torch.remainder(d2, 360.0) - 180.0)], -1)


def poly_t(model, X):
    """A PolyWell's energy (eV) at internal coordinates X (N, n) in torch, with the powers built by repeated
    multiplication so that derivatives stay finite where a coordinate sits exactly at the expansion centre."""
    import torch
    z = (X - model.center) / model.scale
    pw = model.pw.long()
    order = int(pw.max())
    tab = [torch.ones_like(z)]
    for _ in range(order):
        tab.append(tab[-1] * z)
    T = torch.stack(tab, 0)                                # (order+1, N, n)
    terms = torch.ones(z.shape[0], pw.shape[0], dtype=z.dtype)
    for v in range(z.shape[1]):
        terms = terms * T[pw[:, v], :, v].T                # (N, T)
    return terms @ model.coef


def _split_tables(model, k=4):
    """Cached factorisation of a PolyWell's monomials: each term = a(z[:k]) b(z[k:]); unique powers of each group
    (ua, ub) and the coefficient matrix C (len ua, len ub)."""
    key = id(model.pw), k
    cache = getattr(model, '_split_cache', None)
    if cache is not None and cache[0] == key:
        return cache[1]
    pw = np.asarray(model.pw, int)
    coef = np.asarray(model.coef, float).reshape(-1)
    ua, ia = np.unique(pw[:, :k], axis=0, return_inverse=True)
    ub, ib = np.unique(pw[:, k:], axis=0, return_inverse=True)
    C = np.zeros((len(ua), len(ub)))
    np.add.at(C, (ia.reshape(-1), ib.reshape(-1)), coef)
    tab = (ua, ub, C)
    try:
        model._split_cache = (key, tab)
    except Exception:
        pass
    return tab


def poly_energy_grad(model, X, k=4):
    """Energy (N,) and gradient dE/dX (N, n) of a PolyWell at internal coordinates X (N, n), numpy, analytic.
    Every monomial factorises into a part in the first k coordinates and a part in the rest, so
    E = sum_rows((A @ C) * B) with A, B the (few hundred) distinct partial monomials and C their coefficient matrix;
    a derivative replaces one partial monomial by its derivative. Matrix products do the work: several times faster
    than autograd through poly_t for large batches, and equal to it to rounding."""
    ua, ub, C = _split_tables(model, k)
    c = np.asarray(model.center, float)
    s = np.asarray(model.scale, float)
    z = (np.asarray(X, float) - c) / s
    N, n = z.shape
    order = int(max(ua.max(), ub.max()))
    pows = np.ones((n, N, order + 1))
    for q in range(1, order + 1):
        pows[:, :, q] = pows[:, :, q - 1] * z.T
    dpows = np.zeros_like(pows)
    dpows[:, :, 1:] = pows[:, :, :-1] * np.arange(1, order + 1)

    def part(U, cols, dv=None):
        out = np.ones((N, len(U)))
        for j, v in enumerate(cols):
            out = out * np.take(dpows[v] if v == dv else pows[v], U[:, j], axis=1)
        return out
    ga, gb = list(range(k)), list(range(k, n))
    A, B = part(ua, ga), part(ub, gb)
    AC = A @ C
    E = (AC * B).sum(1)
    G = np.empty((N, n))
    for v in ga:
        G[:, v] = ((part(ua, ga, v) @ C) * B).sum(1)
    for v in gb:
        G[:, v] = (AC * part(ub, gb, v)).sum(1)
    return E, G / s


def energy_forces9(model, P, e0=0.0):
    """Energies (N,) eV and Cartesian forces (N, 5, 3) eV/A of the 9-D surface at positions P (N, 5, 3): the
    analytic polynomial gradient (poly_energy_grad) chained through the internal coordinates by autograd."""
    import torch
    R = torch.tensor(np.asarray(P, float), dtype=torch.float64, requires_grad=True)
    Xt = internal9_t(R)
    E, G = poly_energy_grad(model, Xt.detach().numpy())
    (Xt * torch.as_tensor(G)).sum().backward()
    return E - e0, -R.grad.numpy()


def cartesian_hessian_fn(model):
    """P (5, 3) -> Cartesian Hessian (15, 15) of the 9-D polynomial (eV/A^2), by automatic differentiation."""
    import torch

    def f(x):
        return poly_t(model, internal9_t(x.reshape(1, 5, 3)))[0]

    def hess(P):
        x = torch.as_tensor(np.asarray(P, float).reshape(-1), dtype=torch.float64)
        return torch.autograd.functional.hessian(f, x).numpy()
    return hess


def exact_K(model, x):
    """Force constants (9, 9) of a PolyWell at x in derivative units (eV per A^2, A rad, rad^2), by automatic
    differentiation (finite differences of a quintic polynomial err by a few cm-1 in the bends)."""
    import torch
    f = lambda y: poly_t(model, y[None])[0]
    H = torch.autograd.functional.hessian(f, torch.as_tensor(np.asarray(x, float), dtype=torch.float64)).numpy()
    u = unit(NAMES)
    return H / np.outer(u, u)


def refine_minimum(model, x0, tol=1e-9, max_iter=30):
    """Newton iterations on the polynomial in internal coordinates (exact gradient and Hessian). Returns the
    minimum, its energy (eV) and the final gradient norm (eV per A or degree)."""
    import torch
    f = lambda x: poly_t(model, x[None])[0]
    x = torch.as_tensor(np.asarray(x0, float), dtype=torch.float64)
    for _ in range(max_iter):
        g = torch.autograd.functional.jacobian(f, x)
        H = torch.autograd.functional.hessian(f, x)
        step = torch.linalg.solve(H, g)
        x = x - step
        if float(step.abs().max()) < tol:
            break
    g = torch.autograd.functional.jacobian(f, x)
    return x.numpy(), float(f(x)), float(g.abs().max())


# ------------------------------------------------------------------------------------------ 9-D basis-set Delta
def _metric(hz, X):
    """Energy-scaled normal coordinates of the Hessian (the design's farthest-point metric) at X."""
    from ..vibdvr import EV_CM, HBAR2
    floor = (_LAM_FLOOR_CM / EV_CM) ** 2 / HBAR2
    lam = np.maximum(hz['lam'], floor)
    Q = np.linalg.solve(hz['L'], ((X - hz['center']) * unit(NAMES)).T).T
    return Q * np.sqrt(lam)


def stage_well9d_tz(run, jobs=None, orca=None):
    """A larger-basis CCSD(T) set on a spread subset of the 9-D points: the 9-D basis-set correction for
    well9d_fit. [well9d] tz_basis (aug-cc-pVTZ), tz_points (200), tz_e_max_eV (0.6). Finished points are reused,
    so tz_points can be raised and the stage run again."""
    from .shell import _fps as fps
    wc, cc, gc = run.cfg['well9d'], run.cfg['compute'], run.cfg['gs_grid']
    p = run.p('data', 'well9d.npz')
    if not p.exists():
        raise SystemExit(f'{p} not found: run well9d first')
    jobs = int(jobs or wc.get('tz_jobs') or wc.get('jobs') or gc.get('jobs') or cc['jobs'])
    exe = AU._orca_exe(run, orca or cc.get('orca') or run.cfg['audit'].get('orca') or None)
    basis = str(wc.get('tz_basis', 'aug-cc-pVTZ'))
    z = np.load(p, allow_pickle=False)
    hz = np.load(run.p('data', 'well9d_hessian.npz'), allow_pickle=False)
    X, Edz = z['coords'], z['E_total_hartree'][:, 0]
    E = (Edz - np.nanmin(Edz)) * HARTREE_EV
    emax = float(wc.get('tz_e_max_eV', 0.6))
    ok = np.isfinite(E) & (E <= emax)
    idx = np.where(ok)[0]
    n = min(int(wc.get('tz_points', 200)), len(idx))
    start = int(np.argmin(np.linalg.norm(X[idx] - hz['center'], axis=1)))
    pick = idx[fps(_metric(hz, X[idx]), n, start)]
    run.log(f'{basis}: {n} of the {len(idx)} points within {emax:g} eV, spread by farthest-point sampling in the '
            f'normal coordinates; {jobs} parallel')
    for j, nm in enumerate(NAMES):
        run.detail(f'{nm:7s} {X[pick, j].min():9.3f} - {X[pick, j].max():9.3f}')
    Etz, T1 = _compute(run, exe, run.p('data', 'well9d_tz'), X[pick], basis, cc, jobs, basis)
    good = np.isfinite(Etz)
    d = (Etz[good] - Edz[pick][good]) * HARTREE_EV
    save_tz(run, X[pick][good], Etz[good], Edz[pick][good], T1[good], basis, str(hz['basis']))
    run.log(f'{basis} - {hz["basis"]}: {np.ptp(d) * 1000:.0f} meV range over {int(good.sum())} points (shape only '
            f'matters); T1 {np.nanmin(T1):.3f}-{np.nanmax(T1):.3f}; wrote data/well9d_tz.npz. Next: well9d_fit, then '
            f'well9d_vpt2')


def save_tz(run, X, Etz, Edz, T1, basis, base_basis):
    """Write data/well9d_tz.npz with these rows, keeping earlier rows of the same basis pair not among them (so the
    points well9d_extend adds survive a later well9d_tz). Returns the number of rows."""
    tp = run.p('data', 'well9d_tz.npz')
    rows = {}
    if tp.exists():
        t = np.load(tp, allow_pickle=False)
        if str(t['basis']) == basis and str(t['base_basis']) == base_basis:
            for x, a, b, c in zip(t['coords'], t['E_tz_hartree'], t['E_dz_hartree'], t['T1']):
                rows[_key(x)] = (x, a, b, c)
    for x, a, b, c in zip(X, Etz, Edz, T1):
        rows[_key(x)] = (x, a, b, c)
    R = list(rows.values())
    np.savez(tp, coords=np.array([r[0] for r in R]).reshape(-1, len(NAMES)), E_tz_hartree=np.array([r[1] for r in R]),
             E_dz_hartree=np.array([r[2] for r in R]), T1=np.array([r[3] for r in R]), basis=basis,
             base_basis=base_basis)
    return len(R)


def delta9(run, wc, X):
    """The 9-D basis-set correction at X (eV) from data/well9d_tz.npz: a Cs-even polynomial (orders tz_orders,
    the one with the lowest 5-fold held-out error), zero on average over the tz points. Returns (dE, info) or
    (None, None) without enough tz points."""
    tp = run.p('data', 'well9d_tz.npz')
    if not tp.exists():
        return None, None
    t = np.load(tp, allow_pickle=False)
    Xt, d = t['coords'], (t['E_tz_hartree'] - t['E_dz_hartree']) * HARTREE_EV
    if len(Xt) < 30:
        run.log(f'only {len(Xt)} points in data/well9d_tz.npz: 9-D correction skipped')
        return None, None
    x0 = np.load(run.p('data', 'well9d_hessian.npz'))['center']
    even = [NAMES.index(n) for n in NINE_OUT_OF_PLANE]
    scale = _scale(NAMES)
    folds = [np.arange(len(Xt)) % 5 == k for k in range(5)]
    cv = {}
    for order in [int(o) for o in wc.get('tz_orders', [1, 2, 3])]:
        pw = np.array(cs_powers(len(NAMES), order, even))
        if len(pw) > 0.6 * len(Xt):
            continue
        e = np.concatenate([_predict(Xt[m], x0, pw, scale, _fit(Xt[~m], d[~m], x0, pw, scale)) - d[m] for m in folds])
        cv[order] = {'terms': int(len(pw)), 'heldout_rms_meV': float(np.sqrt(np.mean(e ** 2)) * 1000)}
    if not cv:
        run.log('too few tz points for any correction order: skipped')
        return None, None
    best = min(cv, key=lambda o: cv[o]['heldout_rms_meV'])
    pw = np.array(cs_powers(len(NAMES), best, even))
    c = _fit(Xt, d, x0, pw, scale)
    shift = float(np.mean(_predict(Xt, x0, pw, scale, c)))
    dE = _predict(X, x0, pw, scale, c) - shift
    info = {'basis': str(t['basis']), 'base_basis': str(t['base_basis']), 'n': int(len(Xt)), 'orders': cv,
            'order': best, 'range_meV': float(np.ptp(d) * 1000), 'box': {n: [float(Xt[:, j].min()), float(Xt[:, j].max())]
                                                                        for j, n in enumerate(NAMES)}}
    run.log(f'9-D correction {info["basis"]} - {info["base_basis"]} from {len(Xt)} points ({info["range_meV"]:.0f} meV '
            f'range): held-out ' + ', '.join(f'order {o} {v["heldout_rms_meV"]:.2f} meV' for o, v in cv.items())
            + f' -> order {best}')
    return dE, info


# ------------------------------------------------------------------------------------------ VPT2
def _herzberg(info, omega):
    """Herzberg numbers for Cs: a' modes by decreasing wavenumber v1.., then a'' modes."""
    ap = sorted([k for k, m in enumerate(info) if m['symmetry'] == "a'"], key=lambda k: -omega[k])
    app = sorted([k for k, m in enumerate(info) if m['symmetry'] != "a'"], key=lambda k: -omega[k])
    return {k: f'v{n + 1}' for n, k in enumerate(ap + app)}


def stage_well9d_vpt2(run):
    """Anharmonic fundamentals of the 9-D surface by VPT2 with Fermi resonances (GVPT2); see diabml/vpt2.py."""
    from .. import vpt2 as V
    from ..vibdvr import ISOTOPIC_MASS
    from ..wellfit import PolyWell
    wc = run.cfg['well9d']
    mp = run.p('models', 'well9d.npz')
    if not mp.exists():
        raise SystemExit(f'{mp} not found: run well9d_fit first')
    model = PolyWell.load(mp)
    rp = run.p('results', 'well9d.json')
    x0 = np.array([json.loads(rp.read_text())['minimum'][n] for n in NAMES]) if rp.exists() else \
        model.center.numpy()
    xm, Em, gmax = refine_minimum(model, x0)
    run.log(f'model {mp.name} ({model.meta.get("source", "")}, order {model.meta.get("order")}); minimum refined '
            f'by Newton: largest gradient {gmax:.1e} eV per A or degree; moved {np.abs(xm - x0).max():.1e}')
    P0 = ch2oo_positions9(xm[None])[0]
    hess = cartesian_hessian_fn(model)
    modes = V.cartesian_modes(hess(P0), P0, ISOTOPIC_MASS)
    from ..vibdvr import g_matrix
    K, G0 = exact_K(model, xm), g_matrix(xm[None], NAMES, {})[0]
    lam, L, nu_int = normal_modes(K, G0)
    info = mode_info(K, L)
    dev = float(np.abs(np.sort(modes.omega) - np.sort(nu_int)).max())
    run.log(f'harmonic from the Cartesian Hessian {", ".join(f"{v:.1f}" for v in modes.omega[::-1])} cm-1; '
            f'internal-coordinate GF agrees within {dev:.2f} cm-1')
    hz = _herzberg(info, modes.omega)
    step = float(wc.get('vpt2_step', 0.02))
    f3, f4, asym = V.force_constants(hess, modes, step)
    zeta, B = V.coriolis(modes)
    res = V.find_resonances(modes.omega, f3, float(wc.get('vpt2_max_gap', 200.0)), float(wc.get('vpt2_martin', 1.0)))
    out = V.gvpt2(modes.omega, f3, f4, B, zeta, res)
    run.detail(f'force constants by finite differences of exact Hessians, step {step:g} (dimensionless); cubic '
               f'permutation asymmetry {asym:.1e} (relative)')
    ref = json.loads((W.C.PACKAGE_ROOT / 'data' / 'reference' / 'ch2oo_rotational.json').read_text())['band_origins']
    names_v = [hz[k] for k in range(len(modes.omega))]
    lab = lambda st: ' + '.join(f'{c}{names_v[i]}' if c > 1 else names_v[i]
                                for i, c in enumerate(st) if c) if sum(st) else '0'
    for r in res:
        run.log(f'Fermi resonance {names_v[r["a"]]} ~ {lab([1 if i in (r["b"], r["c"]) else 0 for i in range(9)]) if r["b"] != r["c"] else "2" + names_v[r["b"]]}: '
                f'gap {r["gap"]:+.1f} cm-1, coupling {r["W"]:+.1f} cm-1 (Martin {r["martin"]:.1f})')
    rows = []
    order = sorted(range(len(modes.omega)), key=lambda k: int(names_v[k][1:]))
    run.log(f'{"":4s} {"mode":20s} {"sym":4s} {"harm":>7s} {"VPT2":>8s} {"GVPT2":>8s} {"expt":>8s}')
    for k in order:
        ex = ref.get(names_v[k])
        rows.append({'mode': names_v[k], 'label': info[k]['label'], 'symmetry': info[k]['symmetry'],
                     'harmonic': float(modes.omega[k]), 'vpt2': float(out['fundamentals_vpt2'][k]),
                     'vpt2_deperturbed': float(out['fundamentals_dvpt2'][k]), 'gvpt2': float(out['fundamentals'][k]),
                     'expt': ex})
        run.log(f'{names_v[k]:4s} {info[k]["label"]:20s} {info[k]["symmetry"]:4s} {modes.omega[k]:7.1f} '
                f'{out["fundamentals_vpt2"][k]:8.1f} {out["fundamentals"][k]:8.1f} '
                + (f'{ex:8.1f}  ({out["fundamentals"][k] - ex:+.1f})' if ex else ''))
    # two-quantum levels in the region of the experimental overtone/combination bands
    two = []
    n = len(modes.omega)
    for i in range(n):
        for j in range(i, n):
            v = [0] * n
            v[i] += 1
            v[j] += 1
            E = V.level(modes.omega, out['chi'], v)
            if 1100 <= E <= 1600:
                two.append({'state': lab(v), 'deperturbed': E})
    two.sort(key=lambda t: t['deperturbed'])
    run.log('two-quantum levels 1100-1600 cm-1 (deperturbed): ' + ', '.join(f'{t["state"]} {t["deperturbed"]:.0f}'
                                                                            for t in two)
            + f'; expt. 2v9 {ref.get("2v9")}, combination {ref.get("combination")}')
    for pd in out['polyads']:
        st = [lab(s) for s in pd['states']]
        comp = []
        for e, vec in zip(pd['energies'], np.array(pd['vectors']).T):
            comp.append(f'{e:.1f} (' + ', '.join(f'{100 * c * c:.0f}% {s}' for s, c in zip(st, vec) if c * c > 0.05) + ')')
        run.log('polyad ' + ' / '.join(st) + ': ' + '; '.join(comp))
    run.log(f'zero-point energy {out["zpe"]:.1f} cm-1 (harmonic {out["zpe_harmonic"]:.1f})')
    run.detail('chi (deperturbed, cm-1) ' + ' '.join(f'{names_v[j]:>7s}' for j in order))
    for i in order:
        run.detail(f'chi {names_v[i]:4s}                ' + ' '.join(f'{out["chi"][i, j]:7.2f}' for j in order))
    big = sorted(((abs(f3[i, j, k]), i, j, k) for i in range(n) for j in range(i, n) for k in range(j, n)), reverse=True)
    run.detail('largest cubic constants (cm-1): ' + ', '.join(f'{names_v[i]},{names_v[j]},{names_v[k]} {f3[i, j, k]:+.0f}'
                                                              for _, i, j, k in big[:10]))
    zb = sorted(((abs(zeta[a, i, j]), a, i, j) for a in range(3) for i in range(n) for j in range(i + 1, n)),
                reverse=True)
    run.detail('largest Coriolis zeta: ' + ', '.join(f'{"abc"[a]} {names_v[i]},{names_v[j]} {zeta[a, i, j]:+.3f}'
                                                     for _, a, i, j in zb[:6]) + f'; B {", ".join(f"{b:.4f}" for b in B)}')
    res_out = {'model': str(mp), 'source': model.meta.get('source'), 'minimum': dict(zip(NAMES, xm.tolist())),
               'gradient_max': gmax, 'harmonic_gf_vs_cartesian_cm-1': dev, 'step': step, 'cubic_asymmetry': asym,
               'modes': rows, 'two_quantum': two, 'resonances': [dict(r, a=names_v[r['a']], b=names_v[r['b']],
                                                                    c=names_v[r['c']]) for r in res],
               'polyads': [{'states': [lab(s) for s in pd['states']], 'energies': pd['energies'],
                            'vectors': pd['vectors']} for pd in out['polyads']],
               'zpe_cm-1': out['zpe'], 'zpe_harmonic_cm-1': out['zpe_harmonic'], 'B_e_cm-1': B.tolist(),
               'chi': out['chi'].tolist(), 'mode_names': names_v}
    run.p('results', 'well9d_vpt2.json').write_text(json.dumps(res_out, indent=1, default=float))
    run.log('wrote results/well9d_vpt2.json')


# ------------------------------------------------------------------------------------------ 9-D dipole surface
BOHR = 1.8897261246               # bohr per Angstrom
DEBYE = 2.541746                  # Debye per atomic unit
_Z = np.array([6.0, 8.0, 8.0, 1.0, 1.0])
_MASS = np.array([12.0, 15.994915, 15.994915, 1.007825, 1.007825])
_FIELDS = {'xp': (0, 1), 'xm': (0, -1), 'yp': (1, 1), 'ym': (1, -1), 'zp': (2, 1), 'zm': (2, -1)}


def ff_input(atoms, basis, maxcore, field):
    """A CCSD(T) single point in a uniform electric field (a.u.; ORCA %scf EField, no symmetry: the field stays in
    the frame of the input coordinates)."""
    from .method_cut import cc_input
    t = cc_input(atoms, basis, maxcore).replace('written by diabml method_cut: CCSD(T) check of the NEVPT2 surface',
                                                'written by diabml well9d_dipole: CCSD(T) in a finite field')
    f = ', '.join(f'{v:.6f}' for v in field)
    return t.replace('* xyz 0 1', f'%scf\n  EField {f}\nend\n\n* xyz 0 1')


def parse_scf_dipole(path):
    """ORCA's (SCF) 'Total Dipole Moment' (a.u., input frame), the last one in the output, or None."""
    import re
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return None
    m = re.findall(r'Total Dipole Moment\s*:\s*(\S+)\s+(\S+)\s+(\S+)', p.read_text(errors='replace'))
    return np.array([float(v) for v in m[-1]]) if m else None


def field_convention(d, mu_scf, P):
    """How ORCA's finite-field derivative d = -(E(+F) - E(-F)) / 2F (a.u.) relates to the dipole: the sign, and
    whether the nuclear dipole (about the input origin or the centre of mass) must be added. Each candidate is
    compared with ORCA's own SCF dipole of the same runs; the closest one is returned with its RMS (a.u.) and the
    runner-up's. Correlation changes the dipole by a few tenths of a debye; the wrong choices miss by many."""
    Pb = np.asarray(P, float) * BOHR
    nuc0 = np.einsum('a,nab->nb', _Z, Pb)
    com = (np.einsum('a,nab->nb', _MASS, Pb) / _MASS.sum())[:, None]
    nucc = np.einsum('a,nab->nb', _Z, Pb - com)
    cands = {}
    for s in (1, -1):
        cands[f'sign {s:+d}, total'] = s * d
        cands[f'sign {s:+d}, + nuclear (input origin)'] = s * d + nuc0
        cands[f'sign {s:+d}, + nuclear (centre of mass)'] = s * d + nucc
    rms = {k: float(np.sqrt(np.nanmean((v - mu_scf) ** 2))) for k, v in cands.items()}
    best = sorted(rms, key=rms.get)
    return best[0], cands[best[0]], rms[best[0]], rms[best[1]]


def stage_well9d_dipole(run, jobs=None, orca=None):
    """Finite-field CCSD(T) dipoles at a spread subset of the 9-D points and the 9-D dipole surface fitted to
    them (diabml/dipole9.py): data/well9d_dipoles.npz, models/dipole9d.npz. Rerunning reuses finished outputs
    (so it refits for free, or adds points after dipole_points is raised)."""
    from .shell import _fps as fps
    from ..dipole9 import VectorDipole9
    wc, cc, gc = run.cfg['well9d'], run.cfg['compute'], run.cfg['gs_grid']
    p = run.p('data', 'well9d.npz')
    if not p.exists():
        raise SystemExit(f'{p} not found: run well9d first')
    jobs = int(jobs or wc.get('dipole_jobs') or wc.get('jobs') or gc.get('jobs') or cc['jobs'])
    exe = AU._orca_exe(run, orca or cc.get('orca') or run.cfg['audit'].get('orca') or None)
    basis = str(wc.get('dipole_basis') or _basis(run, wc))
    F = float(wc.get('dipole_field_au', 0.001))
    z = np.load(p, allow_pickle=False)
    hz = np.load(run.p('data', 'well9d_hessian.npz'), allow_pickle=False)
    X, Edz = z['coords'], z['E_total_hartree'][:, 0]
    E = (Edz - np.nanmin(Edz)) * HARTREE_EV
    emax = float(wc.get('dipole_e_max_eV', 0.8))
    idx = np.where(np.isfinite(E) & (E <= emax))[0]
    n = min(int(wc.get('dipole_points', 500)), len(idx))
    start = int(np.argmin(np.linalg.norm(X[idx] - hz['center'], axis=1)))
    Xs = X[idx[fps(_metric(hz, X[idx]), n, start)]]
    planar = np.all(np.abs(Xs[:, [NAMES.index(m) for m in NINE_OUT_OF_PLANE]]) < 1e-6, axis=1)
    jobs_list = []
    wd = run.p('data', 'well9d_dipole')
    wd.mkdir(parents=True, exist_ok=True)
    for x, pl in zip(Xs, planar):
        for tag, (ax, s) in _FIELDS.items():
            if pl and ax == 2:
                continue
            f = [0.0, 0.0, 0.0]
            f[ax] = s * F
            nm = f'{_key(x)}_{tag}'
            if not (wd / f'{nm}.inp').exists():
                (wd / f'{nm}.inp').write_text(ff_input(ch2oo_xyz9(x), basis, cc['maxcore'], f))
            jobs_list.append(nm)
    run.log(f'dipoles: {n} of the {len(idx)} points within {emax:g} eV (farthest-point sampling); finite field '
            f'{F:g} a.u., CCSD(T)/{basis}, {len(jobs_list)} runs ({int(planar.sum())} planar points need no z field); '
            f'{jobs} parallel')
    from .method_cut import parse_cc

    def one(nm):
        if parse_cc(wd / f'{nm}.out') is not None:
            return 'reused'
        return AU._run_one(exe, wd, nm, float(cc['timeout_s']))[1]
    t0, done, status = time.time(), 0, {}
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        for s in ex.map(one, jobs_list):
            done += 1
            status[s] = status.get(s, 0) + 1
            if done % max(1, len(jobs_list) // 10) == 0:
                run.log(f'  finite field: {done}/{len(jobs_list)} runs ({(time.time() - t0) / 60:.1f} min)')
    run.detail('finite field: ' + ', '.join(f'{k} {v}' for k, v in sorted(status.items())))
    d = np.full((n, 3), np.nan)
    scf = np.full((n, 3), np.nan)
    for i, (x, pl) in enumerate(zip(Xs, planar)):
        sc = []
        for ax, (tp, tm) in enumerate((('xp', 'xm'), ('yp', 'ym'), ('zp', 'zm'))):
            if pl and ax == 2:
                d[i, 2] = 0.0
                continue
            rp, rm = parse_cc(wd / f'{_key(x)}_{tp}.out'), parse_cc(wd / f'{_key(x)}_{tm}.out')
            if rp is None or rm is None:
                continue
            d[i, ax] = -(rp['E'] - rm['E']) / (2 * F)
            for t in (tp, tm):
                v = parse_scf_dipole(wd / f'{_key(x)}_{t}.out')
                if v is not None:
                    sc.append(v)
        if sc:
            scf[i] = np.mean(sc, 0)
    ok = np.all(np.isfinite(d), 1) & np.all(np.isfinite(scf), 1)
    if ok.sum() < 30:
        raise SystemExit(f'only {int(ok.sum())} complete dipoles (outputs in {wd}); rerun to retry failed points')
    P = ch2oo_positions9(Xs[ok])
    conv, mu_au, rms, rms2 = field_convention(d[ok], scf[ok], P)
    run.log(f'field convention: {conv} (RMS vs ORCA SCF dipole {rms * DEBYE:.3f} D; next best '
            f'{rms2 * DEBYE:.2f} D)')
    if rms * DEBYE > 1.5 or rms2 < 3 * rms:
        raise SystemExit('the finite-field dipoles do not match the SCF dipoles under any convention: check the '
                         f'outputs in {wd} (field direction, orientation)')
    mu = mu_au * DEBYE
    corr = np.linalg.norm(mu - scf[ok] * DEBYE, axis=1)
    run.log(f'{int(ok.sum())} dipoles: |mu| {np.linalg.norm(mu, axis=1).min():.2f}-{np.linalg.norm(mu, axis=1).max():.2f} D; '
            f'CCSD(T) - SCF {corr.mean():.2f} D on average (max {corr.max():.2f})')
    np.savez(run.p('data', 'well9d_dipoles.npz'), coords=Xs[ok], positions=P, mu_D=mu, mu_scf_D=scf[ok] * DEBYE,
             field_au=F, basis=basis, convention=conv)
    m = VectorDipole9.fit(P, mu, [int(o) for o in wc.get('dipole_orders', [1, 2, 3])],
                          center=hz['center'], log=run.log)
    for o, v in m.meta['orders'].items():
        run.detail(f'dipole order {o}: {v["coefficients"]} coefficients, held-out RMS {v["heldout_rms_D"]:.4f} D '
                   f'(max {v["heldout_max_D"]:.3f})')
    m.meta.update({'basis': basis, 'field_au': F, 'convention': conv, 'method': 'CCSD(T) finite field'})
    m.save(run.p('models', 'dipole9d.npz'))
    run.log('dipole surface: held-out ' + ', '.join(f'order {o} {v["heldout_rms_D"]:.4f} D'
                                                   for o, v in m.meta['orders'].items())
            + f' -> order {m.meta["order"]}; wrote models/dipole9d.npz')
    mp = run.p('models', 'well9d.npz')
    if mp.exists():
        from ..dynamics.ir import harmonic_ir9
        rows, xm = harmonic_ir9(str(mp), str(run.p('models', 'dipole9d.npz')))
        mu0 = m.predict(ch2oo_positions9(xm[None]))[0]
        run.log(f'at the 9-D minimum: |mu| {np.linalg.norm(mu0):.3f} D; harmonic intensities (relative) '
                + ', '.join(f'{r["cm-1"]:.0f} {r["intensity"]:.2f}' for r in sorted(rows, key=lambda r: -r['cm-1'])))
    run.p('results', 'well9d_dipole.json').write_text(json.dumps(
        {'n': int(ok.sum()), 'basis': basis, 'field_au': F, 'convention': conv, 'rms_vs_scf_D': rms * DEBYE,
         'fit': m.meta}, indent=1, default=float))
