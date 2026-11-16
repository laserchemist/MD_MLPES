"""VPT2 / GVPT2 (diabml/vpt2.py) against exact levels, and its CH2OO plumbing in well9d."""
import itertools

import numpy as np
import pytest

from diabml import vpt2 as V


def _sym3(n, d):
    f = np.zeros((n, n, n))
    for (a, b, c), v in d.items():
        for p in set(itertools.permutations((a, b, c))):
            f[p] = v
    return f


def _dvr2(w, f3, f4, n=64, L=8.0):
    """Exact levels (cm-1 above the ground state) of the two-mode model
    H = sum w_i (p_i^2 + q_i^2)/2 + (1/6) sum f3 qqq + (1/24) f4_iiii q_i^4 + (1/4) f4_0011 q_0^2 q_1^2 (sinc DVR)."""
    q = np.linspace(-L, L, n)
    dq = q[1] - q[0]
    i = np.arange(n)
    D = i[:, None] - i[None, :]
    T = np.where(D == 0, np.pi ** 2 / 3, 2 * (-1.0) ** D / np.where(D == 0, 1, D) ** 2) / (2 * dq ** 2)
    Q1, Q2 = np.meshgrid(q, q, indexing='ij')
    qs = [Q1, Q2]
    Vg = sum(w[k] / 2 * qs[k] ** 2 for k in range(2))
    for a, b, c in itertools.product(range(2), repeat=3):
        Vg = Vg + f3[a, b, c] * qs[a] * qs[b] * qs[c] / 6
    for a in range(2):
        Vg = Vg + f4[a, a] * qs[a] ** 4 / 24
    Vg = Vg + f4[0, 1] * Q1 ** 2 * Q2 ** 2 / 4
    I = np.eye(n)
    E = np.linalg.eigvalsh(w[0] * np.kron(T, I) + w[1] * np.kron(I, T) + np.diag(Vg.ravel()))
    return E - E[0]


@pytest.mark.parametrize('term', ['diag', 'cross', 'f000', 'f001', 'f011', 'all'])
def test_vpt2_matches_exact_levels_without_resonance(term):
    """Every term of chi (diagonal and cross quartic, each cubic pattern) against exact two-mode levels; the
    remaining difference is higher order (it falls 16x when the anharmonicity is halved)."""
    w = np.array([1000.0, 1700.0])
    z = np.zeros((2, 2, 2))
    f3 = {'f000': _sym3(2, {(0, 0, 0): -60}), 'f001': _sym3(2, {(0, 0, 1): 30}), 'f011': _sym3(2, {(0, 1, 1): 25}),
          'all': _sym3(2, {(0, 0, 0): -60, (1, 1, 1): -80, (0, 0, 1): 30, (0, 1, 1): 25})}.get(term, z)
    f4 = {'diag': np.diag([12.0, 15.0]), 'cross': np.array([[0, 5.0], [5.0, 0]]),
          'all': np.array([[12.0, 5.0], [5.0, 15.0]])}.get(term, np.zeros((2, 2)))
    r = V.gvpt2(w, f3, f4, res=[])
    E = _dvr2(w, f3, f4)
    assert np.allclose(r['fundamentals_vpt2'], E[[1, 2]], atol=0.01)
    assert np.allclose(r['fundamentals'], r['fundamentals_vpt2'])                 # nothing to treat


def test_gvpt2_fermi_resonance_matches_exact_and_plain_vpt2_fails():
    w = np.array([800.0, 1610.0])                                # w1 ~ 2 w0: a type-1 Fermi resonance
    f3 = _sym3(2, {(0, 0, 0): -30, (1, 1, 1): -40, (0, 0, 1): 60})
    f4 = np.array([[6.0, 3.0], [3.0, 8.0]])
    res = V.find_resonances(w, f3)
    assert len(res) == 1 and (res[0]['a'], res[0]['b'], res[0]['c']) == (1, 0, 0)
    assert res[0]['W'] == pytest.approx(60 / 4) and res[0]['gap'] == pytest.approx(-10.0)
    r = V.gvpt2(w, f3, f4)
    E = _dvr2(w, f3, f4)
    assert abs(r['fundamentals'][0] - E[1]) < 0.01
    # the resonant fundamental: GVPT2 within 0.1 cm-1 of the exact level; plain VPT2 12 cm-1 off
    assert abs(r['fundamentals'][1] - E[3]) < 0.1 and abs(r['fundamentals_vpt2'][1] - E[3]) > 10
    assert np.allclose(sorted(r['polyads'][0]['energies']), E[2:4], atol=0.1)    # v1 and 2 v0, mixed
    assert len(r['polyads']) == 1 and len(r['polyads'][0]['states']) == 2


def test_type2_resonance_detection_and_coupling():
    w = np.array([700.0, 1000.0, 1705.0])
    f3 = _sym3(3, {(0, 1, 2): 50.0})
    res = V.find_resonances(w, f3)
    assert [(r['a'], r['b'], r['c']) for r in res] == [(2, 0, 1)]
    assert res[0]['W'] == pytest.approx(50 / (2 * np.sqrt(2)))
    chi_p = V.chi_matrix(w, f3, np.zeros((3, 3)))
    chi_d = V.chi_matrix(w, f3, np.zeros((3, 3)), res=[(2, 0, 1)])
    assert abs(chi_p[0, 1]) > 10 * abs(chi_d[0, 1])                               # the -5 cm-1 denominator removed


def test_level_and_zpe_bookkeeping():
    w = np.array([1000.0, 1700.0])
    chi = np.array([[-5.0, -3.0], [-3.0, -8.0]])
    assert V.level(w, chi, [1, 0]) == pytest.approx(1000 + 2 * -5 + 0.5 * -3)
    assert V.level(w, chi, [0, 2]) == pytest.approx(2 * 1700 + 6 * -8 + -3)
    assert V.zpe(w, chi) == pytest.approx(1350 + (-5 - 3 - 8) / 4)


# ------------------------------------------------------------------------------------------ CH2OO plumbing
def _model(order=4, seed=0):
    """A Cs-even 9-D polynomial well with realistic curvature (eV, A, deg) as a PolyWell."""
    from diabml.wellfit import PolyWell
    from diabml.workflow import well9d as N9
    x0 = np.array([1.35, 117.7, 1.274, 1.094, 1.092, 118.7, 114.8, 0.0, 0.0])
    s = N9._scale(N9.NAMES)
    pw = np.array(N9.cs_powers(9, order, (7, 8)))
    rng = np.random.default_rng(seed)
    coef = np.zeros(len(pw))
    k = {'r': [38.0, 0, 55.0, 33.0, 33.0], 'a': 0.0}
    diag = [38.0, 8.0 * (np.pi / 180) ** 2, 55.0, 33.0, 33.0, 4.0 * (np.pi / 180) ** 2, 4.0 * (np.pi / 180) ** 2,
            1.2 * (np.pi / 180) ** 2, 1.2 * (np.pi / 180) ** 2]
    for j in range(9):
        e = [0] * 9
        e[j] = 2
        coef[[tuple(p) for p in pw].index(tuple(e))] = 0.5 * diag[j] * s[j] ** 2
    hi = pw.sum(1) >= 3
    coef[hi] += rng.normal(0, 2e-3, hi.sum())
    return PolyWell(N9.NAMES, x0, s, pw, coef, {n: [-1e9, 1e9] for n in N9.NAMES}), x0


def test_internal9_torch_equals_numpy():
    import torch
    from diabml.geometry import ch2oo_internal9, ch2oo_positions9
    from diabml.workflow import well9d as N9
    rng = np.random.default_rng(5)
    X = np.array([1.35, 117.7, 1.27, 1.09, 1.09, 118, 115, 0, 0]) + rng.normal(0, 1, (20, 9)) * [0.03, 3, 0.03, 0.03,
                                                                                                 0.03, 3, 3, 15, 15]
    P = ch2oo_positions9(X)
    assert np.allclose(N9.internal9_t(torch.as_tensor(P)).numpy(), ch2oo_internal9(P), atol=1e-9)


def test_poly_t_matches_polywell_and_has_finite_gradient_at_centre():
    import torch
    from diabml.workflow import well9d as N9
    m, x0 = _model()
    X = x0 + np.random.default_rng(1).normal(0, 0.01, (5, 9))
    assert np.allclose(N9.poly_t(m, torch.as_tensor(X)).numpy(), m.predict(X)[:, 0], atol=1e-12)
    x = torch.tensor(x0[None], dtype=torch.float64, requires_grad=True)
    N9.poly_t(m, x).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_cartesian_modes_match_internal_gf_and_zeta_symmetry():
    """Harmonic wavenumbers from the Eckart-projected Cartesian Hessian equal the Wilson GF ones; zeta is
    antisymmetric and, for the planar molecule, zeta^c couples in-plane modes only (a, b: in-plane with
    out-of-plane)."""
    import torch
    from diabml.geometry import ch2oo_positions9
    from diabml.vibdvr import ISOTOPIC_MASS, _harmonic
    from diabml.workflow import well9d as N9
    m, x0 = _model()
    xm, _, g = N9.refine_minimum(m, x0)
    assert g < 1e-8
    P0 = ch2oo_positions9(xm[None])[0]
    hess = N9.cartesian_hessian_fn(m)
    modes = V.cartesian_modes(hess(P0), P0, ISOTOPIC_MASS)
    from diabml.vibdvr import g_matrix
    K, G0 = N9.exact_K(m, xm), g_matrix(xm[None], N9.NAMES, {})[0]
    _, L, nu = N9.normal_modes(K, G0)
    # the GF route differentiates G numerically: agreement to ~0.03%
    assert np.allclose(np.sort(modes.omega), np.sort(nu), rtol=5e-4)
    zeta, B = V.coriolis(modes)
    assert np.allclose(zeta, -zeta.transpose(0, 2, 1), atol=1e-12) and B[0] > B[1] > B[2]
    info = N9.mode_info(K, L)
    oop = np.array([i['symmetry'] == "a''" for i in info])
    # match Cartesian modes to internal ones by wavenumber (both ascending)
    o_c, o_i = np.argsort(modes.omega), np.argsort(nu)
    oop_c = np.zeros(9, bool)
    oop_c[o_c] = oop[o_i]
    zc = zeta[2]
    assert np.abs(zc[np.ix_(oop_c, ~oop_c)]).max() < 1e-8                      # c: in-plane rotation
    assert np.abs(zeta[0][np.ix_(~oop_c, ~oop_c)]).max() < 1e-8


def test_force_constants_converge_with_step():
    from diabml.geometry import ch2oo_positions9
    from diabml.vibdvr import ISOTOPIC_MASS
    from diabml.workflow import well9d as N9
    m, x0 = _model()
    xm, _, _ = N9.refine_minimum(m, x0)
    P0 = ch2oo_positions9(xm[None])[0]
    hess = N9.cartesian_hessian_fn(m)
    modes = V.cartesian_modes(hess(P0), P0, ISOTOPIC_MASS)
    a3, a4, asym = V.force_constants(hess, modes, 0.02)
    b3, b4, _ = V.force_constants(hess, modes, 0.01)
    assert np.abs(a3 - b3).max() < 1e-3 * np.abs(a3).max() and asym < 1e-2
    assert np.abs(a4 - b4).max() < 1e-2 * np.abs(a4).max() + 1e-3
