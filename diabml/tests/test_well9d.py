"""well9d: the CH2OO well in nine internal coordinates (geometry, Hessian, sampling, symmetric fit, stage)."""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from diabml.geometry import NINE, ch2oo_internal9, ch2oo_positions9, ch2oo_xyz
from diabml.workflow import ch2oo as W
from diabml.workflow import well9d as N9

DATA = Path(__file__).parent / 'data'
X0 = np.array([1.35, 118.0, 1.27, 1.08, 1.09, 117.0, 119.5, 0.0, 0.0])


# ------------------------------------------------------------------------------------------ geometry
def test_positions9_planar_matches_reduced_builder():
    """Planar, symmetric 9-D geometry = the 4-D builder's geometry (same atoms, same frame)."""
    a = ch2oo_xyz(r_oo=1.35, a_coo=118.0, r_co=1.27, r_ch=1.08, a_hco=117.0)
    P = ch2oo_positions9([[1.35, 118.0, 1.27, 1.08, 1.08, 117.0, 117.0, 0.0, 0.0]])[0]
    assert np.abs(np.array([p for _, p in a]) - P).max() < 1e-12


def test_internal9_round_trip_any_orientation():
    """ch2oo_internal9 inverts ch2oo_positions9, also after a rotation and translation of the molecule."""
    rng = np.random.default_rng(1)
    X = X0 + rng.normal(0, 1, (50, 9)) * np.array([0.05, 5, 0.05, 0.05, 0.05, 5, 5, 20, 20])
    P = ch2oo_positions9(X)
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]) @ \
        np.array([[1, 0, 0], [0, np.cos(1.1), -np.sin(1.1)], [0, np.sin(1.1), np.cos(1.1)]])
    P2 = P @ R.T + np.array([0.3, -2.0, 5.0])
    assert np.allclose(ch2oo_internal9(P2), X, atol=1e-9)


def test_out_of_plane_is_a_mirror_pair():
    """(d_H1, d_H2) -> (-d_H1, -d_H2) reflects the molecule through its plane (z -> -z)."""
    x = X0.copy()
    x[7], x[8] = 12.0, -7.0
    y = x.copy()
    y[7], y[8] = -12.0, 7.0
    P, Q = ch2oo_positions9(np.array([x, y]))
    assert np.allclose(P * [1, 1, -1], Q, atol=1e-12)
    assert abs(P[3, 2]) > 0.1 and abs(P[4, 2]) > 0.05      # both H really leave the plane


# ------------------------------------------------------------------------------------------ basis
def test_cs_powers_counts_and_parity():
    for n, order in ((3, 3), (4, 4), (9, 2)):
        allp = N9.cs_powers(n, order)
        assert len(allp) == math.comb(n + order, order)
        ref = [p for d in range(order + 1) for p in itertools.product(range(d + 1), repeat=n) if sum(p) == d]
        assert sorted(allp) == sorted(ref)
    even = N9.cs_powers(9, 4, (7, 8))
    assert all((p[7] + p[8]) % 2 == 0 for p in even)
    assert len(even) < math.comb(13, 4) and (0,) * 7 + (1, 1) in even and (0,) * 7 + (1, 0) not in even


# ------------------------------------------------------------------------------------------ Hessian
def test_fd_hessian_recovers_a_known_quadratic_plus_cubic():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(9, 9))
    K = A @ A.T + 9 * np.eye(9)
    g = rng.normal(size=9)
    u = N9.unit(N9.NAMES)
    h = N9.fd_steps({})
    V = lambda X: ((X - X0) * u) @ g + 0.5 * np.einsum('ni,ij,nj->n', (X - X0) * u, K, (X - X0) * u) \
        + 0.3 * (((X - X0) * u) ** 3).sum(1)
    X, tags = N9.fd_design(X0, h)
    assert len(X) == 1 + 2 * 9 + 4 * 36
    g2, K2 = N9.fd_derivatives(V(X), tags, h)
    assert np.allclose(g2, g, atol=2e-3 * np.abs(g).max()) and np.allclose(K2, K, atol=1e-6 * np.abs(K).max())


def test_normal_modes_normalisation_and_sampling_energy():
    """L^T G^-1 L = 1 and L^T K L = diag(lambda); candidates carry the harmonic energy they were drawn with."""
    from diabml.vibdvr import g_matrix
    G = g_matrix(X0[None], N9.NAMES, {})[0]
    rng = np.random.default_rng(2)
    A = rng.normal(size=(9, 9))
    K = A @ A.T * 5 + 20 * np.eye(9)
    lam, L, nu = N9.normal_modes(K, G)
    assert np.allclose(L.T @ np.linalg.inv(G) @ L, np.eye(9), atol=1e-8)
    assert np.allclose(L.T @ K @ L, np.diag(lam), atol=1e-6 * lam.max())
    assert (nu > 0).all() and np.all(np.diff(nu) >= 0)
    X, Eh, Z = N9.sample_pool(X0, lam, L, 400, 0.5, d_max=90.0, seed=3)
    f = N9.harmonic_model(X0, np.zeros(9), K)
    assert len(X) == 400 and np.allclose(f(X), Eh, rtol=1e-8) and Eh.max() <= 0.5
    assert np.allclose(0.5 * (Z ** 2).sum(1), Eh, rtol=1e-8)               # FPS metric = harmonic energy


def test_g_matrix_planar_block_structure():
    """At a planar geometry the out-of-plane dihedrals do not couple kinetically to the in-plane coordinates."""
    from diabml.vibdvr import g_matrix
    G = g_matrix(X0[None], N9.NAMES, {})[0]
    io, oo = slice(0, 7), slice(7, 9)
    assert np.abs(G[io, oo]).max() < 1e-8 * np.abs(G).max()
    assert np.all(np.linalg.eigvalsh(G) > 0)


def test_mode_labels_from_pair_signs():
    names = N9.NAMES
    K = np.eye(9)
    L = np.zeros((9, 6))
    pairs = [('r_CH1', 'r_CH2', 1), ('r_CH1', 'r_CH2', -1), ('a_HCO1', 'a_HCO2', 1), ('a_HCO1', 'a_HCO2', -1),
             ('d_H1', 'd_H2', 1), ('d_H1', 'd_H2', -1)]
    for k, (a, b, s) in enumerate(pairs):
        L[names.index(a), k], L[names.index(b), k] = 1.0, s
    lab = [m['label'] for m in N9.mode_info(K, L, names)]
    assert lab == ['CH2 sym. stretch', 'CH2 asym. stretch', 'CH2 scissor', 'CH2 rock', 'CH2 torsion', 'CH2 wag']
    sym = [m['symmetry'] for m in N9.mode_info(K, L, names)]
    assert sym == ["a'", "a'", "a'", "a'", "a''", "a''"]


# ------------------------------------------------------------------------------------------ fit
def test_symmetric_polynomial_fit_and_heldout_orders():
    """A Cs-even quartic is recovered exactly at order 4; order 2 is worse out of sample."""
    rng = np.random.default_rng(4)
    s = N9._scale(N9.NAMES)
    even = (7, 8)
    pw = np.array(N9.cs_powers(9, 4, even))
    true = rng.normal(0, 0.02, len(pw))
    X = X0 + rng.uniform(-1, 1, (900, 9)) * s * 1.5
    E = N9._predict(X, X0, pw, s, true)
    batch = np.r_[np.zeros(100, int), np.repeat([1, 2, 3, 4], 200)]
    cv = N9.fit_orders(X, E, batch, X0, [2, 4], log=lambda m: None)
    assert cv[4]['heldout_rms_meV'] < 1e-3 and cv[2]['heldout_rms_meV'] > 100 * cv[4]['heldout_rms_meV']
    c = N9._fit(X, E, X0, pw, s, ridge=0)
    Y = X.copy()
    Y[:, 7:] *= -1                                               # the mirror image: the same energies
    assert np.allclose(N9._predict(Y, X0, pw, s, c), E, atol=1e-10)


def test_project_and_lift():
    X = np.array([[1.35, 118.0, 1.27, 1.07, 1.09, 116.0, 120.0, 3.0, -2.0]])
    P = N9.project4(X, ['r_OO', 'a_COO', 'r_CO', 'a_HCO'])
    assert np.allclose(P, [[1.35, 118.0, 1.27, 118.0]])
    Y = N9.lift4(P, ['r_OO', 'a_COO', 'r_CO', 'a_HCO'], {'r_CH': 1.08})
    assert np.allclose(Y, [[1.35, 118.0, 1.27, 1.08, 1.08, 118.0, 118.0, 0.0, 0.0]])


# ------------------------------------------------------------------------------------------ stage, mock ORCA
def _mock_env(tmp_path, monkeypatch, xref):
    P = ch2oo_positions9(np.asarray(xref, float)[None])[0]
    ref = [float(np.linalg.norm(P[a] - P[b])) for a in range(5) for b in range(a + 1, 5)]
    (tmp_path / 'ref.json').write_text(json.dumps(ref))
    (tmp_path / 'empty.json').write_text('{}')
    monkeypatch.setenv('MOCK_ORCA_CC', 'pairs')
    monkeypatch.setenv('MOCK_ORCA_REF', str(tmp_path / 'ref.json'))
    monkeypatch.setenv('MOCK_ORCA_TABLE', str(tmp_path / 'empty.json'))


def test_well9d_stage_end_to_end(tmp_path, monkeypatch, capsys):
    """Hessian (batch 0), two FPS batches, the fit with order selection; verbose detail only with -v."""
    xref = [1.36, 117.5, 1.28, 1.085, 1.082, 118.0, 117.0, 0.0, 0.0]
    _mock_env(tmp_path, monkeypatch, xref)
    d = tmp_path / 'w9'
    W.stage_init(d, 'quick')
    cen = '\n'.join(f'{n} = {v}' for n, v in zip(NINE, X0))
    with open(d / 'config.toml', 'a') as fh:
        fh.write(f'\n[well9d]\nn_pool = 1500\nn_points = 400\nbatches = 2\ne_max_eV = 0.4\nbasis = "aug-cc-pVDZ"\n'
                 f'fit_orders = [2, 3]\nfit_window_eV = 2.0\ntz_points = 100\ntz_e_max_eV = 2.0\ntz_orders = [1, 2]\n'
                 f'[well9d.center]\n{cen}\n')
    orca = str(DATA / 'mock_orca.py')
    W.main(['well9d', str(d), '--orca', orca, '--jobs', '4', '--batch', '0'])
    hz = json.loads((d / 'results' / 'well9d_hessian.json').read_text())
    assert hz['n_points'] == 163 and len(hz['harmonic_cm-1']) == 9
    assert hz['mirror_asymmetry_meV'] < 1e-3                                 # the mock potential is mirror-symmetric
    syms = [m['symmetry'] for m in hz['harmonic_cm-1']]
    assert syms.count("a''") == 2
    z = np.load(d / 'data' / 'well9d.npz')
    assert len(z['coords']) == 163 and set(z['batch'].tolist()) == {0}
    capsys.readouterr()
    W.main(['well9d', str(d), '--orca', orca, '--jobs', '4', '--batch', '1'])
    quiet = capsys.readouterr().out
    W.main(['well9d', str(d), '--orca', orca, '--jobs', '4', '--batch', '2', '-v'])
    loud = capsys.readouterr().out
    assert 'r_CH1' not in quiet.split('batch 1:')[1].split('\n')[1] and '   r_CH1' in loud   # coordinate ranges with -v
    z = np.load(d / 'data' / 'well9d.npz')
    assert len(z['coords']) == 163 + 400 and set(z['batch'].tolist()) == {0, 1, 2}
    log = (d / 'logs' / 'well9d.log').read_text()
    assert log.count('r_CH1 ') >= 2                                          # details are always in the log
    # rerun: nothing recomputed, the design is reused
    W.main(['well9d', str(d), '--orca', orca, '--jobs', '4', '--batch', '1'])
    assert len(np.load(d / 'data' / 'well9d.npz')['coords']) == 563
    W.main(['well9d_fit', str(d), '-v'])
    out = json.loads((d / 'results' / 'well9d.json').read_text())
    assert out['order'] == 3 and set(out['orders']) == {'2', '3'}
    assert out['orders']['3']['heldout_rms_meV'] < out['orders']['2']['heldout_rms_meV']
    # the fitted minimum is the potential's reference geometry
    xm = np.array([out['minimum'][n] for n in NINE])
    assert np.allclose(xm[:7], xref[:7], atol=np.array([2e-3, 0.3, 2e-3, 2e-3, 2e-3, 0.3, 0.3]))
    assert np.abs(xm[7:]).max() < 0.5
    assert len(out['harmonic_cm-1']) == 9 and (d / 'models' / 'well9d.npz').exists()
    from diabml.wellfit import PolyWell
    w = PolyWell.load(d / 'models' / 'well9d.npz')
    assert w.names == list(NINE) and w.meta['order'] == 3
    assert (d / 'figures' / 'W9_well9d.png').exists() and out['delta9'] is None
    # the 9-D basis-set correction: aug-cc-pVTZ at 100 spread points, then the refit uses it
    W.main(['well9d_tz', str(d), '--orca', orca, '--jobs', '4'])
    t = np.load(d / 'data' / 'well9d_tz.npz')
    assert len(t['coords']) == 100 and str(t['basis']) == 'aug-cc-pVTZ'
    W.main(['well9d_fit', str(d)])
    out2 = json.loads((d / 'results' / 'well9d.json').read_text())
    assert out2['delta9']['n'] == 100 and out2['delta9']['order'] == 2                 # the mock difference is quadratic
    assert out2['delta9']['orders']['2']['heldout_rms_meV'] < 0.05 and 'aug-cc-pVTZ' in out2['source']
    # the mock correction shortens the C-H bonds (linear terms in r_CH)
    assert out2['minimum']['r_CH1'] < out['minimum']['r_CH1'] and out2['minimum']['r_CH2'] < out['minimum']['r_CH2']
    # VPT2 on the refitted surface
    W.main(['well9d_vpt2', str(d), '-v'])
    v = json.loads((d / 'results' / 'well9d_vpt2.json').read_text())
    assert v['gradient_max'] < 1e-8 and v['harmonic_gf_vs_cartesian_cm-1'] < 2.0
    assert [m['mode'] for m in v['modes']] == [f'v{k}' for k in range(1, 10)]
    assert sum(m['symmetry'] == "a''" for m in v['modes']) == 2
    assert all(np.isfinite(m['gvpt2']) for m in v['modes']) and v['zpe_cm-1'] < v['zpe_harmonic_cm-1'] + 50
    assert 'polyad' in (d / 'logs' / 'well9d_vpt2.log').read_text() or v['polyads'] == []


def test_well9d_gp_selection_runs_batch_by_batch(tmp_path, monkeypatch):
    xref = [1.36, 117.5, 1.28, 1.085, 1.082, 118.0, 117.0, 0.0, 0.0]
    _mock_env(tmp_path, monkeypatch, xref)
    d = tmp_path / 'gp'
    W.stage_init(d, 'quick')
    cen = '\n'.join(f'{n} = {v}' for n, v in zip(NINE, X0))
    with open(d / 'config.toml', 'a') as fh:
        fh.write(f'\n[well9d]\nn_pool = 500\nn_points = 60\nbatches = 3\ngp_pool = 200\nselect = "gp"\n'
                 f'basis = "aug-cc-pVDZ"\n[well9d.center]\n{cen}\n')
    orca = str(DATA / 'mock_orca.py')
    with pytest.raises(SystemExit):                                          # gp: one batch at a time
        W.main(['well9d', str(d), '--orca', orca, '--jobs', '4'])
    W.main(['well9d', str(d), '--orca', orca, '--jobs', '4', '--batch', '1'])
    with pytest.raises(SystemExit):                                          # batch 3 needs batch 2
        W.main(['well9d', str(d), '--orca', orca, '--jobs', '4', '--batch', '3'])
    W.main(['well9d', str(d), '--orca', orca, '--jobs', '4', '--batch', '2'])
    assert (d / 'data' / 'well9d_gp_batch2.npz').exists()
    z = np.load(d / 'data' / 'well9d.npz')
    assert int((z['batch'] == 2).sum()) == 20 and len(set(map(tuple, np.round(z['coords'], 4)))) == len(z['coords'])
