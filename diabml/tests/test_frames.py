"""Physics guarantees of the frames, checked with random (untrained) slot networks, plus model plumbing."""
from pathlib import Path

import numpy as np
import pytest
import torch

from diabml.refset import RefSet
from diabml.frames import LondonFrame, ChannelFrame, Diabat, eigvals_sorted
from diabml.geometry import H3_PERMUTATIONS
from diabml.models import FramedModel, CDNN, train

DATA = Path(__file__).resolve().parents[1] / 'data'


def randomise(model, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return model


@pytest.fixture(scope='module')
def london():
    return LondonFrame(RefSet.load(DATA / 'h2_curves.npz'))


def test_london_degenerate_at_d3h(london):
    m = randomise(FramedModel(london, 2))
    E = m.predict([[r, r, r] for r in (0.8, 1.08, 1.5, 2.5)])
    assert np.allclose(E[:, 0], E[:, 1], atol=2e-6)


def test_london_permutation_invariant(london):
    m = randomise(FramedModel(london, 2))
    R = np.random.default_rng(0).uniform(0.7, 3.0, (20, 3))
    E0 = m.predict(R)
    for p in H3_PERMUTATIONS:
        assert np.allclose(m.predict(R[:, p]), E0, atol=1e-10)


def test_london_h2_plus_h_limit(london):
    m = randomise(FramedModel(london, 2))
    h2 = RefSet.load(DATA / 'h2_curves.npz')
    r = 0.74
    E = m.predict([[r, 30.0, 30.0]])
    S, T = london.S(r), london.T(r)                 # the frame's own fragment curves (cubic splines)
    assert abs(E[0, 0] - S) < 1e-5 and abs(E[0, 1] - T) < 1e-5     # 1e-6 eV regulariser in the 2x2 formula


def test_untrained_framed_model_equals_prior(london):
    R = np.random.default_rng(1).uniform(0.7, 3.0, (10, 3))
    m = FramedModel(london, 2)
    assert np.allclose(m.predict(R), london.prior_energies(R))


def test_channel_frame_asymptotes():
    d = [Diabat('cov', 0.0), Diabat('cov2', 0.01), Diabat('ion', 7.0, charge_product=-1.0, kind='ionic')]
    f = ChannelFrame(d, n_coords=2).fit_scaler(np.array([[1.3, 110.0], [3.0, 130.0]]))
    m = randomise(FramedModel(f, 3))
    M = m.matrix([[12.0, 117.0]]).detach().numpy()[0]
    off = M - np.diag(np.diag(M))
    assert np.abs(off).max() < 1e-6
    assert np.allclose(np.diag(M)[:2], [0.0, 0.01], atol=1e-6)
    assert abs(M[2, 2] - (7.0 - 14.399645 / np.sqrt(144 + 0.25))) < 1e-6


def test_eigvals_2x2_matches_eigh():
    A = torch.randn(50, 2, 2); A = A + A.transpose(1, 2)
    assert torch.allclose(eigvals_sorted(A), torch.linalg.eigvalsh(A), atol=1e-6)


def test_cdnn_trains_on_toy_avoided_crossing():
    x = np.linspace(-1, 1, 60)[:, None]
    E = np.stack([-np.sqrt(x[:, 0] ** 2 + 0.01), np.sqrt(x[:, 0] ** 2 + 0.01)], 1)
    m = train(CDNN(1, 2, hidden=(32, 32)), x, E, epochs=1500, lr=3e-3, restarts=1)
    assert np.abs(m.predict(x) - E).mean() < 0.02


def test_refset_roundtrip(tmp_path):
    r = RefSet('t', np.arange(6.).reshape(3, 2), np.ones((3, 2)), ['a', 'b'], {'q': np.arange(3)}, {'zero': 'x'})
    r.save(tmp_path / 't.npz')
    s = RefSet.load(tmp_path / 't.npz')
    assert np.allclose(s.coords, r.coords) and s.meta['zero'] == 'x' and np.allclose(s.extra['q'], [0, 1, 2])
