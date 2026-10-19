"""Covalent DIM frame for H2O: projector algebra, exact limits, agreement with the exploratory notebook."""
from pathlib import Path

import numpy as np
import pytest

from diabml.refset import RefSet
from diabml.units import EV_KCAL

DATA = Path(__file__).resolve().parents[1] / 'data'


@pytest.fixture(scope='module')
def frame():
    from diabml.frames.dim_oh import DIMWaterFrame
    return DIMWaterFrame(RefSet.load(DATA / 'oh_curves.npz'), RefSet.load(DATA / 'h2_631g.npz'))


def test_projector_ranks_and_completeness():
    from diabml.frames.dim_oh import oh_projectors, P_SA, _embA
    P, _ = oh_projectors(np.array([1.0, 0, 0]), P_SA, _embA)
    expected = {'2S-': 4, '2P1': 8, '4S-': 8, '4P': 16, '2S+1': 4, '2P2': 8, '2D': 8, '2S+2': 4}
    for k, Pk in P.items():
        assert int(round(np.trace(Pk))) == expected[k]
    assert np.allclose(sum(P.values()), np.eye(60))


def test_limits(frame):
    oh = RefSet.load(DATA / 'oh_curves.npz')
    # fragment curves end at 8 A, where they sit within 1 meV of the true separated-atom limit
    E_far = frame.prior_energies([[40.0, 40.0, 104.5]])[0, 0]
    assert abs(E_far) < 1e-3                                   # O(3P) + 2H
    E1 = frame.prior_energies([[0.97, 40.0, 104.5]])[0, 0]
    assert abs(E1 - (frame.oh[1](0.97) + frame.oh[1](40.0) + frame.hh[1](40.0))) < 1e-6   # OH X2Pi + H exactly


def test_equilibrium_value_matches_notebook(frame):
    E = frame.prior_energies([[0.96, 0.96, 104.5]])[0, 0] * EV_KCAL
    assert abs(E - (-133.33)) < 0.05
