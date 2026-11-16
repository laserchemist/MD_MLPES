"""Asymmetric-rotor levels, line strengths and band contours (diabml/rotor.py), and the rotational constants of
vibrational levels."""
import numpy as np
import pytest

from diabml.rotor import band_lines, contour, rotor_levels, strengths

ABC = (2.5934, 0.41579, 0.35763)                    # CH2OO ground state (cm-1)


def test_rotor_limits():
    """Prolate symmetric top: E = B J(J+1) + (A - B) K^2; the asymmetric top's 0_00, 1_01, 1_11, 1_10 levels:
    B + C, A + C, A + B."""
    E, _ = rotor_levels(3, 2.0, 0.5, 0.5)
    want = sorted(0.5 * 12 + 1.5 * k * k for k in range(-3, 4))
    assert np.allclose(E, want)
    E1, _ = rotor_levels(1, *ABC)
    A, B, C = ABC
    assert np.allclose(sorted(E1), sorted([B + C, A + C, A + B]))
    assert rotor_levels(0, *ABC)[0][0] == 0.0


@pytest.mark.parametrize('comp', ['a', 'b', 'c'])
def test_sum_rule(comp):
    """sum over all upper levels of S(J'n' <- Jn) = 2J + 1 for each lower level (unit moment)."""
    J = 7
    El, Vl = rotor_levels(J, *ABC)
    tot = np.zeros(2 * J + 1)
    for Jp in (J - 1, J, J + 1):
        Eu, Vu = rotor_levels(Jp, *ABC)
        tot += strengths(J, Jp, Vl, Vu, comp).sum(0)
    assert np.allclose(tot, 2 * J + 1)


def test_honl_london_parallel_and_perpendicular():
    """Symmetric top (B = C): a-type R(J, K) = ((J+1)^2 - K^2) / (J+1), Q = (2J+1) K^2 / (J(J+1)); b-type
    K -> K+1, R branch: (J+K+1)(J+K+2) / (2(J+1)) (each summed over the +-K pair, here per |K| state)."""
    J = 5
    A, B = 2.0, 0.5
    El, Vl = rotor_levels(J, A, B, B)
    Kl = np.abs(np.arange(-J, J + 1))[np.argmax(np.abs(Vl), 0)]
    for Jp, f in ((J + 1, lambda K: ((J + 1) ** 2 - K ** 2) / (J + 1)), (J, lambda K: (2 * J + 1) * K ** 2 / (J * (J + 1)))):
        Eu, Vu = rotor_levels(Jp, A, B, B)
        S = strengths(J, Jp, Vl, Vu, 'a')
        for K in range(1, J + 1):
            got = S[:, Kl == K].sum() / 2                   # two degenerate lower states (+K, -K)
            assert got == pytest.approx(f(K), rel=1e-9)
    Eu, Vu = rotor_levels(J + 1, A, B, B)
    S = strengths(J, J + 1, Vl, Vu, 'b')
    Ku = np.abs(np.arange(-J - 1, J + 2))[np.argmax(np.abs(Vu), 0)]
    K = 2
    got = S[np.ix_(Ku == K + 1, Kl == K)].sum() / 2
    assert got == pytest.approx((J + K + 1) * (J + K + 2) / (2 * (J + 1)) / 2, rel=1e-9)


def test_band_contour_shapes():
    """a-type band of a near-prolate top: P and R maxima either side of a sharp Q branch, separated by about
    2 sqrt(2 B kT) (the classical P-R spacing of a parallel band, (B + C)/2 for B; the K structure widens it);
    total intensity ~ nu0; emission ~ nu0^3."""
    lines = band_lines(900.0, ABC, ABC, 'a', 300.0)
    grid = np.arange(700.0, 1100.0, 0.05)
    y = contour([lines], grid, 1.0)
    assert np.trapezoid(y, grid) == pytest.approx(900.0, rel=0.01)
    left, right = grid < 899, grid > 901
    pP = grid[left][np.argmax(y[left])]
    pR = grid[right][np.argmax(y[right])]
    Bm = (ABC[1] + ABC[2]) / 2
    assert pR - pP == pytest.approx(2 * np.sqrt(2 * Bm * 0.695 * 300), rel=0.2)
    assert y[np.abs(grid - 900) < 0.5].max() > y[left].max()          # sharp Q branch
    lc = band_lines(900.0, ABC, ABC, 'c', 300.0)
    yc = contour([lc], grid, 1.0)
    assert np.trapezoid(yc, grid) == pytest.approx(900.0, rel=0.01)
    em = band_lines(900.0, ABC, ABC, 'a', 300.0, kind='emission')
    ye = contour([em], grid, 1.0)
    assert np.trapezoid(ye, grid) == pytest.approx(900.0 ** 3, rel=0.01)
