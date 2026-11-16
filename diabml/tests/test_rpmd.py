"""Ring-polymer dynamics (diabml/dynamics/rpmd.py) on harmonic systems, where the answers are exact."""
import numpy as np
import pytest

from diabml.dynamics.rpmd import FCONV, KB, HBAR, RingPolymer, normal_mode_matrix

C_CM = 2.99792458e-5


def _harmonic(nu_cm, m=1.0):
    w = 2 * np.pi * C_CM * nu_cm                      # fs^-1
    k = m * w ** 2 / FCONV                           # eV / A^2

    def f(X):
        return 0.5 * k * (X ** 2).sum((1, 2)), -k * X
    return f, w


def test_normal_mode_matrix_orthogonal_and_frequencies():
    for P in (1, 2, 5, 8):
        C = normal_mode_matrix(P)
        assert np.allclose(C.T @ C, np.eye(P))
    rp = RingPolymer([1.0], 8, 300.0, None)
    assert rp.wk[0] == 0 and np.isclose(rp.wk[4], 2 * rp.wP)


def test_pimd_potential_energy_matches_exact_finite_P():
    """PIMD (PILE-L, centroid thermostat): bead-averaged potential energy of a 3-D harmonic oscillator against the
    exact finite-P value (omega^2 / 2 P beta_P) sum_k 1 / (omega^2 + omega_k^2) per dimension, and the P -> infinity
    quantum value (hbar omega / 4) coth(beta hbar omega / 2) for large P."""
    f, w = _harmonic(1000.0)
    rp = RingPolymer([1.0], 16, 300.0, f)
    rng = np.random.default_rng(1)
    q, p = rp.initial(np.zeros((1, 3)), 64, rng)
    q, p = rp.run(q, p, 1500, 0.5, gamma0=0.02, lam=1.0, rng=rng)
    acc = []
    for i, (q, p, E) in enumerate(rp.trajectory(q, p, 3000, 0.5, lam=1.0, gamma0=0.02, rng=rng)):
        if i % 10 == 0:
            acc.append(E.mean())
    got = np.mean(acc)
    exact = 3 * (w ** 2 / (2 * rp.P * rp.betaP)) * np.sum(1 / (w ** 2 + rp.wk ** 2)) * 103.6427
    quantum = 3 * HBAR * w / 4 / np.tanh(HBAR * w / (2 * KB * 300.0)) * 103.6427
    assert got == pytest.approx(exact, rel=0.03)
    assert exact == pytest.approx(quantum, rel=0.06)                         # 16 beads: within a few % at 1000 cm-1
    assert got > 2 * 1.5 * KB * 300.0 * 103.6427 / 2                          # well above the classical 3/2 kT


def test_rpmd_conserves_the_ring_polymer_hamiltonian():
    f, w = _harmonic(1500.0)
    rp = RingPolymer([1.0], 8, 300.0, f)
    rng = np.random.default_rng(2)
    q, p = rp.initial(np.zeros((1, 3)), 4, rng)
    q, p = rp.run(q, p, 400, 0.25, gamma0=0.02, lam=1.0, rng=rng)
    E0 = rp.hamiltonian(q, p, rp.forces(q)[0])
    for q, p, E in rp.trajectory(q, p, 2000, 0.25, lam=0.0, gamma0=0.0, rng=rng):
        pass
    E1 = rp.hamiltonian(q, p, E)
    assert np.abs(E1 - E0).max() < 1e-3 * np.abs(E0).max()


@pytest.mark.parametrize('P', [1, 8])
def test_trpmd_spectrum_peaks_at_the_harmonic_frequency(P):
    """For a harmonic oscillator with a linear dipole RPMD and TRPMD are exact: the peak sits at omega."""
    from diabml.dynamics.ir import ir_spectrum
    f, w = _harmonic(1000.0)
    rp = RingPolymer([1.0], P, 300.0, f)
    rng = np.random.default_rng(3)
    q, p = rp.initial(np.zeros((1, 3)), 16, rng)
    q, p = rp.run(q, p, 1000, 0.25, gamma0=0.02, lam=1.0, rng=rng)
    mu = []
    for i, (q, p, E) in enumerate(rp.trajectory(q, p, 16000, 0.25, lam=0.5, rng=rng)):
        if i % 4 == 0:
            mu.append(q.mean(1)[:, 0, :])
    nu, I = ir_spectrum(np.stack(mu, 1), 1.0, max_cm=3000.0, min_cm=200.0)
    pk = nu[(nu > 200)][np.argmax(I[nu > 200])]
    assert pk == pytest.approx(1000.0, abs=3.0)


def test_centroid_temperature_is_thermal():
    f, w = _harmonic(800.0)
    rp = RingPolymer([2.0, 1.0], 8, 400.0, lambda X: (0.5 * 5.0 * (X ** 2).sum((1, 2)), -5.0 * X))
    rng = np.random.default_rng(4)
    q, p = rp.initial(np.zeros((2, 3)), 200, rng)
    q, p = rp.run(q, p, 400, 0.5, gamma0=0.05, lam=1.0, rng=rng)
    T = [rp.centroid_temperature(p).mean() for i, (q, p, E) in
         enumerate(rp.trajectory(q, p, 1000, 0.5, lam=1.0, gamma0=0.05, rng=rng)) if i % 20 == 0]
    assert np.mean(T) == pytest.approx(400.0, rel=0.03)


def test_fast_polynomial_gradient_equals_autograd():
    """poly_energy_grad / energy_forces9 (factorised monomials, matrix products) against autograd through poly_t."""
    import torch
    from test_vpt2 import _model
    from diabml.geometry import ch2oo_positions9
    from diabml.workflow.well9d import energy_forces9, internal9_t, poly_t
    m, x0 = _model(order=5)
    P = ch2oo_positions9(x0[None] + np.random.default_rng(0).normal(0, 1, (40, 9)) * [0.03, 3, 0.03, 0.03, 0.03, 3, 3, 10, 10])
    R = torch.tensor(P, dtype=torch.float64, requires_grad=True)
    E = poly_t(m, internal9_t(R))
    E.sum().backward()
    E2, F2 = energy_forces9(m, P)
    assert np.allclose(E2, E.detach().numpy(), atol=1e-12) and np.allclose(F2, -R.grad.numpy(), atol=1e-10)
