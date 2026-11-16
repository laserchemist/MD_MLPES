"""Dipole surface (charge model, frames), IR spectrum from a dipole time series, finite-field dipoles."""
import numpy as np
import pytest
import torch

from diabml.dipole import DipoleModel, body_axes, ch2oo_positions, fit_dipole_model, lab_dipole
from diabml.dynamics.ir import find_peaks, ir_spectrum
from diabml.geometry import ch2oo_xyz


def test_positions_match_the_geometry_builder():
    X = np.array([[1.35, 117.7, 1.272], [1.9, 100.0, 1.4]])
    P = ch2oo_positions(X, ['r_OO', 'a_COO', 'r_CO'], {'r_CH': 1.08, 'a_HCO': 120.0}).numpy()
    for x, p in zip(X, P):
        ref = np.array([v for _, v in ch2oo_xyz(r_oo=x[0], a_coo=x[1], r_co=x[2])])
        assert np.allclose(p, ref)
    ax = body_axes(P)
    assert np.allclose(ax, np.eye(3)[None].repeat(2, 0), atol=1e-12)     # the builder frame is the body frame


def test_lab_dipole_rotates_with_the_molecule():
    X = np.array([[1.4, 115.0, 1.3]])
    P = ch2oo_positions(X, ['r_OO', 'a_COO', 'r_CO']).numpy()[0]
    th = 0.7
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(0.4), 0, np.sin(0.4)], [0, 1, 0], [-np.sin(0.4), 0, np.cos(0.4)]])
    Q = (Ry @ Rz @ P.T).T + np.array([3.0, -1.0, 2.0])
    mb = np.array([[0.5, -0.2, 0.0]])
    assert np.allclose(lab_dipole(mb, Q[None])[0], Ry @ Rz @ mb[0])


def test_fit_recovers_a_charge_model():
    rng = np.random.default_rng(0)
    names = ['r_OO', 'a_COO', 'r_CO']
    X = np.c_[rng.uniform(1.2, 2.5, 300), rng.uniform(100, 140, 300), rng.uniform(1.2, 1.45, 300)]
    P = ch2oo_positions(X, names).numpy()
    zw = 1 / (1 + np.exp(4 * (X[:, 0] - 1.8)))                           # zwitterion weight, smooth in r_OO
    q = np.c_[0.1 + 0.2 * zw, 0.1 * zw, -0.3 * zw - 0.2, 0.05 * np.ones(300)]
    q = np.c_[q, -q.sum(1)]
    mu = np.einsum('na,nax->nx', q, P)
    m, val = fit_dipole_model(X[:250], mu[:250], names, n_members=2, epochs=1500)
    err = np.sqrt(np.mean(np.sum((m.body_dipole(X[250:]) - mu[250:]) ** 2, 1))) * 4.8032
    assert err < 0.1                                                     # Debye
    assert np.allclose(m.charges(X[:5]).sum(2).detach().numpy(), 0)    # neutral


def test_save_load_roundtrip(tmp_path):
    m = DipoleModel(['r_OO', 'a_COO'], {'r_CO': 1.272}, n_members=2, mean=[1.5, 115.0], std=[0.3, 10.0])
    m.meta = {'level': 'nevpt2'}
    m.save(tmp_path / 'd.pt')
    m2 = DipoleModel.load(tmp_path / 'd.pt')
    X = np.array([[1.4, 117.0], [2.0, 110.0]])
    assert np.allclose(m.body_dipole(X), m2.body_dipole(X)) and m2.meta['level'] == 'nevpt2'


def test_spectrum_finds_the_frequencies():
    dt = 0.4
    t = np.arange(20000) * dt                                            # 8 ps
    c = 2.99792458e-5
    mu = np.c_[0.3 * np.cos(2 * np.pi * 900 * c * t), 0.1 * np.cos(2 * np.pi * 1400 * c * t + 1.0), 0 * t]
    nu, I = ir_spectrum(mu, dt, broaden_cm=4.0)
    pk = [p for p, a in find_peaks(nu, I)]
    assert len(pk) == 2 and abs(pk[0] - 900) < 5 and abs(pk[1] - 1400) < 5


def test_finite_field_dipole_of_water():
    pytest.importorskip('pyscf')
    from diabml.refs.finite_field import state_dipoles
    atoms = [('O', (0.0, 0.0, 0.0)), ('H', (0.757, 0.586, 0.0)), ('H', (-0.757, 0.586, 0.0))]
    spec = {'basis': 'sto-3g', 'ncas': 4, 'nelecas': 4, 'nroots': 2, 'symmetry': 'Cs', 'wfnsym': "A'"}
    r = state_dipoles(atoms, spec, F=0.002)
    # the average of the finite-field CASSCF dipoles is the analytic dipole of the state-averaged density
    assert np.allclose(r['mu_cas'].mean(0)[:2], r['mu_sa'][:2], atol=2e-4)
    assert abs(r['mu_cas'][0, 0]) < 1e-4 and r['mu_cas'][0, 1] > 0.3           # along the C2 axis (+y)
    assert np.all(np.isfinite(r['mu_pt']))


def test_well_polynomial_recovers_a_smooth_surface(tmp_path):
    from diabml import RefSet
    from diabml.wellfit import PolyWell, fit_well
    rng = np.random.default_rng(0)
    X = np.c_[rng.uniform(1.2, 1.5, 400), rng.uniform(105, 130, 400), rng.uniform(1.18, 1.38, 400)]
    q = (X - [1.35, 118.0, 1.28]) / [0.1, 10.0, 0.05]
    E0 = -2.2 + 0.2 * q[:, 0] ** 2 + 0.05 * q[:, 1] ** 2 + 0.15 * q[:, 2] ** 2 + 0.03 * q[:, 0] * q[:, 2] + 0.01 * q[:, 0] ** 3
    ref = RefSet('t', X, np.c_[E0 + rng.normal(0, 0.01, 400), E0 + 4], ['r_OO', 'a_COO', 'r_CO'])
    m, info = fit_well(ref, order=3, window_eV=2.0)
    assert info['residual_rms_meV'] < 12 and info['n_points'] > 100
    m.save(tmp_path / 'w.npz')
    m2 = PolyWell.load(tmp_path / 'w.npz')
    Xt = np.array([[1.35, 118.0, 1.28], [1.40, 112.0, 1.30]])
    true = -2.2 + np.array([0.0, 0.2 * 0.25 + 0.05 * 0.36 + 0.15 * 0.16 + 0.03 * 0.2 + 0.01 * 0.125])
    assert np.allclose(m2.predict(Xt)[:, 0], true, atol=0.01) and m2.inside(Xt).all()


def test_restraints_free_symmetric_hco():
    """With a_HCO in the model, the restraints leave the symmetric H-C-O angle free and hold only the difference."""
    import torch
    from diabml.dynamics.ch2oo import ch2oo_atoms, ch2oo_coordinates, reduced_restraints, RestrainedSurface
    atoms = ch2oo_atoms(a_hco=117.0)
    R0 = torch.tensor(atoms.positions)

    class Flat:
        n_states = 1

        def matrix(self, R):
            return torch.zeros((1, 1), dtype=R.dtype)
    s = reduced_restraints(Flat(), atoms.positions, free=['r_OO', 'a_COO', 'r_CO', 'a_HCO'])
    sym = torch.tensor(ch2oo_atoms(a_hco=121.0).positions)                      # both angles +4 deg
    assert float(s.restraint_energy(sym)) < 1e-10
    asym = R0.clone()
    asym[3] = sym[3]                                                             # one H only
    assert float(s.restraint_energy(asym)) > 1e-3
    fx = ch2oo_coordinates(['a_HCO'])
    assert abs(float(fx(sym)[0]) - 121.0) < 1e-6
    s3 = reduced_restraints(Flat(), atoms.positions, free=['r_OO', 'a_COO', 'r_CO'])
    assert float(s3.restraint_energy(sym)) > 1e-3                                # 3-D: the scissor is restrained
