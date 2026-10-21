"""Model files and the dynamics layer: save/load, differentiable coordinates, forces, couplings, invariance, NVE."""
import numpy as np
import pytest
import torch

from diabml import RefSet
from diabml.frames import AsymptoteTable, ChannelFrame, Diabat
from diabml.models import CDNN, FramedModel, load_model, save_model
from diabml.workflow.config import PACKAGE_ROOT

REF = RefSet.load(PACKAGE_ROOT / 'data' / 'ch2oo_casscf.npz')


def framed_prior(seed=1):
    """A framed CH2OO model with random (nonzero) slot weights, so every part of the matrix is exercised."""
    far = np.isclose(REF.coords[:, 0], REF.coords[:, 0].max())
    asym = AsymptoteTable([0.0], np.nanmean(REF.energies[far, :4], 0)[None], extra=[(4, [0, 1, 2], 7.45)])
    c = asym.vals[0]
    w = dict(wall_A=2000.0, wall_rho=0.1)
    fr = ChannelFrame([Diabat('a', c[0], **w), Diabat('b', c[1], **w), Diabat('c', c[2], **w),
                       Diabat('t', c[3], kind='biradical', **w),
                       Diabat('ion', c[:3].mean() + 7.45, charge_product=-1.0, kind='ionic',
                              wall_A=Diabat.born_mayer_balance(1.35), wall_rho=0.3)], 2, asymptote_fn=asym)
    fr.fit_scaler(REF.coords)
    m = FramedModel(fr, 4, hidden=(16,), slot_bias={'coup': 0.1})
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.02 * torch.randn(p.shape, generator=g))
    return m


def test_save_load_roundtrip(tmp_path):
    m = framed_prior()
    c = CDNN(2, 4, n_matrix=5, hidden=(8,)); c.reset(0); c.fit_scaler(REF.coords)
    for model in (m, c):
        path = save_model(model, tmp_path / 'm.pt', meta={'coord_names': ['r_OO', 'a_COO']})
        back = load_model(path)
        assert np.allclose(back.predict(REF.coords), model.predict(REF.coords), atol=1e-12)
        assert back.meta['coord_names'] == ['r_OO', 'a_COO'] and (tmp_path / 'm.json').exists()


def test_matrix_t_matches_matrix():
    m = framed_prior()
    X = REF.coords[::7]
    assert np.allclose(m.matrix(X).detach().numpy(), m.matrix_t(torch.tensor(X)).detach().numpy(), atol=1e-12)


def test_forces_couplings_and_invariance():
    pytest.importorskip('ase')
    from diabml.dynamics import CoordinateSurface, DistanceSurface, DiabmlCalculator, ch2oo_atoms, ch2oo_coordinates
    m = framed_prior()
    surf = CoordinateSurface(m, ch2oo_coordinates(['r_OO', 'a_COO']))
    atoms = ch2oo_atoms(r_oo=1.55, a_coo=112.0)
    atoms.calc = DiabmlCalculator(surf)
    E = atoms.calc.state_energies(atoms).copy()
    F = atoms.calc.state_forces(atoms).copy()
    assert np.allclose(E, m.predict([[1.55, 112.0]])[0], atol=1e-10)
    h, fd = 1e-5, np.zeros_like(F)
    for i in range(5):
        for x in range(3):
            for s in (1, -1):
                a = atoms.copy(); a.positions[i, x] += s * h; a.calc = DiabmlCalculator(surf)
                fd[:, i, x] -= s * a.calc.state_energies(a) / (2 * h)
    assert np.abs(F - fd).max() < 1e-6
    d = atoms.calc.results['couplings']
    assert np.abs(d + d.transpose(1, 0, 2, 3)).max() < 1e-10
    # full-dimensional distances surface with H-swap averaging: exactly invariant
    cd = CDNN(10, 4, n_matrix=5, hidden=(8,)); cd.reset(2); cd.fit_scaler(np.random.default_rng(0).random((40, 10)) + 1)
    ds = DistanceSurface(cd, permutations=[[0, 1, 2, 3, 4], [0, 1, 2, 4, 3]])
    A = ch2oo_atoms(); A.positions += np.random.default_rng(1).normal(0, 0.05, (5, 3)); A.calc = DiabmlCalculator(ds)
    B = A.copy(); B.positions[[3, 4]] = B.positions[[4, 3]]; B.calc = DiabmlCalculator(ds)
    assert np.allclose(A.calc.state_energies(A), B.calc.state_energies(B), atol=1e-12)


def test_nve_energy_conservation():
    pytest.importorskip('ase')
    from ase import units
    from ase.md.verlet import VelocityVerlet
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from diabml.dynamics import CoordinateSurface, DiabmlCalculator, ch2oo_atoms, ch2oo_coordinates, reduced_restraints
    m = framed_prior()
    atoms = ch2oo_atoms()
    surf = reduced_restraints(CoordinateSurface(m, ch2oo_coordinates(['r_OO', 'a_COO'])), atoms.positions,
                              free=['r_OO', 'a_COO'])
    atoms.calc = DiabmlCalculator(surf, couplings=False)
    MaxwellBoltzmannDistribution(atoms, temperature_K=1000, rng=np.random.default_rng(0))
    dyn, E = VelocityVerlet(atoms, timestep=0.2 * units.fs), []
    dyn.attach(lambda: E.append(atoms.get_total_energy()), interval=5)
    dyn.run(300)
    assert np.ptp(E) < 2e-3          # eV


def test_no_hole_below_the_data():
    """With Born-Mayer walls the framed model's ground state rises below the shortest reference distance."""
    m = framed_prior()
    x = np.linspace(0.5, 1.2, 50)
    E0 = m.predict(np.c_[x, np.full_like(x, 117.7)])[:, 0]
    assert E0.min() > np.nanmin(REF.energies[:, 0]) - 0.5 and E0[0] > 5.0
