"""The 9-D dipole surface (diabml/dipole9.py), the finite-field stage (well9d_dipole) and 9-D IR dynamics."""
import json
from pathlib import Path

import numpy as np
import pytest

from diabml.dipole9 import VectorDipole9, bond_vectors, is_vector9
from diabml.geometry import NINE, ch2oo_internal9, ch2oo_positions9

DATA = Path(__file__).parent / 'data'
X0 = np.array([1.35, 117.7, 1.274, 1.083, 1.080, 118.7, 114.8, 0.0, 0.0])
SPREAD = np.array([0.04, 4, 0.04, 0.04, 0.04, 4, 4, 15, 15])


def _charges_dipole(P):
    """Point charges that depend on the bond lengths (a.u. charges, D): a neutral molecule's dipole."""
    q = ch2oo_internal9(P)
    c = np.stack([0.2 + 0.5 * (q[:, 2] - 1.27), 0.25 - 0.4 * (q[:, 0] - 1.35), np.zeros(len(q)),
                  0.05 + 0.3 * (q[:, 3] - 1.08), 0.05 + 0.3 * (q[:, 4] - 1.08)], 1)
    c[:, 2] = -(c[:, 0] + c[:, 1] + c[:, 3] + c[:, 4])
    return 4.80321 * np.einsum('na,nab->nb', c, P)


def _mock_dipole(P):
    """The dipole tests/data/mock_orca.py builds into its finite-field energies (D)."""
    q = ch2oo_internal9(P)
    c = np.stack([0.2 + 0.3 * (q[:, 2] - 1.27), 0.25 - 0.4 * (q[:, 0] - 1.35), np.zeros(len(q)),
                  0.05 + 0.2 * (q[:, 3] - 1.08), 0.05 + 0.2 * (q[:, 4] - 1.08)], 1)
    c[:, 2] = -(c[:, 0] + c[:, 1] + c[:, 3] + c[:, 4])
    return 1.8897261246 * 2.541746 * np.einsum('na,nab->nb', c, P)


def _sample(n, seed=0):
    rng = np.random.default_rng(seed)
    return ch2oo_positions9(X0 + rng.normal(0, 1, (n, 9)) * SPREAD)


def test_vector_dipole_recovers_bond_dependent_charges():
    P = _sample(300)
    m = VectorDipole9.fit(P, _charges_dipole(P), orders=[1, 2])
    assert m.meta['order'] == 1 and m.meta['orders'][1]['heldout_rms_D'] < 1e-5
    Pt = _sample(50, seed=1)
    assert np.abs(m.predict(Pt) - _charges_dipole(Pt)).max() < 1e-4


def test_vector_dipole_is_covariant_and_mirror_correct(tmp_path):
    P = _sample(300)
    mu = _charges_dipole(P) + 0.3 * np.einsum('n,na->na', ch2oo_internal9(P)[:, 7] ** 2 / 100,
                                              bond_vectors(P)[:, 2])          # some non-charge-like polarisation
    m = VectorDipole9.fit(P, mu, orders=[2])
    Pt = _sample(20, seed=2)
    th = 0.9
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]) @ \
        np.array([[1, 0, 0], [0, 0.6, -0.8], [0, 0.8, 0.6]])
    assert np.abs(m.predict(Pt @ R.T + 3.0) - m.predict(Pt) @ R.T).max() < 1e-10       # rotation and translation
    assert np.abs(m.predict(Pt * [1, 1, -1]) - m.predict(Pt) * [1, 1, -1]).max() < 1e-10   # reflection in the plane
    Xp = ch2oo_internal9(Pt)
    Xp[:, 7:] = 0
    assert np.abs(m.predict(ch2oo_positions9(Xp))[:, 2]).max() < 1e-12                 # planar: no z dipole
    m.save(tmp_path / 'd.npz')
    assert is_vector9(tmp_path / 'd.npz')
    m2 = VectorDipole9.load(tmp_path / 'd.npz')
    assert np.allclose(m2.predict(Pt), m.predict(Pt)) and m2.meta['order'] == 2


def test_field_convention_finds_sign_and_nuclear_term():
    from diabml.workflow.well9d import BOHR, _Z, field_convention
    P = _sample(40)
    mu = _charges_dipole(P) / 2.541746                                              # a.u.
    nuc = np.einsum('a,nab->nb', _Z, P * BOHR)
    d = -(mu - nuc)                              # electronic part only, opposite sign
    conv, got, rms, rms2 = field_convention(d, 1.05 * mu, P)
    assert conv == 'sign -1, + nuclear (input origin)' and np.allclose(got, mu) and rms2 > 20 * rms


def test_ff_input_has_the_field_and_no_symmetry():
    from diabml.geometry import ch2oo_xyz9
    from diabml.workflow.well9d import ff_input
    t = ff_input(ch2oo_xyz9(X0), 'aug-cc-pVDZ', 2000, [0.0, 0.001, 0.0])
    assert 'EField 0.000000, 0.001000, 0.000000' in t and 'UseSym' not in t and 'CCSD(T) aug-cc-pVDZ' in t
    assert t.index('EField') < t.index('* xyz 0 1')


def test_dipole_stage_and_9d_ir_end_to_end(tmp_path, monkeypatch):
    """well9d (Hessian + one batch), well9d_fit, well9d_dipole with the mock's field convention, then the ir stage
    on the 9-D well: microcanonical and thermal trajectories, spectra and figures."""
    from test_well9d import _mock_env
    from diabml.workflow import ch2oo as W
    xref = [1.36, 117.5, 1.28, 1.085, 1.082, 118.0, 117.0, 0.0, 0.0]
    _mock_env(tmp_path, monkeypatch, xref)
    d = tmp_path / 'r'
    W.stage_init(d, 'quick')
    cen = '\n'.join(f'{n} = {v}' for n, v in zip(NINE, xref))
    with open(d / 'config.toml', 'a') as fh:
        fh.write(f'\n[well9d]\nn_pool = 1500\nn_points = 400\nbatches = 1\ne_max_eV = 0.4\nbasis = "aug-cc-pVDZ"\n'
                 f'fit_orders = [3]\nfit_window_eV = 2.0\ndipole_points = 60\ndipole_orders = [1, 2]\n'
                 f'dipole_e_max_eV = 2.0\n[well9d.center]\n{cen}\n'
                 f'[ir]\nsurface = "well9d"\ntemperatures = [300.0]\nenergies_kJmol = [13.0]\ntrajectories = 2\n'
                 f'equil_fs = 50.0\nrun_fs = 200.0\ndt = 0.2\nevery = 2\njobs = 2\nreference = ""\n')
    orca = str(DATA / 'mock_orca.py')
    W.main(['well9d', str(d), '--orca', orca, '--jobs', '4'])
    W.main(['well9d_fit', str(d)])
    W.main(['well9d_dipole', str(d), '--orca', orca, '--jobs', '4', '-v'])
    r = json.loads((d / 'results' / 'well9d_dipole.json').read_text())
    assert r['convention'] == 'sign -1, + nuclear (input origin)' and r['n'] == 60
    assert r['fit']['order'] == 1 and r['fit']['orders']['1']['heldout_rms_D'] < 1e-4
    z = np.load(d / 'data' / 'well9d_dipoles.npz')
    assert np.allclose(z['mu_D'], _mock_dipole(z['positions']), atol=1e-4)       # the mock's true dipoles
    assert 'harmonic intensities' in (d / 'logs' / 'well9d_dipole.log').read_text()
    W.main(['ir', str(d)])
    s = json.loads((d / 'results' / 'ir_9d.json').read_text())
    assert set(s['temperatures']) == {'300 K', '13 kJ/mol'}
    for T, v in s['temperatures'].items():
        assert v['trajectories'] == 2 and max(abs(x) for x in v['drift_meV']) < 5 and v['peaks_cm-1']
    assert len(s['harmonic']) == 9 and not any(h['spring'] for h in s['harmonic'])
    assert (d / 'ir_9d.html').exists() and (d / 'figures' / 'IR3_zoom_9d.png').exists()
    page = (d / 'ir_9d.html').read_text()
    assert 'restraint spring (not physical)' not in page and 'held by restraint springs' not in page
    assert 'all nine internal coordinates' in page
    assert np.load(d / 'results' / 'ir_9d_13kJpermol.npz')['x'].shape[-1] == 9
