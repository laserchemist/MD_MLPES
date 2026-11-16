"""VCI (diabml/vci.py) against exact levels and analytic results, and the well9d_vci stage end to end."""
import itertools
import json

import numpy as np
import pytest

from diabml.vci import KM_MOL, VCI, enumerate_basis, line_list, temperature_for_energy, boltzmann, \
    transition_moments
from test_vpt2 import _dvr2, _sym3


def _model2():
    w = np.array([800.0, 1610.0])                       # the Fermi-resonant pair of test_vpt2
    f3 = _sym3(2, {(0, 0, 0): -30, (1, 1, 1): -40, (0, 0, 1): 60})
    f4 = np.array([[6.0, 3.0], [3.0, 8.0]])

    def pot(q):
        v = 0.5 * (w * q ** 2).sum(1)
        for a, b, c in itertools.product(range(2), repeat=3):
            v = v + f3[a, b, c] * q[:, a] * q[:, b] * q[:, c] / 6
        return v + f4[0, 0] * q[:, 0] ** 4 / 24 + f4[1, 1] * q[:, 1] ** 4 / 24 + f4[0, 1] * q[:, 0] ** 2 * q[:, 1] ** 2 / 4
    return w, f3, f4, pot


def test_two_mode_levels_equal_exact_dvr():
    """With two modes the n-mode representation is exact: the variational levels equal a sinc-DVR's, including
    the Fermi dyad that VPT2 gets wrong without the resonance treatment."""
    w, f3, f4, pot = _model2()
    E, _ = VCI(w, pot, ecut=12000, nmax=40, nmode=2).solve(8)
    assert np.allclose(E[:6] - E[0], _dvr2(w, f3, f4)[:6], atol=1e-3)


def test_folding_lower_order_terms_into_clusters_is_exact():
    """Pairwise couplings only: the 2-mode and the (folded) 3-mode representations agree to round-off; with
    three-body terms, 3- and 4-mode agree and 2-mode does not."""
    w = np.array([700., 1000., 1300., 1650.])
    A = np.random.default_rng(0).normal(0, 8, (4, 4))

    def pot2(q):
        v = 0.5 * (w * q ** 2).sum(1) - 0.01 * (w * q ** 3).sum(1) + 0.001 * (w * q ** 4).sum(1)
        for i in range(4):
            for j in range(i + 1, 4):
                v = v + A[i, j] * q[:, i] ** 2 * q[:, j] + 0.5 * A[j, i] * q[:, i] ** 2 * q[:, j] ** 2
        return v

    def pot3(q):
        return pot2(q) + 6.0 * q[:, 0] * q[:, 1] * q[:, 2] + 3.0 * q[:, 1] * q[:, 2] * q[:, 3]
    kw = dict(ecut=7000, nmax=10)
    assert np.allclose(VCI(w, pot2, nmode=2, **kw).solve(10)[0], VCI(w, pot2, nmode=3, **kw).solve(10)[0], atol=1e-8)
    e3 = VCI(w, pot3, nmode=3, **kw).solve(10)[0]
    assert np.allclose(e3, VCI(w, pot3, nmode=4, n4max=10, **kw).solve(10)[0], atol=1e-8)
    assert np.abs(VCI(w, pot3, nmode=2, **kw).solve(10)[0] - e3).max() > 0.01


def test_coriolis_ground_state_shift():
    """Harmonic pair with zeta^c_12: first-order shift of the ground level B zeta^2 (w1 - w2)^2 / (4 w1 w2)."""
    w = np.array([600., 1500.])
    z = np.zeros((3, 2, 2))
    z[2, 0, 1], z[2, 1, 0] = 0.8, -0.8
    B = np.array([0.0, 0.0, 0.5])
    v = VCI(w, lambda q: 0.5 * (w * q ** 2).sum(1), ecut=12000, nmax=20, nmode=2)
    v.build()
    v.add_coriolis(B, z)
    E0 = v.solve(3)[0][0] - w.sum() / 2 + 0.25 * B.sum()
    assert E0 == pytest.approx(0.5 * 0.64 * 900 ** 2 / (4 * 600 * 1500), rel=0.02)


def test_intensity_units_match_harmonic_formula():
    """A harmonic mode with a linear dipole d Q (D per amu^1/2 A): 42.2561 d^2 km/mol."""
    from diabml.vpt2 import C_CM
    w = np.array([600., 1500.])
    g = np.sqrt(w / C_CM)
    d = 1.3
    v = VCI(w, lambda q: 0.5 * (w * q ** 2).sum(1), ecut=5000, nmax=8, nmode=2)
    E, C = v.solve(4)
    M = v.operator(lambda q: np.stack([d * q[:, 0] / g[0], 0 * q[:, 0], 0 * q[:, 0]], 1))
    T = transition_moments(C, M)
    f = v.fundamentals(C)[0][0]
    assert KM_MOL * (E[f] - E[0]) * (T[:, f, 0] ** 2).sum() == pytest.approx(42.2561 * d * d, rel=1e-4)
    lines = line_list(E - E[0], T, boltzmann(E - E[0], 0.0), 'absorption')
    assert len(lines) == 1 and lines[0, 0] == pytest.approx(600.0)


def test_product_basis_and_helpers():
    w = np.array([500., 900., 3000.])
    S = enumerate_basis(w, 2000, 8, separate=w > 2500, sep_quanta=2)
    assert S[:, 2].max() == 2 and ((S[:, :2] @ w[:2]) <= 2000).all()
    assert len(S) == 3 * len(enumerate_basis(w[:2], 2000, 8))                  # every low state x C-H 0, 1, 2
    E = np.array([0.0, 500.0, 900.0, 1000.0, 1400.0, 1500.0])
    T = temperature_for_energy(E, 300.0)
    assert (boltzmann(E, T) * E).sum() == pytest.approx(300.0, rel=1e-6)


def test_vci_stage_end_to_end(tmp_path, monkeypatch):
    """well9d, well9d_fit, well9d_vpt2, well9d_dipole on the mock ORCA, then well9d_vci with a small basis: VCI
    fundamentals near GVPT2, intensities, spectra and figures written."""
    from test_well9d import _mock_env
    from diabml.geometry import NINE
    from diabml.workflow import ch2oo as W
    xref = [1.36, 117.5, 1.28, 1.085, 1.082, 118.0, 117.0, 0.0, 0.0]
    _mock_env(tmp_path, monkeypatch, xref)
    d = tmp_path / 'r'
    W.stage_init(d, 'quick')
    cen = '\n'.join(f'{n} = {v}' for n, v in zip(NINE, xref))
    with open(d / 'config.toml', 'a') as fh:
        fh.write(f'\n[well9d]\nn_pool = 1500\nn_points = 400\nbatches = 1\ne_max_eV = 0.4\nbasis = "aug-cc-pVDZ"\n'
                 f'fit_orders = [3]\nfit_window_eV = 2.0\ndipole_points = 60\ndipole_orders = [1]\n'
                 f'dipole_e_max_eV = 2.0\nvci_ecut = 2600.0\nvci_ch_quanta = 1\nvci_ecut_check = 2000.0\n'
                 f'vci_ch_quanta_check = 1\nvci_states_cm = 1800.0\nvci_extra = 4\nband_max = 12\niso_vci = ["CD2OO"]\niso_ecut = 2000.0\n[well9d.center]\n{cen}\n'
                 f'[ir]\nreference = ""\nsurface = "well9d"\ntemperatures = [300.0]\nenergies_kJmol = [13.0]\ntrajectories = 2\n'
                 f'equil_fs = 50.0\nrun_fs = 200.0\ndt = 0.2\nevery = 2\njobs = 2\n')
    orca = str(__import__('pathlib').Path(__file__).parent / 'data' / 'mock_orca.py')
    W.main(['well9d', str(d), '--orca', orca, '--jobs', '4'])
    W.main(['well9d_fit', str(d)])
    W.main(['well9d_vpt2', str(d)])
    W.main(['well9d_dipole', str(d), '--orca', orca, '--jobs', '4'])
    W.main(['well9d_vci', str(d), '-v'])
    r = json.loads((d / 'results' / 'well9d_vci.json').read_text())
    assert r['basis_size'] > 20 and r['convergence'] is not None and r['zpe_cm-1'] < r['zpe_harmonic_cm-1'] + 50
    low = [f for f in r['fundamentals'] if f['harmonic'] < 1800]
    assert all(abs(f['vci'] - f['gvpt2']) < 25 for f in low)
    assert all(f['km_mol'] >= 0 and abs(sum(f['type'].values()) - 1) < 1e-6 for f in r['fundamentals'])
    sp = r['spectra']
    assert sp['T_vib_K'] > 300 and sp['strongest_absorption'] and sp['strongest_emission'] and sp['oo_band_centre']
    z = np.load(d / 'results' / 'well9d_vci.npz')
    assert z['T'].shape[0] == 3 and z['spectrum_absorption'].shape == z['grid'].shape
    assert (d / 'figures' / 'VCI1_spectra.png').exists() and (d / 'figures' / 'VCI2_oo_band.png').exists()
    rot = r['rotational']
    assert len(rot['B0']) == 3 and rot['B0'][0] > rot['B0'][1] > rot['B0'][2] > 0 and z['B'].shape == (len(z['E']), 3)
    # rotational band shapes from the saved levels
    with open(d / 'config.toml', 'a') as fh:
        fh.write('\n')
    W.main(['well9d_bands', str(d)])
    b = json.loads((d / 'results' / 'well9d_bands.json').read_text())
    assert b['absorption_bands'] > 0
    pa = b['oo']['absorption_PR']
    assert pa[0] is None or pa[0] < b['oo']['origin'] < pa[1]
    assert 'sensitivity' in b['oo'] and (d / 'figures' / 'VCI3_bands.png').exists()
    page = (d / 'vci_9d.html').read_text()                       # the report, written by well9d_bands
    for s in ('Electronic structure', 'Potential energy surface', 'Dipole moment surface', 'Watson', 'asymmetric rotor',
              'References', 'data:image/png;base64'):
        assert s in page
    # isotopologues: microwave constants and shifts from the same surface
    W.main(['well9d_isotopes', str(d)])
    iso = json.loads((d / 'results' / 'well9d_isotopes.json').read_text())
    assert len(iso['species']) == 9
    par, cd2 = iso['species']['CH2OO'], iso['species']['CD2OO']
    assert all(a < b for a, b in zip(cd2['B0_MHz'], par['B0_MHz']))               # heavier: smaller constants
    assert all(abs(x) < 400 for x in par['dB_vib_MHz']) and all(m['overlap'] > 0.98 for m in par['modes'])
    assert max(m['harmonic'] for m in cd2['modes']) < 0.8 * max(m['harmonic'] for m in par['modes'])   # C-D stretch
    assert all('vci' in m for m in cd2['modes']) and 'vci' not in par['modes'][0]
    assert 'Isotopologues: microwave constants' in (d / 'vci_9d.html').read_text()
    assert (d / 'results' / 'well9d_isotopes.npz').exists() and 'CD2OO' in iso.get('spectra', {})
    assert (d / 'figures' / 'VCI4_isotopes.png').exists()
    # classical dynamics against VCI
    W.main(['ir', str(d)])
    W.main(['well9d_compare', str(d)])
    c = json.loads((d / 'results' / 'well9d_compare.json').read_text())
    assert len(c['regions']) == 3 and 'dynamics' in c['regions'][0] and c['md_resolution_cm-1'] > 0
    assert (d / 'figures' / 'VCI5_dynamics.png').exists()
    assert 'Dynamics against VCI' in (d / 'vci_9d.html').read_text()
    # TRPMD on the same surface, then the comparison again with the TRPMD trace
    with open(d / 'config.toml', 'a') as fh:
        fh.write('\n[rpmd]\nbeads = [1, 4]\ntrajectories = 4\nbatch = 2\njobs = 2\nequil_fs = 40.0\nrun_fs = 200.0\n'
                 'lam_scan = [0.0]\ncheck_points = 6\ncheck_controls = 4\n')
    W.main(['rpmd9d', str(d)])
    rp = json.loads((d / 'results' / 'rpmd_9d.json').read_text())
    assert set(rp['beads']) == {'1', '4'} and rp['beads']['4']['V_mean_eV'] > rp['beads']['1']['V_mean_eV']   # zero point
    assert (d / 'figures' / 'RPMD1_spectra.png').exists()
    c = json.loads((d / 'results' / 'well9d_compare.json').read_text())
    assert c.get('trpmd_beads') == 4 and all('trpmd_max' in b for b in c['bands'])   # mock: few bands in range
    assert set(rp['lam_scan']['values']) == {'0', '0.5'} and rp['beads']['4']['ms_per_step_per_trajectory'] > 0
    page = (d / 'dynamics_9d.html').read_text()                  # trajectory diagnostics report
    for s in ('Classical molecular dynamics', 'Thermostatted ring-polymer', 'radius of gyration', 'Friction scan',
              'force evaluations'):
        assert s in page
    for f in ('DYN1_classical.png', 'DYN2_trpmd.png', 'DYN3_quantum_widths.png'):
        assert (d / 'figures' / f).exists()
    # CCSD(T) (mock) at bead geometries outside the fitted range
    assert sum(rp['beads']['4']['outside_by_coordinate'].values()) >= 0
    W.main(['rpmd_check', str(d), '--orca', orca, '--jobs', '4'])
    ck = json.loads((d / 'results' / 'rpmd_check.json').read_text())
    assert ck['outside']['n'] > 0 and ck['inside']['n'] >= 3 and ck['inside']['rms_meV'] < 5 and 'verdict' in ck
    assert (d / 'figures' / 'RPMD2_check.png').exists() and 'Surface check at bead geometries' in (d / 'dynamics_9d.html').read_text()
    # extend the surface: new points from the bead samples and a harmonic sample, aVTZ subset, refit
    nb = len(np.load(d / 'data' / 'well9d.npz')['coords'])
    with open(d / 'config.toml', 'a') as fh:
        fh.write('\n')
    import tomllib
    cfg = (d / 'config.toml').read_text().replace('fit_window_eV = 2.0\n', 'fit_window_eV = 2.0\nextend_points = 30\n'
                                                   'extend_tz_points = 35\nextend_fit_window_eV = 2.5\nextend_pool = 3000\n'
                                                   'fit_weight_eV = 1.0\n', 1)
    (d / 'config.toml').write_text(cfg)
    tomllib.loads(cfg)
    W.main(['well9d_extend', str(d), '--orca', orca, '--jobs', '4'])
    ex = json.loads((d / 'results' / 'well9d_extend.json').read_text())
    z = np.load(d / 'data' / 'well9d.npz')
    assert ex['n_new'] >= 30 and len(z['coords']) == nb + ex['n_new'] and set(ex['batches']) <= set(z['batch'].tolist())
    assert ex['sources'].get('beads', 0) > 0 and ex['sources'].get('rpmd_check', 0) > 0 and ex['old_surface_test']
    assert ex['tz']['n_new'] > 0 and len(np.load(d / 'data' / 'well9d_tz.npz')['coords']) == ex['tz']['n_total']
    assert ex['refit']['window_eV'][1] == 2.5 and ex['refit']['heldout_low_rms_meV'] is not None
    assert (d / 'models' / 'well9d_pre_extend.npz').exists() and (d / 'figures' / 'W9X_extend.png').exists()
    assert json.loads((d / 'results' / 'well9d.json').read_text())['weight_eV'] == 1.0
    W.main(['well9d_extend', str(d), '--orca', orca, '--jobs', '4'])       # rerun: the selection is reused
    assert len(np.load(d / 'data' / 'well9d.npz')['coords']) == len(z['coords'])
