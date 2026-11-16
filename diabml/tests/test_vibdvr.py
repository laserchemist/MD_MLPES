"""Grid vibrational levels: on a quadratic surface the DVR fundamentals equal the G-F harmonic frequencies."""
import numpy as np
import torch

from diabml.vibdvr import _harmonic, g_matrix, vib_levels


class _Quad:
    names = ['r_OO', 'a_COO', 'r_CO']

    def __init__(self):
        self.x0 = np.array([1.35, 118.0, 1.275])
        self.K = np.array([[30.0, 0.0, 2.0], [0.0, 3.0e-3, 0.0], [2.0, 0.0, 45.0]])   # eV per A^2 / deg^2
        self.center = torch.tensor(self.x0)
        self.meta = {'coord_names': self.names, 'fixed_coordinates': {}}

    def predict(self, X):
        d = np.atleast_2d(X) - self.x0
        return (0.5 * np.einsum('ni,ij,nj->n', d, self.K, d))[:, None]


def test_quadratic_levels_match_harmonic():
    s = _Quad()
    out = vib_levels(s, None, n=22, n_states=6, ecut_eV=1.2, log=lambda *a: None)
    nu = np.array(out['harmonic_cm-1'])
    E = [st['cm-1'] for st in out['states']]
    fund = sorted(st['cm-1'] for st in out['states'] if sum(st['quanta']) == 1)
    assert np.allclose(fund, np.sort(nu), atol=0.5)        # G(q) varies over the grid: small deviation allowed
    assert out['harmonic_grid_error_cm-1'] < 0.1
    assert abs(E[0]) < 1e-9


def test_g_matrix_symmetric_positive():
    G = g_matrix(np.array([[1.35, 118.0, 1.275], [1.25, 110.0, 1.35]]), ['r_OO', 'a_COO', 'r_CO'], {})
    assert np.allclose(G, np.swapaxes(G, 1, 2))
    assert (np.linalg.eigvalsh(G) > 0).all()
    assert abs(G[0, 0, 0] - (1 / 15.999 + 1 / 15.999)) < 0.05      # O-O stretch ~ 1/mu_OO, slightly coupled to C


class _Quad4(_Quad):
    names = ['r_OO', 'a_COO', 'r_CO', 'a_HCO']

    def __init__(self):
        super().__init__()
        self.x0 = np.array([1.35, 118.0, 1.275, 120.0])
        K = np.zeros((4, 4))
        K[:3, :3] = _Quad().K
        K[3, 3] = 4e-3
        K[2, 3] = K[3, 2] = 0.05
        self.K = K
        self.center = torch.tensor(self.x0)
        self.meta = {'coord_names': self.names, 'fixed_coordinates': {'r_CH': 1.08}}


def test_quadratic_levels_4d():
    out = vib_levels(_Quad4(), None, n=14, n_states=6, ecut_eV=1.2, log=lambda *a: None)
    fund = sorted(st['cm-1'] for st in out['states'] if sum(st['quanta']) == 1)
    assert len(fund) == 4 and 'HCO' in out['harmonic_labels'] or 'C-O' in out['harmonic_labels']
    assert np.allclose(fund, np.sort(out['harmonic_cm-1']), atol=2.0)


def test_rotational_constants_rigid_and_averaged():
    """A, B, C of CH2OO near its structure (expt. 2.59 / 0.416 / 0.358 cm-1); a state localised at the minimum gives
    the equilibrium constants; vib_levels reports them per state."""
    from diabml.geometry import ch2oo_xyz
    from diabml.vibdvr import inertia_abc, rotational_constants
    P = np.array([[p for _, p in ch2oo_xyz(r_oo=1.345, a_coo=117.4, r_co=1.272, r_ch=1.08, a_hco=118.0)]])
    A, B, C = inertia_abc(P)[0]
    assert A > B > C and abs(A - 2.59) < 0.05 and abs(B - 0.416) < 0.01 and abs(C - 0.358) < 0.01
    X = np.array([[1.345, 117.4, 1.272], [1.40, 117.4, 1.272]])
    U = np.array([[1.0], [0.0]])
    r = rotational_constants(X, ['r_OO', 'a_COO', 'r_CO'], {'r_CH': 1.08, 'a_HCO': 118.0}, U, X[0], n_keep=1)
    assert np.allclose(r['states'][0], [A, B, C]) and np.allclose(r['equilibrium'], [A, B, C])
    out = vib_levels(_Quad(), None, n=18, n_states=4, ecut_eV=1.2, log=lambda *a: None)
    assert all(len(s['ABC_cm-1']) == 3 for s in out['states']) and len(out['rotational_cm-1']['equilibrium']) == 3


def test_coriolis_planar_model():
    """Planar model: only zeta^c is non-zero, antisymmetric, with sum_s zeta_rs^2 <= 1; states carry the shift."""
    out = vib_levels(_Quad(), None, n=18, n_states=4, ecut_eV=1.2, log=lambda *a: None)
    c = out['rotational_cm-1']['coriolis']
    z = np.array(c['zeta'])
    assert np.abs(z[:2]).max() < 1e-10 and np.allclose(z[2], -z[2].T) and (np.sum(z[2] ** 2, 1) <= 1 + 1e-9).all()
    s0 = out['states'][0]
    assert np.allclose(np.array(s0['ABC_with_coriolis_cm-1']) - np.array(s0['ABC_cm-1']),
                       -0.5 * np.array(c['alpha_cm-1']).sum(0))
