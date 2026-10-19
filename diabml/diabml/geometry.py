"""Geometry helpers for the three systems. Distances in Angstrom, angles in degrees."""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- H3
def h3_xy(r12, r13, r23):
    """Planar coordinates (H1, H2, H3) from the three distances; H2 at the origin, H3 on +x."""
    x = (r12 ** 2 - r13 ** 2 + r23 ** 2) / (2 * r23)
    y = np.sqrt(max(r12 ** 2 - x ** 2, 0.0))
    return np.array([[x, y, 0.0], [0.0, 0.0, 0.0], [r23, 0.0, 0.0]])


def h3_from_bend(r1, r2, theta):
    """(r12, r13, r23) for H1-H2-H3 with H2 in the middle: r12 = r1, r23 = r2, angle theta at H2."""
    t = np.radians(theta)
    return (r1, float(np.sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * np.cos(t))), r2)


H3_PERMUTATIONS = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]   # act on (r12, r13, r23)


# ---------------------------------------------------------------- H2O
def h2o_xyz(r1, r2, theta):
    t = np.radians(theta)
    return np.array([[0, 0, 0], [r1, 0, 0], [r2 * np.cos(t), r2 * np.sin(t), 0]], float)


# ---------------------------------------------------------------- CH2OO (planar, Cs)
CH2OO_REF = dict(r_oo=1.345, a_coo=117.7, r_co=1.272, r_ch=1.080, a_hco=120.0)


def ch2oo_xyz(r_oo=CH2OO_REF['r_oo'], a_coo=CH2OO_REF['a_coo'], r_co=CH2OO_REF['r_co'],
              r_ch=CH2OO_REF['r_ch'], a_hco=CH2OO_REF['a_hco']):
    """Planar CH2OO. Atom order C, O1 (central), O2 (terminal), H, H. O1 at the origin, C on -x.
    Only r_oo and a_coo vary in the default scans; the CH2O fragment is held rigid."""
    C = np.array([-r_co, 0.0, 0.0])
    O1 = np.zeros(3)
    t = np.radians(180.0 - a_coo)
    O2 = np.array([r_oo * np.cos(t), r_oo * np.sin(t), 0.0])
    def h(sign):
        a = np.radians(180.0 - a_hco)
        return C + r_ch * np.array([-np.cos(a), sign * np.sin(a), 0.0])
    return [('C', C), ('O', O1), ('O', O2), ('H', h(+1)), ('H', h(-1))]


def pairwise_distances(X):
    X = np.asarray(X)
    return np.linalg.norm(X[:, None] - X[None], axis=-1)
