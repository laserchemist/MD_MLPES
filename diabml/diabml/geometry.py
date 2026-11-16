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


# ---------------------------------------------------------------- CH2OO, all nine internal coordinates
# Atom order as ch2oo_xyz: C, O1 (central), O2 (terminal), H1 (syn: the same side as O2), H2 (anti).
# O1 at the origin, C on -x, O2 in the xy plane (+y).  The nine coordinates (Angstrom, degrees):
#   r_OO, a_COO (C-O1-O2), r_CO (C-O1), r_CH1, r_CH2, a_HCO1, a_HCO2 (H-C-O1),
#   d_H1  dihedral H1-C-O1-O2              (0 for the planar molecule)
#   d_H2  dihedral H2-C-O1-O2 minus 180    (0 for the planar molecule)
# The mirror plane of the planar molecule (z -> -z) sends (d_H1, d_H2) -> (-d_H1, -d_H2) and leaves the other seven
# unchanged: a Cs-symmetric potential is even in (d_H1, d_H2) jointly.  At planar geometries with r_CH1 = r_CH2 and
# a_HCO1 = a_HCO2 the positions are those of ch2oo_xyz.
NINE = ('r_OO', 'a_COO', 'r_CO', 'r_CH1', 'r_CH2', 'a_HCO1', 'a_HCO2', 'd_H1', 'd_H2')
NINE_OUT_OF_PLANE = ('d_H1', 'd_H2')
NINE_REF = dict(r_OO=1.345, a_COO=117.7, r_CO=1.272, r_CH1=1.080, r_CH2=1.080, a_HCO1=120.0, a_HCO2=120.0,
                d_H1=0.0, d_H2=0.0)


def ch2oo_positions9(X, names=NINE, fixed=None):
    """(N, 5, 3) positions (numpy) from internal coordinates X (N, len(names)); coordinates not in `names` come from
    `fixed` (keys as NINE, case-insensitive) or NINE_REF."""
    X = np.atleast_2d(np.asarray(X, float))
    N = len(X)
    low = {k.lower(): float(v) for k, v in (fixed or {}).items()}
    v = {k: np.full(N, low.get(k.lower(), NINE_REF[k])) for k in NINE}
    for j, n in enumerate(names):
        if n not in v:
            raise KeyError(f'{n} is not one of the nine CH2OO coordinates {NINE}')
        v[n] = X[:, j]
    z = np.zeros(N)
    C = np.stack([-v['r_CO'], z, z], 1)
    O1 = np.zeros((N, 3))
    t = np.radians(180.0 - v['a_COO'])
    O2 = np.stack([v['r_OO'] * np.cos(t), v['r_OO'] * np.sin(t), z], 1)

    def h(r, a, phi):                      # bond r from C, angle a to C->O1 (+x), rotated by phi about the C-O1 axis
        a, phi = np.radians(a), np.radians(phi)
        return C + r[:, None] * np.stack([np.cos(a), np.sin(a) * np.cos(phi), np.sin(a) * np.sin(phi)], 1)
    H1 = h(v['r_CH1'], v['a_HCO1'], v['d_H1'])
    H2 = h(v['r_CH2'], v['a_HCO2'], 180.0 + v['d_H2'])
    return np.stack([C, O1, O2, H1, H2], 1)


def ch2oo_xyz9(x, names=NINE, fixed=None):
    """Atoms list [(symbol, xyz)] for one geometry, as ch2oo_xyz returns (for ORCA inputs)."""
    P = ch2oo_positions9(np.asarray(x, float)[None], names, fixed)[0]
    return list(zip(('C', 'O', 'O', 'H', 'H'), P))


def _angle(a, b, c):
    u, w = a - b, c - b
    cs = np.sum(u * w, -1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1))
    return np.degrees(np.arccos(np.clip(cs, -1.0, 1.0)))


def _dihedral(a, b, c, d):
    b0, b1, b2 = a - b, c - b, d - c
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1n, -1, keepdims=True) * b1n
    w = b2 - np.sum(b2 * b1n, -1, keepdims=True) * b1n
    x = np.sum(v * w, -1)
    y = np.sum(np.cross(b1n, v) * w, -1)
    return np.degrees(np.arctan2(y, x))


def ch2oo_internal9(P):
    """The nine internal coordinates (N, 9), in NINE order, of Cartesian positions P (N, 5, 3) in the atom order
    C, O1, O2, H1, H2 (any orientation).  Inverse of ch2oo_positions9; used to put dynamics frames on the 9-D
    surface.  The sign of the dihedrals follows the frame built by ch2oo_positions9."""
    P = np.asarray(P, float)
    if P.ndim == 2:
        P = P[None]
    C, O1, O2, H1, H2 = (P[:, k] for k in range(5))
    d2 = _dihedral(H2, C, O1, O2)
    out = np.stack([np.linalg.norm(O2 - O1, axis=-1), _angle(C, O1, O2), np.linalg.norm(C - O1, axis=-1),
                    np.linalg.norm(H1 - C, axis=-1), np.linalg.norm(H2 - C, axis=-1),
                    _angle(H1, C, O1), _angle(H2, C, O1),
                    -_dihedral(H1, C, O1, O2), -((d2 % 360.0) - 180.0)], 1)
    return out
