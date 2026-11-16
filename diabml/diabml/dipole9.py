"""A dipole surface for CH2OO in all nine internal coordinates, exact under rotation and reflection.

    m = VectorDipole9.fit(P, mu, orders=[1, 2, 3])     # P (N, 5, 3) positions (A), mu (N, 3) dipoles (D), same frame
    m.predict(P)                                       # (N, 3) dipoles (D) in the frame of P
    m.save('models/dipole9d.npz'); VectorDipole9.load(...)

Form. mu(P) = sum_j f_j(q) b_j(P) with the four bond vectors b = (O1 - C, O2 - O1, H1 - C, H2 - C) and scalar
coefficients f_j (D/A) that are polynomials in the nine internal coordinates q (diabml.geometry.NINE), even in the
two out-of-plane dihedrals jointly. The f_j are invariant and the b_j rotate with the molecule, so mu rotates with
it exactly; a reflection through the molecular plane reflects the b_j and leaves the f_j unchanged, so mu reflects
as a polar vector should (no out-of-plane dipole for a planar geometry, an antisymmetric one when the H atoms leave
the plane). Unlike point charges on the atoms, the form also carries atomic polarisation along the bonds.
The four vectors are linearly dependent (three dimensions); a small ridge term picks the smallest coefficients.
Each polynomial order in `orders` is scored by k-fold held-out error and the best is kept.
"""
from __future__ import annotations

import json

import numpy as np

from .geometry import NINE, NINE_OUT_OF_PLANE, ch2oo_internal9

BONDS = ((0, 1), (1, 2), (0, 3), (0, 4))          # (from, to) atoms in the order C, O1, O2, H1, H2
_SCALE = {'r': 0.05, 'a': 5.0, 'd': 15.0}


def _kind(n):
    return 'd' if n.startswith('d_') else 'a' if n.startswith('a_') else 'r'


def bond_vectors(P):
    P = np.asarray(P, float)
    return np.stack([P[:, b] - P[:, a] for a, b in BONDS], 1)          # (N, 4, 3)


class VectorDipole9:
    def __init__(self, center, scale, pw, coef, meta=None):
        self.center = np.asarray(center, float)
        self.scale = np.asarray(scale, float)
        self.pw = np.asarray(pw, int)
        self.coef = np.asarray(coef, float)                              # (4, T)
        self.meta = dict(meta or {})

    # ---------------------------------------------------------------- evaluation
    def _mono(self, X):
        from .workflow.well9d import design_matrix
        return design_matrix((np.atleast_2d(X) - self.center) / self.scale, self.pw)

    def coefficients(self, P):
        return self._mono(ch2oo_internal9(P)) @ self.coef.T              # (N, 4)

    def predict(self, P):
        """Dipoles (N, 3) in Debye, in the frame of the positions P (N, 5, 3) or (5, 3)."""
        P = np.asarray(P, float)
        one = P.ndim == 2
        P = P[None] if one else P
        mu = np.einsum('nj,nja->na', self.coefficients(P), bond_vectors(P))
        return mu[0] if one else mu

    # ---------------------------------------------------------------- fit
    @staticmethod
    def _solve(M, B, mu, ridge):
        N, T = M.shape
        A = np.einsum('nt,nja->najt', M, B).reshape(N * 3, 4 * T)   # rows (n, axis), cols (j, t)
        s = np.linalg.norm(A, axis=0)
        s[s == 0] = 1.0
        An = A / s
        c = np.linalg.solve(An.T @ An + ridge * np.eye(An.shape[1]), An.T @ mu.reshape(-1))
        return (c / s).reshape(4, T)

    @classmethod
    def fit(cls, P, mu, orders=(1, 2, 3), center=None, folds=5, ridge=1e-8, log=None):
        """Fit to dipoles mu (N, 3; Debye) at positions P (N, 5, 3) in the same frame. Returns the model of the
        order with the lowest `folds`-fold held-out RMS (per component, D); meta holds the scores."""
        from .workflow.well9d import cs_powers, design_matrix
        P, mu = np.asarray(P, float), np.asarray(mu, float)
        X = ch2oo_internal9(P)
        c0 = X.mean(0) if center is None else np.asarray(center, float)
        c0[[NINE.index(n) for n in NINE_OUT_OF_PLANE]] = 0.0
        sc = np.array([_SCALE[_kind(n)] for n in NINE])
        even = [NINE.index(n) for n in NINE_OUT_OF_PLANE]
        B = bond_vectors(P)
        k = np.arange(len(P)) % folds
        scores = {}
        for order in orders:
            pw = np.array(cs_powers(9, int(order), even))
            if 4 * len(pw) > 0.6 * 3 * len(P):
                (log or (lambda m: None))(f'dipole order {order}: {4 * len(pw)} coefficients for {3 * len(P)} '
                                          f'components, too many: skipped')
                continue
            M = design_matrix((X - c0) / sc, pw)
            err = []
            for f in range(folds):
                tr, te = k != f, k == f
                c = cls._solve(M[tr], B[tr], mu[tr], ridge)
                err.append(np.einsum('nj,nja->na', M[te] @ c.T, B[te]) - mu[te])
            e = np.concatenate(err)
            scores[int(order)] = {'coefficients': int(4 * len(pw)), 'heldout_rms_D': float(np.sqrt(np.mean(e ** 2))),
                                  'heldout_max_D': float(np.abs(e).max())}
        if not scores:
            raise ValueError('no dipole order could be fitted: more points needed')
        best = min(scores, key=lambda o: scores[o]['heldout_rms_D'])
        pw = np.array(cs_powers(9, best, even))
        M = design_matrix((X - c0) / sc, pw)
        coef = cls._solve(M, B, mu, ridge)
        res = np.einsum('nj,nja->na', M @ coef.T, B) - mu
        meta = {'order': best, 'orders': scores, 'n_points': int(len(P)),
                'residual_rms_D': float(np.sqrt(np.mean(res ** 2))), 'box': {n: [float(X[:, j].min()), float(X[:, j].max())]
                                                                          for j, n in enumerate(NINE)}}
        return cls(c0, sc, pw, coef, meta)

    # ---------------------------------------------------------------- io
    def save(self, path):
        np.savez(path, kind='vector9', center=self.center, scale=self.scale, pw=self.pw, coef=self.coef,
                 meta=json.dumps(self.meta))
        return path

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        if str(z['kind']) != 'vector9':
            raise ValueError(f'{path} is not a 9-D vector dipole model')
        return cls(z['center'], z['scale'], z['pw'], z['coef'], json.loads(str(z['meta'])))


def is_vector9(path):
    try:
        return str(path).endswith('.npz') and str(np.load(path, allow_pickle=False)['kind']) == 'vector9'
    except Exception:
        return False
