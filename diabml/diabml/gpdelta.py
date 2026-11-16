"""A Gaussian-process model for a smooth correction (Delta) over internal coordinates, with its uncertainty.

    gp = GPDelta.fit(X, y)                  # X (N, d) internal coordinates (A, degrees), y (N,) in eV
    mean, std = gp.predict(Xnew)            # eV; std is the posterior standard deviation (interpolation error)
    gp.save('models/delta_gp.npz'); GPDelta.load(...)

Kernel: Matern 5/2 with one length scale per coordinate (automatic relevance), a signal variance and a noise
variance; zero prior mean, so the prediction falls back to zero (no correction) away from the data, which is the
behaviour wanted for a Delta that must vanish where it is not known. Hyperparameters maximise the log marginal
likelihood (L-BFGS on their logarithms), on at most `max_opt` points.

The standard deviation is the model's interpolation uncertainty given its kernel: calibrate it on held-out points
(shell_check does) before trusting it, and remember it knows nothing about the error of the reference method.
"""
from __future__ import annotations

import numpy as np

_DEFAULT_LS = {'r_OO': 0.15, 'a_COO': 15.0, 'r_CO': 0.1, 'r_CH': 0.05, 'a_HCO': 10.0}


def _matern52(A, B, ls):
    r2 = np.zeros((len(A), len(B)))
    for j in range(A.shape[1]):                               # (M, N) memory only
        r2 += ((A[:, j:j + 1] - B[None, :, j]) / ls[j]) ** 2
    d = np.sqrt(r2) * np.sqrt(5.0)
    return (1 + d + d ** 2 / 3) * np.exp(-d)


class GPDelta:
    def __init__(self, names, X, alpha, L, ls, sf2, sn2, ymean=0.0, meta=None):
        self.names, self.X, self.alpha, self.L = list(names), np.asarray(X, float), alpha, L
        self.ls, self.sf2, self.sn2, self.ymean = np.asarray(ls, float), float(sf2), float(sn2), float(ymean)
        self.meta = dict(meta or {})

    # ------------------------------------------------------------------ fit
    @staticmethod
    def _nll(theta, X, y):
        d = X.shape[1]
        ls, sf2, sn2 = np.exp(theta[:d]), np.exp(theta[d]), np.exp(theta[d + 1])
        K = sf2 * _matern52(X, X, ls) + (sn2 + 1e-12) * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e25
        a = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return float(0.5 * y @ a + np.log(np.diag(L)).sum())

    @classmethod
    def fit(cls, X, y, names, noise_eV=1e-4, max_opt=1200, seed=0, optimise=True, init_ls=None):
        from scipy.optimize import minimize
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
        X, y = X[ok], y[ok]
        ls0 = np.array([float((init_ls or {}).get(n, _DEFAULT_LS.get(n, 1.0))) for n in names])
        sf2 = max(float(np.var(y)), 1e-8)
        theta = np.log(np.r_[ls0, sf2, noise_eV ** 2])
        if optimise and len(X) > 5:
            idx = np.random.default_rng(seed).choice(len(X), min(len(X), max_opt), replace=False)
            lo = np.log(np.r_[ls0 / 20, sf2 / 100, (noise_eV / 10) ** 2])
            hi = np.log(np.r_[ls0 * 20, sf2 * 100, (100 * noise_eV) ** 2 + 1e-12])
            r = minimize(cls._nll, theta, args=(X[idx], y[idx]), method='L-BFGS-B', bounds=list(zip(lo, hi)))
            theta = r.x
        d = X.shape[1]
        ls, sf2, sn2 = np.exp(theta[:d]), np.exp(theta[d]), np.exp(theta[d + 1])
        K = sf2 * _matern52(X, X, ls) + (sn2 + 1e-12) * np.eye(len(X))
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return cls(names, X, alpha, L, ls, sf2, sn2,
                   meta={'n': int(len(X)), 'length_scales': dict(zip(names, ls.tolist())),
                         'signal_std_meV': float(np.sqrt(sf2) * 1000), 'noise_std_meV': float(np.sqrt(sn2) * 1000)})

    # ------------------------------------------------------------------ predict
    def predict(self, Xn, return_std=True, chunk=4000):
        Xn = np.atleast_2d(np.asarray(Xn, float))
        mu, sd = [], []
        for a in range(0, len(Xn), chunk):
            Ks = self.sf2 * _matern52(Xn[a:a + chunk], self.X, self.ls)
            mu.append(Ks @ self.alpha)
            if return_std:
                v = np.linalg.solve(self.L, Ks.T)
                sd.append(np.sqrt(np.maximum(self.sf2 - (v ** 2).sum(0), 0.0)))
        mu = np.concatenate(mu)
        return (mu, np.concatenate(sd)) if return_std else mu

    def posterior_cov_to(self, C, P):
        """Posterior covariance between candidates C (M, d) and points P (k, d): (M, k)."""
        kcp = self.sf2 * _matern52(C, P, self.ls)
        vc = np.linalg.solve(self.L, (self.sf2 * _matern52(C, self.X, self.ls)).T)
        vp = np.linalg.solve(self.L, (self.sf2 * _matern52(P, self.X, self.ls)).T)
        return kcp - vc.T @ vp

    def sample(self, Xn, n_samples=20, seed=0, jitter=1e-10):
        """Posterior function samples at Xn (n_samples, len(Xn)): for error bars on derived quantities."""
        mu = self.predict(Xn, return_std=False)
        C = self.posterior_cov_to(Xn, Xn)
        Lc = np.linalg.cholesky(C + jitter * np.eye(len(Xn)) * max(self.sf2, 1e-12))
        z = np.random.default_rng(seed).standard_normal((n_samples, len(Xn)))
        return mu[None] + z @ Lc.T

    # ------------------------------------------------------------------ io
    def save(self, path):
        import json
        np.savez(path, names=np.array(self.names), X=self.X, alpha=self.alpha, L=self.L, ls=self.ls,
                 sf2=self.sf2, sn2=self.sn2, ymean=self.ymean, meta=json.dumps(self.meta))
        return path

    @classmethod
    def load(cls, path):
        import json
        z = np.load(path, allow_pickle=False)
        return cls([str(n) for n in z['names']], z['X'], z['alpha'], z['L'], z['ls'], float(z['sf2']),
                   float(z['sn2']), float(z['ymean']), json.loads(str(z['meta'])))


def select_uncertain(gp, C, n, min_std=0.0):
    """Greedy batch selection: the candidate with the largest posterior std, then condition on it (as if observed)
    and repeat, so a batch does not pile up in one region. Returns indices into C."""
    C = np.asarray(C, float)
    _, sd = gp.predict(C)
    var = sd ** 2
    picks, V = [], []
    for _ in range(min(n, len(C))):
        k = int(np.argmax(var))
        if var[k] <= min_std ** 2:
            break
        picks.append(k)
        c0 = gp.posterior_cov_to(C, C[k:k + 1])[:, 0]
        for v in V:                                           # condition on the earlier picks of this batch
            c0 = c0 - v * v[k]
        v = c0 / np.sqrt(max(var[k], 1e-30))
        V.append(v)
        var = np.maximum(var - v ** 2, 0.0)
        var[picks] = 0.0
    return np.array(picks, int)
