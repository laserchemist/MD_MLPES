"""Vibrational configuration interaction (VCI) on an analytic potential in rectilinear normal coordinates.

    vci = VCI(omega, potential, ecut=9000, nmax=8, nmode=3)      # potential(q (M, n) dimensionless) -> cm-1
    vci.add_coriolis(B, zeta)                                    # optional: Watson vibrational angular momentum
    E, C = vci.solve(n_states)                                   # levels (cm-1, absolute) and eigenvectors
    M = vci.operator(dipole_fn, nmode=2)                         # (3,) sparse matrices of a vector property
    T = transition_moments(C, M)                                 # (3, k, k)

Hamiltonian (Watson, J = 0, effective moments of inertia at equilibrium):
    H = sum_i (omega_i / 2) p_i^2 + V(q) + sum_a B_a pi_a^2 - (1/4) sum_a B_a,
    pi_a = sum_{i != j} zeta^a_ij sqrt(omega_j / omega_i) q_i p_j,
with dimensionless normal coordinates q_i = Q_i sqrt(omega_i / C) (diabml.vpt2), all energies in cm-1.

Basis. Products of harmonic-oscillator functions |n_1 ... n_n> with harmonic excitation energy
sum_i n_i omega_i <= ecut and n_i <= nmax (an energy-truncated basis). Modes marked `separate` (for CH2OO the two
C-H stretches) are truncated apart, by total quanta sep_quanta, and the basis is the product of the two parts.
Reason: in rectilinear coordinates the CH2 bends and torsions couple strongly to the C-H stretches; with one
energy cutoff for everything the ground state gets more C-H configurations than the excited states (whose own
energy uses up the cutoff), which biases the fundamentals by 10-20 cm-1. The product basis gives every low level
the same C-H room. Convergence is checked by repeating with a smaller basis.

Potential: n-mode representation (Bowman, Carter, Huang, Int. Rev. Phys. Chem. 22, 533 (2003)),
    V = sum_i V_i + sum_{i<j} V_ij + sum_{i<j<k} V_ijk [+ sum V_ijkl],
each increment the potential with all other coordinates at zero minus its lower-order parts, evaluated on a
direct-product Gauss-Hermite grid (nmax_i + 1 + extra points per mode) and transformed to the HO basis by
quadrature. With nmode = n (all modes) the representation is exact; nmode = 3 is the usual choice, and 4-mode
terms can be added for low quanta (`n4max`). For two modes the result is the exact variational answer in the
basis (tests/test_vci.py checks it against a DVR).

Kinetic energy in the HO basis is analytic. The vibrational angular momentum term is built as
B_a Pi_a^T Pi_a with Pi_a the real antisymmetric matrix of -i pi_a in the basis (the product truncated at the
basis edge: exact for states well inside it).

Matrix assembly. Every potential increment of order <= K (K = min(nmode, 3)) is folded into the K-mode clusters
that contain it, W_M = sum_{S subset M} V_S / binom(n - |S|, K - |S|), so that H = sum_{|M| = K} W_M; each
cluster's matrix elements fill all pairs of basis states that agree outside M.
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np
import scipy.sparse as sp

# integrated absorption (km/mol) per (cm-1 * Debye^2): 8 pi^3 N_A / (3 h c 4 pi eps0), SI with unit conversions
_NA, _H, _C, _EPS0, _DEBYE = 6.02214076e23, 6.62607015e-34, 2.99792458e8, 8.8541878128e-12, 3.33564095e-30
KM_MOL = 8 * math.pi ** 3 * _NA / (3 * _H * _C * 4 * math.pi * _EPS0) * 100 * _DEBYE ** 2 / 1000   # 2.5066


def ho_functions(G, N):
    """Gauss-Hermite nodes x (G,) and u (N, G): u[n, g] = sqrt(w_g) exp(x_g^2 / 2) psi_n(x_g), psi_n the
    normalised HO functions; sum_g u[n, g] u[m, g] f(x_g) is <n|f|m> by quadrature (exact for polynomial f of
    degree <= 2 (G - N) + 1)."""
    x, w = np.polynomial.hermite.hermgauss(G)
    h = np.zeros((N, G))
    h[0] = np.pi ** -0.25
    if N > 1:
        h[1] = np.sqrt(2.0) * x * h[0]
    for n in range(1, N - 1):
        h[n + 1] = np.sqrt(2.0 / (n + 1)) * x * h[n] - np.sqrt(n / (n + 1)) * h[n - 1]
    return x, h * np.sqrt(w)


def _kinetic(N, omega):
    """(omega / 2) p^2 in the HO basis (N, N)."""
    T = np.zeros((N, N))
    n = np.arange(N)
    T[n, n] = omega * (2 * n + 1) / 4
    m = np.arange(N - 2)
    T[m, m + 2] = T[m + 2, m] = -omega * np.sqrt((m + 1) * (m + 2)) / 4
    return T


def enumerate_basis(omega, ecut, nmax, separate=None, sep_quanta=2):
    """Quanta vectors (Nb, n), sorted by harmonic energy: sum n_i omega_i <= ecut over the modes not in `separate`
    (a boolean mask), times all states of the `separate` modes with at most sep_quanta quanta in total (a product
    basis: every low-mode state gets the same room in the separate modes); n_i <= nmax_i throughout."""
    omega = np.asarray(omega, float)
    n = len(omega)
    nmax = np.broadcast_to(np.asarray(nmax, int), omega.shape)
    sep = np.zeros(n, bool) if separate is None else np.asarray(separate, bool)
    S = np.zeros((1, n), int)
    E = np.zeros(1)
    for i in range(n):
        k = np.arange(nmax[i] + 1)
        Sn = np.repeat(S, len(k), 0)
        Sn[:, i] = np.tile(k, len(S))
        En = np.repeat(E, len(k)) + (0.0 if sep[i] else np.tile(k, len(S)) * omega[i])
        ok = (En <= ecut + 1e-9) & (Sn[:, sep].sum(1) <= sep_quanta)
        S, E = Sn[ok], En[ok]
    Eh = S @ omega
    o = np.lexsort((np.arange(len(Eh)), Eh))
    return S[o]


class VCI:
    def __init__(self, omega, potential, ecut=9000.0, nmax=8, nmode=3, extra=6, n4max=0, separate=None, sep_quanta=2, chunk=200000,
                 log=None):
        self.omega = np.asarray(omega, float)
        if (self.omega <= 0).any():
            raise ValueError('VCI needs real harmonic wavenumbers at the reference')
        self.n = len(self.omega)
        self.potential = potential
        self.ecut = float(ecut)
        self.nmode = int(min(nmode, self.n))
        self.extra = int(extra)
        self.n4max = int(n4max)
        self.chunk = int(chunk)
        self.log = log or (lambda m: None)
        self.S = enumerate_basis(self.omega, self.ecut, nmax, separate, sep_quanta)
        self.N = self.S.max(0) + 1                               # basis functions per mode
        self.G = self.N + self.extra                             # quadrature points per mode
        self.grid = [ho_functions(int(g), int(N)) for g, N in zip(self.G, self.N)]
        self.stride = np.cumprod(np.concatenate([[1], self.N[::-1][:-1]]))[::-1].astype(np.int64)
        self.keys = self.S @ self.stride
        self._order = np.argsort(self.keys)
        self.H = None
        self.diag = {'n_basis': int(len(self.S)), 'quanta_per_mode': self.N.tolist(), 'grid_per_mode': self.G.tolist()}
        self.timing = {}

    # ---------------------------------------------------------------- state lookup
    def index(self, S):
        """Indices of quanta vectors S (M, n) in the basis, -1 where absent."""
        S = np.asarray(S)
        ok = ((S >= 0) & (S < self.N)).all(1)
        k = np.where(ok, np.clip(S, 0, self.N - 1) @ self.stride, -1)
        sk = self.keys[self._order]
        p = np.clip(np.searchsorted(sk, k), 0, len(sk) - 1)
        found = ok & (sk[p] == k)
        return np.where(found, self._order[p], -1)

    # ---------------------------------------------------------------- property on grids
    def _evaluate(self, fn, modes, ncomp):
        """fn on the direct-product grid of `modes` (other coordinates 0): (ncomp, G_i, G_j, ...)."""
        axes = [self.grid[m][0] for m in modes]
        shape = [len(a) for a in axes]
        mesh = np.meshgrid(*axes, indexing='ij')
        npts = int(np.prod(shape))
        q = np.zeros((npts, self.n))
        for m, g in zip(modes, mesh):
            q[:, m] = g.ravel()
        out = np.empty((npts, ncomp))
        for s in range(0, npts, self.chunk):
            v = np.asarray(fn(q[s:s + self.chunk]), float)
            out[s:s + self.chunk] = v.reshape(len(v), ncomp)
        return out.T.reshape([ncomp] + shape)

    def _increments(self, fn, order, ncomp, caps=None):
        """n-mode increments of fn up to `order`: {modes: (ncomp, grid...)}; also the minimum of the full
        function on every cluster grid (for ncomp == 1: holes in the potential)."""
        inc = {(): self._evaluate(fn, (), ncomp).reshape(ncomp)}
        vmin = np.inf
        for k in range(1, order + 1):
            for M in itertools.combinations(range(self.n), k):
                full = self._evaluate(fn, M, ncomp)
                vmin = min(vmin, float(full.min()))
                d = full - inc[()].reshape([ncomp] + [1] * k)
                for s in range(1, k):
                    for Sb in itertools.combinations(range(k), s):
                        shape = [ncomp] + [full.shape[1 + a] if a in Sb else 1 for a in range(k)]
                        d = d - inc[tuple(M[a] for a in Sb)].reshape(shape)
                inc[M] = d
        return inc, vmin

    def _tensor(self, values, modes, N=None):
        """HO-basis matrix elements of grid values (G_i, G_j, ...) of `modes`: (N_i, N_i, N_j, N_j, ...)."""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        ops, subs = [], []
        out = ''
        for a, m in enumerate(modes):
            n_, m_, g_ = letters[3 * a], letters[3 * a + 1], letters[3 * a + 2]
            u = self.grid[m][1] if N is None else self.grid[m][1][:N[a]]
            ops.append(np.einsum('ng,mg->nmg', u, u))
            subs.append(n_ + m_ + g_)
            out += n_ + m_
        gsub = ''.join(letters[3 * a + 2] for a in range(len(modes)))
        return np.einsum(','.join(subs) + ',' + gsub + '->' + out, *ops, values, optimize=True)

    # ---------------------------------------------------------------- assembly
    def _pairs(self, M):
        """All pairs (a, b) of basis states that agree outside the modes M, in chunks of index arrays."""
        kc = self.keys - self.S[:, list(M)] @ self.stride[list(M)]
        o = np.argsort(kc, kind='stable')
        ks = kc[o]
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        sizes = np.diff(np.r_[starts, len(ks)])
        npair = sizes.astype(np.int64) ** 2
        cum = np.cumsum(npair)
        lim = 4_000_000
        g0 = 0
        while g0 < len(starts):
            g1 = int(np.searchsorted(cum, (cum[g0 - 1] if g0 else 0) + lim, side='right'))
            g1 = max(g1, g0 + 1)
            gs = np.arange(g0, g1)
            rep = np.repeat(gs, npair[gs])
            base = np.repeat(np.r_[0, np.cumsum(npair[gs])[:-1]], npair[gs])
            t = np.arange(len(rep)) - base
            sz = sizes[rep]
            yield o[starts[rep] + t // sz], o[starts[rep] + t % sz]
            g0 = g1

    def _assemble(self, tensors, K, cap=None):
        """Sparse matrix sum over clusters M (|M| = K) of tensors[M] (N_i, N_i, ...) placed on all pairs of states
        agreeing outside M; with `cap`, only pairs whose quanta in M are all <= cap."""
        Nb = len(self.S)
        out = sp.csr_matrix((Nb, Nb))
        rows, cols, vals, cnt = [], [], [], 0
        for M, T in tensors.items():
            for a, b in self._pairs(M):
                if cap is not None:
                    ok = (self.S[a][:, list(M)] <= cap).all(1) & (self.S[b][:, list(M)] <= cap).all(1)
                    a, b = a[ok], b[ok]
                idx = []
                for j, m in enumerate(M):
                    idx += [self.S[a, m], self.S[b, m]]
                v = T[tuple(idx)]
                keep = v != 0
                rows.append(a[keep].astype(np.int32))
                cols.append(b[keep].astype(np.int32))
                vals.append(v[keep])
                cnt += int(keep.sum())
                if cnt > 30_000_000:
                    out = out + sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                              shape=(Nb, Nb))
                    rows, cols, vals, cnt = [], [], [], 0
        if vals:
            out = out + sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                      shape=(Nb, Nb))
        return out

    def _folded(self, inc, comp, K, kinetic=False):
        """Fold increments of order <= K (component comp) into the K-mode clusters: {M: tensor}."""
        n = self.n
        small = {}
        for S_, v in inc.items():
            if 0 < len(S_) <= K:
                t = self._tensor(v[comp], S_)
                if kinetic and len(S_) == 1:
                    t = t + _kinetic(self.N[S_[0]], self.omega[S_[0]])
                small[S_] = t / math.comb(n - len(S_), K - len(S_))
        const = float(inc[()][comp])
        out = {}
        for M in itertools.combinations(range(n), K):
            T = np.zeros([int(self.N[m]) for m in M for _ in (0, 1)])
            for s in range(1, K + 1):
                for Sb in itertools.combinations(range(K), s):
                    t = small[tuple(M[a] for a in Sb)]
                    shape, eyes = [], []
                    for a in range(K):
                        if a in Sb:
                            shape += [int(self.N[M[a]])] * 2
                        else:
                            shape += [1, 1]
                    t = t.reshape(shape)
                    for a in range(K):
                        if a not in Sb:
                            e = np.eye(int(self.N[M[a]]))
                            es = [1] * (2 * K)
                            es[2 * a] = es[2 * a + 1] = int(self.N[M[a]])
                            t = t * e.reshape(es)
                    T = T + t
            out[M] = T
        return out, const

    def build(self):
        """The Hamiltonian matrix (cm-1, absolute: the potential's zero is the reference)."""
        t0 = time.time()
        K = min(self.nmode, 3)
        inc, vmin = self._increments(self.potential, K, 1)
        self.diag['potential_min_on_grids_cm-1'] = vmin
        self.timing['potential_s'] = time.time() - t0
        t1 = time.time()
        tens, const = self._folded(inc, 0, K, kinetic=True)
        H = self._assemble(tens, K)
        H = H + const * sp.identity(len(self.S), format='csr')
        if self.nmode >= 4 and self.n >= 4 and self.n4max > 0:
            cap = self.n4max
            t4 = {}
            for M in itertools.combinations(range(self.n), 4):
                full = self._evaluate(self.potential, M, 1)
                d = full - inc[()].reshape([1] * 5)
                for s in range(1, 4):
                    for Sb in itertools.combinations(range(4), s):
                        shape = [1] + [full.shape[1 + a] if a in Sb else 1 for a in range(4)]
                        d = d - inc[tuple(M[a] for a in Sb)].reshape(shape)
                self.diag['potential_min_on_grids_cm-1'] = min(self.diag['potential_min_on_grids_cm-1'],
                                                                float(full.min()))
                t4[M] = self._tensor(d[0], M, N=[min(cap + 1, int(self.N[m])) for m in M])
            H = H + self._assemble(t4, 4, cap=cap)
        self.H = (H + H.T) * 0.5
        self.diag['nnz'] = int(self.H.nnz)
        self.timing['assemble_s'] = time.time() - t1
        return self.H

    def add_coriolis(self, B, zeta):
        """Add sum_a B_a pi_a^2 - (1/4) sum B_a (B (3,) cm-1, zeta (3, n, n) antisymmetric)."""
        if self.H is None:
            self.build()
        Nb = len(self.S)
        w = self.omega
        Hc = sp.csr_matrix((Nb, Nb))
        for a in range(3):
            rows, cols, vals = [], [], []
            for i in range(self.n):
                for j in range(self.n):
                    if i == j or abs(zeta[a, i, j]) < 1e-12:
                        continue
                    c = zeta[a, i, j] * np.sqrt(w[j] / w[i])
                    for di in (1, -1):
                        for dj in (1, -1):
                            T = self.S.copy()
                            T[:, i] += di
                            T[:, j] += dj
                            k = self.index(T)
                            ok = k >= 0
                            ni, nj = self.S[ok, i], self.S[ok, j]
                            qf = np.sqrt((ni + 1) / 2) if di == 1 else np.sqrt(ni / 2)
                            pf = np.sqrt((nj + 1) / 2) if dj == 1 else -np.sqrt(nj / 2)
                            rows.append(k[ok])
                            cols.append(np.flatnonzero(ok))
                            vals.append(c * qf * pf)
            if not vals or B[a] == 0:
                continue
            Pi = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(Nb, Nb))
            Hc = Hc + B[a] * (Pi.T @ Pi)
        self.H = self.H + Hc - 0.25 * float(np.sum(B)) * sp.identity(Nb, format='csr')
        self.diag['coriolis'] = True
        return self.H

    def solve(self, k, tol=1e-9):
        """The k lowest levels (cm-1) and eigenvectors (Nb, k)."""
        if self.H is None:
            self.build()
        t0 = time.time()
        Nb = self.H.shape[0]
        k = int(min(k, Nb - 1))
        if Nb <= 3000:
            E, C = np.linalg.eigh(self.H.toarray())
            E, C = E[:k], C[:, :k]
        else:
            from scipy.sparse.linalg import eigsh
            E, C = eigsh(self.H, k=k, which='SA', tol=tol, ncv=min(Nb - 1, max(2 * k + 1, k + 64)))
            o = np.argsort(E)
            E, C = E[o], C[:, o]
        self.timing['solve_s'] = time.time() - t0
        return E, C

    def operator(self, fn, ncomp=3, nmode=2):
        """Sparse matrices (ncomp,) of a property fn(q) -> (M, ncomp) in its n-mode representation (order
        nmode <= 3, no kinetic term)."""
        K = min(nmode, 3, self.n)
        inc, _ = self._increments(fn, K, ncomp)
        mats = []
        for c in range(ncomp):
            tens, const = self._folded(inc, c, K)
            mats.append(self._assemble(tens, K) + const * sp.identity(len(self.S), format='csr'))
        return mats

    # ---------------------------------------------------------------- analysis
    def assign(self, C, top=3):
        """Per eigenvector: the largest basis components [(quanta tuple, weight), ...]."""
        out = []
        for c in C.T:
            w = c * c
            o = np.argsort(-w)[:top]
            out.append([(tuple(int(x) for x in self.S[j]), float(w[j])) for j in o])
        return out

    def fundamentals(self, C):
        """For each mode i: (eigenstate index with the largest weight of |1_i>, that weight)."""
        res = []
        for i in range(self.n):
            e = np.zeros(self.n, int)
            e[i] = 1
            j = self.index(e[None])[0]
            w = C[j] ** 2
            res.append((int(np.argmax(w)), float(w.max())))
        return res


def transition_moments(C, mats):
    """<psi_f|mu_a|psi_i> (ncomp, k, k) for eigenvectors C (Nb, k)."""
    return np.stack([C.T @ (m @ C) for m in mats])


def boltzmann(E, T):
    """Populations of levels E (cm-1 above the lowest) at T (K)."""
    if T <= 0:
        p = np.zeros(len(E))
        p[0] = 1
        return p
    x = np.exp(-(E - E.min()) / (0.6950348 * T))
    return x / x.sum()


def temperature_for_energy(E, e_mean):
    """The temperature at which the mean energy of levels E (cm-1 above the lowest) is e_mean."""
    lo, hi = 1.0, 1e5
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if (boltzmann(E, mid) * E).sum() < e_mean:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def line_list(E, T, pops, kind='absorption', emin=1.0):
    """Lines (wavenumber, strength, i, f) between levels E (k,) with transition moments T (3, k, k; Debye) and
    populations pops (k,). absorption: strength in km/mol, K nu |mu|^2 (p_i - p_f), from lower i to upper f;
    emission: photon emission rate (arbitrary units) nu^3 |mu|^2 p_i from upper i to lower f."""
    mu2 = (T ** 2).sum(0)                                         # [f, i]
    out = []
    k = len(E)
    for i in range(k):
        for f in range(k):
            nu = E[f] - E[i]
            if nu < emin:
                continue
            if kind == 'absorption':
                s = KM_MOL * nu * mu2[f, i] * (pops[i] - pops[f])
                out.append((nu, s, i, f))
            else:
                s = nu ** 3 * mu2[i, f] * pops[f]                 # upper f -> lower i
                out.append((nu, s, f, i))
    a = np.array(out, float).reshape(-1, 4)
    return a[a[:, 1] > 1e-10 * max(a[:, 1].max(initial=0.0), 1e-300)]


def broaden(lines, grid, fwhm):
    """Gaussian-broadened spectrum of lines (nu, strength, ...) on grid (cm-1)."""
    s = fwhm / 2.3548
    y = np.zeros_like(grid)
    for nu, st in lines[:, :2]:
        y += st * np.exp(-0.5 * ((grid - nu) / s) ** 2)
    return y / (s * np.sqrt(2 * np.pi))
