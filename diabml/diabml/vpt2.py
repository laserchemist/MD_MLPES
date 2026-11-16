"""Second-order vibrational perturbation theory (VPT2 / GVPT2) on an analytic potential.

    modes = cartesian_modes(hess, P0, masses)                 # rectilinear normal modes, Eckart-projected
    f3, f4 = force_constants(hess_fn, P0, masses, modes)      # cubic, semi-diagonal quartic (dimensionless, cm-1)
    zeta, B = coriolis(P0, masses, modes)                     # Coriolis zeta in the principal-axis frame
    res = find_resonances(modes.omega, f3)                    # Fermi resonances (Martin test)
    out = gvpt2(modes.omega, f3, f4, B, zeta, res)            # anharmonic constants, fundamentals, polyads

Conventions. Normal coordinates Q (amu^1/2 A) diagonalise the mass-weighted Cartesian Hessian after translations
and rotations are projected out; dimensionless q_i = Q_i sqrt(omega_i / C) with C = hbar^2/(amu A^2) = 33.715 cm-1.
The potential is V = sum_i omega_i q_i^2 / 2 + (1/6) sum f3_ijk q_i q_j q_k + (1/24) sum f4_ijkl q_i q_j q_k q_l,
force constants in cm-1. VPT2 needs the cubic constants and the semi-diagonal quartic ones f4_iijj (f4[i, j]).

Anharmonic constants (Mills; Barone, J. Chem. Phys. 122, 014108 (2005)), with the resonant denominators written
as partial fractions so that a resonance can be removed term by term (deperturbed VPT2):
    chi_ii = (1/16) [f_iiii - sum_k f_iik^2 T1(i, k)],
        T1 = 2/w_k + 1/(2(2w_i + w_k)) - 1/(2(2w_i - w_k))
    chi_ij = (1/4) [f_iijj - sum_k f_iik f_jjk / w_k] + (1/2) sum_k f_ijk^2 T2(i, j, k)
             + sum_a B_a (zeta^a_ij)^2 (w_i/w_j + w_j/w_i),
        T2 = w_k (w_k^2 - w_i^2 - w_j^2) / Delta_ijk,
        Delta_ijk = (w_i+w_j+w_k)(-w_i+w_j+w_k)(w_i-w_j+w_k)(w_i+w_j-w_k)
           = (1/4) [-1/(w_i+w_j+w_k) + 1/(w_i+w_j-w_k) - 1/(w_i-w_j+w_k) + 1/(w_i-w_j-w_k)]
(the signs are checked against exact variational levels of two-mode models in tests/test_vpt2.py).
Levels E(v) = sum w_i (v_i + 1/2) + sum_{i<=j} chi_ij (v_i + 1/2)(v_j + 1/2); fundamentals
nu_i = w_i + 2 chi_ii + (1/2) sum_{j != i} chi_ij.

Resonances. A Fermi resonance (a; b, c) has w_a ~ w_b + w_c (b = c for an overtone). It is kept when
|w_b + w_c - w_a| <= max_gap and W^4 / |D|^3 >= martin (Martin et al., J. Phys. Chem. 99, 17049 (1995)), with
the coupling W = <1_a|H|1_b 1_c> = f_abc / (2 sqrt 2), or <1_a|H|2_b> = f_abb / 4. Its terms are dropped from chi
(deperturbed), and the fundamental and its partners are diagonalised together (GVPT2): each polyad's eigenstates
are reported with their composition.

Not included: Darling-Dennison (2-2) resonances, vibration-rotation constants alpha, intensities.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .vibdvr import B_CONV, EV_CM, HBAR2

C_CM = HBAR2 * EV_CM                              # hbar^2 / (amu A^2) in cm-1 (33.715)


# ------------------------------------------------------------------------------------------ normal modes
@dataclass
class Modes:
    omega: np.ndarray        # (n,) harmonic wavenumbers, cm-1 (negative = imaginary)
    L: np.ndarray            # (3N, n) orthonormal mass-weighted Cartesian displacement vectors
    masses: np.ndarray       # (N,) amu
    P0: np.ndarray           # (N, 3) reference positions (A)

    def cart(self, Q):
        """Cartesian displacements (N, 3) of normal-coordinate displacements Q (n,) in amu^1/2 A."""
        m = np.repeat(self.masses, 3)
        return (self.L @ np.asarray(Q, float) / np.sqrt(m)).reshape(-1, 3)

    def hess_q(self, H):
        """A Cartesian Hessian (3N, 3N; eV/A^2) in normal coordinates (n, n; eV/(amu A^2))."""
        m = np.repeat(self.masses, 3)
        Hm = H / np.sqrt(np.outer(m, m))
        return self.L.T @ Hm @ self.L


def _external(P, masses):
    """Orthonormal mass-weighted translation and rotation vectors (3N, 6 or 5) at P."""
    m = np.asarray(masses, float)
    Pc = P - (m[:, None] * P).sum(0) / m.sum()
    sm = np.repeat(np.sqrt(m), 3)
    V = []
    for a in range(3):
        t = np.zeros_like(Pc)
        t[:, a] = 1.0
        V.append(t.ravel() * sm)
    for a in range(3):
        e = np.zeros(3)
        e[a] = 1.0
        V.append(np.cross(e, Pc).ravel() * sm)
    V = np.array(V).T
    U, s, _ = np.linalg.svd(V, full_matrices=False)
    return U[:, s > 1e-8 * s.max()]


def cartesian_modes(H, P0, masses):
    """Harmonic normal modes from the Cartesian Hessian H (3N, 3N; eV/A^2) at P0 (N, 3): Eckart-projected,
    ascending wavenumber."""
    m = np.repeat(np.asarray(masses, float), 3)
    Hm = H / np.sqrt(np.outer(m, m))
    E = _external(P0, masses)
    Pr = np.eye(len(m)) - E @ E.T
    Hp = Pr @ Hm @ Pr
    lam, U = np.linalg.eigh(Hp)
    # drop the external (zero) directions: the eigenvectors with the largest overlap with E
    ext = np.argsort(-np.linalg.norm(E.T @ U, axis=0))[:E.shape[1]]
    keep = np.setdiff1d(np.arange(len(lam)), ext)
    lam, U = lam[keep], U[:, keep]
    o = np.argsort(lam)
    lam, U = lam[o], U[:, o]
    omega = np.sign(lam) * np.sqrt(np.abs(lam) * HBAR2) * EV_CM
    return Modes(omega, U, np.asarray(masses, float), np.asarray(P0, float))


# ------------------------------------------------------------------------------------------ force constants
def force_constants(hess_fn, modes, step=0.02):
    """Cubic f3 (n, n, n) and semi-diagonal quartic f4 (n, n; f4[i, j] = f_iijj) in cm-1, dimensionless
    normal coordinates, from Hessians at +-step (dimensionless) along every normal coordinate:
    f_ijk = dH_ij/dq_k, f_iikk = d^2 H_ii/dq_k^2 (central differences). hess_fn(P) returns the Cartesian
    Hessian (3N, 3N) in eV/A^2 at positions P (N, 3). f3 is averaged over index permutations; the asymmetry
    before averaging is returned as a check (cm-1)."""
    w = modes.omega
    n = len(w)
    g = np.sqrt(np.abs(w) / C_CM)                         # q = g Q
    H0 = modes.hess_q(hess_fn(modes.P0)) * EV_CM          # cm-1 / (amu A^2)
    d3 = np.zeros((n, n, n))
    f4 = np.zeros((n, n))
    for k in range(n):
        dQ = step / g[k]
        Q = np.zeros(n)
        Q[k] = dQ
        Hp = modes.hess_q(hess_fn(modes.P0 + modes.cart(Q))) * EV_CM
        Hm = modes.hess_q(hess_fn(modes.P0 - modes.cart(Q))) * EV_CM
        d3[:, :, k] = (Hp - Hm) / (2 * dQ)
        f4[:, k] = np.diag(Hp - 2 * H0 + Hm) / dQ ** 2
    perms = [d3.transpose(p) for p in itertools.permutations(range(3))]
    asym = float(max(np.abs(p - d3).max() for p in perms) / max(np.abs(d3).max(), 1e-30))
    d3 = sum(perms) / 6
    f4 = 0.5 * (f4 + f4.T)
    f3 = d3 / (g[:, None, None] * g[None, :, None] * g[None, None, :])
    f4 = f4 / (g[:, None] ** 2 * g[None, :] ** 2)
    return f3, f4, asym


# ------------------------------------------------------------------------------------------ rotation
def principal_axes(P, masses):
    """Principal moments (3,; amu A^2, ascending: a, b, c) and axes (3, 3; columns)."""
    m = np.asarray(masses, float)
    Pc = P - (m[:, None] * P).sum(0) / m.sum()
    I = np.zeros((3, 3))
    for mi, r in zip(m, Pc):
        I += mi * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    Iv, R = np.linalg.eigh(I)
    return Iv, R


def coriolis(modes):
    """zeta (3, n, n) about the a, b, c principal axes and B (3,) rotational constants (cm-1) at the reference.
    zeta^a_ij = sum_atoms (l_i x l_j)_a with l the mass-weighted displacement of each atom, in the principal frame."""
    Iv, R = principal_axes(modes.P0, modes.masses)
    n = modes.L.shape[1]
    l = modes.L.T.reshape(n, -1, 3) @ R                   # (n, N, 3) in the principal frame
    zeta = np.zeros((3, n, n))
    for a, (b, c) in enumerate(((1, 2), (2, 0), (0, 1))):
        zeta[a] = np.einsum('in,jn->ij', l[:, :, b], l[:, :, c]) - np.einsum('in,jn->ij', l[:, :, c], l[:, :, b])
    return zeta, B_CONV / Iv


# ------------------------------------------------------------------------------------------ VPT2
def coupling(f3, a, b, c):
    """<1_a|H|1_b 1_c> = f_abc / (2 sqrt 2), or <1_a|H|2_b> = f_abb / 4 (cm-1)."""
    return f3[a, b, c] / 4.0 if b == c else f3[a, b, c] / (2 * np.sqrt(2))


def find_resonances(w, f3, max_gap=200.0, martin=1.0):
    """Fermi resonances (a, b, c), b <= c, b and c != a: w_a ~ w_b + w_c with |D| <= max_gap and
    W^4/|D|^3 >= martin. Returns a list of dicts (a, b, c, gap D = w_b + w_c - w_a, W, Martin value)."""
    n = len(w)
    out = []
    for a in range(n):
        for b in range(n):
            for c in range(b, n):
                if a in (b, c):
                    continue
                D = w[b] + w[c] - w[a]
                if abs(D) > max_gap:
                    continue
                W = coupling(f3, a, b, c)
                mv = W ** 4 / max(abs(D), 1e-6) ** 3
                if mv >= martin:
                    out.append({'a': a, 'b': b, 'c': c, 'gap': float(D), 'W': float(W), 'martin': float(mv)})
    return out


def chi_matrix(w, f3, f4, B=None, zeta=None, res=()):
    """Anharmonic constants chi (n, n; symmetric, chi[i, j] = chi_ij for i != j) with the terms of the
    resonances `res` (tuples (a, b, c), b <= c, w_a ~ w_b + w_c) removed."""
    w = np.asarray(w, float)
    n = len(w)
    R = {(int(a), *sorted((int(b), int(c)))) for a, b, c in res}
    chi = np.zeros((n, n))
    for i in range(n):
        s = 0.0
        for k in range(n):
            t1 = 2 / w[k] + 1 / (2 * (2 * w[i] + w[k]))
            if (k, i, i) not in R:
                t1 -= 1 / (2 * (2 * w[i] - w[k]))
            s += f3[i, i, k] ** 2 * t1
        chi[i, i] = (f4[i, i] - s) / 16
    for i in range(n):
        for j in range(i + 1, n):
            s1 = sum(f3[i, i, k] * f3[j, j, k] / w[k] for k in range(n))
            s2 = 0.0
            for k in range(n):
                t2 = -1 / (w[i] + w[j] + w[k])
                if (k, *sorted((i, j))) not in R:
                    t2 += 1 / (w[i] + w[j] - w[k])
                if (j, *sorted((i, k))) not in R:
                    t2 -= 1 / (w[i] - w[j] + w[k])
                if (i, *sorted((j, k))) not in R:
                    t2 += 1 / (w[i] - w[j] - w[k])
                s2 += f3[i, j, k] ** 2 * t2 / 4
            v = (f4[i, j] - s1) / 4 + s2 / 2
            if B is not None and zeta is not None:
                v += sum(B[a] * zeta[a, i, j] ** 2 for a in range(3)) * (w[i] / w[j] + w[j] / w[i])
            chi[i, j] = chi[j, i] = v
    return chi


def level(w, chi, v):
    """E(v) - E(0) (cm-1) for the quantum numbers v (n,)."""
    v = np.asarray(v, float)
    h = v + 0.5
    E = w @ h + sum(chi[i, j] * h[i] * h[j] for i in range(len(w)) for j in range(i, len(w)))
    h0 = np.full(len(w), 0.5)
    E0 = w @ h0 + sum(chi[i, j] * h0[i] * h0[j] for i in range(len(w)) for j in range(i, len(w)))
    return float(E - E0)


def zpe(w, chi):
    n = len(w)
    return float(np.sum(w) / 2 + sum(chi[i, j] / 4 for i in range(n) for j in range(i, n)))


def gvpt2(w, f3, f4, B=None, zeta=None, res=None, max_gap=200.0, martin=1.0):
    """Deperturbed VPT2 plus variational treatment of the Fermi polyads. Returns a dict:
    chi (deperturbed), chi_plain (no resonance removed), fundamentals_vpt2 (plain), fundamentals_dvpt2,
    resonances, polyads [{states, energies, vectors}], fundamentals (GVPT2: the eigenstate with the largest
    weight on each fundamental), zpe."""
    w = np.asarray(w, float)
    n = len(w)
    if res is None:
        res = find_resonances(w, f3, max_gap, martin)
    tups = [(r['a'], r['b'], r['c']) for r in res]
    chi0 = chi_matrix(w, f3, f4, B, zeta)
    chi = chi_matrix(w, f3, f4, B, zeta, tups)

    def fund(c):
        return np.array([w[i] + 2 * c[i, i] + 0.5 * sum(c[i, j] for j in range(n) if j != i) for i in range(n)])
    nu0, nud = fund(chi0), fund(chi)
    # polyads: connected components of fundamentals and their partner states
    def st(*idx):
        v = [0] * n
        for i in idx:
            v[i] += 1
        return tuple(v)
    edges = [(st(a), st(b, c), coupling(f3, a, b, c)) for a, b, c in tups]
    nodes = sorted({e[0] for e in edges} | {e[1] for e in edges})
    parent = {s: s for s in nodes}

    def find(s):
        while parent[s] != s:
            s = parent[s]
        return s
    for a, b, _ in edges:
        parent[find(a)] = find(b)
    groups = {}
    for s in nodes:
        groups.setdefault(find(s), []).append(s)
    polyads = []
    nu = nud.copy()
    for states in groups.values():
        states = sorted(states, key=lambda s: level(w, chi, s))
        idx = {s: k for k, s in enumerate(states)}
        Hm = np.diag([level(w, chi, s) for s in states])
        for a, b, W in edges:
            if a in idx and b in idx:
                Hm[idx[a], idx[b]] = Hm[idx[b], idx[a]] = W
        E, U = np.linalg.eigh(Hm)
        polyads.append({'states': [list(s) for s in states], 'diagonal': np.diag(Hm).tolist(), 'energies': E.tolist(),
                        'vectors': U.tolist()})
        for s in states:
            if sum(s) == 1:
                i = s.index(1)
                k = int(np.argmax(U[idx[s]] ** 2))
                nu[i] = E[k]
    return {'chi': chi, 'chi_plain': chi0, 'fundamentals_vpt2': nu0, 'fundamentals_dvpt2': nud,
            'fundamentals': nu, 'resonances': res, 'polyads': polyads, 'zpe': zpe(w, chi), 'zpe_harmonic':
            float(np.sum(w) / 2)}
