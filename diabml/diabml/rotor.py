"""Rotational structure of vibrational bands: asymmetric-rotor line lists and the rotational constants of
vibrational levels.

    E, V = rotor_levels(J, A, B, C)                         # levels of one J (cm-1), eigenvectors in |J K>
    lines = band_lines(nu0, lower, upper, comp, T, Jmax)    # (nu, strength) of one vibrational band
    y = contour(lines_list, grid, fwhm)                     # binned and Gaussian-broadened

Rigid asymmetric rotor in the Ir representation (z = a, x = b, y = c):
    H = A Ja^2 + B Jb^2 + C Jc^2, diagonalised in the symmetric-top basis |J K>, K = -J..J (real symmetric).

Line strengths (summed over M, M' and space-fixed components) for a transition moment with body-fixed spherical
components mu_q (q = 0: a; q = +-1: -+(mu_b +- i mu_c)/sqrt 2):
    S(J'n' <- Jn) = (2J + 1) | sum_{K, q} V'_{K+q, n'} V_{K, n} <J K 1 q | J' K+q> mu_q |^2,
which satisfies sum_{J'n'} S = (2J + 1) |mu|^2 (tests/test_rotor.py checks it, and the symmetric-top Honl-London
factors). The a, b and c components have different selection rules (parity of Ka, Kc) and do not interfere, so a
hybrid band is the sum of its a-, b- and c-type parts weighted by mu_a^2, mu_b^2, mu_c^2.

Intensities: absorption nu S exp(-E''/kT) / Q_rot (lower level populations); emission (photon rate)
nu^3 S exp(-E'/kT) / Q_rot (upper level populations, the degeneracy 2J'+1 cancelling against the rate's 1/(2J'+1)).
CH2OO has no symmetry exchanging its H atoms (Cs, both H in the plane but inequivalent): no spin statistics.

Rotational constants of vibrational levels: Watson's inverse effective inertia tensor to second order in the
normal coordinates,
    mu = I^-1 - sum_k I^-1 a_k I^-1 Q_k + (3/4) sum_kl I^-1 a_k I^-1 a_l I^-1 Q_k Q_l,     a_k = dI/dQ_k,
B_beta(Q) = B_CONV mu_beta,beta; its expectation value over a vibrational wavefunction gives the harmonic and
anharmonic parts of the vibration-rotation constants. The Coriolis part (second order in the rotation-vibration
coupling) is added perturbatively: alpha_k^beta(Cor) = -(2 B_beta^2 / omega_k) sum_j (zeta^beta_kj)^2
(3 omega_k^2 + omega_j^2) / (omega_k^2 - omega_j^2), with B_v = B - sum_k alpha_k (v_k + 1/2).
"""
from __future__ import annotations

import numpy as np

from .vibdvr import B_CONV

AXES = 'abc'


# ------------------------------------------------------------------------------------------ rotor
def rotor_levels(J, A, B, C):
    """Energies (2J+1,) ascending and eigenvectors (2J+1, 2J+1) in the basis K = -J..J (Ir: z = a)."""
    K = np.arange(-J, J + 1)
    H = np.diag((B + C) / 2 * (J * (J + 1) - K ** 2) + A * K ** 2).astype(float)
    k = K[:-2]
    off = (B - C) / 4 * np.sqrt((J * (J + 1) - k * (k + 1)) * (J * (J + 1) - (k + 1) * (k + 2)))
    i = np.arange(len(k))
    H[i, i + 2] = H[i + 2, i] = off
    return np.linalg.eigh(H)


def _cg(J, K, q, Jp):
    """Clebsch-Gordan <J K 1 q | Jp K+q> for arrays K (Condon-Shortley; the j2 = 1 table)."""
    K = np.asarray(K, float)
    m = K + q
    J = float(J)
    with np.errstate(invalid='ignore', divide='ignore'):
        if Jp == J + 1:
            if q == 1:
                v = np.sqrt((J + m) * (J + m + 1) / ((2 * J + 1) * (2 * J + 2)))
            elif q == 0:
                v = np.sqrt((J - m + 1) * (J + m + 1) / ((2 * J + 1) * (J + 1)))
            else:
                v = np.sqrt((J - m) * (J - m + 1) / ((2 * J + 1) * (2 * J + 2)))
        elif Jp == J:
            if J == 0:
                return np.zeros_like(K)
            if q == 1:
                v = -np.sqrt((J + m) * (J - m + 1) / (2 * J * (J + 1)))
            elif q == 0:
                v = m / np.sqrt(J * (J + 1))
            else:
                v = np.sqrt((J - m) * (J + m + 1) / (2 * J * (J + 1)))
        else:
            if J == 0:
                return np.zeros_like(K)
            if q == 1:
                v = np.sqrt((J - m) * (J - m + 1) / (2 * J * (2 * J + 1)))
            elif q == 0:
                v = -np.sqrt((J - m) * (J + m) / (J * (2 * J + 1)))
            else:
                v = np.sqrt((J + m + 1) * (J + m) / (2 * J * (2 * J + 1)))
    v = np.nan_to_num(v)
    v[np.abs(m) > Jp] = 0.0
    return v


_MUQ = {'a': {0: 1.0}, 'b': {1: -1 / np.sqrt(2), -1: 1 / np.sqrt(2)}, 'c': {1: -1j / np.sqrt(2), -1: -1j / np.sqrt(2)}}


def strengths(J, Jp, V, Vp, comp):
    """Line strengths S (2Jp+1, 2J+1) between the eigenvectors V (lower, J) and Vp (upper, Jp) for a unit
    transition moment along axis comp ('a', 'b' or 'c')."""
    K = np.arange(-J, J + 1)
    M = np.zeros((2 * Jp + 1, 2 * J + 1), complex)          # <Jp K'| mu |J K> in the |J K> bases
    for q, c in _MUQ[comp].items():
        cg = _cg(J, K, q, Jp)
        Kp = K + q
        ok = np.abs(Kp) <= Jp
        M[Kp[ok] + Jp, np.flatnonzero(ok)] += c * cg[ok]
    amp = Vp.T @ M @ V
    return (2 * J + 1) * np.abs(amp) ** 2


def band_lines(nu0, lower, upper, comp, T, Jmax=None, kind='absorption', smin=1e-7):
    """Rotational lines (nu, intensity) of one vibrational band of type comp with band origin nu0 (cm-1),
    lower and upper rotational constants (A, B, C), rotational temperature T (K). Absorption intensities
    nu S exp(-E''/kT) / Q, emission nu^3 S exp(-E'/kT) / Q (Q from the emitting / absorbing level's
    constants); both sum to about nu |mu|^2 (absorption) for unit vibrational moment. Lines weaker than smin of
    the strongest are dropped."""
    kT = 0.6950348 * T
    pop_c = lower if kind == 'absorption' else upper
    if Jmax is None:
        Jmax = int(np.sqrt(12 * kT / max(min(pop_c[1:]), 1e-6))) + 5
    lev_lo = [rotor_levels(J, *lower) for J in range(Jmax + 2)]
    lev_up = [rotor_levels(J, *upper) for J in range(Jmax + 2)]
    pop_lev = lev_lo if kind == 'absorption' else lev_up
    Q = sum((2 * J + 1) * np.exp(-E / kT).sum() for J, (E, _) in enumerate(pop_lev))
    nus, ints = [], []
    for J in range(Jmax + 1):
        El, Vl = lev_lo[J]
        for Jp in (J - 1, J, J + 1):
            if Jp < 0 or (J == 0 and Jp == 0):
                continue
            Eu, Vu = lev_up[Jp]
            S = strengths(J, Jp, Vl, Vu, comp)
            nu = nu0 + Eu[:, None] - El[None, :]
            if kind == 'absorption':
                w = nu * S * np.exp(-El[None, :] / kT) / Q
            else:
                w = nu ** 3 * S * np.exp(-Eu[:, None] / kT) / Q
            keep = S > 1e-12
            nus.append(nu[keep])
            ints.append(w[keep])
    nu, I = np.concatenate(nus), np.concatenate(ints)
    keep = I > smin * I.max()
    return nu[keep], I[keep]


def contour(bands, grid, fwhm):
    """Sum of line lists [(nu, I), ...] on a uniform grid, Gaussian FWHM fwhm (cm-1); area-preserving."""
    d = grid[1] - grid[0]
    y = np.zeros(len(grid))
    for nu, I in bands:
        idx = np.rint((nu - grid[0]) / d).astype(int)
        ok = (idx >= 0) & (idx < len(grid))
        y += np.bincount(idx[ok], weights=I[ok], minlength=len(grid))
    s = fwhm / 2.3548 / d
    k = np.arange(-int(5 * s) - 1, int(5 * s) + 2)
    g = np.exp(-0.5 * (k / s) ** 2)
    g /= g.sum()
    return np.convolve(y, g, mode='same') / d


# ------------------------------------------------------------------------------------------ constants
def inertia_derivatives(modes):
    """Principal moments I (3,) (amu A^2), principal axes R, and a (n, 3, 3) = dI/dQ_k in the principal frame
    (amu^1/2 A)."""
    from .vpt2 import principal_axes
    m = modes.masses
    I, R = principal_axes(modes.P0, m)
    Pc = modes.P0 - (m[:, None] * modes.P0).sum(0) / m.sum()
    r = Pc @ R
    n = modes.L.shape[1]
    d = (modes.L.T.reshape(n, -1, 3) / np.sqrt(m)[None, :, None]) @ R        # (n, N, 3) per unit Q
    a = np.zeros((n, 3, 3))
    for k in range(n):
        rd = (m * (r * d[k]).sum(1)).sum()
        a[k] = 2 * rd * np.eye(3) - np.einsum('i,ia,ib->ab', m, r, d[k]) - np.einsum('i,ia,ib->ab', m, d[k], r)
    return I, R, a


def rotational_constant_fn(modes):
    """B(q): (M, 3) the a, b, c rotational constants (cm-1) of Watson's mu tensor (second order) at
    dimensionless normal coordinates q (M, n)."""
    from .vpt2 import C_CM
    I, _, a = inertia_derivatives(modes)
    g = np.sqrt(modes.omega / C_CM)
    Ii = 1.0 / I
    b = a * Ii[None, :, None] * Ii[None, None, :]             # I^-1 a_k I^-1
    c = np.einsum('kab,lbc->klac', b, a * Ii[None, None, :])  # I^-1 a_k I^-1 a_l I^-1 (left factor in b)

    def fn(q):
        Q = np.asarray(q, float) / g
        lin = np.einsum('mk,kaa->ma', Q, b)
        quad = np.einsum('mk,ml,klaa->ma', Q, Q, c)
        return B_CONV * (Ii[None, :] - lin + 0.75 * quad)
    return fn


def coriolis_alpha(modes, cap=None):
    """Coriolis contributions to the vibration-rotation constants alpha (n, 3) (cm-1; B_v = B - sum alpha
    (v + 1/2)). Pairs closer than `cap` (cm-1, None: no cap) are left out (Coriolis resonances, where the
    perturbative form fails and the contributions of the two partners cancel in their sum)."""
    from .vpt2 import coriolis
    zeta, B = coriolis(modes)
    w = modes.omega
    n = len(w)
    al = np.zeros((n, 3))
    for k in range(n):
        for j in range(n):
            if j == k or (cap is not None and abs(w[k] - w[j]) < cap):
                continue
            al[k] += -(2 * B ** 2 / w[k]) * zeta[:, k, j] ** 2 * (3 * w[k] ** 2 + w[j] ** 2) / (w[k] ** 2 - w[j] ** 2)
    return al, B
