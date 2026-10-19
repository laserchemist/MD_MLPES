"""Covalent diatomics-in-molecules (DIM) frame for H2O, built from eight OH states and the H2 singlet/triplet.

Polyatomic basis: O(2p^4) determinants (15) x spin of H_a (2) x spin of H_b (2) = 60 states, reduced to the 9
total-singlet states. Each fragment Hamiltonian is sum_n E_n(r) P_n, with P_n the projector onto OH state n,
identified by the O term (3P, 1D, 1S), |Lambda| along the bond and the O-H pair spin:

    H_DIM = H_OHa + H_OHb + H_HaHb - H_O - H_Ha - H_Hb

In eV relative to O(3P) + 2H this is  H = sum_a E_a P_a + sum_b E_b P_b + E_HH - sum_t (E_t - E_3P) P_t.

Limits are exact by construction (H + OH, O + 2H). Known physics gap: with only O 2p orbitals and no ionic O-H+
structures, the frame cannot bend water (it equals bond additivity for angles above ~60 degrees).

Slots:
  mix      per OH bond, input r: mixing angles between same-symmetry OH states of different O terms
           (2Pi: 3P/1D, 2Sigma+: 1D/1S); zero = pure-term states
  generic  optional dense 9x9 correction from invariant features, times an envelope that vanishes when either
           H atom leaves (so H + OH stays exact); a CDNN living inside the DIM frame
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from .base import Frame, Curve1D, slater_overlap
from ..geometry import h2o_xyz

# ------------------------------------------------------------------ O 2p^4 operator algebra (built once)
def _jw(nso):
    I2, Z = np.eye(2), np.diag([1.0, -1.0])
    a1 = np.array([[0.0, 1.0], [0.0, 0.0]])
    ops = []
    for p in range(nso):
        m = np.array([[1.0]])
        for q in range(nso):
            m = np.kron(m, Z if q < p else (a1 if q == p else I2))
        ops.append(m)
    return ops


_a = _jw(6)
_ad = [x.T for x in _a]
_occ = np.array([[(s >> (5 - p)) & 1 for p in range(6)] for s in range(64)])
_keep = np.where(_occ.sum(1) == 4)[0]
_P = lambda M: M[np.ix_(_keep, _keep)]


def _one_body(h):
    M = np.zeros((64, 64), dtype=complex)
    for i in range(3):
        for j in range(3):
            if h[i, j] != 0:
                for s in (0, 1):
                    M += h[i, j] * _ad[2 * i + s] @ _a[2 * j + s]
    return _P(M)


_eps = np.zeros((3, 3, 3))
for _i, _j, _k in itertools.permutations(range(3)):
    _eps[_i, _j, _k] = np.linalg.det(np.eye(3)[[_i, _j, _k]])
L_O = [_one_body(-1j * _eps[k]) for k in range(3)]
_Sz = _P(sum(0.5 * (_ad[2 * k] @ _a[2 * k] - _ad[2 * k + 1] @ _a[2 * k + 1]) for k in range(3))).astype(complex)
_Sp = _P(sum(_ad[2 * k] @ _a[2 * k + 1] for k in range(3))).astype(complex)
S_O = [(_Sp + _Sp.conj().T) / 2, (_Sp - _Sp.conj().T) / 2j, _Sz]
_PAULI = [np.array([[0, 1], [1, 0]]) / 2, np.array([[0, -1j], [1j, 0]]) / 2, np.diag([0.5, -0.5])]


def _p_along(n):
    return _one_body(np.outer(n, n).astype(complex))


def _spin_along(n):
    out = []
    for sk in _PAULI:
        M = np.zeros((64, 64), dtype=complex)
        for i in range(3):
            for j in range(3):
                if n[i] * n[j] == 0:
                    continue
                for s1 in (0, 1):
                    for s2 in (0, 1):
                        if sk[s1, s2] != 0:
                            M += n[i] * n[j] * sk[s1, s2] * _ad[2 * i + s1] @ _a[2 * j + s2]
        out.append(_P(M))
    return out


_I2, _I4, _IO = np.eye(2), np.eye(4), np.eye(15)
_embO = lambda M: np.kron(M, _I4)
_embA = lambda m: np.kron(_IO, np.kron(m, _I2))
_embB = lambda m: np.kron(_IO, np.kron(_I2, m))
_sq = lambda vec: sum(v @ v for v in vec)


def _projector(M, value, tol=1e-6):
    w, U = np.linalg.eigh((M + M.conj().T) / 2)
    sel = np.abs(w - value) < tol
    P = U[:, sel] @ U[:, sel].conj().T
    assert np.abs(P.imag).max() < 1e-10
    return P.real


P_TERM = {'3P': _projector(_embO(_sq(L_O) + 10 * _sq(S_O)), 22), '1D': _projector(_embO(_sq(L_O) + 10 * _sq(S_O)), 6),
          '1S': _projector(_embO(_sq(L_O) + 10 * _sq(S_O)), 0)}
_SA = [_embO(S_O[k]) + _embA(_PAULI[k]) for k in range(3)]
_SB = [_embO(S_O[k]) + _embB(_PAULI[k]) for k in range(3)]
_SHH = [_embA(_PAULI[k]) + _embB(_PAULI[k]) for k in range(3)]
_STOT = [_embO(S_O[k]) + _embA(_PAULI[k]) + _embB(_PAULI[k]) for k in range(3)]
P_SA = {0.5: _projector(_sq(_SA), 0.75), 1.5: _projector(_sq(_SA), 3.75)}
P_SB = {0.5: _projector(_sq(_SB), 0.75), 1.5: _projector(_sq(_SB), 3.75)}
P_HH = {'S': _projector(_sq(_SHH), 0.0), 'T': _projector(_sq(_SHH), 2.0)}
_w, _U = np.linalg.eigh(_sq(_STOT).real)
SINGLET = _U[:, np.abs(_w) < 1e-6]              # 60 x 9

OH_STATES = ['2S-', '2P1', '2P2', '4S-', '4P', '2S+1', '2S+2', '2D']       # column order of oh_curves RefSet
CLASSES = {('3P', 0.5, 0): '2S-', ('3P', 0.5, 1): '2P1', ('3P', 1.5, 0): '4S-', ('3P', 1.5, 1): '4P',
           ('1D', 0.5, 0): '2S+1', ('1D', 0.5, 1): '2P2', ('1D', 0.5, 2): '2D', ('1S', 0.5, 0): '2S+2'}
MIX_PAIRS = [('2P1', '2P2'), ('2S+1', '2S+2')]


def oh_projectors(n, P_S, emb_h):
    """Projectors onto the eight OH states for a bond along unit vector n, and the two term-mixing operators."""
    nL = _embO(sum(n[k] * L_O[k] for k in range(3)))
    lam2 = nL @ nL
    P_lam = {l: _projector(lam2, l * l) for l in (0, 1, 2)}
    P = {key: (P_TERM[t] @ P_lam[lam] @ P_S[s]).real for (t, s, lam), key in CLASSES.items()}
    T = _spin_along(n)
    ops = {MIX_PAIRS[0]: sum(_embO(T[k]) @ emb_h(_PAULI[k]) for k in range(3)), MIX_PAIRS[1]: _embO(_p_along(n))}
    coup = {}
    for (ka, kb), C in ops.items():
        Cab = (P[kb] @ C @ P[ka]).real
        sv = np.linalg.svd(Cab, compute_uv=False)
        c = sv[sv > 1e-8]
        assert np.allclose(c, c[0])
        coup[(ka, kb)] = (Cab + Cab.T) / c[0]
    return P, coup


_red = lambda M: SINGLET.T @ M @ SINGLET
_S_PHH = np.stack([_red(P_HH['S']), _red(P_HH['T'])])
_S_PTERM = np.stack([_red(P_TERM[t]) for t in ('3P', '1D', '1S')])
_IU = np.triu_indices(9)


def h2o_invariants(coords):
    """Permutation-invariant features from Slater overlaps of r1, r2 and r(H-H)."""
    out = []
    for r1, r2, th in np.atleast_2d(coords):
        X = h2o_xyz(r1, r2, th)
        r3 = np.linalg.norm(X[2] - X[1])
        s1, s2, s3 = [np.array([slater_overlap(x, z) for z in (0.5, 1.1, 2.0)]) for x in (r1, r2, r3)]
        out.append(np.concatenate([s1 + s2, s1 * s2, s3]))
    return np.array(out)


class DIMWaterFrame(Frame):
    name = 'covalent DIM (H2O, 9 singlet structures)'
    n_diabats = 9
    diabat_labels = [f'singlet structure {i + 1}' for i in range(9)]

    def __init__(self, oh_refset, h2_refset, generic=False):
        r = oh_refset.coords[:, 0]
        self.oh = [Curve1D(r, oh_refset.energies[:, i]) for i in range(8)]
        rh = h2_refset.coords[:, 0]
        self.hh = [Curve1D(rh, h2_refset.energies[:, i]) for i in range(2)]
        m = oh_refset.meta
        from ..units import HARTREE_EV
        self.dE_term = np.array([0.0, m['E_O1D'] - m['E_O3P'], m['E_O1S'] - m['E_O3P']]) * HARTREE_EV
        self.generic = generic
        self.slots = {'mix': (1, 2)}
        if generic:
            self.slots['generic'] = (9, len(_IU[0]))
        self._cache = {}

    def _geometry(self, x):
        key = tuple(np.round(x, 10))
        if key not in self._cache:
            r1, r2, th = x
            X = h2o_xyz(r1, r2, th)
            Pa, Ca = oh_projectors((X[1] - X[0]) / r1, P_SA, _embA)
            Pb, Cb = oh_projectors((X[2] - X[0]) / r2, P_SB, _embB)
            self._cache[key] = (np.stack([_red(Pa[k]) for k in OH_STATES]), np.stack([_red(Ca[p]) for p in MIX_PAIRS]),
                                np.stack([_red(Pb[k]) for k in OH_STATES]), np.stack([_red(Cb[p]) for p in MIX_PAIRS]),
                                float(np.linalg.norm(X[2] - X[1])))
        return self._cache[key]

    def prepare(self, coords):
        coords = np.atleast_2d(np.asarray(coords, float))
        geo = [self._geometry(x) for x in coords]
        t = lambda a: torch.tensor(np.asarray(a, float))
        r1, r2 = coords[:, 0], coords[:, 1]
        r3 = np.array([g[4] for g in geo])
        c = {'PA': t([g[0] for g in geo]), 'CA': t([g[1] for g in geo]), 'PB': t([g[2] for g in geo]),
             'CB': t([g[3] for g in geo]),
             'EA': t(np.stack([f(r1) for f in self.oh], 1)), 'EB': t(np.stack([f(r2) for f in self.oh], 1)),
             'EHH': t(np.stack([f(r3) for f in self.hh], 1)),
             'slot_in': {'mix': t(np.stack([r1, r2], 1))[..., None]}}
        if self.generic:
            sa, sb = slater_overlap(r1, 1.1), slater_overlap(r2, 1.1)
            c['env'] = t(sa * sb / (slater_overlap(0.96, 1.1) ** 2))
            c['slot_in']['generic'] = t(h2o_invariants(coords))[:, None, :]
        return c

    def assemble(self, c, out):
        N = c['EA'].shape[0]
        H = torch.einsum('nk,nkij->nij', c['EA'], c['PA']) + torch.einsum('nk,nkij->nij', c['EB'], c['PB'])
        idx = {k: i for i, k in enumerate(OH_STATES)}
        for E, P, C, b in [(c['EA'], c['PA'], c['CA'], 0), (c['EB'], c['PB'], c['CB'], 1)]:
            for m, (ka, kb) in enumerate(MIX_PAIRS):
                f = out['mix'][:, b, m]
                cs, sn = torch.cos(f), torch.sin(f)
                ea, eb = E[:, idx[ka]], E[:, idx[kb]]
                Pa, Pb = P[:, idx[ka]], P[:, idx[kb]]
                H = H + ((ea * (cs * cs - 1) + eb * sn * sn)[:, None, None] * Pa
                         + (ea * sn * sn + eb * (cs * cs - 1))[:, None, None] * Pb
                         + ((ea - eb) * cs * sn)[:, None, None] * C[:, m])
        H = H + torch.einsum('nk,kij->nij', c['EHH'], torch.tensor(_S_PHH))
        H = H - torch.einsum('k,kij->ij', torch.tensor(self.dE_term), torch.tensor(_S_PTERM))[None]
        if self.generic:
            g = out['generic'][:, 0]
            G = torch.zeros(N, 9, 9)
            G[:, _IU[0], _IU[1]] = g
            G = G + G.transpose(1, 2) - torch.diag_embed(torch.diagonal(G, dim1=1, dim2=2))
            H = H + c['env'][:, None, None] * G
        return H


def bond_additivity(coords, frame: DIMWaterFrame):
    """Baseline: two OH X2Pi bonds plus H-H triplet repulsion (eV relative to O(3P) + 2H)."""
    coords = np.atleast_2d(coords)
    out = []
    for r1, r2, th in coords:
        X = h2o_xyz(r1, r2, th)
        r3 = np.linalg.norm(X[2] - X[1])
        out.append(frame.oh[1](r1) + frame.oh[1](r2) + frame.hh[1](r3))
    return np.array(out)
