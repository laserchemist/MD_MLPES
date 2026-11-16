"""Ring-polymer molecular dynamics (path-integral sampling, RPMD and thermostatted RPMD) for batches of trajectories.

    rp = RingPolymer(masses, beads=32, temperature=300.0, force_fn=f)   # f(X (M, N, 3) A) -> (E (M,) eV, F (M, N, 3) eV/A)
    q, p = rp.initial(X0, n_traj, rng)                                  # all beads at X0, momenta at the bead temperature
    q, p = rp.run(q, p, steps, dt, gamma0=0.01, lam=1.0, rng=rng)      # PIMD: thermostat everything (PILE-L)
    for q, p, E in rp.trajectory(q, p, steps, dt, lam=0.5, rng=rng): ...    # TRPMD: centroid free, internal modes damped

State: q, p (n_traj, P, N, 3) in A and amu A / fs; all trajectories and beads are propagated together and every force
call is one batched evaluation of n_traj * P geometries.

Ring-polymer Hamiltonian (Craig and Manolopoulos 2004), P beads at T_P = P T:
    H_P = sum_j [ |p_j|^2 / 2m + V(q_j) ] + sum_j (m omega_P^2 / 2) |q_j - q_{j+1}|^2,   omega_P = P k_B T / hbar.
Integrator (OBABO, Ceriotti, Parrinello, Markland and Manolopoulos 2010): the free ring polymer is propagated exactly
in its normal modes (frequencies omega_k = 2 omega_P sin(k pi / P)); the O step is the PILE-L Langevin thermostat in
the normal modes, friction gamma_k = 2 lam omega_k for the internal modes and gamma0 for the centroid. lam = 1 with a
centroid thermostat samples the quantum Boltzmann distribution (PIMD); lam = 0 and no centroid thermostat is RPMD;
lam = 1/2 (or as chosen) with the centroid free is thermostatted RPMD (TRPMD, Rossi, Ceriotti and Manolopoulos 2014),
which damps the spurious resonances between internal ring-polymer modes and physical vibrations. P = 1 is classical
molecular dynamics.

Spectrum: the Kubo-transformed dipole autocorrelation of the bead-averaged dipole; n(w) alpha(w) ~ w^2 C_Kubo(w),
which diabml.dynamics.ir.ir_spectrum computes from the bead-averaged dipole time series as it does for classical MD.
"""
from __future__ import annotations

import numpy as np

HBAR = 6.350779924e-3            # amu A^2 / fs
KB = 8.314462618e-7              # amu A^2 / fs^2 / K
FCONV = 9.648533212e-3           # (eV / A) * fs -> amu A / fs
ECONV = 103.6427                 # amu A^2 / fs^2 -> eV


def normal_mode_matrix(P):
    """Orthogonal C (P, P): bead j -> normal mode k (C[j, k]); k = 0 centroid (times sqrt P)."""
    C = np.zeros((P, P))
    j = np.arange(P)
    C[:, 0] = 1 / np.sqrt(P)
    for k in range(1, P):
        if k < P / 2:
            C[:, k] = np.sqrt(2 / P) * np.cos(2 * np.pi * j * k / P)
        elif k == P / 2:
            C[:, k] = (-1.0) ** j / np.sqrt(P)
        else:
            C[:, k] = np.sqrt(2 / P) * np.sin(2 * np.pi * j * k / P)
    return C


class RingPolymer:
    def __init__(self, masses, beads, temperature, force_fn):
        self.m = np.asarray(masses, float)                 # (N,) amu
        self.P = int(beads)
        self.T = float(temperature)
        self.f = force_fn
        self.betaP = 1.0 / (KB * self.T * self.P)          # 1 / (k_B T_P)
        self.wP = self.P * KB * self.T / HBAR              # fs^-1
        k = np.arange(self.P)
        self.wk = 2 * self.wP * np.sin(k * np.pi / self.P)
        self.C = normal_mode_matrix(self.P)

    # ---------------------------------------------------------------- transforms
    def to_nm(self, x):
        return np.einsum('jk,tjna->tkna', self.C, x)

    def from_nm(self, x):
        return np.einsum('jk,tkna->tjna', self.C, x)

    def forces(self, q):
        t, P, N, _ = q.shape
        E, F = self.f(q.reshape(t * P, N, 3))
        return np.asarray(E).reshape(t, P), np.asarray(F).reshape(t, P, N, 3)

    # ---------------------------------------------------------------- steps
    def _free(self, q, p, dt):
        """Exact free ring-polymer propagation for dt (normal modes)."""
        Q, Pm = self.to_nm(q), self.to_nm(p)
        m = self.m[None, :, None]
        Q[:, 0] += Pm[:, 0] / m * dt
        w = self.wk[1:][None, :, None, None]
        c, s = np.cos(w * dt), np.sin(w * dt)
        mm = self.m[None, None, :, None]
        Qi, Pi = Q[:, 1:].copy(), Pm[:, 1:].copy()
        Pm[:, 1:] = c * Pi - mm * w * s * Qi
        Q[:, 1:] = s / (mm * w) * Pi + c * Qi
        return self.from_nm(Q), self.from_nm(Pm)

    def _thermo(self, p, dt, gamma0, lam, rng):
        """PILE-L half step (dt is the half step): gamma_k = 2 lam omega_k, centroid gamma0 (fs^-1; 0: none)."""
        g = 2 * lam * self.wk.copy()
        g[0] = gamma0
        if not np.any(g > 0):
            return p
        Pm = self.to_nm(p)
        c1 = np.exp(-g * dt)[None, :, None, None]
        c2 = np.sqrt(1 - c1 ** 2)
        sd = np.sqrt(self.m / self.betaP)[None, None, :, None]
        Pm = c1 * Pm + c2 * sd * rng.standard_normal(Pm.shape)
        return self.from_nm(Pm)

    def initial(self, X0, n_traj, rng):
        q = np.repeat(np.asarray(X0, float)[None, None], n_traj, 0).repeat(self.P, 1)
        p = rng.standard_normal(q.shape) * np.sqrt(self.m / self.betaP)[None, None, :, None]
        return q, p

    def step(self, q, p, F, dt, gamma0, lam, rng):
        p = self._thermo(p, dt / 2, gamma0, lam, rng)
        p = p + 0.5 * dt * FCONV * F
        q, p = self._free(q, p, dt)
        E, F = self.forces(q)
        p = p + 0.5 * dt * FCONV * F
        p = self._thermo(p, dt / 2, gamma0, lam, rng)
        return q, p, E, F

    def run(self, q, p, steps, dt, gamma0=0.01, lam=1.0, rng=None):
        """PIMD (or any thermostat setting) without recording; returns the final state."""
        rng = rng or np.random.default_rng()
        E, F = self.forces(q)
        for _ in range(int(steps)):
            q, p, E, F = self.step(q, p, F, dt, gamma0, lam, rng)
        return q, p

    def trajectory(self, q, p, steps, dt, lam=0.5, gamma0=0.0, rng=None):
        """Generator of (q, p, E_beads) after every step."""
        rng = rng or np.random.default_rng()
        E, F = self.forces(q)
        for _ in range(int(steps)):
            q, p, E, F = self.step(q, p, F, dt, gamma0, lam, rng)
            yield q, p, E

    # ---------------------------------------------------------------- observables
    def spring_energy(self, q):
        d = q - np.roll(q, -1, axis=1)
        return 0.5 * (self.m[None, None, :, None] * self.wP ** 2 * d ** 2).sum((1, 2, 3)) * ECONV   # eV

    def hamiltonian(self, q, p, E):
        kin = (p ** 2 / (2 * self.m[None, None, :, None])).sum((1, 2, 3)) * ECONV
        return kin + self.spring_energy(q) + E.sum(1)                                              # eV

    def centroid_temperature(self, p):
        pc = p.mean(1)                                               # centroid momentum per atom (t, N, 3)
        ke = (pc ** 2 / (2 * self.m[None, :, None])).sum((1, 2))      # amu A^2/fs^2, physical-mass centroid
        dof = 3 * len(self.m)
        return 2 * ke / (dof * KB)                                   # K

    def potential_estimator(self, E):
        """Thermal average of the potential energy per trajectory (bead average, eV)."""
        return E.mean(1)
