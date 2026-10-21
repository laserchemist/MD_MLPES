"""CH2OO helpers for dynamics: internal coordinates from Cartesians (differentiable), an ASE Atoms builder, and the
restraints that make reduced-dimensional (planar, rigid-CH2O) models usable for test trajectories.

Atom order everywhere: C, O1 (central), O2 (terminal), H, H  (symbols "COOHH"), as in diabml.geometry.ch2oo_xyz.
"""
from __future__ import annotations

import numpy as np
import torch

from ..geometry import ch2oo_xyz, CH2OO_REF

CH2OO_SYMBOLS = 'COOHH'
C, O1, O2, H1, H2 = range(5)


def _dihedral(a, b, c, d):
    """Dihedral a-b-c-d in radians (torch, differentiable; atan2 form, smooth through 0 and 180 degrees)."""
    b0, b1, b2 = a - b, c - b, d - c
    n1, n2 = torch.linalg.cross(b0, b1), torch.linalg.cross(b1, b2)
    m1 = torch.linalg.cross(n1, b1 / torch.linalg.norm(b1))
    return torch.atan2(m1 @ n2, n1 @ n2)


def _angle(a, b, c):
    """Angle a-b-c in degrees (torch, differentiable)."""
    u, v = a - b, c - b
    cos = (u @ v) / (torch.linalg.norm(u) * torch.linalg.norm(v))
    return torch.rad2deg(torch.arccos(cos.clamp(-1 + 1e-12, 1 - 1e-12)))


_COORD = {
    'r_OO': lambda R: torch.linalg.norm(R[O2] - R[O1]),
    'a_COO': lambda R: _angle(R[C], R[O1], R[O2]),
    'r_CO': lambda R: torch.linalg.norm(R[O1] - R[C]),
}


def ch2oo_coordinates(names):
    """coord_fn for CoordinateSurface: R (5, 3) -> tensor of the named internal coordinates, in that order."""
    unknown = [n for n in names if n not in _COORD]
    if unknown:
        raise ValueError(f'no Cartesian map for {unknown}; available: {sorted(_COORD)}')
    return lambda R: torch.stack([_COORD[n](R) for n in names])


def ch2oo_atoms(**geometry):
    """ase.Atoms for planar CH2OO; keyword arguments as diabml.geometry.ch2oo_xyz (r_oo, a_coo, r_co, r_ch, a_hco)."""
    from ase import Atoms
    xyz = ch2oo_xyz(**{k: v for k, v in geometry.items() if k in CH2OO_REF})
    return Atoms(CH2OO_SYMBOLS, positions=np.array([x for _, x in xyz]))


def reduced_restraints(surface, positions, free=('r_OO', 'a_COO', 'r_CO'), k_bond=30.0, k_angle=4.0, k_dihedral=1.0):
    """Wrap a reduced CH2OO surface with stiff harmonic restraints on everything the model does not describe.

    Restrained around their values in `positions` (Angstrom): both C-H bonds, both H-C-O1 angles and the two
    H-C-O1-O2 dihedrals (a planar CH2O frame holding O2 in its plane); also the C-O1 bond and the C-O1-O2 angle
    unless listed in `free`. Force constants: eV/Angstrom^2 for bonds, eV/rad^2 for angles, eV for dihedrals
    (k (1 - cos(phi - phi0))). The defaults give C-H stretch and bend frequencies of a real CH2O frame order of
    magnitude, so use a time step of 0.25 fs or less. The restraint energy is added to every state equally (a
    multiple of the identity in the matrix), so eigenvectors and couplings are untouched and energy is conserved.

    Why restraints and not ase.constraints.FixInternals: with velocity Verlet the latter does not conserve energy
    for these coordinates (hundreds of meV in tests), which would hide real problems of the surface."""
    if 'r_OO' not in free:
        raise ValueError('r_OO must stay free')
    R0 = torch.as_tensor(np.asarray(positions, float))
    bonds = [(C, H1), (C, H2)] + ([] if 'r_CO' in free else [(C, O1)])
    angles = [(H1, C, O1), (H2, C, O1)] + ([] if 'a_COO' in free else [(C, O1, O2)])
    dihedrals = [(H1, C, O1, O2), (H2, C, O1, O2)]
    terms = ([('bond', b, float(torch.linalg.norm(R0[b[0]] - R0[b[1]])), k_bond) for b in bonds] +
             [('angle', a, float(torch.deg2rad(_angle(*(R0[i] for i in a)))), k_angle) for a in angles] +
             [('dihedral', d, float(_dihedral(*(R0[i] for i in d))), k_dihedral) for d in dihedrals])
    return RestrainedSurface(surface, terms)


class RestrainedSurface:
    """A surface plus restraint energy U(R) added to every state: matrix(R) + U(R) * identity."""

    def __init__(self, surface, terms):
        self.surface, self.terms = surface, terms
        self.n_states = surface.n_states
        self.n_atoms = getattr(surface, 'n_atoms', None)

    def restraint_energy(self, R):
        U = torch.zeros((), dtype=R.dtype)
        for kind, idx, x0, k in self.terms:
            if kind == 'bond':
                U = U + 0.5 * k * (torch.linalg.norm(R[idx[0]] - R[idx[1]]) - x0) ** 2
            elif kind == 'angle':
                U = U + 0.5 * k * (torch.deg2rad(_angle(*(R[i] for i in idx))) - x0) ** 2
            else:
                U = U + k * (1 - torch.cos(_dihedral(*(R[i] for i in idx)) - x0))
        return U

    def matrix(self, R):
        M = self.surface.matrix(R)
        return M + self.restraint_energy(R) * torch.eye(M.shape[0], dtype=M.dtype)
