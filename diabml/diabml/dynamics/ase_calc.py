"""ASE calculator for diabml matrix models: energies, forces and couplings of every state.

    from diabml.models import load_model
    from diabml.dynamics import CoordinateSurface, ch2oo_coordinates, ch2oo_atoms, DiabmlCalculator

    model = load_model('runs/01_baseline/models/random_framed.pt')
    surf = CoordinateSurface(model, ch2oo_coordinates(model.meta['coord_names']))
    atoms = ch2oo_atoms()
    atoms.calc = DiabmlCalculator(surf, state=0)
    atoms.get_potential_energy(), atoms.get_forces()          # the active state (eV, eV/Angstrom)
    atoms.calc.results['energies'], atoms.calc.results['all_forces'], atoms.calc.results['couplings']

Physics. With Omega(R) the learned matrix, eigenpairs (E_k, v_k) and G_kl = v_k^T (dOmega/dR) v_l:
    E_k     adiabatic energy of state k
    F_k = -G_kk                         (Hellmann-Feynman for matrices: exact for the model surface)
    d_kl = G_kl / (E_l - E_k), k != l   derivative coupling between adiabatic states, in this matrix's basis.
For a compatible matrix (CDNN without diabatic restraints) d_kl is a property of the matrix, not of the molecule;
methods that need only energies and gradients (curvature-driven surface hopping, CSDM) are the safe choice until
the matrix is restrained toward diabatic. d_kl is singular where E_k = E_l; near such points it is reported as is.

Ensembles (EnsembleModel, mode 'adiabatic'): every member is solved separately and energies, forces, structure
weights and phase-aligned couplings are averaged; results['member_energies'] (K, n_states) is the members' spread,
a running uncertainty estimate along a trajectory.
"""
from __future__ import annotations

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes


class DiabmlCalculator(Calculator):
    implemented_properties = ['energy', 'free_energy', 'forces']

    def __init__(self, surface, state=0, couplings=True, **kwargs):
        super().__init__(**kwargs)
        self.surface = surface
        self._state = int(state)
        self.couplings = couplings

    # --- active state (switch it for surface hopping; cached results are dropped)
    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, k):
        if int(k) != self._state:
            self._state = int(k)
            self.reset()

    def _solve(self, surface, R):
        """Energies, forces, structure weights and couplings of every state for one matrix surface."""
        with torch.no_grad():
            M = surface.matrix(R)
        f = surface.matrix
        try:
            J = torch.autograd.functional.jacobian(f, R, vectorize=True)
        except Exception:
            J = torch.autograd.functional.jacobian(f, R)                # (n, n, n_atoms, 3)
        E, V = torch.linalg.eigh(M)
        ns = min(surface.n_states, len(E))
        Vs = V[:, :ns]
        G = torch.einsum('ik,ijab,jl->klab', Vs, J, Vs)              # (ns, ns, n_atoms, 3)
        out = {'energies': E[:ns].numpy().copy(), 'all_forces': -torch.stack([G[s, s] for s in range(ns)]).numpy(),
               'matrix': M.numpy().copy(), 'eigenvectors': Vs.numpy().copy(),
               'structure_weights': (Vs ** 2).numpy().T.copy()}
        if self.couplings:
            dE = E[:ns][None, :] - E[:ns][:, None]                    # E_l - E_k
            with np.errstate(divide='ignore', invalid='ignore'):
                d = G.numpy() / dE.numpy()[:, :, None, None]
            for s in range(ns):
                d[s, s] = 0.0
            out['couplings'] = d                                      # (ns, ns, n_atoms, 3), 1/Angstrom
        return out

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        R = torch.tensor(self.atoms.get_positions(), dtype=torch.float64)
        if getattr(self.surface, 'is_ensemble', False):
            outs = [self._solve(s, R) for s in self.surface.member_surfaces()]
            res = {'energies': np.mean([o['energies'] for o in outs], 0),
                   'all_forces': np.mean([o['all_forces'] for o in outs], 0),
                   'structure_weights': np.mean([o['structure_weights'] for o in outs], 0),
                   'member_energies': np.array([o['energies'] for o in outs]),
                   'matrix': outs[0]['matrix'], 'eigenvectors': outs[0]['eigenvectors']}
            if self.couplings:
                # eigenvector phases are arbitrary in each member: align every coupling vector with member 0's
                ref = outs[0]['couplings']
                acc = np.zeros_like(ref)
                for o in outs:
                    d = o['couplings']
                    sgn = np.sign(np.nansum(d * ref, axis=(2, 3)))
                    sgn[sgn == 0] = 1.0
                    acc += sgn[:, :, None, None] * d
                res['couplings'] = acc / len(outs)
        else:
            res = self._solve(self.surface, R)
        ns = len(res['energies'])
        k = self._state
        if not 0 <= k < ns:
            raise ValueError(f'state {k} outside 0..{ns - 1}')
        res.update({'energy': float(res['energies'][k]), 'free_energy': float(res['energies'][k]),
                    'forces': res['all_forces'][k].copy()})
        self.results = res

    def state_energies(self, atoms):
        self.get_property('energy', atoms)
        return self.results['energies']

    def state_forces(self, atoms):
        self.get_property('forces', atoms)
        return self.results['all_forces']
