"""CH2OO reference data: SA4-CASSCF(8,6)/6-31G* singlet A' states on a planar (r_OO, angle COO) grid.

Active space (8 electrons in 6 orbitals): the three pi orbitals of C-O-O (A''), the O-O sigma and sigma* (A'),
and the in-plane lone pair of the terminal O (A'). That is enough for O-O cleavage to CH2O + O(1D), whose three
A' components are the three lowest singlet A' asymptotes, and for the biradical channel CH2O(T) + O(3P).
"""
from __future__ import annotations

import numpy as np

from ..geometry import ch2oo_xyz, CH2OO_REF
from ..refset import RefSet
from ..units import HARTREE_EV
from .casscf import CASSpec, scan

R_OO = [1.20, 1.275, 1.35, 1.425, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.15, 2.30, 2.50, 2.75, 3.00, 3.50, 4.00, 5.00, 6.50, 8.00]
A_COO = [65.0, 85.0, 100.0, 117.7, 135.0, 155.0]
SPEC = CASSpec(ncas=6, nelecas=8, nroots=4, basis='6-31g*', symmetry='Cs', wfnsym="A'")


def default_grid():
    return np.array([(r, a) for a in A_COO for r in R_OO])


def generate(path, grid=None, spec: CASSpec = SPEC, log=print) -> RefSet:
    grid = default_grid() if grid is None else np.asarray(grid)
    # start the scan at the reference geometry so the first orbitals are the well-behaved equilibrium ones
    i0 = int(np.argmin(np.sum(((grid - [CH2OO_REF['r_oo'], CH2OO_REF['a_coo']]) / [1.0, 50.0]) ** 2, 1)))
    order = np.r_[i0, np.delete(np.arange(len(grid)), i0)]
    res = scan(grid[order], lambda r, a: ch2oo_xyz(r_oo=r, a_coo=a), spec, fragment_atoms=(2,), log=log)
    inv = np.argsort(order)
    res = {k: v[inv] for k, v in res.items()}
    E = res['E_hartree']
    # zero of energy: the lowest state at the largest O-O distance, reference angle (CH2O + O(1D) asymptote)
    far = np.where(grid[:, 0] == grid[:, 0].max())[0]
    zero = float(np.nanmin(E[far, 0]))
    rs = RefSet('CH2OO SA4-CASSCF(8,6)/6-31G*', grid, (E - zero) * HARTREE_EV, ['r_OO', 'a_COO'],
                extra={'q_O_terminal': res['charge'], 'n_unpaired': res['n_unpaired'],
                       'converged': res['converged'], 'active_space_consistent': res['active_space_consistent']},
                meta={'zero': 'lowest state at largest r_OO (CH2O + O(1D))', 'zero_hartree': zero,
                      'method': 'SA4-CASSCF(8,6)/6-31G*, singlet A\' (Cs), planar, CH2O fragment rigid',
                      'fixed': {k: v for k, v in CH2OO_REF.items() if k not in ('r_oo', 'a_coo')}})
    rs.save(path)
    return rs


if __name__ == '__main__':
    import sys
    generate(sys.argv[1] if len(sys.argv) > 1 else 'data/ch2oo_casscf.npz')
