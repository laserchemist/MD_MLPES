"""The 'ladder' for CH2OO: fragment charge and spin states that pin the asymptotes of ionic diabats.

The zwitterionic structure H2C=O(+)-O(-) dissociates (diabatically) to CH2O(+) + O(-). Its asymptote relative to
the covalent channel CH2O(S0) + O(1D) is

    dE_ion = IE(CH2O) - EA(O) - [E(O 1D) - E(O 3P)]

IE and EA are computed here by delta-CCSD(T) with an augmented basis (diffuse functions are essential for O-).
The O(1D)-O(3P) splitting is a multireference quantity; the experimental value is used by default and reported.
Experimental values (for comparison): IE(CH2O) = 10.88 eV (adiabatic), EA(O) = 1.461 eV, O 1D-3P = 1.967 eV.
"""
from __future__ import annotations

import numpy as np

from ..geometry import ch2oo_xyz
from ..units import HARTREE_EV

EXPERIMENT = {'IE_CH2O_adiabatic': 10.88, 'EA_O': 1.461, 'O_1D_3P': 1.967}


def _ccsd_t(atoms, charge, spin, basis):
    from pyscf import gto, scf, cc
    mol = gto.M(atom=atoms, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = (scf.RHF(mol) if spin == 0 else scf.UHF(mol)).run()
    c = (cc.CCSD(mf) if spin == 0 else cc.UCCSD(mf)).run()
    return mf.e_tot + c.e_corr + c.ccsd_t()


def compute(basis='aug-cc-pvtz', log=print):
    """Vertical IE of the CH2O fragment at the geometry used in the CH2OO scans, and EA of O."""
    frag = [(a, tuple(x)) for i, (a, x) in enumerate(ch2oo_xyz()) if i != 2]     # CH2O: drop the terminal O
    e0 = _ccsd_t(frag, 0, 0, basis)
    e1 = _ccsd_t(frag, 1, 1, basis)
    log('CH2O done')
    o0 = _ccsd_t([('O', (0, 0, 0))], 0, 2, basis)
    om = _ccsd_t([('O', (0, 0, 0))], -1, 1, basis)
    IE, EA = (e1 - e0) * HARTREE_EV, (o0 - om) * HARTREE_EV
    out = {'IE_CH2O_vertical': IE, 'EA_O': EA, 'O_1D_3P': EXPERIMENT['O_1D_3P'], 'basis': basis,
           'method': 'delta-CCSD(T); O 1D-3P experimental'}
    out['dE_ion'] = IE - EA - out['O_1D_3P']
    return out
