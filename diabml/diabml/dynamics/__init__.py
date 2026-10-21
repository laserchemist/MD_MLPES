"""Dynamics on learned matrices: Cartesian surfaces (energies, forces and couplings of every state by automatic
differentiation) and an ASE calculator.

    from diabml.models import load_model
    from diabml.dynamics import CoordinateSurface, DistanceSurface, DiabmlCalculator, ch2oo_coordinates

ase is needed only for DiabmlCalculator (pip install ase, or conda install -c conda-forge ase).
"""
from .surfaces import Surface, CoordinateSurface, DistanceSurface, pair_distances
from .ch2oo import ch2oo_coordinates, ch2oo_atoms, reduced_restraints, RestrainedSurface, CH2OO_SYMBOLS


def __getattr__(name):          # import the ASE calculator lazily so diabml works without ase installed
    if name == 'DiabmlCalculator':
        from .ase_calc import DiabmlCalculator
        return DiabmlCalculator
    raise AttributeError(name)

__all__ = ['Surface', 'CoordinateSurface', 'DistanceSurface', 'pair_distances', 'ch2oo_coordinates', 'ch2oo_atoms',
           'reduced_restraints', 'RestrainedSurface', 'CH2OO_SYMBOLS', 'DiabmlCalculator']
