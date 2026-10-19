"""diabml: physics-framed learned matrices for coupled potential energy surfaces.

Layers
  refs/      reference engines (FCI, SA-CASSCF scans with state-character diagnostics, ladder fragments)
  refset     RefSet: one .npz per data set; notebooks never need quantum chemistry to load data
  frames/    physics frames: London (H3), covalent DIM (H2O), fragment-channel (CH2OO and other A + B channels)
  models     CDNN (Shu et al. 2025), independent-surface NN, FramedModel, KRR, one shared training loop
  metrics    errors, seam gaps, permutation errors
  plotting   standard figures;  ui  optional interactive cut explorer
"""
from .refset import RefSet
from .units import HARTREE_EV, EV_KCAL

__version__ = '0.1.0'
