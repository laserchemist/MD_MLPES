"""Run configuration for the CH2OO workflow: one config.toml per run folder.

A run folder holds everything one study produces:

    <run>/config.toml     the only input file (edit it, then run the stages)
    <run>/data/           refs.npz (reference energies + diagnostics), scan_checkpoint.npz, ladder.json
    <run>/models/         trained models: name.pt (weights + JSON spec) and name.json; load with diabml.models.load_model
    <run>/results/        metrics.json, curves.npz (numbers behind every figure and table)
    <run>/figures/        PNG figures
    <run>/tables/         CSV and Markdown tables
    <run>/logs/           one log per stage
    <run>/report.html     figures and tables on one page

Templates (python -m diabml.workflow.ch2oo init <run> --template NAME):
    quick        smoke test on the shipped CASSCF data, small epochs (minutes)
    baseline     shipped CASSCF data, full training settings: reproduces the talk's CH2OO results
    nevpt2_2d    new SA-CASSCF + SC-NEVPT2 / cc-pVDZ scan on the planar (r_OO, angle COO) grid
    cas_3d       adds the C-O1 stretch as a third coordinate (planar 3D grid)
    restraint    trains on an existing refs.npz with a diabatic restraint on the zwitterion structure
"""
from __future__ import annotations

import copy
from pathlib import Path

try:                       # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

PACKAGE_ROOT = Path(__file__).resolve().parents[2]      # the folder holding pyproject.toml and data/

DEFAULTS = {
    'run': {'name': '', 'description': '', 'threads': 8},
    'refs': {
        'source': 'compute',            # compute | file
        'file': '',                     # a RefSet .npz when source = "file" ({diabml} = package root)
        'basis': '6-31g*', 'ncas': 6, 'nelecas': 8, 'nroots': 4,
        'spin': 'singlet',              # singlet | triplet
        'dynamic': 'none',              # none | nevpt2
        'energy': 'casscf',             # which energies the models learn: casscf | nevpt2
        'max_cycle': 100,
        'grid': {'r_OO': [1.20, 1.275, 1.35, 1.425, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.15, 2.30, 2.50, 2.75,
                          3.00, 3.50, 4.00, 5.00, 6.50, 8.00],
                 'a_COO': [65.0, 85.0, 100.0, 117.7, 135.0, 155.0]},
    },
    'ladder': {'compute': False, 'basis': 'aug-cc-pvtz', 'use': 'experiment'},
    'frame': {'zeta_env': 1.0, 'r_ref': 1.35, 'feature_zetas': [0.6, 1.0, 1.6],
              'ion_wall_r_eq': 1.35, 'ion_wall_rho': 0.3, 'covalent_wall': [2000.0, 0.1]},
    'train': {
        'models': ['framed', 'cdnn'],
        'exclude_flagged': True,        # drop unconverged / inconsistent-active-space points
        'hidden_framed': [128, 64], 'hidden_cdnn': [128, 64], 'coupling_init': 0.1,
        'lr_framed': 2e-3, 'lr_cdnn': 1e-3, 'alpha2': 1e-6,
        'epochs': 6000, 'restarts': 2, 'seeds': [0, 1, 2],
        'random_split': {'enabled': True, 'train_fraction': 0.75, 'val_fraction': 0.125},
        'extrapolation': {'enabled': True, 'r_train_max': 2.3},
        'learning_curve': {'enabled': True, 'n_train': [20, 40, 60, 90], 'test_fraction': 0.2},
        'restraint': {'enabled': False, 'weights': [0.0, 0.1, 1.0, 10.0], 'q_ref': -0.6, 'min_gap_eV': 0.1},
    },
}

_COMMON_TRAIN = '''
[frame]
zeta_env = 1.0                 # overlap envelope exponent (1/bohr): how fast learned corrections die off with r_OO
r_ref = 1.35                   # envelope = 1 at this O-O distance (Angstrom)
feature_zetas = [0.6, 1.0, 1.6]
ion_wall_r_eq = 1.35           # Born-Mayer wall on the ion pair, balanced against the Coulomb pull at this r_OO (Angstrom)
ion_wall_rho = 0.3             # its range (Angstrom)
covalent_wall = [2000.0, 0.1]  # A (eV), rho (Angstrom) for the other structures: ~0.01 eV at 1.2 Angstrom, steep below

[train]
models = ["framed", "cdnn"]    # framed = ChannelFrame + small networks; cdnn = plain CDNN (Shu et al. 2025)
exclude_flagged = true         # skip unconverged points and points whose active space changed character
hidden_framed = [128, 64]
hidden_cdnn = [128, 64]
coupling_init = 0.1            # eV; framed couplings must not start at exactly 0 (their energy gradient vanishes there)
lr_framed = 2e-3
lr_cdnn = 1e-3
alpha2 = 1e-6                  # L2 weight penalty (eq. 5 of the CDNN paper)
epochs = {epochs}
restarts = {restarts}
seeds = {seeds}

[train.random_split]           # interpolation test: random train / validation / test split of all usable points
enabled = true
train_fraction = 0.75
val_fraction = 0.125

[train.extrapolation]          # train only on r_OO <= r_train_max, test beyond it
enabled = true
r_train_max = 2.3

[train.learning_curve]         # test error vs number of training points (fixed test set)
enabled = true
n_train = {n_train}
test_fraction = 0.2

[train.restraint]              # diabatic restraint: ground-state weight of the ionic structure follows the CASSCF charge
enabled = {restraint}
weights = [0.0, 0.1, 1.0, 10.0]
q_ref = -0.6                   # terminal-O Mulliken charge taken to mean "fully zwitterionic" (an assumption: vary it)
min_gap_eV = 0.1               # apply only where S1 - S0 exceeds this (eigenvector derivatives blow up at degeneracies)
'''

TEMPLATES = {
    'quick': '''# Smoke test: shipped SA4-CASSCF(8,6)/6-31G* data, short training. Takes a few minutes.
[run]
name = "quick"
description = "Smoke test of the workflow on the shipped CASSCF data"
threads = 8

[refs]
source = "file"
file = "{diabml}/data/ch2oo_casscf.npz"

[ladder]
compute = false
use = "experiment"
''' + _COMMON_TRAIN.format(epochs=400, restarts=1, seeds='[0]', n_train='[20, 60]', restraint='true'),

    'baseline': '''# Baseline: the shipped SA4-CASSCF(8,6)/6-31G* data with the talk's training settings.
[run]
name = "baseline"
description = "Shipped CASSCF data; reproduces the CH2OO results in the talk"
threads = 8

[refs]
source = "file"
file = "{diabml}/data/ch2oo_casscf.npz"

[ladder]
compute = false
use = "experiment"
''' + _COMMON_TRAIN.format(epochs=6000, restarts=2, seeds='[0, 1, 2]', n_train='[20, 40, 60, 90]', restraint='false'),

    'nevpt2_2d': '''# Step 1: dynamic correlation. SA4-CASSCF(8,6) + SC-NEVPT2 with cc-pVDZ on the planar 2D grid (120 points).
[run]
name = "nevpt2_2d"
description = "SA4-CASSCF(8,6)/cc-pVDZ + SC-NEVPT2, planar (r_OO, angle COO)"
threads = 8                    # PySCF and PyTorch threads; use the number of performance cores

[refs]
source = "compute"
basis = "cc-pvdz"
ncas = 6                       # active space: C-O-O pi (3), O-O sigma/sigma*, terminal-O in-plane lone pair
nelecas = 8
nroots = 4                     # singlet A' states averaged with equal weights
spin = "singlet"
dynamic = "nevpt2"             # none | nevpt2 (state-specific SC-NEVPT2 on the SA-CASSCF orbitals)
energy = "nevpt2"              # which energies the models learn
max_cycle = 100

[refs.grid]                    # one list per coordinate; a single value holds that coordinate fixed
r_OO = [1.20, 1.275, 1.35, 1.425, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.15, 2.30, 2.50, 2.75, 3.00, 3.50, 4.00, 5.00, 6.50, 8.00]
a_COO = [65.0, 85.0, 100.0, 117.7, 135.0, 155.0]
r_CO = [1.272]

[ladder]
compute = true                 # delta-CCSD(T) IE(CH2O) and EA(O); about 10-30 min
basis = "aug-cc-pvtz"
use = "experiment"             # experiment | computed: which ion-pair asymptote the frame uses
''' + _COMMON_TRAIN.format(epochs=6000, restarts=2, seeds='[0, 1, 2]', n_train='[20, 40, 60, 90]', restraint='false'),

    'cas_3d': '''# Step 2: add the C-O1 stretch. The zwitterion (H2C=O+ -O-) and biradical (H2C*-O-O*) structures prefer
# different C-O lengths, so r_CO is the next coordinate that changes state character. 20 x 6 x 4 = 480 points.
[run]
name = "cas_3d"
description = "SA4-CASSCF(8,6)/cc-pVDZ + SC-NEVPT2, planar (r_OO, angle COO, r_CO)"
threads = 8

[refs]
source = "compute"
basis = "cc-pvdz"
ncas = 6
nelecas = 8
nroots = 4
spin = "singlet"
dynamic = "nevpt2"
energy = "nevpt2"
max_cycle = 100

[refs.grid]
r_OO = [1.20, 1.275, 1.35, 1.425, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.15, 2.30, 2.50, 2.75, 3.00, 3.50, 4.00, 5.00, 6.50, 8.00]
a_COO = [65.0, 85.0, 100.0, 117.7, 135.0, 155.0]
r_CO = [1.20, 1.272, 1.35, 1.45]

[ladder]
compute = false
use = "experiment"
''' + _COMMON_TRAIN.format(epochs=6000, restarts=2, seeds='[0, 1, 2]', n_train='[40, 80, 160, 320]', restraint='false'),

    'restraint': '''# Step 3: diabatic restraint. Reuse the references of an earlier run and tie the zwitterion structure's
# weight in the ground state to the computed terminal-O charge.
[run]
name = "restraint"
description = "Framed model with a charge-based diabatic restraint on the ionic structure"
threads = 8

[refs]
source = "file"
file = "../02_nevpt2_2d/data/refs.npz"  # relative to this run folder: the step-2 references

[ladder]
compute = false
use = "experiment"
''' + _COMMON_TRAIN.format(epochs=6000, restarts=2, seeds='[0, 1, 2]', n_train='[20, 40, 60, 90]', restraint='true'),
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(run_dir) -> dict:
    run_dir = Path(run_dir)
    with open(run_dir / 'config.toml', 'rb') as fh:
        user = tomllib.load(fh)
    cfg = _merge(DEFAULTS, user)
    if 'grid' in user.get('refs', {}):            # a grid in the file replaces the default grid entirely
        cfg['refs']['grid'] = user['refs']['grid']
    return cfg


def resolve(path_str, run_dir) -> Path:
    s = str(path_str).replace('{diabml}', str(PACKAGE_ROOT))
    p = Path(s).expanduser()
    return p if p.is_absolute() else (Path(run_dir) / p).resolve()
