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
    active       one active-learning round: extend an existing refs.npz where a committee of models disagrees
    explore      one round of active learning from dynamics: trajectories find where the surface becomes uncertain
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
    'nacs': {'enabled': False, 'points': 'cut', 'pairs': [[0, 1], [1, 2], [2, 3]], 'r_max': 4.0,
             'energy_tolerance_eV': 0.01},
    'frame': {'zeta_env': 1.0, 'r_ref': 1.35, 'feature_zetas': [0.6, 1.0, 1.6],
              'ion_wall_r_eq': 1.35, 'ion_wall_rho': 0.3, 'covalent_wall': [2000.0, 0.1],
              # guard: networks frozen outside the data (smooth clamp), plus a wall k (r_w - r)^2 on every structure
              # starting `offset` inside the shortest r_OO of the data: [k (eV/A^2), offset (A)]
              'guard': False, 'guard_margin_r': 0.05, 'guard_soft_r': 0.0125, 'inner_wall': [30.0, 0.1],
              # fragment (CH2O) energy vs r_CO: 'spline' (smooth; parabolas curving up at least asymptote_curv_min
              # eV/A^2 beyond the data) or 'linear' (the old form: kinks at the grid r_CO values, flat outside)
              'asymptote': 'spline', 'asymptote_curv_min': 20.0},
    'train': {
        'models': ['framed', 'cdnn'],
        'exclude_flagged': True,        # drop unconverged / inconsistent-active-space points
        'max_casci_mismatch_hartree': 1e-3,  # NEVPT2 data: drop points whose CASCI states differ from SA-CASSCF
        'sort_states': True,            # train on energies in ascending order (state-specific NEVPT2 can reorder roots)
        'hidden_framed': [128, 64], 'hidden_cdnn': [128, 64], 'coupling_init': 0.1,
        'lr_framed': 2e-3, 'lr_cdnn': 1e-3, 'alpha2': 1e-6,
        'epochs': 6000, 'restarts': 2, 'seeds': [0, 1, 2],
        'random_split': {'enabled': True, 'train_fraction': 0.75, 'val_fraction': 0.125},
        'extrapolation': {'enabled': True, 'r_train_max': 2.3},
        'learning_curve': {'enabled': True, 'n_train': [20, 40, 60, 90], 'test_fraction': 0.2},
        'restraint': {'enabled': False, 'weights': [0.0, 0.1, 1.0, 10.0], 'states': 'all', 'q_ref': 'auto',
                      'min_gap_eV': 0.1},
        'coupling_fit': {'enabled': False, 'weights': [0.0, 0.1, 1.0], 'charge_weight': 1.0, 'min_gap_eV': 0.1,
                         'cap': 5.0},
    },
    'active': {
        'enabled': False,
        'committee': 5, 'bootstrap': 0.9, 'epochs': 4000,
        'n_candidates': 20000, 'r_max': 3.0, 'bounds': {}, 'exclude': [],
        'states': [0, 1, 2, 3], 'state_weights': [1.0, 1.0, 0.5, 0.5],
        'max_S0_eV': 3.0,
        'n_new': 60, 'min_distance': 1.0,
        'length_scales': {'r_OO': 0.1, 'a_COO': 10.0, 'r_CO': 0.05, 'r_CH': 0.05, 'a_HCO': 10.0},
        'nacs': True, 'seed': 0,
    },
    'explore': {
        'enabled': False,
        'r_oo': [1.35, 1.6, 1.9], 'temperatures': [1000.0, 1500.0, 2000.0], 'seeds': 1,   # one trajectory each
        'steps': 3000, 'dt': 0.1, 'state': 0,
        'spread_eV': 0.25, 'data_distance': 1.5, 'jump_meV': 5.0, 'force_max': 25.0, 'lookahead_fs': 5.0,
        'gap_eV': 0.25, 'massey': 0.1,
        'max_new': 40, 'train_after': False,
    },
    'audit': {
        'orca': '',                 # ORCA folder or executable (or --orca, or the ORCA_PATH environment variable)
        'jobs': 6, 'tol_uEh': 50.0, 'retries': 2, 'qd': True, 'maxcore': 2000, 'timeout_s': 3600,
    },
    'dipoles': {
        'jobs': 6,                  # PySCF processes (one thread each)
        'field_au': 0.002,          # finite field (a.u.); 0.001 gives the same dipoles to < 0.001 D
        'tol_uEh': 100.0,           # field-free SA-CASSCF energies must match the data this closely (same solution)
        'retries': 2,               # retries from matched neighbours' orbitals
    },
    'dipole_fit': {
        'members': 5, 'hidden': [64, 64], 'epochs': 3000, 'lr': 3e-3,
        'gap_mEh': 5.0,             # leave out points where S1 - S0 is smaller (finite differences of near-degenerate states)
        'test_fraction': 0.1,       # held-out test set (reported), then the final model is trained on everything
        'final': True, 'r_max': None,
    },
    'ir': {
        'model': 'production.pt', 'dipole': 'dipole.pt',
        'surface': 'production',    # or 'well': a polynomial fitted to the reference S0 energies in the well (low-energy
                                    # spectra; outputs ir_well.html etc. next to the production ones); or 'well9d':
                                    # the 9-D CCSD(T) well, no restraints (outputs ir_9d.html etc.)
        'model9d': 'well9d.npz', 'dipole9d': 'dipole9d.npz',   # the surfaces of surface = "well9d" (in models/)
        'well_order': 3, 'well_window_eV': 0.6,
        'well_source': 'refs',      # 'gs_grid': fit the well to data/gs_grid.npz (smooth ground-state grid) instead
        'zoom_broaden_cm': 3.0,     # broadening of the zoom and band-contour figures (IR3, IR4)
        'contour_align': 'centroid',   # IR4 right panel: band centre on the quantum fundamental; or 'shift'
        'save_every': 10,           # results/ir_*.npz: dipole at every record; x, E_tot, spread every this many (float32)
        'restraint_scale': 1.0,     # stiffness of the CH2 angle/dihedral restraints of a reduced model (x this); 9 = 3x
                                    # their frequencies: a nearly rigid CH2 that no longer mixes with the heavy-atom bands
        'temperatures': [500.0, 1500.0],
        'energies_kJmol': [],       # also microcanonical runs: this vibrational energy above the S0 minimum (e.g. 13)
        'rot_temperature': 300.0,   # rotation of the microcanonical runs (K)
        'trajectories': 6,          # per temperature, run in parallel (jobs)
        'equil_fs': 1000.0,         # Langevin at the temperature (vibration and rotation)
        'run_fs': 5000.0,           # constant energy, recorded: resolution ~ 1 / run time (5 ps -> ~7 cm-1)
        'dt': 0.2, 'every': 2,      # fs; record every 2 steps (0.4 fs: spectrum up to ~40000 cm-1)
        'jobs': 6, 'broaden_cm': 8.0, 'max_cm': 4000.0,
        'min_cm': 200.0,            # normalise and pick peaks above this (below: the rotational band of the tumbling molecule)
        'max_spread_eV': 0.25,      # flag trajectories whose ensemble spread on S0 exceeded this
        'reference': '{diabml}/data/reference/su2013_ch2oo_fig2B.csv',   # experimental spectrum drawn for comparison
        'reference_label': 'expt., Su et al., Science 340, 174 (2013)',
    },
    'track': {
        'box': {},                  # coordinate ranges to track in; empty = r_OO 1.2-1.5, a_COO 105-130, r_CO 1.15-1.40
        'seed': {},                 # coordinates of the starting point; empty = lowest S0 in the box
        'neighbours': 2,            # MORead starts per point: the tree parent + the nearest other tracked points
        'jump_eV': 0.2,             # flag a point whose S1..S(n-2) excitation energies jump more than this from its parent
        'drop_flagged': False,      # True: leave flagged points out (they are only reported by default)
        'jobs': 0,                  # 0: [compute] jobs
    },
    'vib': {                        # quantum vibrational levels on the well polynomial (diabml/workflow/vib_levels.py)
        'source': 'refs',           # 'refs': the run's reference data; 'gs_grid': data/gs_grid.npz (gs_grid stage)
        'order': 3, 'window_eV': 0.6,   # the well fit (as [ir] well_order / well_window_eV)
        'only_audit': [],           # fit only reference points with these audit codes (e.g. [5]: tracked); [] = all
        'points': 28,               # grid points per coordinate
        'ecut_eV': 1.2,             # grid spans the harmonic classically allowed region up to this energy
        'n_states': 16, 'dipole': 'dipole.pt',
        'check_orders': [2, 3, 4], 'check_windows_eV': [0.3, 0.6, 0.9],
    },
    'gs_grid': {                    # dense ground-state grid around the S0 minimum (diabml/workflow/gs_grid.py)
        'center': {},               # {} = minimum of the cubic well polynomial of the run's data
        'step': {'r_OO': 0.03, 'a_COO': 3.0, 'r_CO': 0.025},
        'half_points': 5,           # 2 * half_points + 1 values per coordinate (11^3 = 1331 points)
        'nroots': 1,                # CASSCF roots averaged (1 = state-specific ground state)
        'method': 'nevpt2',         # 'nevpt2': CASSCF(+orbital tracking) + SC-NEVPT2; 'ccsd_t': RHF-CCSD(T), T1 kept
        'basis': '',                # '' = the reference data's basis; e.g. 'aug-cc-pVDZ' for a basis-set check
        'delta': {},                # {run = "../32_tz", order = 2}: add (that grid - this grid), fitted, to this grid
        'extra': {},                # coordinates held fixed in the data to add, e.g. {a_HCO = {center, step, half_points}}
        'neighbours': 2,            # MORead starts per point (tree parent + nearest finished point)
        'jobs': 0,                  # 0: [compute] jobs
    },
    'shell': {                      # CCSD(T) shell beyond the well grid for the Delta model (diabml/workflow/shell.py)
        'box': {'r_OO': [1.15, 1.80], 'a_COO': [98.0, 140.0], 'r_CO': [1.12, 1.46], 'a_HCO': [102.0, 134.0]},
        'step': {'r_OO': 0.05, 'a_COO': 6.0, 'r_CO': 0.04, 'a_HCO': 4.0},
        'window_eV': [0.4, 1.8],    # keep candidates whose energy estimate lies here above the minimum
        'energy_model': '',         # diabatic network for the estimate (e.g. "../24_spec/models/production.pt")
        'n_points': 1200, 'batches': 4,   # farthest-point sampling, then split round-robin: each batch spans the shell
        'select': 'fps',            # 'fps': all batches by farthest-point sampling; 'gp': batch 1 so, then each batch
                                    # where a GP of Delta (CCSD(T) - network) fitted to the batches before is least sure
        'basis': '',                # '' = the gs_grid basis
        'jobs': 0,
        'benchmark': 'benchmark',   # the frozen surface shell_check compares with (models/surfaces/<name>.npz)
    },
    'well9d': {                     # the well in all nine internal coordinates, CCSD(T) (diabml/workflow/well9d.py)
        'center': {},               # {} = the 4-D well minimum (frozen benchmark, else gs_grid) with the C-H of the grid
        'fd_step': {'r': 0.01, 'a': 1.0, 'd': 2.0},   # finite-difference steps: bonds (A), angles, dihedrals (deg)
        'e_max_eV': 0.6,            # candidates: harmonic energy uniform in (0, e_max] (power e_power: E = e_max u^p)
        'e_power': 1.0,
        'd_max': 45.0,              # |dihedral| limit for candidates (deg)
        'n_pool': 20000,            # candidate pool before farthest-point sampling
        'n_points': 3000, 'batches': 6,   # FPS in energy-scaled normal coordinates, split round-robin into batches
        'select': 'fps',            # 'fps', or 'gp': batch 1 by FPS, each later one where a GP of the anharmonic part
                                    # (CCSD(T) - harmonic) of the batches before is least sure
        'gp_pool': 6000,            # candidates (FPS-thinned) the GP scores per batch
        'basis': '',                # '' = the gs_grid basis
        'jobs': 0,
        'seed': 0,
        'fit_orders': [3, 4, 5],    # polynomial orders tried; the one with the lowest held-out (by batch) error is kept
        'fit_window_eV': 0.8,       # fit points within this energy of the lowest one
        'fit_weight_eV': 0.0,       # > 0: weight the fit by E0 / (E0 + E) with E0 = this (Braams-Bowman); 0: unweighted
        'fit_low_eV': 0.8,          # also report the held-out error of the points within this energy (the spectra's region)
        'extend_points': 400,       # well9d_extend: new CCSD(T) points, chosen from the rpmd9d bead samples and a
        'extend_bead_fraction': 0.5,  # harmonic sample (this share from the beads), spread from the computed ones
        'extend_e_min_eV': 0.6,     # candidates at least this high (bead geometries outside the fitted box always count)
        'extend_e_max_eV': 2.0,     # ... and at most this high
        'extend_e_power': 0.5,      # harmonic sample energies E = e_max u^power: < 1 puts more of them near the top
        'extend_pool': 20000,       # candidates kept from each source
        'extend_batches': 2,        # the new points become this many batches (held-out error by batch)
        'extend_tz_points': 50,     # of them also in tz_basis, added to data/well9d_tz.npz (the 9-D correction)
        'extend_refit': True,       # refit at the end (the old surface is kept as models/well9d_pre_extend.npz)
        'extend_fit_window_eV': 1.5,  # the refit's window
        'apply_delta': True,        # add [gs_grid.delta] (e.g. aug-cc-pVTZ - aug-cc-pVDZ) on the 4-D projection
        'benchmark': 'benchmark',   # the frozen 4-D surface to compare with (models/surfaces/<name>.npz)
        'tz_basis': 'aug-cc-pVTZ',  # well9d_tz: the larger basis, at tz_points of the computed points within
        'tz_points': 200,           # tz_e_max_eV, spread by farthest-point sampling
        'tz_e_max_eV': 0.6,
        'tz_orders': [1, 2, 3],     # orders tried for the 9-D correction (held-out error picks one)
        'tz_jobs': 0,
        'vpt2_step': 0.02,          # well9d_vpt2: finite-difference step for the force constants (dimensionless)
        'vpt2_max_gap': 200.0,      # Fermi resonances: |w_b + w_c - w_a| at most this (cm-1) ...
        'vpt2_martin': 1.0,         # ... and coupling^4 / gap^3 at least this (cm-1)
        'dipole_points': 500,       # well9d_dipole: finite-field CCSD(T) dipoles at this many computed points within
        'dipole_e_max_eV': 0.8,     # dipole_e_max_eV, spread by farthest-point sampling (6 runs each, 4 if planar)
        'dipole_field_au': 0.001,
        'dipole_basis': '',         # '' = the 9-D well's basis
        'dipole_orders': [1, 2, 3], # polynomial orders of the dipole surface's coefficients (held-out error picks)
        'dipole_jobs': 0,
        'vci_ecut': 7000.0,         # well9d_vci: basis = modes below 2500 cm-1 to this harmonic excitation energy (cm-1)
        'vci_ch_quanta': 3,         # ... times the two C-H stretches to this many quanta (product basis; 3 -> 4
                                    # changes the CH2OO fundamentals by < 1.5 cm-1)
        'vci_ecut_check': 6000.0,   # a smaller basis for the convergence report (0: none)
        'vci_ch_quanta_check': 3,
        'vci_nmax': 8,              # at most this many quanta in any one mode
        'vci_nmode': 3,             # n-mode representation of the potential (3; 4 adds 4-mode terms to vci_n4max quanta)
        'vci_n4max': 0,
        'vci_extra': 6,             # Gauss-Hermite points per mode beyond the basis size
        'vci_guard_cm': 12000.0,    # grid points this far up (harmonic) where the fit falls below half: harmonic value
        'vci_states_cm': 4500.0,    # levels computed to about this far above the zero-point level (hot populations)
        'vci_max_states': 1500,
        'vci_dipole_nmode': 2,      # n-mode representation of the dipole
        'vci_temperature': 300.0,   # absorption spectrum temperature (K)
        'vci_emission_kJmol': 13.0, # emission: vibrational energy above the zero-point level (kJ/mol), levels
                                    # populated at the vibrational temperature with that mean energy
        'vci_broaden_cm': 12.0,     # Gaussian FWHM of the spectra (cm-1)
        'band_T_rot': 300.0,        # well9d_bands: rotational temperature (K)
        'band_fwhm_cm': 1.0,        # instrument function, Gaussian FWHM (cm-1)
        'band_ground': 'experiment',  # ground-level rotational constants: "experiment" (Chang et al.) or "computed"
        'band_fraction': 0.98,      # rotationally resolve the strongest bands carrying this share of 780-1560 cm-1
        'band_max': 150,            # ... at most this many
        'iso_vci': ['CD2OO', 'CH2(18O)(18O)'],   # well9d_isotopes: species that also get VCI fundamentals
        'iso_ecut': 6000.0,         # ... with this basis cutoff (cm-1; C-H/C-D quanta as vci_ch_quanta)
    },
    'rpmd': {                       # thermostatted ring-polymer MD on the 9-D surface (diabml/workflow/rpmd9.py)
        'beads': [1, 16, 32],       # P = 1: classical MD with the same protocol (the control)
        'temperature': 300.0,       # K
        'trajectories': 64,         # per bead number
        'batch': 16,                # trajectories per process (one batched surface call per step)
        'jobs': 4,                  # processes in parallel
        'torch_threads': 1,
        'equil_fs': 2000.0,         # path-integral MD (PILE-L, every mode thermostatted) before recording
        'centroid_tau_fs': 100.0,   # centroid thermostat time constant during equilibration
        'run_fs': 10000.0,          # TRPMD, recorded (resolution ~ 1.44 / run time: 10 ps -> 4.8 cm-1)
        'dt': 0.25,                 # fs
        'every': 4,                 # record the bead-averaged dipole every 4 steps (1 fs)
        'lam': 0.5,                 # TRPMD internal-mode friction gamma_k = 2 lam omega_k (0: RPMD)
        'lam_scan': [],             # further lam values run at the largest bead number, e.g. [0.0, 0.25, 1.0]
        'bootstrap': 20,            # resamples for the band-centre error bars
        'progress_minutes': 5.0,    # log progress and the expected finish time this often
        'check_points': 40,         # rpmd_check: CCSD(T) at this many bead geometries outside the fitted range ...
        'check_controls': 10,       # ... and this many inside (they fix the energy offset)
        'check_beads': 0,           # the bead number whose sample is checked (0: the largest run)
        'check_jobs': 0,            # ORCA runs in parallel (0: [well9d] jobs)
        'broaden_cm': 0.0,
    },
    'method_cut': {                 # CCSD(T) and FIC-NEVPT2 along r(O-O) cuts of a gs_grid (diabml/workflow/method_cut.py)
        'a_COO': 117.7,             # nearest grid value is used
        'r_CO': [1.244, 1.269, 1.294],
        'methods': ['ccsd_t', 'fic_nevpt2'],   # also 'qd_nevpt2': SA-CASSCF + QD-NEVPT2 (S0 with state mixing)
        'qd_nroots': 3,             # states in the QD-NEVPT2 average
        'jobs': 0,                  # 0: [compute] jobs
    },
    'fill': {
        'box': [],                  # [[fill.box]] lists of grid values per coordinate; empty = the inner-wall default
        'min_distance': 0.4,        # grid steps ([active.length_scales]) from the data and from each other
        'committee': True,          # train the committee first (error before training, T9 / A2)
        'max_S0_eV': None,          # skip candidates whose committee S0 is higher (None: keep all)
        'max_points': 400,
    },
    'compute': {
        'engine': 'pyscf',          # active-learning compute stage: 'pyscf' or 'orca' (see diabml/workflow/orca_compute.py)
        'orca': '',                 # ORCA folder (else [audit] orca, else ORCA_PATH)
        'jobs': 6, 'neighbours': 2, 'scratch': True, 'maxcore': 2000, 'timeout_s': 3600,
    },
    'repair': {
        'extras_tol_mEh': 1.0,      # repaired points closer than this to our solution keep their PySCF charges etc.
        'keep_unverified': True,    # keep points ORCA could not check ('orca higher', 'failed') in the repaired set
    },
    'production': {
        'members': 5, 'test_fraction': 0.1, 'final': True,
        'coupling_weight': 0.1, 'charge_weight': 1.0,
        'low_energy_weight': 0.0,   # training weight 1 + this * exp(-(S0 - min)/weight_width_eV): spectroscopy (e.g. 10)
        'weight_width_eV': 0.3,
        'max_S0_eV': 1.0,                               # dynamics region: S0 at most this far above CH2O + O(1D)
        'corner': {'r_OO': [0.0, 2.0], 'a_COO': [0.0, 70.0]},   # compressed low-angle corner, reported separately
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

[train.restraint]              # diabatic restraint: each state's zwitterion weight follows its CASSCF terminal-O charge
enabled = {restraint}
weights = [0.0, 0.1, 1.0, 10.0]
states = "all"                 # all: S0-S3 | ground: S0 only
q_ref = "auto"                 # charge of a pure zwitterion: "auto" = four-state charge sum at short range, or a number (e)
min_gap_eV = 0.1               # apply only where neighbouring states are this far apart (eigenvector derivatives blow up at degeneracies)

[train.coupling_fit]           # fit the ab initio couplings along r_OO (needs [nacs] points = "all"): makes the matrix diabatic along r
enabled = {coupfit}
weights = [0.0, 0.1, 1.0]
charge_weight = 1.0            # charge restraint applied together with the coupling fit
min_gap_eV = 0.1
cap = 5.0                      # 1/Angstrom: larger couplings (narrow avoided crossings) are left to the energies
'''

_NACS = '''
[nacs]                         # ab initio derivative couplings along r_OO at the equilibrium cut: tests how diabatic the matrix is
enabled = {nacs}
points = "{points}"                 # cut: the equilibrium-angle cut only (a test) | all: every grid point (needed to fit couplings)
pairs = [[0, 1], [1, 2], [2, 3]]
r_max = 4.0                    # skip larger O-O distances (couplings vanish; O(1D) components are degenerate)
energy_tolerance_eV = 0.01     # discard a geometry whose state energies differ from the reference data by more than this
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
''' + _NACS.format(nacs='false', points='cut') + _COMMON_TRAIN.format(epochs=400, restarts=1, seeds='[0]', n_train='[20, 60]', restraint='true', coupfit='false'),

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
''' + _NACS.format(nacs='true', points='all') + _COMMON_TRAIN.format(epochs=6000, restarts=2, seeds='[0, 1, 2]', n_train='[20, 40, 60, 90]', restraint='true', coupfit='true'),

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
''' + _NACS.format(nacs='true', points='all') + _COMMON_TRAIN.format(epochs=6000, restarts=2, seeds='[0, 1, 2]', n_train='[20, 40, 60, 90]', restraint='false', coupfit='false'),

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
''' + _NACS.format(nacs='true', points='cut') + _COMMON_TRAIN.format(epochs=6000, restarts=2, seeds='[0, 1, 2]', n_train='[40, 80, 160, 320]', restraint='false', coupfit='false'),

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
''' + _NACS.format(nacs='false', points='cut') + _COMMON_TRAIN.format(epochs=6000, restarts=2, seeds='[0, 1, 2]', n_train='[20, 40, 60, 90]', restraint='true', coupfit='true'),
    'active': '''# One active-learning round: add reference points where a committee of models disagrees, then retrain.
# Make the next round with a new folder whose refs.file points at this folder's data/refs.npz.
[run]
name = "active"
description = "Active-learning round: new NEVPT2 points where the committee disagrees"
threads = 6

[refs]
source = "file"
file = "../03_cas_3d/data/refs.npz"   # the data to extend (its folder also supplies couplings and orbitals)

[ladder]
compute = false
use = "experiment"

[active]
enabled = true
committee = 5                  # framed models with different seeds, each on a 90 % bootstrap sample
bootstrap = 0.9
epochs = 4000
n_candidates = 20000           # random geometries inside the bounds where the committee is evaluated
r_max = 3.0                    # Angstrom: largest r_OO to propose (beyond it the frame fixes the surface)
states = [0, 1, 2, 3]
state_weights = [1.0, 1.0, 0.5, 0.5]   # S2 and S3 count half: their NEVPT2 is least smooth near crossings
max_S0_eV = 3.0                # skip candidates whose committee S0 is higher (eV above CH2O + O(1D)): not reached in dynamics
n_new = 60                     # geometries per round
min_distance = 1.0             # in units of length_scales, from existing points and from each other
nacs = true                    # also compute couplings along r_OO at the new points (r_OO <= [nacs] r_max)

[active.bounds]                # defaults: the range of the usable data (r_OO capped at r_max)
r_OO = [1.2, 3.0]

[active.length_scales]         # what counts as "one grid step" in each coordinate
r_OO = 0.1
a_COO = 10.0
r_CO = 0.05

# leave out the compressed dioxirane-like corner, where SA-CASSCF has more than one solution; delete to include it
[[active.exclude]]
r_OO = [1.2, 1.5]
a_COO = [65.0, 85.0]
''' + _NACS.format(nacs='false', points='cut') + _COMMON_TRAIN.format(epochs=6000, restarts=2, seeds='[0, 1, 2]', n_train='[90]', restraint='false', coupfit='true')
      .replace('models = ["framed", "cdnn"]', 'models = ["framed"]')
      .replace('[train.extrapolation]          # train only on r_OO <= r_train_max, test beyond it\nenabled = true',
               '[train.extrapolation]          # train only on r_OO <= r_train_max, test beyond it\nenabled = false')
      .replace('[train.learning_curve]         # test error vs number of training points (fixed test set)\nenabled = true',
               '[train.learning_curve]         # test error vs number of training points (fixed test set)\nenabled = false')
      .replace('weights = [0.0, 0.1, 1.0]\ncharge_weight', 'weights = [0.1]\ncharge_weight'),
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


TEMPLATES['explore'] = (TEMPLATES['active']
    .replace('# One active-learning round: add reference points where a committee of models disagrees, then retrain.',
             '# One round of active learning from dynamics: watched trajectories on the committee average stop where the\n'
             '# surface becomes uncertain; those geometries are computed, merged, and the model retrained.')
    .replace('name = "active"', 'name = "explore"')
    .replace('description = "Active-learning round: new NEVPT2 points where the committee disagrees"',
             'description = "Active learning from dynamics: new NEVPT2 points where trajectories found the surface uncertain"')
    .replace('\n[nacs]', '''
[explore]
enabled = true
r_oo = [1.35, 1.6, 1.9]          # starting O-O distances (Angstrom); every start runs at every temperature
temperatures = [1000.0, 1500.0, 2000.0]
seeds = 1                        # trajectories per (r_oo, temperature)
steps = 3000                     # x dt
dt = 0.1                         # fs
state = 0                        # adiabatic state the trajectories run on
spread_eV = 0.25                 # stop: ensemble members disagree this much on the active state (typical in the data: < 0.2)
data_distance = 1.5              # stop: this many grid steps ([active.length_scales]) from every reference point
jump_meV = 5.0                   # stop: total energy changes this much in one step
force_max = 25.0                 # stop: a force above this (eV/Angstrom)
lookahead_fs = 5.0               # after a stop signal, run this much longer to see where it was heading
gap_eV = 0.25                    # log a nonadiabatic region below this gap to a neighbouring state
massey = 0.1                     # ... or Massey parameter hbar|d.v|/dE above this
max_new = 40                     # new reference geometries per round
train_after = false              # all: skip the train and figures stages (the next round trains its own committee)

[nacs]''', 1))

_COMPUTE_ORCA = """
[compute]
engine = "orca"                  # new points with ORCA: lowest SA-CASSCF solution of scratch + neighbour orbitals
neighbours = 2                   # MORead attempts from the nearest points' converged orbitals (orca_orbitals.json)
jobs = 6                         # ORCA runs in parallel (ORCA from ORCA_PATH, or set orca = "/path/to/orca_folder")
"""
TEMPLATES['active'] += _COMPUTE_ORCA
TEMPLATES['explore'] += _COMPUTE_ORCA
