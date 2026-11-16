"""CH2OO workflow: reference scans -> ladder -> training -> figures -> tables -> report, one run folder per study.

    python -m diabml.workflow.ch2oo init     RUN --template nevpt2_2d     # make RUN/ and RUN/config.toml
    python -m diabml.workflow.ch2oo refs     RUN     # SA-CASSCF (+ NEVPT2) scan; resumable, checkpoint every point
    python -m diabml.workflow.ch2oo status   RUN     # progress and time per point of a running scan
    python -m diabml.workflow.ch2oo ladder   RUN     # fragment IE / EA (delta-CCSD(T)) for the ion-pair asymptote
    python -m diabml.workflow.ch2oo nacs     RUN     # ab initio derivative couplings along r_OO (test of diabaticity)
    python -m diabml.workflow.ch2oo train    RUN     # framed CDNN vs CDNN: random split, extrapolation, learning curve,
                                                     # optional diabatic restraint
    python -m diabml.workflow.ch2oo figures  RUN
    python -m diabml.workflow.ch2oo tables   RUN
    python -m diabml.workflow.ch2oo report   RUN     # RUN/report.html
    python -m diabml.workflow.ch2oo all      RUN     # refs, ladder, train, figures, tables, report

Active learning (template "active", see diabml/workflow/active.py): one run folder per round,
    python -m diabml.workflow.ch2oo propose  RUN     # committee of models; geometries where they disagree most
    python -m diabml.workflow.ch2oo compute  RUN     # new references there, at the level of the base data
    python -m diabml.workflow.ch2oo merge    RUN     # base + new -> data/refs.npz
    (all = refs, ladder, propose, compute, merge, train, figures, tables, report)

Active learning from dynamics (template "explore", see diabml/workflow/explore.py):
    python -m diabml.workflow.ch2oo explore  RUN     # watched trajectories; flag where the surface becomes uncertain
    (all = refs, ladder, explore, compute, merge, train, figures, tables, report; plus RUN/dynamics.html)

Fill a region with a grid of new points (see diabml/workflow/fill.py), in an active/explore run folder:
    python -m diabml.workflow.ch2oo fill     RUN     # grid in [[fill.box]] minus points near the data -> proposed.npz
    (then compute and merge as for any round)

Audit of the reference data with ORCA (see diabml/workflow/audit.py):
    python -m diabml.workflow.ch2oo audit RUN --orca /path/to/orca_folder --jobs 6   # -> RUN/audit.html
    python -m diabml.workflow.ch2oo repair RUN    # -> RUN/data/repaired/{nevpt2,qd_nevpt2}/ (diabml/workflow/repair.py)

Follow one SA-CASSCF solution through a region (see diabml/workflow/track.py):
    python -m diabml.workflow.ch2oo track      RUN --jobs 6   # -> RUN/data/tracked/refs.npz (smooth near the minimum)

Dipoles and IR (see diabml/workflow/dipoles.py and ir.py):
    python -m diabml.workflow.ch2oo dipoles    RUN --jobs 6   # finite-field dipoles at every point -> data/dipoles.npz
    python -m diabml.workflow.ch2oo dipole_fit RUN            # dipole surface -> models/dipole.pt
    python -m diabml.workflow.ch2oo ir         RUN --jobs 6   # hot-molecule IR from dynamics -> RUN/ir.html
    python -m diabml.workflow.ch2oo gs_grid    RUN --jobs 6   # ground-state CASSCF+NEVPT2 grid at the minimum (ORCA)
    python -m diabml.workflow.ch2oo method_cut RUN --jobs 8   # CCSD(T) and FIC-NEVPT2 along r(O-O) cuts of the grid
    python -m diabml.workflow.ch2oo freeze     RUN --name benchmark   # fix the validated well surface (a switch)
    python -m diabml.workflow.ch2oo shell      RUN --batch 1          # CCSD(T) shell beyond the well, one batch
    python -m diabml.workflow.ch2oo shell_check RUN                   # benchmark error and Delta on the shell
    python -m diabml.workflow.ch2oo well9d     RUN --batch 1 [-v]     # 9-D CCSD(T) well: Hessian, then sampled batches
    python -m diabml.workflow.ch2oo well9d_fit RUN [-v]               # 9-D polynomial, harmonic modes, checks
    python -m diabml.workflow.ch2oo well9d_tz  RUN [-v]               # aug-cc-pVTZ subset: 9-D basis correction
    python -m diabml.workflow.ch2oo well9d_vpt2 RUN [-v]              # anharmonic fundamentals, Fermi resonances
    python -m diabml.workflow.ch2oo well9d_dipole RUN [-v]            # finite-field dipoles -> 9-D dipole surface
    python -m diabml.workflow.ch2oo well9d_vci RUN [-v]               # variational levels, absorption and emission
    python -m diabml.workflow.ch2oo well9d_bands RUN                  # rotationally resolved bands from the VCI levels
    python -m diabml.workflow.ch2oo well9d_isotopes RUN [-v]          # isotopologues: microwave constants, shifts
    python -m diabml.workflow.ch2oo well9d_compare RUN                # classical dynamics vs VCI on the same surface
    python -m diabml.workflow.ch2oo rpmd_plan RUN                     # TRPMD planning: beads, resolution, run time
    python -m diabml.workflow.ch2oo rpmd9d RUN [-v]                   # TRPMD spectra (P = 1, 16, 32) on the 9-D surface
    python -m diabml.workflow.ch2oo rpmd_check RUN [--jobs N] -v       # CCSD(T) where the beads leave the fitted range
    python -m diabml.workflow.ch2oo well9d_extend RUN [--jobs N] -v    # new CCSD(T) to 2 eV from the beads, then refit
    python -m diabml.workflow.ch2oo dynamics_report RUN               # dynamics_9d.html: trajectory diagnostics
    python -m diabml.workflow.ch2oo vci_report RUN                    # vci_9d.html: results and full methods
    python -m diabml.workflow.ch2oo ir RUN    ([ir] surface = "well9d") # spectra on the 9-D well, no restraints
    python -m diabml.workflow.ch2oo ir_replot  RUN            # redraw IR3/IR4 and ir.html from the saved spectra
    python -m diabml.workflow.ch2oo vib_levels RUN            # quantum levels on the well polynomial -> RUN/vib_levels.html

Production surface (see diabml/workflow/production.py):
    python -m diabml.workflow.ch2oo production RUN   # averaged ensemble -> models/production.pt; T10, P1, P2

Everything a stage needs is read from RUN/config.toml and from files earlier stages wrote, so stages can be rerun
one at a time (e.g. change training settings and rerun train, figures, tables, report without touching the scan).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import config as C

AXES = ['r_OO', 'a_COO', 'r_CO', 'r_CH', 'a_HCO']                   # allowed grid coordinates, in this order
KW = {'r_OO': 'r_oo', 'a_COO': 'a_coo', 'r_CO': 'r_co', 'r_CH': 'r_ch', 'a_HCO': 'a_hco'}
UNITS = {'r_OO': 'Å', 'a_COO': '°', 'r_CO': 'Å', 'r_CH': 'Å', 'a_HCO': '°'}
LABEL = {'r_OO': 'r(O–O)', 'a_COO': '∠C–O–O', 'r_CO': 'r(C–O)', 'r_CH': 'r(C–H)', 'a_HCO': '∠H–C–O'}
ION = 4                                                             # index of the ionic structure in the frame
EV_KCAL = 23.060548


# ------------------------------------------------------------------------------------------------ plumbing
class Run:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        for d in ('data', 'models', 'results', 'figures', 'tables', 'logs'):
            (self.path / d).mkdir(parents=True, exist_ok=True)
        self.cfg = C.load(self.path)
        self._log = None
        self.verbose = False            # --verbose: detail() lines also go to the terminal

    def p(self, *parts):
        return self.path.joinpath(*parts)

    def open_log(self, stage):
        self._log = open(self.p('logs', f'{stage}.log'), 'a')
        self.log(f'==== {stage}  ({time.strftime("%Y-%m-%d %H:%M:%S")})  run: {self.path}')

    def log(self, msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        if self._log:
            self._log.write(line + '\n'); self._log.flush()

    def detail(self, msg):
        """A detail line: always written to the stage log, printed to the terminal only with --verbose."""
        line = f'[{time.strftime("%H:%M:%S")}]   {msg}'
        if self.verbose:
            print(line, flush=True)
        if self._log:
            self._log.write(line + '\n'); self._log.flush()

    def threads(self):
        n = int(self.cfg['run'].get('threads', 8))
        os.environ.setdefault('OMP_NUM_THREADS', str(n))
        try:
            import torch
            torch.set_num_threads(n)
        except Exception:
            pass
        try:
            from pyscf import lib
            lib.num_threads(n)
        except Exception:
            pass
        return n


def _reference_values():
    from ..geometry import CH2OO_REF
    return {'r_OO': CH2OO_REF['r_oo'], 'a_COO': CH2OO_REF['a_coo'], 'r_CO': CH2OO_REF['r_co'],
            'r_CH': CH2OO_REF['r_ch'], 'a_HCO': CH2OO_REF['a_hco']}


def load_refs(run):
    from ..refset import RefSet
    path = run.p('data', 'refs.npz')
    if not path.exists():
        sys.exit(f'{path} not found: run the refs stage first')
    ref = RefSet.load(path)
    if run.cfg['train'].get('sort_states', True):
        ref = adiabatic_order(ref)
    return ref


PER_STATE = ('q_O_terminal', 'n_unpaired')        # E_casscf_eV / E_nevpt2_eV stay per SA-CASSCF root


def adiabatic_order(ref):
    """Put every row's state energies in ascending order, the order of the model's eigenvalues.

    State-specific SC-NEVPT2 corrects each SA-CASSCF root separately, and where roots are close the corrected
    energies can come out in a different order (S1 below S0, S3 below S2, ...). A model's eigenvalues are always
    sorted, so an unsorted target column cannot be fitted. Per-state diagnostics move with their energies;
    extra['state_order'][i, k] is the SA-CASSCF root that became state k (for remapping couplings)."""
    E = np.asarray(ref.energies, float)
    n = E.shape[1]
    P = np.argsort(np.where(np.isfinite(E), E, np.inf), 1, kind='stable')
    ref.extra['state_order'] = P
    if not np.any(P != np.arange(n)):
        return ref
    rows = np.arange(len(E))[:, None]
    ref.energies = E[rows, P]
    for k in PER_STATE:
        v = ref.extra.get(k)
        if v is not None and np.ndim(v) == 2 and v.shape[1] >= n:
            v = np.array(v, float)
            v[:, :n] = v[rows, P]
            ref.extra[k] = v
    moved = np.any(P != np.arange(n), 1)
    ref.meta = dict(ref.meta, n_reordered=int(moved.sum()),
                    largest_reorder_eV=float(np.nanmax(np.abs(ref.energies - E))))
    return ref


def slice_values(ref):
    """For every coordinate except r_OO: the grid value closest to the CH2OO equilibrium (where cuts are drawn)."""
    rv = _reference_values()
    out = {}
    for j, name in enumerate(ref.coord_names):
        if name == 'r_OO':
            continue
        vals = np.unique(ref.coords[grid_mask(ref), j])
        out[name] = float(vals[np.argmin(np.abs(vals - rv.get(name, vals.mean())))])
    return out


def cut_mask(ref, values=None):
    values = slice_values(ref) if values is None else values
    m = np.ones(len(ref), bool)
    for j, name in enumerate(ref.coord_names):
        if name in values:
            m &= np.isclose(ref.coords[:, j], values[name])
    return m


def cut_coords(ref, x, values=None):
    values = slice_values(ref) if values is None else values
    cols = [x if name == 'r_OO' else np.full_like(x, values[name]) for name in ref.coord_names]
    return np.stack(cols, 1)


# ------------------------------------------------------------------------------------------------ init
def stage_init(path, template, refs=None):
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    cfg = path / 'config.toml'
    if cfg.exists():
        print(f'{cfg} exists; leaving it unchanged')
    else:
        if template not in C.TEMPLATES:
            sys.exit(f'unknown template {template}; choose from {sorted(C.TEMPLATES)}')
        text = C.TEMPLATES[template]
        if refs:
            line = f'file = "{refs}"'
            if re.search(r'^file = ".*?"', text, re.M):
                text = re.sub(r'^file = ".*?"', lambda m: line, text, count=1, flags=re.M)
            else:
                text = re.sub(r'^source = "compute"', lambda m: f'source = "file"\n{line}', text, count=1, flags=re.M)
        cfg.write_text(text)
        print(f'wrote {cfg} (template "{template}"' + (f', refs.file = "{refs}")' if refs else ')'))
    Run(path)
    print('folders: data/ models/ results/ figures/ tables/ logs/')


# ------------------------------------------------------------------------------------------------ refs
def _grid(cfg):
    grid = cfg['refs']['grid']
    unknown = set(grid) - set(AXES)
    if unknown:
        sys.exit(f'unknown grid coordinates {sorted(unknown)}; allowed: {AXES}')
    if 'r_OO' not in grid:
        sys.exit('the grid needs r_OO (the dissociation coordinate)')
    rv = _reference_values()
    axes = [a for a in AXES if a in grid]
    values = [sorted(float(v) for v in np.atleast_1d(grid[a])) for a in axes]
    fixed = {a: rv[a] for a in AXES if a not in grid}
    fixed.update({a: v[0] for a, v in zip(axes, values) if len(v) == 1})
    varying = [a for a, v in zip(axes, values) if len(v) > 1]
    pts = np.array(list(itertools.product(*values)), float)
    return axes, pts, varying, fixed


def stage_refs(run, force=False):
    from ..refset import RefSet
    cfg = run.cfg['refs']
    out = run.p('data', 'refs.npz')
    if run.cfg['active'].get('enabled', False):
        from . import active
        return active.setup_refs(run)
    if cfg['source'] == 'file':
        src = C.resolve(cfg['file'], run.path)
        if not src.exists():
            hint = ''
            if src.name == 'refs.npz' and src.parent.name == 'data':
                hint = (f' It is written by the refs stage of the run in {src.parent.parent}: finish that run first '
                        f'(python -m diabml.workflow.ch2oo refs {src.parent.parent}),')
            sys.exit(f'refs.file in {run.p("config.toml")} points to {src}, which does not exist.{hint} or set '
                     f'refs.file to another refs.npz (e.g. "{{diabml}}/data/ch2oo_casscf.npz" for the shipped CASSCF data).')
        ref = RefSet.load(src)
        ref.meta = dict(ref.meta, copied_from=str(src))
        ref.save(out)
        sib = src.parent / 'nacs.npz'
        if sib.exists():
            shutil.copy(sib, run.p('data', 'nacs.npz'))
            run.log(f'copied {sib} (ab initio couplings for F11/T7)')
        run.log(f'copied {src} -> {out}: {ref}')
        return ref
    if out.exists() and not force:
        run.log(f'{out} exists (use --force to recompute); nothing to do')
        return RefSet.load(out)
    from ..geometry import ch2oo_xyz
    from ..refs.casscf import CASSpec, scan
    from ..units import HARTREE_EV
    n = run.threads()
    axes, pts, varying, fixed = _grid(run.cfg)
    spin_ss = {'singlet': 0.0, 'triplet': 2.0}[cfg['spin']]
    spec = CASSpec(ncas=int(cfg['ncas']), nelecas=int(cfg['nelecas']), nroots=int(cfg['nroots']), basis=cfg['basis'],
                   symmetry='Cs', wfnsym="A'", spin_ss=spin_ss, max_cycle=int(cfg['max_cycle']))
    dynamic = None if cfg['dynamic'] in (None, '', 'none') else cfg['dynamic']
    run.log(f'{len(pts)} geometries over {axes} (varying {varying}; fixed {fixed}); {spec}; dynamic={dynamic}; '
            f'{n} threads')
    # start at the point nearest equilibrium so the first orbitals are the well-behaved ones
    rv = _reference_values()
    scale = np.array([1.0 if a.startswith('r') else 50.0 for a in axes])
    i0 = int(np.argmin(np.sum(((pts - [rv[a] for a in axes]) / scale) ** 2, 1)))
    perm = np.r_[i0, np.delete(np.arange(len(pts)), i0)]

    def atoms_fn(*p):
        return ch2oo_xyz(**{KW[a]: v for a, v in zip(axes, p)})

    R = scan(pts[perm], atoms_fn, spec, fragment_atoms=(2,), log=run.log,
             checkpoint=str(run.p('data', 'scan_checkpoint.npz')), dynamic=dynamic)
    inv = np.argsort(perm)
    R = {k: v[inv] for k, v in R.items()}
    # zero of energy: lowest state at the largest r_OO, other coordinates nearest equilibrium (CH2O + O(1D))
    far = np.isclose(pts[:, 0], pts[:, 0].max())
    for j, a in enumerate(axes[1:], 1):
        vals = np.unique(pts[:, j])
        far &= np.isclose(pts[:, j], vals[np.argmin(np.abs(vals - rv[a]))])
    levels = {'casscf': R['E_hartree']}
    if dynamic == 'nevpt2':
        levels['nevpt2'] = R['E_nevpt2_hartree']
    zeros = {k: float(np.nanmin(v[far, 0])) for k, v in levels.items()}
    ev = {k: (v - zeros[k]) * HARTREE_EV for k, v in levels.items()}
    use = cfg['energy'] if cfg['energy'] in ev else 'casscf'
    cols = [axes.index(a) for a in varying]
    extra = {'q_O_terminal': R['charge'], 'n_unpaired': R['n_unpaired'], 'converged': R['converged'],
             'active_space_consistent': R['active_space_consistent'], 'seconds': R['seconds'],
             'E_casscf_eV': ev['casscf']}
    if 'nevpt2' in ev:
        extra['E_nevpt2_eV'] = ev['nevpt2']
        extra['casci_mismatch_hartree'] = R['casci_mismatch_hartree']
    method = f"SA{spec.nroots}-CASSCF({spec.nelecas},{spec.ncas})/{spec.basis}" + (' + SC-NEVPT2' if dynamic else '')
    ref = RefSet(f'CH2OO {method} ({cfg["spin"]}s)', pts[:, cols], ev[use], varying, extra=extra,
                 meta={'method': method, 'energy_level': use, 'spin': cfg['spin'], 'fixed': fixed,
                       'zero': 'lowest state at the largest r_OO, other coordinates nearest equilibrium',
                       'zero_hartree': zeros, 'grid_axes': axes, 'spec': spec.__dict__})
    ref.save(out)
    run.log(f'saved {out}: {ref}; converged {int(R["converged"].sum())}/{len(pts)}, '
            f'consistent active space {int(R["active_space_consistent"].sum())}/{len(pts)}, '
            f'{R["seconds"].sum() / 3600:.2f} h of compute')
    return ref


def stage_status(run):
    ck = run.p('data', 'scan_checkpoint.npz')
    if not ck.exists():
        print('no scan checkpoint yet'); return
    z = np.load(ck)
    done, sec = z['done'], z['seconds']
    n, k = len(done), int(done.sum())
    per = sec[done].mean() if k else float('nan')
    print(f'{k}/{n} points done; {per:.1f} s per point; about {(n - k) * per / 60:.0f} min left; '
          f'{int((~z["converged"][done]).sum())} unconverged so far')


# ------------------------------------------------------------------------------------------------ ladder
def stage_ladder(run):
    from ..refs import ladder as L
    lc = run.cfg['ladder']
    X = L.EXPERIMENT
    out = {'experiment': dict(X), 'dE_ion_experiment': X['IE_CH2O_adiabatic'] - X['EA_O'] - X['O_1D_3P']}
    old = run.p('data', 'ladder.json')
    if old.exists():
        prev = json.loads(old.read_text())
        if 'computed' in prev and prev['computed'].get('basis') == lc['basis']:
            out['computed'] = prev['computed']
    if lc['compute'] and 'computed' not in out:
        run.threads()
        run.log(f'delta-CCSD(T)/{lc["basis"]}: CH2O, CH2O+, O(3P), O-')
        out['computed'] = L.compute(basis=lc['basis'], log=run.log)
    out['use'] = lc['use'] if (lc['use'] == 'experiment' or 'computed' in out) else 'experiment'
    out['dE_ion_used'] = out['computed']['dE_ion'] if out['use'] == 'computed' else out['dE_ion_experiment']
    out['r_cross_A'] = 14.399645 / out['dE_ion_used']
    old.write_text(json.dumps(out, indent=1))
    run.log(f'ion-pair asymptote {out["dE_ion_used"]:.3f} eV ({out["use"]}); Coulomb crossing r* = '
            f'{out["r_cross_A"]:.2f} Å')
    return out


def _dE_ion(run):
    path = run.p('data', 'ladder.json')
    if not path.exists():
        run.log('no ladder.json: using experimental IE, EA and O(1D)-O(3P) (run the ladder stage to record them)')
        return stage_ladder(run)['dE_ion_used']
    return json.loads(path.read_text())['dE_ion_used']


# ------------------------------------------------------------------------------------------------ nacs
def _geometry_kwargs(ref, coords_row):
    """ch2oo_xyz keyword arguments for one row of ref.coords plus the run's fixed coordinates."""
    kw = {}
    for k, v in ref.meta.get('fixed', {}).items():
        kw[KW.get(k, k.lower())] = float(v)
    for name, v in zip(ref.coord_names, coords_row):
        kw[KW[name]] = float(v)
    return kw


def _ref_casscf_relative(ref):
    """Reference SA-CASSCF state energies relative to S0 (eV). The couplings come from SA-CASSCF, so the check that
    they belong to the same states must compare SA-CASSCF with SA-CASSCF, even when the run learns NEVPT2 energies."""
    E = ref.extra.get('E_casscf_eV') if hasattr(ref, 'extra') and ref.extra else None
    E = np.asarray(ref.energies if E is None else E, float)
    return E - E[:, [0]]


def _check_and_write_nacs(run, ref, want, pairs, tau, vec, Es, conv, nroots):
    """Compare the NAC run's SA-CASSCF energies with the reference SA-CASSCF energies, flag mismatches and write
    data/nacs.npz. tau_all keeps every computed value; tau has the mismatched geometries set to NaN."""
    from ..units import HARTREE_EV
    nc = run.cfg['nacs']
    rr = ref.coords[want, 0]
    zero = float(np.nanmin(Es[np.argmax(rr), :]))
    Ec = (Es - Es[:, [0]]) * HARTREE_EV
    Er = _ref_casscf_relative(ref)[want][:, :nroots]
    mismatch = np.nanmax(np.abs(Ec - Er), 1)
    tol = float(nc.get('energy_tolerance_eV', 0.01))
    bad = ~(mismatch <= tol) | ~conv
    tau_all = np.array(tau, float)
    tau = tau_all.copy()
    if bad.any():
        tau[bad] = np.nan
        run.log(f'  {int(bad.sum())}/{len(want)} geometries discarded: unconverged, or SA-CASSCF state energies differ '
                f'from the reference SA-CASSCF energies by more than {tol * 1000:.0f} meV (a different solution); worst '
                f'{np.nanmax(mismatch):.3f} eV')
    else:
        run.log(f'  all {len(want)} geometries reproduce the reference SA-CASSCF state energies (largest difference '
                f'{np.nanmax(mismatch) * 1000:.2f} meV)')
    np.savez(run.p('data', 'nacs.npz'), r=rr, idx=want, coords=ref.coords[want], pairs=np.array(pairs), tau=tau,
             tau_all=tau_all, vectors=vec, E_eV=(Es - zero) * HARTREE_EV, converged=conv, energy_mismatch_eV=mismatch,
             cut=json.dumps(slice_values(ref)))
    run.log(f'wrote data/nacs.npz ({int((~bad).sum())}/{len(want)} geometries usable)')


def _tau_from_vectors(ref, want, vec):
    """|projection of each coupling vector on the O1->O2 unit vector| (recovers tau from the stored vectors)."""
    from ..geometry import ch2oo_xyz
    tau = np.full(vec.shape[:2], np.nan)
    for k, i in enumerate(want):
        xyz = np.array([x for _, x in ch2oo_xyz(**_geometry_kwargs(ref, ref.coords[i]))])
        e = (xyz[2] - xyz[1]) / np.linalg.norm(xyz[2] - xyz[1])
        tau[k] = np.abs(vec[k, :, 2] @ e)
    return tau


def stage_nacs(run):
    """Ab initio derivative couplings along the O-O stretch (SA-CASSCF, PySCF), at the cut or at every grid point.

    Couplings only mean something for the same electronic states as the reference data, and SA-CASSCF can converge to
    different solutions depending on the starting orbitals. So the SA-CASSCF is redone along exactly the path the
    reference scan took (start nearest equilibrium, greedy nearest neighbour, orbitals from the nearest finished point),
    or, when the run's own scan checkpoint is present, from the reference orbitals at each geometry. Every result is
    then checked against the reference state energies and discarded if they differ (energy_tolerance_eV).

    For each requested pair the derivative coupling vector d_kl = <k|grad l> is projected on the O1->O2 unit vector of
    the terminal O: tau_kl along r_OO (1/Angstrom). Phases are arbitrary, so magnitudes are stored. Resumable: results
    and orbitals are written to data/nacs_checkpoint.npz after every geometry."""
    from ..geometry import ch2oo_xyz
    from ..refs.casscf import CASSpec, run_sa_casscf, _visit_order, _mol as _mol_for
    from ..units import BOHR_ANG, HARTREE_EV
    from pyscf.nac import sacasscf as NAC
    nc = run.cfg['nacs']
    if not nc.get('enabled', False):
        run.log('nacs disabled in config.toml ([nacs] enabled = false)'); return
    n = run.threads()
    ref = load_refs(run)
    cfg = run.cfg['refs']
    spin_ss = {'singlet': 0.0, 'triplet': 2.0}[cfg['spin']]
    spec = CASSpec(ncas=int(cfg['ncas']), nelecas=int(cfg['nelecas']), nroots=int(cfg['nroots']), basis=cfg['basis'],
                   symmetry='Cs', wfnsym="A'", spin_ss=spin_ss, max_cycle=int(cfg['max_cycle']))
    pairs = [tuple(int(i) for i in p) for p in nc['pairs']]
    sel = cut_mask(ref) if nc.get('points', 'cut') == 'cut' else np.ones(len(ref), bool)
    if nc.get('r_max'):
        sel &= ref.coords[:, 0] <= float(nc['r_max'])
    want = np.where(sel)[0]                                  # where couplings are computed
    N = len(ref)
    # the reference scan's path over all its points (same construction as stage_refs / refs.ch2oo.generate)
    rv = _reference_values()
    axes = ref.coord_names
    scale = np.array([1.0 if a.startswith('r') else 50.0 for a in axes])
    i0 = int(np.argmin(np.sum(((ref.coords - [rv.get(a, 0.0) for a in axes]) / scale) ** 2, 1)))
    perm = np.r_[i0, np.delete(np.arange(N), i0)]
    zc = (ref.coords[perm] - ref.coords[perm].mean(0)) / (ref.coords[perm].std(0) + 1e-12)
    path = [int(perm[k]) for k in _visit_order(zc)]          # reference-point indices in visiting order
    zpt = np.empty_like(zc); zpt[perm] = zc                  # standardised coords indexed by reference point
    # reference orbitals, if this run made the scan itself
    ref_mo = None
    sck = run.p('data', 'scan_checkpoint.npz')
    if sck.exists():
        z = np.load(sck)
        if 'mo' in z.files and z['points'].shape[0] == N:
            pts_full = z['points']
            # the scan stored points in its own (permuted) order with all grid axes; map them to reference rows
            cols = [list(ref.meta.get('grid_axes', axes)).index(a) for a in axes] if 'grid_axes' in ref.meta else None
            if cols is not None:
                key = {tuple(np.round(p[cols], 6)): i for i, p in enumerate(pts_full)}
                rows = [key.get(tuple(np.round(c, 6))) for c in ref.coords]
                if all(r_ is not None for r_ in rows):
                    ref_mo = z['mo'][rows]
                    run.log('  starting every geometry from the reference scan\'s own orbitals')
    nsel = len(want)
    old = run.p('data', 'nacs.npz')
    if old.exists() and not run.p('data', 'nacs_checkpoint.npz').exists():
        z = np.load(old)
        if (np.array_equal(z['idx'], want) and np.array_equal(z['pairs'], np.array(pairs)) and 'vectors' in z.files
                and np.isfinite(z['vectors']).any()):
            run.log('data/nacs.npz already holds the computed couplings: re-checking them against the reference '
                    '(delete data/nacs.npz to recompute from scratch)')
            from ..units import HARTREE_EV
            vec = z['vectors']
            tau = z['tau_all'] if 'tau_all' in z.files else _tau_from_vectors(ref, want, vec)
            _check_and_write_nacs(run, ref, want, pairs, tau, vec, z['E_eV'] / HARTREE_EV, z['converged'], spec.nroots)
            return
    pos = {int(i): k for k, i in enumerate(want)}
    tau = np.full((nsel, len(pairs)), np.nan)
    vec = np.full((nsel, len(pairs), 5, 3), np.nan)
    E = np.full((N, spec.nroots), np.nan)
    conv = np.zeros(N, bool)
    done = np.zeros(N, bool)
    mos = None
    ck = run.p('data', 'nacs_checkpoint.npz')
    if ck.exists():
        z = np.load(ck)
        if np.array_equal(z['want'], want) and np.array_equal(z['pairs'], np.array(pairs)) and len(z['done']) == N:
            tau, vec, E, conv, done = z['tau'], z['vectors'], z['E_hartree'], z['converged'], z['done']
            mos = z['mo'] if 'mo' in z.files else None
            run.log(f'  resuming: {int(done.sum())}/{N} geometries already done')
    visited = [i for i in path if done[i]]
    live = {}

    def guess_for(i):
        if i in live:
            return live[i]
        return mos[i], _mol_for(ch2oo_xyz(**_geometry_kwargs(ref, ref.coords[i])), spec)

    def checkpoint():
        tmp = str(ck) + '.tmp.npz'
        np.savez(tmp, want=want, pairs=np.array(pairs), tau=tau, vectors=vec, E_hartree=E, converged=conv, done=done,
                 **({} if mos is None else {'mo': mos}))
        os.replace(tmp, ck)

    # only walk the path as far as needed: up to the last point where couplings are wanted
    last = max(path.index(int(i)) for i in want)
    todo = [i for i in path[:last + 1] if not done[i]]
    run.log(f'{nsel} geometries for couplings ({nc.get("points", "cut")}), pairs {pairs}; SA-CASSCF along the reference '
            f'path at {len(todo)} geometries; {spec}; {n} threads')
    t0 = time.time()
    n_off = 0
    ref_rel = _ref_casscf_relative(ref)
    for step, i in enumerate(todo):
        atoms = ch2oo_xyz(**_geometry_kwargs(ref, ref.coords[i]))
        if ref_mo is not None:
            guess = (ref_mo[i], _mol_for(atoms, spec))
        elif visited:
            j = min(visited, key=lambda m: float(np.sum((zpt[m] - zpt[i]) ** 2)))
            guess = guess_for(j)
        else:
            guess = None
        mol, mc = run_sa_casscf(atoms, spec, guess)
        if not mc.converged and visited:                   # same retry as the reference scan
            mol, mc = run_sa_casscf(atoms, spec, guess_for(visited[0]))
        live[i] = (mc.mo_coeff, mol)
        if mos is None:
            mos = np.full((N,) + mc.mo_coeff.shape, np.nan)
        mos[i] = mc.mo_coeff
        E[i], conv[i] = mc.e_states, mc.converged
        mm = float(np.max(np.abs((E[i] - E[i, 0]) * HARTREE_EV - ref_rel[i, :spec.nroots])))
        n_off += mm > float(nc.get('energy_tolerance_eV', 0.01))
        if i in pos:
            k = pos[i]
            xyz = np.array([x for _, x in atoms])
            e = (xyz[2] - xyz[1]) / np.linalg.norm(xyz[2] - xyz[1])
            nac = NAC.NonAdiabaticCouplings(mc)
            nac.verbose = 0
            for c, pr in enumerate(pairs):
                v = np.asarray(nac.kernel(state=pr)) / BOHR_ANG          # 1/bohr -> 1/Angstrom
                vec[k, c] = v
                tau[k, c] = abs(float(v[2] @ e))
        done[i] = True
        visited.append(i)
        checkpoint()
        if step % 10 == 0 or step == len(todo) - 1:
            run.log(f'  {step + 1}/{len(todo)}  {np.round(ref.coords[i], 3)}  '
                    f'{"|tau| " + str(np.round(tau[pos[i]], 3)) if i in pos else "(path only)"}  energy vs reference '
                    f'{mm * 1000:.2f} meV; {n_off} off so far  {time.time() - t0:.0f} s')
    _check_and_write_nacs(run, ref, want, pairs, tau, vec, E[want], conv[want], spec.nroots)
    ck.unlink(missing_ok=True)


def coupling_comparison(run, ref):
    """Model |tau_kl| along r_OO at the ab initio points and on a fine grid, for every saved framed/CDNN model."""
    from ..models import load_model
    path = run.p('data', 'nacs.npz')
    if not path.exists():
        return None
    z = np.load(path)
    pairs = [tuple(int(x) for x in p) for p in z['pairs']]
    sv = json.loads(str(z['cut']))
    on_cut = cut_mask(ref, sv)[z['idx']]
    if on_cut.sum() < 2:
        return None
    o = np.argsort(z['r'][on_cut])
    xr = z['r'][on_cut][o]
    xf = np.linspace(max(1.15, xr.min()), min(4.0, xr.max()), 300)
    out = {'r': xr, 'tau_ref': z['tau'][on_cut][o], 'E_ref': z['E_eV'][on_cut][o], 'pairs': pairs, 'x': xf, 'models': {}}
    names = (['random_framed', 'random_cdnn'] + sorted(p.stem for p in run.p('models').glob('restraint_w*.pt'))
             + sorted(p.stem for p in run.p('models').glob('coupfit_w*.pt')))
    for nm in names:
        f = run.p('models', f'{nm}.pt')
        if not f.exists():
            continue
        try:
            m = load_model(f)
        except Exception as exc:                     # an old-format file: skip it, say why
            run.log(f'coupling comparison: cannot load {f.name} ({exc.__class__.__name__}); retrain to include it')
            continue
        Tp = model_couplings_r(m, cut_coords(ref, xr, sv))
        Tf = model_couplings_r(m, cut_coords(ref, xf, sv))
        out['models'][nm] = {'at_ref': np.array([np.abs(Tp[:, a, b]) for a, b in pairs]).T,
                             'fine': np.array([np.abs(Tf[:, a, b]) for a, b in pairs]).T}
    return out


# ------------------------------------------------------------------------------------------------ frame
def asymptote_table(ref, dE_ion, form='spline', curv_min=20.0):
    """Asymptotes of the five CH2OO structures (an AsymptoteTable).

    States 0-2 (CH2O + O(1D)) and 3 (CH2O(T1) + O(3P)) come from the reference data at the largest r_OO, averaged
    over the angle; if r_CO varies they are interpolated in r_CO (the CH2O fragment energy depends on it). The ion
    pair CH2O(+) + O(-) sits dE_ion (ladder) above the mean of the three O(1D) components. form / curv_min: see
    AsymptoteTable ([frame] asymptote, asymptote_curv_min; 'spline' keeps forces continuous in r_CO and the C-O bond
    bound beyond the data)."""
    from ..frames import AsymptoteTable
    r = ref.coords[:, 0]
    far = np.isclose(r, r.max())
    j = ref.coord_names.index('r_CO') if 'r_CO' in ref.coord_names else None
    if j is None:
        knots, vals = [0.0], np.nanmean(ref.energies[far, :4], 0)[None, :]
    else:
        knots = np.unique(ref.coords[far, j])
        vals = np.array([np.nanmean(ref.energies[far & np.isclose(ref.coords[:, j], k), :4], 0) for k in knots])
    return AsymptoteTable(knots, vals, j=j, extra=[(4, [0, 1, 2], float(dE_ion))], form=form, curv_min=curv_min)


def build_frame(run, ref):
    from ..frames import ChannelFrame, Diabat
    fc = run.cfg['frame']
    dE = _dE_ion(run)
    asy = asymptote_table(ref, dE, fc.get('asymptote', 'spline'), float(fc.get('asymptote_curv_min', 20.0)))
    c = asy.vals.mean(0)
    cA, crho = fc.get('covalent_wall', [2000.0, 0.1])
    rho_ion = float(fc.get('ion_wall_rho', 0.3))
    A_ion = Diabat.born_mayer_balance(float(fc.get('ion_wall_r_eq', 1.35)), -1.0, rho_ion)
    diabats = [Diabat('CH₂O + O(¹D) a', c[0], wall_A=cA, wall_rho=crho),
               Diabat('CH₂O + O(¹D) b', c[1], wall_A=cA, wall_rho=crho),
               Diabat('CH₂O + O(¹D) c', c[2], wall_A=cA, wall_rho=crho),
               Diabat('CH₂O(T₁) + O(³P)', c[3], kind='biradical', wall_A=cA, wall_rho=crho),
               Diabat('CH₂O⁺ + O⁻', c[:3].mean() + dE, charge_product=-1.0, kind='ionic', wall_A=A_ion, wall_rho=rho_ion)]
    fr = ChannelFrame(diabats, n_coords=len(ref.coord_names), r_index=0, zeta_env=fc['zeta_env'],
                      r_ref=fc['r_ref'], feature_zetas=tuple(fc['feature_zetas']), asymptote_fn=asy)
    if fc.get('guard', True):
        # beyond the data the networks are frozen (smooth clamp) and a wall on every structure supplies the
        # short-range repulsion; without it a trajectory that compresses r_OO below the data can fall into a hole
        X = ref.coords[np.all(np.isfinite(ref.energies[:, :4]), 1)]
        lo, hi = X.min(0), X.max(0)
        width = np.maximum(hi - lo, 1e-6)
        g_lo = [float(lo[0]) - float(fc.get('guard_margin_r', 0.05))] + [float(v) for v in lo[1:]]
        g_hi = [None] + [float(v) for v in hi[1:]]                   # r_OO: the envelope handles large r
        soft = [float(fc.get('guard_soft_r', 0.0125))] + [0.02 * float(w) for w in width[1:]]
        k, off = fc.get('inner_wall', [30.0, 0.1])
        fr.set_guard(g_lo, g_hi, soft, wall_r=float(lo[0]) + float(off), wall_k=float(k))
    return fr


def make_model(run, ref, kind):
    from ..models import CDNN, FramedModel
    tc = run.cfg['train']
    if kind == 'framed':
        # couplings start at coupling_init (eV), not 0: with every coupling exactly 0 the matrix is diagonal and the
        # energy gradient with respect to a coupling vanishes, so the coupling network would never learn
        return FramedModel(build_frame(run, ref), 4, hidden=tuple(tc['hidden_framed']),
                           slot_bias={'coup': float(tc.get('coupling_init', 0.1))})
    if kind == 'cdnn':
        return CDNN(len(ref.coord_names), 4, n_matrix=5, hidden=tuple(tc['hidden_cdnn']))
    raise ValueError(kind)


def ionic_weights(model, coords, n=4):
    """Weight of the ionic (zwitterion) structure in each of the model's lowest n adiabatic states: (N, n)."""
    M = model.matrix(coords).detach().numpy()
    _, V = np.linalg.eigh(M)
    return V[:, ION, :n] ** 2


def ionic_weight(model, coords):
    return ionic_weights(model, coords)[:, 0]


def q_ion_auto(ref, usable, r_max=1.5):
    """Terminal-O charge of a pure zwitterion, from the data: the four-state sum of terminal-O charges at short
    range, where the ionic structure lies inside the four computed states (median over r_OO <= r_max)."""
    q = np.asarray(ref.extra['q_O_terminal'], float)[:, :4]
    qs = q.sum(1)                                   # NaN where a point has no charges (e.g. repaired from ORCA)
    m = usable & (ref.coords[:, 0] <= r_max) & np.isfinite(qs)
    return float(np.median(qs[m]))


def charge_targets(ref, idx, q_ion, min_gap, states='all'):
    """Targets for the ionic weight of each state, q_k / q_ion (clipped to 0..1), and where to apply them.

    states = 'all': S0-S3 at points where every neighbouring reference gap exceeds min_gap (eigenvector derivatives
    diverge at degeneracies); 'ground': S0 only, where S1 - S0 > min_gap. Returns (targets (N, 4), mask (N, 4))."""
    q = ref.extra['q_O_terminal'][idx, :4]
    t = np.clip(q / q_ion, 0.0, 1.0)
    gaps = np.diff(ref.energies[idx, :4], axis=1)
    mask = np.zeros_like(t, bool)
    if states == 'ground':
        mask[:, 0] = gaps[:, 0] > min_gap
    else:
        mask[:] = np.all(gaps > min_gap, 1)[:, None]
    mask &= np.isfinite(t)                          # no charges at this point: no restraint
    return np.nan_to_num(t), mask


def restraint_penalty(weight, target, mask):
    import torch
    rows = mask.any(1)
    t = torch.tensor(target[rows])
    m = torch.tensor(mask[rows])
    r = torch.tensor(rows)

    def pen(model, cache):
        if weight == 0 or not m.any():
            return torch.zeros(())
        M = model._matrix(cache)[r]
        _, V = torch.linalg.eigh(M)
        return weight * ((V[:, ION, :4] ** 2 - t) ** 2)[m].mean()
    return pen


def load_nacs(run, ref):
    """Ab initio |tau_kl| along r_OO aligned with the reference points: (tau (N, n_pairs) with NaN where not computed,
    pairs, E_eV (N, n_states) of the coupling calculation)."""
    path = run.p('data', 'nacs.npz')
    if not path.exists():
        return None
    z = np.load(path)
    pairs = [tuple(int(x) for x in p) for p in z['pairs']]
    raw = np.full((len(ref), len(pairs)), np.nan)
    raw[z['idx']] = z['tau']
    P = ref.extra.get('state_order')
    if P is None:
        return raw, pairs
    # couplings were computed between SA-CASSCF roots; state k (energy order) is root P[i, k]
    look = {frozenset(p): c for c, p in enumerate(pairs)}
    tau = np.full_like(raw, np.nan)
    for c, (a, b) in enumerate(pairs):
        src = [look.get(frozenset((int(pa), int(pb)))) for pa, pb in zip(P[:, a], P[:, b])]
        for i, j in enumerate(src):
            if j is not None:
                tau[i, c] = raw[i, j]
    return tau, pairs


def coupling_targets(ref, idx, tau, pairs, min_gap, cap):
    """Which ab initio couplings to fit: computed, states more than min_gap apart, |tau| below cap (narrow
    avoided-crossing spikes are left to the energies, which already fix them)."""
    T = tau[idx]
    gaps = np.stack([ref.energies[idx, b] - ref.energies[idx, a] for a, b in pairs], 1)
    mask = np.isfinite(T) & (np.abs(gaps) > min_gap) & (T < cap)
    return np.nan_to_num(T), mask


def coupling_penalty(weight, coords, target, mask, pairs, h=1e-3, delta=0.02, huber=1.0):
    """weight * mean (|tau_model| - |tau_ref|)^2 over the masked training couplings, with tau_model from a central
    difference of the (differentiable) matrix along r_OO. Robust form: tau_model = G dE / (dE^2 + delta^2) (delta in
    eV) so a model gap that closes during training cannot blow up, and misfits beyond `huber` (1/Angstrom) count
    linearly instead of quadratically."""
    import torch
    rows = mask.any(1)
    X = torch.tensor(np.asarray(coords, float)[rows])
    Xp, Xm = X.clone(), X.clone()
    Xp[:, 0] += h; Xm[:, 0] -= h
    t, m, r = torch.tensor(target[rows]), torch.tensor(mask[rows]), torch.tensor(rows)
    a = torch.tensor([p[0] for p in pairs]); b = torch.tensor([p[1] for p in pairs])

    def pen(model, cache):
        if weight == 0 or not m.any():
            return torch.zeros(())
        M = model._matrix(cache)[r]
        dM = (model.matrix_t(Xp) - model.matrix_t(Xm)) / (2 * h)
        w, V = torch.linalg.eigh(M)
        G = torch.einsum('nik,nij,njl->nkl', V, dM, V)
        n = torch.arange(len(w))[:, None]
        dw = w[n, b[None, :]] - w[n, a[None, :]]
        # the reference states are > min_gap apart here, but the model's need not be during training: a plain
        # 1/dw would send the coupling (and its gradient) to infinity and wreck the fit, so the denominator is
        # regularised (|tau| <= 1/(2 delta) * |G|) and large misfits count linearly (Huber)
        tau = G[n, a[None, :], b[None, :]] * dw / (dw ** 2 + delta ** 2)
        d = (tau.abs() - t)[m]
        return weight * torch.where(d.abs() < huber, 0.5 * d ** 2 / huber, d.abs() - 0.5 * huber).mean() * 2 * huber
    return pen


def model_couplings_r(model, coords, h=1e-4, n=4):
    """Derivative couplings of the model's adiabatic states along r_OO (column 0), tau_kl = v_k^T (dOmega/dr) v_l /
    (E_l - E_k), in 1/Angstrom: (N, n, n). Signs are arbitrary (eigenvector phases), so compare magnitudes."""
    if getattr(model, 'mode', None) == 'adiabatic' and hasattr(model, 'members'):
        # ensemble: magnitudes averaged over members (each member's phases are its own)
        return np.mean([np.abs(model_couplings_r(m, coords, h, n)) for m in model.members], 0)
    X = np.atleast_2d(np.asarray(coords, float))
    Xp, Xm = X.copy(), X.copy()
    Xp[:, 0] += h; Xm[:, 0] -= h
    M = model.matrix(X).detach().numpy()
    dM = (model.matrix(Xp).detach().numpy() - model.matrix(Xm).detach().numpy()) / (2 * h)
    w, V = np.linalg.eigh(M)
    V = V[:, :, :n]
    G = np.einsum('nik,nij,njl->nkl', V, dM, V)
    dE = w[:, None, :n] - w[:, :n, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        tau = G / dE
    tau[:, np.arange(n), np.arange(n)] = 0.0
    return tau


def usable_points(run, ref, log=True):
    """Indices of reference points the models train on: finite energies and, with exclude_flagged, converged, same
    active space, and (NEVPT2 data) CASCI states that reproduce SA-CASSCF."""
    tc = run.cfg['train']
    usable = np.all(np.isfinite(ref.energies[:, :4]), 1)
    if tc['exclude_flagged']:
        for k in ('converged', 'active_space_consistent'):
            if k in ref.extra:
                usable &= ref.extra[k].astype(bool)
        # NEVPT2 is computed on CASCI states built from the SA-CASSCF orbitals; if CASCI does not reproduce the
        # SA-CASSCF energies, the NEVPT2 energies belong to different states and the point is not usable
        if 'casci_mismatch_hartree' in ref.extra and ref.meta.get('energy_level') == 'nevpt2':
            cm = np.nan_to_num(np.asarray(ref.extra['casci_mismatch_hartree'], float), nan=0.0)
            bad = cm > float(tc.get('max_casci_mismatch_hartree', 1e-3))
            if bad.any() and log:
                run.log(f'  excluding {int((usable & bad).sum())} points where CASCI does not reproduce SA-CASSCF '
                        f'(> {1000 * float(tc.get("max_casci_mismatch_hartree", 1e-3)):.1f} mEh): '
                        f'{np.round(ref.coords[bad], 3).tolist()}')
            usable &= ~bad
    return np.where(usable)[0]


def grid_mask(ref):
    """Rows on the original scan grid (active-learning points are off-grid and are left out of grid figures)."""
    return ref.extra['al_round'] == 0 if 'al_round' in ref.extra else np.ones(len(ref), bool)


# ------------------------------------------------------------------------------------------------ train
def stage_train(run):
    import torch
    from ..models import train, save_model
    from .. import metrics as Mx
    run.threads()
    ref = load_refs(run)
    tc = run.cfg['train']
    U = usable_points(run, ref)
    usable = np.zeros(len(ref), bool); usable[U] = True
    run.log(f'{ref}; {len(U)} usable points; models {tc["models"]}')
    r = ref.coords[:, 0]
    E = ref.energies[:, :4]
    kw = dict(epochs=int(tc['epochs']), restarts=int(tc['restarts']), alpha2=float(tc['alpha2']))
    lr = {'framed': float(tc['lr_framed']), 'cdnn': float(tc['lr_cdnn'])}
    metrics = {'n_points': int(len(ref)), 'n_usable': int(len(U)), 'coord_names': ref.coord_names,
               'method': ref.meta.get('method', ref.name), 'config': tc}
    x = np.linspace(max(1.15, r.min() - 0.05), r.max(), 400)
    X = cut_coords(ref, x)
    xs = np.linspace(0.6, 1.6, 200)                 # below the data: is there a hole?
    curves = {'x': x, 'x_short': xs, 'cut_values': json.dumps(slice_values(ref))}

    def save(m, name, experiment, tr):
        save_model(m, run.p('models', f'{name}.pt'), meta={
            'run': str(run.path), 'experiment': experiment, 'seed': int(tc['seeds'][0]), 'n_train': int(len(tr)),
            'coord_names': ref.coord_names, 'coord_units': [UNITS[c] for c in ref.coord_names],
            'energy_unit': 'eV', 'energy_zero': ref.meta.get('zero', ''), 'method': ref.meta.get('method', ref.name),
            'fixed_coordinates': ref.meta.get('fixed', {}), 'atom_order': ['C', 'O1 (central)', 'O2 (terminal)', 'H', 'H'],
            'states': 'lowest 4 eigenvalues = S0..S3 (singlet A\'); matrix order = ' + ', '.join(
                ['CH2O+O(1D) a', 'b', 'c', 'CH2O(T1)+O(3P)', 'CH2O(+)+O(-)'])})

    def fit(kind, tr, va, seed, penalty=None):
        torch.manual_seed(seed)
        m = make_model(run, ref, kind)
        t0 = time.time()
        m = train(m, ref.coords[tr], E[tr], ref.coords[va], E[va], lr=lr[kind], penalty=penalty, **kw)
        return m, time.time() - t0

    # ---- random split (interpolation)
    rs = tc['random_split']
    if rs['enabled']:
        res = {k: [] for k in tc['models']}
        for seed in tc['seeds']:
            p = np.random.default_rng(seed).permutation(U)
            ntr, nva = int(rs['train_fraction'] * len(p)), int(rs['val_fraction'] * len(p))
            tr, va, te = p[:ntr], p[ntr:ntr + nva], p[ntr + nva:]
            for kind in tc['models']:
                m, dt = fit(kind, tr, va, seed)
                res[kind].append({'seed': seed, 'test_rmse_kcal': Mx.rmse_kcal(m.predict(ref.coords[te]), E[te]).tolist(),
                                  'val_rmse_kcal': Mx.rmse_kcal(m.predict(ref.coords[va]), E[va]).tolist(),
                                  'n_train': int(len(tr)), 'seconds': dt})
                run.log(f'random split seed {seed} {kind}: test RMSE {np.round(res[kind][-1]["test_rmse_kcal"], 2)} '
                        f'kcal/mol ({dt:.0f} s)')
                if seed == tc['seeds'][0]:
                    save(m, f'random_{kind}', 'random split', tr)
                    curves[f'random_{kind}_E'] = m.predict(X)
                    curves[f'random_{kind}_short'] = m.predict(cut_coords(ref, xs))
                    curves[f'random_{kind}_test_pred'] = m.predict(ref.coords[te])
                    curves['random_test_idx'] = te
                    if kind == 'framed':
                        M = m.matrix(X).detach().numpy()
                        curves['random_framed_diag'] = np.diagonal(M, axis1=1, axis2=2)
                        iu = np.triu_indices(5, 1)
                        curves['random_framed_coup'] = M[:, iu[0], iu[1]]
                        curves['random_framed_ionic_weight'] = ionic_weights(m, X)
        metrics['random_split'] = res
    # ---- extrapolation
    ex = tc['extrapolation']
    if ex['enabled']:
        inner = U[r[U] <= ex['r_train_max']]
        outer = U[r[U] > ex['r_train_max']]
        res = {k: [] for k in tc['models']}
        for seed in tc['seeds']:
            p = np.random.default_rng(seed).permutation(inner)
            tr, va = p[:int(0.8 * len(p))], p[int(0.8 * len(p)):]
            for kind in tc['models']:
                m, dt = fit(kind, tr, va, seed)
                res[kind].append({'seed': seed, 'outer_rmse_kcal': Mx.rmse_kcal(m.predict(ref.coords[outer]), E[outer]).tolist(),
                                  'inner_val_rmse_kcal': Mx.rmse_kcal(m.predict(ref.coords[va]), E[va]).tolist(),
                                  'seconds': dt})
                run.log(f'extrapolation seed {seed} {kind}: beyond {ex["r_train_max"]} Å RMSE '
                        f'{np.round(res[kind][-1]["outer_rmse_kcal"], 1)} kcal/mol')
                if seed == tc['seeds'][0]:
                    save(m, f'extrap_{kind}', f'trained on r_OO <= {ex["r_train_max"]}', tr)
                    curves[f'extrap_{kind}_E'] = m.predict(X)
        metrics['extrapolation'] = dict(res, r_train_max=ex['r_train_max'], n_inner=int(len(inner)),
                                        n_outer=int(len(outer)))
    # ---- learning curve
    lc = tc['learning_curve']
    if lc['enabled']:
        p = np.random.default_rng(12345).permutation(U)
        nte = int(lc['test_fraction'] * len(p))
        te, pool = p[:nte], p[nte:]
        res = {k: {} for k in tc['models']}
        for n in lc['n_train']:
            nva = max(5, min(n // 5, len(pool) - n))
            if n + nva > len(pool):
                run.log(f'learning curve: skipping n={n} (only {len(pool)} points outside the test set)')
                continue
            for kind in tc['models']:
                res[kind][str(n)] = []
                for seed in tc['seeds']:
                    q = np.random.default_rng(1000 + seed).permutation(pool)
                    m, _ = fit(kind, q[:n], q[n:n + nva], seed)
                    res[kind][str(n)].append(Mx.rmse_kcal(m.predict(ref.coords[te]), E[te]).tolist())
                run.log(f'learning curve n={n} {kind}: mean test RMSE '
                        f'{np.mean(res[kind][str(n)]):.2f} kcal/mol')
        metrics['learning_curve'] = dict(res, n_test=int(nte))
    # ---- diabatic restraint
    rc = tc['restraint']
    if rc['enabled'] and 'framed' in tc['models']:
        if 'q_O_terminal' not in ref.extra:
            run.log('restraint skipped: the reference data have no q_O_terminal')
        else:
            res = {}
            states = rc.get('states', 'all')
            q_ion = q_ion_auto(ref, usable) if str(rc['q_ref']) == 'auto' else float(rc['q_ref'])
            run.log(f'restraint on {states} states; charge of a pure zwitterion q_ion = {q_ion:.3f} e'
                    + (' (from the data: four-state sum at r_OO <= 1.5 Å)' if str(rc['q_ref']) == 'auto' else ''))
            eqv = cut_mask(ref) & usable
            for wgt in rc['weights']:
                res[str(wgt)] = []
                for seed in tc['seeds']:
                    p = np.random.default_rng(seed).permutation(U)
                    ntr = int(tc['random_split']['train_fraction'] * len(p))
                    nva = int(tc['random_split']['val_fraction'] * len(p))
                    tr, va, te = p[:ntr], p[ntr:ntr + nva], p[ntr + nva:]
                    tt, mt = charge_targets(ref, tr, q_ion, rc['min_gap_eV'], states)
                    m, dt = fit('framed', tr, va, seed, penalty=restraint_penalty(float(wgt), tt, mt))
                    te_t, te_m = charge_targets(ref, te, q_ion, rc['min_gap_eV'], states)
                    iw = ionic_weights(m, ref.coords[te])
                    res[str(wgt)].append({
                        'seed': seed, 'test_rmse_kcal': Mx.rmse_kcal(m.predict(ref.coords[te]), E[te]).tolist(),
                        'ionic_weight_mae': float(np.mean(np.abs(iw[te_m] - te_t[te_m]))) if te_m.any() else None,
                        'ionic_weight_at_eq': ionic_weights(m, cut_coords(ref, np.array([1.35])))[0].tolist()})
                    if seed == tc['seeds'][0]:
                        curves[f'restraint_{wgt}_ionic_weight'] = ionic_weights(m, X)
                        save(m, f'restraint_w{wgt}', f'restraint weight {wgt} ({states} states, q_ion {q_ion:.3f} e)', tr)
                run.log(f'restraint weight {wgt}: energy RMSE '
                        f'{np.round(np.mean([v["test_rmse_kcal"] for v in res[str(wgt)]], 0), 2)} kcal/mol, '
                        f'ionic-weight MAE {np.mean([v["ionic_weight_mae"] or np.nan for v in res[str(wgt)]]):.3f}')
            metrics['restraint'] = dict(res, q_ref=q_ion, states=states, min_gap_eV=rc['min_gap_eV'],
                                        target_cut=np.clip(ref.extra['q_O_terminal'][eqv, :4] / q_ion, 0, 1).tolist(),
                                        target_cut_r=r[eqv].tolist())
    # ---- fit the couplings themselves (with the charge restraint), test on held-out couplings
    cf = tc.get('coupling_fit', {})
    nac = load_nacs(run, ref)
    if cf.get('enabled', False) and 'framed' in tc['models']:
        if nac is None:
            run.log('coupling fit skipped: no data/nacs.npz (run the nacs stage with points = "all")')
        else:
            tau, pairs = nac
            q_ion = q_ion_auto(ref, usable) if str(rc['q_ref']) == 'auto' else float(rc['q_ref'])
            res = {}
            for wc in cf['weights']:
                res[str(wc)] = []
                for seed in tc['seeds']:
                    p = np.random.default_rng(seed).permutation(U)
                    ntr = int(tc['random_split']['train_fraction'] * len(p))
                    nva = int(tc['random_split']['val_fraction'] * len(p))
                    tr, va, te = p[:ntr], p[ntr:ntr + nva], p[ntr + nva:]
                    tt, mt = charge_targets(ref, tr, q_ion, rc['min_gap_eV'], rc.get('states', 'all'))
                    ct, cm_ = coupling_targets(ref, tr, tau, pairs, cf['min_gap_eV'], cf['cap'])
                    pq = restraint_penalty(float(cf['charge_weight']), tt, mt)
                    pc = coupling_penalty(float(wc), ref.coords[tr], ct, cm_, pairs)
                    m, dt = fit('framed', tr, va, seed, penalty=lambda mod, c, pq=pq, pc=pc: pq(mod, c) + pc(mod, c))
                    te_ct, te_cm = coupling_targets(ref, te, tau, pairs, cf['min_gap_eV'], cf['cap'])
                    Tm = model_couplings_r(m, ref.coords[te])
                    tm = np.stack([np.abs(Tm[:, a_, b_]) for a_, b_ in pairs], 1)
                    err = [float(np.sqrt(np.mean((tm[te_cm[:, c], c] - te_ct[te_cm[:, c], c]) ** 2))) if te_cm[:, c].any()
                           else None for c in range(len(pairs))]
                    te_t, te_m = charge_targets(ref, te, q_ion, rc['min_gap_eV'], rc.get('states', 'all'))
                    iw = ionic_weights(m, ref.coords[te])
                    res[str(wc)].append({'seed': seed, 'test_rmse_kcal': Mx.rmse_kcal(m.predict(ref.coords[te]), E[te]).tolist(),
                                         'coupling_rmse': err, 'n_test_couplings': int(te_cm.sum()),
                                         'ionic_weight_mae': float(np.mean(np.abs(iw[te_m] - te_t[te_m]))) if te_m.any() else None})
                    if seed == tc['seeds'][0]:
                        save(m, f'coupfit_w{wc}', f'coupling fit weight {wc} + charge restraint {cf["charge_weight"]}', tr)
                run.log(f'coupling fit weight {wc}: energy RMSE '
                        f'{np.round(np.mean([v["test_rmse_kcal"] for v in res[str(wc)]], 0), 2)} kcal/mol, held-out |tau| RMSE '
                        f'{np.round(np.nanmean([[e if e is not None else np.nan for e in v["coupling_rmse"]] for v in res[str(wc)]], 0), 3)} 1/Å')
            metrics['coupling_fit'] = dict(res, pairs=[list(p) for p in pairs], charge_weight=cf['charge_weight'],
                                           cap=cf['cap'], min_gap_eV=cf['min_gap_eV'], q_ion=q_ion)
    run.p('results', 'metrics.json').write_text(json.dumps(metrics, indent=1))
    np.savez(run.p('results', 'curves.npz'), **curves)
    run.log('wrote results/metrics.json, results/curves.npz, models/*.pt (+ .json spec; load with diabml.models.load_model)')


# ------------------------------------------------------------------------------------------------ figures
INK, BLUE, ORANGE, GREEN, PLUM, COPPER, GREY = '#2E2A36', '#2a78d6', '#eb6834', '#1baf7a', '#4E3A6E', '#C8873A', '#9A94A0'
SCOL = [INK, BLUE, ORANGE, GREEN]
MCOL = {'framed': GREEN, 'cdnn': ORANGE}
MNAME = {'framed': 'framed CDNN', 'cdnn': 'CDNN'}


def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 200, 'font.size': 12, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'legend.frameon': False})
    return plt


def stage_figures(run):
    plt = _mpl()
    ref = load_refs(run)
    r = ref.coords[:, 0]
    E = ref.energies
    sv = slice_values(ref)
    cm = cut_mask(ref)
    cap = {}
    note = ', '.join(f'{LABEL[k]} = {v:g}{UNITS[k]}' for k, v in sv.items())
    method = ref.meta.get('method', ref.name)
    flag = ~(ref.extra.get('converged', np.ones(len(ref), bool)).astype(bool)
             & ref.extra.get('active_space_consistent', np.ones(len(ref), bool)).astype(bool))

    def save(fig, name, caption):
        fig.tight_layout()
        fig.savefig(run.p('figures', name), bbox_inches='tight')
        plt.close(fig)
        cap[name] = caption
        run.log(f'figures/{name}')

    # F1 cuts at every angle
    ja = ref.coord_names.index('a_COO') if 'a_COO' in ref.coord_names else None
    gm = grid_mask(ref)
    angles = np.unique(ref.coords[gm, ja]) if ja is not None else [None]
    base = {k: v for k, v in sv.items() if k != 'a_COO'}
    ncol = min(3, len(angles)); nrow = int(np.ceil(len(angles) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.8 * nrow), squeeze=False)
    for ax, a in zip(axes.flat, angles):
        vals = dict(base, **({'a_COO': a} if a is not None else {}))
        m = cut_mask(ref, vals) & gm
        o = np.argsort(r[m])
        for s in range(min(4, ref.n_states)):
            ax.plot(r[m][o], E[m][o, s], '-o', ms=3, color=SCOL[s], label=f'S{s}')
        bad = m & flag
        if bad.any():
            ax.plot(r[bad], E[bad, 0], 'x', color='red', ms=8, label='flagged')
        ax.set(xlabel='r(O–O) (Å)', ylabel='E (eV)', title=f'∠COO = {a:g}°' if a is not None else '', xlim=(1.1, 5.2))
    for ax in list(axes.flat)[len(angles):]:
        ax.axis('off')
    axes.flat[0].legend(ncol=2, fontsize=10)
    save(fig, 'F1_ref_cuts.png', f'Reference energies of the four singlet A′ states along r(O–O) at each C–O–O angle '
         f'({method}; energies relative to CH₂O + O(¹D)). Red crosses mark unconverged points or points whose active '
         f'space changed character.')

    # F2 character
    o = np.argsort(r[cm])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for s in range(min(4, ref.n_states)):
        axes[0].plot(r[cm][o], E[cm][o, s], '-o', ms=3, color=SCOL[s], label=f'S{s}')
        if 'q_O_terminal' in ref.extra:
            axes[1].plot(r[cm][o], ref.extra['q_O_terminal'][cm][o, s], '-o', ms=3, color=SCOL[s])
            axes[2].plot(r[cm][o], ref.extra['n_unpaired'][cm][o, s], '-o', ms=3, color=SCOL[s])
    axes[0].set(xlabel='r(O–O) (Å)', ylabel='E (eV)', title='Energies', xlim=(1.1, 5.2))
    axes[1].set(xlabel='r(O–O) (Å)', ylabel='Mulliken charge, terminal O (e)', title='Charge separation', xlim=(1.1, 5.2))
    axes[2].set(xlabel='r(O–O) (Å)', ylabel='effective unpaired electrons', title='Open-shell character', xlim=(1.1, 5.2))
    axes[0].legend(ncol=2)
    save(fig, 'F2_ref_character.png', f'State character along r(O–O) at {note}: energy, terminal-O charge (zwitterion '
         f'proxy) and effective unpaired electrons (biradical proxy), each from the state\'s own density.')

    # F3 maps over (r_OO, angle)
    if ja is not None and len(angles) > 2:
        m = (cut_mask(ref, base) if base else np.ones(len(ref), bool)) & gm
        rr, aa = np.unique(r[m]), np.unique(ref.coords[m, ja])
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
        for s, ax in enumerate(axes):
            Z = np.full((len(aa), len(rr)), np.nan)
            for i in np.where(m)[0]:
                Z[np.searchsorted(aa, ref.coords[i, ja]), np.searchsorted(rr, r[i])] = E[i, s]
            pc = ax.pcolormesh(rr, aa, Z, shading='nearest', cmap='viridis')
            ax.contour(rr, aa, Z, levels=10, colors='white', linewidths=0.5)
            fig.colorbar(pc, ax=ax, label='E (eV)')
            ax.set(xlabel='r(O–O) (Å)', title=f'S{s}', xscale='log')
            ax.grid(False)
        axes[0].set_ylabel('∠C–O–O (°)')
        save(fig, 'F3_ref_maps.png', 'Each state over the planar (r(O–O), ∠C–O–O) grid' +
             (f' at {", ".join(f"{LABEL[k]} = {v:g}{UNITS[k]}" for k, v in base.items())}' if base else '') +
             '; log scale in r to show the short-range region.')

    # F3b r_CO dependence
    if 'r_CO' in ref.coord_names:
        jc = ref.coord_names.index('r_CO')
        rcs = np.unique(ref.coords[gm, jc])
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
        cmap = plt.get_cmap('plasma')
        for k, rc in enumerate(rcs):
            vals = dict(sv, r_CO=rc)
            m = cut_mask(ref, vals) & gm; o = np.argsort(r[m])
            for ax, s in zip(axes, (0, 1)):
                ax.plot(r[m][o], E[m][o, s], '-o', ms=3, color=cmap(k / max(1, len(rcs) - 1)), label=f'r(C–O) = {rc:g} Å')
        for ax, s in zip(axes, (0, 1)):
            ax.set(xlabel='r(O–O) (Å)', ylabel='E (eV)', title=f'S{s}', xlim=(1.1, 5.2))
        axes[0].legend()
        save(fig, 'F3b_ref_rCO.png', 'How the C–O1 bond length changes the two lowest states along r(O–O) '
             '(at the equilibrium C–O–O angle): the zwitterion and biradical structures prefer different C–O lengths.')

    # F4 dynamic correlation
    if 'E_nevpt2_eV' in ref.extra:
        Ec, En = ref.extra['E_casscf_eV'][cm][o], ref.extra['E_nevpt2_eV'][cm][o]
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
        for s in range(4):
            axes[0].plot(r[cm][o], Ec[:, s], '--', color=SCOL[s], lw=1.5)
            axes[0].plot(r[cm][o], En[:, s], '-o', ms=3, color=SCOL[s], label=f'S{s}')
            axes[1].plot(r[cm][o], En[:, s] - Ec[:, s], '-o', ms=3, color=SCOL[s], label=f'S{s}')
        axes[0].set(xlabel='r(O–O) (Å)', ylabel='E (eV)', title='SC-NEVPT2 (solid) vs SA-CASSCF (dashed)', xlim=(1.1, 5.2))
        axes[1].set(xlabel='r(O–O) (Å)', ylabel='E(NEVPT2) − E(CASSCF) (eV)', title='Dynamic-correlation shift', xlim=(1.1, 5.2))
        axes[0].legend(ncol=2)
        save(fig, 'F4_dynamic_correlation.png', f'Effect of dynamic correlation along r(O–O) at {note}. Each level is '
             'zeroed at its own CH₂O + O(¹D) asymptote, so the right panel shows how binding changes. State-specific '
             'NEVPT2 can be unreliable where states nearly cross; look for kinks there.')

    mpath = run.p('results', 'metrics.json')
    if not mpath.exists():
        run.p('figures', 'captions.json').write_text(json.dumps(cap, indent=1, ensure_ascii=False))
        run.log('no results/metrics.json yet: model figures skipped (run train)'); return
    M = json.loads(mpath.read_text())
    cv = np.load(run.p('results', 'curves.npz'))
    x = cv['x']
    models = M['config']['models']

    # F5 parity
    if 'random_test_idx' in cv.files:
        te = cv['random_test_idx']
        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5.4), squeeze=False)
        for ax, k in zip(axes[0], models):
            P = cv[f'random_{k}_test_pred']
            for s in range(4):
                ax.plot(E[te, s], P[:, s], 'o', ms=4, color=SCOL[s], alpha=0.8, label=f'S{s}')
            lo, hi = np.nanmin(E[te, :4]), np.nanmax(E[te, :4])
            ax.plot([lo, hi], [lo, hi], color=GREY, lw=1)
            rm = np.mean([v['test_rmse_kcal'] for v in M['random_split'][k]], 0)
            ax.set(xlabel='reference E (eV)', ylabel='model E (eV)',
                   title=f'{MNAME[k]}: test RMSE ' + ' / '.join(f'{v:.1f}' for v in rm) + ' kcal/mol')
            ax.legend()
        save(fig, 'F5_parity.png', 'Held-out test points (random split, first seed): model against reference for each '
             'state. Titles give the test RMSE per state, averaged over seeds.')

    # F6 extrapolation
    if 'extrapolation' in M:
        rmax = M['extrapolation']['r_train_max']
        fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5), squeeze=False)
        o = np.argsort(r[cm])
        for ax, k in zip(axes[0], models):
            Pk = cv[f'extrap_{k}_E']
            ax.axvspan(rmax, x.max(), color=COPPER, alpha=0.12, lw=0)
            for s in range(4):
                ax.plot(r[cm][o], E[cm][o, s], 'o', color=INK, ms=4, mfc='white' if s else INK,
                        label='reference' if s == 0 else None)
                ax.plot(x, Pk[:, s], color=MCOL[k], lw=2, label=MNAME[k] if s == 0 else None)
            rm = np.mean([v['outer_rmse_kcal'] for v in M['extrapolation'][k]], 0)
            ax.set(xlabel='r(O–O) (Å)', ylabel='E (eV)', ylim=(np.nanmin(E[:, :4]) - 1, np.nanmax(E[cm, :4]) + 1),
                   title=f'{MNAME[k]}: beyond {rmax} Å RMSE ' + ' / '.join(f'{v:.0f}' for v in rm) + ' kcal/mol')
            ax.legend(loc='lower right')
        save(fig, 'F6_extrapolation.png', f'Trained only on r(O–O) ≤ {rmax} Å (shaded: no training data), drawn at '
             f'{note}. The frame fixes the asymptotes and the ion-pair Coulomb tail; plain CDNN has to guess them.')

    # F7 learned matrix
    if 'random_framed_diag' in cv.files:
        D, U, Ead = cv['random_framed_diag'], cv['random_framed_coup'], cv['random_framed_E']
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        kc = {0: BLUE, 1: BLUE, 2: BLUE, 3: GREEN, 4: ORANGE}
        lab = {0: 'CH₂O + O(¹D) (3 components)', 3: 'CH₂O(T₁) + O(³P)', 4: 'CH₂O⁺ + O⁻ (zwitterion)'}
        for i in range(5):
            axes[0].plot(x, D[:, i], '--', lw=2.4, color=kc[i], label=lab.get(i))
        for s in range(4):
            axes[0].plot(x, Ead[:, s], color=GREY, lw=1.4, label='adiabatic (model)' if s == 0 else None)
        o = np.argsort(r[cm])
        axes[0].plot(r[cm][o], E[cm][o, 0], 'o', color=INK, ms=3, mfc='white', label='reference S0')
        axes[0].set(xlabel='r(O–O) (Å)', ylabel='E (eV)', xlim=(x.min(), 5), ylim=(np.nanmin(E[:, :4]) - 1.5, 11),
                    title='Diagonal elements (structures)')
        axes[0].legend(fontsize=10)
        iu = np.triu_indices(5, 1)
        big = np.max(np.abs(U), 0)
        for c in np.argsort(-big)[:6]:
            i, j = iu[0][c], iu[1][c]
            axes[1].plot(x, np.abs(U[:, c]) * 1000, lw=1.6, label=f'|Ω{i}{j}|' + (' (ionic)' if j == ION else ''))
        axes[1].set_ylabel('|coupling| (meV)')
        axes[1].set(xlabel='r(O–O) (Å)', xlim=(x.min(), 5), title='Six largest off-diagonal elements')
        axes[1].legend(fontsize=9, ncol=2)
        save(fig, 'F7_learned_matrix.png', f'The framed model\'s matrix along r(O–O) at {note}: structures (dashed) '
             'behind the adiabatic states, and the learned couplings. Energies alone do not fix how much the '
             'structures mix (see F9).')

    # F8 learning curve
    if 'learning_curve' in M:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for k in models:
            ns = sorted(int(n) for n in M['learning_curve'][k])
            if not ns:
                continue
            mean = [np.mean(M['learning_curve'][k][str(n)]) for n in ns]
            per_seed = [np.mean(M['learning_curve'][k][str(n)], 1) for n in ns]
            lo = [m - np.min(p) for m, p in zip(mean, per_seed)]
            hi = [np.max(p) - m for m, p in zip(mean, per_seed)]
            ax.errorbar(ns, mean, yerr=[lo, hi], marker='o', color=MCOL[k], label=MNAME[k], capsize=4, lw=2)
        ax.set(xlabel='training points', ylabel='test RMSE, mean of 4 states (kcal/mol)', xscale='log', yscale='log',
               title=f'Learning curve ({M["learning_curve"]["n_test"]} fixed test points)')
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        save(fig, 'F8_learning_curve.png', 'Test error against the number of training points; bars span the seeds.')

    # F9 restraint
    if 'restraint' in M:
        R = M['restraint']
        ws = [w for w in R if w not in ('q_ref', 'states', 'min_gap_eV', 'target_cut', 'target_cut_r')]
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        cmap = plt.get_cmap('viridis')
        for k, w in enumerate(ws):
            key = f'restraint_{w}_ionic_weight'
            if key in cv.files:
                W = np.atleast_2d(cv[key].T).T
                c = cmap(k / max(1, len(ws) - 1))
                axes[0].plot(x, W[:, 0], color=c, lw=2, label=f'weight {w}')
                if W.shape[1] > 1:
                    axes[0].plot(x, W[:, 1], color=c, lw=1.6, ls='--')
        T = np.atleast_2d(np.array(R['target_cut']).T).T
        axes[0].plot(R['target_cut_r'], T[:, 0], 'o', color=INK, ms=5, label=f'S0 target q₀ / ({R["q_ref"]:.2f} e)')
        if T.shape[1] > 1:
            axes[0].plot(R['target_cut_r'], T[:, 1], 'o', color=INK, ms=5, mfc='white', label='S1 target (dashed: model S1)')
        axes[0].set(xlabel='r(O–O) (Å)', ylabel='zwitterion weight', xlim=(x.min(), 5), ylim=(-0.05, 1.05),
                    title='Zwitterion character of S0 (solid) and S1 (dashed)')
        axes[0].legend(fontsize=10)
        e = [np.mean([np.mean(v['test_rmse_kcal']) for v in R[w]]) for w in ws]
        iw = [np.mean([v['ionic_weight_mae'] for v in R[w] if v['ionic_weight_mae'] is not None]) for w in ws]
        axes[1].plot(iw, e, '-o', color=PLUM)
        for w, a, b in zip(ws, iw, e):
            axes[1].annotate(f' {w}', (a, b))
        axes[1].set(xlabel='ionic-weight error vs charge target (MAE)', ylabel='energy test RMSE, mean of states (kcal/mol)',
                    title='Trade-off as the restraint weight grows')
        save(fig, 'F9_restraint.png', f'Diabatic restraint ({R.get("states", "ground")} states): each state\'s weight on '
             'the CH₂O⁺ + O⁻ structure is pulled toward its CASSCF terminal-O charge divided by the charge of a pure '
             f'zwitterion, {R["q_ref"]:.2f} e. Left: along r(O–O); right: energy accuracy against agreement with the targets.')
    # F11 derivative couplings vs ab initio
    cc = coupling_comparison(run, ref)
    if cc and cc['models']:
        pairs = cc['pairs']
        fig, axes = plt.subplots(1, len(pairs), figsize=(5.6 * len(pairs), 4.6), squeeze=False)
        style = {'random_framed': (GREEN, '-', 'framed CDNN'), 'random_cdnn': (ORANGE, '-', 'CDNN')}
        rw = [k for k in cc['models'] if k.startswith('restraint_w')]
        cw = [k for k in cc['models'] if k.startswith('coupfit_w')]
        if cw:                                   # with a coupling fit, show the strongest charge-only model for comparison
            rw = rw[-1:]
            cc['models'] = {k: v for k, v in cc['models'].items() if not k.startswith('restraint_w') or k in rw}
        cmap = plt.get_cmap('viridis')
        for i, k in enumerate(rw):
            style[k] = (cmap(i / max(1, len(rw) - 1)) if not cw else PLUM, '--', 'charge restraint, weight ' + k[len('restraint_w'):])
        cmap2 = plt.get_cmap('autumn')
        for i, k in enumerate(cw):
            style[k] = (cmap2(i / max(1, len(cw))), '-', 'coupling fit, weight ' + k[len('coupfit_w'):])
        for c, (a, b) in enumerate(pairs):
            ax = axes[0][c]
            for k, d in cc['models'].items():
                col, ls, lab = style.get(k, (GREY, ':', k))
                ax.plot(cc['x'], d['fine'][:, c], color=col, ls=ls, lw=1.8, label=lab)
            ax.plot(cc['r'], cc['tau_ref'][:, c], 'o', color=INK, ms=6, label='SA-CASSCF (PySCF)')
            top = np.nanmax(cc['tau_ref'][:, c])
            ax.set(xlabel='r(O–O) (Å)', ylabel=f'|τ{a}{b}| along r(O–O) (1/Å)', title=f'S{a}–S{b}',
                   xlim=(cc['x'].min(), cc['x'].max()), ylim=(0, max(1.0, 1.6 * top)))
        axes[0][0].legend(fontsize=9)
        save(fig, 'F11_couplings.png', 'Derivative couplings between adjacent states along the O–O stretch: the models\' '
             '(computed from their matrices, never trained on) against ab initio SA-CASSCF couplings. A quasi-diabatic '
             'matrix reproduces these; a matrix that only fits energies need not. Magnitudes, because phases are arbitrary.')
    # F10 short range: below the data
    if 'x_short' in cv.files and all(f'random_{k}_short' in cv.files for k in models):
        xs, rmin = cv['x_short'], float(r.min())
        fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 4.6), squeeze=False)
        for ax, k in zip(axes[0], models):
            S = cv[f'random_{k}_short']
            ax.axvspan(xs.min(), rmin, color=COPPER, alpha=0.12, lw=0)
            for s in range(4):
                ax.plot(xs, S[:, s], color=SCOL[s], lw=2, label=f'S{s}')
            o = np.argsort(r[cm])
            ax.plot(r[cm][o], E[cm][o, 0], 'o', color=INK, ms=4, mfc='white', label='reference S0')
            ax.axhline(float(np.nanmin(E[:, 0])), color=GREY, lw=1, ls=':')
            ax.set(xlabel='r(O–O) (Å)', ylabel='E (eV)', xlim=(xs.min(), xs.max()), ylim=(float(np.nanmin(E[:, 0])) - 3, 15),
                   title=f'{MNAME[k]}: below the data (shaded)')
            ax.legend(loc='upper right', fontsize=10)
        save(fig, 'F10_short_range.png', f'Model energies at O–O distances shorter than any reference point (shaded, '
             f'r < {rmin:g} Å), drawn at {note}. A curve that drops below the dotted line (the lowest reference energy) is a '
             'hole: trajectories that reach it fall in. The framed model has Born–Mayer walls for this; plain CDNN does not.')
    run.p('figures', 'captions.json').write_text(json.dumps(cap, indent=1, ensure_ascii=False))


# ------------------------------------------------------------------------------------------------ tables
def _write_table(run, name, header, rows, title):
    with open(run.p('tables', f'{name}.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    md = [f'**{title}**', '', '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    md += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    run.p('tables', f'{name}.md').write_text('\n'.join(md) + '\n')
    run.log(f'tables/{name}.csv')
    return {'name': name, 'title': title, 'header': header, 'rows': rows}


def _f(x, d=2):
    return '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


def stage_tables(run):
    ref = load_refs(run)
    r = ref.coords[:, 0]
    tables = []
    ex = ref.extra
    conv = ex.get('converged', np.ones(len(ref), bool)).astype(bool)
    cons = ex.get('active_space_consistent', np.ones(len(ref), bool)).astype(bool)
    far = np.isclose(r, r.max()) & cut_mask(ref, {k: v for k, v in slice_values(ref).items() if k != 'a_COO'})
    rows = [['method', ref.meta.get('method', ref.name)], ['energies used for training', ref.meta.get('energy_level', 'casscf')],
            ['geometries whose state energies were put in ascending order (largest shift, eV)',
             f'{ref.meta.get("n_reordered", 0)} ({ref.meta.get("largest_reorder_eV", 0.0):.2f})'],
            ['geometries', len(ref)],
            ['coordinates', ', '.join(f'{n} {np.min(ref.coords[:, j]):g}–{np.max(ref.coords[:, j]):g}' for j, n in enumerate(ref.coord_names))],
            ['fixed', ', '.join(f'{k} = {v:g}' for k, v in ref.meta.get('fixed', {}).items())],
            ['converged', f'{conv.sum()} / {len(ref)}'], ['consistent active space', f'{cons.sum()} / {len(ref)}'],
            ['O(¹D) components at largest r_OO: spread (meV)', _f(1000 * float(np.nanmax(np.ptp(ref.energies[far, :3], 1))) if far.any() else None, 1)],
            ['S3 asymptote (eV)', _f(float(np.nanmean(ref.energies[far, 3])) if far.any() else None, 3)]]
    if 'seconds' in ex:
        rows += [['mean time per geometry (s)', _f(float(np.mean(ex['seconds'])), 1)],
                 ['total compute (h)', _f(float(np.sum(ex['seconds'])) / 3600, 2)]]
    if 'casci_mismatch_hartree' in ex:
        rows += [['largest CASCI vs SA-CASSCF energy difference before NEVPT2 (µEh)', _f(1e6 * float(np.nanmax(ex['casci_mismatch_hartree'])), 2)]]
    cm = cut_mask(ref); o = np.argsort(r[cm])
    s0 = ref.energies[cm][o, 0]
    rows += [['S0 well depth along the cut (eV)', _f(float(np.nanmax(s0[-1:]) - np.nanmin(s0)), 3)],
             ['S0 minimum at r_OO (Å)', _f(float(r[cm][o][np.nanargmin(s0)]), 3)]]
    tables.append(_write_table(run, 'T1_reference_summary', ['quantity', 'value'], rows, 'T1. Reference data'))

    lp = run.p('data', 'ladder.json')
    if lp.exists():
        L = json.loads(lp.read_text())
        X, Cc = L['experiment'], L.get('computed', {})
        rows = [['IE(CH₂O) (eV)', _f(X['IE_CH2O_adiabatic'], 3) + ' (adiabatic)', _f(Cc.get('IE_CH2O_vertical'), 3) + (' (vertical)' if Cc else '')],
                ['EA(O) (eV)', _f(X['EA_O'], 3), _f(Cc.get('EA_O'), 3)],
                ['O ¹D – ³P (eV)', _f(X['O_1D_3P'], 3), _f(Cc.get('O_1D_3P'), 3) + (' (experiment)' if Cc else '')],
                ['ΔE(ion pair) = IE − EA − ΔE(¹D) (eV)', _f(L['dE_ion_experiment'], 3), _f(Cc.get('dE_ion'), 3)],
                ['Coulomb crossing r* = 14.40 / ΔE (Å)', _f(14.399645 / L['dE_ion_experiment'], 2), _f(14.399645 / Cc['dE_ion'], 2) if Cc else '']]
        tables.append(_write_table(run, 'T2_ladder', ['quantity', 'experiment', f'computed ({Cc.get("basis", "not run")})'], rows,
                                   f'T2. Ladder anchors for the ion-pair structure (frame uses: {L["use"]})'))

    mp = run.p('results', 'metrics.json')
    if mp.exists():
        M = json.loads(mp.read_text())
        models = M['config']['models']
        rows = []
        def add(test, k, vals):
            a = np.array(vals)
            rows.append([test, MNAME[k]] + [f'{m:.2f} ± {s:.2f}' for m, s in zip(a.mean(0), a.std(0))] + [len(a)])
        if 'random_split' in M:
            for k in models:
                add('random split: test', k, [v['test_rmse_kcal'] for v in M['random_split'][k]])
        if 'extrapolation' in M:
            rt = M['extrapolation']['r_train_max']
            for k in models:
                add(f'trained r ≤ {rt} Å: validation', k, [v['inner_val_rmse_kcal'] for v in M['extrapolation'][k]])
            for k in models:
                add(f'trained r ≤ {rt} Å: beyond {rt} Å', k, [v['outer_rmse_kcal'] for v in M['extrapolation'][k]])
        if rows:
            tables.append(_write_table(run, 'T3_model_errors', ['test', 'model', 'S0', 'S1', 'S2', 'S3', 'seeds'], rows,
                                       'T3. RMSE (kcal/mol), mean ± s.d. over seeds'))
        if 'learning_curve' in M:
            rows = []
            for k in models:
                for n in sorted(M['learning_curve'][k], key=int):
                    a = np.mean(M['learning_curve'][k][n], 1)
                    rows.append([n, MNAME[k], f'{a.mean():.2f} ± {a.std():.2f}'] +
                                [f'{v:.2f}' for v in np.mean(M['learning_curve'][k][n], 0)])
            tables.append(_write_table(run, 'T4_learning_curve', ['n train', 'model', 'mean RMSE', 'S0', 'S1', 'S2', 'S3'],
                                       rows, f'T4. Learning curve, test RMSE (kcal/mol) on {M["learning_curve"]["n_test"]} fixed points'))
        if 'restraint' in M:
            R = M['restraint']
            rows = []
            for w in [w for w in R if w not in ('q_ref', 'states', 'min_gap_eV', 'target_cut', 'target_cut_r')]:
                e = np.mean([v['test_rmse_kcal'] for v in R[w]], 0)
                iw = [v['ionic_weight_mae'] for v in R[w] if v['ionic_weight_mae'] is not None]
                eq = np.mean([np.atleast_1d(v['ionic_weight_at_eq'])[:2] for v in R[w]], 0)
                rows.append([w] + [f'{v:.2f}' for v in e] + [_f(float(np.mean(iw)), 3) if iw else '',
                                                             ' / '.join(f'{v:.2f}' for v in eq)])
            tables.append(_write_table(run, 'T5_restraint', ['weight', 'S0', 'S1', 'S2', 'S3', 'zwitterion-weight MAE',
                                                             'zwitterion weight S0 / S1 at 1.35 Å'], rows,
                                       f'T5. Diabatic restraint ({R.get("states", "ground")} states, pure zwitterion '
                                       f'{R["q_ref"]:.2f} e): test RMSE (kcal/mol) and agreement with the charge targets'))
    cc = coupling_comparison(run, ref)
    if cc and cc['models']:
        rows = []
        Eg = cc['E_ref']
        for k, d in cc['models'].items():
            row = [k.replace('random_', '').replace('restraint_w', 'charge restraint, weight ').replace('coupfit_w', 'coupling fit, weight ')]
            for c, (a, b) in enumerate(cc['pairs']):
                ok = np.isfinite(cc['tau_ref'][:, c]) & (np.abs(Eg[:, b] - Eg[:, a]) > 0.1)
                err = np.sqrt(np.mean((d['at_ref'][ok, c] - cc['tau_ref'][ok, c]) ** 2)) if ok.any() else np.nan
                row.append(_f(float(err), 3))
            i_pk = int(np.nanargmax(d['fine'][:, 0]))
            row.append(_f(float(cc['x'][i_pk]), 2))
            rows.append(row)
        i_ref = int(np.nanargmax(cc['tau_ref'][:, 0]))
        rows.append(['SA-CASSCF reference'] + ['0'] * len(cc['pairs']) + [_f(float(cc['r'][i_ref]), 2)])
        tables.append(_write_table(run, 'T7_couplings', ['model'] + [f'|τ{a}{b}| RMS error (1/Å)' for a, b in cc['pairs']]
                                   + ['peak of |τ01| at r_OO (Å)'], rows,
                                   'T7. Derivative couplings along r_OO against SA-CASSCF (points where the two states '
                                   'are more than 0.1 eV apart)'))
    if mp.exists():
        M = json.loads(mp.read_text())
        if 'coupling_fit' in M:
            Cf = M['coupling_fit']
            prs = Cf['pairs']
            rows = []
            for w in [k for k in Cf if k not in ('pairs', 'charge_weight', 'cap', 'min_gap_eV', 'q_ion')]:
                e = np.mean([v['test_rmse_kcal'] for v in Cf[w]], 0)
                ce = np.nanmean([[x if x is not None else np.nan for x in v['coupling_rmse']] for v in Cf[w]], 0)
                iw = [v['ionic_weight_mae'] for v in Cf[w] if v['ionic_weight_mae'] is not None]
                rows.append([w] + [f'{v:.2f}' for v in e] + [f'{v:.3f}' for v in ce] + [_f(float(np.mean(iw)), 3) if iw else ''])
            tables.append(_write_table(run, 'T8_coupling_fit', ['coupling weight', 'S0', 'S1', 'S2', 'S3']
                                       + [f'|τ{a}{b}| (1/Å)' for a, b in prs] + ['zwitterion-weight MAE'], rows,
                                       f'T8. Fitting the ab initio couplings (with charge restraint weight {Cf["charge_weight"]}): '
                                       'held-out energy RMSE (kcal/mol) and held-out coupling RMSE'))
    if 'E_nevpt2_eV' in ex:
        rows = []
        Ec, En = ex['E_casscf_eV'][cm][o], ex['E_nevpt2_eV'][cm][o]
        i_eq = int(np.argmin(np.abs(r[cm][o] - 1.35)))
        for s in range(4):
            rows.append([f'S{s}', _f(float(Ec[i_eq, s]), 3), _f(float(En[i_eq, s]), 3), _f(float(En[i_eq, s] - Ec[i_eq, s]), 3)])
        rows.append(['S0 well depth', _f(float(Ec[-1, 0] - np.nanmin(Ec[:, 0])), 3), _f(float(En[-1, 0] - np.nanmin(En[:, 0])), 3),
                     _f(float((En[-1, 0] - np.nanmin(En[:, 0])) - (Ec[-1, 0] - np.nanmin(Ec[:, 0]))), 3)])
        tables.append(_write_table(run, 'T6_dynamic_correlation', ['state (near r_OO = 1.35 Å)', 'SA-CASSCF (eV)',
                                                                   'SC-NEVPT2 (eV)', 'shift (eV)'], rows,
                                   'T6. Dynamic correlation along the equilibrium cut (each level zeroed at its own asymptote)'))
    from . import active
    rows = active.table_rows(run)
    if rows:
        tables.append(_write_table(run, 'T9_active_learning', ['quantity', 'value'], rows,
                                   'T9. Active learning: committee spread and the error where it was largest'))
    from . import explore
    rows = explore.table_rows(run)
    if rows:
        tables.append(_write_table(run, 'T11_dynamics_exploration', ['quantity', 'value'], rows,
                                   'T11. Active learning from dynamics (details and figures: dynamics.html)'))
    from . import production
    rows = production.table_rows(run)
    if rows:
        tables.append(_write_table(run, 'T10_production_regions', ['region (held-out test points)', 'n',
                                   'one model S0 / S1 / S2 / S3', 'ensemble S0 / S1 / S2 / S3', 'member spread'], rows,
                                   'T10. Production ensemble: held-out RMSE (kcal/mol) by region'))
    run.p('tables', 'tables.json').write_text(json.dumps(tables, indent=1, ensure_ascii=False))


# ------------------------------------------------------------------------------------------------ report
def stage_report(run):
    import html
    cfg_text = run.p('config.toml').read_text()
    caps = json.loads(run.p('figures', 'captions.json').read_text()) if run.p('figures', 'captions.json').exists() else {}
    for extra in ('active_captions.json', 'production_captions.json'):
        if run.p('figures', extra).exists():
            caps = dict(json.loads(run.p('figures', extra).read_text()), **caps)
    tabs = json.loads(run.p('tables', 'tables.json').read_text()) if run.p('tables', 'tables.json').exists() else []
    esc = html.escape
    parts = []
    for t in tabs:
        head = ''.join(f'<th>{esc(str(h))}</th>' for h in t['header'])
        body = ''.join('<tr>' + ''.join(f'<td>{esc(str(c))}</td>' for c in row) + '</tr>' for row in t['rows'])
        parts.append(f'<h3>{esc(t["title"])}</h3><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
                     f'<p class="src">tables/{t["name"]}.csv</p>')
    figs = ''.join(f'<figure><img src="figures/{esc(n)}" alt="{esc(c)}"><figcaption><b>{esc(n.split("_")[0])}.</b> '
                   f'{esc(c)}</figcaption></figure>' for n, c in caps.items())
    name = run.cfg['run'].get('name') or run.path.name
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CH2OO run: {esc(name)}</title>
<style>
:root{{--paper:#F6F4F0;--ink:#2E2A36;--muted:#7A7480;--rule:#DAD3C8;--plum:#3D2C58;--copper:#C8873A;--panel:#EFEBF3}}
@media (prefers-color-scheme:dark){{:root{{--paper:#17131f;--ink:#ece8f2;--muted:#a69cb6;--rule:#3d3250;--panel:#241b33}}}}
body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,-apple-system,sans-serif}}
header{{background:var(--plum);color:#fff;border-bottom:5px solid var(--copper);padding:28px 24px}}
header h1{{margin:0;font:600 30px Georgia,serif}} header p{{margin:6px 0 0;color:#E9B872}}
main{{max-width:1180px;margin:0 auto;padding:16px}}
h2{{font:600 24px Georgia,serif;border-bottom:1px solid var(--rule);padding-bottom:4px;margin-top:36px}}
table{{border-collapse:collapse;width:100%;margin:8px 0;font-size:14px}} th,td{{border-bottom:1px solid var(--rule);padding:6px 8px;text-align:left}}
th{{background:var(--panel)}} figure{{margin:22px 0}} img{{max-width:100%;background:#fff;border:1px solid var(--rule)}}
figcaption,.src{{color:var(--muted);font-size:14px}} pre{{background:var(--panel);padding:12px;overflow:auto;font-size:13px}}
</style></head><body>
<header><h1>CH₂OO workflow run: {esc(name)}</h1><p>{esc(run.cfg['run'].get('description', ''))} · {time.strftime('%Y-%m-%d %H:%M')}</p></header>
<main>{'<p><a href="dynamics.html">Dynamics log (dynamics.html)</a>: every trajectory of this round, with diagnostics.</p>' if run.p('dynamics.html').exists() else ''}{'<p><a href="audit.html">Reference audit (audit.html)</a>: every point checked against ORCA.</p>' if run.p('audit.html').exists() else ''}<h2>Tables</h2>{''.join(parts) or '<p>No tables yet.</p>'}
<h2>Figures</h2>{figs or '<p>No figures yet.</p>'}
<h2>Configuration</h2><pre>{esc(cfg_text)}</pre></main></body></html>'''
    run.p('report.html').write_text(page)
    run.log(f'wrote {run.p("report.html")}')


# ------------------------------------------------------------------------------------------------ cli
STAGES = ['init', 'refs', 'status', 'ladder', 'nacs', 'propose', 'compute', 'merge', 'train', 'figures', 'tables',
          'report', 'production', 'explore', 'fill', 'audit', 'repair', 'track', 'dipoles', 'dipole_fit', 'ir', 'gs_grid', 'vib_levels', 'method_cut', 'ir_replot', 'freeze', 'shell', 'shell_check', 'well9d', 'well9d_fit', 'well9d_tz', 'well9d_vpt2', 'well9d_dipole', 'well9d_vci', 'well9d_bands', 'well9d_isotopes', 'well9d_compare', 'rpmd9d', 'rpmd_plan', 'dynamics_report', 'rpmd_check', 'well9d_extend', 'vci_report', 'all']


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m diabml.workflow.ch2oo', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=STAGES)
    ap.add_argument('run', help='run folder (holds config.toml)')
    ap.add_argument('--template', default='quick', help=f'for init: {", ".join(C.TEMPLATES)}')
    ap.add_argument('--refs', help='init: the refs.npz this run starts from (refs.file, relative to the run folder)')
    ap.add_argument('--orca', help='audit, track: the ORCA folder (or the orca executable)')
    ap.add_argument('--jobs', type=int, help='audit: ORCA runs in parallel; dipoles, ir: processes')
    ap.add_argument('--name', help='freeze: the name of the frozen surface (models/surfaces/<name>.npz)')
    ap.add_argument('--batch', type=int, help='shell, well9d: compute this batch only (default: all)')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='print detail lines to the terminal too (they always go to logs/<stage>.log); '
                         'used by the well9d stages')
    ap.add_argument('--force', action='store_true', help='refs: recompute even if data/refs.npz exists; '
                    'propose: propose again even if data/proposed.npz exists')
    a = ap.parse_args(argv)
    if a.stage == 'init':
        return stage_init(a.run, a.template, a.refs)
    run = Run(a.run)
    run.verbose = bool(a.verbose)
    if a.stage == 'status':
        return stage_status(run)
    todo = ['refs', 'ladder', 'nacs', 'train', 'figures', 'tables', 'report'] if a.stage == 'all' else [a.stage]
    if a.stage == 'all' and not run.cfg['nacs'].get('enabled', False):
        todo.remove('nacs')
    active_run = run.cfg['active'].get('enabled', False)
    explore_run = run.cfg['explore'].get('enabled', False)
    if a.stage == 'all' and active_run:
        todo = ['refs', 'ladder', 'explore' if explore_run else 'propose', 'compute', 'merge', 'train', 'figures',
                'tables', 'report']
        if explore_run and not run.cfg['explore'].get('train_after', False):
            # the next round's committee is the model that matters; train a production surface at the end instead
            todo = [t for t in todo if t not in ('train', 'figures')]
    if a.stage == 'explore' and not active_run:
        sys.exit('explore belongs to active-learning runs: make one with --template explore')
    if a.stage in ('propose', 'compute', 'merge', 'fill') and not active_run:
        sys.exit(f'{a.stage} belongs to active-learning runs: make one with --template active ([active] enabled = true)')
    from . import active, audit, dipoles, explore, fill, ir, production, repair, track, vib_levels, gs_grid, method_cut, shell
    from . import well9d, vci9, vci_report, isotopes9, compare9, rpmd9, dynamics_report, rpmd_check, extend9
    for st in todo:
        run.open_log(st)
        t0 = time.time()
        {'refs': lambda: stage_refs(run, a.force), 'ladder': lambda: stage_ladder(run), 'nacs': lambda: stage_nacs(run),
         'propose': lambda: active.stage_propose(run, a.force), 'compute': lambda: active.stage_compute(run),
         'merge': lambda: active.stage_merge(run),
         'production': lambda: production.stage_production(run),
         'explore': lambda: explore.stage_explore(run, a.force),
         'fill': lambda: fill.stage_fill(run, a.force),
         'audit': lambda: audit.stage_audit(run, a.orca, a.jobs),
         'repair': lambda: repair.stage_repair(run),
         'track': lambda: track.stage_track(run, a.jobs, a.orca),
         'dipoles': lambda: dipoles.stage_dipoles(run, a.jobs),
         'dipole_fit': lambda: dipoles.stage_dipole_fit(run),
         'ir': lambda: ir.stage_ir(run, a.jobs),
         'vib_levels': lambda: vib_levels.stage_vib_levels(run),
         'ir_replot': lambda: ir.stage_ir_replot(run),
         'freeze': lambda: shell.stage_freeze(run, a.name),
         'shell': lambda: shell.stage_shell(run, a.jobs, a.orca, a.batch),
         'shell_check': lambda: shell.stage_shell_check(run),
         'well9d': lambda: well9d.stage_well9d(run, a.jobs, a.orca, a.batch),
         'well9d_fit': lambda: well9d.stage_well9d_fit(run),
         'well9d_tz': lambda: well9d.stage_well9d_tz(run, a.jobs, a.orca),
         'well9d_vpt2': lambda: well9d.stage_well9d_vpt2(run),
         'well9d_dipole': lambda: well9d.stage_well9d_dipole(run, a.jobs, a.orca),
         'well9d_vci': lambda: vci9.stage_well9d_vci(run),
         'well9d_bands': lambda: vci9.stage_well9d_bands(run),
         'well9d_isotopes': lambda: isotopes9.stage_well9d_isotopes(run),
         'well9d_compare': lambda: compare9.stage_well9d_compare(run),
         'rpmd9d': lambda: rpmd9.stage_rpmd9d(run),
         'rpmd_plan': lambda: rpmd9.stage_rpmd_plan(run),
         'dynamics_report': lambda: dynamics_report.stage_dynamics_report(run),
         'rpmd_check': lambda: rpmd_check.stage_rpmd_check(run, a.jobs, a.orca),
         'well9d_extend': lambda: extend9.stage_well9d_extend(run, a.jobs, a.orca),
         'vci_report': lambda: vci_report.stage_vci_report(run),
         'gs_grid': lambda: gs_grid.stage_gs_grid(run, a.jobs, a.orca),
         'method_cut': lambda: method_cut.stage_method_cut(run, a.jobs, a.orca),
         'train': lambda: stage_train(run),
         'figures': lambda: stage_figures(run), 'tables': lambda: stage_tables(run),
         'report': lambda: stage_report(run)}[st]()
        run.log(f'{st} done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
