"""Fill a region with a regular grid of new reference points, instead of waiting for trajectories to find it.

    python -m diabml.workflow.ch2oo init    runs/22_wall --template explore --refs ../21_md8/data/refs.npz
    python -m diabml.workflow.ch2oo refs    runs/22_wall
    python -m diabml.workflow.ch2oo fill    runs/22_wall     # -> data/proposed.npz (the grid, minus points near the data)
    python -m diabml.workflow.ch2oo compute runs/22_wall     # ORCA (engine = "orca"), lowest of several starts
    python -m diabml.workflow.ch2oo merge   runs/22_wall     # -> data/refs.npz; the next explore round starts from it

The boxes are lists of grid values per coordinate ([[fill.box]] in config.toml; the defaults below cover the inner
O-O wall that hot trajectories reach, r_OO 1.0-1.15 A with the C-O bond anywhere from 1.1 to 1.6 A, and stretched C-O
beyond the original grid). Every combination of the values is a candidate; candidates closer than min_distance grid
steps ([active.length_scales]) to an existing point or to each other are dropped. With committee = true the round's
committee is trained first, so the compute stage can report how wrong the surface was at the new points before
training on them (T9, A2), and max_S0_eV (if set) skips candidates whose committee S0 is higher than that.
"""
from __future__ import annotations

import itertools
import json

import numpy as np

from . import active as A
from . import ch2oo as W

DEFAULT_BOXES = [
    # inner O-O wall: below the original grid (which starts at r_OO 1.2 A), angles and C-O lengths trajectories visit
    {'r_OO': [1.00, 1.05, 1.10, 1.15], 'a_COO': [95.0, 105.0, 115.0, 125.0, 135.0, 145.0],
     'r_CO': [1.10, 1.20, 1.30, 1.40, 1.50, 1.60]},
    # C-O stretched beyond the grid (1.45 A) from the compressed side to the transition region
    {'r_OO': [1.20, 1.275, 1.35, 1.425, 1.50, 1.60, 1.70], 'a_COO': [100.0, 117.7, 135.0], 'r_CO': [1.55, 1.65]},
]


def grid_candidates(names, boxes):
    """Every combination of each box's values, in the order of the model coordinates."""
    out = []
    for b in boxes:
        missing = [n for n in names if n not in b]
        if missing:
            raise SystemExit(f'[[fill.box]] {b} has no values for {missing} (the data use {names})')
        out += list(itertools.product(*[[float(v) for v in np.atleast_1d(b[n])] for n in names]))
    return np.array(out, float).reshape(-1, len(names))


def stage_fill(run, force=False):
    from ..models import EnsembleModel
    from ..refset import RefSet
    out = run.p('data', 'proposed.npz')
    if out.exists() and not force:
        run.log(f'{out.name} exists (use --force to make the grid again); nothing to do')
        return
    fc, ac = run.cfg['fill'], A._ac(run)
    ref = RefSet.load(run.p('data', 'base_refs.npz'))
    if run.cfg['train'].get('sort_states', True):
        ref = W.adiabatic_order(ref)
    boxes = fc.get('box') or DEFAULT_BOXES
    C = grid_candidates(ref.coord_names, boxes)
    sc = A._scales(ref.coord_names, ac)
    md = float(fc['min_distance'])
    D = ref.coords / sc
    keep = []
    for i, x in enumerate(C):
        z = x / sc
        if np.min(np.sum((D - z) ** 2, 1)) < md ** 2:
            continue
        if keep and np.min(np.sum((C[keep] / sc - z) ** 2, 1)) < md ** 2:
            continue
        keep.append(i)
    X = C[keep]
    run.log(f'{len(C)} grid points in {len(boxes)} box(es); {len(X)} more than {md} grid steps from the data '
            f'and from each other')
    nr = 4
    mean = np.full((len(X), nr), np.nan)
    std = np.full((len(X), nr), np.nan)
    if fc['committee'] and len(X):
        run.threads()
        U = W.usable_points(run, ref)
        models, _ = A.train_committee(run, ref, U)
        P = EnsembleModel(models).member_predictions(X)[:, :, :nr]
        mean, std = P.mean(0), P.std(0)
        if fc.get('max_S0_eV') not in (None, '', 'none'):
            ok = mean[:, 0] <= float(fc['max_S0_eV'])
            run.log(f'  {int((~ok).sum())} candidates with committee S0 above {fc["max_S0_eV"]} eV skipped')
            X, mean, std = X[ok], mean[ok], std[ok]
    if len(X) > int(fc['max_points']):
        idx = np.sort(np.random.default_rng(0).choice(len(X), int(fc['max_points']), replace=False))
        run.log(f'  {len(X)} candidates: keeping a random {fc["max_points"]} (max_points)')
        X, mean, std = X[idx], mean[idx], std[idx]
    np.savez(out, coords=X, score=np.nanmax(std, 1) if len(X) else np.zeros(0), mean=mean, std=std,
             cand=C, cand_score=np.zeros(len(C)), round=A._round_of(ref))
    with open(run.p('results', 'proposed.csv'), 'w') as fh:
        fh.write(','.join(ref.coord_names + ['why', 'committee_S0_eV', 'committee_spread_S0_meV']) + '\n')
        for x, m, s in zip(X, mean, std):
            fh.write(','.join(f'{v:.4f}' for v in x) + f',fill grid,{m[0]:.3f},{s[0] * 1000:.0f}\n')
    A._write_json(run, {'n_grid': int(len(C)), 'n_proposed': int(len(X)), 'boxes': boxes, 'min_distance': md}, 'fill')
    if len(X):
        lo, hi = X.min(0), X.max(0)
        run.log(f'{len(X)} geometries -> {out.name} (' + ', '.join(f'{n} {a:g}-{b:g}' for n, a, b in
                                                                   zip(ref.coord_names, lo, hi)) + '); next: compute')
        if fc['committee']:
            run.log(f'  committee spread there: median S0 {np.median(std[:, 0]) * 1000:.0f} meV, '
                    f'max {np.max(std[:, 0]) * 1000:.0f} meV')
