"""Error and symmetry metrics. Energies in eV in, kcal/mol or meV out (as named)."""
from __future__ import annotations

import numpy as np

from .units import EV_KCAL


def rmse_kcal(pred, ref):
    """Per-state RMSE in kcal/mol."""
    pred, ref = np.atleast_2d(pred), np.atleast_2d(ref)
    n = min(pred.shape[1], ref.shape[1])
    return np.sqrt(np.mean((pred[:, :n] - ref[:, :n]) ** 2, axis=0)) * EV_KCAL


def mue_mev(pred, ref):
    pred, ref = np.atleast_2d(pred), np.atleast_2d(ref)
    n = min(pred.shape[1], ref.shape[1])
    return np.mean(np.abs(pred[:, :n] - ref[:, :n]), axis=0) * 1000


def gap_mev(model_or_fn, coords, i=0, j=1):
    """Mean |E_j - E_i| (meV) at the given geometries (e.g. points on an intersection seam, where it should be 0)."""
    f = model_or_fn.predict if hasattr(model_or_fn, 'predict') else model_or_fn
    E = f(np.atleast_2d(coords))
    return float(np.mean(np.abs(E[:, j] - E[:, i])) * 1000)


def permutation_error_mev(model_or_fn, coords, perms):
    """Mean |E(P x) - E(x)| (meV) over the given permutations of the coordinate columns."""
    f = model_or_fn.predict if hasattr(model_or_fn, 'predict') else model_or_fn
    coords = np.atleast_2d(coords)
    E0 = f(coords)
    return float(np.mean([np.abs(f(coords[:, p]) - E0).mean() for p in perms[1:]]) * 1000)


def summary_table(rows: dict, columns: list[str]):
    """Plain-text table: rows = {name: [values...]}."""
    w = max(len(k) for k in rows) + 2
    head = ''.join(f'{c:>12s}' for c in columns)
    lines = [' ' * w + head]
    for k, v in rows.items():
        lines.append(f'{k:<{w}s}' + ''.join(f'{x:12.2f}' if isinstance(x, (float, np.floating)) else f'{str(x):>12s}' for x in v))
    return '\n'.join(lines)
