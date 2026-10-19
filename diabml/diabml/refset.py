"""RefSet: a small container for reference data (geometries, multistate energies, diagnostics).

Everything is stored in one .npz file so a notebook can load a data set without any quantum chemistry installed.
Energies are kept in eV relative to a stated zero (``meta['zero']``), with the absolute zero in hartree recorded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RefSet:
    name: str
    coords: np.ndarray                  # (N, d) the coordinates models see (e.g. distances or internal coords)
    energies: np.ndarray                # (N, n_states) adiabatic energies in eV relative to meta['zero']
    coord_names: list[str]
    extra: dict[str, np.ndarray] = field(default_factory=dict)   # diagnostics, tags, raw geometries ...
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basic access
    def __len__(self):
        return len(self.coords)

    @property
    def n_states(self):
        return self.energies.shape[1]

    def subset(self, idx) -> "RefSet":
        idx = np.asarray(idx)
        return RefSet(self.name, self.coords[idx], self.energies[idx], list(self.coord_names),
                      {k: v[idx] if (isinstance(v, np.ndarray) and len(v) == len(self)) else v for k, v in self.extra.items()},
                      dict(self.meta))

    def where(self, mask) -> "RefSet":
        return self.subset(np.where(mask)[0])

    def split(self, n_train, n_val, seed=0, pool=None):
        """Random split into train / val / test index arrays (optionally drawn from a pool of indices)."""
        pool = np.arange(len(self)) if pool is None else np.asarray(pool)
        p = np.random.default_rng(seed).permutation(pool)
        return p[:n_train], p[n_train:n_train + n_val], p[n_train + n_val:]

    # ------------------------------------------------------------------ io
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {'coords': self.coords, 'energies': self.energies}
        arrays.update({'extra__' + k: np.asarray(v) for k, v in self.extra.items()})
        header = {'name': self.name, 'coord_names': self.coord_names, 'meta': self.meta}
        np.savez(path, __header__=np.array(json.dumps(header)), **arrays)

    @classmethod
    def load(cls, path) -> "RefSet":
        z = np.load(path, allow_pickle=False)
        h = json.loads(str(z['__header__']))
        extra = {k[len('extra__'):]: z[k] for k in z.files if k.startswith('extra__')}
        return cls(h['name'], z['coords'], z['energies'], h['coord_names'], extra, h['meta'])

    def __repr__(self):
        return (f"RefSet('{self.name}', {len(self)} geometries, {self.n_states} states, coords={self.coord_names}, "
                f"extras={sorted(self.extra)})")
