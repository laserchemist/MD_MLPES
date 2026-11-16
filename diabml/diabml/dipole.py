"""Dipole surface for CH2OO: learned atomic charges, mu = sum_i q_i(x) r_i.

Each member of a DipoleModel is a small network from the internal coordinates x (the same ones the energy surface
uses) to the charges of C, O1, O2 and H1; H2 carries the rest, so the molecule stays neutral. The dipole is
sum_i q_i r_i with the atoms where the internal coordinates put them (diabml.geometry.ch2oo_xyz: O1 at the origin,
C on -x, the molecule in the xy plane). This charge form rotates with the molecule by construction and carries over
unchanged to models with more coordinates (planar or full-dimensional CH2OO): only the list of coordinates and the
geometry builder grow. The members are averaged.

For dynamics: body_dipole(x) gives the dipole in the molecule's frame from the internal coordinates; lab_dipole
rotates it onto the actual orientation of C, O1, O2 in a trajectory. Using the ideal positions keeps motions the
model does not describe (the restraint springs on the CH2 frame of a reduced model) out of the dipole.

    m = DipoleModel.load('runs/23_md9/models/dipole.pt')
    mu = m.body_dipole(X)                 # (N, 3) in e*Angstrom; * 4.8032 -> Debye
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

EA_DEBYE = 4.80320471            # 1 e*Angstrom in Debye
AU_EA = 0.529177210903           # 1 a.u. of dipole (e*bohr) in e*Angstrom
_DEF = dict(r_OO=1.345, a_COO=117.7, r_CO=1.272, r_CH=1.080, a_HCO=120.0)


def ch2oo_positions(X, names, fixed=None):
    """Torch (N, 5, 3) positions of C, O1, O2, H1, H2 from internal coordinates (columns `names`), the rest from
    `fixed` (e.g. {'r_CH': 1.08, 'a_HCO': 120}) or CH2OO reference values. Same construction as ch2oo_xyz."""
    X = torch.as_tensor(np.asarray(X, float)) if not isinstance(X, torch.Tensor) else X
    N = X.shape[0]
    val = {}
    for k, v in _DEF.items():
        f = None if fixed is None else {kk.lower(): vv for kk, vv in fixed.items()}.get(k.lower())
        val[k] = torch.full((N,), float(f if f is not None else v), dtype=X.dtype)
    for j, n in enumerate(names):
        val[n] = X[:, j]
    z = torch.zeros(N, dtype=X.dtype)
    C = torch.stack([-val['r_CO'], z, z], 1)
    O1 = torch.zeros(N, 3, dtype=X.dtype)
    t = torch.deg2rad(180.0 - val['a_COO'])
    O2 = torch.stack([val['r_OO'] * torch.cos(t), val['r_OO'] * torch.sin(t), z], 1)
    a = torch.deg2rad(180.0 - val['a_HCO'])
    H1 = C + val['r_CH'][:, None] * torch.stack([-torch.cos(a), torch.sin(a), z], 1)
    H2 = C + val['r_CH'][:, None] * torch.stack([-torch.cos(a), -torch.sin(a), z], 1)
    return torch.stack([C, O1, O2, H1, H2], 1)


class _Charges(torch.nn.Module):
    def __init__(self, n_in, hidden=(64, 64), seed=0):
        super().__init__()
        torch.manual_seed(seed)
        layers, d = [], n_in
        for h in hidden:
            layers += [torch.nn.Linear(d, h, dtype=torch.float64), torch.nn.GELU()]
            d = h
        layers += [torch.nn.Linear(d, 4, dtype=torch.float64)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, z):
        q = self.net(z)                                           # C, O1, O2, H1
        return torch.cat([q, -q.sum(1, keepdim=True)], 1)         # H2: neutral molecule


class DipoleModel(torch.nn.Module):
    def __init__(self, names, fixed=None, n_members=5, hidden=(64, 64), mean=None, std=None):
        super().__init__()
        self.names, self.fixed, self.hidden = list(names), dict(fixed or {}), tuple(hidden)
        self.members = torch.nn.ModuleList([_Charges(len(names), hidden, seed=s) for s in range(n_members)])
        self.register_buffer('mean', torch.as_tensor(np.zeros(len(names)) if mean is None else mean, dtype=torch.float64))
        self.register_buffer('std', torch.as_tensor(np.ones(len(names)) if std is None else std, dtype=torch.float64))
        self.meta = {}

    def _z(self, X):
        X = torch.as_tensor(np.asarray(X, float)) if not isinstance(X, torch.Tensor) else X
        return X, (X - self.mean) / self.std

    def charges(self, X):
        """(n_members, N, 5) charges (e)."""
        X, z = self._z(X)
        return torch.stack([m(z) for m in self.members])

    def member_dipoles(self, X):
        """(n_members, N, 3) body-frame dipoles (e*Angstrom), torch."""
        X, z = self._z(X)
        R = ch2oo_positions(X, self.names, self.fixed)
        return torch.stack([torch.einsum('na,nax->nx', m(z), R) for m in self.members])

    def body_dipole(self, X):
        """(N, 3) ensemble-mean body-frame dipole (e*Angstrom), numpy."""
        with torch.no_grad():
            return self.member_dipoles(X).mean(0).numpy()

    # ---------------------------------------------------------------- io
    def save(self, path):
        path = Path(path)
        spec = {'format': 'diabml-dipole', 'names': self.names, 'fixed': self.fixed, 'hidden': list(self.hidden),
                'n_members': len(self.members), 'meta': self.meta}
        torch.save({'spec': spec, 'state_dict': self.state_dict()}, path)
        path.with_suffix('.json').write_text(json.dumps(spec, indent=1, default=float))
        return path

    @classmethod
    def load(cls, path):
        d = torch.load(path, weights_only=True)
        s = d['spec']
        m = cls(s['names'], s['fixed'], s['n_members'], tuple(s['hidden']))
        m.load_state_dict(d['state_dict'])
        m.meta = s.get('meta', {})
        m.eval()
        return m


def fit_dipole_model(X, mu, names, fixed=None, n_members=5, hidden=(64, 64), epochs=3000, lr=3e-3, val_frac=0.1,
                     seed=0, log=None):
    """Fit a DipoleModel to body-frame dipoles mu (N, 3; e*Angstrom). Each member: its own seed and random
    90/10 split, Adam + cosine schedule, best epoch by validation error. Returns (model, per-member validation RMSE
    in Debye)."""
    X = np.asarray(X, float)
    mu = np.asarray(mu, float)
    model = DipoleModel(names, fixed, n_members, hidden, mean=X.mean(0), std=X.std(0) + 1e-12)
    Xt, Yt = torch.tensor(X), torch.tensor(mu)
    R = ch2oo_positions(Xt, names, fixed)
    Z = (Xt - model.mean) / model.std
    val = []
    for k, m in enumerate(model.members):
        p = np.random.default_rng(seed + k).permutation(len(X))
        nv = max(1, int(val_frac * len(X)))
        va, tr = p[:nv], p[nv:]
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best = (np.inf, None)
        for ep in range(epochs):
            opt.zero_grad()
            pred = torch.einsum('na,nax->nx', m(Z[tr]), R[tr])
            loss = ((pred - Yt[tr]) ** 2).sum(1).mean()
            loss.backward(); opt.step(); sch.step()
            if ep % 25 == 0 or ep == epochs - 1:
                with torch.no_grad():
                    pv = torch.einsum('na,nax->nx', m(Z[va]), R[va])
                    v = float(torch.sqrt(((pv - Yt[va]) ** 2).sum(1).mean()))
                if v < best[0]:
                    best = (v, {kk: t.clone() for kk, t in m.state_dict().items()})
        m.load_state_dict(best[1])
        val.append(best[0] * EA_DEBYE)
        if log:
            log(f'  dipole member {k}: validation RMSE {best[0] * EA_DEBYE:.3f} D')
    model.eval()
    return model, val


def body_axes(P):
    """(N, 3, 3) rows = body x, y, z axes from Cartesian positions P (N, 5, 3) of C, O1, O2, H, H:
    x along C -> O1, y in the C-O1-O2 plane towards O2, z = x cross y (the frame of ch2oo_xyz)."""
    P = np.asarray(P, float)
    ex = P[:, 1] - P[:, 0]
    ex /= np.linalg.norm(ex, axis=1, keepdims=True)
    v = P[:, 2] - P[:, 1]
    ey = v - np.sum(v * ex, 1, keepdims=True) * ex
    ey /= np.linalg.norm(ey, axis=1, keepdims=True)
    ez = np.cross(ex, ey)
    return np.stack([ex, ey, ez], 1)


def lab_dipole(mu_body, P):
    """Rotate body-frame dipoles (N, 3) onto the orientation of the molecule in positions P (N, 5, 3)."""
    return np.einsum('ni,nij->nj', np.asarray(mu_body, float), body_axes(P))
