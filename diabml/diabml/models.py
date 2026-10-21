"""Learned models. All share one interface so notebooks can swap them freely:

    m.prepare(coords) -> cache        geometry-dependent tensors (computed once per data set)
    m(cache)          -> (N, n_states) lowest eigenvalues in eV
    m.matrix(coords)  -> (N, n, n)    the learned matrix (CPEM / diabatic-like matrix)
    m.predict(coords) -> numpy (N, n_states)
    m.reset(seed)                     re-initialise for a new random restart

Models
  CDNN           Shu, Varga, Zhang, Meng, Parameswaran, Zhou & Truhlar, JCTC 21, 3342 (2025): coordinates -> MLP ->
                 dense symmetric matrix -> eigenvalues. No physics, no symmetry beyond what the inputs carry.
  IndependentNN  the paper's negative control: the same MLP predicting the adiabatic energies directly.
  FramedModel    a physics Frame (London, DIM, fragment channel, ...) whose slots are small MLPs, zero-initialised
                 so training starts from the physics prior.
"""
from __future__ import annotations

import numpy as np
import torch

from .frames.base import eigvals_sorted

torch.set_default_dtype(torch.float64)


def mlp(n_in, hidden, n_out, act=torch.nn.GELU):
    layers, d = [], n_in
    for h in hidden:
        layers += [torch.nn.Linear(d, h), act()]
        d = h
    layers.append(torch.nn.Linear(d, n_out))
    return torch.nn.Sequential(*layers)


def _init(module, seed, zero_last=False):
    g = torch.Generator().manual_seed(seed)
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            std = np.sqrt(2.0 / (m.in_features + m.out_features))
            with torch.no_grad():
                m.weight.copy_(torch.randn(m.weight.shape, generator=g) * std)
                m.bias.zero_()
    if zero_last:
        last = [m for m in module.modules() if isinstance(m, torch.nn.Linear)][-1]
        with torch.no_grad():
            last.weight.zero_(); last.bias.zero_()


class _Base(torch.nn.Module):
    n_states: int

    def predict(self, coords):
        with torch.no_grad():
            return self(self.prepare(coords)).numpy()

    def matrix(self, coords):
        raise NotImplementedError


class CDNN(_Base):
    """Coordinates (standardised) -> MLP -> dense symmetric n x n matrix -> lowest n_states eigenvalues."""

    def __init__(self, n_in, n_states, n_matrix=None, hidden=(256, 128, 64), scaler=None):
        super().__init__()
        self.n_states, self.n = n_states, n_matrix or n_states
        self.iu = torch.triu_indices(self.n, self.n)
        self.net = mlp(n_in, hidden, len(self.iu[0]))
        self.scaler = scaler

    def fit_scaler(self, coords):
        coords = np.asarray(coords, float)
        self.scaler = (coords.mean(0), coords.std(0) + 1e-9)
        return self

    def prepare(self, coords):
        coords = np.atleast_2d(np.asarray(coords, float))
        if self.scaler is None:
            self.fit_scaler(coords)
        return {'x': torch.tensor((coords - self.scaler[0]) / self.scaler[1])}

    def _matrix(self, cache):
        o = self.net(cache['x'])
        N = o.shape[0]
        M = torch.zeros(N, self.n, self.n)
        M[:, self.iu[0], self.iu[1]] = o
        return M + M.transpose(1, 2) - torch.diag_embed(torch.diagonal(M, dim1=1, dim2=2))

    def forward(self, cache):
        return eigvals_sorted(self._matrix(cache))[:, :self.n_states]

    def matrix(self, coords):
        return self._matrix(self.prepare(coords))

    def reset(self, seed):
        _init(self.net, seed)


class IndependentNN(CDNN):
    """Same MLP, predicting the adiabatic energies directly (no matrix, no diagonalisation)."""

    def __init__(self, n_in, n_states, hidden=(256, 128, 64), scaler=None):
        torch.nn.Module.__init__(self)
        self.n_states, self.net, self.scaler = n_states, mlp(n_in, hidden, n_states), scaler

    def forward(self, cache):
        return self.net(cache['x'])

    def matrix(self, coords):
        raise TypeError('IndependentNN has no matrix')


class FramedModel(_Base):
    """A physics Frame with each of its slots learned by a small MLP (zero-initialised => starts at the prior)."""

    def __init__(self, frame, n_states, hidden=(32, 32), use_slots=None, slot_bias=None):
        super().__init__()
        self.frame, self.n_states = frame, n_states
        # slot_bias: {slot: value} starts that slot's outputs at a constant instead of zero. Needed for couplings
        # between structures that the physics prior leaves uncoupled: at Omega_ij = 0 the eigenvalue gradient
        # d lambda_k / d Omega_ij = 2 v_ik v_jk vanishes, so a zero-initialised coupling slot never learns.
        self.slot_bias = dict(slot_bias or {})
        names = use_slots if use_slots is not None else list(frame.slots)
        self.nets = torch.nn.ModuleDict({k: mlp(frame.slots[k][0], hidden, frame.slots[k][1]) for k in names})
        self.reset(0)

    def prepare(self, coords):
        return self.frame.prepare(coords)

    def _matrix(self, cache):
        out = self.frame.zero_outputs(cache)
        for k, net in self.nets.items():
            out[k] = net(cache['slot_in'][k])
        return self.frame.assemble(cache, out)

    def forward(self, cache):
        return eigvals_sorted(self._matrix(cache))[:, :self.n_states]

    def matrix(self, coords):
        return self._matrix(self.prepare(coords))

    def reset(self, seed):
        for i, (name, net) in enumerate(self.nets.items()):
            _init(net, seed * 101 + i, zero_last=True)
            if name in getattr(self, 'slot_bias', {}):
                last = [m for m in net.modules() if isinstance(m, torch.nn.Linear)][-1]
                with torch.no_grad():
                    last.bias.fill_(self.slot_bias[name])
        hook = getattr(self, 'post_reset', None)      # optional picklable callable(model), e.g. to seed couplings
        if hook is not None:
            hook(self)


# ---------------------------------------------------------------------------------------------------- training
def train(model, coords, energies, val_coords=None, val_energies=None, epochs=4000, lr=1e-3, alpha2=1e-6,
          restarts=3, augment=None, log=None, weights=None, penalty=None):
    """Adam + cosine schedule on the CDNN cost (Shu et al. eq. 5): mean squared eigenvalue error / 2
    + alpha2/2 * sum of squared weights. Keeps the restart and epoch with the lowest validation MUE.

    augment(coords, energies) -> (coords, energies) is applied to training and validation data (e.g. adding all
    atom permutations). weights: optional per-state weights in the loss.
    penalty(model, cache) -> scalar tensor: an extra loss term evaluated on the training cache (e.g. a diabatic
    restraint on eigenvector character). Model selection still uses the validation energy error."""
    if augment is not None:
        coords, energies = augment(coords, energies)
        if val_coords is not None:
            val_coords, val_energies = augment(val_coords, val_energies)
    if hasattr(model, 'fit_scaler') and getattr(model, 'scaler', 1) is None:
        model.fit_scaler(coords)
    if hasattr(model, 'frame') and hasattr(model.frame, 'fit_scaler') and model.frame.coord_scale is None:
        model.frame.fit_scaler(coords)
    c_tr, y_tr = model.prepare(coords), torch.tensor(np.asarray(energies, float)[:, :model.n_states])
    has_val = val_coords is not None
    if has_val:
        c_va, y_va = model.prepare(val_coords), torch.tensor(np.asarray(val_energies, float)[:, :model.n_states])
    w = torch.ones(model.n_states) if weights is None else torch.tensor(weights, dtype=torch.float64)
    best_all = (np.inf, None)
    for rs in range(restarts):
        model.reset(rs)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best = (np.inf, None)
        for ep in range(epochs):
            opt.zero_grad()
            loss = (w * (model(c_tr) - y_tr) ** 2).mean() / 2
            loss = loss + alpha2 / 2 * sum((p ** 2).sum() for n, p in model.named_parameters() if 'weight' in n)
            if penalty is not None:
                loss = loss + penalty(model, c_tr)
            loss.backward()
            opt.step(); sched.step()
            if ep % 50 == 0 or ep == epochs - 1:
                with torch.no_grad():
                    v = ((model(c_va) - y_va).abs().mean() if has_val else (model(c_tr) - y_tr).abs().mean()).item()
                if v < best[0]:
                    best = (v, {k: x.clone() for k, x in model.state_dict().items()})
        if log:
            log(f'  restart {rs}: best validation MUE {best[0] * 1000:.1f} meV')
        if best[0] < best_all[0]:
            best_all = best
    model.load_state_dict(best_all[1])
    model.val_mue = best_all[0]
    return model


# ---------------------------------------------------------------------------------------------------- baselines
class KRR:
    """RBF kernel ridge regression; hyperparameters by 5-fold cross-validation on the training set."""

    def __init__(self, featurizer=None):
        self.featurizer = featurizer or (lambda x: np.asarray(x, float))

    def fit(self, coords, y, seed=0):
        X = self.featurizer(coords)
        self.m, self.s = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.m) / self.s
        y = np.asarray(y, float)
        self.mu = y.mean(0)
        yc = y - self.mu
        folds = np.array_split(np.random.default_rng(seed).permutation(len(Z)), min(5, len(Z)))
        D2 = ((Z[:, None] - Z[None]) ** 2).sum(-1)
        best = (np.inf, None)
        for g in np.geomspace(1e-3, 3, 12):
            K = np.exp(-g * D2)
            for lam in np.geomspace(1e-6, 1, 7):
                err = 0.0
                for f in folds:
                    tr = np.setdiff1d(np.arange(len(Z)), f)
                    wts = np.linalg.solve(K[np.ix_(tr, tr)] + lam * np.eye(len(tr)), yc[tr])
                    err += np.sum((K[np.ix_(f, tr)] @ wts - yc[f]) ** 2)
                if err < best[0]:
                    best = (err, (g, lam))
        self.g, lam = best[1]
        self.Z = Z
        self.w = np.linalg.solve(np.exp(-self.g * D2) + lam * np.eye(len(Z)), yc)
        return self

    def predict(self, coords):
        Z = (self.featurizer(coords) - self.m) / self.s
        return np.exp(-self.g * ((Z[:, None] - self.Z[None]) ** 2).sum(-1)) @ self.w + self.mu
