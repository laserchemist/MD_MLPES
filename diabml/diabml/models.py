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
  EnsembleModel  several trained matrix models used as one production surface (averaged adiabatic energies).
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
        self.n_in, self.hidden = n_in, tuple(hidden)
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

    def matrix_t(self, coords):
        """Matrix from a torch tensor of coordinates, differentiable with respect to them (forces)."""
        mean = torch.as_tensor(np.asarray(self.scaler[0], float), dtype=coords.dtype)
        std = torch.as_tensor(np.asarray(self.scaler[1], float), dtype=coords.dtype)
        return self._matrix({'x': (coords - mean) / std})

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
        self.hidden = tuple(hidden)
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

    def matrix_t(self, coords):
        """Matrix from a torch tensor of coordinates, differentiable with respect to them (needs frame.prepare_torch)."""
        return self._matrix(self.frame.prepare_torch(coords))

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


class EnsembleModel(_Base):
    """Several trained matrix models used as one production surface.

    mode = 'adiabatic' (default): the ensemble's state k is the mean of the members' state-k eigenvalues, and its
    forces the mean of their forces (so energy is conserved); the ASE calculator also averages structure weights and
    phase-aligned derivative couplings, and reports each member's energies as an uncertainty. This is the safe
    average: two equally good members can describe the same states with different matrices, because the three
    CH2O + O(1D) structures share one asymptote and any rotation among them changes the matrix but no energy.
    `matrix()` returns member 0's matrix (for figures of matrix elements and structure weights).

    mode = 'matrix': the members' matrices are averaged after aligning structure signs (align_signs). Only
    meaningful when every member uses the same diabatic representation; kept for comparison."""

    def __init__(self, members, signs=None, mode='adiabatic'):
        super().__init__()
        self.members = torch.nn.ModuleList(members)
        self.n_states = members[0].n_states
        self.mode = mode
        signs = np.ones((len(members), 1)) if signs is None else np.asarray(signs, float)
        self.register_buffer('signs', torch.tensor(signs, dtype=torch.float64))

    @property
    def frame(self):
        return getattr(self.members[0], 'frame', None)

    def prepare(self, coords):
        return np.atleast_2d(np.asarray(coords, float))

    def _combine(self, Ms):
        S = self.signs
        if S.shape[1] == 1:
            return torch.stack(Ms).mean(0)
        out = 0
        for k, M in enumerate(Ms):
            out = out + S[k][None, :, None] * M * S[k][None, None, :]
        return out / len(Ms)

    def matrix_average(self, coords):
        X = self.prepare(coords)
        return self._combine([m.matrix(X) for m in self.members])

    def matrix(self, coords):
        if self.mode == 'matrix':
            return self.matrix_average(coords)
        return self.members[0].matrix(self.prepare(coords))

    def matrix_t(self, coords):
        if self.mode == 'matrix':
            return self._combine([m.matrix_t(coords) for m in self.members])
        return self.members[0].matrix_t(coords)

    def forward(self, cache):
        return torch.tensor(self.predict(cache))

    def member_predictions(self, coords):
        """(K, N, n_states): each member's own eigenvalues (their spread is the ensemble's uncertainty)."""
        return np.stack([m.predict(coords) for m in self.members])

    def predict(self, coords):
        if self.mode == 'matrix':
            with torch.no_grad():
                return eigvals_sorted(self.matrix_average(coords))[:, :self.n_states].numpy()
        return self.member_predictions(coords).mean(0)

    def align_signs(self, coords):
        """Choose each member's structure signs to agree best with member 0 over the given geometries."""
        import itertools
        with torch.no_grad():
            Ms = [m.matrix(np.atleast_2d(coords)).numpy() for m in self.members]
        n = Ms[0].shape[1]
        iu = np.triu_indices(n, 1)
        ref = Ms[0][:, iu[0], iu[1]]
        S = [np.ones(n)]
        for M in Ms[1:]:
            best = (-np.inf, None)
            for t in itertools.product([1.0, -1.0], repeat=n - 1):
                s = np.r_[1.0, t]
                score = float(np.sum(ref * (s[iu[0]] * s[iu[1]]) * M[:, iu[0], iu[1]]))
                if score > best[0]:
                    best = (score, s)
            S.append(best[1])
        self.signs = torch.tensor(np.array(S), dtype=torch.float64)
        return self.signs.numpy()


# ---------------------------------------------------------------------------------------------------- saving
def model_spec(model) -> dict:
    """Plain-data description of a model (no code, no pickles): enough to rebuild it with load_model."""
    if isinstance(model, IndependentNN):
        raise TypeError('IndependentNN is a negative control and is not saved')
    if isinstance(model, EnsembleModel):
        return {'class': 'EnsembleModel', 'members': [model_spec(m) for m in model.members],
                'signs': model.signs.numpy().tolist(), 'mode': model.mode}
    if isinstance(model, CDNN):
        return {'class': 'CDNN', 'n_in': model.n_in, 'n_states': model.n_states, 'n_matrix': model.n,
                'hidden': list(model.hidden),
                'scaler': [np.asarray(v, float).tolist() for v in model.scaler]}
    if isinstance(model, FramedModel):
        if not hasattr(model.frame, 'spec'):
            raise TypeError(f'{type(model.frame).__name__} has no spec(); it cannot be saved this way')
        return {'class': 'FramedModel', 'n_states': model.n_states, 'hidden': list(model.hidden),
                'slots': list(model.nets.keys()), 'slot_bias': model.slot_bias, 'frame': model.frame.spec()}
    raise TypeError(type(model).__name__)


def save_model(model, path, meta=None):
    """Save weights and a JSON spec together: path (.pt, tensors + plain data) and path.json (readable copy).

    The .pt file loads with torch.load(..., weights_only=True): no pickled classes, so it survives code changes
    that keep the architecture, and loading it cannot run code."""
    import json
    from pathlib import Path
    path = Path(path)
    def plain(o):                     # numpy scalars / arrays -> Python, so torch.load(weights_only=True) accepts it
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o).__name__)
    spec = {'format': 'diabml-model', 'version': 1, 'model': model_spec(model), 'meta': meta or {}}
    spec = json.loads(json.dumps(spec, default=plain))
    torch.save({'spec': spec, 'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)
    shapes = {k: list(v.shape) for k, v in model.state_dict().items()}
    path.with_suffix('.json').write_text(json.dumps(dict(spec, parameter_shapes=shapes,
                                                         n_parameters=int(sum(v.numel() for v in model.state_dict().values()))),
                                                    indent=1, ensure_ascii=False))
    return path


def load_model(path):
    """Rebuild a model saved with save_model. Returns the model (eval mode); its meta is in model.meta."""
    d = torch.load(path, weights_only=True)
    m = _build(d['spec']['model'])
    m.load_state_dict(d['state_dict'])
    m.meta = d['spec'].get('meta', {})
    return m.eval()


def _build(spec):
    if spec['class'] == 'EnsembleModel':
        return EnsembleModel([_build(s) for s in spec['members']], signs=spec['signs'],
                             mode=spec.get('mode', 'adiabatic'))
    if spec['class'] == 'CDNN':
        m = CDNN(spec['n_in'], spec['n_states'], n_matrix=spec['n_matrix'], hidden=tuple(spec['hidden']),
                 scaler=tuple(np.array(v) for v in spec['scaler']))
    elif spec['class'] == 'FramedModel':
        from .frames.channel import ChannelFrame
        frames = {'ChannelFrame': ChannelFrame}
        fr = frames[spec['frame']['type']].from_spec(spec['frame'])
        m = FramedModel(fr, spec['n_states'], hidden=tuple(spec['hidden']), use_slots=spec['slots'],
                        slot_bias=spec['slot_bias'])
    else:
        raise ValueError(spec['class'])
    return m


# ---------------------------------------------------------------------------------------------------- training
def train(model, coords, energies, val_coords=None, val_energies=None, epochs=4000, lr=1e-3, alpha2=1e-6,
          restarts=3, augment=None, log=None, weights=None, penalty=None, sample_weights=None, val_sample_weights=None):
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
    # per-point weights (normalised to mean 1): e.g. more weight near a minimum for spectroscopic accuracy
    sw = None if sample_weights is None else torch.tensor(np.asarray(sample_weights, float) /
                                                         np.mean(sample_weights))[:, None]
    vw = None if val_sample_weights is None else torch.tensor(np.asarray(val_sample_weights, float) /
                                                             np.mean(val_sample_weights))[:, None]
    best_all = (np.inf, None)
    for rs in range(restarts):
        model.reset(rs)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best = (np.inf, None)
        for ep in range(epochs):
            opt.zero_grad()
            r2 = w * (model(c_tr) - y_tr) ** 2
            loss = (r2 if sw is None else sw * r2).mean() / 2
            loss = loss + alpha2 / 2 * sum((p ** 2).sum() for n, p in model.named_parameters() if 'weight' in n)
            if penalty is not None:
                loss = loss + penalty(model, c_tr)
            loss.backward()
            opt.step(); sched.step()
            if ep % 50 == 0 or ep == epochs - 1:
                with torch.no_grad():
                    if has_val:
                        a = (model(c_va) - y_va).abs()
                        v = (a if vw is None else vw * a).mean().item()
                    else:
                        a = (model(c_tr) - y_tr).abs()
                        v = (a if sw is None else sw * a).mean().item()
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
