"""Production ensemble: sign alignment, save/load, ASE forces, and the production stage on the shipped data."""
import json
import re

import numpy as np
import torch

from diabml import RefSet
from diabml.models import EnsembleModel, FramedModel, load_model, save_model
from diabml.workflow import ch2oo as W


def _member(seed):
    from diabml.frames import ChannelFrame, Diabat
    fr = ChannelFrame([Diabat('a', 0.0), Diabat('b', 0.5), Diabat('c', 7.0, charge_product=-1.0)], 2)
    m = FramedModel(fr, 3, hidden=(8,), slot_bias={'coup': 0.1})
    m.reset(seed)
    ref = RefSet.load(W.C.PACKAGE_ROOT / 'data' / 'ch2oo_casscf.npz')
    fr.fit_scaler(ref.coords)
    return m, ref


class _Flipped(torch.nn.Module):
    """A member whose structure 1 has its sign flipped: same eigenvalues, couplings to structure 1 negated."""
    def __init__(self, m):
        super().__init__()
        self.m, self.n_states = m, m.n_states
        self.s = torch.tensor([1.0, -1.0, 1.0])

    def matrix(self, X):
        M = self.m.matrix(X)
        return self.s[None, :, None] * M * self.s[None, None, :]

    def predict(self, X):
        return self.m.predict(X)


def test_sign_alignment_and_io(tmp_path):
    a, ref = _member(0)
    X = ref.coords[:40]
    ens = EnsembleModel([a, _Flipped(a)], mode="matrix")
    S = ens.align_signs(X)
    assert np.allclose(S[0], [1, 1, 1]) and np.allclose(S[1], [1, -1, 1])
    with torch.no_grad():
        assert np.allclose(ens.matrix(X).numpy(), a.matrix(X).numpy())       # aligned average of identical members
    b, _ = _member(1)
    ens2 = EnsembleModel([a, b])
    ens2.align_signs(X)
    save_model(ens2, tmp_path / 'e.pt', meta={'coord_names': ['r_OO', 'a_COO']})
    back = load_model(tmp_path / 'e.pt')
    assert np.allclose(back.predict(X), ens2.predict(X))
    assert back.member_predictions(X).shape == (2, 40, 3)


def test_production_stage_and_forces(tmp_path):
    run_dir = tmp_path / 'p'
    W.stage_init(run_dir, 'quick')
    cfg = (run_dir / 'config.toml').read_text()
    cfg = re.sub(r'epochs = \d+', 'epochs = 40', cfg)
    cfg = re.sub(r'restarts = \d+', 'restarts = 1', cfg)
    cfg += '\n[production]\nmembers = 2\nfinal = false\n'
    (run_dir / 'config.toml').write_text(cfg)
    W.main(['refs', str(run_dir)])
    W.main(['production', str(run_dir)])
    r = json.loads((run_dir / 'results' / 'production.json').read_text())
    assert r['members'] == 2 and r['regions']['all']['n'] == r['n_test']
    assert sum(v['n'] for k, v in r['regions'].items() if k in ('dynamics', 'high energy', 'corner')) == r['n_test']
    W.main(['tables', str(run_dir)])
    W.main(['report', str(run_dir)])
    assert (run_dir / 'tables' / 'T10_production_regions.csv').exists()
    assert 'P1_production_parity.png' in (run_dir / 'report.html').read_text()
    # the saved ensemble drives the ASE calculator, and its forces match finite differences
    from diabml.dynamics import CoordinateSurface, DiabmlCalculator, ch2oo_atoms, ch2oo_coordinates, reduced_restraints
    model = load_model(run_dir / 'models' / 'production.pt')
    names = model.meta['coord_names']
    atoms = ch2oo_atoms(r_oo=1.45)
    surf = reduced_restraints(CoordinateSurface(model, ch2oo_coordinates(names)), atoms.positions, free=names)
    atoms.calc = DiabmlCalculator(surf, state=0)
    F = atoms.get_forces()
    h, i = 1e-4, (2, 0)
    p = atoms.positions.copy()
    E = []
    for d in (h, -h):
        q = p.copy(); q[i] += d; atoms.positions = q
        E.append(atoms.get_potential_energy())
    atoms.positions = p
    assert abs(-(E[0] - E[1]) / (2 * h) - F[i]) < 1e-5


def test_guard_freezes_networks_below_data_and_adds_wall():
    """Below the data the learned parts stop changing (smooth clamp) and the wall raises every state equally."""
    from diabml.frames import ChannelFrame, Diabat
    fr = ChannelFrame([Diabat('a', 0.0), Diabat('b', 0.5), Diabat('c', 7.0, charge_product=-1.0)], 2)
    ref = RefSet.load(W.C.PACKAGE_ROOT / 'data' / 'ch2oo_casscf.npz')
    fr.fit_scaler(ref.coords)
    fr.set_guard([1.15, 65.0], [None, 155.0], [0.0125, 1.8], wall_r=1.3, wall_k=30.0)
    X = np.array([[0.7, 117.7], [0.9, 117.7], [1.0, 117.7]])
    f = fr.features(X)
    assert np.allclose(f[0], f[1], atol=1e-6) and np.allclose(f[1], f[2], atol=1e-4)
    base = fr.tails(X[:, 0], X)
    plain = ChannelFrame.from_spec(dict(fr.spec(), guard=None)).tails(X[:, 0], X)
    assert np.allclose(base - plain, (30.0 * (1.3 - X[:, 0]) ** 2)[:, None])
    # numpy and torch paths agree, and the guard survives save/load
    c_np = fr.prepare(X)
    c_t = ChannelFrame.from_spec(fr.spec()).prepare_torch(torch.tensor(X))
    assert np.allclose(c_np['base'].numpy(), c_t['base'].detach().numpy())
    assert np.allclose(c_np['slot_in']['diag'].numpy(), c_t['slot_in']['diag'].detach().numpy())
