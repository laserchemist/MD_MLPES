"""Smoke test of the CH2OO workflow on the shipped CASSCF data (no quantum chemistry needed)."""
import json
import re

import numpy as np

from diabml.workflow import ch2oo as W


def test_quick_workflow(tmp_path):
    run_dir = tmp_path / 'quick'
    W.stage_init(run_dir, 'quick')
    cfg = (run_dir / 'config.toml').read_text()
    cfg = re.sub(r'epochs = \d+', 'epochs = 60', cfg).replace('n_train = [20, 60]', 'n_train = [20]')
    cfg = cfg.replace('weights = [0.0, 0.1, 1.0, 10.0]', 'weights = [0.0, 1.0]')
    (run_dir / 'config.toml').write_text(cfg)
    W.main(['all', str(run_dir)])
    m = json.loads((run_dir / 'results' / 'metrics.json').read_text())
    assert m['n_usable'] == 120
    for k in ('random_split', 'extrapolation', 'learning_curve', 'restraint'):
        assert k in m
    for f in ('F1_ref_cuts.png', 'F2_ref_character.png', 'F5_parity.png', 'F6_extrapolation.png', 'F9_restraint.png'):
        assert (run_dir / 'figures' / f).exists()
    for t in ('T1_reference_summary', 'T2_ladder', 'T3_model_errors', 'T5_restraint'):
        assert (run_dir / 'tables' / f'{t}.csv').exists()
    assert (run_dir / 'report.html').exists()


def test_asymptotes_follow_r_co():
    from diabml import RefSet
    coords = np.array([[8.0, 117.7, 1.2], [8.0, 117.7, 1.4], [1.35, 117.7, 1.2], [1.35, 117.7, 1.4]])
    E = np.array([[0, 0, 0, 1.0], [1, 1, 1, 2.0], [-1, 3, 6, 9], [-1, 3, 6, 9.0]])
    ref = RefSet('t', coords, E, ['r_OO', 'a_COO', 'r_CO'])
    a = W.Asymptotes(ref, 7.0)
    out = a(np.array([[2.0, 117.7, 1.3]]))[0]
    assert np.allclose(out[:4], [0.5, 0.5, 0.5, 1.5]) and np.isclose(out[4], 7.5)


def test_framed_couplings_can_learn():
    """With slot_bias on 'coup', the energy loss reaches the coupling network (it is exactly zero at zero coupling)."""
    import torch
    from diabml import RefSet
    from diabml.frames import ChannelFrame, Diabat
    from diabml.models import FramedModel
    ref = RefSet.load(W.C.PACKAGE_ROOT / 'data' / 'ch2oo_casscf.npz')
    fr = ChannelFrame([Diabat('a', 0.0), Diabat('b', 0.5), Diabat('c', 7.0, charge_product=-1.0)], 2)
    grads = {}
    for bias in (None, {'coup': 0.1}):
        m = FramedModel(fr, 3, hidden=(8,), slot_bias=bias)
        c = m.prepare(ref.coords)
        ((m(c) - torch.tensor(ref.energies[:, :3])) ** 2).mean().backward()
        grads[bias is None] = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if 'coup' in n)
    assert grads[True] == 0.0 and grads[False] > 0.0
