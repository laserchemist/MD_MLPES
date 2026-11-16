"""Active learning without quantum chemistry: propose on the shipped data, fake the new references, merge, train."""
import json
import re

import numpy as np

from diabml import RefSet
from diabml.workflow import active as A
from diabml.workflow import ch2oo as W


def _base(tmp_path):
    ref = RefSet.load(W.C.PACKAGE_ROOT / 'data' / 'ch2oo_casscf.npz')
    ref.meta = dict(ref.meta, spec={'ncas': 6, 'nelecas': 8, 'nroots': 4, 'basis': '6-31g*', 'symmetry': 'Cs',
                                    'wfnsym': "A'", 'spin_ss': 0.0, 'max_cycle': 100},
                    zero_hartree={'casscf': float(ref.meta['zero_hartree'])}, energy_level='casscf')
    d = tmp_path / 'base' / 'data'
    ref.save(d / 'refs.npz')
    idx = np.where((ref.coords[:, 0] <= 4.0) & W.cut_mask(ref))[0][:10]
    np.savez(d / 'nacs.npz', r=ref.coords[idx, 0], idx=idx, coords=ref.coords[idx], pairs=np.array([[0, 1], [1, 2]]),
             tau=np.full((len(idx), 2), 0.3), vectors=np.zeros((len(idx), 2, 5, 3)), E_eV=ref.energies[idx],
             converged=np.ones(len(idx), bool), energy_mismatch_eV=np.zeros(len(idx)),
             cut=json.dumps(W.slice_values(ref)))
    return ref


def test_active_round(tmp_path):
    base = _base(tmp_path)
    run_dir = tmp_path / 'r1'
    W.stage_init(run_dir, 'active')
    cfg = (run_dir / 'config.toml').read_text()
    cfg = cfg.replace('../03_cas_3d/data/refs.npz', '../base/data/refs.npz')
    cfg = re.sub(r'epochs = \d+', 'epochs = 40', cfg)
    cfg = re.sub(r'committee = \d+', 'committee = 2', cfg)
    cfg = re.sub(r'n_candidates = \d+', 'n_candidates = 400', cfg)
    cfg = re.sub(r'n_new = \d+', 'n_new = 5', cfg)
    cfg = re.sub(r'restarts = \d+', 'restarts = 1', cfg).replace('seeds = [0, 1, 2]', 'seeds = [0]')
    (run_dir / 'config.toml').write_text(cfg)
    W.main(['refs', str(run_dir)])
    W.main(['propose', str(run_dir)])
    z = np.load(run_dir / 'data' / 'proposed.npz')
    X = z['coords']
    assert 1 <= len(X) <= 5
    assert np.all(X[:, 0] <= 3.0) and not np.any((X[:, 0] <= 1.5) & (X[:, 1] <= 85.0))     # bounds and exclude box
    # stand-in for the compute stage: "references" = committee mean + 0.05 eV, on the base data's zero
    run = W.Run(run_dir)
    n = len(X)
    new = RefSet(base.name, X, z['mean'] + 0.05, list(base.coord_names),
                 extra={'q_O_terminal': np.full((n, 4), -0.2), 'n_unpaired': np.zeros((n, 4)),
                        'converged': np.ones(n, bool), 'active_space_consistent': np.ones(n, bool),
                        'seconds': np.ones(n), 'E_casscf_eV': z['mean'] + 0.05, 'al_round': np.ones(n, int)},
                 meta=dict(RefSet.load(run.p('data', 'base_refs.npz')).meta))
    new.save(run.p('data', 'new_refs.npz'))
    np.savez(run.p('data', 'new_nacs.npz'), tau=np.full((n, 2), 0.5), vectors=np.zeros((n, 2, 5, 3)),
             pairs=np.array([[0, 1], [1, 2]]), E_eV=z['mean'])
    W.main(['merge', str(run_dir)])
    merged = RefSet.load(run.p('data', 'refs.npz'))
    assert len(merged) == 120 + n
    assert np.array_equal(merged.extra['al_round'], np.r_[np.zeros(120, int), np.ones(n, int)])
    assert np.allclose(merged.coords[120:], X)
    zn = np.load(run.p('data', 'nacs.npz'))
    assert len(zn['idx']) == 10 + n and np.array_equal(zn['idx'][10:], 120 + np.arange(n))
    # the grid figures still see the original grid
    assert W.slice_values(merged) == W.slice_values(base)
    W.main(['train', str(run_dir)])
    W.main(['figures', str(run_dir)])
    W.main(['tables', str(run_dir)])
    W.main(['report', str(run_dir)])
    assert (run_dir / 'figures' / 'A1_proposal.png').exists()
    assert (run_dir / 'tables' / 'T9_active_learning.csv').exists()
    assert 'A1_proposal.png' in (run_dir / 'report.html').read_text()
    m = json.loads((run_dir / 'results' / 'metrics.json').read_text())
    assert m['n_points'] == 120 + n


def test_adiabatic_order_and_coupling_remap(tmp_path):
    """Rows whose (state-specific NEVPT2) energies are out of order are sorted, per-state charges move with them,
    and couplings computed between SA-CASSCF roots are relabelled to the sorted states."""
    E = np.array([[0.0, 1.0, 2.0, 3.0], [1.2, 1.0, 2.0, 3.0]])
    q = np.array([[-0.1, -0.2, -0.3, -0.4], [-0.1, -0.2, -0.3, -0.4]])
    ref = RefSet('t', np.array([[1.4, 117.7], [1.5, 117.7]]), E.copy(), ['r_OO', 'a_COO'], extra={'q_O_terminal': q.copy()})
    out = W.adiabatic_order(ref)
    assert np.allclose(out.energies[1], [1.0, 1.2, 2.0, 3.0]) and np.allclose(out.extra['q_O_terminal'][1, :2], [-0.2, -0.1])
    assert out.meta['n_reordered'] == 1
    run_dir = tmp_path / 'r'
    W.stage_init(run_dir, 'quick')
    np.savez(run_dir / 'data' / 'nacs.npz', idx=np.array([0, 1]), pairs=np.array([[0, 1], [1, 2]]),
             tau=np.array([[0.5, 0.7], [0.5, 0.7]]))
    tau, pairs = W.load_nacs(W.Run(run_dir), out)
    assert np.allclose(tau[0], [0.5, 0.7])
    # row 1: state 0 = root 1, state 1 = root 0, state 2 = root 2 -> (0,1) is roots (1,0); (1,2) is roots (0,2): not computed
    assert np.isclose(tau[1, 0], 0.5) and np.isnan(tau[1, 1])


def test_explore_round(tmp_path):
    """Active learning from dynamics: watched trajectories on the committee, flagged geometries -> proposed.npz, and
    the dynamics log; compute is replaced by a stand-in (no quantum chemistry in the tests)."""
    _base(tmp_path)
    run_dir = tmp_path / 'md1'
    W.stage_init(run_dir, 'explore')
    cfg = (run_dir / 'config.toml').read_text()
    cfg = cfg.replace('../03_cas_3d/data/refs.npz', '../base/data/refs.npz')
    cfg = re.sub(r'epochs = \d+', 'epochs = 40', cfg)
    cfg = re.sub(r'committee = \d+', 'committee = 2', cfg)
    cfg = re.sub(r'(?m)^r_oo = \[.*\]', 'r_oo = [1.6]', cfg)
    cfg = re.sub(r'(?m)^temperatures = \[.*\]', 'temperatures = [1500.0]', cfg)
    cfg = re.sub(r'(?m)^steps = \d+', 'steps = 200', cfg)
    cfg = re.sub(r'(?m)^spread_eV = [0-9.]+', 'spread_eV = 0.0001', cfg)        # untrained committee: always stops
    cfg = re.sub(r'restarts = \d+', 'restarts = 1', cfg).replace('seeds = [0, 1, 2]', 'seeds = [0]')
    (run_dir / 'config.toml').write_text(cfg)
    W.main(['refs', str(run_dir)])
    W.main(['explore', str(run_dir)])
    z = np.load(run_dir / 'data' / 'proposed.npz')
    assert len(z['coords']) >= 1 and z['coords'].shape[1] == 2
    e = json.loads((run_dir / 'results' / 'explore.json').read_text())
    assert e['n_completed'] == 0 and e['trajectories'][0]['stopped'].startswith('spread')
    assert (run_dir / 'dynamics.html').exists() and 'D1_energies' in (run_dir / 'dynamics.html').read_text()
    W.main(['tables', str(run_dir)])
    W.main(['report', str(run_dir)])
    assert (run_dir / 'tables' / 'T11_dynamics_exploration.csv').exists()
    assert 'dynamics.html' in (run_dir / 'report.html').read_text()
