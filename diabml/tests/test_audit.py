"""Audit stage: ORCA input/output handling (a real ORCA 6.1 output) and the classification, with a mock ORCA."""
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from diabml import RefSet
from diabml.workflow import audit as AU
from diabml.workflow import ch2oo as W

DATA = Path(__file__).parent / 'data'
H = 27.211386245988


def test_parse_real_orca_output():
    r = AU.parse_orca(DATA / 'orca_ch2oo_eq_qdnevpt2.out')
    assert np.allclose(r['casscf'], [-188.6497941349, -188.4922118705, -188.3729493155, -188.2773206549])
    assert np.allclose(r['sc'], [-189.082586472, -188.940269560, -188.814694562, -188.701858914], atol=1e-8)
    assert np.allclose(r['qd'], [-189.083004, -188.939858, -188.814730, -188.701818])
    # ORCA can abort in the QD-NEVPT2 property step after printing the energies: the energies are still used
    t = (DATA / 'orca_ch2oo_eq_qdnevpt2.out').read_text()
    cut = t[:t.index('QD-NEVPT2 UV, CD spectra')] + 'ORCA finished by error termination in CASSCF\n'
    (Path(os.environ.get('TMPDIR', '/tmp')) / 'cut.out').write_text(cut)
    r2 = AU.parse_orca(Path(os.environ.get('TMPDIR', '/tmp')) / 'cut.out')
    assert r2 is not None and np.allclose(r2['qd'], r['qd']) and not r2['terminated_normally']
    inp = AU.orca_input([('C', (0, 0, 0)), ('O', (1.3, 0, 0))], {'basis': 'cc-pvdz', 'ncas': 6, 'nelecas': 8,
                        'nroots': 4, 'symmetry': 'Cs', 'wfnsym': "A'"}, moread='/x/p0001.gbw')
    assert 'MORead' in inp and '%moinp "/x/p0001.gbw"' in inp and 'QD_VanVleck' in inp and 'irrep  0' in inp
    assert 'NoFrozenCore' in inp


def test_audit_stage_with_mock_orca(tmp_path, monkeypatch):
    ref = RefSet.load(W.C.PACKAGE_ROOT / 'data' / 'ch2oo_casscf.npz')
    z = float(ref.meta['zero_hartree'])
    ref.meta = dict(ref.meta, spec={'ncas': 6, 'nelecas': 8, 'nroots': 4, 'basis': '6-31g*', 'symmetry': 'Cs',
                                    'wfnsym': "A'", 'spin_ss': 0.0, 'max_cycle': 100},
                    zero_hartree={'casscf': z, 'nevpt2': z - 0.43}, energy_level='casscf')
    ref.extra['E_casscf_eV'] = ref.energies.copy()
    ref.extra['E_nevpt2_eV'] = ref.energies.copy()
    run_dir = tmp_path / 'a'
    W.stage_init(run_dir, 'quick')
    ref.save(run_dir / 'data' / 'refs.npz')
    Ec = z + ref.energies / H
    table = {f'p{i:04d}': Ec[i].tolist() for i in range(len(ref))}
    table['p0005'] = (Ec[5] - 0.01).tolist()           # ORCA lower: ours on a higher solution
    table['p0007'] = (Ec[7] + 0.02).tolist()           # ORCA higher on the first try ...
    table['p0007_r1'] = Ec[7].tolist()                 # ... and the same solution from a neighbour's orbitals
    table['p0009'] = 'fail'
    (tmp_path / 'table.json').write_text(json.dumps(table))
    monkeypatch.setenv('MOCK_ORCA_TABLE', str(tmp_path / 'table.json'))
    W.main(['audit', str(run_dir), '--orca', str(DATA / 'mock_orca.py'), '--jobs', '4'])
    a = np.load(run_dir / 'results' / 'audit.npz')
    st = a['status']
    assert st[5] == 'ours higher' and st[7] == 'same' and st[9] == 'failed' and st[0] == 'same'
    assert int(a['attempt'][7]) == 1
    assert np.sum(st == 'same') == len(ref) - 2
    assert (run_dir / 'audit.html').exists() and 'Points on a higher solution (1)' in (run_dir / 'audit.html').read_text()
    W.main(['report', str(run_dir)])
    assert 'audit.html' in (run_dir / 'report.html').read_text()


def test_repair_after_audit(tmp_path, monkeypatch):
    ref = RefSet.load(W.C.PACKAGE_ROOT / 'data' / 'ch2oo_casscf.npz')
    z = float(ref.meta['zero_hartree'])
    ref.meta = dict(ref.meta, spec={'ncas': 6, 'nelecas': 8, 'nroots': 4, 'basis': '6-31g*', 'symmetry': 'Cs',
                                    'wfnsym': "A'", 'spin_ss': 0.0, 'max_cycle': 100},
                    zero_hartree={'casscf': z, 'nevpt2': z - 0.43}, energy_level='nevpt2',
                    method='SA4-CASSCF(8,6)/6-31g* + SC-NEVPT2')
    N = len(ref)
    ref.extra['E_casscf_eV'] = ref.energies.copy()
    ref.extra['E_nevpt2_eV'] = ref.energies.copy()
    ref.extra['q_O_terminal'] = np.full((N, 4), -0.3)
    ref.extra['casci_mismatch_hartree'] = np.zeros(N)
    run_dir = tmp_path / 'r'
    W.stage_init(run_dir, 'quick')
    ref.save(run_dir / 'data' / 'refs.npz')
    np.savez(run_dir / 'data' / 'nacs.npz', idx=np.array([3, 5, 8]), tau=np.ones((3, 2)), pairs=np.array([[0, 1], [1, 2]]),
             r=ref.coords[[3, 5, 8], 0])
    np.savez(run_dir / 'data' / 'orbitals.npz', coords=ref.coords[[3, 5, 8]], mo=np.zeros((3, 2, 2)),
             ref_irr=np.array('null'))
    Ec = z + ref.energies / H
    table = {f'p{i:04d}': Ec[i].tolist() for i in range(N)}
    table['p0005'] = (Ec[5] - 0.01).tolist()           # ours on a higher solution, by 10 mEh
    table['p0009'] = 'fail'
    (tmp_path / 'table.json').write_text(json.dumps(table))
    monkeypatch.setenv('MOCK_ORCA_TABLE', str(tmp_path / 'table.json'))
    W.main(['audit', str(run_dir), '--orca', str(DATA / 'mock_orca.py'), '--jobs', '4', ])
    W.main(['repair', str(run_dir)])
    sc = RefSet.load(run_dir / 'data' / 'repaired' / 'nevpt2' / 'refs.npz')
    qd = RefSet.load(run_dir / 'data' / 'repaired' / 'qd_nevpt2' / 'refs.npz')
    # the repaired point moved down by 10 mEh at both levels; the others are untouched
    assert np.allclose(sc.energies[5], ref.energies[5] - 0.01 * H, atol=1e-6)
    assert np.allclose(np.delete(sc.energies, 5, 0), np.delete(ref.energies, 5, 0))
    assert sc.extra['audit'][5] == 1 and sc.extra['audit'][9] == 3 and sc.extra['from_orca'].sum() == 1
    assert np.isnan(sc.extra['q_O_terminal'][5]).all() and np.isfinite(sc.extra['q_O_terminal'][4]).all()
    assert np.isfinite(sc.energies[9]).all()                           # unverified, kept
    assert np.isnan(qd.energies[9]).all() and qd.meta['energy_level'] == 'qd_nevpt2'
    assert 'qd_nevpt2' in qd.meta['zero_hartree'] and 'QD-NEVPT2' in qd.meta['method']
    # mock QD: S0 is 0.1 mEh below SC, the other states equal (up to the zero shift)
    d = (qd.energies - np.sort(sc.energies, 1)) / H - (qd.meta['zero_hartree']['nevpt2'] - qd.meta['zero_hartree']['qd_nevpt2'])
    ok = np.isfinite(d).all(1)
    assert np.allclose(d[ok][:, 0], -1e-4, atol=1e-6) and np.allclose(d[ok][:, 1:], 0, atol=1e-6)
    # couplings and orbitals at the repaired geometry are gone, in both folders
    for name in ('nevpt2', 'qd_nevpt2'):
        zn = np.load(run_dir / 'data' / 'repaired' / name / 'nacs.npz')
        assert list(zn['idx']) == [3, 8] and zn['tau'].shape == (2, 2) and zn['pairs'].shape == (2, 2)
        zo = np.load(run_dir / 'data' / 'repaired' / name / 'orbitals.npz')
        assert len(zo['coords']) == 2
    lib = json.loads((run_dir / 'data' / 'repaired' / 'nevpt2' / 'orca_orbitals.json').read_text())['points']
    assert len(lib) == N - 1 and all(q['gbw'].endswith('.gbw') for q in lib)          # every point but the failed one
    info = json.loads((run_dir / 'results' / 'repair.json').read_text())
    assert info['n_repaired'] == 1 and info['counts']['failed'] == 1
    assert (run_dir / 'figures' / 'R1_repair.png').exists()
    # a run made from the repaired set with init --refs
    new = tmp_path / 'r_sc'
    W.main(['init', str(new), '--template', 'explore', '--refs', '../r/data/repaired/nevpt2/refs.npz'])
    assert 'file = "../r/data/repaired/nevpt2/refs.npz"' in (new / 'config.toml').read_text()
    # charges missing at a point: no restraint there, and q_ion from the points that have them
    t, m = W.charge_targets(sc, np.arange(N), -0.6, 0.0)
    assert not m[5].any() and np.isfinite(t).all()


def test_compute_with_orca_keeps_the_lowest_solution(tmp_path, monkeypatch):
    ref = RefSet.load(W.C.PACKAGE_ROOT / 'data' / 'ch2oo_casscf.npz')
    z = float(ref.meta['zero_hartree'])
    ref.meta = dict(ref.meta, spec={'ncas': 6, 'nelecas': 8, 'nroots': 4, 'basis': '6-31g*', 'symmetry': 'Cs',
                                    'wfnsym': "A'", 'spin_ss': 0.0, 'max_cycle': 100},
                    zero_hartree={'casscf': z, 'nevpt2': z - 0.43}, energy_level='nevpt2')
    N = len(ref)
    ref.extra['E_casscf_eV'] = ref.energies.copy()
    ref.extra['E_nevpt2_eV'] = ref.energies.copy()
    src = tmp_path / 'base'
    src.mkdir()
    ref.save(src / 'refs.npz')
    from diabml.workflow import orca_compute as OC
    gb = []
    for i in range(3):
        (src / f'g{i}.gbw').write_text('mock')
        gb.append(str(src / f'g{i}.gbw'))
    OC.save_library(src / 'orca_orbitals.json', ref.coord_names, ref.coords[:3], gb)
    run_dir = tmp_path / 'r'
    W.main(['init', str(run_dir), '--template', 'explore', '--refs', str(src / 'refs.npz')])
    assert 'engine = "orca"' in (run_dir / 'config.toml').read_text()
    W.main(['refs', str(run_dir)])
    assert (run_dir / 'data' / 'orca_orbitals_base.json').exists()
    X = ref.coords[[4, 6]] + 0.01
    E = ref.energies[[4, 6]]
    np.savez(run_dir / 'data' / 'proposed.npz', coords=X, score=np.ones(2), mean=E, std=np.full((2, 4), 0.01),
             cand=X, cand_score=np.ones(2), round=1)
    Eh = z + E / H
    table = {'c000_0': (Eh[0] + 0.02).tolist(),         # scratch: a higher solution
             'c000_1': Eh[0].tolist(), 'c000_2': (Eh[0] + 0.001).tolist(),
             'c001_0': 'fail', 'c001_1': 'fail', 'c001_2': 'fail'}
    (tmp_path / 'table.json').write_text(json.dumps(table))
    monkeypatch.setenv('MOCK_ORCA_TABLE', str(tmp_path / 'table.json'))
    monkeypatch.setenv('ORCA_PATH', str(DATA / 'mock_orca.py'))
    W.main(['compute', str(run_dir)])
    new = RefSet.load(run_dir / 'data' / 'new_refs.npz')
    assert np.allclose(new.energies[0], E[0], atol=1e-6)                   # the lowest of three attempts
    assert np.isnan(new.energies[1]).all() and not new.extra['converged'][1]
    assert new.extra['from_orca'][0] and np.isnan(new.extra['q_O_terminal'][0]).all()
    st = json.loads((run_dir / 'results' / 'active.json').read_text())['compute']
    assert st['engine'] == 'orca' and st['points_with_several_solutions'] == 1 and st['orca_runs'] == 6
    W.main(['merge', str(run_dir)])
    merged = RefSet.load(run_dir / 'data' / 'refs.npz')
    assert len(merged) == N + 2
    lib = json.loads((run_dir / 'data' / 'orca_orbitals.json').read_text())['points']
    assert len(lib) == 4 and lib[-1]['gbw'].endswith('c000_1.gbw')


def test_fill_grid_skips_points_near_the_data(tmp_path):
    ref = RefSet.load(W.C.PACKAGE_ROOT / 'data' / 'ch2oo_casscf.npz')
    z = float(ref.meta['zero_hartree'])
    ref.meta = dict(ref.meta, spec={'ncas': 6, 'nelecas': 8, 'nroots': 4, 'basis': '6-31g*', 'symmetry': 'Cs',
                                    'wfnsym': "A'", 'spin_ss': 0.0, 'max_cycle': 100},
                    zero_hartree={'casscf': z}, energy_level='casscf')
    src = tmp_path / 'base'
    src.mkdir()
    ref.save(src / 'refs.npz')
    run_dir = tmp_path / 'f'
    W.main(['init', str(run_dir), '--template', 'explore', '--refs', str(src / 'refs.npz')])
    r0 = float(ref.coords[0, 0]); a0 = float(ref.coords[0, 1])
    with open(run_dir / 'config.toml', 'a') as fh:
        fh.write(f'\n[fill]\ncommittee = false\n\n[[fill.box]]\nr_OO = [0.9, 1.0, {r0}]\na_COO = [{a0}, 200.0]\n')
    W.main(['refs', str(run_dir)])
    W.main(['fill', str(run_dir)])
    p = np.load(run_dir / 'data' / 'proposed.npz')
    X = p['coords']
    assert len(X) == 5                                                   # 6 combinations, one is an existing point
    assert not any(np.allclose(x, [r0, a0]) for x in X)
    assert (run_dir / 'results' / 'proposed.csv').exists()


def test_spanning_tree_and_roughness():
    from diabml.workflow.track import line_roughness, spanning_tree
    g = np.array([[x, y] for x in range(4) for y in range(3)], float)
    parent, depth = spanning_tree(g, 0)
    assert parent[0] == -1 and (parent[1:] >= 0).all() and depth.max() >= 3
    for k in range(1, len(g)):                                  # every edge joins grid neighbours
        assert np.sum(np.abs(g[k] - g[parent[k]])) == 1
    E = g[:, 0] ** 2 + g[:, 1]                                  # smooth (curved): third differences vanish
    r, n = line_roughness(g, E)
    assert n == 3 and np.isclose(r, 0.0)                        # 3 lines of 4 points along x; lines along y too short
    E2 = E.copy(); E2[4] += 1.0                                 # a step raises it
    assert line_roughness(g, E2)[0] > 0.5


def test_track_stage_with_mock_orca(tmp_path, monkeypatch):
    ref = RefSet.load(W.C.PACKAGE_ROOT / 'data' / 'ch2oo_casscf.npz')
    z = float(ref.meta['zero_hartree'])
    ref.meta = dict(ref.meta, spec={'ncas': 6, 'nelecas': 8, 'nroots': 4, 'basis': '6-31g*', 'symmetry': 'Cs',
                                    'wfnsym': "A'", 'spin_ss': 0.0, 'max_cycle': 100},
                    zero_hartree={'casscf': z, 'nevpt2': z - 0.43}, energy_level='nevpt2')
    ref.extra['E_casscf_eV'] = ref.energies.copy()
    ref.extra['E_nevpt2_eV'] = ref.energies.copy()
    run_dir = tmp_path / 't'
    W.stage_init(run_dir, 'quick')
    ref.save(run_dir / 'data' / 'refs.npz')
    X = ref.coords
    lo, hi = np.sort(np.unique(X[:, 0]))[[1, 4]]
    with open(run_dir / 'config.toml', 'a') as fh:
        fh.write(f'\n[track]\njump_eV = 100.0\n[track.box]\nr_OO = [{lo}, {hi}]\na_COO = [100.0, 140.0]\n')
    monkeypatch.setenv('MOCK_ORCA_GEOM', '1')
    monkeypatch.setenv('MOCK_ORCA_TABLE', str(tmp_path / 'empty.json'))
    (tmp_path / 'empty.json').write_text('{}')
    W.main(['track', str(run_dir), '--orca', str(DATA / 'mock_orca.py'), '--jobs', '3'])
    info = json.loads((run_dir / 'results' / 'track.json').read_text())
    assert info['n'] > 3 and info['n_tracked'] == info['n'] and info['n_failed'] == 0
    t = RefSet.load(run_dir / 'data' / 'tracked' / 'refs.npz')
    inb = (X[:, 0] >= lo) & (X[:, 0] <= hi) & (X[:, 1] >= 100) & (X[:, 1] <= 140)
    assert not np.allclose(t.energies[inb], ref.energies[inb])          # replaced inside the box
    assert np.allclose(t.energies[~inb], ref.energies[~inb])            # unchanged outside
    assert (run_dir / 'data' / 'tracked' / 'orca_orbitals.json').exists()
    assert (run_dir / 'figures' / 'TR1_track.png').exists()


def test_gs_grid_stage_with_mock_orca(tmp_path, monkeypatch):
    ref = RefSet.load(W.C.PACKAGE_ROOT / 'data' / 'ch2oo_casscf.npz')
    z = float(ref.meta['zero_hartree'])
    ref.meta = dict(ref.meta, spec={'ncas': 6, 'nelecas': 8, 'nroots': 4, 'basis': '6-31g*', 'symmetry': 'Cs',
                                    'wfnsym': "A'", 'spin_ss': 0.0, 'max_cycle': 100},
                    zero_hartree={'casscf': z, 'nevpt2': z - 0.43}, energy_level='nevpt2')
    run_dir = tmp_path / 'g'
    W.stage_init(run_dir, 'quick')
    ref.save(run_dir / 'data' / 'refs.npz')
    names = ref.coord_names
    cen = '\n'.join(f'{n} = {v}' for n, v in zip(names, [1.35, 118.0]))
    with open(run_dir / 'config.toml', 'a') as fh:
        fh.write(f'\n[gs_grid]\nhalf_points = 2\nbasis = "aug-cc-pVDZ"\n[gs_grid.center]\n{cen}\n'
                 '[gs_grid.step]\nr_OO = 0.03\na_COO = 3.0\n[vib]\nsource = "gs_grid"\norder = 2\npoints = 16\n'
                 'n_states = 4\ncheck_orders = [2]\ncheck_windows_eV = [0.6]\n')
    monkeypatch.setenv('MOCK_ORCA_GEOM', '1')
    monkeypatch.setenv('MOCK_ORCA_TABLE', str(tmp_path / 'empty.json'))
    (tmp_path / 'empty.json').write_text('{}')
    W.main(['gs_grid', str(run_dir), '--orca', str(DATA / 'mock_orca.py'), '--jobs', '3'])
    info = json.loads((run_dir / 'results' / 'gs_grid.json').read_text())
    assert info['n'] == 25 and info['n_done'] == 25 and info['nroots'] == 1
    g = np.load(run_dir / 'data' / 'gs_grid.npz')
    assert g['E_nevpt2_hartree'].shape == (25, 1) and np.isfinite(g['E_nevpt2_hartree']).all()
    inp = (run_dir / 'data' / 'gs_grid' / 'g0000_0.inp').read_text()      # a corner: started from a neighbour
    assert 'nroots 1' in inp and 'QDType' not in inp and 'MORead' in inp and 'aug-cc-pVDZ' in inp
    from diabml.workflow import vib_levels as VL
    run = W.Run(run_dir)
    g = VL._refs(run, run.cfg['vib'])                     # source = "gs_grid": the grid, one state, eV above its minimum
    assert len(g) == 25 and g.energies.shape == (25, 1) and np.isclose(np.nanmin(g.energies), 0.0)
    w, inf = VL._fit(run, g, 2, 0.6)
    assert inf['n_points'] == 25


def test_gs_grid_extra_coordinate(tmp_path, monkeypatch):
    """[gs_grid.extra] a_HCO: a coordinate the data hold fixed becomes a grid axis; geometries follow it."""
    ref = RefSet.load(W.C.PACKAGE_ROOT / 'data' / 'ch2oo_casscf.npz')
    z = float(ref.meta['zero_hartree'])
    ref.meta = dict(ref.meta, spec={'ncas': 6, 'nelecas': 8, 'nroots': 4, 'basis': '6-31g*', 'symmetry': 'Cs',
                                    'wfnsym': "A'", 'spin_ss': 0.0, 'max_cycle': 100},
                    zero_hartree={'casscf': z, 'nevpt2': z - 0.43}, energy_level='nevpt2')
    run_dir = tmp_path / 'g4'
    W.stage_init(run_dir, 'quick')
    ref.save(run_dir / 'data' / 'refs.npz')
    from diabml.workflow.orca_compute import save_library
    (tmp_path / 'lib.gbw').write_text('mock')
    save_library(run_dir / 'data' / 'orca_orbitals.json', ref.coord_names, ref.coords[:5],
                 [str(tmp_path / 'lib.gbw')] * 5)                 # a 2-D library with a 3-D grid (as 27_gs4d)
    cen = '\n'.join(f'{n} = {v}' for n, v in zip(ref.coord_names, [1.35, 118.0]))
    with open(run_dir / 'config.toml', 'a') as fh:
        fh.write(f'\n[gs_grid]\nhalf_points = 1\n[gs_grid.center]\n{cen}\n[gs_grid.step]\nr_OO = 0.03\na_COO = 3.0\n'
                 '[gs_grid.extra.a_HCO]\ncenter = 120.0\nstep = 4.0\nhalf_points = 1\n')
    monkeypatch.setenv('MOCK_ORCA_GEOM', '1')
    monkeypatch.setenv('MOCK_ORCA_TABLE', str(tmp_path / 'empty.json'))
    (tmp_path / 'empty.json').write_text('{}')
    W.main(['gs_grid', str(run_dir), '--orca', str(DATA / 'mock_orca.py'), '--jobs', '3'])
    g = np.load(run_dir / 'data' / 'gs_grid.npz')
    assert list(g['coord_names']) == list(ref.coord_names) + ['a_HCO'] and len(g['coords']) == 27
    assert sorted(set(np.round(g['coords'][:, -1], 6))) == [116.0, 120.0, 124.0]
    assert 'a_HCO' not in json.loads(str(g['fixed']))
    E = g['E_nevpt2_hartree'][:, 0]
    assert np.isfinite(E).all() and np.ptp(E[g['coords'][:, -1] != 120.0]) > 0     # H positions follow a_HCO


def test_method_cut_with_mock_orca(tmp_path, monkeypatch):
    """CCSD(T) and FIC-NEVPT2 along r(O-O) cuts of a gs_grid: inputs, parsing, curvature summary."""
    ref = RefSet.load(W.C.PACKAGE_ROOT / 'data' / 'ch2oo_casscf.npz')
    z = float(ref.meta['zero_hartree'])
    ref.meta = dict(ref.meta, spec={'ncas': 6, 'nelecas': 8, 'nroots': 4, 'basis': 'cc-pVDZ', 'symmetry': 'Cs',
                                    'wfnsym': "A'", 'spin_ss': 0.0, 'max_cycle': 100},
                    zero_hartree={'casscf': z, 'nevpt2': z - 0.43}, energy_level='nevpt2')
    run_dir = tmp_path / 'mc'
    W.stage_init(run_dir, 'quick')
    ref.save(run_dir / 'data' / 'refs.npz')
    cen = '\n'.join(f'{n} = {v}' for n, v in zip(ref.coord_names, [1.35, 118.0]))
    with open(run_dir / 'config.toml', 'a') as fh:
        fh.write(f'\n[gs_grid]\nhalf_points = 2\nbasis = "aug-cc-pVDZ"\n[gs_grid.center]\n{cen}\n'
                 '[gs_grid.step]\nr_OO = 0.03\na_COO = 3.0\n[gs_grid.extra.r_CO]\ncenter = 1.272\nstep = 0.025\n'
                 'half_points = 1\n[method_cut]\na_COO = 118.0\nr_CO = [1.247, 1.272]\n'
                 'methods = ["ccsd_t", "fic_nevpt2", "qd_nevpt2"]\n')
    monkeypatch.setenv('MOCK_ORCA_GEOM', '1')
    monkeypatch.setenv('MOCK_ORCA_TABLE', str(tmp_path / 'empty.json'))
    (tmp_path / 'empty.json').write_text('{}')
    W.main(['gs_grid', str(run_dir), '--orca', str(DATA / 'mock_orca.py'), '--jobs', '3'])
    W.main(['method_cut', str(run_dir), '--orca', str(DATA / 'mock_orca.py'), '--jobs', '3'])
    out = json.loads((run_dir / 'results' / 'method_cut.json').read_text())
    assert out['basis'] == 'aug-cc-pVDZ' and len(out['rows']) == 10           # 2 cuts x 5 r_OO values
    r = out['rows'][0]
    assert np.isfinite(r['ccsd_t']) and np.isfinite(r['fic_nevpt2']) and abs(r['T1'] - 0.015) < 1e-9
    assert abs(r['fic_casscf'] - r['casscf']) < 1e-8                         # FIC on the grid's own CASSCF
    inp = next((run_dir / 'data' / 'method_cut').glob('fic*.inp')).read_text()
    assert 'FIC_NEVPT2' in inp and 'MORead' in inp and 'aug-cc-pVDZ' in inp
    assert 'CCSD(T) aug-cc-pVDZ' in next((run_dir / 'data' / 'method_cut').glob('cc*.inp')).read_text()
    assert set(out['summary']['1.247']) >= {'casscf', 'sc_nevpt2', 'ccsd_t', 'fic_nevpt2', 'qd_nevpt2', 'sa_sc_nevpt2'}
    assert np.isfinite(r['qd_nevpt2']) and r['qd_gap_eV'] > 0
    qd = next((run_dir / 'data' / 'method_cut').glob('qd*.inp')).read_text()
    assert 'nroots 3' in qd and 'QDType' in qd and 'MORead' in qd
    assert (run_dir / 'figures' / 'MC1_method_cut.png').exists()


def test_gs_grid_ccsd_t(tmp_path, monkeypatch):
    """[gs_grid] method = "ccsd_t": RHF-CCSD(T) at every point, T1 kept; vib_levels reads E_total_hartree."""
    ref = RefSet.load(W.C.PACKAGE_ROOT / 'data' / 'ch2oo_casscf.npz')
    z = float(ref.meta['zero_hartree'])
    ref.meta = dict(ref.meta, spec={'ncas': 6, 'nelecas': 8, 'nroots': 4, 'basis': 'cc-pVDZ', 'symmetry': 'Cs',
                                    'wfnsym': "A'", 'spin_ss': 0.0, 'max_cycle': 100},
                    zero_hartree={'casscf': z, 'nevpt2': z - 0.43}, energy_level='nevpt2')
    run_dir = tmp_path / 'cc'
    W.stage_init(run_dir, 'quick')
    ref.save(run_dir / 'data' / 'refs.npz')
    cen = '\n'.join(f'{n} = {v}' for n, v in zip(ref.coord_names, [1.35, 118.0]))
    with open(run_dir / 'config.toml', 'a') as fh:
        fh.write(f'\n[gs_grid]\nhalf_points = 2\nmethod = "ccsd_t"\nbasis = "aug-cc-pVDZ"\n[gs_grid.center]\n{cen}\n'
                 '[gs_grid.step]\nr_OO = 0.03\na_COO = 3.0\n[vib]\nsource = "gs_grid"\n')
    monkeypatch.setenv('MOCK_ORCA_GEOM', '1')
    monkeypatch.setenv('MOCK_ORCA_TABLE', str(tmp_path / 'empty.json'))
    (tmp_path / 'empty.json').write_text('{}')
    W.main(['gs_grid', str(run_dir), '--orca', str(DATA / 'mock_orca.py'), '--jobs', '3'])
    g = np.load(run_dir / 'data' / 'gs_grid.npz')
    assert str(g['method']) == 'ccsd_t' and np.isfinite(g['E_total_hartree']).all() and np.allclose(g['T1'], 0.015)
    assert 'CCSD(T) aug-cc-pVDZ' in (run_dir / 'data' / 'gs_grid' / 'c0000.inp').read_text()
    info = json.loads((run_dir / 'results' / 'gs_grid.json').read_text())
    assert info['n_done'] == 25 and info['method'] == 'ccsd_t'
    from diabml.workflow import vib_levels as VL
    run = W.Run(run_dir)
    gg = VL._refs(run, run.cfg['vib'])
    assert len(gg) == 25 and np.isclose(np.nanmin(gg.energies), 0.0)
    with pytest.raises(SystemExit):                                   # method_cut needs a NEVPT2 grid
        W.main(['method_cut', str(run_dir), '--orca', str(DATA / 'mock_orca.py')])


def test_gs_grid_delta_correction(tmp_path, monkeypatch):
    """[gs_grid.delta]: a small higher-level grid on a subset of the points; its difference, fitted, corrects the
    dense grid that vib_levels and ir read."""
    ref = RefSet.load(W.C.PACKAGE_ROOT / 'data' / 'ch2oo_casscf.npz')
    z = float(ref.meta['zero_hartree'])
    ref.meta = dict(ref.meta, spec={'ncas': 6, 'nelecas': 8, 'nroots': 4, 'basis': 'cc-pVDZ', 'symmetry': 'Cs',
                                    'wfnsym': "A'", 'spin_ss': 0.0, 'max_cycle': 100},
                    zero_hartree={'casscf': z, 'nevpt2': z - 0.43}, energy_level='nevpt2')
    monkeypatch.setenv('MOCK_ORCA_GEOM', '1')
    monkeypatch.setenv('MOCK_ORCA_TABLE', str(tmp_path / 'empty.json'))
    (tmp_path / 'empty.json').write_text('{}')
    cen = '\n'.join(f'{n} = {v}' for n, v in zip(ref.coord_names, [1.35, 118.0]))
    for name, basis, step, hp in (('dz', 'aug-cc-pVDZ', (0.03, 3.0), 2), ('tz', 'aug-cc-pVTZ', (0.06, 6.0), 1)):
        d = tmp_path / name
        W.stage_init(d, 'quick')
        ref.save(d / 'data' / 'refs.npz')
        with open(d / 'config.toml', 'a') as fh:
            fh.write(f'\n[gs_grid]\nmethod = "ccsd_t"\nbasis = "{basis}"\nhalf_points = {hp}\n[gs_grid.center]\n{cen}\n'
                     f'[gs_grid.step]\nr_OO = {step[0]}\na_COO = {step[1]}\n'
                     + ('[gs_grid.delta]\nrun = "../tz"\norder = 2\n[vib]\nsource = "gs_grid"\n' if name == 'dz' else ''))
        W.main(['gs_grid', str(d), '--orca', str(DATA / 'mock_orca.py'), '--jobs', '3'])
    # make the "aug-cc-pVTZ" energies differ by a known quadratic in r_OO
    p = tmp_path / 'tz' / 'data' / 'gs_grid.npz'
    g = dict(np.load(p))
    g['E_total_hartree'] = g['E_total_hartree'] + (0.5 * 2.0 * (g['coords'][:, :1] - 1.35) ** 2) / 27.211386
    np.savez(p, **g)
    from diabml.workflow import vib_levels as VL
    from diabml.workflow.gs_grid import load_gs_grid
    run = W.Run(tmp_path / 'dz')
    corr = load_gs_grid(run, W.load_refs(run))
    run.cfg['gs_grid']['delta'] = {}
    base = load_gs_grid(run, W.load_refs(run))
    X = base.coords
    diff = corr.energies[:, 0] - base.energies[:, 0]
    want = 0.5 * 2.0 * (X[:, 0] - 1.35) ** 2
    assert np.allclose(diff - diff.mean(), want - want.mean(), atol=1e-6)    # recovered on all 25 points
    assert 'delta' in corr.meta['method'] and 'aug-cc-pVTZ' in corr.meta['method']


def test_freeze_switch_and_shell(tmp_path, monkeypatch):
    """freeze: a fixed copy of the well surface that vib_levels and ir can use; shell: CCSD(T) on batches of points
    outside the grid; shell_check: compares, changes nothing."""
    ref = RefSet.load(W.C.PACKAGE_ROOT / 'data' / 'ch2oo_casscf.npz')
    z = float(ref.meta['zero_hartree'])
    ref.meta = dict(ref.meta, spec={'ncas': 6, 'nelecas': 8, 'nroots': 4, 'basis': 'cc-pVDZ', 'symmetry': 'Cs',
                                    'wfnsym': "A'", 'spin_ss': 0.0, 'max_cycle': 100},
                    zero_hartree={'casscf': z, 'nevpt2': z - 0.43}, energy_level='nevpt2')
    monkeypatch.setenv('MOCK_ORCA_GEOM', '1')
    monkeypatch.setenv('MOCK_ORCA_TABLE', str(tmp_path / 'empty.json'))
    (tmp_path / 'empty.json').write_text('{}')
    d = tmp_path / 'w'
    W.stage_init(d, 'quick')
    ref.save(d / 'data' / 'refs.npz')
    cen = '\n'.join(f'{n} = {v}' for n, v in zip(ref.coord_names, [1.35, 118.0]))
    with open(d / 'config.toml', 'a') as fh:
        fh.write(f'\n[gs_grid]\nmethod = "ccsd_t"\nhalf_points = 2\n[gs_grid.center]\n{cen}\n'
                 '[gs_grid.step]\nr_OO = 0.03\na_COO = 3.0\n[vib]\nsource = "gs_grid"\norder = 2\n'
                 '[shell]\nn_points = 12\nbatches = 3\n[shell.box]\nr_OO = [1.2, 1.6]\na_COO = [100.0, 136.0]\n'
                 '[shell.step]\nr_OO = 0.05\na_COO = 6.0\n')
    W.main(['gs_grid', str(d), '--orca', str(DATA / 'mock_orca.py'), '--jobs', '3'])
    W.main(['freeze', str(d), '--name', 'benchmark'])
    assert (d / 'models' / 'surfaces' / 'benchmark.npz').exists()
    man = json.loads((d / 'models' / 'surfaces' / 'benchmark.json').read_text())
    assert man['data']['gs_grid.npz'] and man['order'] == 2
    with pytest.raises(SystemExit):                                         # never overwritten
        W.main(['freeze', str(d), '--name', 'benchmark'])
    from diabml.workflow.shell import design, load_frozen
    run = W.Run(d)
    dz = design(run)
    g = np.load(d / 'data' / 'gs_grid.npz')
    lo, hi = g['coords'].min(0), g['coords'].max(0)
    assert len(dz['coords']) == 12 and not np.all((dz['coords'] >= lo) & (dz['coords'] <= hi), 1).any()
    assert sorted(set(dz['batch'])) == [1, 2, 3] and (np.bincount(dz['batch'])[1:] == 4).all()
    W.main(['shell', str(d), '--orca', str(DATA / 'mock_orca.py'), '--jobs', '3', '--batch', '1'])
    s = np.load(d / 'data' / 'shell.npz')
    assert len(s['coords']) == 4 and set(s['batch'].tolist()) == {1}
    W.main(['shell', str(d), '--orca', str(DATA / 'mock_orca.py'), '--jobs', '3', '--batch', '2'])
    s = np.load(d / 'data' / 'shell.npz')
    assert len(s['coords']) == 8 and set(s['batch'].tolist()) == {1, 2}              # merged, batch 1 reused
    W.main(['shell_check', str(d)])
    out = json.loads((d / 'results' / 'shell_check.json').read_text())
    assert out['n'] == 8 and 'benchmark' in out
    # the switch: frozen surface for the levels, unchanged by the grid
    w1, _ = load_frozen(run, 'frozen:benchmark')
    assert np.allclose(w1.coef.numpy(), load_frozen(W.Run(d), 'frozen:benchmark')[0].coef.numpy())
