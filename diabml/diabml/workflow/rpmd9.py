"""Thermostatted ring-polymer molecular dynamics (TRPMD) infrared spectra of CH2OO on the 9-D surface.

    python -m diabml.workflow.ch2oo rpmd9d RUN [-v]
        -> results/rpmd_9d.json, results/rpmd_9d_spectrum.csv, results/rpmd_9d_P{P}.npz, figures/RPMD1_spectra.png

The approximate quantum dynamics that can go to larger Criegee intermediates, benchmarked here against the exact
(VCI) result on the same surface and dipole. For each bead number P in [rpmd] beads (P = 1 is classical molecular
dynamics run with the same protocol, the control):
  1. path-integral MD (PILE-L thermostat on every ring-polymer normal mode, centroid time constant
     [rpmd] centroid_tau_fs) for equil_fs from the minimum: quantum Boltzmann sampling at [rpmd] temperature,
     zero-point motion included through the bead spread;
  2. TRPMD for run_fs: centroid unthermostatted, internal modes damped with gamma_k = 2 lam omega_k
     (Rossi, Ceriotti and Manolopoulos 2014); the bead-averaged dipole (9-D dipole surface) every `every` steps;
  3. spectrum: Fourier transform of the Kubo-transformed dipole autocorrelation, n alpha ~ w^2 C(w) (as ir).
All trajectories of a batch and all their beads go through the surface in one batched evaluation (PyTorch, exact
forces). Reported per P: band maxima of the strong bands, the centroid temperature, the mean potential energy (the
quantum value includes zero-point energy), and how much of the bead sampling left the fitted region of the surface.
P must be large enough for the highest frequency: P >~ hbar omega_max / k_B T (16 at 300 K for the C-H stretches;
32 recommended); the change from 16 to 32 is the convergence check.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import ch2oo as W

BANDS = [('v6', 870.0, 950.0), ('v4', 1250.0, 1340.0), ('v3', 1400.0, 1500.0)]   # band regions (cm-1)


DIAG_FS = 10.0                                          # interval of the saved diagnostic traces (fs)
PROGRESS_STEPS = 500                                    # steps between progress-file updates
SAMPLE_OUT, SAMPLE_IN = 4, 1                            # bead geometries kept per check (every 200 steps) for rpmd_check
HIST = [('r_OO', 0.35), ('r_CO', 0.35), ('r_CH1', 0.45), ('a_COO', 35.0), ('d_H1', 80.0)]   # coordinate, half range
HIST_BINS = 140


def _centre(nu, I, lo, hi):
    m = (nu >= lo) & (nu <= hi)
    s = I[m].sum()
    return float((nu[m] * I[m]).sum() / s) if s > 0 else float('nan')


def band_centres(mu, dt, n_boot=20, seed=0):
    """Intensity-weighted centre of each band region (robust where the band is broad and the maximum noisy) and
    its standard error from a bootstrap over trajectories. mu (n_traj, n_t, 3)."""
    from ..dynamics.ir import ir_spectrum

    def c(m):
        nu, I = ir_spectrum(m, dt)
        return [_centre(nu, I, lo, hi) for _, lo, hi in BANDS]
    base = c(mu)
    rng = np.random.default_rng(seed)
    bs = np.array([c(mu[rng.integers(0, len(mu), len(mu))]) for _ in range(n_boot)]) if len(mu) > 1 else None
    return {b: {'centre': base[i], 'err': float(bs[:, i].std()) if bs is not None else None}
            for i, (b, _, _) in enumerate(BANDS)}


def _worker(args):
    """One batch of trajectories at one bead number: PIMD equilibration then TRPMD; returns the bead-averaged
    dipoles and diagnostics."""
    (mp, dp, P, T, n, seed, equil_fs, run_fs, dt, every, lam, tau, threads) = args[:13]
    prog = args[13] if len(args) > 13 else None              # progress file (json), written every PROGRESS_STEPS

    def report(phase, done, total, t_start):
        if prog:
            try:
                with open(prog, 'w') as fh:
                    json.dump({'phase': phase, 'done': done, 'total': total, 'elapsed': time.time() - t_start}, fh)
            except OSError:
                pass
    import torch
    torch.set_num_threads(int(threads))
    from ..dipole9 import VectorDipole9
    from ..dynamics.rpmd import RingPolymer
    from ..geometry import NINE, ch2oo_internal9, ch2oo_positions9
    from ..vibdvr import ISOTOPIC_MASS
    from ..wellfit import PolyWell
    from .well9d import energy_forces9, refine_minimum
    model = PolyWell.load(mp)
    dm = VectorDipole9.load(dp)
    xm, Em, _ = refine_minimum(model, model.center.numpy() if hasattr(model.center, 'numpy') else model.center)
    X0 = ch2oo_positions9(xm[None])[0]

    def force(X):
        return energy_forces9(model, X, Em)

    rng = np.random.default_rng(seed)
    rp = RingPolymer(ISOTOPIC_MASS, P, T, force)
    q, p = rp.initial(X0, n, rng)
    t_all = time.time()
    n_eq = int(round(equil_fs / dt))
    steps = int(round(run_fs / dt))
    total = n_eq + steps
    t0 = time.time()
    done = 0
    while done < n_eq:                                       # equilibration in chunks, with progress
        k = min(PROGRESS_STEPS, n_eq - done)
        q, p = rp.run(q, p, k, dt, gamma0=1.0 / tau, lam=1.0, rng=rng)
        done += k
        report('equilibration', done, total, t_all)
    t_eq = time.time() - t0
    mu, Tc, Vb, out_box, emin, nchk = [], [], [], 0, np.inf, 0
    dstep = max(1, int(round(DIAG_FS / dt)))             # diagnostics every DIAG_FS
    cen, rg, tT, tV = [], [], [], []
    hist = np.zeros((len(HIST), HIST_BINS))
    lo_b = np.array([model.box[k][0] for k in NINE])
    hi_b = np.array([model.box[k][1] for k in NINE])
    out_coord = np.zeros(len(NINE))
    out_x, in_x = [], []
    t0 = time.time()
    for i, (q, p, E) in enumerate(rp.trajectory(q, p, steps, dt, lam=lam, gamma0=0.0, rng=rng)):
        if i % PROGRESS_STEPS == 0:
            report('TRPMD', n_eq + i, total, t_all)
        if i % every == 0:
            mu.append(dm.predict(q.reshape(-1, 5, 3)).reshape(n, P, 3).mean(1))
        if i % dstep == 0:                              # traces: centroid geometry, bead spread, temperature, energy
            qc = q.mean(1)
            cen.append(ch2oo_internal9(qc))
            rg.append(np.sqrt(((q - qc[:, None]) ** 2).sum(3).mean(1)))
            tT.append(rp.centroid_temperature(p).mean())
            tV.append(E.mean())
        if i % 200 == 0:
            Tc.append(rp.centroid_temperature(p))
            Vb.append(E.mean(1))
            X = ch2oo_internal9(q.reshape(-1, 5, 3))
            outside = ~model.inside(X)
            out_box += int(outside.sum())
            nchk += X.shape[0]
            out_coord += ((X < lo_b) | (X > hi_b)).sum(0)
            io = np.flatnonzero(outside)
            ii = np.flatnonzero(~outside)
            if len(io):
                out_x.append(X[rng.choice(io, min(SAMPLE_OUT, len(io)), replace=False)])
            if len(ii):
                in_x.append(X[rng.choice(ii, min(SAMPLE_IN, len(ii)), replace=False)])
            emin = min(emin, float(E.min()))
            for h, (name, half) in enumerate(HIST):     # distribution of the bead geometries (quantum widths)
                j = NINE.index(name)
                hist[h] += np.histogram(X[:, j] - xm[j], bins=HIST_BINS, range=(-half, half))[0]
    return {'mu': np.stack(mu, 1).astype(np.float32), 'dt_record': dt * every, 'T_centroid': float(np.mean(Tc)),
            'V_mean_eV': float(np.mean(Vb)), 'outside_box': out_box / max(nchk, 1), 'E_min_eV': emin,
            'seconds_equil': t_eq, 'seconds_run': time.time() - t0, 'n': n,
            'centroid_x': np.stack(cen, 1).astype(np.float32), 'rg': np.stack(rg, 1).astype(np.float32),
            'T_t': np.array(tT), 'V_t': np.array(tV), 'hist': hist, 'diag_dt': dstep * dt, 'xm': xm,
            'out_x': np.concatenate(out_x) if out_x else np.zeros((0, len(NINE))),
            'in_x': np.concatenate(in_x) if in_x else np.zeros((0, len(NINE))), 'out_coord': out_coord / max(nchk, 1)}


def captured(nu_cm, P, T):
    """Share of the exact quantum potential energy of a harmonic mode (wavenumber nu_cm) that P beads capture at T:
    <V>_P / <V>_quantum = [(1 / 2 P beta_P hbar w) sum_k w^2 / (w^2 + w_k^2)] / [(1/4) coth(beta hbar w / 2)]."""
    x = nu_cm / (0.6950348 * T)
    k = np.arange(P)
    wk = 2 * P / x * np.sin(k * np.pi / P)
    return float((P / (2 * x)) * np.sum(1 / (1 + wk ** 2)) / P / (0.25 / np.tanh(x / 2)))


def plan(run, probe=True):
    """Bead and run-length planning: zero-point capture per mode for candidate P, spectral resolution of the run
    length, and the predicted wall time from a short timing probe of one batch."""
    rc = run.cfg['rpmd']
    T = float(rc['temperature'])
    vj = run.p('results', 'well9d_vci.json')
    if vj.exists():
        v = json.loads(vj.read_text())
        w = np.array(v['harmonic'])
        names = v['mode_names']
    else:
        w = np.array([3300.0, 1500.0, 900.0, 500.0])
        names = ['C-H', 'bend', 'O-O', 'COO']
    o = np.argsort(-w)
    beads = sorted(set([int(b) for b in rc['beads']] + [8, 16, 32, 48, 64]))
    run.log(f'planning at {T:g} K: share of the quantum (zero-point) potential energy captured by P beads, '
            f'harmonic, per mode (100% = converged); P ~ hbar w_max / k_B T = {w.max() / (0.695 * T):.0f}')
    run.log('   P  ' + ' '.join(f'{names[i]:>5s}' for i in o) + '   (wavenumbers ' +
            ', '.join(f'{w[i]:.0f}' for i in o) + ')')
    for P in beads:
        run.log(f'{P:4d}  ' + ' '.join(f'{100 * captured(w[i], P, T):5.1f}' for i in o))
    rf = float(rc['run_fs'])
    run.log('resolution (FWHM of a Hann-windowed transform, 1.44 / run time): ' + ', '.join(
        f'{x / 1000:g} ps {1.44 * 33356.4 / x:.1f} cm-1' for x in sorted({5000.0, 10000.0, 20000.0, 50000.0, rf})) +
        f'; TRPMD friction adds an intrinsic width of a few cm-1, so runs beyond about 20 ps gain little')
    if not probe:
        return
    from multiprocessing import get_context
    mp = run.p('models', str(run.cfg['ir'].get('model9d', 'well9d.npz')))
    dp = run.p('models', str(run.cfg['ir'].get('dipole9d', 'dipole9d.npz')))
    batch, jobs, ntraj = int(rc['batch']), int(rc['jobs']), int(rc['trajectories'])
    dt, every = float(rc['dt']), int(rc['every'])
    steps = (float(rc['equil_fs']) + rf) / dt
    nb = -(-ntraj // batch)
    rounds = -(-nb // jobs)
    tot = 0.0
    with get_context('spawn').Pool(1) as pool:
        for P in sorted(set(int(b) for b in rc['beads'])):
            r = pool.apply(_worker, ((str(mp), str(dp), P, T, batch, 0, 10 * dt, 40 * dt, dt, every,
                                      float(rc['lam']), float(rc['centroid_tau_fs']), int(rc.get('torch_threads', 1))),))
            sps = r['seconds_run'] / 40
            m = sps * steps * rounds / 60
            tot += m
            run.log(f'P = {P:3d}: {1000 * sps:.0f} ms per step for {batch} trajectories -> about {m:.0f} min for '
                    f'{ntraj} trajectories ({nb} batches, {jobs} in parallel), {(float(rc["equil_fs"]) + rf) / 1000:g} ps each')
    run.log(f'predicted total about {tot:.0f} min on this machine (CPU cores busy with other work slow it down)')


def stage_rpmd_plan(run):
    plan(run, probe=True)


def _band_max(nu, I, lo, hi):
    m = (nu >= lo) & (nu <= hi)
    return float(nu[m][np.argmax(I[m])]) if m.any() else None


def stage_rpmd9d(run):
    from multiprocessing import get_context
    from ..dynamics.ir import ir_spectrum
    rc = run.cfg['rpmd']
    mp = run.p('models', str(run.cfg['ir'].get('model9d', 'well9d.npz')))
    dp = run.p('models', str(run.cfg['ir'].get('dipole9d', 'dipole9d.npz')))
    for f, st in ((mp, 'well9d_fit'), (dp, 'well9d_dipole')):
        if not f.exists():
            raise SystemExit(f'{f} not found: run {st} first')
    T = float(rc['temperature'])
    beads = [int(b) for b in rc['beads']]
    ntraj, batch, jobs = int(rc['trajectories']), int(rc['batch']), int(rc['jobs'])
    dt, every, lam = float(rc['dt']), int(rc['every']), float(rc['lam'])
    equil, run_fs, tau = float(rc['equil_fs']), float(rc['run_fs']), float(rc['centroid_tau_fs'])
    run.log(f'TRPMD at {T:g} K: beads {beads}, {ntraj} trajectories each ({batch} per batch, {jobs} batches in '
            f'parallel), PIMD equilibration {equil:g} fs then {run_fs:g} fs, dt {dt:g} fs, lam {lam:g} '
            f'(internal-mode friction 2 lam omega_k); P needed for the C-H stretches about '
            f'{3300 / (0.695 * T):.0f}')
    plan(run, probe=False)
    out = {'temperature_K': T, 'lam': lam, 'run_fs': run_fs, 'dt_fs': dt, 'trajectories': ntraj, 'beads': {}}
    spectra = {}
    ctx = get_context('spawn')
    pdir = run.p('logs', 'rpmd_progress')
    pdir.mkdir(parents=True, exist_ok=True)
    every_min = float(rc.get('progress_minutes', 5.0))
    todo = [(P, lam if P > 1 else 0.0) for P in beads]
    if [float(x) for x in rc.get('lam_scan', [])] and max(beads) > 1:
        todo += [(max(beads), float(x)) for x in rc['lam_scan'] if abs(float(x) - lam) > 1e-9]
    cost_done, cost_total = 0.0, float(sum(P for P, _ in todo))          # work is about proportional to P
    rate = {'s_per_P': None}                                             # seconds per unit P, measured

    def run_set(P, lam_, tag):
        nonlocal cost_done
        sizes = [min(batch, ntraj - i) for i in range(0, ntraj, batch)]
        files = [pdir / f'{tag}_b{k}.json' for k in range(len(sizes))]
        for f in files:
            f.unlink(missing_ok=True)
        args = [(str(mp), str(dp), P, T, s, 1000 * P + k, equil, run_fs, dt, every, lam_, tau,
                 int(rc.get('torch_threads', 1)), str(files[k])) for k, s in enumerate(sizes)]
        nconc = min(jobs, len(args))
        rounds = -(-len(args) // nconc)
        if rate['s_per_P']:
            est = rate['s_per_P'] * P
            rest = rate['s_per_P'] * (cost_total - cost_done)
            run.log(f'{tag}: starting {len(args)} batches ({nconc} at a time); expected about {est / 60:.0f} min, '
                    f'finishing near {time.strftime("%H:%M", time.localtime(time.time() + est))}; whole stage near '
                    f'{time.strftime("%H:%M", time.localtime(time.time() + rest))}')
        else:
            run.log(f'{tag}: starting {len(args)} batches ({nconc} at a time); time estimate after the first progress report')
        t0 = time.time()
        last = t0
        with ctx.Pool(nconc) as pool:
            async_res = pool.map_async(_worker, args)
            while not async_res.ready():
                async_res.wait(timeout=20)
                if time.time() - last < 60 * every_min or async_res.ready():
                    continue
                last = time.time()
                st = []
                for f in files:
                    try:
                        st.append(json.loads(f.read_text()))
                    except (OSError, ValueError):
                        pass
                if not st:
                    continue
                frac = np.mean([s['done'] / s['total'] for s in st] + [0.0] * (len(files) - len(st)))
                el = time.time() - t0
                # batches run nconc at a time: total work = rounds x one batch
                left = el * (1 - frac) / max(frac, 1e-6)
                phases = sorted({s['phase'] for s in st})
                run.log(f'  {tag}: {100 * frac:.0f}% of steps done ({"/".join(phases)}), {el / 60:.0f} min elapsed; '
                        f'about {left / 60:.0f} min left, finishing near '
                        f'{time.strftime("%H:%M", time.localtime(time.time() + left))}' +
                        (f'; whole stage near {time.strftime("%H:%M", time.localtime(time.time() + left + (rate["s_per_P"] or (el + left) / P) * (cost_total - cost_done - P)))}'
                         if cost_total - cost_done - P > 0 else ''))
            res = async_res.get()
            for k, r in enumerate(res):
                run.detail(f'  {tag}: batch {k + 1}/{len(args)} ({r["n"]} trajectories) done; centroid '
                           f'{r["T_centroid"]:.0f} K, equilibration {r["seconds_equil"]:.0f} s, run '
                           f'{r["seconds_run"]:.0f} s')
        wall = time.time() - t0
        cost_done += P
        rate['s_per_P'] = wall / P if P > 1 else rate['s_per_P']
        mu = np.concatenate([r['mu'] for r in res], 0)
        nu, I = ir_spectrum(mu, res[0]['dt_record'], broaden_cm=float(rc.get('broaden_cm', 0.0)),
                            max_cm=4000.0, min_cm=200.0)
        w = np.array([r['n'] for r in res], float)
        nsteps = int(round((equil + run_fs) / dt))
        cpu = sum(r['seconds_equil'] + r['seconds_run'] for r in res)
        rec = {'beads': P, 'lam': lam_, 'T_centroid_K': float(np.average([r['T_centroid'] for r in res], weights=w)),
               'V_mean_eV': float(np.average([r['V_mean_eV'] for r in res], weights=w)),
               'outside_box': float(np.average([r['outside_box'] for r in res], weights=w)),
               'E_min_eV': float(min(r['E_min_eV'] for r in res)), 'minutes': wall / 60,
               'trajectories': int(w.sum()), 'steps_per_trajectory': nsteps, 'cpu_seconds': cpu,
               'ms_per_step_per_trajectory': 1000 * cpu / (nsteps * w.sum()),
               'force_evaluations': int(nsteps * w.sum() * P),
               'outside_by_coordinate': dict(zip(NINE9, np.average([r['out_coord'] for r in res], axis=0,
                                                                   weights=w).tolist())),
               'bands': band_centres(mu, res[0]['dt_record'], int(rc.get('bootstrap', 20)))}
        np.savez(run.p('results', f'rpmd_9d_{tag}.npz'), mu=mu, dt_record=res[0]['dt_record'], nu=nu, intensity=I,
                 centroid_x=np.concatenate([r['centroid_x'] for r in res], 0), rg=np.concatenate([r['rg'] for r in res], 0),
                 T_t=np.average([r['T_t'] for r in res], axis=0, weights=w),
                 V_t=np.average([r['V_t'] for r in res], axis=0, weights=w), hist=sum(r['hist'] for r in res),
                 diag_dt=res[0]['diag_dt'], xm=res[0]['xm'], out_x=np.concatenate([r['out_x'] for r in res]),
                 in_x=np.concatenate([r['in_x'] for r in res]))
        run.log(f'{tag}: centroid {rec["T_centroid_K"]:.0f} K, <V> {1000 * rec["V_mean_eV"]:.0f} meV '
                f'({rec["V_mean_eV"] * 8065.5:.0f} cm-1), bead geometries outside the fitted box '
                f'{100 * rec["outside_box"]:.1f}%, lowest bead energy {1000 * rec["E_min_eV"]:+.0f} meV; band centres '
                + ', '.join(f'{b} {v["centre"]:.1f}' + (f' ± {v["err"]:.1f}' if v['err'] is not None else '')
                            for b, v in rec['bands'].items()) +
                f' ({rec["minutes"]:.1f} min; {rec["ms_per_step_per_trajectory"]:.2f} ms per step per trajectory)')
        ob = sorted(rec['outside_by_coordinate'].items(), key=lambda kv: -kv[1])
        run.detail(f'  {tag}: bead geometries outside the fitted range, by coordinate: ' + ', '.join(
            f'{k} {100 * v:.1f}%' for k, v in ob if v > 0.0005))
        if rec['E_min_eV'] < -0.05:
            run.log('  warning: beads reached geometries below the minimum (a hole of the polynomial outside the '
                    'fitted region): check outside_box and consider more points there')
        return rec, (nu, I)

    from ..geometry import NINE as NINE9
    out.update({'equil_fs': equil, 'every': every, 'batch': batch, 'jobs': jobs, 'centroid_tau_fs': tau,
                'resolution_cm-1': 1.44 * 33356.4 / run_fs})
    for P in beads:
        out['beads'][str(P)], spectra[P] = run_set(P, lam if P > 1 else 0.0, f'P{P}')
    # friction scan at the largest bead number: lam = 0 is plain RPMD
    scan = [float(x) for x in rc.get('lam_scan', [])]
    if scan and max(beads) > 1:
        out['lam_scan'] = {'beads': max(beads), 'values': {f'{lam:g}': out['beads'][str(max(beads))]}}
        for lm in scan:
            if abs(lm - lam) < 1e-9:
                continue
            out['lam_scan']['values'][f'{lm:g}'], _ = run_set(max(beads), lm, f'P{max(beads)}_lam{lm:g}')
        run.log(f'friction scan at P = {max(beads)} (band centres; lam = 0: RPMD): ' + '; '.join(
            f'lam {k}: ' + ', '.join(f'{b} {v["centre"]:.1f}±{v["err"]:.1f}' for b, v in r['bands'].items())
            for k, r in sorted(out['lam_scan']['values'].items(), key=lambda kv: float(kv[0]))))
    bp = run.p('results', 'well9d_bands.npz')
    vj = run.p('results', 'well9d_vci.json')
    if vj.exists():
        v = {f['mode']: f for f in json.loads(vj.read_text())['fundamentals']}
        out['vci'] = {b: v[b]['vci'] for b, _, _ in BANDS if b in v}
        out['harmonic'] = {b: v[b]['harmonic'] for b, _, _ in BANDS if b in v}
        out['expt'] = {b: v[b].get('expt') for b, _, _ in BANDS if b in v}
    if bp.exists():                                     # the same centre measure on the VCI rotational contour
        z = np.load(bp)
        g, va = z['grid'], z['absorption']
        out['vci_centre'] = {b: _centre(g, va, lo, hi) for b, lo, hi in BANDS}
        run.log('band centres (intensity-weighted over ' + ', '.join(f'{b} {lo:.0f}-{hi:.0f}' for b, lo, hi in BANDS)
                + ' cm-1), the VCI value from its rotational contour (well9d_bands):')
        run.log('band  ' + '  '.join(f'{"P=" + str(P):>12s}' for P in beads) + f'  {"VCI":>8s}   share of the '
                f'classical-to-VCI gap closed at P = {beads[-1]}')
        out['gap_closed'] = {}
        cref = None                                   # the classical reference: the more precise of P = 1 and ir
        irf = run.p('results', f'ir_9d_{T:g}K.npz')
        if irf.exists():
            with np.load(irf) as zz:
                if 'mu' in zz.files:
                    cref = band_centres(zz['mu'], float(zz['t'][1] - zz['t'][0]), int(rc.get('bootstrap', 20)))
                    out['classical_ir'] = {'trajectories': int(zz['mu'].shape[0]), 'bands': cref}
                    run.log(f'classical reference from the ir stage ({zz["mu"].shape[0]} trajectories): ' + ', '.join(
                        f'{b} {v["centre"]:.1f} ± {v["err"]:.1f}' for b, v in cref.items()))
        for b, _, _ in BANDS:
            cells = [out['beads'][str(P)]['bands'][b] for P in beads]
            vc = out['vci_centre'][b]
            gap = None
            c0 = cells[0] if beads[0] == 1 else None
            if cref is not None and (c0 is None or (cref[b]['err'] or 1e9) < (c0['err'] or 1e9)):
                c0 = cref[b]
            e0 = (c0 or {}).get('err') or 0.0
            if c0 is not None and np.isfinite(vc) and abs(c0['centre'] - vc) > max(3 * e0, 1.0):    # a gap above the noise
                gap = (c0['centre'] - cells[-1]['centre']) / (c0['centre'] - vc)
                out['gap_closed'][b] = gap
            run.log(f'{b:4s}  ' + '  '.join(f'{x["centre"]:7.1f}' + (f'±{x["err"]:.1f}' if x['err'] is not None else '    ')
                                             for x in cells) + f'  {vc:8.1f}   ' + (f'{100 * gap:.0f}%' if gap is not None else '-'))
    nu0 = spectra[beads[0]][0]
    cols = [np.interp(nu0, *spectra[P]) for P in beads]
    np.savetxt(run.p('results', 'rpmd_9d_spectrum.csv'), np.c_[nu0, np.array(cols).T], delimiter=',', fmt='%.5f',
               header='wavenumber_cm-1,' + ','.join(f'I_P{P}' for P in beads), comments='')
    run.p('results', 'rpmd_9d.json').write_text(json.dumps(out, indent=1, default=float))
    _figure(run, spectra, out)
    run.log('wrote results/rpmd_9d.json, results/rpmd_9d_spectrum.csv, figures/RPMD1_spectra.png')
    from .dynamics_report import stage_dynamics_report
    stage_dynamics_report(run)
    if run.p('results', 'well9d_bands.npz').exists() and run.p('results', 'ir_9d_spectrum.csv').exists():
        from .compare9 import stage_well9d_compare
        stage_well9d_compare(run)


def _figure(run, spectra, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    beads = sorted(spectra)
    cmap = {1: '#c8641e'}
    shades = ['#7a3fa0', '#2a7f62', '#b02a6b', '#1f5fa8']
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={'width_ratios': [1.6, 1]})
    for ax, (lo, hi) in zip(axs, ((800, 1520), (860, 960))):
        for i, P in enumerate(beads):
            nu, I = spectra[P]
            m = (nu >= lo) & (nu <= hi)
            ax.plot(nu[m], I[m] / max(I[m].max(), 1e-300), lw=1.2,
                    color=cmap.get(P, shades[i % len(shades)]), label='classical (P = 1)' if P == 1 else f'TRPMD, P = {P}')
        for b, v in (out.get('vci_centre') or out.get('vci') or {}).items():
            if lo <= v <= hi:
                ax.axvline(v, color='#1f5fa8', ls='--', lw=0.8)
                ax.text(v, 1.04, b, ha='center', fontsize=8, color='#1f5fa8')
        for b, v in (out.get('harmonic') or {}).items():
            if lo <= v <= hi:
                ax.axvline(v, color='0.6', ls=':', lw=0.8)
        ax.set_xlim(hi, lo)
        ax.set_ylim(-0.03, 1.12)
        ax.set_yticks([])
        ax.set_xlabel('wavenumber (cm⁻¹)')
    axs[0].set_title(f'TRPMD on the 9-D surface, {out["temperature_K"]:g} K (dashed: VCI band centres; dotted: harmonic)',
                     fontsize=10.5)
    axs[1].set_title('O–O band', fontsize=10.5)
    axs[0].legend(fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(0.3, 0.95))
    fig.tight_layout()
    fig.savefig(run.p('figures', 'RPMD1_spectra.png'), dpi=150)
    plt.close(fig)
