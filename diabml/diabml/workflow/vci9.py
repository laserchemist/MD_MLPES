"""Variational vibrational levels, IR absorption and IR emission of CH2OO on the 9-D surface (VCI).

    python -m diabml.workflow.ch2oo well9d_vci RUN [-v]      # needs models/well9d.npz (well9d_fit) and, for
                                                              # intensities, models/dipole9d.npz (well9d_dipole)
    -> results/well9d_vci.json, results/well9d_vci.npz, figures/VCI1_spectra.png, figures/VCI2_oo_band.png

Quantum levels of the full 9-D Watson Hamiltonian (J = 0) on the fitted surface by vibrational configuration
interaction (diabml/vci.py): harmonic-oscillator product basis in the Eckart normal coordinates of the minimum,
n-mode representation of the potential and of the dipole, Coriolis coupling included. Zero-point energy, Fermi and
Darling-Dennison resonances and all mode couplings are in the variational answer; there is no perturbation theory.

What it gives:
  - the fundamentals, compared with GVPT2 (results/well9d_vpt2.json) and with experiment (Su et al. 2013, band
    origins from the rotational analysis), and the zero-point energy;
  - absolute absorption intensities (km/mol) of every transition among the computed levels, with the band type
    (a/b/c: the transition moment along the principal axes);
  - spectra: absorption at [well9d] vci_temperature (Boltzmann populations over the levels: hot bands included),
    and emission from CH2OO with vci_emission_kJmol of vibrational energy above the zero-point level, the levels
    populated at the vibrational temperature with that mean energy; emission strength nu^3 |mu|^2 (photon rate)
    from each upper level to every lower one. The population the computed levels leave out is estimated
    (harmonic) and reported.

Convergence. The basis is a product: the seven modes below 2500 cm-1 truncated by harmonic excitation energy
vci_ecut, times the two C-H stretches with up to vci_ch_quanta quanta (diabml/vci.py explains why: in rectilinear
coordinates the CH2 bends couple strongly to the C-H stretches, and one energy cutoff for all modes biases the
fundamentals). A second, smaller basis (vci_ecut_check, vci_ch_quanta_check) is run and the change in every
fundamental is reported.
Grid points far outside the fitted region (harmonic energy above vci_guard_cm where the polynomial falls below half
of the harmonic value) use the harmonic potential; their number is reported (they lie where the wavefunctions
vanish).

Line shapes are Gaussian (vci_broaden_cm). Rotational band contours are not included here: the 9-D dynamics
(ir stage) give them; the band positions and relative strengths are what this stage adds.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import ch2oo as W
from .well9d import (NAMES, _herzberg, cartesian_hessian_fn, ch2oo_positions9, exact_K, mode_info, normal_modes,
                     refine_minimum)

_H_TO_K = 1.0 / 0.6950348                      # K per cm-1
KJMOL_CM = 83.5935                             # cm-1 per kJ/mol


def setup(run, log=True, masses=None):
    """The 9-D surface, its minimum and Eckart normal modes, Herzberg mode names, and the potential (cm-1 above the
    minimum) and dipole (Debye, principal-axis frame) as functions of dimensionless normal coordinates."""
    from .. import vpt2 as V
    from ..geometry import ch2oo_internal9
    from ..vibdvr import EV_CM, ISOTOPIC_MASS, g_matrix
    from ..wellfit import PolyWell
    wc = run.cfg['well9d']
    mp = run.p('models', 'well9d.npz')
    if not mp.exists():
        raise SystemExit(f'{mp} not found: run well9d_fit first')
    model = PolyWell.load(mp)
    rp = run.p('results', 'well9d.json')
    x0 = np.array([json.loads(rp.read_text())['minimum'][n] for n in NAMES]) if rp.exists() else \
        model.center.numpy()
    xm, Em, gmax = refine_minimum(model, x0)
    P0 = ch2oo_positions9(xm[None])[0]
    parent = masses is None
    masses = ISOTOPIC_MASS if parent else np.asarray(masses, float)
    modes = V.cartesian_modes(cartesian_hessian_fn(model)(P0), P0, masses)
    K, G0 = exact_K(model, xm), g_matrix(xm[None], NAMES, {})[0]
    _, L, _ = normal_modes(K, G0)
    info = mode_info(K, L)
    hz = _herzberg(info, modes.omega)
    names = [hz[k] for k in range(len(modes.omega))]
    labels = [info[k]['label'] for k in range(len(modes.omega))]
    sym = [info[k]['symmetry'] for k in range(len(modes.omega))]
    w = modes.omega
    g = np.sqrt(w / V.C_CM)
    sm = np.repeat(masses, 3)
    guard = float(wc.get('vci_guard_cm', 12000.0))
    stats = {'points': 0, 'outside_box': 0, 'guarded': 0}

    def positions(q):
        D = ((np.asarray(q, float) / g) @ modes.L.T / np.sqrt(sm)).reshape(len(q), -1, 3)
        return P0[None] + D

    def potential(q):
        P = positions(q)
        X = ch2oo_internal9(P)
        v = (model.predict(X)[:, 0] - Em) * EV_CM
        h = 0.5 * (w * np.asarray(q) ** 2).sum(1)
        bad = (h > guard) & (v < 0.5 * h)
        stats['points'] += len(q)
        stats['outside_box'] += int((~model.inside(X)).sum())
        stats['guarded'] += int(bad.sum())
        return np.where(bad, h, v)

    _, R = V.principal_axes(P0, masses)
    dipole = None
    dp = run.p('models', str(run.cfg['ir'].get('dipole9d', 'dipole9d.npz')))
    if dp.exists():
        from ..dipole9 import VectorDipole9
        dm = VectorDipole9.load(dp)

        def dipole(q):
            return dm.predict(positions(q)) @ R                    # a, b, c components
    if log:
        run.log(f'model {mp.name} ({model.meta.get("source", "")}, order {model.meta.get("order")}); minimum refined '
                f'(largest gradient {gmax:.1e}); dipole surface: {dp.name if dipole else "none (no intensities)"}')
    return {'model': model, 'mp': mp, 'dp': dp if dipole else None, 'modes': modes, 'names': names, 'labels': labels,
            'symmetry': sym, 'potential': potential, 'dipole': dipole, 'stats': stats, 'xm': xm, 'P0': P0, 'masses': masses}


def _label(st, names):
    parts = [(names[i], c) for i, c in enumerate(st) if c]
    parts.sort(key=lambda p: int(p[0][1:]))
    return ' + '.join(f'{c}{n}' if c > 1 else n for n, c in parts) if parts else '0'


def _level_constants(run, s, v, C, names):
    """Rotational constants (A, B, C; cm-1) of every computed level: the VCI expectation value of Watson's mu
    tensor (harmonic and anharmonic vibration-rotation terms) plus the perturbative Coriolis terms weighted by the
    level's mean quanta (diabml/rotor.py)."""
    from ..rotor import coriolis_alpha, rotational_constant_fn
    t0 = time.time()
    Bm = v.operator(rotational_constant_fn(s['modes']), ncomp=3, nmode=2)
    Bw = np.stack([np.einsum('nk,nk->k', C, m @ C) for m in Bm], 1)
    nq = (C ** 2).T @ v.S
    al, Be = coriolis_alpha(s['modes'])
    B = Bw - (nq + 0.5) @ al
    ref = json.loads((W.C.PACKAGE_ROOT / 'data' / 'reference' / 'ch2oo_rotational.json').read_text())['states']
    g = ref['ground']
    B0x = np.array([g['A'], g['B'], g['C']])
    run.log(f'rotational constants (cm-1): equilibrium A, B, C {Be[0]:.4f}, {Be[1]:.5f}, {Be[2]:.5f}; ground level '
            f'{B[0, 0]:.4f}, {B[0, 1]:.5f}, {B[0, 2]:.5f}; expt. {B0x[0]:.4f}, {B0x[1]:.5f}, {B0x[2]:.5f} '
            f'({100 * (B[0, 0] / B0x[0] - 1):+.1f}, {100 * (B[0, 1] / B0x[1] - 1):+.1f}, '
            f'{100 * (B[0, 2] / B0x[2] - 1):+.1f}%) ({time.time() - t0:.0f} s)')
    summary = {'Be': Be.tolist(), 'B0': B[0].tolist(), 'B0_expt': B0x.tolist(), 'alpha_coriolis': al.tolist()}
    fund = v.fundamentals(C)
    for state, d in ref.items():
        if state == 'ground' or state not in names:
            continue
        j = fund[names.index(state)][0]
        r = B[j] / B[0]
        x = np.array([d['ratios'][a] for a in 'ABC'])
        summary[f'{state}_ratios'] = {'computed': r.tolist(), 'expt': x.tolist()}
        run.log(f'  {state}: B_v / B_0 computed A {r[0]:.5f}, B {r[1]:.5f}, C {r[2]:.5f}; expt. {x[0]:.5f}, '
                f'{x[1]:.5f}, {x[2]:.5f}')
    return {'B': B, 'nq': nq, 'summary': summary}


def _run_vci(run, s, ecut, ch_quanta, wc, coriolis=True, quiet=False, states_cm=None):
    from .. import vpt2 as V
    from ..vci import VCI
    w = s['modes'].omega
    t0 = time.time()
    v = VCI(w, s['potential'], ecut=ecut, nmax=int(wc.get('vci_nmax', 8)), nmode=int(wc.get('vci_nmode', 3)),
            extra=int(wc.get('vci_extra', 6)), n4max=int(wc.get('vci_n4max', 0)), separate=w > 2000,     # C-H / C-D stretches
            sep_quanta=int(ch_quanta))
    v.build()
    if coriolis:
        zeta, B = V.coriolis(s['modes'])
        v.add_coriolis(B, zeta)
    harm = v.S @ w
    top = float(wc.get('vci_states_cm', 4500.0)) if states_cm is None else float(states_cm)
    k = int(np.sum(harm <= top + 200)) + 10
    k = min(k, int(wc.get('vci_max_states', 1500)), len(v.S) - 2)
    E, C = v.solve(k)
    if not quiet:
        run.log(f'VCI basis {len(v.S)} states (CH2 bends, torsions, C-O and O-O to {ecut:g} cm-1 x C-H stretches to '
                f'{ch_quanta} quanta; quanta per mode {v.N.tolist()}), {v.nmode}-mode potential, H nnz {v.diag["nnz"]:,}: '
                f'{k} levels in {time.time() - t0:.0f} s (potential {v.timing["potential_s"]:.0f} s, '
                f'assembly {v.timing["assemble_s"]:.0f} s, solve {v.timing["solve_s"]:.0f} s)')
    return v, E, C


def stage_well9d_vci(run):
    from ..vci import KM_MOL, transition_moments
    wc = run.cfg['well9d']
    s = setup(run)
    names, w = s['names'], s['modes'].omega
    run.log('harmonic (Eckart normal modes): ' + ', '.join(f'{names[k]} {w[k]:.1f}' for k in np.argsort(
        [int(n[1:]) for n in names])))
    ecut, chq = float(wc.get('vci_ecut', 7000.0)), int(wc.get('vci_ch_quanta', 3))
    v, E, C = _run_vci(run, s, ecut, chq, wc)
    st = s['stats']
    run.log(f'potential grids: {st["points"]:,} points, {100 * st["outside_box"] / max(st["points"], 1):.0f}% outside '
            f'the fitted box, {st["guarded"]} far points (harmonic energy > {wc.get("vci_guard_cm", 12000.0):g} cm-1) '
            f'given the harmonic value')
    zpe = float(E[0])
    Ex = E - zpe
    asg = v.assign(C, top=3)
    fund = v.fundamentals(C)
    # convergence: a smaller basis
    chk = float(wc.get('vci_ecut_check', 0) or 0)
    chq2 = int(wc.get('vci_ch_quanta_check', chq))
    conv = None
    if chk > 0:
        v2, E2, C2 = _run_vci(run, s, chk, chq2, wc, quiet=True, states_cm=float(w.max()) + 100)   # fundamentals only
        f2 = v2.fundamentals(C2)
        conv = {'ecut_check': chk, 'ch_quanta_check': chq2, 'zpe_change': zpe - float(E2[0]),
                'fundamental_change': [float(Ex[f[0]] - (E2[g[0]] - E2[0])) for f, g in zip(fund, f2)]}
        run.log(f'convergence: basis {len(v2.S)} ({chk:g} cm-1, C-H {chq2}) -> {len(v.S)} ({ecut:g} cm-1, C-H {chq}): '
                f'zero-point energy '
                f'{conv["zpe_change"]:+.2f} cm-1; fundamentals ' + ', '.join(
                    f'{names[i]} {d:+.1f}' for i, d in sorted(enumerate(conv['fundamental_change']),
                                                              key=lambda t: int(names[t[0]][1:]))))
        del v2, C2
    # dipole, intensities
    T = None
    if s['dipole'] is not None:
        t0 = time.time()
        M = v.operator(s['dipole'], ncomp=3, nmode=int(wc.get('vci_dipole_nmode', 2)))
        T = transition_moments(C, M)
        run.detail(f'dipole operator ({wc.get("vci_dipole_nmode", 2)}-mode) and transition moments in '
                   f'{time.time() - t0:.0f} s')
    rot = _level_constants(run, s, v, C, names)
    vp = run.p('results', 'well9d_vpt2.json')
    gv = {m['mode']: m['gvpt2'] for m in json.loads(vp.read_text())['modes']} if vp.exists() else {}
    ref = json.loads((W.C.PACKAGE_ROOT / 'data' / 'reference' / 'ch2oo_rotational.json').read_text())['band_origins']
    bands = json.loads((W.C.PACKAGE_ROOT / 'data' / 'reference' / 'su2013_ch2oo_bands.json').read_text())
    ex_int = {m[0]: m[2][1] for m in bands['modes'] if m[2]}
    rows = []
    order = sorted(range(len(w)), key=lambda i: int(names[i][1:]))
    i6 = None
    run.log(f'{"":4s} {"mode":20s} {"sym":4s} {"harm":>7s} {"GVPT2":>7s} {"VCI":>7s} {"wt":>5s} {"expt":>7s} '
            f'{"km/mol":>7s} {"type":>9s}')
    for i in order:
        j, wt = fund[i]
        nu = float(Ex[j])
        row = {'mode': names[i], 'label': s['labels'][i], 'symmetry': s['symmetry'][i], 'harmonic': float(w[i]),
               'gvpt2': gv.get(names[i]), 'vci': nu, 'weight': wt, 'state': int(j), 'expt': ref.get(names[i]),
               'expt_rel_intensity': ex_int.get(names[i])}
        if T is not None:
            m = T[:, j, 0]
            row['km_mol'] = float(KM_MOL * nu * (m ** 2).sum())
            row['type'] = {ax: float(m[a] ** 2 / max((m ** 2).sum(), 1e-30)) for a, ax in enumerate('abc')}
        if conv:
            row['convergence'] = conv['fundamental_change'][i]
        rows.append(row)
        if names[i] == 'v6':
            i6 = row
        typ = ''.join(f'{ax}{100 * f:.0f} ' for ax, f in row.get('type', {}).items() if f > 0.05).strip()
        run.log(f'{names[i]:4s} {row["label"]:20s} {row["symmetry"]:4s} {w[i]:7.1f} '
                f'{row["gvpt2"] if row["gvpt2"] is not None else float("nan"):7.1f} {nu:7.1f} {wt:5.2f} '
                f'{row["expt"] if row["expt"] else float("nan"):7.1f} {row.get("km_mol", float("nan")):7.1f} {typ:>9s}')
    if T is not None and i6 and i6.get('km_mol'):
        run.log('relative intensities (O-O = 100), VCI vs expt: ' + ', '.join(
            f'{r["mode"]} {100 * r["km_mol"] / i6["km_mol"]:.0f}' + (f' ({r["expt_rel_intensity"]})'
                                                                       if r['expt_rel_intensity'] else '')
            for r in rows))
    run.log(f'zero-point energy {zpe:.1f} cm-1 (harmonic {0.5 * w.sum():.1f}' +
            (f', GVPT2 {json.loads(vp.read_text())["zpe_cm-1"]:.1f}' if vp.exists() else '') + ')')
    levels = []
    for j in range(len(E)):
        comp = asg[j]
        levels.append({'E': float(Ex[j]), 'label': _label(comp[0][0], names), 'weight': comp[0][1],
                       'components': [(_label(c, names), wgt) for c, wgt in comp if wgt > 0.05]})
    low = [l for l in levels if 0 < l['E'] <= 2000]
    run.log('levels to 2000 cm-1: ' + ', '.join(f'{l["E"]:.0f} {l["label"]}'
                                                + (f' ({100 * l["weight"]:.0f}%)' if l['weight'] < 0.7 else '')
                                                for l in low))
    for l in levels:
        if l['E'] <= 3200 and len(l['components']) > 1 and l['weight'] < 0.7:
            run.detail(f'mixed level {l["E"]:.1f}: ' + ', '.join(f'{100 * wgt:.0f}% {c}' for c, wgt in l['components']))
    out = {'model': str(s['mp']), 'dipole': str(s['dp']) if s['dp'] else None, 'ecut': ecut,
           'basis_size': int(len(v.S)), 'quanta_per_mode': v.N.tolist(), 'nmode': v.nmode,
           'ch_quanta': chq, 'coriolis': True, 'zpe_cm-1': zpe,
           'zpe_harmonic_cm-1': float(0.5 * w.sum()), 'mode_names': names, 'harmonic': w.tolist(),
           'fundamentals': rows, 'convergence': conv, 'grid_stats': st, 'levels': levels,
           'potential_min_on_grids_cm-1': v.diag['potential_min_on_grids_cm-1']}
    out['rotational'] = rot['summary']
    npz = {'E': Ex, 'zpe': zpe, 'quanta': np.array([c[0][0] for c in asg]), 'B': rot['B'], 'nq': rot['nq']}
    if T is not None:
        out.update(_spectra(run, wc, Ex, T, rows, npz, w))
        npz['T'] = T
    np.savez(run.p('results', 'well9d_vci.npz'), **npz)
    run.p('results', 'well9d_vci.json').write_text(json.dumps(out, indent=1, default=float))
    run.log('wrote results/well9d_vci.json, results/well9d_vci.npz')


def _missing_population(omega, top, T):
    """Fraction of a harmonic oscillator set's Boltzmann population at T above excitation energy top (cm-1): how
    much a level list ending at top leaves out."""
    from ..vci import enumerate_basis
    kT = 0.6950348 * T
    Q = np.prod(1.0 / (1.0 - np.exp(-np.asarray(omega) / kT)))
    E = enumerate_basis(omega, top, 200) @ np.asarray(omega)
    return float(max(0.0, 1.0 - np.exp(-E / kT).sum() / Q))


def _spectra(run, wc, Ex, T, rows, npz, omega):
    """Absorption at vci_temperature and emission from CH2OO with vci_emission_kJmol above the zero-point level
    (thermal-equivalent populations: the vibrational temperature with that mean energy)."""
    from ..dynamics.ir import find_peaks
    from ..vci import boltzmann, broaden, line_list, temperature_for_energy
    Tk = float(wc.get('vci_temperature', 300.0))
    e_exc = float(wc.get('vci_emission_kJmol', 13.0)) * KJMOL_CM
    fw = float(wc.get('vci_broaden_cm', 12.0))
    top = float(Ex[-1])
    grid = np.arange(400.0, 3500.0, 0.5)
    absl = line_list(Ex, T, boltzmann(Ex, Tk), 'absorption')
    Tv = temperature_for_energy(Ex, e_exc)
    emt = line_list(Ex, T, boltzmann(Ex, Tv), 'emission')
    miss = _missing_population(omega, top, Tv)
    spec = {'absorption': broaden(absl, grid, fw), 'emission': broaden(emt, grid, fw)}
    run.log(f'absorption at {Tk:g} K: {len(absl)} lines; emission with {e_exc:.0f} cm-1 '
            f'({wc.get("vci_emission_kJmol", 13.0)} kJ/mol) above the zero-point level, populated at the vibrational '
            f'temperature with that mean energy, T_vib {Tv:.0f} K: {len(emt)} lines; population above the highest '
            f'computed level ({top:.0f} cm-1), harmonic estimate: {100 * miss:.1f}%')
    if miss > 0.02:
        run.log(f'  warning: {100 * miss:.1f}% of the T_vib population lies above the computed levels: raise '
                f'vci_states_cm')
    for k, y in spec.items():
        npz['spectrum_' + k] = y
    npz['grid'] = grid
    peaks = {}
    for k, y in spec.items():
        peaks[k] = find_peaks(grid, y / y.max(), min_cm=400.0)[:8]
        run.log(f'  {k}: peaks ' + ', '.join(f'{p:.0f} ({a:.2f})' for p, a in peaks[k]))
    # O-O band: shift of the hot emission from the cold absorption (intensity-weighted over 870-950 cm-1)
    sel = (grid >= 870) & (grid <= 950)
    cen = {k: float((grid[sel] * y[sel]).sum() / y[sel].sum()) for k, y in spec.items()}
    hot = emt[(emt[:, 0] >= 870) & (emt[:, 0] <= 950)]
    frac_hot = float(hot[hot[:, 3] != 0, 1].sum() / max(hot[:, 1].sum(), 1e-300)) if len(hot) else 0.0
    run.log(f'O-O band centre (870-950 cm-1, intensity-weighted): absorption {cen["absorption"]:.1f}, emission '
            f'{cen["emission"]:.1f} cm-1 ({cen["emission"] - cen["absorption"]:+.1f}); hot bands (lower level not '
            f'the ground state) carry {100 * frac_hot:.0f}% of the emission there')
    _figures(run, grid, spec, absl, rows, Tk, Tv, e_exc, fw)
    return {'spectra': {'temperature_K': Tk, 'emission_energy_cm-1': e_exc, 'T_vib_K': Tv, 'broaden_cm-1': fw,
                        'peaks': peaks, 'oo_band_centre': cen, 'oo_emission_hot_fraction': frac_hot,
                        'missing_population': miss,
                        'strongest_absorption': [{'nu': float(l[0]), 'km_mol': float(l[1]), 'from': int(l[2]),
                                                  'to': int(l[3])} for l in absl[np.argsort(-absl[:, 1])][:25]],
                        'strongest_emission': [{'nu': float(l[0]), 'rate': float(l[1]), 'from': int(l[2]),
                                                'to': int(l[3])} for l in emt[np.argsort(-emt[:, 1])][:25]]}}


def _figures(run, grid, spec, absl, rows, Tk, Tv, e_exc, fw):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from .ir import _reference
    ref = _reference(run)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={'width_ratios': [1.5, 1]})
    ax = axs[0]
    lo, hi = 780, 1560
    sel = (grid >= lo) & (grid <= hi)
    y = spec['absorption']
    ax.plot(grid[sel], y[sel] / y[sel].max(), color='#1f5fa8', lw=1.6)
    h = [Line2D([], [], color='#1f5fa8', lw=1.6, label=f'VCI, {Tk:g} K ({fw:g} cm⁻¹ Gaussian, no rotation)')]
    if ref is not None:
        rs = (ref[0] >= lo) & (ref[0] <= hi)
        ax.plot(ref[0][rs], ref[1][rs] + 1.15, color='0.2', lw=1)
        h.append(Line2D([], [], color='0.2', lw=1, label=ref[2]))
    inw = absl[(absl[:, 0] >= lo) & (absl[:, 0] <= hi)]
    mx = inw[:, 1].max()
    for l in inw:
        if l[1] > 0.005 * mx:
            ax.vlines(l[0], -0.27, -0.27 + 0.22 * l[1] / mx, color='#1f5fa8' if l[2] == 0 else '#d08a2a', lw=1)
    h += [Line2D([], [], color='#1f5fa8', lw=1, label='lines from the ground state'),
          Line2D([], [], color='#d08a2a', lw=1, label='hot bands'),
          Line2D([], [], color='0.6', ls=':', lw=0.8, label='expt. band origins')]
    for r in rows:
        if r['expt'] and lo <= r['expt'] <= hi:
            ax.axvline(r['expt'], color='0.6', ls=':', lw=0.8)
            ax.text(r['expt'] + 4, 2.5, r['mode'], ha='right', fontsize=8, color='0.35')
    ax.set_xlim(hi, lo)
    ax.set_ylim(-0.32, 2.62)
    ax.set_yticks([])
    ax.set_xlabel('wavenumber (cm⁻¹)')
    ax.set_title('Absorption: VCI on the 9-D surface vs experiment', fontsize=11)
    ax.legend(handles=h, fontsize=7.5, loc='upper left', bbox_to_anchor=(0.42, 0.475), frameon=False)
    ax = axs[1]
    sel = (grid >= 400) & (grid <= 3400)
    for k, c, lab in (('emission', '#b5442c', f'emission, {e_exc / KJMOL_CM:.0f} kJ/mol (T_vib {Tv:.0f} K)'),
                      ('absorption', '#1f5fa8', f'absorption, {Tk:g} K')):
        y = spec[k][sel]
        ax.plot(grid[sel], y / y.max(), color=c, lw=1.2, label=lab)
    ax.set_xlim(3400, 400)
    ax.set_yticks([])
    ax.set_xlabel('wavenumber (cm⁻¹)')
    ax.set_title('Emission (photon rate) and absorption, full range', fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(run.p('figures', 'VCI1_spectra.png'), dpi=150)
    plt.close(fig)
    # O-O region, absorption vs emission
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    sel = (grid >= 800) & (grid <= 1000)
    for k, c, lab in (('absorption', '#1f5fa8', f'VCI absorption, {Tk:g} K'),
                      ('emission', '#b5442c', f'VCI emission, {e_exc / KJMOL_CM:.0f} kJ/mol (T_vib {Tv:.0f} K)')):
        y = spec[k][sel]
        ax.plot(grid[sel], y / y.max(), color=c, lw=1.5, label=lab)
    if ref is not None:
        rs = (ref[0] >= 800) & (ref[0] <= 1000)
        ax.plot(ref[0][rs], ref[1][rs] / ref[1][rs].max(), color='0.45', lw=0.8, label='expt. absorption (Su et al. 2013)')
    ax.set_xlim(1000, 800)
    ax.set_ylim(-0.03, 1.12)
    ax.set_yticks([])
    ax.set_xlabel('wavenumber (cm⁻¹)')
    ax.set_title('O–O stretch and CH₂ wag: cold absorption vs hot emission', fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.text(0.99, 0.97, 'VCI bands carry no rotational structure:\nthe expt. P/R wings are rotation',
            transform=ax.transAxes, ha='right', va='top', fontsize=7, color='0.4')
    fig.tight_layout()
    fig.savefig(run.p('figures', 'VCI2_oo_band.png'), dpi=150)
    plt.close(fig)
    run.log('figures/VCI1_spectra.png, figures/VCI2_oo_band.png')


# ------------------------------------------------------------------------------------------ rotational band shapes
def _select(lines, lo, hi, frac, nmax):
    """The strongest vibrational lines in [lo, hi] that together carry `frac` of the intensity there (at most
    nmax)."""
    m = (lines[:, 0] >= lo) & (lines[:, 0] <= hi)
    L = lines[m]
    L = L[np.argsort(-L[:, 1])]
    c = np.cumsum(L[:, 1]) / L[:, 1].sum()
    n = min(int(np.searchsorted(c, frac)) + 1, nmax, len(L))
    return L[:n], float(c[n - 1])


def _rot_spectrum(Ex, T, Bl, lines, kind, Trot, grid, fwhm, log):
    """Rotationally resolved spectrum: every selected vibrational line (nu, strength, lower, upper) spread over its
    asymmetric-rotor band (a/b/c parts from the transition moment, each level's own constants); each band keeps
    its vibrational strength."""
    from ..rotor import band_lines, contour
    bands = []
    t0 = time.time()
    for nu, s, lo_, up_ in lines:
        lo_, up_ = int(lo_), int(up_)
        lower, upper = (lo_, up_) if kind == 'absorption' else (up_, lo_)     # emission lines: (nu, s, upper, lower)
        m = T[:, upper, lower] ** 2
        m = m / m.sum()
        for a, comp in enumerate('abc'):
            if m[a] < 0.01:
                continue
            nus, I = band_lines(nu, tuple(Bl[lower]), tuple(Bl[upper]), comp, Trot, kind=kind)
            bands.append((nus, I * (s * m[a] / I.sum())))
    log(f'  {kind}: {len(lines)} vibrational bands, {sum(len(b[0]) for b in bands):,} rotational lines '
        f'({time.time() - t0:.0f} s)')
    return contour(bands, grid, fwhm)


def _pr(p):
    return f'{p[0]:.1f} / {p[1]:.1f} (spacing {p[1] - p[0]:.1f})' if p[0] is not None else 'not found'


def _side_peaks(nu, y, centre, lo=8.0, hi=25.0, smooth=6.0):
    """P and R branch maxima: the largest values 8-25 cm-1 below and above the band centre after Gaussian
    smoothing (FWHM `smooth` cm-1), so that sharp hot-band Q branches are not taken for the branch maxima."""
    nu, y = np.asarray(nu, float), np.asarray(y, float)
    o = np.argsort(nu)
    g = np.arange(centre - hi - 10, centre + hi + 10, 0.1)
    if g[0] < nu[o][0] or g[-1] > nu[o][-1]:
        return None, None
    yg = np.interp(g, nu[o], y[o])
    s = smooth / 2.3548 / 0.1
    k = np.arange(-int(4 * s) - 1, int(4 * s) + 2)
    w = np.exp(-0.5 * (k / s) ** 2)
    ys = np.convolve(yg, w / w.sum(), mode='same')
    pm = (g >= centre - hi) & (g <= centre - lo)
    rm = (g >= centre + lo) & (g <= centre + hi)
    return float(g[pm][np.argmax(ys[pm])]), float(g[rm][np.argmax(ys[rm])])


def stage_well9d_bands(run):
    """Rotationally resolved absorption and emission from the VCI levels (results/well9d_vci.npz): every strong
    vibrational band, hot bands included, gets its asymmetric-rotor structure at [well9d] band_T_rot, with the
    rotational constants of its own two levels (ground level anchored to experiment: band_ground = "experiment",
    or "computed"); Gaussian instrument function band_fwhm_cm. Compared with Su et al. (2013).
        -> results/well9d_bands.json, results/well9d_bands.npz, figures/VCI3_bands.png, and the report vci_9d.html
           (diabml/workflow/vci_report.py)"""
    from ..vci import boltzmann, line_list, temperature_for_energy
    from .ir import _reference
    wc = run.cfg['well9d']
    zp = run.p('results', 'well9d_vci.npz')
    if not zp.exists():
        raise SystemExit(f'{zp} not found: run well9d_vci first')
    z = np.load(zp)
    if 'B' not in z or 'T' not in z:
        raise SystemExit(f'{zp} has no rotational constants or transition moments: rerun well9d_vci (with the '
                         f'dipole surface)')
    r = json.loads(run.p('results', 'well9d_vci.json').read_text())
    Ex, T, B = z['E'], z['T'], z['B'].copy()
    ground = str(wc.get('band_ground', 'experiment'))
    if ground == 'experiment':
        B = B * (np.array(r['rotational']['B0_expt']) / B[0])[None, :]
    Trot = float(wc.get('band_T_rot', 300.0))
    Tabs = float(wc.get('vci_temperature', 300.0))
    fw = float(wc.get('band_fwhm_cm', 1.0))
    frac = float(wc.get('band_fraction', 0.98))
    nmax = int(wc.get('band_max', 150))
    lo, hi = 780.0, 1560.0
    grid = np.arange(lo - 40, hi + 40, 0.05)
    run.log(f'rotational constants: ground level {ground} ({B[0, 0]:.4f}, {B[0, 1]:.5f}, {B[0, 2]:.5f} cm-1), other '
            f'levels by their computed ratio to it; T_rot {Trot:g} K, instrument FWHM {fw:g} cm-1')
    absl = line_list(Ex, T, boltzmann(Ex, Tabs), 'absorption')
    A_sel, A_cov = _select(absl, lo, hi, frac, nmax)
    e_exc = float(wc.get('vci_emission_kJmol', 13.0)) * KJMOL_CM
    Tv = temperature_for_energy(Ex, e_exc)
    eml = line_list(Ex, T, boltzmann(Ex, Tv), 'emission')
    E_sel, E_cov = _select(eml, lo, hi, frac, nmax)
    run.log(f'bands: absorption {len(A_sel)} (carrying {100 * A_cov:.1f}% of the {lo:g}-{hi:g} cm-1 intensity), '
            f'emission ({e_exc / KJMOL_CM:g} kJ/mol, T_vib {Tv:.0f} K) {len(E_sel)} ({100 * E_cov:.1f}%)')
    ya = _rot_spectrum(Ex, T, B, A_sel, 'absorption', Trot, grid, fw, run.log)
    ye = _rot_spectrum(Ex, T, B, E_sel, 'emission', Trot, grid, fw, run.log)
    ref = _reference(run)
    names = r['mode_names']
    f6 = next(f for f in r['fundamentals'] if f['mode'] == 'v6')
    c6 = f6['vci']
    out = {'T_rot_K': Trot, 'fwhm_cm-1': fw, 'ground': ground, 'absorption_bands': len(A_sel),
           'emission_bands': len(E_sel), 'T_vib_K': Tv}
    pa = _side_peaks(grid, ya, c6)
    pe = _side_peaks(grid, ye, c6)
    msg = (f'O-O band (VCI origin {c6:.1f}): P/R branch maxima (6 cm-1 smoothing) absorption {_pr(pa)}; '
           f'emission {_pr(pe)}')
    out['oo'] = {'origin': c6, 'absorption_PR': pa, 'emission_PR': pe}
    if ref is not None:
        px = _side_peaks(ref[0], ref[1], f6['expt'] or c6)
        msg += f'; expt. {_pr(px)}'
        out['oo']['expt_PR'] = px
        # shape agreement: correlation over 840-1000 cm-1 after shifting the computed band onto the expt. origin
        sh = (f6['expt'] or c6) - c6
        m = (ref[0] >= 840) & (ref[0] <= 1000)
        yc = np.interp(ref[0][m], grid + sh, ya)
        out['oo']['shape_correlation'] = float(np.corrcoef(yc, ref[1][m])[0, 1])
        msg += f'; shape correlation with expt. (840-1000, computed shifted {sh:+.1f}): {out["oo"]["shape_correlation"]:.3f}'
    run.log(msg)
    # sensitivity to the vibration-rotation constants (the Coriolis part is perturbative and uncertain for the
    # near-resonant pairs v7/v9, v6/v8, v4/v5): the O-O region again with every level at the ground constants
    sel = A_sel[(A_sel[:, 0] >= 840) & (A_sel[:, 0] <= 1000)]
    if not len(sel):
        sel = A_sel[:3]
    Br = np.repeat(B[:1], len(B), 0)
    yr = _rot_spectrum(Ex, T, Br, sel, 'absorption', Trot, grid, fw, run.detail)
    ys = _rot_spectrum(Ex, T, B, sel, 'absorption', Trot, grid, fw, run.detail)
    pr = _side_peaks(grid, yr, c6)
    m = (grid >= 840) & (grid <= 1000)
    nr = lambda y: y[m] / max(y[m].max(), 1e-300)
    sens = {'rigid_PR': pr, 'rigid_vs_full_rms': float(np.sqrt(np.mean((nr(yr) - nr(ys)) ** 2)))}
    if ref is not None:
        sh = (f6['expt'] or c6) - c6
        mx = (ref[0] >= 840) & (ref[0] <= 1000)
        sens['rigid_shape_correlation'] = float(np.corrcoef(np.interp(ref[0][mx], grid + sh, yr), ref[1][mx])[0, 1])
    out['oo']['sensitivity'] = sens
    run.log(f'  with rigid rotational constants (all levels as the ground level): P/R {_pr(pr)}' + (f', shape correlation {sens["rigid_shape_correlation"]:.3f}'
                                        if 'rigid_shape_correlation' in sens else '') +
            f'; rms change of the normalised O-O band {sens["rigid_vs_full_rms"]:.3f}')
    np.savez(run.p('results', 'well9d_bands.npz'), grid=grid, absorption=ya, emission=ye)
    run.p('results', 'well9d_bands.json').write_text(json.dumps(out, indent=1, default=float))
    _band_figure(run, grid, ya, ye, ref, r, Tabs, Tv, Trot, fw, e_exc)
    run.log('wrote results/well9d_bands.json, results/well9d_bands.npz, figures/VCI3_bands.png')
    from .vci_report import stage_vci_report
    stage_vci_report(run)


def _band_figure(run, grid, ya, ye, ref, r, Tabs, Tv, Trot, fw, e_exc):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f6 = next(f for f in r['fundamentals'] if f['mode'] == 'v6')
    sh = (f6['expt'] or f6['vci']) - f6['vci']
    fig = plt.figure(figsize=(13, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.32, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    lo, hi = 780, 1560
    m = (grid >= lo) & (grid <= hi)
    ax.plot(grid[m], ya[m] / ya[m].max(), color='#1f5fa8', lw=1.0,
            label=f'VCI + asymmetric rotor, absorption ({Tabs:g} K vib., {Trot:g} K rot., {fw:g} cm⁻¹)')
    if ref is not None:
        rs = (ref[0] >= lo) & (ref[0] <= hi)
        ax.plot(ref[0][rs], ref[1][rs] / ref[1][rs].max() + 1.05, color='0.2', lw=0.9, label=ref[2])
    for f in r['fundamentals']:
        if lo <= f['vci'] <= hi and f.get('km_mol', 0) > 3:
            ax.text(f['vci'], -0.12, f['mode'], ha='center', fontsize=8, color='#1f5fa8')
        if f['expt'] and lo <= f['expt'] <= hi:
            ax.text(f['expt'], 2.12, f['mode'], ha='center', fontsize=8, color='0.3')
    ax.set_xlim(hi, lo)
    ax.set_ylim(-0.2, 2.25)
    ax.set_yticks([])
    ax.set_xlabel('wavenumber (cm⁻¹)')
    ax.set_title('Absorption with rotational structure: computed (bottom) vs experiment (top)', fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(0.0, 0.5))
    ax = fig.add_subplot(gs[1, 0])
    a, b = 860, 960
    m = (grid >= a - 5) & (grid <= b + 5)
    ax.plot(grid[m] + sh, ya[m] / ya[m].max(), color='#1f5fa8', lw=1.3,
            label=f'VCI absorption, shifted {sh:+.1f} cm⁻¹ onto the expt. origin')
    if ref is not None:
        rs = (ref[0] >= a) & (ref[0] <= b)
        ax.plot(ref[0][rs], ref[1][rs] / ref[1][rs].max(), color='0.25', lw=1.0, label='expt. (Su et al. 2013)')
    ax.set_xlim(b, a)
    ax.set_ylim(-0.03, 1.2)
    ax.set_yticks([])
    ax.set_xlabel('wavenumber (cm⁻¹)')
    ax.set_title('O–O stretch band shape (a-type), absorption', fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax = fig.add_subplot(gs[1, 1])
    m = (grid >= a - 5) & (grid <= b + 5)
    ax.plot(grid[m], ya[m] / ya[m].max(), color='#1f5fa8', lw=1.2, label=f'absorption ({Tabs:g} K)')
    ax.plot(grid[m], ye[m] / ye[m].max(), color='#b5442c', lw=1.2,
            label=f'emission, {e_exc / KJMOL_CM:g} kJ/mol (T_vib {Tv:.0f} K)')
    ax.set_xlim(b, a)
    ax.set_ylim(-0.03, 1.2)
    ax.set_yticks([])
    ax.set_xlabel('wavenumber (cm⁻¹)')
    ax.set_title('O–O and wag region: cold absorption vs hot emission (computed)', fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    fig.savefig(run.p('figures', 'VCI3_bands.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
