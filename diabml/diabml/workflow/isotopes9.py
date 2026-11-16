"""Isotopologues of CH2OO on the 9-D surface: microwave rotational constants and vibrational shifts.

    python -m diabml.workflow.ch2oo well9d_isotopes RUN [-v]
        -> results/well9d_isotopes.json (and a section of vci_9d.html)

The surface does not depend on the masses, so every isotopologue is a prediction with no new electronic-structure
work. For each species of data/reference/ch2oo_isotopologues.json (Nakajima and Endo 2013; McCarthy et al. 2013):

  rotational constants  B_0 = B_e + dB_vib, with dB_vib from second-order perturbation theory: the ground-state
      average of Watson's inverse inertia tensor (linear term with <q_k> = -sum_l phi_kll / 4 omega_k from the
      cubic force field, quadratic term with <q_k^2> = 1/2) and the Coriolis term -1/2 sum_k alpha_k(Cor)
      (diabml/rotor.py). Reported twice: with B_e at the surface's own minimum, and with B_e at the
      semi-experimental equilibrium structure. The second separates the two things a surface can get wrong: the
      position of the minimum (B_e) and the shape around it (dB_vib).
  fundamentals  harmonic and GVPT2 for every species; VCI (as well9d_vci, basis [well9d] iso_ecut and
      vci_ch_quanta, levels to the highest fundamental) with intensities for the species in iso_vci.
      Modes are labelled by the parent (CH2OO) normal mode they overlap most (mass-weighted displacement).
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import ch2oo as W
from . import vci9

MHZ = 29979.2458


def reference():
    return json.loads((W.C.PACKAGE_ROOT / 'data' / 'reference' / 'ch2oo_isotopologues.json').read_text())


def vib_correction(modes, f3):
    """B_0 - B_e (3,) in cm-1 from the cubic force constants f3 (dimensionless normal coordinates)."""
    from ..rotor import coriolis_alpha, rotational_constant_fn
    w = modes.omega
    n = len(w)
    fn = rotational_constant_fn(modes)
    Be = fn(np.zeros((1, n)))[0]
    e = np.eye(n)
    g = np.array([(fn(e[k:k + 1])[0] - fn(-e[k:k + 1])[0]) / 2 for k in range(n)])
    h = np.array([(fn(e[k:k + 1])[0] + fn(-e[k:k + 1])[0] - 2 * Be) / 2 for k in range(n)])
    qm = np.array([-sum(f3[k, l, l] for l in range(n)) / (4 * w[k]) for k in range(n)])
    al, _ = coriolis_alpha(modes)
    return qm @ g + 0.5 * h.sum(0) - 0.5 * al.sum(0)


def _parent_labels(modes, parent, names):
    """For each mode: (parent mode name, squared overlap), a one-to-one assignment that maximises the total
    overlap of the Cartesian displacement patterns (deuteration mixes modes: the labels are then approximate)."""
    from scipy.optimize import linear_sum_assignment

    def disp(m):
        d = m.L / np.sqrt(np.repeat(m.masses, 3))[:, None]
        return d / np.linalg.norm(d, axis=0)
    O = (disp(modes).T @ disp(parent)) ** 2
    r, c = linear_sum_assignment(-O)
    lab = dict(zip(r, c))
    return [(names[int(lab[k])], float(O[k, lab[k]])) for k in range(len(modes.omega))]


GRID = np.arange(760.0, 1500.0, 0.05)
REF_SPECTRA = {'CH2OO': 'huang2016_ch2oo_fig2a.csv', 'CD2OO': 'huang2016_cd2oo_fig2b.csv'}
# interference in the measured spectra (precursor depletion / off-scale): left out of the comparisons
MASKED = {'CH2OO': [(1085.0, 1135.0)], 'CD2OO': [(826.0, 850.0)]}


def _contour(run, wc, s, v, E, C, T, B0x):
    """Rotationally resolved 300 K absorption of one isotopologue from its VCI levels (as well9d_bands): level
    rotational constants = experimental ground-state constants x computed B_v / B_0."""
    from ..rotor import coriolis_alpha, rotational_constant_fn
    from ..vci import boltzmann, line_list
    Bm = v.operator(rotational_constant_fn(s['modes']), ncomp=3, nmode=2)
    Bw = np.stack([np.einsum('nk,nk->k', C, m @ C) for m in Bm], 1)
    al, _ = coriolis_alpha(s['modes'])
    B = Bw - ((C ** 2).T @ v.S + 0.5) @ al
    B = B * (B0x / B[0])[None, :]
    Ex = E - E[0]
    lines = line_list(Ex, T, boltzmann(Ex, float(wc.get('vci_temperature', 300.0))), 'absorption')
    sel, _ = vci9._select(lines, GRID[0] + 20, GRID[-1] - 20, float(wc.get('band_fraction', 0.98)),
                          int(wc.get('band_max', 150)))
    return vci9._rot_spectrum(Ex, T, B, sel, 'absorption', float(wc.get('band_T_rot', 300.0)), GRID,
                              float(wc.get('band_fwhm_cm', 1.0)), run.detail)


def _load_ref(name):
    f = W.C.PACKAGE_ROOT / 'data' / 'reference' / REF_SPECTRA.get(name, '-')
    if not f.exists():
        return None
    d = np.loadtxt(f, delimiter=',', comments='#')
    return d[:, 0], d[:, 1]


def _compare(run, spec, out):
    """Computed contours against the digitized spectra of Huang et al. (2016): sharp Q-branch maxima near each
    strong computed fundamental, and the correlation of the two traces. Figure VCI4_isotopes.png."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bp = run.p('results', 'well9d_bands.npz')
    if 'CH2OO' not in spec and bp.exists():
        z = np.load(bp)
        spec = dict(spec, CH2OO=np.interp(GRID, z['grid'], z['absorption']))
    vj = run.p('results', 'well9d_vci.json')
    funds = {n: [(m['parent'], m['vci'], m.get('km_mol', 0.0)) for m in r['modes'] if 'vci' in m]
             for n, r in out['species'].items()}
    if vj.exists():
        funds['CH2OO'] = [(f['mode'], f['vci'], f.get('km_mol', 0.0)) for f in json.loads(vj.read_text())['fundamentals']]
    res = {}
    names = [n for n in ('CH2OO', 'CD2OO') + tuple(spec) if n in spec and _load_ref(n) is not None]
    names = list(dict.fromkeys(names))
    if not names:
        return res
    fig, axs = plt.subplots(len(names), 1, figsize=(12, 3.9 * len(names)), squeeze=False)
    for ax, name in zip(axs[:, 0], names):
        nu, A = _load_ref(name)
        y = spec[name]
        keep = np.ones(len(nu), bool)
        for lo, hi in MASKED.get(name, []):
            keep &= ~((nu >= lo) & (nu <= hi))
        m = keep & (nu >= 800) & (nu <= 1450)
        yc = np.interp(nu, GRID, y)
        sc = float((yc[m] * A[m]).sum() / (yc[m] ** 2).sum())           # least-squares scale of the computed trace
        corr = float(np.corrcoef(yc[m], A[m])[0, 1])
        rows = []
        for par, f, km in sorted(funds.get(name, []), key=lambda t: -t[2]):
            if not (805 <= f <= 1445) or km < 15:
                continue
            w = keep & (np.abs(nu - f) <= 18)
            if w.sum() < 5:
                continue
            # the sharpest feature of the measured band (Q branch): largest excess over a 9-point running median
            base = np.array([np.median(A[max(0, i - 6):i + 7]) for i in range(len(A))])
            ex = (A - base)[w]
            rows.append({'mode': par, 'vci_origin': float(f), 'km_mol': float(km),
                         'expt_Q': float(nu[w][np.argmax(ex)]), 'expt_Q_sharpness': float(ex.max())})
        res[name] = {'correlation_800_1450': corr, 'bands': rows}
        run.log(f'{name} vs digitized expt. (Huang et al. 2016): correlation {corr:.3f} over 800-1450 cm-1; VCI '
                f'origin / sharpest expt. feature within 18 cm-1: ' + ', '.join(
                    f'{r["mode"]} {r["vci_origin"]:.1f} / {r["expt_Q"]:.1f} ({r["vci_origin"] - r["expt_Q"]:+.1f})'
                    for r in rows))
        ax.plot(nu[keep], np.where(np.r_[False, np.diff(nu[keep]) > 3], np.nan, A[keep]), color='0.25', lw=1.0,
                label='expt., digitized (Huang et al. 2016)')
        ax.plot(GRID, -sc * y, color='#1f5fa8' if name == 'CH2OO' else '#b02a6b', lw=0.9,
                label='VCI + asymmetric rotor (this surface), inverted')
        for lo, hi in MASKED.get(name, []):
            ax.axvspan(lo, hi, color='0.9', lw=0)
        top = np.nanmax(A[m])
        ax.set_xlim(1450, 800)
        ax.set_ylim(-1.15 * top, 1.15 * top)
        ax.axhline(0, color='0.6', lw=0.5)
        ax.set_yticks([])
        ax.set_title(f'{name}: measured (up) and computed (down); correlation {corr:.3f}; grey: interference, excluded',
                     fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc='lower left')
    axs[-1, 0].set_xlabel('wavenumber (cm⁻¹)')
    fig.tight_layout()
    fig.savefig(run.p('figures', 'VCI4_isotopes.png'), dpi=150)
    plt.close(fig)
    run.log('figures/VCI4_isotopes.png')
    return res


def stage_well9d_isotopes(run):
    from .. import vpt2 as V
    from ..geometry import NINE, ch2oo_positions9
    from ..vci import KM_MOL, transition_moments
    from ..vibdvr import B_CONV
    from .well9d import cartesian_hessian_fn
    wc = run.cfg['well9d']
    ref = reference()
    s0 = vci9.setup(run)
    model, P0, names = s0['model'], s0['P0'], s0['names']
    hess = cartesian_hessian_fn(model)
    H0 = hess(P0)
    xse = np.array([ref['structure_reSE'][n] for n in NINE])
    Pse = ch2oo_positions9(xse[None])[0]
    xm = s0['xm']
    run.log('equilibrium structure, surface vs semi-experimental (McCarthy et al. 2013): ' + ', '.join(
        f'{n} {xm[j]:.4f} / {xse[j]:.4f} ({xm[j] - xse[j]:+.4f})' for j, n in enumerate(NINE[:7])))
    do_vci = list(wc.get('iso_vci', ['CD2OO', 'CH2(18O)(18O)']))
    spec = {}
    out = {'structure': {'surface': dict(zip(NINE, xm.tolist())), 'reSE': dict(zip(NINE, xse.tolist()))},
           'species': {}, 'sources': ref['sources']}
    run.log(f'{"species":15s} {"A":>9s} {"B":>8s} {"C":>8s}  expt. (MHz); error %: surface minimum | at the '
            f'semi-experimental structure')
    worst = [0.0, 0.0]
    for name, d in ref['species'].items():
        m = np.array([ref['masses'][a] for a in d['atoms']])
        modes = V.cartesian_modes(H0, P0, m)
        f3, f4, _ = V.force_constants(hess, modes, float(wc.get('vpt2_step', 0.02)))
        dB = vib_correction(modes, f3)
        be = lambda P: B_CONV / V.principal_axes(P, m)[0]
        ex = np.array(d['ABC'])
        b_own, b_se = (be(P0) + dB) * MHZ, (be(Pse) + dB) * MHZ
        e_own, e_se = 100 * (b_own / ex - 1), 100 * (b_se / ex - 1)
        worst = [max(worst[0], np.abs(e_own).max()), max(worst[1], np.abs(e_se).max())]
        zeta, B = V.coriolis(modes)
        gv = V.gvpt2(modes.omega, f3, f4, B=B, zeta=zeta, max_gap=float(wc.get('vpt2_max_gap', 200.0)),
                     martin=float(wc.get('vpt2_martin', 1.0)))
        lab = _parent_labels(modes, s0['modes'], names)
        rec = {'atoms': d['atoms'], 'source': d['source'], 'expt_MHz': d['ABC'], 'Be_MHz': (be(P0) * MHZ).tolist(),
               'dB_vib_MHz': (dB * MHZ).tolist(), 'B0_MHz': b_own.tolist(), 'B0_at_reSE_MHz': b_se.tolist(),
               'error_pct': e_own.tolist(), 'error_at_reSE_pct': e_se.tolist(),
               'modes': [{'parent': lab[k][0], 'overlap': lab[k][1], 'harmonic': float(modes.omega[k]),
                          'gvpt2': float(gv['fundamentals'][k])} for k in range(len(modes.omega))]}
        run.log(f'{name:15s} {ex[0]:9.1f} {ex[1]:8.1f} {ex[2]:8.1f}  {e_own[0]:+.2f} {e_own[1]:+.2f} {e_own[2]:+.2f} | '
                f'{e_se[0]:+.3f} {e_se[1]:+.3f} {e_se[2]:+.3f}')
        run.detail('  GVPT2: ' + ', '.join(f'{r["parent"]} {r["gvpt2"]:.1f}' for r in rec['modes']))
        if name in do_vci:
            t0 = time.time()
            s = vci9.setup(run, log=False, masses=m)
            w = s['modes'].omega
            v, E, C = vci9._run_vci(run, s, float(wc.get('iso_ecut', 6000.0)), int(wc.get('vci_ch_quanta', 3)), wc,
                                    quiet=True, states_cm=float(w.max()) + 100)
            fund = v.fundamentals(C)
            T = transition_moments(C, v.operator(s['dipole'], ncomp=3, nmode=2)) if s['dipole'] is not None else None
            for k, (j, wt) in enumerate(fund):
                rec['modes'][k]['vci'] = float(E[j] - E[0])
                rec['modes'][k]['vci_weight'] = wt
                if T is not None:
                    mu = T[:, j, 0] ** 2
                    rec['modes'][k]['km_mol'] = float(KM_MOL * (E[j] - E[0]) * mu.sum())
                    rec['modes'][k]['type'] = {ax: float(mu[a] / max(mu.sum(), 1e-30)) for a, ax in enumerate('abc')}
            if T is not None:
                spec[name] = _contour(run, wc, s, v, E, C, T, np.array(d['ABC']) / MHZ)
            rec['vci_basis'] = int(len(v.S))
            rec['zpe_cm-1'] = float(E[0])
            run.log(f'  {name} VCI ({len(v.S)} basis functions, {time.time() - t0:.0f} s): ' + ', '.join(
                f'{r["parent"]} {r["vci"]:.1f}' + (f' ({r["km_mol"]:.0f})' if 'km_mol' in r else '')
                for r in sorted(rec['modes'], key=lambda r: int(r['parent'][1:]))) + '  [cm-1 (km/mol)]')
        out['species'][name] = rec
    out['worst_error_pct'] = {'surface_minimum': float(worst[0]), 'at_reSE': float(worst[1])}
    run.log(f'largest error over {3 * len(out["species"])} constants: {worst[0]:.2f}% with the surface\'s own minimum, '
            f'{worst[1]:.3f}% with the semi-experimental structure: the vibrational corrections (shape of the well) '
            f'are accurate; the remaining error is the position of the minimum')
    if spec:
        out['spectra'] = _compare(run, spec, out)
        np.savez(run.p('results', 'well9d_isotopes.npz'), grid=GRID, **{k: y for k, y in spec.items()})
    run.p('results', 'well9d_isotopes.json').write_text(json.dumps(out, indent=1, default=float))
    run.log('wrote results/well9d_isotopes.json')
    if run.p('results', 'well9d_vci.json').exists():
        from .vci_report import stage_vci_report
        stage_vci_report(run)
