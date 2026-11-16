"""Infrared spectra from dynamics: dipole time series on a learned surface, and their spectrum.

The IR absorption of a classical trajectory, with the harmonic quantum correction, is proportional to the power
spectrum of the dipole's time derivative:

    alpha(w) n(w)  ~  w^2 FT< mu(0) . mu(t) >  =  FT< mu'(0) . mu'(t) >  =  | FT[ mu'(t) ] |^2   (averaged)

(Wiener-Khinchin; the w^2 is the harmonic correction factor beta*hbar*w/(1 - exp(-beta*hbar*w)) times the
classical prefactor). So each trajectory's dipole is differentiated, windowed (Hann), Fourier transformed, and
|.|^2 is averaged over components and trajectories. Peak positions are what to compare with experiment; relative
intensities within one spectrum are meaningful; absolute units are arbitrary.
"""
from __future__ import annotations

import time

import numpy as np

C_CM_PER_FS = 2.99792458e-5          # speed of light in cm/fs


def ir_spectrum(mu, dt_fs, broaden_cm=0.0, max_cm=4000.0, min_cm=200.0):
    """mu: (n_traj, n_t, 3) or (n_t, 3) lab-frame dipoles at spacing dt_fs. Returns (wavenumbers cm-1, intensity
    normalised to a maximum of 1 between min_cm and max_cm). Below ~100 cm-1 a lab-frame spectrum holds the pure
    rotational band of the tumbling molecule (real, but it would dominate the normalisation)."""
    mu = np.asarray(mu, float)
    if mu.ndim == 2:
        mu = mu[None]
    d = np.gradient(mu, dt_fs, axis=1)
    d = d - d.mean(1, keepdims=True)
    n = d.shape[1]
    w = np.hanning(n)[None, :, None]
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    P = (np.abs(np.fft.rfft(d * w, n=nfft, axis=1)) ** 2).sum(2).mean(0)
    nu = np.fft.rfftfreq(nfft, dt_fs) / C_CM_PER_FS
    if broaden_cm and broaden_cm > 0:
        step = nu[1] - nu[0]
        s = broaden_cm / step
        k = np.arange(-int(4 * s) - 1, int(4 * s) + 2)
        g = np.exp(-0.5 * (k / s) ** 2)
        P = np.convolve(P, g / g.sum(), mode='same')
    m = nu <= max_cm
    nu, P = nu[m], P[m]
    ref = P[nu >= min_cm].max() if (nu >= min_cm).any() else P.max()
    return nu, P / ref if ref > 0 else P


def find_peaks(nu, I, min_rel=0.05, min_sep_cm=40.0, min_cm=200.0):
    """Local maxima above min_cm and above min_rel of the maximum there, at least min_sep_cm apart (strongest kept)."""
    top = I[nu >= min_cm].max() if (nu >= min_cm).any() else I.max()
    i = np.where((I[1:-1] > I[:-2]) & (I[1:-1] >= I[2:]) & (I[1:-1] > min_rel * top) & (nu[1:-1] >= min_cm))[0] + 1
    out = []
    for j in i[np.argsort(-I[i])]:
        if all(abs(nu[j] - nu[k]) >= min_sep_cm for k in out):
            out.append(j)
    return sorted((float(nu[j]), float(I[j])) for j in out)


def load_surface_model(path):
    """A model for dynamics: a workflow model (.pt) or a polynomial well (.npz, diabml.wellfit.PolyWell)."""
    if str(path).endswith('.npz'):
        from ..wellfit import PolyWell
        return PolyWell.load(path)
    from ..models import load_model
    return load_model(path)


def _normal_modes(atoms, delta=0.005):
    """Mass-weighted Hessian eigenpairs at the current geometry: (lambda in eV/(amu A^2), vectors (3N, 3N))."""
    import tempfile
    from ase.vibrations import Vibrations
    with tempfile.TemporaryDirectory() as tmp:
        vib = Vibrations(atoms, name=f'{tmp}/vib', delta=delta)
        vib.run()
        H = vib.get_vibrations().get_hessian_2d()
    m = np.repeat(atoms.get_masses(), 3)
    return np.linalg.eigh(H / np.sqrt(np.outer(m, m)))


def microcanonical_start(atoms, energy_eV, rng, rot_temperature=300.0, min_cm=50.0):
    """Classical microcanonical normal-mode sampling at the minimum: the vibrational energy energy_eV split over the
    normal modes uniformly on the simplex (the microcanonical distribution of harmonic modes), each with a random
    phase; plus a thermal rotation at rot_temperature. Sets positions and velocities of `atoms` (at a minimum)."""
    from ase import units
    lam, V = _normal_modes(atoms)
    m = np.repeat(atoms.get_masses(), 3)
    nu = 521.47 * np.sqrt(np.clip(lam, 0, None))
    vib = np.where(nu > min_cm)[0]
    e = rng.dirichlet(np.ones(len(vib))) * energy_eV
    dx, dv = np.zeros(len(m)), np.zeros(len(m))
    for k, ek in zip(vib, e):
        A = np.sqrt(2 * ek / lam[k])                     # amplitude in amu^1/2 A; lam = omega^2 in ASE units
        ph = rng.uniform(0, 2 * np.pi)
        dx += V[:, k] * A * np.cos(ph) / np.sqrt(m)
        dv += -V[:, k] * A * np.sqrt(lam[k]) * np.sin(ph) / np.sqrt(m)
    atoms.set_positions(atoms.get_positions() + dx.reshape(-1, 3))
    v = dv.reshape(-1, 3)
    if rot_temperature and rot_temperature > 0:
        kT = units.kB * rot_temperature
        r = atoms.get_positions() - atoms.get_center_of_mass()
        mm = atoms.get_masses()
        I = np.einsum('i,ij->', mm, r ** 2) * np.eye(3) - np.einsum('i,ij,ik->jk', mm, r, r)
        Lc = np.linalg.cholesky(kT * I + 1e-12 * np.eye(3)) @ rng.standard_normal(3)
        w = np.linalg.solve(I + 1e-12 * np.eye(3), Lc)
        v = v + np.cross(w, r)
    atoms.set_velocities(v)


def run_ir_trajectory(model_path, dipole_path, temperature=None, seed=0, equil_fs=1000.0, run_fs=5000.0, dt=0.2,
                      every=2, state=0, friction=0.02, energy_eV=None, rot_temperature=300.0, label=None,
                      restraint_scale=1.0):
    """One trajectory, then constant energy for run_fs, recording every `every` steps. Start: either Langevin at
    `temperature` for equil_fs (canonical: thermalises vibration and rotation), or (energy_eV given) classical
    microcanonical normal-mode sampling with that vibrational energy above the S0 minimum plus a thermal rotation at
    rot_temperature (e.g. a nascent molecule with a known excess energy). Returns time series: t (fs), the lab dipole
    (e*Angstrom), internal coordinates, total and potential energy (eV), the ensemble spread on `state` (eV)."""
    import torch
    from ase import units
    from ase.constraints import FixCom
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from ase.md.verlet import VelocityVerlet
    from ..dipole import DipoleModel, lab_dipole
    from ..models import load_model
    from . import CoordinateSurface, DiabmlCalculator, ch2oo_atoms, ch2oo_coordinates, reduced_restraints
    torch.set_num_threads(1)
    model = load_surface_model(model_path)
    if is_well9d(model):                         # the 9-D well: no restraints, the 9-D dipole surface
        return run_ir_trajectory9(model_path, dipole_path, temperature=temperature, seed=seed, equil_fs=equil_fs,
                                  run_fs=run_fs, dt=dt, every=every, friction=friction, energy_eV=energy_eV,
                                  rot_temperature=rot_temperature, label=label)
    dip = DipoleModel.load(dipole_path)
    names = model.meta.get('coord_names', ['r_OO', 'a_COO'])
    fixed = {k.lower(): v for k, v in model.meta.get('fixed_coordinates', {}).items()}
    atoms = ch2oo_atoms(**fixed)
    coord_fn = ch2oo_coordinates(names)
    surface = reduced_restraints(CoordinateSurface(model, coord_fn), atoms.positions, free=names,
                                 k_bond=30.0, k_angle=4.0 * restraint_scale,
                                 k_dihedral=1.0 * restraint_scale)
    atoms.calc = DiabmlCalculator(surface, state=state, couplings=False)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    e_min = None
    if energy_eV is not None:
        from ase.optimize import BFGS
        BFGS(atoms, logfile=None).run(fmax=1e-4, steps=500)
        e_min = atoms.get_potential_energy()
        microcanonical_start(atoms, float(energy_eV), rng, rot_temperature)
        atoms.set_constraint(FixCom())
    else:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)
        Stationary(atoms)
        # FixCom, not Langevin(fixcm=True): for a 5-atom molecule the latter heats far above the target temperature
        # (ASE 3.28+ deprecates it for that reason; tested here: 700-900 K for a 500 K target)
        atoms.set_constraint(FixCom())
        if equil_fs > 0:
            Langevin(atoms, dt * units.fs, temperature_K=temperature, friction=friction / units.fs, fixcm=False,
                     rng=rng).run(int(round(equil_fs / dt)))
    dyn = VelocityVerlet(atoms, timestep=dt * units.fs)
    rec = {k: [] for k in ('P', 'x', 'E_tot', 'E_pot', 'spread')}

    def record():
        res = atoms.calc.results
        P = atoms.get_positions()
        rec['P'].append(P.copy())
        rec['x'].append(coord_fn(torch.tensor(P)).numpy())
        ep = atoms.get_potential_energy()
        rec['E_pot'].append(ep)
        rec['E_tot'].append(ep + atoms.get_kinetic_energy())
        me = res.get('member_energies')
        rec['spread'].append(float(np.std(np.asarray(me)[:, state])) if me is not None else np.nan)

    dyn.attach(record, interval=int(every))
    dyn.run(int(round(run_fs / dt)))
    P = np.array(rec['P'])
    x = np.array(rec['x'])
    from ..vibdvr import model_dipole
    mu_b = model_dipole(dip, x, list(names), fixed)
    mu = lab_dipole(mu_b, P)
    return {'t': np.arange(len(P)) * dt * every, 'mu': mu, 'mu_body': mu_b, 'x': x,
            'E_tot': np.array(rec['E_tot']), 'E_pot': np.array(rec['E_pot']), 'spread': np.array(rec['spread']),
            'temperature': temperature, 'energy_eV': energy_eV, 'E_min': e_min, 'seed': seed,
            'label': label or (f'{temperature:.0f} K' if energy_eV is None else f'{energy_eV * 96.485:.0f} kJ/mol'),
            'dt_record': dt * every, 'seconds': time.time() - t0, 'names': names}


def _mark_springs(rows, names):
    """A restraint spring is a mode of mostly H motion that the model's own coordinates do not carry. With a_HCO
    (r_CH) in the model the mode with the largest a_HCO (r_CH) amplitude, and any within 30% of it, is the physical
    CH2 scissor (symmetric C-H stretch), not a spring, although it is H motion."""
    hq = [j for j, n in enumerate(names) if n in ('a_HCO', 'r_CH')]
    for j in hq:
        top = max((abs(r['dq'][j]) for r in rows), default=0.0)
        for r in rows:
            if top > 0 and abs(r['dq'][j]) > 0.3 * top:
                r['spring'] = False


def harmonic_ir(model_path, dipole_path, state=0, delta=0.01, restraint_scale=1.0):
    """Harmonic analysis at the S0 minimum of a reduced CH2OO model: frequencies (cm-1), relative IR intensities
    (|d mu/dQ|^2 from the dipole surface) and how much of each mode is H motion (the restraint springs of the CH2
    frame in a reduced model are not physical modes). Returns (rows, optimised internal coordinates)."""
    import torch
    from ase.optimize import BFGS
    from ase.vibrations import Vibrations
    from ..dipole import DipoleModel, lab_dipole
    from ..models import load_model
    from ..vibdvr import model_dipole
    from . import CoordinateSurface, DiabmlCalculator, ch2oo_atoms, ch2oo_coordinates, reduced_restraints
    import tempfile
    model = load_surface_model(model_path)
    dip = DipoleModel.load(dipole_path) if dipole_path else None
    names = model.meta.get('coord_names', ['r_OO', 'a_COO'])
    fixed = {k.lower(): v for k, v in model.meta.get('fixed_coordinates', {}).items()}
    atoms = ch2oo_atoms(**fixed)
    coord_fn = ch2oo_coordinates(names)
    surface = reduced_restraints(CoordinateSurface(model, coord_fn), atoms.positions, free=names,
                                 k_bond=30.0, k_angle=4.0 * restraint_scale,
                                 k_dihedral=1.0 * restraint_scale)
    atoms.calc = DiabmlCalculator(surface, state=state, couplings=False)
    BFGS(atoms, logfile=None).run(fmax=1e-4, steps=500)
    x0 = coord_fn(torch.tensor(atoms.get_positions())).numpy()
    with tempfile.TemporaryDirectory() as tmp:
        vib = Vibrations(atoms, name=f'{tmp}/vib', delta=0.005)
        vib.run()
        H = vib.get_vibrations().get_hessian_2d()                 # eV / Angstrom^2
    m = np.repeat(atoms.get_masses(), 3)
    lam, V = np.linalg.eigh(H / np.sqrt(np.outer(m, m)))
    P0 = atoms.get_positions()
    rows = []
    for k in range(len(lam)):
        if lam[k] <= 0:
            continue
        nu = 521.47 * np.sqrt(lam[k])                              # sqrt(eV / (A^2 amu)) -> cm-1
        if nu < 50:                                                # translations and rotations
            continue
        L = (V[:, k] / np.sqrt(m)).reshape(-1, 3)                  # Cartesian displacement per unit normal coordinate
        amp = (V[:, k] ** 2).reshape(-1, 3).sum(1)                 # mass-weighted: fraction of the mode per atom
        dq = (coord_fn(torch.tensor(P0 + delta * L)).numpy() - coord_fn(torch.tensor(P0 - delta * L)).numpy()) / (2 * delta)
        # a restraint spring: mostly H motion that the model's own coordinates do not carry (with a_HCO / r_CH in
        # the model the CH2 scissor / symmetric stretch are physical modes although they are H motion)
        spring = bool(amp[3:].sum() > 0.5)                      # refined below once all modes are known
        if dip is None:
            rows.append({'cm-1': float(nu), 'intensity': float('nan'), 'H_fraction': float(amp[3:].sum()),
                         'dq': dq.tolist(), 'spring': spring})
            continue
        dmu = []
        for s in (+1, -1):
            P = P0 + s * delta * L
            xb = coord_fn(torch.tensor(P)).numpy()[None]
            dmu.append(lab_dipole(model_dipole(dip, xb, list(names), fixed), P[None])[0])
        d = (dmu[0] - dmu[1]) / (2 * delta)
        rows.append({'cm-1': float(nu), 'intensity': float(d @ d), 'H_fraction': float(amp[3:].sum()), 'dq': dq.tolist(),
                     'spring': spring})
    _mark_springs(rows, names)
    if dip is None:
        return rows, x0
    imax = max((r['intensity'] for r in rows), default=1.0) or 1.0
    for r in rows:
        r['intensity'] /= imax
    return rows, x0


# ------------------------------------------------------------------------------------------ nine coordinates
def is_well9d(model):
    from ..geometry import NINE
    return list(getattr(model, 'names', [])) == list(NINE)


class Well9Calculator:
    """ASE calculator for the 9-D polynomial well (diabml.workflow.well9d): energy and exact forces of all five
    atoms by automatic differentiation through the internal coordinates. No restraints: every mode is physical."""

    def __new__(cls, model):
        from ase.calculators.calculator import Calculator, all_changes
        import torch
        from ..workflow.well9d import internal9_t, poly_t

        class _Calc(Calculator):
            implemented_properties = ['energy', 'free_energy', 'forces']

            def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                R = torch.tensor(self.atoms.get_positions(), dtype=torch.float64, requires_grad=True)
                E = poly_t(model, internal9_t(R[None]))[0]
                E.backward()
                e = float(E.detach())
                self.results = {'energy': e, 'free_energy': e, 'forces': -R.grad.numpy().copy()}
        return _Calc()


def _atoms9(model):
    from ase import Atoms
    from ..geometry import ch2oo_positions9
    from ..workflow.well9d import refine_minimum
    from .ch2oo import CH2OO_SYMBOLS
    xm, _, _ = refine_minimum(model, model.center.numpy() if hasattr(model.center, 'numpy') else model.center)
    atoms = Atoms(CH2OO_SYMBOLS, positions=ch2oo_positions9(xm[None])[0])
    atoms.calc = Well9Calculator(model)
    return atoms, xm


def run_ir_trajectory9(model_path, dipole_path, temperature=None, seed=0, equil_fs=1000.0, run_fs=5000.0, dt=0.2,
                       every=2, friction=0.02, energy_eV=None, rot_temperature=300.0, label=None, **_):
    """run_ir_trajectory for the 9-D well: all nine modes free, the lab dipole from the 9-D dipole surface
    (diabml/dipole9.py; it rotates with the molecule, so no body frame is needed). Same starts (Langevin at
    `temperature`, or microcanonical normal-mode sampling with energy_eV above the minimum plus a thermal rotation)
    and the same returned time series."""
    import torch
    from ase import units
    from ase.constraints import FixCom
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from ase.md.verlet import VelocityVerlet
    from ..dipole9 import VectorDipole9
    from ..geometry import NINE, ch2oo_internal9
    torch.set_num_threads(1)
    model = load_surface_model(model_path)
    dip = VectorDipole9.load(dipole_path)
    atoms, _ = _atoms9(model)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    e_min = None
    if energy_eV is not None:
        e_min = atoms.get_potential_energy()
        microcanonical_start(atoms, float(energy_eV), rng, rot_temperature)
        atoms.set_constraint(FixCom())
    else:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)
        Stationary(atoms)
        atoms.set_constraint(FixCom())
        if equil_fs > 0:
            Langevin(atoms, dt * units.fs, temperature_K=temperature, friction=friction / units.fs, fixcm=False,
                     rng=rng).run(int(round(equil_fs / dt)))
    dyn = VelocityVerlet(atoms, timestep=dt * units.fs)
    rec = {k: [] for k in ('P', 'E_tot', 'E_pot')}

    def record():
        rec['P'].append(atoms.get_positions().copy())
        ep = atoms.get_potential_energy()
        rec['E_pot'].append(ep)
        rec['E_tot'].append(ep + atoms.get_kinetic_energy())

    dyn.attach(record, interval=int(every))
    dyn.run(int(round(run_fs / dt)))
    P = np.array(rec['P'])
    x = ch2oo_internal9(P)
    mu = dip.predict(P)
    return {'t': np.arange(len(P)) * dt * every, 'mu': mu, 'mu_body': mu, 'x': x,
            'E_tot': np.array(rec['E_tot']), 'E_pot': np.array(rec['E_pot']), 'spread': np.full(len(P), np.nan),
            'temperature': temperature, 'energy_eV': energy_eV, 'E_min': e_min, 'seed': seed,
            'label': label or (f'{temperature:.0f} K' if energy_eV is None else f'{energy_eV * 96.485:.0f} kJ/mol'),
            'dt_record': dt * every, 'seconds': time.time() - t0, 'names': list(NINE)}


def harmonic_ir9(model_path, dipole_path, delta=0.01):
    """Harmonic modes of the 9-D well at its minimum (exact Hessian, Eckart-projected) with relative intensities
    |d mu / dQ|^2 from the 9-D dipole surface. Rows as harmonic_ir (no restraint springs: every mode is physical);
    dq are the internal-coordinate displacements per unit normal coordinate. Returns (rows, minimum)."""
    from ..dipole9 import VectorDipole9
    from ..geometry import ch2oo_internal9, ch2oo_positions9
    from ..vibdvr import ISOTOPIC_MASS
    from ..vpt2 import cartesian_modes
    from ..workflow.well9d import cartesian_hessian_fn, refine_minimum
    model = load_surface_model(model_path)
    dip = VectorDipole9.load(dipole_path) if dipole_path else None
    xm, _, _ = refine_minimum(model, model.center.numpy() if hasattr(model.center, 'numpy') else model.center)
    P0 = ch2oo_positions9(xm[None])[0]
    modes = cartesian_modes(cartesian_hessian_fn(model)(P0), P0, ISOTOPIC_MASS)
    rows = []
    for k, nu in enumerate(modes.omega):
        Q = np.zeros(len(modes.omega))
        Q[k] = delta
        L = modes.cart(Q)
        amp = (modes.L[:, k] ** 2).reshape(-1, 3).sum(1)
        dq = (ch2oo_internal9((P0 + L)[None]) - ch2oo_internal9((P0 - L)[None]))[0] / (2 * delta)
        inten = float('nan')
        if dip is not None:
            dm = (dip.predict(P0 + L) - dip.predict(P0 - L)) / (2 * delta)
            inten = float(dm @ dm)
        rows.append({'cm-1': float(nu), 'intensity': inten, 'H_fraction': float(amp[3:].sum()), 'dq': dq.tolist(),
                     'spring': False})
    if dip is not None:
        top = max(r['intensity'] for r in rows) or 1.0
        for r in rows:
            r['intensity'] /= top
    return rows, xm
