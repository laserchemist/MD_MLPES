"""Quantum vibrational levels of a reduced CH2OO model: a grid (sinc-DVR) solution of the J = 0 vibrational
Schroedinger equation in the model's internal coordinates (r_OO, angle COO, r_CO), CH2 frame rigid.

Why: classical dynamics at 300 K samples only the bottom of the well and gives nearly harmonic band positions; the
real molecule carries zero-point energy, which puts it on the anharmonic part of the surface. Solving for the levels
on the same surface gives the quantum fundamentals (and overtones and combinations), directly comparable with
experiment and with the anharmonic (VPT2) values of Su et al., Science 340, 174 (2013).

Hamiltonian (Wilson G-matrix form, J = 0):

    H = -(hbar^2 / 2) sum_ij d_i G_ij(q) d_j + V(q)

G is the vibrational block of the inverse of the full metric g = J^T M J, where the columns of J are the Cartesian
derivatives of the atom positions with respect to the three internal coordinates, the three translations and the three
infinitesimal rotations (so translation and rotation are removed exactly at J = 0). The atom positions come from the
same builder as the dipole surface (diabml.dipole.ch2oo_positions), so the CH2 group moves rigidly with C along C-O.
The small extrapotential term of the exact (Podolsky) form is left out: it is nearly constant over the well and
shifts the fundamentals by at most a few cm-1.

Discretisation: a uniform grid per coordinate with the sinc first-derivative matrix D (Colbert & Miller); the kinetic
operator is sum_ij D_i^T diag(G_ij) D_j, symmetric and position dependent. The lowest states come from an iterative
eigensolver on the operator (no matrix is stored). Assignment: overlap with the states of the harmonic problem (the
quadratic expansion of V with G fixed at the minimum), whose quantum numbers are known. Intensities: transition
dipoles <0|mu|v> from the dipole surface in the body frame, times the transition wavenumber.

    from diabml.vibdvr import vib_levels
    out = vib_levels(PolyWell.load('models/well_poly.npz'), DipoleModel.load('models/dipole.pt'), n=32)
"""
from __future__ import annotations

import itertools

import numpy as np

HBAR2 = 4.180159e-3          # hbar^2 in eV * amu * Angstrom^2
EV_CM = 8065.54393734921     # 1 eV in cm-1
_MASS = np.array([12.011, 15.999, 15.999, 1.008, 1.008])     # C, O1, O2, H1, H2 (ASE defaults, as in the dynamics)
_ANG = ('a_', 'd_')          # coordinates in degrees (angles a_*, dihedrals d_*; converted to radians on the grid)


def _is_angle(name):
    return name.startswith(_ANG)


def _positions(Xdeg, names, fixed):
    """Atom positions (N, 5, 3). Reduced models (r_OO, a_COO, r_CO, r_CH, a_HCO) use the dipole surface's builder;
    names from the nine-coordinate set (r_CH1, a_HCO1, d_H1, ...; diabml.geometry.NINE) use ch2oo_positions9."""
    from .geometry import NINE, ch2oo_positions9
    if any(n in NINE and n not in ('r_OO', 'a_COO', 'r_CO') for n in names):
        return ch2oo_positions9(np.asarray(Xdeg, float), names, fixed)
    from .dipole import ch2oo_positions
    return ch2oo_positions(np.asarray(Xdeg, float), names, fixed).numpy()


def g_matrix(Xdeg, names, fixed=None, masses=_MASS, h=1e-4):
    """(N, n, n) vibrational G matrix (inverse kinetic metric, J = 0) at internal coordinates X (degrees for angles);
    units 1/(amu A^2) for lengths, 1/(amu rad^2) for angles."""
    X = np.atleast_2d(np.asarray(Xdeg, float))
    n = X.shape[1]
    unit = np.array([np.pi / 180 if _is_angle(c) else 1.0 for c in names])
    P = _positions(X, names, fixed)                                          # (N, 5, 3)
    cols = []
    for j in range(n):
        dX = np.zeros(n)
        dX[j] = h
        cols.append((_positions(X + dX, names, fixed) - _positions(X - dX, names, fixed)) / (2 * h * unit[j]))
    m = np.asarray(masses, float)
    R = P - (m[None, :, None] * P).sum(1, keepdims=True) / m.sum()
    for a in range(3):                                                        # translations
        e = np.zeros(3)
        e[a] = 1
        cols.append(np.broadcast_to(e, P.shape))
    for a in range(3):                                                        # infinitesimal rotations
        e = np.zeros(3)
        e[a] = 1
        cols.append(np.cross(e, R))
    J = np.stack([c.reshape(len(X), -1) for c in cols], 2)                   # (N, 15, n + 6)
    w = np.repeat(m, 3)
    g = np.einsum('nki,k,nkj->nij', J, w, J)
    return np.linalg.inv(g)[:, :n, :n]


def _sinc_d1(n, dx):
    k = np.arange(n)
    d = k[:, None] - k[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        D = np.where(d == 0, 0.0, (-1.0) ** d / (d * dx))
    return D


def _apply(D, psi, axis):
    return np.moveaxis(np.tensordot(D, psi, axes=([1], [axis])), 0, axis)


class GridHamiltonian:
    """H on a product grid: V (n1, n2, n3) in eV, G (n1, n2, n3, 3, 3), grid spacings in internal units (A, rad)."""

    def __init__(self, axes, V, G):
        self.axes = [np.asarray(a, float) for a in axes]
        self.shape = tuple(len(a) for a in self.axes)
        self.D = [_sinc_d1(len(a), a[1] - a[0]) for a in self.axes]
        self.V = np.asarray(V, float)
        self.G = np.asarray(G, float)

    def matvec(self, x):
        psi = x.reshape(self.shape)
        d = [_apply(Di, psi, i) for i, Di in enumerate(self.D)]
        out = self.V * psi
        for i, Di in enumerate(self.D):
            w = sum(self.G[..., i, j] * d[j] for j in range(len(d)))
            out = out + 0.5 * HBAR2 * _apply(Di.T, w, i)
        return out.ravel()

    def eigh(self, k=20, tol=1e-10):
        from scipy.sparse.linalg import LinearOperator, eigsh
        N = int(np.prod(self.shape))
        op = LinearOperator((N, N), matvec=self.matvec, dtype=float)
        v0 = np.exp(-(self.V - self.V.min()) / 0.05).ravel()
        E, U = eigsh(op, k=k, which='SA', tol=tol, v0=v0, ncv=max(2 * k + 1, 60))
        o = np.argsort(E)
        return E[o], U[:, o]


def _harmonic(Vfun, x0deg, names, fixed, h=None):
    """Hessian (eV per internal unit^2, radians for angles), G at x0, normal-mode wavenumbers and the transform."""
    n = len(names)
    unit = np.array([np.pi / 180 if _is_angle(c) else 1.0 for c in names])
    h = h if h is not None else np.array([0.5 if _is_angle(c) else 0.005 for c in names])     # degrees / A
    K = np.zeros((n, n))
    f0 = Vfun(x0deg[None])[0]
    for i in range(n):
        for j in range(n):
            pts = []
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                x = x0deg.copy()
                x[i] += si * h[i]
                x[j] += sj * h[j]
                pts.append(x)
            f = Vfun(np.array(pts))
            K[i, j] = (f[0] - f[1] - f[2] + f[3]) / (4 * h[i] * h[j] * unit[i] * unit[j])
    G0 = g_matrix(x0deg[None], names, fixed)[0]
    lam, L = np.linalg.eig(G0 @ K)
    o = np.argsort(lam.real)
    lam, L = lam.real[o], L.real[:, o]
    nu = np.sqrt(np.maximum(lam, 0) * HBAR2) * EV_CM                         # hbar * omega in cm-1
    return K, G0, nu, L, f0


def minimise(Vfun, x0deg, box=None):
    from scipy.optimize import minimize
    bounds = None if box is None else list(box)
    r = minimize(lambda x: float(Vfun(x[None])[0]), np.asarray(x0deg, float), method='L-BFGS-B', bounds=bounds,
                 options={'ftol': 1e-14, 'gtol': 1e-10})
    return r.x


def _quanta(nu, nmax=6, emax=None):
    combos = []
    for q in itertools.product(range(nmax + 1), repeat=len(nu)):
        e = float(np.dot(q, nu))
        if emax is None or e <= emax:
            combos.append((e, q))
    return sorted(combos)


def vib_levels(surface, dipole=None, n=28, n_states=16, ecut_eV=1.2, names=None, fixed=None, x0=None,
               ranges=None, log=print):
    """Vibrational levels of a one-state surface (anything with predict(X) -> (N, >=1) energies in eV, X with
    angles in degrees, e.g. a PolyWell). Grid: n points per coordinate over `ranges` ({name: (lo, hi)}) or, by
    default, the classically allowed region of the harmonic potential up to ecut_eV above the minimum. Returns a dict
    with the levels (cm-1 above the ground state), assignments, intensities and diagnostics."""
    names = list(names or surface.meta.get('coord_names'))
    fixed = {k.lower(): v for k, v in (fixed if fixed is not None
                                       else surface.meta.get('fixed_coordinates', {})).items()}
    Vfun = lambda X: np.asarray(surface.predict(np.asarray(X, float)))[:, 0]
    box = getattr(surface, 'box', None)
    bx = [tuple(box[c]) for c in names] if box else None
    x0 = minimise(Vfun, np.asarray(x0 if x0 is not None else surface.center.numpy(), float), bx)
    K, G0, nu_h, L, V0 = _harmonic(Vfun, x0, names, fixed)
    unit = np.array([np.pi / 180 if _is_angle(c) else 1.0 for c in names])
    # grid: the harmonic classically allowed region at ecut_eV (Delta q_i = sqrt(2 E (K^-1)_ii))
    span = np.sqrt(2 * ecut_eV * np.diag(np.linalg.inv(K))) / unit                 # internal units (deg, A)
    lims = []
    for j, c in enumerate(names):
        lo, hi = (ranges or {}).get(c, (x0[j] - span[j], x0[j] + span[j]))
        lims.append((float(lo), float(hi)))
    nn = [n] * len(names) if np.isscalar(n) else list(n)
    clipped = False
    while True:
        axes_deg = [np.linspace(lo, hi, k) for (lo, hi), k in zip(lims, nn)]
        mesh = np.stack(np.meshgrid(*axes_deg, indexing='ij'), -1)
        X = mesh.reshape(-1, len(names))
        V = Vfun(X).reshape(mesh.shape[:-1]) - V0
        vmin = float(V.min())
        if vmin >= -1e-3 or clipped or not box:
            break
        # a polynomial diverges outside the data it was fitted to: keep the grid inside the fitted region
        new = [(max(lo, box[c][0]), min(hi, box[c][1])) for (lo, hi), c in zip(lims, names)]
        log(f'the surface dips {vmin * 1000:.0f} meV below its minimum outside the fitted data: grid narrowed to the '
            f'data box ' + ', '.join(f'{c} {a:.3f}-{b:.3f}' for c, (a, b) in zip(names, new)))
        lims, clipped = new, True
    if vmin < -1e-3:
        log(f'warning: the surface dips {vmin * 1000:.1f} meV below its minimum on the grid (extrapolation outside '
            f'the fitted region); levels are unreliable - fit a lower order or narrow the grid')
    G = g_matrix(X, names, fixed).reshape(*mesh.shape[:-1], len(names), len(names))
    outside = None
    if box:
        ok = np.ones(len(X), bool)
        for j, c in enumerate(names):
            ok &= (X[:, j] >= box[c][0] - 1e-9) & (X[:, j] <= box[c][1] + 1e-9)
        outside = (~ok).reshape(mesh.shape[:-1])
    axes = [a * u for a, u in zip(axes_deg, unit)]
    Hq = GridHamiltonian(axes, V, G)
    E, U = Hq.eigh(k=n_states)
    # harmonic reference on the same grid: quadratic V, constant G -> known quantum numbers
    dq = (mesh - x0) * unit
    Vh = 0.5 * np.einsum('...i,ij,...j->...', dq, K, dq)
    Hh = GridHamiltonian(axes, Vh, np.broadcast_to(G0, G.shape))
    Eh, Uh = Hh.eigh(k=n_states)
    combos = _quanta(nu_h, emax=(Eh[-1] - Eh[0]) * EV_CM + 100)
    zpe_h = 0.5 * nu_h.sum()
    lab_h = []
    for e in (Eh - Eh[0]) * EV_CM:
        d = [abs(c[0] - e) for c in combos]
        k = int(np.argmin(d))
        close = sorted(d)[1] < 2.0 if len(d) > 1 else False
        lab_h.append((combos[k][1], float(d[k]), close))
    harm_err = max(l[1] for l in lab_h)
    S = Uh.T @ U                                                                  # overlaps harmonic x anharmonic
    w = np.abs(Uh) ** 2
    # edge density per state: fraction of |psi|^2 on the outer two grid layers
    edge = []
    for k in range(U.shape[1]):
        p = (U[:, k] ** 2).reshape(Hq.shape)
        inner = p[tuple(slice(2, -2) for _ in Hq.shape)].sum()
        edge.append(float(1 - inner / p.sum()))
    mu = None
    if dipole is not None:
        mu = model_dipole(dipole, X, names, fixed)                                # (N, 3) e*A
    states = []
    for k in range(len(E)):
        j = int(np.argmax(np.abs(S[:, k])))
        q, err, close = lab_h[j]
        row = {'n': k, 'cm-1': float((E[k] - E[0]) * EV_CM), 'quanta': list(map(int, q)),
               'overlap': float(abs(S[j, k])), 'edge_density': edge[k],
               'harmonic_cm-1': float(np.dot(q, nu_h))}
        if mu is not None:
            t = (U[:, 0] * U[:, k]) @ mu
            row['transition_dipole_eA'] = t.tolist()
            row['intensity'] = float(row['cm-1'] * (t @ t)) if k else 0.0
        if outside is not None:
            row['outside_box_density'] = float((U[:, k] ** 2 * outside.ravel()).sum())
        states.append(row)
    if mu is not None:
        imax = max(s['intensity'] for s in states[1:]) or 1.0
        for s in states:
            s['intensity'] /= imax
    labels = mode_labels(K, L, names)
    fund = {}
    for i in range(len(names)):
        q = [0] * len(names)
        q[i] = 1
        c = [s for s in states if s['quanta'] == q]
        if c:
            best = max(c, key=lambda s: s['overlap'])
            fund[labels[i]] = best['cm-1']
    for s in states:
        s['label'] = ' + '.join((f'{v}' if v > 1 else '') + labels[i] for i, v in enumerate(s['quanta']) if v) or 'ground'
    rot = rotational_constants(X, names, fixed, U, x0, n_keep=min(len(states), 12))
    cor = coriolis(x0, names, fixed, L, nu_h)
    for s, abc in zip(states, rot['states']):
        s['ABC_cm-1'] = abc
        # Coriolis part of B_v = B_e - sum_r alpha_r (v_r + 1/2), added to the averaged (inertial) constants
        dv = -np.sum(np.asarray(cor['alpha_cm-1']) * (np.asarray(s['quanta'])[:, None] + 0.5), 0)
        s['ABC_coriolis_shift_cm-1'] = dv.tolist()
        s['ABC_with_coriolis_cm-1'] = (np.asarray(abc) + dv).tolist()
    return {'rotational_cm-1': {'equilibrium': rot['equilibrium'], 'masses': rot['masses'], 'coriolis': cor},
            'names': names, 'fixed': fixed, 'minimum': x0.tolist(), 'grid': {'n': nn, 'ranges': lims},
            'zpe_cm-1': float((E[0] - 0.0) * EV_CM), 'zpe_harmonic_cm-1': float(zpe_h),
            'harmonic_cm-1': nu_h.tolist(), 'harmonic_labels': labels, 'harmonic_modes': L.tolist(), 'harmonic_grid_error_cm-1': float(harm_err),
            'G0': G0.tolist(), 'hessian': K.tolist(), 'surface_min_on_grid_eV': vmin,
            'grid_clipped_to_data': clipped,
            'reliable': bool(vmin >= -1e-3 and max(edge[:len(names) + 1]) < 1e-2),
            'grid_outside_box_fraction': None if outside is None else float(outside.mean()),
            'states': states, 'fundamentals_cm-1': fund}


ISOTOPIC_MASS = np.array([12.0, 15.99491462, 15.99491462, 1.00782503, 1.00782503])   # 12C 16O 16O 1H 1H
B_CONV = 16.857629206                     # B (cm-1) = B_CONV / I (amu A^2)


def inertia_abc(P, masses=ISOTOPIC_MASS):
    """(N, 3) rotational constants A >= B >= C (cm-1) of geometries P (N, 5, 3)."""
    m = np.asarray(masses, float)
    R = P - (m[None, :, None] * P).sum(1, keepdims=True) / m.sum()
    I = np.einsum('a,nai,naj->nij', m, R, R)
    I = np.einsum('n,ij->nij', np.einsum('nii->n', I), np.eye(3)) - I
    ev = np.linalg.eigvalsh(I)                                                  # ascending: I_a <= I_b <= I_c
    return B_CONV / ev


def rotational_constants(X, names, fixed, U, x0, n_keep=12, masses=ISOTOPIC_MASS, chunk=200000):
    """Rotational constants (cm-1, main isotopologue) at the minimum and averaged over the vibrational states:
    <v| hbar^2 / (2 I_g) |v> for g = a, b, c with the principal moments of each grid geometry. This is the
    'inertial' part of the vibration-rotation interaction (anharmonicity and the change of the moments along the
    vibrations); the Coriolis part is not included, so compare differences between states (ratios B'/B'') with care."""
    from .dipole import ch2oo_positions
    fx = {k: v for k, v in (fixed or {}).items() if k not in names}
    abc = []
    for a in range(0, len(X), chunk):
        abc.append(inertia_abc(ch2oo_positions(np.asarray(X[a:a + chunk], float), names, fx).numpy(), masses))
    abc = np.concatenate(abc)
    w = U[:, :n_keep] ** 2
    states = (w.T @ abc).tolist()
    eq = inertia_abc(ch2oo_positions(np.asarray(x0, float)[None], names, fx).numpy(), masses)[0].tolist()
    return {'equilibrium': eq, 'states': states, 'masses': list(map(float, masses))}


def coriolis(x0deg, names, fixed, L, nu_h, masses=ISOTOPIC_MASS, h=1e-4):
    """Coriolis zeta constants of the model's normal modes and their contribution to the vibration-rotation
    constants (second-order perturbation theory, Mills 1972):

        alpha_r^g(Coriolis) = -(2 B_g^2 / omega_r) sum_s (zeta_rs^g)^2 (3 omega_r^2 + omega_s^2) / (omega_r^2 - omega_s^2)

    with B_g the equilibrium constants (g = a, b, c) and omega the harmonic wavenumbers of the model. The normal
    modes are made Cartesian (mass-weighted, Eckart: translation and rotation projected out) from the internal ones;
    zeta^g_rs = sum_atoms (l_r x l_s)_g in the principal-axis frame. For a planar model only in-plane modes exist, so
    only zeta^c (about the out-of-plane axis) is non-zero: the Coriolis terms of A and B need the out-of-plane modes
    and coupling partners outside the model (the CH2 rock, the C-H stretches) are missing too."""
    from .dipole import ch2oo_positions
    fx = {k: v for k, v in (fixed or {}).items() if k not in names}
    n = len(names)
    x0 = np.asarray(x0deg, float)
    unit = np.array([np.pi / 180 if _is_angle(c) else 1.0 for c in names])
    pos = lambda x: ch2oo_positions(np.asarray(x, float)[None], names, fx).numpy()[0]
    P0 = pos(x0)
    m = np.asarray(masses, float)
    com = (m[:, None] * P0).sum(0) / m.sum()
    R = P0 - com
    I = np.einsum('a,ai,aj->ij', m, R, R)
    I = np.trace(I) * np.eye(3) - I
    ev, V = np.linalg.eigh(I)                                     # columns: a, b, c (I_a <= I_b <= I_c)
    Be = B_CONV / ev
    J = []
    for j in range(n):
        d = np.zeros(n)
        d[j] = h
        J.append(((pos(x0 + d) - pos(x0 - d)) / (2 * h * unit[j])).ravel())
    J = np.array(J).T                                             # (15, n): Cartesian per internal unit
    sq = np.repeat(np.sqrt(m), 3)
    ext = []
    for a in range(3):
        e = np.zeros(3)
        e[a] = 1
        ext.append((np.broadcast_to(e, R.shape) * np.sqrt(m)[:, None]).ravel())
        ext.append((np.cross(e, R) * np.sqrt(m)[:, None]).ravel())
    Q, _ = np.linalg.qr(np.array(ext).T)
    proj = np.eye(len(sq)) - Q @ Q.T
    lm = []
    for k in range(n):
        v = proj @ (sq * (J @ L[:, k]))
        lm.append((v / np.linalg.norm(v)).reshape(-1, 3) @ V)     # mass-weighted, principal-axis frame
    lm = np.array(lm)                                             # (n, 5, 3)
    zeta = np.zeros((3, n, n))
    for g, (b, c) in enumerate(((1, 2), (2, 0), (0, 1))):
        zeta[g] = np.einsum('ri,si->rs', lm[:, :, b], lm[:, :, c]) - np.einsum('ri,si->rs', lm[:, :, c], lm[:, :, b])
    w = np.asarray(nu_h, float)
    alpha = np.zeros((n, 3))
    for r in range(n):
        for g in range(3):
            t = 0.0
            for s_ in range(n):
                if s_ != r and abs(w[r] - w[s_]) > 1e-6:
                    t += zeta[g, r, s_] ** 2 * (3 * w[r] ** 2 + w[s_] ** 2) / (w[r] ** 2 - w[s_] ** 2)
            alpha[r, g] = -2 * Be[g] ** 2 / w[r] * t
    return {'zeta': zeta.tolist(), 'alpha_cm-1': alpha.tolist(), 'Be_cm-1': Be.tolist(), 'omega_cm-1': w.tolist(),
            'axes': 'a, b, c (principal, I_a <= I_b <= I_c)'}


_DESC = {'r_OO': 'O-O', 'a_COO': 'COO', 'r_CO': 'C-O', 'a_HCO': 'HCO', 'r_CH': 'C-H',
         'r_CH1': 'C-H (syn)', 'r_CH2': 'C-H (anti)', 'a_HCO1': 'HCO (syn)', 'a_HCO2': 'HCO (anti)',
         'd_H1': 'H out of plane (syn)', 'd_H2': 'H out of plane (anti)'}


def model_dipole(dipole, X, names, fixed=None, chunk=200000):
    """Body-frame dipole (N, 3) at internal coordinates X (columns `names`). If the dipole surface has fewer
    coordinates than the energy surface (e.g. a 3-D charge model with a 4-D surface that adds angle HCO), its charges
    are evaluated on its own coordinates and placed on the atoms where all of `names` put them: the extra motion
    carries its fixed-charge dipole change."""
    import torch
    from .dipole import ch2oo_positions
    if list(dipole.names) == list(names):
        return dipole.body_dipole(X)
    idx = [names.index(n) for n in dipole.names]
    fx = dict(dipole.fixed or {}, **{k: v for k, v in (fixed or {}).items()})
    out = []
    with torch.no_grad():
        for a in range(0, len(X), chunk):
            Xa = np.asarray(X[a:a + chunk], float)
            q = dipole.charges(Xa[:, idx]).mean(0)                               # (N, 5)
            R = ch2oo_positions(Xa, names, {k: v for k, v in fx.items() if k not in names})
            out.append(torch.einsum('na,nax->nx', q, R).numpy())
    return np.concatenate(out)


def mode_labels(K, L, names):
    """Name each normal mode (columns of L) by the internal coordinate with the largest share of its potential
    energy (potential energy distribution)."""
    out = []
    for k in range(L.shape[1]):
        ped = L[:, k] * (K @ L[:, k])
        out.append(_DESC.get(names[int(np.argmax(ped))], names[int(np.argmax(ped))]))
    return out


def harmonic_frequencies(surface, names=None, fixed=None):
    """Harmonic wavenumbers and labels at the minimum of a one-state surface (the vib_levels sensitivity check)."""
    names = list(names or surface.meta.get('coord_names'))
    fixed = {k.lower(): v for k, v in (fixed if fixed is not None
                                       else surface.meta.get('fixed_coordinates', {})).items()}
    Vfun = lambda X: np.asarray(surface.predict(np.asarray(X, float)))[:, 0]
    box = getattr(surface, 'box', None)
    x0 = minimise(Vfun, surface.center.numpy(), [tuple(box[c]) for c in names] if box else None)
    K, G0, nu, L, _ = _harmonic(Vfun, x0, names, fixed)
    return dict(zip(mode_labels(K, L, names), nu.tolist())), x0
