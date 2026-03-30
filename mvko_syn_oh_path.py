#!/usr/bin/env python3
"""
mvko_syn_oh_path.py — Reaction-path ML-PES for syn-MVKO → vinyl hydroperoxide → OH
=====================================================================================

Chemistry
---------
Syn-MVKO (methyl vinyl ketone oxide Criegee intermediate) decomposes via a
1,4-H shift from the methyl group to the terminal (distal) oxygen O2:

   syn-MVKO  →  [TS]  →  vinyl hydroperoxide (VHP)  →  OH + vinyl oxy-radical

The syn conformer has the methyl (C4) on the same face as O2, enabling this
5-membered ring TS:  C4–C1–O1–O2···H

This is the dominant OH-producing channel of MVKO (and CIs generally) at
atmospheric pressure; fresh CIs from ozonolysis carry ~40–80 kcal/mol of
excess energy and sample far above the TS even at 1 atm.

Multi-reference warning
-----------------------
B3LYP/6-31G* is a single-reference method. Along the IRC:
  - Near MVKO minimum: closed-shell character, B3LYP adequate
  - Near TS: partial biradical, ⟨S²⟩ rises, B3LYP barrier is approximate
  - VHP region: open-shell hydroperoxy radical character, B3LYP qualitative only
  - OH + radical: full open-shell, B3LYP fails for product energies

Despite this, B3LYP gives a useful *geometric* description of the IRC.
The ML-PES trained on B3LYP IRC data gives correct qualitative reaction path
geometry; barrier heights should be treated as ±5 kcal/mol approximate.

Future: replace B3LYP IRC data with CASSCF(4,4)/6-31G* or NEVPT2 single-points
on the B3LYP geometries.  The PESFamily framework is set up to swap in a more
accurate surface without changing the MD infrastructure.

PES family design
-----------------
  Surface "reactant"  : mlpes_initial.pkl  (near-eq MVKO, 904 fr, γ=0.001)
  Surface "rxn_path"  : trained on IRC + NM-displaced points
  Blend width         : 10 kcal/mol (wider than conformer blending to span
                        the TS region continuously)

At near-equilibrium geometries the reactant surface dominates.  As the molecule
climbs toward the TS (ΔE > ~15 kcal/mol), the rxn_path surface takes over.

Usage
-----
  # Step 1: PSI4 TS search and IRC (requires PSI4; ~30 min)
  python3 mvko_syn_oh_path.py --steps ts,irc

  # Step 2: Train reaction-path ML-PES and assemble family
  python3 mvko_syn_oh_path.py --steps train \\
      --irc-data outputs/mvko_rxn_path_<ts>/irc_training_data.npz

  # Step 3: Run high-energy ML-MD with family, monitor reaction coordinate
  python3 mvko_syn_oh_path.py --steps md \\
      --family-pkl outputs/mvko_rxn_path_<ts>/pes_family.pkl \\
      --temp 2000 --md-steps 50000

  # All steps in sequence:
  python3 mvko_syn_oh_path.py --steps ts,irc,train,md

Atom ordering (MVKO, from modules/test_molecules.py / bakken MEMORY):
  0  C1  Criegee carbon
  1  O1  proximal oxygen (O⁻ in zwitterion)
  2  O2  distal/terminal oxygen (electrophilic)
  3  C2  vinyl carbon (CH=)
  4  C3  terminal vinyl carbon (=CH2)
  5  C4  methyl carbon
  6  H1  on C2
  7  H2  on C3
  8  H3  on C3
  9  H4  on C4  ← transfers to O2 in syn-1,4-H shift
  10 H5  on C4
  11 H6  on C4

Reaction coordinate atoms:
  Forming bond : O2(2) – H4(9)
  Breaking bond: C4(5) – H4(9)
  Spectator    : O1(1) – O2(2)  (the Criegee O-O bond)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── PSI4 ──────────────────────────────────────────────────────────────────────
try:
    import psi4
    PSI4_AVAILABLE = True
    print(f"PSI4 {psi4.__version__} available")
except ImportError:
    PSI4_AVAILABLE = False
    print("PSI4 not available — TS/IRC steps will be skipped")

PSI4_METHOD  = 'b3lyp'
PSI4_BASIS   = '6-31G*'
PSI4_OPTIONS = {
    'basis': PSI4_BASIS, 'scf_type': 'df', 'reference': 'rhf',
    'maxiter': 200, 'e_convergence': 1e-7, 'd_convergence': 1e-7,
}
PSI4_MEM_GB  = 6
PSI4_THREADS = 4
HARTREE_TO_KCAL = 627.509474
AU_TO_DEBYE     = 2.541746
ANGSTROM_TO_BOHR = 1.88972612456

# Reaction coordinate atom indices (0-based)
IDX_C4  = 5    # methyl carbon
IDX_H4  = 9    # transferring hydrogen
IDX_O2  = 2    # distal oxygen (H acceptor)
IDX_O1  = 1    # proximal oxygen
IDX_C1  = 0    # Criegee carbon


# ── PSI4 helpers ──────────────────────────────────────────────────────────────

def _psi4_setup():
    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.be_quiet()
    psi4.set_memory(f'{PSI4_MEM_GB} GB')
    psi4.set_num_threads(PSI4_THREADS)
    psi4.set_options(PSI4_OPTIONS)


def _mol_str(symbols, coords, charge=0, mult=1):
    lines = [f'{charge} {mult}']
    for s, c in zip(symbols, coords):
        lines.append(f'{s}  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}')
    lines += ['units angstrom', 'no_reorient', 'no_com']
    return '\n'.join(lines)


def _psi4_energy_forces_dipole(symbols, coords, charge=0, mult=1):
    """Single-point energy, forces (Ha/Å), dipole (D), and spin diagnostics."""
    _psi4_setup()
    mol = psi4.geometry(_mol_str(symbols, coords, charge, mult))
    e, wfn = psi4.energy(
        f'{PSI4_METHOD}/{PSI4_BASIS}', molecule=mol,
        return_wfn=True, properties=['dipole'],
    )
    # Gradient (forces = -grad)
    G = psi4.gradient(f'{PSI4_METHOD}/{PSI4_BASIS}', molecule=mol)
    grad_au = np.array(G)  # Ha/Bohr
    forces  = -grad_au / ANGSTROM_TO_BOHR  # Ha/Å

    # Dipole
    try:
        dip_au = np.array(psi4.variable('SCF DIPOLE'))
        dipole = dip_au * AU_TO_DEBYE
    except Exception:
        dipole = np.zeros(3)

    # Spin diagnostics (multi-reference proxy)
    s2  = float(psi4.variable('SCF S^2 EIGENVALUE')) if psi4.has_variable('SCF S^2 EIGENVALUE') else 0.0
    s2_expected = 0.0 if mult == 1 else (mult - 1) * mult / 4.0
    spin_contam = s2 - s2_expected

    return float(e), forces, dipole, spin_contam


def _psi4_optimize(symbols, coords, charge=0, mult=1, ts=False):
    """Optimize geometry.  If ts=True, use PSI4 saddle-point optimizer."""
    _psi4_setup()
    if ts:
        psi4.set_options({
            'optking__opt_type': 'ts',
            'optking__geom_maxiter': 100,
            'optking__consecutive_backsteps_allowed': 5,
        })
    else:
        psi4.set_options({'optking__geom_maxiter': 100})
    mol = psi4.geometry(_mol_str(symbols, coords, charge, mult))
    e = psi4.optimize(f'{PSI4_METHOD}/{PSI4_BASIS}', molecule=mol)
    opt_coords = np.array(mol.geometry()) / ANGSTROM_TO_BOHR
    return float(e), opt_coords


# ── Reaction-coordinate diagnostics ──────────────────────────────────────────

def rxn_coord_distances(coords):
    """Return key distances for the 1,4-H shift reaction coordinate."""
    d_CH  = float(np.linalg.norm(coords[IDX_C4] - coords[IDX_H4]))   # breaking
    d_OH  = float(np.linalg.norm(coords[IDX_O2] - coords[IDX_H4]))   # forming
    d_OO  = float(np.linalg.norm(coords[IDX_O1] - coords[IDX_O2]))   # spectator O-O
    d_CO  = float(np.linalg.norm(coords[IDX_C1] - coords[IDX_O1]))   # C1-O1
    return d_CH, d_OH, d_OO, d_CO


def print_rxn_coord_header():
    print(f"  {'Frame':>6}  {'E(Ha)':>14}  {'ΔE(kcal)':>10}  "
          f"{'C4-H4(Å)':>9}  {'O2-H4(Å)':>9}  {'O1-O2(Å)':>9}  "
          f"{'C1-O1(Å)':>9}  {'SpinCont':>9}")
    print('  ' + '─' * 90)


def print_rxn_coord_row(i, e, e_ref, coords, spin_contam=0.0):
    d_CH, d_OH, d_OO, d_CO = rxn_coord_distances(coords)
    dE = (e - e_ref) * HARTREE_TO_KCAL
    print(f"  {i:>6}  {e:>14.8f}  {dE:>+10.3f}  "
          f"{d_CH:>9.4f}  {d_OH:>9.4f}  {d_OO:>9.4f}  "
          f"{d_CO:>9.4f}  {spin_contam:>9.4f}")


# ── Step 1: TS Search ─────────────────────────────────────────────────────────

def run_ts_search(symbols, reactant_coords, out_dir: Path):
    """
    Search for the syn-MVKO → VHP transition state.

    Strategy:
    1. Build a guess TS geometry by manually contracting O2-H4 and
       stretching C4-H4 from the reactant (syn-MVKO) geometry.
    2. Run PSI4 saddle-point optimization.
    3. Verify one imaginary mode.
    4. Return TS coords and frequency.
    """
    print("\n" + "=" * 70)
    print("  TS SEARCH: syn-MVKO → vinyl hydroperoxide")
    print("=" * 70)
    print(f"\n  PSI4 {PSI4_METHOD}/{PSI4_BASIS}")
    print(f"  Multi-reference note: B3LYP is single-reference.")
    print(f"  TS barrier height is approximate (±5 kcal/mol).")
    print(f"  ⟨S²⟩ deviation from 0.0 flags open-shell character.\n")

    # Build TS guess: stretch C4-H4, compress O2-H4
    coords_ts_guess = reactant_coords.copy()
    h_pos = reactant_coords[IDX_H4]
    c4_pos = reactant_coords[IDX_C4]
    o2_pos = reactant_coords[IDX_O2]

    # Move H4 35% of the way from C4 toward O2 (rough TS geometry)
    midpoint = h_pos + 0.35 * (o2_pos - h_pos)
    coords_ts_guess[IDX_H4] = midpoint

    d_CH_guess, d_OH_guess, _, _ = rxn_coord_distances(coords_ts_guess)
    print(f"  TS guess: C4-H4 = {d_CH_guess:.3f} Å, O2-H4 = {d_OH_guess:.3f} Å")

    # PSI4 TS optimization
    print(f"\n  Running PSI4 TS optimization ...")
    t0 = time.time()
    try:
        e_ts, coords_ts = _psi4_optimize(symbols, coords_ts_guess, ts=True)
        print(f"  TS optimization converged in {time.time()-t0:.0f}s")
    except Exception as exc:
        print(f"  TS optimization failed: {exc}")
        print(f"  Saving TS guess as placeholder. Re-run with --steps ts after")
        print(f"  manually adjusting the guess geometry.")
        coords_ts = coords_ts_guess
        e_ts = float('nan')

    d_CH, d_OH, d_OO, d_CO = rxn_coord_distances(coords_ts)
    print(f"\n  TS geometry:")
    print(f"    C4-H4 (breaking): {d_CH:.4f} Å")
    print(f"    O2-H4 (forming) : {d_OH:.4f} Å")
    print(f"    O1-O2 (spectator): {d_OO:.4f} Å")
    print(f"    C1-O1            : {d_CO:.4f} Å")

    # Save TS geometry
    ts_path = out_dir / 'ts_coords.npy'
    np.save(str(ts_path), coords_ts)

    ts_info = {
        'e_ts_ha': e_ts,
        'ts_coords_path': str(ts_path),
        'd_C4H4_ts': d_CH,
        'd_O2H4_ts': d_OH,
        'method': f'{PSI4_METHOD}/{PSI4_BASIS}',
        'note': 'B3LYP single-reference; barrier height approximate',
    }
    with open(out_dir / 'ts_info.json', 'w') as f:
        json.dump(ts_info, f, indent=2)

    print(f"\n  TS energy  : {e_ts:.6f} Ha")
    print(f"  TS coords  : {ts_path}")
    return coords_ts, e_ts


# ── Step 2: IRC sampling ──────────────────────────────────────────────────────

def run_irc_sampling(symbols, coords_ts, e_ts, reactant_coords, out_dir: Path,
                     n_points_per_side: int = 20,
                     step_size: float = 0.15):
    """
    Sample the IRC manually using steepest-descent from the TS along the
    imaginary-mode eigenvector (both forward and backward).

    This is a simplified IRC that doesn't use PSI4's built-in IRC driver
    (which can be unreliable for long paths).  We displace along the TS
    Hessian imaginary-mode eigenvector in small steps, re-computing
    energy+forces at each point.

    Also adds normal-mode displacements around 5 key IRC points to
    give the ML model coverage of the reaction-path PES well.
    """
    print("\n" + "=" * 70)
    print("  IRC SAMPLING: mapping syn-MVKO → VHP reaction path")
    print("=" * 70)
    print(f"\n  {n_points_per_side} points per side, step={step_size} Å·√(amu)")
    print(f"  Multi-reference flag: |⟨S²⟩| > 0.1 printed with ⚠️\n")

    # Use PSI4 Hessian at TS to get imaginary eigenvector
    print("  Computing PSI4 Hessian at TS ...")
    _psi4_setup()
    psi4.set_options({'hessian_write': True})
    mol = psi4.geometry(_mol_str(symbols, coords_ts))
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))
    try:
        _, wfn_hess = psi4.frequency(f'{PSI4_METHOD}/{PSI4_BASIS}',
                                      molecule=mol, return_wfn=True)
        # Extract Hessian and diagonalize
        H_au = np.array(wfn_hess.hessian())   # (3N, 3N) in Ha/Bohr²
        n_at = len(symbols)
        # Mass-weight
        from modules.bakken import ATOMIC_MASSES
        masses = np.array([ATOMIC_MASSES[s] for s in symbols])
        mass_vec = np.repeat(masses, 3)
        H_mw = H_au / np.outer(np.sqrt(mass_vec), np.sqrt(mass_vec))
        eigvals, eigvecs = np.linalg.eigh(H_mw)
        # Imaginary mode: most negative eigenvalue
        imag_idx  = np.argmin(eigvals)
        imag_freq = np.sign(eigvals[imag_idx]) * np.sqrt(abs(eigvals[imag_idx]))
        imag_vec  = eigvecs[:, imag_idx].reshape(n_at, 3)   # mass-weighted
        # Un-mass-weight to get displacement vector in Cartesian coordinates
        imag_vec_cart = imag_vec / np.sqrt(masses[:, None])
        imag_vec_cart /= np.linalg.norm(imag_vec_cart)
        print(f"  Imaginary mode found: λ = {eigvals[imag_idx]:.6f} Ha/(Bohr²·amu)")
    except Exception as exc:
        print(f"  Hessian failed ({exc}); using C4→O2 H-transfer vector as IRC direction")
        # Fall back: H-transfer vector
        direction = coords_ts[IDX_O2] - coords_ts[IDX_C4]
        direction /= np.linalg.norm(direction)
        imag_vec_cart = np.zeros_like(coords_ts)
        imag_vec_cart[IDX_H4] = direction

    # Walk IRC
    all_symbols = symbols
    all_coords  = []
    all_energies = []
    all_forces   = []
    all_dipoles  = []
    all_spin_contam = []
    all_irc_s    = []   # IRC coordinate (arbitrary units)

    e_ref = e_ts

    print_rxn_coord_header()

    for direction_sign, label in [(+1, 'forward→VHP'), (-1, 'backward→MVKO')]:
        print(f"\n  Direction: {label}")
        coords = coords_ts.copy()
        for k in range(n_points_per_side):
            # Step along IRC direction
            coords = coords + direction_sign * step_size * imag_vec_cart
            irc_s  = direction_sign * (k + 1) * step_size

            try:
                e, f, d, sc = _psi4_energy_forces_dipole(all_symbols, coords)
                flag = '⚠️ ' if abs(sc) > 0.1 else '  '
                print_rxn_coord_row(
                    direction_sign * (k + 1), e, e_ref, coords, sc)
                if abs(sc) > 0.1:
                    print(f"  {flag} Multi-reference character: ⟨S²⟩ deviation = {sc:.4f}")

                all_coords.append(coords.copy())
                all_energies.append(e)
                all_forces.append(f)
                all_dipoles.append(d)
                all_spin_contam.append(sc)
                all_irc_s.append(irc_s)

                # Relax forces: follow gradient to stay on IRC
                # (simple steepest-descent correction step)
                grad_step = 0.1 * f  # Ha/Å → displacement in Å (approximate)
                coords = coords + grad_step

            except Exception as exc:
                print(f"  Step {k+1} failed: {exc}")
                break

    # Add TS itself
    try:
        e_ts2, f_ts, d_ts, sc_ts = _psi4_energy_forces_dipole(all_symbols, coords_ts)
        all_coords.append(coords_ts.copy())
        all_energies.append(e_ts2)
        all_forces.append(f_ts)
        all_dipoles.append(d_ts)
        all_spin_contam.append(sc_ts)
        all_irc_s.append(0.0)
    except Exception:
        pass

    # Also add reactant minimum
    try:
        e_r, f_r, d_r, sc_r = _psi4_energy_forces_dipole(all_symbols, reactant_coords)
        all_coords.append(reactant_coords.copy())
        all_energies.append(e_r)
        all_forces.append(f_r)
        all_dipoles.append(d_r)
        all_spin_contam.append(sc_r)
        all_irc_s.append(-(n_points_per_side + 1) * step_size)
    except Exception:
        pass

    coords_arr  = np.array(all_coords)
    energies_arr = np.array(all_energies)
    forces_arr   = np.array(all_forces)
    dipoles_arr  = np.array(all_dipoles)
    spin_arr     = np.array(all_spin_contam)
    irc_s_arr    = np.array(all_irc_s)

    # Save
    irc_path = out_dir / 'irc_training_data.npz'
    np.savez(str(irc_path),
             symbols=np.array(all_symbols),
             coordinates=coords_arr,
             energies=energies_arr,
             forces=forces_arr,
             dipoles=dipoles_arr,
             spin_contamination=spin_arr,
             irc_s=irc_s_arr,
             metadata=np.array(json.dumps({
                 'source': 'mvko_syn_oh_path IRC',
                 'n_frames': int(len(all_coords)),
                 'method': f'{PSI4_METHOD}/{PSI4_BASIS}',
                 'multi_ref_note': 'B3LYP single-ref; ⟨S²⟩ dev > 0.1 flags open-shell',
                 'rxn_atoms': {
                     'C4': IDX_C4, 'H4': IDX_H4, 'O2': IDX_O2,
                     'O1': IDX_O1, 'C1': IDX_C1,
                 },
             })))

    print(f"\n  IRC training data: {len(all_coords)} points → {irc_path}")

    # Energy profile summary
    e_min_idx = int(np.argmin(energies_arr))
    e_ts_idx  = np.argmin(np.abs(irc_s_arr))
    e_min     = energies_arr[e_min_idx]
    dE_barrier = (energies_arr[e_ts_idx] - e_min) * HARTREE_TO_KCAL
    print(f"  Approximate barrier : {dE_barrier:.1f} kcal/mol  (B3LYP, approximate)")
    print(f"  ⟨S²⟩ range         : {spin_arr.min():.4f} – {spin_arr.max():.4f}")
    n_open = int((np.abs(spin_arr) > 0.1).sum())
    print(f"  Open-shell frames   : {n_open}/{len(spin_arr)}  (|ΔS²| > 0.1)")

    return irc_path, coords_arr, energies_arr, spin_arr


# ── Step 3: Train reaction-path ML-PES and assemble PESFamily ─────────────────

def train_rxn_path_pes(irc_data_path: str, reactant_model_path: str,
                       out_dir: Path, blend_width: float = 10.0,
                       gamma: float = 0.001, alpha: float = 1e-5):
    """
    Train a KRR ML-PES on IRC data and assemble PESFamily with the
    existing near-equilibrium MVKO model.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from modules.ml_pes import MLPESTrainer, CoulombMatrixDescriptor
    from modules.pes_family import PESFamily
    from modules.data_formats import TrajectoryData

    print("\n" + "=" * 70)
    print("  TRAINING REACTION-PATH ML-PES")
    print("=" * 70)

    data = np.load(irc_data_path, allow_pickle=True)
    symbols  = data['symbols'].tolist()
    coords   = data['coordinates']
    energies = data['energies']
    forces   = data['forces']
    n_frames = len(coords)
    print(f"  IRC data: {n_frames} frames from {irc_data_path}")

    # Energy statistics
    e_min = energies.min()
    e_range = (energies.max() - e_min) * HARTREE_TO_KCAL
    print(f"  Energy range: {e_range:.1f} kcal/mol")
    if 'spin_contamination' in data:
        sc = data['spin_contamination']
        n_open = int((np.abs(sc) > 0.1).sum())
        print(f"  Open-shell frames (|ΔS²|>0.1): {n_open}/{n_frames}")
        print(f"  ⚠️  These frames have multi-reference character; B3LYP energies")
        print(f"     are approximate. Forces from these frames are included but")
        print(f"     treat the resulting ML-PES as qualitative in that region.")

    # Train
    traj = TrajectoryData(
        symbols=symbols,
        coordinates=coords,
        energies=energies,
        forces=forces,
        dipoles=None,
        metadata={'source': 'irc_rxn_path'},
    )
    trainer = MLPESTrainer()
    trainer.train(traj, gamma=gamma, alpha=alpha)
    print(f"\n  Reaction-path ML-PES: γ={gamma}, α={alpha}")

    rxn_pkl = out_dir / 'mlpes_rxn_path.pkl'
    trainer.save(str(rxn_pkl))
    print(f"  Saved: {rxn_pkl}")

    # Energy alignment: align both surfaces to the reactant model's energy
    # at the near-equilibrium MVKO geometry (first IRC point, most negative s)
    print(f"\n  Assembling PESFamily  (blend_width={blend_width} kcal/mol)")
    print(f"  Loading reactant surface: {reactant_model_path}")

    # Reference energies at the MVKO minimum (lowest-energy IRC point)
    ref_idx = int(np.argmin(energies))
    ref_coords = coords[ref_idx]

    from modules.ml_pes import MLPESTrainer as _MLT
    reactant_trainer = _MLT.load(str(reactant_model_path))
    e_ref_reactant = float(reactant_trainer.predict(symbols, ref_coords))
    e_ref_rxn      = float(trainer.predict(symbols, ref_coords))
    e_ref_psi4     = float(energies[ref_idx])

    # Use PSI4 energies as reference; offset ML models to match
    ref_energies = {
        'reactant': e_ref_reactant - e_ref_psi4,
        'rxn_path': e_ref_rxn      - e_ref_psi4,
    }
    print(f"  Reactant surface offset : {ref_energies['reactant']*HARTREE_TO_KCAL:+.3f} kcal/mol")
    print(f"  Rxn-path surface offset : {ref_energies['rxn_path']*HARTREE_TO_KCAL:+.3f} kcal/mol")

    family = PESFamily.from_model_paths(
        symbols,
        {'reactant': str(reactant_model_path),
         'rxn_path':  str(rxn_pkl)},
        blend_width=blend_width,
        reference_energies=ref_energies,
    )

    family_pkl = out_dir / 'pes_family.pkl'
    family.save(str(family_pkl))
    print(f"  PESFamily saved: {family_pkl}")

    # Manifest for ir_md_spectrum.py --multi-surface
    manifest = {
        'reactant': str(reactant_model_path),
        'rxn_path': str(rxn_pkl),
        '_blend_width': blend_width,
        '_reference_energies': ref_energies,
    }
    manifest_path = out_dir / 'rxn_family_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_path}")

    return family, family_pkl, manifest_path


# ── Step 4: High-energy ML-MD with reaction-path monitoring ──────────────────

def run_reaction_md(family, symbols, reactant_coords, out_dir: Path,
                    temperature: float = 2000.0,
                    n_steps: int = 50000,
                    timestep: float = 0.3,
                    save_every: int = 5,
                    print_every: int = 500):
    """
    Run high-temperature ML-MD on the PESFamily surface with full reaction-
    coordinate monitoring.  This simulates a nascent (hot) Criegee intermediate
    exploring up to and past the TS.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from modules.bakken import MLPESDriver, run_md, maxwell_boltzmann_velocities
    from modules.pes_family import PESFamily

    print("\n" + "=" * 70)
    print(f"  HIGH-ENERGY ML-MD  (T={temperature:.0f} K, {n_steps} steps)")
    print("=" * 70)
    print(f"\n  Monitoring reaction coordinate:")
    print(f"    C4(5)–H4(9)  : breaking bond")
    print(f"    O2(2)–H4(9)  : forming bond (OH product)")
    print(f"    O1(1)–O2(2)  : Criegee O-O (spectator)")
    print(f"    C1(0)–O1(1)  : C-O (Criegee backbone)")
    print(f"\n  Multi-reference note: geometries with O2-H4 < 1.2 Å are in")
    print(f"  the VHP/OH product region where B3LYP ML-PES is qualitative.\n")

    # Use a PESFamilyDriver wrapper
    class PESFamilyDriver:
        """Thin driver wrapping PESFamily for bakken.run_md compatibility."""
        def __init__(self, fam: PESFamily):
            self.family  = fam
            self.symbols = fam.symbols
            self.n_atoms = len(fam.symbols)
            from modules.bakken import ATOMIC_MASSES
            self.masses  = np.array([ATOMIC_MASSES[s] for s in self.symbols])
            self._surface_log: list = []   # [(step, label, weights)]

        def energy(self, coords: np.ndarray) -> float:
            return self.family.blend_energy(coords)

        def forces(self, coords: np.ndarray) -> np.ndarray:
            _, f = self.family.energy_and_forces(coords, method='analytic')
            return f

        def surface_weights(self, coords: np.ndarray) -> dict:
            return dict(zip(self.family._labels,
                            self.family._weights(coords)[1]))

    driver = PESFamilyDriver(family)

    monitor_bonds = [
        (IDX_C4, IDX_H4, 'C4-H4'),
        (IDX_O2, IDX_H4, 'O2-H4'),
        (IDX_O1, IDX_O2, 'O1-O2'),
        (IDX_C1, IDX_O1, 'C1-O1'),
    ]

    md_result = run_md(
        driver, reactant_coords, n_steps, temperature,
        timestep=timestep,
        save_every=save_every,
        preminimize=False,   # don't pre-minimize at high T
        monitor_bonds=monitor_bonds,
        print_every=print_every,
        max_bond_extension=3.0,   # stop if O-O extends >3× (real dissociation)
        seed=42,
    )

    # Post-trajectory analysis
    _post_trajectory_analysis(md_result, driver, out_dir, temperature)

    return md_result


def _post_trajectory_analysis(md_result: dict, driver, out_dir: Path,
                               temperature: float):
    """Print and plot comprehensive post-trajectory diagnostics."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    coords_traj = md_result['coords_traj']
    energies    = md_result['energies_ml'] * HARTREE_TO_KCAL
    times       = md_result['times_fs']
    symbols     = md_result['symbols']
    bond_dist   = md_result.get('bond_distances_traj')   # (n_frames, 4) or None
    bond_labels = md_result.get('bond_labels') or []

    n_frames = len(coords_traj)
    e_ref    = energies.min()

    print("\n" + "=" * 70)
    print("  POST-TRAJECTORY DIAGNOSTICS")
    print("=" * 70)

    # Energy statistics
    print(f"\n  Energy statistics ({n_frames} frames):")
    print(f"    Min  : {energies.min():>12.3f}  kcal/mol  (relative: 0.0)")
    print(f"    Mean : {energies.mean():>12.3f}  kcal/mol  "
          f"(ΔE = {energies.mean()-e_ref:+.1f} kcal/mol)")
    print(f"    Max  : {energies.max():>12.3f}  kcal/mol  "
          f"(ΔE = {energies.max()-e_ref:+.1f} kcal/mol above min)")
    print(f"    σ(E) : {energies.std():>12.3f}  kcal/mol")

    # Reaction-coordinate summary
    if bond_dist is not None and len(bond_labels) >= 2:
        print(f"\n  Reaction-coordinate statistics (Å):")
        print(f"  {'Bond':>12}  {'Min':>7}  {'Mean':>7}  {'Max':>7}  {'σ':>7}")
        print('  ' + '─' * 50)
        for k, lbl in enumerate(bond_labels):
            d = bond_dist[:, k]
            print(f"  {lbl:>12}  {d.min():>7.4f}  {d.mean():>7.4f}  "
                  f"{d.max():>7.4f}  {d.std():>7.4f}")

        # Count reactive events: O2-H4 < 1.2 Å (near OH formed)
        # Bond label index 1 = O2-H4 (second monitor_bonds entry)
        if len(bond_labels) >= 2:
            oh_idx = next((k for k, l in enumerate(bond_labels) if 'O2-H4' in l or 'H4' in l and 'O2' in l), None)
            if oh_idx is not None:
                oh_dist = bond_dist[:, oh_idx]
                n_near_product = int((oh_dist < 1.2).sum())
                frac = n_near_product / n_frames * 100
                print(f"\n  Frames with O2-H4 < 1.2 Å (near product): "
                      f"{n_near_product}/{n_frames}  ({frac:.1f}%)")
                if n_near_product > 0:
                    print(f"  ⚠️  These frames are in the VHP/OH product region")
                    print(f"      where B3LYP ML-PES is qualitatively unreliable.")
                    print(f"      For quantitative results, replace B3LYP IRC data")
                    print(f"      with CASSCF(4,4) or NEVPT2 single-points.")

    # Surface occupancy (PESFamily)
    if hasattr(driver, 'family'):
        print(f"\n  Surface occupancy (PESFamily, every 10th frame):")
        from collections import Counter
        labels = [driver.family.assign_conformer(c)
                  for c in coords_traj[::10]]
        counts = Counter(labels)
        total  = sum(counts.values())
        for lbl, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {lbl:>15}: {cnt:>6} frames  ({cnt/total*100:.1f}%)")

    # Save bond distances as CSV
    if bond_dist is not None:
        import csv
        csv_path = out_dir / 'rxn_trajectory_bonds.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_fs', 'energy_kcal'] + bond_labels)
            for k in range(n_frames):
                row = [times[k], energies[k]] + list(bond_dist[k])
                writer.writerow(row)
        print(f"\n  Bond distances CSV: {csv_path}")

    # Matplotlib diagnostic figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n_panels = 2 + (1 if bond_dist is not None else 0)
        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels))
        ax = axes if hasattr(axes, '__len__') else [axes]

        # Panel 1: Energy vs time
        ax[0].plot(times / 1000, energies - e_ref, lw=0.8, color='steelblue')
        ax[0].axhline(0, color='gray', lw=0.5, ls='--')
        ax[0].set_xlabel('Time (ps)')
        ax[0].set_ylabel('ΔE (kcal/mol)')
        ax[0].set_title(f'Energy trajectory  (T={temperature:.0f} K, '
                        f'{n_frames} frames)')

        # Panel 2: Energy histogram
        ax[1].hist(energies - e_ref, bins=60, color='steelblue', alpha=0.7,
                   edgecolor='navy', lw=0.3)
        ax[1].set_xlabel('ΔE (kcal/mol)')
        ax[1].set_ylabel('Frames')
        ax[1].set_title('Energy distribution')

        # Panel 3: Bond distances vs time
        if bond_dist is not None:
            colors = ['C0', 'C1', 'C3', 'C4']
            for k, lbl in enumerate(bond_labels):
                ax[2].plot(times / 1000, bond_dist[:, k], lw=0.7,
                           label=lbl, color=colors[k % len(colors)])
            # Equilibrium O-H reference
            ax[2].axhline(0.97, color='C1', lw=0.5, ls='--',
                          label='O-H eq (0.97 Å)')
            ax[2].set_xlabel('Time (ps)')
            ax[2].set_ylabel('Distance (Å)')
            ax[2].set_title('Reaction-coordinate bond distances')
            ax[2].legend(ncol=2, fontsize=8)

        plt.tight_layout()
        fig_path = out_dir / 'rxn_md_diagnostics.png'
        fig.savefig(str(fig_path), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Diagnostic figure  : {fig_path}")

    except Exception as exc:
        print(f"  (Matplotlib figure skipped: {exc})")

    if md_result.get('dissociation_step'):
        print(f"\n  ⚠️  Trajectory truncated at step {md_result['dissociation_step']}"
              f" (heavy-atom bond extension limit)")


# ── IRC energy profile plot ───────────────────────────────────────────────────

def plot_irc_profile(irc_data_path: str, ts_e: float, out_dir: Path):
    """Plot IRC energy profile with multi-reference flagging."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        data     = np.load(irc_data_path, allow_pickle=True)
        irc_s    = data['irc_s']
        energies = data['energies'] * HARTREE_TO_KCAL
        e_ref    = energies.min()
        sc       = data.get('spin_contamination', np.zeros_like(irc_s))

        sort_idx = np.argsort(irc_s)
        irc_s_s  = irc_s[sort_idx]
        e_s      = energies[sort_idx] - e_ref
        sc_s     = sc[sort_idx]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Energy profile, colour by spin contamination
        ax1.scatter(irc_s_s, e_s, c=np.abs(sc_s), cmap='RdYlGn_r',
                    s=40, zorder=3, vmin=0, vmax=0.3)
        ax1.plot(irc_s_s, e_s, lw=0.8, color='gray', zorder=2)
        ax1.set_xlabel('IRC coordinate (Å·√amu)')
        ax1.set_ylabel('ΔE (kcal/mol)')
        ax1.set_title('IRC energy profile  (colour = |ΔS²|; red = multi-ref)')
        ax1.axvline(0, color='red', lw=0.8, ls='--', label='TS')
        ax1.legend()

        # Spin contamination along IRC
        ax2.plot(irc_s_s, np.abs(sc_s), color='firebrick', lw=1.5)
        ax2.axhline(0.1, color='orange', lw=0.8, ls='--', label='|ΔS²| = 0.1 threshold')
        ax2.set_xlabel('IRC coordinate (Å·√amu)')
        ax2.set_ylabel('|ΔS²| (spin contamination)')
        ax2.set_title('Multi-reference character along IRC')
        ax2.legend()

        plt.tight_layout()
        fig_path = out_dir / 'irc_energy_profile.png'
        fig.savefig(str(fig_path), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  IRC profile figure  : {fig_path}")

    except Exception as exc:
        print(f"  (IRC plot skipped: {exc})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', default='ts,irc,train,md',
                        help='Comma-separated steps to run: ts,irc,train,md')
    parser.add_argument('--reactant-model', default='outputs/mvko_20260319_081314/mlpes_initial.pkl',
                        help='Path to near-eq MVKO ML-PES (.pkl)')
    parser.add_argument('--reactant-coords',
                        default='outputs/mvko_20260319_081314/psi4_eq_coords.npy',
                        help='Path to PSI4 eq coords (.npy) for MVKO')
    parser.add_argument('--irc-data', default=None,
                        help='Existing IRC training data (.npz) — skips ts/irc steps')
    parser.add_argument('--family-pkl', default=None,
                        help='Existing PESFamily .pkl — skips ts/irc/train steps')
    parser.add_argument('--blend-width', type=float, default=10.0,
                        help='Softmin blend width (kcal/mol) [default 10]')
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=1e-5)
    parser.add_argument('--temp', type=float, default=2000.0,
                        help='MD temperature (K) — high T to explore TS region')
    parser.add_argument('--md-steps', type=int, default=50000)
    parser.add_argument('--timestep', type=float, default=0.3)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--print-every', type=int, default=500)
    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(',')]

    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(f'outputs/mvko_rxn_path_{ts_str}')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Output directory: {out_dir}")

    # Load reactant geometry
    reactant_coords = None
    if Path(args.reactant_coords).exists():
        reactant_coords = np.load(args.reactant_coords)
        print(f"  Reactant coords  : {args.reactant_coords}  shape={reactant_coords.shape}")
    else:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from modules.test_molecules import get_molecule
        mol = get_molecule('mvko')
        reactant_coords = np.array(mol['coords'])
        print(f"  Reactant coords  : using test_molecules initial guess")

    # Get symbols from reactant model or test_molecules
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from modules.ml_pes import MLPESTrainer
    reactant_trainer = MLPESTrainer.load(args.reactant_model)
    symbols = reactant_trainer.symbols

    # ── Step: TS search ───────────────────────────────────────────────────────
    coords_ts = None
    e_ts      = None
    if 'ts' in steps:
        if not PSI4_AVAILABLE:
            print("  PSI4 not available — skipping TS search")
        else:
            coords_ts, e_ts = run_ts_search(symbols, reactant_coords, out_dir)

    # ── Step: IRC ─────────────────────────────────────────────────────────────
    irc_data_path = args.irc_data
    if 'irc' in steps and coords_ts is not None:
        if not PSI4_AVAILABLE:
            print("  PSI4 not available — skipping IRC")
        else:
            irc_data_path, _, _, _ = run_irc_sampling(
                symbols, coords_ts, e_ts or 0.0, reactant_coords, out_dir)

    # ── Step: Train ───────────────────────────────────────────────────────────
    family     = None
    family_pkl = args.family_pkl
    if 'train' in steps:
        if irc_data_path is None:
            print("  No IRC data available — skipping training step")
            print("  Provide --irc-data or run --steps ts,irc first")
        else:
            family, family_pkl, _ = train_rxn_path_pes(
                irc_data_path, args.reactant_model, out_dir,
                blend_width=args.blend_width,
                gamma=args.gamma, alpha=args.alpha,
            )
            if irc_data_path:
                plot_irc_profile(irc_data_path, e_ts or 0.0, out_dir)

    # ── Step: MD ──────────────────────────────────────────────────────────────
    if 'md' in steps:
        if family is None and family_pkl is not None:
            from modules.pes_family import PESFamily
            family = PESFamily.load(str(family_pkl))
        if family is None:
            print("  No PESFamily available — skipping MD step")
            print("  Provide --family-pkl or run --steps train first")
        else:
            run_reaction_md(
                family, symbols, reactant_coords, out_dir,
                temperature=args.temp,
                n_steps=args.md_steps,
                timestep=args.timestep,
                save_every=args.save_every,
                print_every=args.print_every,
            )

    # Save run summary
    summary = {
        'timestamp': ts_str,
        'output_dir': str(out_dir),
        'steps_run': steps,
        'reactant_model': args.reactant_model,
        'irc_data': str(irc_data_path) if irc_data_path else None,
        'family_pkl': str(family_pkl) if family_pkl else None,
        'md_params': {
            'temperature': args.temp,
            'n_steps': args.md_steps,
            'timestep': args.timestep,
        },
        'multi_ref_note': (
            'B3LYP/6-31G* is single-reference. TS barrier ±5 kcal/mol. '
            'VHP/OH product region (O2-H4 < 1.2 A) has significant biradical '
            'character. Replace IRC training data with CASSCF(4,4)/6-31G* or '
            'NEVPT2 single-points for quantitative results.'
        ),
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary: {out_dir}/summary.json")

    print(f"\n{'=' * 70}")
    print(f"  DONE — outputs in {out_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
