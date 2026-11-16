#!/usr/bin/env python3
"""Stand-in for ORCA in the tests: writes the parts of an ORCA CASSCF/NEVPT2 output that diabml reads.

Energies come from $MOCK_ORCA_TABLE (JSON: input basename -> [4 SA-CASSCF energies] or "fail")."""
import json, os, sys
name = sys.argv[1][:-4]
table = json.load(open(os.environ['MOCK_ORCA_TABLE']))
key = name if name in table else name.split('_r')[0]
e = table.get(key)
if e is None and os.environ.get('MOCK_ORCA_GEOM'):
    # energies from the geometry: smooth in the coordinates, four roots 0.1 Eh apart
    txt = open(sys.argv[1]).read().split('* xyz 0 1')[1].split('*')[0].split()
    xyz = [float(v) for i, v in enumerate(txt) if i % 4]
    base = -100.0 + 0.01 * sum(v * v for v in xyz)
    e = [base + 0.1 * k for k in range(4)]
inp = open(sys.argv[1]).read()
if 'CCSD(T)' in inp:                       # coupled cluster: the lines method_cut reads
    txt = inp.split('* xyz 0 1')[1].split('*')[0].split()
    xyz = [float(v) for i, v in enumerate(txt) if i % 4]
    ecc = -100.5 + 0.01 * sum(v * v for v in xyz)
    if os.environ.get('MOCK_ORCA_CC') == 'pairs':
        # a molecule-like potential (well9d tests): springs on all ten interatomic distances (reference distances
        # in $MOCK_ORCA_REF, JSON list in the order (0,1),(0,2),...,(3,4)) with a cubic term, plus out-of-plane
        # springs on z (the builder puts the molecular plane at z = 0)
        ref = json.load(open(os.environ['MOCK_ORCA_REF']))
        P = [xyz[3 * a:3 * a + 3] for a in range(5)]
        k = 0
        ecc = -100.5
        for a in range(5):
            for b in range(a + 1, 5):
                d = sum((P[a][c] - P[b][c]) ** 2 for c in range(3)) ** 0.5 - ref[k]
                ecc += 1.0 * d * d - 0.8 * d ** 3
                k += 1
        ecc += 0.2 * sum(P[a][2] ** 2 for a in range(5))
        if 'pVTZ' in inp:                  # a smooth basis-set difference: shorter C-H, softer C-O
            dist = lambda a, b: sum((P[a][c] - P[b][c]) ** 2 for c in range(3)) ** 0.5
            ecc += -0.01 + 0.004 * (dist(0, 3) - 1.08) + 0.003 * (dist(0, 4) - 1.08) - 0.05 * (dist(0, 1) - 1.27) ** 2
    if 'EField' in inp:                    # finite field (well9d_dipole tests): a molecule-like dipole, ORCA-mock
        # convention: E(F) = E0 + F.(mu - mu_nuc(input origin)) (electronic part only, opposite sign), so the stage
        # must find sign -1 plus the nuclear dipole about the input origin; the SCF dipole printed is 5% larger
        fl = [float(v) for v in inp.split('EField')[1].split('\n')[0].replace(',', ' ').split()]
        P = [xyz[3 * a:3 * a + 3] for a in range(5)]
        dist = lambda a, b: sum((P[a][c] - P[b][c]) ** 2 for c in range(3)) ** 0.5
        q = [0.2 + 0.3 * (dist(0, 1) - 1.27), 0.25 - 0.4 * (dist(1, 2) - 1.35), 0.0,
             0.05 + 0.2 * (dist(0, 3) - 1.08), 0.05 + 0.2 * (dist(0, 4) - 1.08)]
        q[2] = -(q[0] + q[1] + q[3] + q[4])
        Z = [6, 8, 8, 1, 1]
        B = 1.8897261246
        mu = [sum(q[a] * P[a][c] * B for a in range(5)) for c in range(3)]
        nuc = [sum(Z[a] * P[a][c] * B for a in range(5)) for c in range(3)]
        ecc += sum(fl[c] * (mu[c] - nuc[c]) for c in range(3))
        print('Total Dipole Moment    :  ' + '  '.join(f'{1.05 * m + 0.002:.6f}' for m in mu))
    open(f'{name}.gbw', 'w').write('mock')
    print('T1 diagnostic                              ...      0.0150')
    print(f'E(CCSD(T))                                 ...   {ecc:.10f}')
    print(f'FINAL SINGLE POINT ENERGY      {ecc:.10f}')
    print('                             ****ORCA TERMINATED NORMALLY****')
    sys.exit(0)
if e == 'fail':
    print('ORCA finished by error termination in CASSCF'); sys.exit(1)
open(f'{name}.gbw', 'w').write('mock')
print('---- THE CAS-SCF GRADIENT HAS CONVERGED ----')
print('CASSCF RESULTS')
for k, x in enumerate(e):
    print(f'ROOT   {k}:  E=    {x:.10f} Eh')
print('NEVPT2 Results')
for x in e:
    print(f'\t Total Energy (E0+dE)    : E  = {x - 0.43:.12f}')
print('QD-NEVPT2 Results')
print('QD-NEVPT2 TOTAL ENERGIES')
for k, x in enumerate(e):
    print(f"   {k}:    {k}    1   A'  {x - 0.43 - 0.0001 * (k == 0):.6f}    EDIAG[{k}] 0.0")
print('                             ****ORCA TERMINATED NORMALLY****')
