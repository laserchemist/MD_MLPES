import numpy as np, pickle, sys
sys.path.insert(0, '/Users/jmsmith1/Documents/Research/code/MD_MLPES')
from modules.ml_pes import CoulombMatrixDescriptor

with open('outputs/ir_mace_anti_cis_300K/dipole_surface.pkl', 'rb') as f:
    dsurf_dict = pickle.load(f)
model = dsurf_dict['model']
scaler_X = dsurf_dict['scaler_X']
scaler_y = dsurf_dict['scaler_y']

with open('outputs/ir_mace_anti_cis_300K/traj_01.xyz') as f:
    lines = f.readlines()

n_atoms = 12
syms = ['C','O','O','C','C','C','H','H','H','H','H','H']

def get_coords(lines, idx):
    start = idx * (n_atoms + 2)
    return np.array([[float(x) for x in lines[start+2+j].split()[1:4]] for j in range(n_atoms)])

n_total = 30000
idxs = np.linspace(0, n_total-1, 500, dtype=int).tolist()
desc = CoulombMatrixDescriptor()
dips = []
for i in idxs:
    coords = get_coords(lines, int(i))
    feat = desc.compute(syms, coords).reshape(1,-1)
    feat_sc = scaler_X.transform(feat)
    dip_sc = model.predict(feat_sc)
    dip = scaler_y.inverse_transform(dip_sc)[0]
    dips.append(dip)

dips = np.array(dips)
print('Dipole statistics (300K traj_01, 500 sampled frames):')
for ci, comp in enumerate(['x','y','z']):
    print(f'  mu_{comp}: std={dips[:,ci].std():.5f} D  range=[{dips[:,ci].min():.4f}, {dips[:,ci].max():.4f}]')
norm = np.linalg.norm(dips, axis=1)
print(f'  |mu|:   mean={norm.mean():.4f}  std={norm.std():.5f}')
