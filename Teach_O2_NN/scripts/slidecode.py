import torch, torch.nn as nn
torch.manual_seed(7)
De, re, a = 5.21, 1.208, 2.66
t = torch.linspace(0, 1, 36)
R = 0.95 + 2.05 * t**1.4
e = torch.exp(-a * (R - re))
V_ref = (De * (1 - e)**2 - De).reshape(-1, 1)
F_ref = (-2 * De * a * e * (1 - e)).reshape(-1, 1)
class PES(nn.Module):
    def __init__(s, h=16):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(1, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
    def forward(s, r):
        return s.net((r - 1.9) / 0.6) * 1.6 - 1.9     # standard units in, eV out
# ---- slide code: TRAIN ----
r = R.reshape(-1, 1)                    # (36,1): every point at once
model = PES()                           # 1 -> 16 -> 16 -> 1, GELU
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(6000):
    opt.zero_grad()
    r.requires_grad_(True)
    V = model(r)                        # (36,1) energies
    dVdr, = torch.autograd.grad(V.sum(), r, create_graph=True)
    F = -dVdr                           # (36,1) forces
    loss = ((V - V_ref)**2).mean() + 0.1 * ((F - F_ref)**2).mean()
    loss.backward()                     # gradient for all 321 weights
    opt.step()                          # one Adam update
    if epoch == 2000: opt.param_groups[0]["lr"] = 0.003
    if epoch == 4000: opt.param_groups[0]["lr"] = 0.001
# ---- slide code: PREDICT ----
model.eval()
r = torch.linspace(0.9, 3.1, 400).reshape(-1, 1)
r.requires_grad_(True)
V = model(r)                            # (400,1)
dVdr, = torch.autograd.grad(V.sum(), r)
F = -dVdr                               # (400,1)
print(r.shape, V.shape, F.shape, "loss", loss.item())
rr = r.detach(); ee = torch.exp(-a*(rr-re)); Vm = De*(1-ee)**2-De; Fm = -2*De*a*ee*(1-ee)
m = (rr>=0.95)&(rr<=3.0)
print("E rmse", ((V.detach()-Vm)[m]**2).mean().sqrt().item(), "F rmse", ((F.detach()-Fm)[m]**2).mean().sqrt().item())
print("\n r(Å)   Morse(eV)   NN(eV)   NN force(eV/Å)")
for rv in [0.3, 0.5, 0.7, 0.9, 1.208, 3.0, 4.0, 6.0, 10.0, 20.0]:
    x = torch.tensor([[rv]], requires_grad=True)
    v = model(x); d, = torch.autograd.grad(v.sum(), x)
    ee = torch.exp(torch.tensor(-a*(rv-re))); vm = De*(1-ee)**2-De
    print(f"{rv:5.2f} {vm.item():11.2f} {v.item():9.2f} {(-d).item():10.2f}")
