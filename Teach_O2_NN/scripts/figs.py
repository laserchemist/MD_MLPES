import torch, torch.nn as nn, math, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
torch.manual_seed(7)
De, re, a = 5.21, 1.208, 2.66
morse = lambda r: De*(1-torch.exp(-a*(r-re)))**2 - De
mforce = lambda r: -2*De*a*torch.exp(-a*(r-re))*(1-torch.exp(-a*(r-re)))
t = torch.linspace(0,1,36); r_tr = (0.95+2.05*t**1.4).unsqueeze(1); V_tr, F_tr = morse(r_tr), mforce(r_tr)
Vs = V_tr.std()
def mlp(h): return nn.Sequential(nn.Linear(1,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,1))
class Plain(nn.Module):
    def __init__(s,h=16): super().__init__(); s.net=mlp(h); s.m,s.s,s.vm=r_tr.mean(),r_tr.std(),V_tr.mean()
    def forward(s,r): return s.net((r-s.m)/s.s)*Vs+s.vm
class Zero(nn.Module):
    def __init__(s,h=16,lam=0.5): super().__init__(); s.net=mlp(h); s.lam=lam; y=torch.exp(-(r_tr-re)/lam); s.ym,s.ys=y.mean(),y.std()
    def forward(s,r):
        x=(torch.exp(-(r-re)/s.lam)-s.ym)/s.ys; xi=((0-s.ym)/s.ys).reshape(1,1)
        return (s.net(x)-s.net(xi))*Vs
def EF(m,r,cg=False):
    r=r.clone().requires_grad_(True); V=m(r); d,=torch.autograd.grad(V.sum(),r,create_graph=cg); return V,-d
def train(m,ep=6000):
    o=torch.optim.Adam(m.parameters(),lr=1e-2); sc=torch.optim.lr_scheduler.StepLR(o,2000,0.3)
    for _ in range(ep):
        o.zero_grad(); V,F=EF(m,r_tr,True); L=((V-V_tr)**2).mean()+0.1*((F-F_tr)**2).mean(); L.backward(); o.step(); sc.step()
P,Z=Plain(),Zero(); train(P); train(Z)
INK="#14213D"; MUTED="#5A6475"; BG="#F7F5EF"; ORANGE="#E07A2F"; BLUE="#2A78D6"; GRID="#E1DED4"
plt.rcParams.update({"font.size":20,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,
  "axes.facecolor":BG,"figure.facecolor":BG,"axes.grid":True,"grid.color":GRID,"axes.spines.top":False,"axes.spines.right":False,"lines.solid_capstyle":"round"})
r=torch.linspace(0.9,3.1,400).unsqueeze(1); V,F=EF(P,r); V,F=V.detach(),F.detach()
fig,ax=plt.subplots(1,2,figsize=(16,6.4))
ax[0].plot(r,morse(r),"--",color=MUTED,lw=2.5,label="Morse reference"); ax[0].plot(r,V,color=BLUE,lw=3.5,label="GELU network")
ax[0].plot(r_tr,V_tr,"o",color=ORANGE,ms=7,label="training points"); ax[0].set_ylim(-6,4); ax[0].set_xlabel("O–O distance r (Å)"); ax[0].set_ylabel("V (eV)"); ax[0].legend(frameon=False)
ax[1].plot(r,mforce(r),"--",color=MUTED,lw=2.5,label="analytic −dV/dr"); ax[1].plot(r,F,color=BLUE,lw=3.5,label="autograd −dV/dr")
ax[1].axhline(0,color=MUTED,lw=0.8); ax[1].set_ylim(-15,40); ax[1].set_xlabel("O–O distance r (Å)"); ax[1].set_ylabel("F (eV/Å)"); ax[1].legend(frameon=False)
plt.tight_layout(); plt.savefig("fit_force.png",dpi=110)
r=torch.linspace(0.9,8,600).unsqueeze(1)
fig,ax=plt.subplots(figsize=(11,6.6))
ax.axvspan(3,8,color=GRID,alpha=0.6,lw=0); ax.text(5.5,2.4,"no training data (extrapolation)",ha="center",color=MUTED)
ax.plot(r,morse(r),"--",color=MUTED,lw=2.5,label="Morse reference")
ax.plot(r,P(r).detach(),color=ORANGE,lw=3.5,label="plain network V(r)")
ax.plot(r,Z(r).detach(),color=BLUE,lw=3.5,label="V = g(y) − g(0)")
ax.axhline(0,color=MUTED,lw=0.8); ax.set_ylim(-6,3); ax.set_xlabel("O–O distance r (Å)"); ax.set_ylabel("V (eV)"); ax.legend(frameon=False,loc="lower right")
plt.tight_layout(); plt.savefig("dissociation.png",dpi=110)
for nm,m in [("plain",P),("zero",Z)]:
    print(nm,[round(m(torch.tensor([[x]])).item(),3) for x in (1.208,3.0,5.0,8.0)])
