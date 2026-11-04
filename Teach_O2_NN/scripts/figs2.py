import torch, torch.nn as nn, math, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
De, re, a = 5.21, 1.208, 2.66
morse = lambda r: De*(1-torch.exp(-a*(r-re)))**2 - De
mforce = lambda r: -2*De*a*torch.exp(-a*(r-re))*(1-torch.exp(-a*(r-re)))
def mcurv(r):
    e=torch.exp(-a*(r-re)); return 2*De*a*a*e*(2*e-1)
t = torch.linspace(0,1,36); r_tr = (0.95+2.05*t**1.4).unsqueeze(1); V_tr, F_tr = morse(r_tr), mforce(r_tr)
Vs, Vm, rm, rsd = V_tr.std(), V_tr.mean(), r_tr.mean(), r_tr.std()
class PES(nn.Module):
    def __init__(s, act, h=16):
        super().__init__(); s.net=nn.Sequential(nn.Linear(1,h),act(),nn.Linear(h,h),act(),nn.Linear(h,1))
    def forward(s,r): return s.net((r-rm)/rsd)*Vs+Vm
def derivs(m,r,cg=False):
    r=r.clone().requires_grad_(True); V=m(r)
    d1,=torch.autograd.grad(V.sum(),r,create_graph=True)
    d2,=torch.autograd.grad(d1.sum(),r,create_graph=cg, allow_unused=True)
    if d2 is None: d2=torch.zeros_like(r)
    return V,-d1,d2
def train(m,ep=6000):
    o=torch.optim.Adam(m.parameters(),lr=1e-2); sc=torch.optim.lr_scheduler.StepLR(o,2000,0.3)
    for _ in range(ep):
        o.zero_grad(); rr=r_tr.clone().requires_grad_(True); V=m(rr); d,=torch.autograd.grad(V.sum(),rr,create_graph=True)
        L=((V-V_tr)**2).mean()+0.1*((-d-F_tr)**2).mean(); L.backward(); o.step(); sc.step()
models={}
for name,act in [("GELU",nn.GELU),("ReLU",nn.ReLU)]:
    torch.manual_seed(7); m=PES(act); train(m); m.eval(); models[name]=m
def omega(m):
    r=torch.linspace(1.0,1.5,5001).unsqueeze(1)
    with torch.no_grad(): rmin=r[m(r).argmin()].reshape(1,1)
    _,_,k=derivs(m,rmin); k=k.item()
    if k<=0: return rmin.item(),k,float('nan')
    return rmin.item(),k,math.sqrt(k*1.602177e-19/1e-20/(8*1.66054e-27))/(2*math.pi*2.99792458e10)
r=torch.linspace(0.95,3.0,1500).unsqueeze(1)
for n,m in models.items():
    V,F,K=derivs(m,r)
    inr=(r>=0.95)&(r<=3.0)
    print(n,"E rmse",((V-morse(r))**2).mean().sqrt().item(),"F rmse",((F-mforce(r))**2).mean().sqrt().item(),"omega",omega(m))
INK="#14213D"; MUTED="#5A6475"; BG="#F7F5EF"; ORANGE="#E07A2F"; BLUE="#2A78D6"; GRID="#E1DED4"
plt.rcParams.update({"font.size":20,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,"axes.titlecolor":INK,
  "axes.facecolor":BG,"figure.facecolor":BG,"axes.grid":True,"grid.color":GRID,"axes.spines.top":False,"axes.spines.right":False})
rp=torch.linspace(0.95,2.2,1500).unsqueeze(1)
fig,ax=plt.subplots(1,3,figsize=(18,5.6))
ref=[morse(rp),mforce(rp),mcurv(rp)]
titles=["Energy V(r)","Force −dV/dr","Curvature d²V/dr²"]; ylabs=["eV","eV/Å","eV/Å²"]; lims=[(-5.6,3.5),(-10,40),(-60,240)]
for n,col,ls,lw in [("ReLU",ORANGE,"-",3),("GELU",BLUE,"-",3.5)]:
    V,F,K=derivs(models[n],rp)
    for i,y in enumerate([V,F,K]): ax[i].plot(rp,y.detach(),color=col,lw=lw,ls=ls,label=n+" network")
for i in range(3):
    ax[i].plot(rp,ref[i],"--",color=INK,lw=2,label="Morse (exact)")
    ax[i].set_title(titles[i]); ax[i].set_ylabel(ylabs[i]); ax[i].set_xlabel("r (Å)"); ax[i].set_ylim(*lims[i])
ax[0].legend(frameon=False,fontsize=18)
plt.tight_layout(); plt.savefig("gelu_vs_relu.png",dpi=110)
# network diagram of trained GELU model
m=models["GELU"]; Ws=[l.weight.detach().T.numpy() for l in m.net if isinstance(l,nn.Linear)]; bs=[l.bias.detach().numpy() for l in m.net if isinstance(l,nn.Linear)]
sizes=[1,16,16,1]; import numpy as np
pos=[[(k*3.2, (15/2) if n==1 else i) for i in range(n)] for k,n in enumerate(sizes)]
wmax=max(np.abs(W).max() for W in Ws); norm=TwoSlopeNorm(vmin=-wmax,vcenter=0,vmax=wmax); cmap=plt.get_cmap("RdBu_r")
fig,ax=plt.subplots(figsize=(15,8.2)); ax.set_facecolor(BG); ax.grid(False)
for k,W in enumerate(Ws):
    order=np.argsort(np.abs(W).ravel())
    for idx in order:
        i,j=np.unravel_index(idx,W.shape); w=W[i,j]
        (x1,y1),(x2,y2)=pos[k][i],pos[k+1][j]
        ax.plot([x1,x2],[y1,y2],color=cmap(norm(w)),lw=0.4+5*abs(w)/wmax,alpha=0.35+0.65*abs(w)/wmax,zorder=1)
bmax=max(np.abs(b).max() for b in bs); bnorm=TwoSlopeNorm(vmin=-bmax,vcenter=0,vmax=bmax)
for k in range(4):
    for i,(x,y) in enumerate(pos[k]):
        fc="#FFFFFF" if k==0 else cmap(bnorm(bs[k-1][i]))
        ax.scatter(x,y,s=420,color=fc,edgecolor=INK,lw=1.5,zorder=3)
ax.text(-0.45,7.5,"r",ha="right",va="center",fontsize=26,color=INK)
ax.text(9.6+0.45,7.5,"V(r)",ha="left",va="center",fontsize=26,color=INK)
for k,lab in enumerate(["input","hidden 1: 16 GELU","hidden 2: 16 GELU","output"]):
    ax.text(k*3.2,16.3,lab,ha="center",fontsize=20,color=MUTED)
ax.set_xlim(-1.6,11.4); ax.set_ylim(-0.8,17); ax.invert_yaxis(); ax.axis("off")
cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=ax,shrink=0.7,pad=0.01); cb.set_label("weight (edges)",color=INK); cb.ax.tick_params(colors=MUTED)
plt.tight_layout(); plt.savefig("network_weights.png",dpi=110)
print("wmax",wmax,"bmax",bmax, "n strong>0.5wmax", sum((np.abs(W)>0.5*wmax).sum() for W in Ws), "params", sum(W.size for W in Ws)+sum(b.size for b in bs))
print("biggest |w| layer2 frac positive", (Ws[1]>0).mean())
