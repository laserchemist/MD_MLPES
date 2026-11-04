import torch, torch.nn as nn, math, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
De, re, a = 5.21, 1.208, 2.66
morse = lambda r: De*(1-torch.exp(-a*(r-re)))**2 - De
mforce = lambda r: -2*De*a*torch.exp(-a*(r-re))*(1-torch.exp(-a*(r-re)))
t = torch.linspace(0,1,36); r_tr=(0.95+2.05*t**1.4).unsqueeze(1); V_tr,F_tr=morse(r_tr),mforce(r_tr)
Vs,Vm,rm,rsd=V_tr.std(),V_tr.mean(),r_tr.mean(),r_tr.std()
class PES(nn.Module):
    def __init__(s,h=16): super().__init__(); s.net=nn.Sequential(nn.Linear(1,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,1))
    def forward(s,r): return s.net((r-rm)/rsd)*Vs+Vm
def EF(m,r,cg):
    r=r.clone().requires_grad_(True); V=m(r); d,=torch.autograd.grad(V.sum(),r,create_graph=cg); return V,-d
def run(wF):
    torch.manual_seed(7); m=PES(); o=torch.optim.Adam(m.parameters(),lr=1e-2); sc=torch.optim.lr_scheduler.StepLR(o,2000,0.3)
    hE,hF=[],[]
    for ep in range(6000):
        o.zero_grad(); V,F=EF(m,r_tr,True); lE=((V-V_tr)**2).mean(); lF=((F-F_tr)**2).mean()
        L=lE+wF*lF; L.backward(); o.step(); sc.step(); hE.append(lE.item()); hF.append(lF.item())
    r=torch.linspace(0.95,3.0,2000).unsqueeze(1); V,F=EF(m,r,False)
    eR=((V-morse(r))**2).mean().sqrt().item(); fR=((F-mforce(r))**2).mean().sqrt().item()
    rr=torch.linspace(1.0,1.5,5001).unsqueeze(1)
    with torch.no_grad(): rmin=rr[m(rr).argmin()].reshape(1,1).clone()
    rmin.requires_grad_(True); d1,=torch.autograd.grad(m(rmin).sum(),rmin,create_graph=True); k,=torch.autograd.grad(d1.sum(),rmin)
    w=math.sqrt(k.item()*1.602177e-19/1e-20/(8*1.66054e-27))/(2*math.pi*2.99792458e10)
    return hE,hF,eR,fR,w
res={wF:run(wF) for wF in (0.0,0.1)}
for wF,(hE,hF,eR,fR,w) in res.items(): print(wF,"E",round(eR,4),"F",round(fR,4),"w",round(w))
INK="#14213D"; MUTED="#5A6475"; BG="#F7F5EF"; ORANGE="#E07A2F"; BLUE="#2A78D6"; GRID="#E1DED4"
plt.rcParams.update({"font.size":20,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,"axes.titlecolor":INK,
  "axes.facecolor":BG,"figure.facecolor":BG,"axes.grid":True,"grid.color":GRID,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(figsize=(9,6.2))
hE,hF,*_=res[0.1]
ax.semilogy([x**0.5 for x in hE],color=BLUE,lw=3,label="energy RMSE (eV)")
ax.semilogy([x**0.5 for x in hF],color=ORANGE,lw=3,label="force RMSE (eV/Å)")
for e in (2000,4000): ax.axvline(e,color=MUTED,ls=":",lw=2)
ax.text(2050,ax.get_ylim()[1]*0.5,"learning rate × 0.3",color=MUTED,fontsize=17)
ax.set_xlabel("epoch"); ax.set_ylabel("training error (log scale)"); ax.legend(frameon=False,loc="upper right",bbox_to_anchor=(1.0,0.8),fontsize=18)
ax.set_title("Training on energies + forces (λ = 0.1)")
plt.tight_layout(); plt.savefig("loss_curve.png",dpi=110)
