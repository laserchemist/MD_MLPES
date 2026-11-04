import torch, torch.nn as nn, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
De, re, a = 5.21, 1.208, 2.66
morse=lambda r: De*(1-np.exp(-a*(r-re)))**2-De
t=torch.linspace(0,1,36); R=0.95+2.05*t**1.4; e=torch.exp(-a*(R-re))
V_ref=(De*(1-e)**2-De).reshape(-1,1); F_ref=(-2*De*a*e*(1-e)).reshape(-1,1)
class PES(nn.Module):
    def __init__(s,h=16): super().__init__(); s.net=nn.Sequential(nn.Linear(1,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,1))
    def forward(s,r): return s.net((r-1.81)/0.64)*1.83-2.15
models=[]
for seed in (7,8,9):
    torch.manual_seed(seed); m=PES(); opt=torch.optim.Adam(m.parameters(),lr=0.01); r=R.reshape(-1,1)
    for ep in range(6000):
        opt.zero_grad(); r.requires_grad_(True); V=m(r); d,=torch.autograd.grad(V.sum(),r,create_graph=True)
        L=((V-V_ref)**2).mean()+0.1*((-d-F_ref)**2).mean(); L.backward(); opt.step()
        if ep==2000: opt.param_groups[0]["lr"]=0.003
        if ep==4000: opt.param_groups[0]["lr"]=0.001
    models.append(m.eval())
def pred(x):
    with torch.no_grad(): return np.stack([m(torch.tensor(x,dtype=torch.float32).reshape(-1,1)).squeeze().numpy() for m in models])
for rv in [0.3,0.5,0.7,1.208,2.0,3.0,5.0,10.0,20.0]:
    p=pred(np.array([rv,rv]))[:,0]; print(f"{rv:5.2f} morse {morse(rv):8.2f}  nets {np.round(p,2)}  spread {p.max()-p.min():.2f}")
INK="#14213D"; MUTED="#5A6475"; BG="#F7F5EF"; ORANGE="#E07A2F"; BLUE="#2A78D6"; GRID="#E1DED4"; SHADE="#E6E1D3"
plt.rcParams.update({"font.size":20,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,"axes.titlecolor":INK,
  "axes.facecolor":BG,"figure.facecolor":BG,"axes.grid":True,"grid.color":GRID,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(1,2,figsize=(17,5.8),gridspec_kw={"width_ratios":[1,1.35]})
cols=[BLUE,"#6C5CE0","#1B9E8A"]
for k,(lo,hi,yl,title) in enumerate([(0.3,1.3,(-10,400),"Short range: atoms pushed together"),(0.9,12,(-7,30),"Long range: atoms pulled apart")]):
    x=np.linspace(lo,hi,800); P=pred(x)
    ax[k].axvspan(0.95,3.0,color=SHADE,lw=0,zorder=0)
    ax[k].plot(x,morse(x),"--",color=INK,lw=2.5,label="Morse (truth)")
    for i in range(3): ax[k].plot(x,P[i],color=cols[i],lw=3,label=f"network, seed {i+1}")
    ax[k].set_xlim(lo,hi); ax[k].set_ylim(*yl); ax[k].set_title(title); ax[k].set_xlabel("r (Å)"); ax[k].set_ylabel("V (eV)")
ax[0].text(1.0,360,"training\nrange →",color=MUTED,fontsize=18,va="top")
ax[1].text(1.05,26,"training\nrange",color=MUTED,fontsize=18)
ax[1].legend(frameon=False,fontsize=17,loc="upper left",bbox_to_anchor=(0.3,1.0))
plt.tight_layout(); plt.savefig("outside.png",dpi=110)
