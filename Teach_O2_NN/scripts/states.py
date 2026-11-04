import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
# Spectroscopic constants (Huber & Herzberg): Te (cm^-1), we (cm^-1), re (Å)
mu=7.9975*1.66054e-27; c=2.99792458e10
D_X=5.21  # eV, well depth used throughout the deck
states=[("X ³Σg⁻  (ground, triplet)",0.0,1580.2,1.2075,"#2A78D6","-"),
        ("a ¹Δg  (singlet, 0.98 eV)",7918.1,1483.5,1.2156,"#E07A2F","--"),
        ("b ¹Σg⁺  (singlet, 1.63 eV)",13195.1,1432.8,1.2269,"#6C5CE0","-.")]
r=np.linspace(0.95,3.5,600)
INK="#14213D"; MUTED="#5A6475"; BG="#F7F5EF"; GRID="#E1DED4"
plt.rcParams.update({"font.size":20,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,
  "axes.facecolor":BG,"figure.facecolor":BG,"axes.grid":True,"grid.color":GRID,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(figsize=(10.5,6.6))
for name,Te,we,re,col,ls in states:
    Te_eV=Te/8065.544; De=D_X-Te_eV
    a=we*2*np.pi*c*np.sqrt(mu/(2*De*1.602177e-19))*1e-10   # 1/Å
    V=De*(1-np.exp(-a*(r-re)))**2 - D_X + Te_eV   # minimum at Te above X, all → 0 at O(³P)+O(³P)
    ax.plot(r,V,color=col,lw=3.5,ls=ls,label=name)
    print(name, "De=%.2f eV a=%.2f 1/Å"%(De,a))
ax.axhline(0,color=MUTED,lw=1); ax.text(3.45,0.15,"O(³P) + O(³P)",ha="right",color=MUTED,fontsize=18)
ax.axvspan(0.95,3.0,color="#E6E1D3",alpha=0.5,lw=0); ax.text(1.55,1.2,"our training range",color=MUTED,fontsize=17)
ax.set_xlim(0.95,3.5); ax.set_ylim(-5.6,1.8); ax.set_xlabel("O–O distance r (Å)"); ax.set_ylabel("V (eV)")
ax.legend(frameon=False,loc="lower right",fontsize=18)
plt.tight_layout(); plt.savefig("o2_states.png",dpi=110)
