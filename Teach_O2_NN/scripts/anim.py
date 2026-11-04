import torch, torch.nn as nn, numpy as np, matplotlib, io
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
De, re, a = 5.21, 1.208, 2.66
morse = lambda r: De*(1-np.exp(-a*(r-re)))**2 - De
t=np.linspace(0,1,36); R=0.95+2.05*t**1.4; Y=morse(R)
xm,xs,ym,ys=1.9,0.6,Y.mean(),Y.std()
X=torch.tensor((R-xm)/xs,dtype=torch.float32).unsqueeze(1); T=torch.tensor((Y-ym)/ys,dtype=torch.float32).unsqueeze(1)
torch.manual_seed(3)
H=8; l1,l2,l3=nn.Linear(1,H),nn.Linear(H,H),nn.Linear(H,1); act=nn.GELU()
params=list(l1.parameters())+list(l2.parameters())+list(l3.parameters())
opt=torch.optim.Adam(params,lr=0.01)
rs=np.linspace(0.9,3.05,300); xg=torch.tensor((rs-xm)/xs,dtype=torch.float32).unsqueeze(1)
INK="#14213D"; MUTED="#5A6475"; BG="#F7F5EF"; ORANGE="#E07A2F"; BLUE="#2A78D6"; GREEN="#1BAF7A"; GRID="#E1DED4"
plt.rcParams.update({"font.size":22,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,
  "axes.facecolor":BG,"figure.facecolor":BG,"axes.grid":True,"grid.color":GRID,"axes.spines.top":False,"axes.spines.right":False})
snap=sorted(set([0]+[int(round(x)) for x in np.geomspace(1,5000,110)]))
frames=[]
def frame(ep,loss):
    with torch.no_grad():
        h2=act(l2(act(l1(xg)))); out=(l3(h2)*ys+ym).squeeze().numpy()
        pieces=(h2*l3.weight.squeeze()*ys).numpy()
    fig,ax=plt.subplots(figsize=(11,6.4),dpi=100)
    for j in range(H): ax.plot(rs,pieces[:,j],color=GREEN,alpha=0.5,lw=1.6,label="neuron pieces" if j==0 else None)
    ax.plot(rs,morse(rs),"--",color=MUTED,lw=2.5,label="Morse")
    ax.plot(rs,out,color=BLUE,lw=4,label="network")
    ax.plot(R,Y,"o",color=ORANGE,ms=8,label="data")
    ax.set_xlim(0.9,3.05); ax.set_ylim(-6,4); ax.set_xlabel("O–O distance r (Å)"); ax.set_ylabel("V (eV)")
    ax.text(0.985,0.95,f"epoch {ep:,}",transform=ax.transAxes,ha="right",va="top",fontsize=26,fontweight="bold",color=INK)
    ax.text(0.985,0.86,"random weights" if loss is None else f"RMSE {np.sqrt(loss)*ys:.3f} eV",transform=ax.transAxes,ha="right",va="top",fontsize=22,color=MUTED)
    h,l=ax.get_legend_handles_labels(); order=[2,3,1,0]
    ax.legend([h[i] for i in order],[l[i] for i in order],loc="lower right",ncol=4,frameon=False,fontsize=17,handlelength=1.6,columnspacing=1.2)
    plt.tight_layout(); buf=io.BytesIO(); plt.savefig(buf,format="png"); plt.close(fig)
    buf.seek(0); return Image.open(buf).convert("RGB")
loss=None
for ep in range(0,5001):
    if ep in snap: frames.append(frame(ep,loss))
    opt.zero_grad(); L=((l3(act(l2(act(l1(X)))))-T)**2).mean(); L.backward(); opt.step(); loss=L.item()
print("frames",len(frames),"final rmse eV",np.sqrt(loss)*ys)
frames += [frames[-1]]*30          # hold ~2 s at 15 fps
import os; os.makedirs("frames",exist_ok=True)
for i,f in enumerate(frames): f.save(f"frames/f{i:04d}.png")
# GIF with a shared palette
pal=frames[-1].quantize(colors=128,method=Image.Quantize.MEDIANCUT)
q=[f.quantize(palette=pal,dither=Image.Dither.NONE) for f in frames]
q[0].save("O2_GELU_training.gif",save_all=True,append_images=q[1:],duration=66,loop=0,optimize=True)
