import sys, numpy as np
sys.path.insert(0,"/scratch/vladimir_albrekht/projects/world-models/phystalk")
import phystalk as ph
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ph.RES=300
S="/scratch/vladimir_albrekht/projects/world-models/TripoSplat_proj/explain/dwm/demo/splats"
objs=[("watermelon","Watermelon"),("onion","Red onion"),("apple","Apple")]
fig,ax=plt.subplots(3,3,figsize=(9,9))
for i,(k,name) in enumerate(objs):
    gw=ph.load(f"{S}/{k}_whole.ply"); ga=ph.load(f"{S}/{k}_cutA.ply"); gb=ph.load(f"{S}/{k}_cutB.ply")
    ax[i,0].imshow(ph.render(gw,az=0.4,elev=0.30)); ax[i,0].set_ylabel(name,fontsize=11)
    ax[i,1].imshow(ph.render(ga,az=1.57,elev=0.12))
    ax[i,2].imshow(ph.render(gb,az=1.57,elev=0.12))
    for j in range(3): ax[i,j].set_xticks([]); ax[i,j].set_yticks([])
ax[0,0].set_title("exterior shell",fontsize=10)
ax[0,1].set_title("cut — texture-fill\n(baseline, wrong)",fontsize=10)
ax[0,2].set_title("cut — VLM-authored\n+ metric-optimized",fontsize=10)
fig.suptitle("Give the splat an inside — VLM authors the structure, a metric loop fits the look",fontsize=11)
fig.tight_layout(); fig.savefig("/scratch/vladimir_albrekht/projects/world-models/TripoSplat_proj/explain/dwm/img/fig1_v2.png",dpi=115)
print("saved fig1_v2.png")
