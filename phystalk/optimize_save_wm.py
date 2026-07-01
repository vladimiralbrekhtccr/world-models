import sys, os, json, numpy as np, torch, torch.nn.functional as F, gsplat
sys.path.insert(0,"/scratch/vladimir_albrekht/projects/world-models/phystalk")
import phystalk as ph, interior as I
from PIL import Image
DEV="cuda"; torch.manual_seed(0)
O="/scratch/vladimir_albrekht/projects/world-models/TripoSplat_proj/interior_objects"
DEMO="/scratch/vladimir_albrekht/projects/world-models/TripoSplat_proj/explain/dwm/demo/splats"
sch=json.load(open(f"{O}/vlm_schemas.json"))
REF="/scratch/vladimir_albrekht/projects/world-models/TripoSplat_proj/interior_inputs/cand/wm_pick.jpg"
def render_dir(X,Q,S,Op,C,d,r=2.7,RES=200):
    d=d/d.norm(); cam=d*r; up=torch.tensor([0.,0.,1.],device=DEV)
    if abs(float((d*up).sum()))>0.95: up=torch.tensor([0.,1.,0.],device=DEV)
    fw=F.normalize(-cam,dim=0);ri=F.normalize(torch.cross(fw,up,dim=0),dim=0);u2=torch.cross(ri,fw,dim=0)
    Rm=torch.stack([ri,-u2,fw],0);w2c=torch.eye(4,device=DEV);w2c[:3,:3]=Rm;w2c[:3,3]=-Rm@cam
    fx=RES*1.2;K=torch.tensor([[fx,0,RES/2],[0,fx,RES/2],[0,0,1]],device=DEV,dtype=torch.float32)
    out,_,_=gsplat.rasterization(X,Q,S,Op,C,w2c[None],K[None],RES,RES,packed=False)
    return out[0].clamp(0,1).permute(2,0,1)
dino=torch.hub.load(os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main"),"dinov2_vitl14_reg",source="local",pretrained=True).to(DEV).eval()
for p in dino.parameters(): p.requires_grad_(False)
MN=torch.tensor([0.485,0.456,0.406],device=DEV).view(1,3,1,1);SD=torch.tensor([0.229,0.224,0.225],device=DEV).view(1,3,1,1)
def feat(chw):
    x=F.interpolate(chw.unsqueeze(0),224,mode='bilinear',align_corners=False); return F.normalize(dino((x-MN)/SD),dim=1)
ref_img=torch.tensor(np.array(Image.open(REF).convert("RGB").resize((224,224)))/255.,dtype=torch.float32,device=DEV).permute(2,0,1)
with torch.no_grad(): ref_f=feat(ref_img)
g=ph.load(f"{O}/watermelon.ply"); gB=I.arm_from_schema(g,sch["watermelon"]); ns=len(g["X"])
Xs,Ss,Qs,Os,Cs=[gB[k][:ns].detach() for k in ["X","S","Q","O","C"]]
Xi=gB["X"][ns:].detach(); Qi=gB["Q"][ns:].detach(); Ci0=gB["C"][ns:].detach().clamp(1e-3,1-1e-3)
col=torch.log(Ci0/(1-Ci0)).clone().requires_grad_(True)
logS=torch.log(gB["S"][ns:].detach().clamp(1e-6)).clone().requires_grad_(True)
opl=torch.logit(gB["O"][ns:].detach().clamp(1e-3,1-1e-3)).clone().requires_grad_(True)
opt=torch.optim.Adam([col,logS,opl],lr=0.03)
DIRS=[torch.tensor(v,device=DEV,dtype=torch.float32) for v in [[1,0,.15],[.2,1,.1],[.7,.6,.1],[.6,.2,.7],[.1,.5,.9],[-1,0,.1],[-.6,.6,.2]]]
def build(): return (torch.cat([Xs,Xi]),torch.cat([Qs,Qi]),torch.cat([Ss,torch.exp(logS)]),torch.cat([Os,torch.sigmoid(opl)]),torch.cat([Cs,torch.sigmoid(col)]))
def cut(d,RES=180):
    X,Q,S,Op,C=build(); t=(X@(d/d.norm())); m=(t<=t.median().detach()).nonzero().squeeze(1)
    return render_dir(X[m],Q[m],S[m],Op[m],C[m],d,RES=RES)
rng=np.random.RandomState(0)
for it in range(300):
    opt.zero_grad(); loss=0.; pick=[DIRS[i] for i in rng.choice(len(DIRS),3,replace=False)]
    for d in pick: loss=loss+(1-(feat(cut(d,180))*ref_f).sum())
    loss=loss/3; anchor=0.15*((torch.sigmoid(col)-Ci0)**2).mean()
    (loss+anchor).backward(); opt.step()
    if it%60==0: print(f"it{it} loss{float(loss):.3f}",flush=True)
# export optimized halved object for the demo
X,Q,S,Op,C=build()
gg=dict(X=X.detach(),Q=Q.detach(),S=S.detach(),O=Op.detach(),C=C.detach())
h=I.halve(gg,axis=0)
I.export_ply(h, f"{DEMO}/watermelon_cutB.ply")
# verify montage (5 dirs)
with torch.no_grad():
    tiles=[(cut(d,300).permute(1,2,0).cpu().numpy()*255).astype('uint8') for d in [DIRS[0],DIRS[1],DIRS[2],DIRS[3],DIRS[4]]]
Image.fromarray(np.concatenate(tiles,1)).save(f"{O}/_opt_saved_wm.png")
print("exported optimized watermelon_cutB.ply + saved _opt_saved_wm.png; gaussians in cutB:",len(h["X"]))
