import sys, os, json, numpy as np, torch, torch.nn.functional as F, gsplat
sys.path.insert(0,"/scratch/vladimir_albrekht/projects/world-models/phystalk")
import phystalk as ph, interior as I
from PIL import Image
DEV="cuda"; torch.manual_seed(0)
O="/scratch/vladimir_albrekht/projects/world-models/TripoSplat_proj/interior_objects"
sch=json.load(open(f"{O}/vlm_schemas.json"))
REF="/scratch/vladimir_albrekht/projects/world-models/TripoSplat_proj/interior_inputs/cand/wm_pick.jpg"

def render_dir(X,Q,S,Op,C,d,r=2.7,RES=180):   # cut plane normal = view dir d; look face-on at the cross-section
    d=d/d.norm(); cam=d*r
    up=torch.tensor([0.,0.,1.],device=DEV)
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
Xi=gB["X"][ns:].detach(); Qi=gB["Q"][ns:].detach()
Ci0=gB["C"][ns:].detach().clamp(1e-3,1-1e-3)
col=torch.log(Ci0/(1-Ci0)).clone().requires_grad_(True)
logS=torch.log(gB["S"][ns:].detach().clamp(1e-6)).clone().requires_grad_(True)
opl=torch.logit(gB["O"][ns:].detach().clamp(1e-3,1-1e-3)).clone().requires_grad_(True)
opt=torch.optim.Adam([col,logS,opl],lr=0.03)

# fibonacci-ish set of cut directions covering the sphere
DIRS=[]
for v in [[1,0,.15],[-1,0,.15],[.2,1,.1],[.2,-1,.1],[.8,.6,.1],[-.7,.7,.1],[.7,.2,.7],[.1,.6,.9],[1,0,.9],[.5,.5,.5]]:
    DIRS.append(torch.tensor(v,device=DEV,dtype=torch.float32))
def build():
    return (torch.cat([Xs,Xi]),torch.cat([Qs,Qi]),torch.cat([Ss,torch.exp(logS)]),torch.cat([Os,torch.sigmoid(opl)]),torch.cat([Cs,torch.sigmoid(col)]))
def cut(d,RES=170):
    X,Q,S,Op,C=build(); t=(X@ (d/d.norm())); m=(t<=t.median().detach()).nonzero().squeeze(1)
    return render_dir(X[m],Q[m],S[m],Op[m],C[m],d,RES=RES)
def montage(path,dirs,RES=300):
    with torch.no_grad():
        tiles=[(cut(d,RES).permute(1,2,0).cpu().numpy()*255).astype('uint8') for d in dirs]
    Image.fromarray(np.concatenate(tiles,1)).save(path)
SHOW=[DIRS[0],DIRS[2],DIRS[5],DIRS[6],DIRS[8]]
montage(f"{O}/_cutany_before.png",SHOW)
rng=np.random.RandomState(0)
for it in range(300):
    opt.zero_grad(); loss=0.; pick=[DIRS[i] for i in rng.choice(len(DIRS),3,replace=False)]
    for d in pick: loss=loss+(1-(feat(cut(d,160))*ref_f).sum())
    loss=loss/3; anchor=0.15*((torch.sigmoid(col)-Ci0)**2).mean()
    (loss+anchor).backward(); opt.step()
    if it%50==0: print(f"it{it} loss{float(loss):.3f}",flush=True)
montage(f"{O}/_cutany_after.png",SHOW)
a=np.array(Image.open(f"{O}/_cutany_before.png"));b=np.array(Image.open(f"{O}/_cutany_after.png"))
Image.fromarray(np.concatenate([a,b],0)).save(f"{O}/_cutany_ba.png")
print("saved _cutany_ba.png (TOP row=VLM init, BOTTOM row=optimized; same object cut 5 different directions)")
