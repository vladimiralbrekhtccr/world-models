import sys, os, json, numpy as np, torch, torch.nn.functional as F, gsplat
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
sys.path.insert(0,"/scratch/vladimir_albrekht/projects/world-models/phystalk")
import phystalk as ph, interior as I
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer
DEV="cuda"; torch.manual_seed(0)
O="/scratch/vladimir_albrekht/projects/world-models/TripoSplat_proj/interior_objects"
DEMO="/scratch/vladimir_albrekht/projects/world-models/TripoSplat_proj/explain/dwm/demo/splats"
sch=json.load(open(f"{O}/vlm_schemas.json"))
def render_dir(X,Q,S,Op,C,d,r=2.7,RES=200):
    d=d/d.norm(); cam=d*r; up=torch.tensor([0.,0.,1.],device=DEV)
    if abs(float((d*up).sum()))>0.95: up=torch.tensor([0.,1.,0.],device=DEV)
    fw=F.normalize(-cam,dim=0);ri=F.normalize(torch.cross(fw,up,dim=0),dim=0);u2=torch.cross(ri,fw,dim=0)
    Rm=torch.stack([ri,-u2,fw],0);w2c=torch.eye(4,device=DEV);w2c[:3,:3]=Rm;w2c[:3,3]=-Rm@cam
    fx=RES*1.2;K=torch.tensor([[fx,0,RES/2],[0,fx,RES/2],[0,0,1]],device=DEV,dtype=torch.float32)
    out,_,_=gsplat.rasterization(X,Q,S,Op,C,w2c[None],K[None],RES,RES,packed=False)
    return out[0].clamp(0,1).permute(2,0,1)
clip=CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEV).eval()
for p in clip.parameters(): p.requires_grad_(False)
tok=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
CM=torch.tensor([0.4815,0.4578,0.4082],device=DEV).view(1,3,1,1);CS=torch.tensor([0.2686,0.2613,0.2758],device=DEV).view(1,3,1,1)
def img_feat(chw):
    x=F.interpolate(chw.unsqueeze(0),224,mode='bilinear',align_corners=False)
    return F.normalize(clip.get_image_features(pixel_values=(x-CM)/CS),dim=1)
DIRS=[torch.tensor(v,device=DEV,dtype=torch.float32) for v in [[1,0,.15],[.2,1,.1],[.7,.6,.1],[.6,.2,.7],[.1,.5,.9],[-1,0,.1],[-.6,.6,.2]]]
def run(name,prompt):
    with torch.no_grad():
        ti=tok([prompt],return_tensors="pt",padding=True).to(DEV); txt=F.normalize(clip.get_text_features(**ti),dim=1)
    g=ph.load(f"{O}/{name}.ply"); gB=I.arm_from_schema(g,sch[name]); ns=len(g["X"])
    Xs,Ss,Qs,Os,Cs=[gB[k][:ns].detach() for k in ["X","S","Q","O","C"]]
    Xi=gB["X"][ns:].detach(); Qi=gB["Q"][ns:].detach(); Ci0=gB["C"][ns:].detach().clamp(1e-3,1-1e-3)
    col=torch.log(Ci0/(1-Ci0)).clone().requires_grad_(True)
    logS=torch.log(gB["S"][ns:].detach().clamp(1e-6)).clone().requires_grad_(True)
    opl=torch.logit(gB["O"][ns:].detach().clamp(1e-3,1-1e-3)).clone().requires_grad_(True)
    opt=torch.optim.Adam([col,logS,opl],lr=0.02)
    def build(): return (torch.cat([Xs,Xi]),torch.cat([Qs,Qi]),torch.cat([Ss,torch.exp(logS)]),torch.cat([Os,torch.sigmoid(opl)]),torch.cat([Cs,torch.sigmoid(col)]))
    def cut(d,RES=200):
        X,Q,S,Op,C=build(); t=(X@(d/d.norm())); m=(t<=t.median().detach()).nonzero().squeeze(1)
        return render_dir(X[m],Q[m],S[m],Op[m],C[m],d,RES=RES)
    rng=np.random.RandomState(0)
    for it in range(280):
        opt.zero_grad(); loss=0.; pick=[DIRS[i] for i in rng.choice(len(DIRS),3,replace=False)]
        for d in pick: loss=loss+(1-(img_feat(cut(d,200))*txt).sum())
        loss=loss/3; anchor=0.2*((torch.sigmoid(col)-Ci0)**2).mean(); (loss+anchor).backward(); opt.step()
    X,Q,S,Op,C=build(); gg=dict(X=X.detach(),Q=Q.detach(),S=S.detach(),O=Op.detach(),C=C.detach())
    I.export_ply(I.halve(gg,axis=0), f"{DEMO}/{name}_cutB.ply")
    with torch.no_grad():
        tiles=[(cut(d,300).permute(1,2,0).cpu().numpy()*255).astype('uint8') for d in [DIRS[0],DIRS[1],DIRS[2],DIRS[3],DIRS[4]]]
    Image.fromarray(np.concatenate(tiles,1)).save(f"{O}/_opt_saved_{name}.png")
    print(f"exported optimized {name}_cutB.ply (loss {float(loss):.3f})",flush=True)
run("onion","a cross-section of a red onion, concentric purple and white rings")
run("apple","a cross-section of a red apple, pale cream flesh, a small brown core with dark seeds, thin red skin")
print("DONE")
