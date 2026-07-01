# Structure-vs-texture go/no-go: does a VLM-authored STRUCTURED interior,
# revealed by a cut, beat a nearest-surface TEXTURE fill (FruitNinja-lite)?
# Arm A = texture-fill.  Arm B = world-knowledge schema (hand-written here as
# a stand-in for the agent's output; the agentic wrapper comes after the visual holds).
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,"/scratch/vladimir_albrekht/projects/world-models/phystalk")
import phystalk as ph
from phystalk import DEV, load, render
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0); ph.RES=240
O="/scratch/vladimir_albrekht/projects/world-models/TripoSplat_proj/interior_objects"

def interior_geom(g,res=80):
    X=g["X"];lo=X.min(0).values;hi=X.max(0).values;ext=hi-lo;lo=lo-ext*0.04;hi=hi+ext*0.04;ext=hi-lo
    vi=((X-lo)/ext*res).long().clamp(0,res-1)
    occ=torch.zeros((res,res,res),dtype=torch.bool,device=DEV);occ[vi[:,0],vi[:,1],vi[:,2]]=True
    # solid-ish fill: voxel is interior if the surface encloses it along >=2 of 3 axes
    # (leak-proof on a non-watertight shell; fills concavities the strict 3-of-3 rule left as voids)
    free=~occ; cmax=lambda o,d:torch.cummax(o,dim=d).values
    ax=lambda a:(cmax(occ,a)&torch.flip(cmax(torch.flip(occ,[a]),a),[a]))
    enc=ax(0).int()+ax(1).int()+ax(2).int()
    interior=free&(enc>=2); idx=interior.nonzero()
    cen=lo+(idx.float()+0.5)/res*ext; m=len(cen); vox=float((ext/res).mean())
    ids=torch.randperm(len(X),device=DEV)[:8000];sub=X[ids];subC=g["C"][ids]
    D=torch.cdist(cen,sub);nn=D.argmin(1);nd=D.min(1).values;dn=(nd/(nd.max()+1e-6)).clamp(0,1)
    return cen,dn,subC[nn],vox,m

def _rand_quat(m):
    q=torch.randn(m,4,device=DEV); return q/(q.norm(dim=1,keepdim=True)+1e-9)
def assemble(g,cen,col,vox,sizes=None,opac=0.985):
    m=len(cen)
    if sizes is None: sizes=torch.full((m,3),vox*1.05,device=DEV)
    pos=cen+(torch.rand(m,3,device=DEV)-0.5)*vox*0.5          # small jitter off the lattice
    aniso=0.75+0.5*torch.rand(m,3,device=DEV)                 # mild anisotropy (not extreme)
    S=sizes*aniso
    Q=_rand_quat(m)                                           # random orientation (kills axis-aligned popping)
    gi=dict(g)
    gi["X"]=torch.cat([g["X"],pos]);gi["S"]=torch.cat([g["S"],S])
    gi["Q"]=torch.cat([g["Q"],Q]);gi["O"]=torch.cat([g["O"],torch.full((m,),opac,device=DEV)]);gi["C"]=torch.cat([g["C"],col])
    return gi

def arm_texture(g):                       # Arm A: nearest-surface colour, depth-darkened
    cen,dn,nnc,vox,m=interior_geom(g)
    col=(nnc*0.62*(1-0.4*dn[:,None])).clamp(0,1)
    return assemble(g,cen,col,vox)

def arm_watermelon(g):                    # Arm B schema: white rind -> red flesh + black seeds
    cen,dn,nnc,vox,m=interior_geom(g)
    rind=torch.tensor([0.86,0.90,0.78],device=DEV);flesh=torch.tensor([0.82,0.12,0.16],device=DEV)
    t=((dn-0.03)/0.12).clamp(0,1)[:,None]; col=rind[None]*(1-t)+flesh[None]*t
    col=(col*(1-0.10*dn[:,None])).clamp(0,1)
    fm=(dn>0.18).nonzero().squeeze(1); ns=min(240,len(fm))
    sel=fm[torch.randperm(len(fm),device=DEV)[:ns]]
    col[sel]=torch.tensor([0.06,0.05,0.05],device=DEV)
    sizes=torch.full((m,3),vox*1.1,device=DEV); sizes[sel]=vox*1.7
    return assemble(g,cen,col,vox,sizes=sizes)

def arm_onion(g):                         # Arm B schema: concentric purple->pale rings + membranes
    cen,dn,nnc,vox,m=interior_geom(g)
    outer=torch.tensor([0.62,0.26,0.38],device=DEV);inner=torch.tensor([0.95,0.90,0.88],device=DEV)
    col=outer[None]*(1-dn[:,None])+inner[None]*dn[:,None]
    membrane=((torch.sin(dn*np.pi*2*6.0)).abs()>0.86).float()[:,None]
    col=(col*(1-0.42*membrane)).clamp(0,1)
    return assemble(g,cen,col,vox)

def halve(g,axis=0,keep_low=True):
    med=g["X"][:,axis].median(); m=(g["X"][:,axis]<=med) if keep_low else (g["X"][:,axis]>=med)
    idx=m.nonzero().squeeze(1); g2=dict(g)
    for k in ["X","S","Q","O","C"]: g2[k]=g[k][idx]
    return g2

def cut_face(g,axis,az,elev,keep_low=True): return render(halve(g,axis,keep_low),az=az,elev=elev)

def main():
    spec={"watermelon (green skin)":dict(ply="watermelon",armB=arm_watermelon,cut=dict(axis=0,az=1.57,elev=0.12)),
          "red onion (purple skin)":dict(ply="onion",armB=arm_onion,cut=dict(axis=2,az=0.0,elev=1.25))}
    fig,ax=plt.subplots(2,3,figsize=(9.2,6.2))
    for i,(name,s) in enumerate(spec.items()):
        g=load(f"{O}/{s['ply']}.ply"); gA=arm_texture(g); gB=s["armB"](g)
        ax[i,0].imshow(render(g,az=0.4,elev=0.30)); ax[i,0].set_title(f"{name}\nexterior shell",fontsize=8)
        ax[i,1].imshow(cut_face(gA,**s["cut"])); ax[i,1].set_title("CUT — Arm A: texture-fill\n(nearest surface colour)",fontsize=8)
        ax[i,2].imshow(cut_face(gB,**s["cut"])); ax[i,2].set_title("CUT — Arm B: structured\n(VLM world-knowledge)",fontsize=8)
        for j in range(3): ax[i,j].axis("off")
    fig.suptitle("Structure vs Texture: the interior a cut reveals  (Arm A = FruitNinja-lite  |  Arm B = world-knowledge schema)",fontsize=9.5)
    fig.tight_layout(); out=f"{O}/AB_structure_vs_texture.png"; fig.savefig(out,dpi=120); print("saved",out)

if __name__=="__main__": main()

# ---- schema-driven interior (consumes a VLM-authored JSON schema directly) ----
import json as _json
def arm_from_schema(g, schema):
    cen,dn,nnc,vox,m=interior_geom(g)
    col=torch.zeros((m,3),device=DEV); assigned=torch.zeros(m,dtype=torch.bool,device=DEV)
    layers=schema.get("layers",[])
    for L in layers:
        lo,hi=L["depth"]; rgb=torch.tensor(L["rgb"],device=DEV,dtype=torch.float32)
        mask=(dn>=lo)&(dn<=hi+1e-6); col[mask]=rgb; assigned|=mask
    if (~assigned).any() and layers:                     # fill any gap with the deepest layer
        col[~assigned]=torch.tensor(layers[-1]["rgb"],device=DEV,dtype=torch.float32)
    sizes=torch.full((m,3),vox*1.1,device=DEV)
    for inc in schema.get("inclusions",[]):
        lo,hi=inc.get("region_depth",[0.2,0.95]); rgb=torch.tensor(inc["rgb"],device=DEV,dtype=torch.float32)
        region=((dn>=lo)&(dn<=hi)).nonzero().squeeze(1)
        if len(region)==0: continue
        cnt=int(inc.get("count",150)); szmul=float(inc.get("size",1.6))
        if "seed" in str(inc.get("type","")):
            cnt=max(cnt,int(0.008*len(region))); szmul=max(szmul,2.4)  # scale seeds w/ fill density so a cut still shows them
        cnt=min(cnt,len(region))
        sel=region[torch.randperm(len(region),device=DEV)[:cnt]]
        col[sel]=rgb; sizes[sel]=vox*szmul
    return assemble(g,cen,col,vox,sizes=sizes)

def render_from_schema(ply, schema, cut):   # cut=dict(axis,az,elev)
    g=load(ply); gB=arm_from_schema(g,schema)
    return render(g,az=0.4,elev=0.30), cut_face(arm_texture(g),**cut), cut_face(gB,**cut)

# ---- export a gaussian dict to a standard 3DGS (INRIA, SH0) PLY ----
from plyfile import PlyData as _PlyData, PlyElement as _PlyElement
def export_ply(g, path):
    import numpy as _np
    X=g["X"].detach().cpu().numpy().astype(_np.float32)
    S=_np.clip(g["S"].detach().cpu().numpy().astype(_np.float32),1e-8,None)
    Q=g["Q"].detach().cpu().numpy().astype(_np.float32)
    O=_np.clip(g["O"].detach().cpu().numpy().astype(_np.float32),1e-6,1-1e-6)
    C=_np.clip(g["C"].detach().cpu().numpy().astype(_np.float32),1e-6,1-1e-6)
    fdc=(C-0.5)/0.2820948; opa=_np.log(O/(1-O)); sc=_np.log(S)
    n=len(X); dt=[('x','f4'),('y','f4'),('z','f4'),('nx','f4'),('ny','f4'),('nz','f4'),
        ('f_dc_0','f4'),('f_dc_1','f4'),('f_dc_2','f4'),('opacity','f4'),
        ('scale_0','f4'),('scale_1','f4'),('scale_2','f4'),('rot_0','f4'),('rot_1','f4'),('rot_2','f4'),('rot_3','f4')]
    a=_np.zeros(n,dtype=dt)
    a['x'],a['y'],a['z']=X[:,0],X[:,1],X[:,2]
    a['f_dc_0'],a['f_dc_1'],a['f_dc_2']=fdc[:,0],fdc[:,1],fdc[:,2]; a['opacity']=opa
    a['scale_0'],a['scale_1'],a['scale_2']=sc[:,0],sc[:,1],sc[:,2]
    a['rot_0'],a['rot_1'],a['rot_2'],a['rot_3']=Q[:,0],Q[:,1],Q[:,2],Q[:,3]
    _PlyData([_PlyElement.describe(a,'vertex')]).write(path)
