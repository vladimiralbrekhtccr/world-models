"""PhysTalk (reimpl) — train-free, agent-authored physics on explicit 3DGS.
Capability that surface-smearing CANNOT do: shatter, reveal interior, change material."""
import sys,glob,os,numpy as np,torch,torch.nn.functional as F,gsplat,matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
DEV="cuda";RES=160
from plyfile import PlyData
def load(p):
    d=PlyData.read(p)['vertex'].data;xyz=np.stack([d['x'],d['y'],d['z']],1).astype(np.float32)
    sc=np.stack([d[f'scale_{i}'] for i in range(3)],1).astype(np.float32);rt=np.stack([d[f'rot_{i}'] for i in range(4)],1).astype(np.float32)
    op=np.asarray(d['opacity'],np.float32);fdc=np.stack([d[f'f_dc_{i}'] for i in range(3)],1).astype(np.float32)
    c=np.median(xyz,0);xyz=xyz-c;s=float(np.percentile(np.linalg.norm(xyz,axis=1),95));xyz/=s
    return dict(X=torch.tensor(xyz,device=DEV),S=torch.tensor(np.exp(sc),device=DEV)/s,Q=F.normalize(torch.tensor(rt,device=DEV),dim=1),
                O=torch.sigmoid(torch.tensor(op,device=DEV)),C=torch.tensor(np.clip(0.2820948*fdc+0.5,0,1),device=DEV))
def render(g,az=0.6,elev=0.32):
    r=2.7;cam=torch.tensor([r*np.cos(elev)*np.sin(az),r*np.sin(elev),r*np.cos(elev)*np.cos(az)],device=DEV,dtype=torch.float32)
    up=torch.tensor([0.,0.,1.],device=DEV);fw=F.normalize(-cam,dim=0);ri=F.normalize(torch.cross(fw,up,dim=0),dim=0);u2=torch.cross(ri,fw,dim=0)
    Rm=torch.stack([ri,-u2,fw],0);w2c=torch.eye(4,device=DEV);w2c[:3,:3]=Rm;w2c[:3,3]=-Rm@cam
    fx=RES*1.2;K=torch.tensor([[fx,0,RES/2],[0,fx,RES/2],[0,0,1]],device=DEV,dtype=torch.float32)
    out,_,_=gsplat.rasterization(g["X"],g["Q"],g["S"],g["O"],g["C"],w2c[None],K[None],RES,RES,packed=False);return out[0].clamp(0,1).cpu().numpy()
def kmeans(P,k,it=18):
    c=P[torch.randperm(len(P),device=DEV)[:k]].clone()
    for _ in range(it):
        lab=torch.cdist(P,c).argmin(1)
        for j in range(k):
            m=lab==j
            if m.sum()>0:c[j]=P[m].mean(0)
    return lab,c
# ---------- C2: solidify — fill the hollow interior via voxel flood-fill ----------
def solidify(g,res=56,shade=0.62,structured=False):
    X=g["X"];lo=X.min(0).values;hi=X.max(0).values;ext=(hi-lo);lo=lo-ext*0.04;hi=hi+ext*0.04;ext=hi-lo
    vi=((X-lo)/ext*res).long().clamp(0,res-1)
    occ=torch.zeros((res,res,res),dtype=torch.bool,device=DEV);occ[vi[:,0],vi[:,1],vi[:,2]]=True
    free=~occ
    def cmax(o,dim):return torch.cummax(o,dim=dim).values
    rx1=cmax(occ,0);rx2=torch.flip(cmax(torch.flip(occ,[0]),0),[0])
    ry1=cmax(occ,1);ry2=torch.flip(cmax(torch.flip(occ,[1]),1),[1])
    rz1=cmax(occ,2);rz2=torch.flip(cmax(torch.flip(occ,[2]),2),[2])
    interior=free&rx1&rx2&ry1&ry2&rz1&rz2
    idx=interior.nonzero()
    if len(idx)==0:return g,0
    cen=lo+(idx.float()+0.5)/res*ext;m=len(cen);vox=float((ext/res).mean())
    ids=torch.randperm(len(X),device=DEV)[:8000];sub=X[ids];subC=g["C"][ids]
    D=torch.cdist(cen,sub);nn=D.argmin(1);nd=D.min(1).values;dn=(nd/(nd.max()+1e-6)).clamp(0,1)
    if structured:                                  # P3 probe: surface-tinted SHELL -> warm dark CORE (radial layers)
        shell=subC[nn]*0.72;core=torch.tensor([0.34,0.27,0.20],device=DEV)
        col=(shell*(1-dn[:,None])+core[None]*dn[:,None]).clamp(0,1)
    else:
        col=(subC[nn]*shade*(1-0.4*dn[:,None])).clamp(0,1)
    gi=dict(g)
    gi["X"]=torch.cat([X,cen]);gi["S"]=torch.cat([g["S"],torch.full((m,3),vox*1.1,device=DEV)])
    gi["Q"]=torch.cat([g["Q"],torch.tensor([[1.,0,0,0]],device=DEV).repeat(m,1)])
    gi["O"]=torch.cat([g["O"],torch.full((m,),0.97,device=DEV)]);gi["C"]=torch.cat([g["C"],col])
    return gi,m
# ---------- ops ----------
def crush(g,t):
    X=g["X"].clone();S=g["S"].clone();z=X[:,2];zmin=z.min();X[:,2]=zmin+(z-zmin)*(1-0.7*t)
    cx,cy=X[:,0].mean(),X[:,1].mean();X[:,0]=cx+(X[:,0]-cx)*(1+0.5*t);X[:,1]=cy+(X[:,1]-cy)*(1+0.5*t)
    S[:,2]*=(1-0.7*t);S[:,0]*=(1+0.5*t);S[:,1]*=(1+0.5*t);g2=dict(g);g2["X"]=X;g2["S"]=S;return g2
def shatter(g,t,K=14,seed=0):
    X=g["X"];lab,cen=kmeans(X,K);oc=X.mean(0);rng=torch.Generator(device=DEV).manual_seed(seed);grav=torch.tensor([0.,0.,-1.6],device=DEV);Xn=X.clone()
    for k in range(K):
        m=lab==k
        if m.sum()==0:continue
        d=cen[k]-oc;d=d/(d.norm()+1e-6);v=d*1.3+0.5*torch.randn(3,generator=rng,device=DEV);v[2]=v[2].abs()*0.6+0.4;Xn[m]=X[m]+(v*1.7*t+0.5*grav*(1.7*t)**2)[None,:]
    g2=dict(g);g2["X"]=Xn;return g2
def slice_cut(g,t,axis=1,gap=0.7):                 # split along a plane and part the two halves (interior shows)
    X=g["X"];m=X[:,axis]>X[:,axis].median();Xn=X.clone();Xn[m,axis]+=gap*t;Xn[~m,axis]-=gap*t;g2=dict(g);g2["X"]=Xn;return g2
def author_physics(g,prompt):
    p=prompt.lower()
    if any(w in p for w in["cut","slice","halve","saw"]):return lambda t:slice_cut(g,t)
    if any(w in p for w in["shatter","break","smash","drop","throw"]):return lambda t:shatter(g,t)
    return lambda t:crush(g,t)
# ---------- demos ----------
def demo_c2():
    objs=sorted(glob.glob("/scratch/vladimir_albrekht/projects/world-models/TripoSplat_proj/objects_maxq/*.ply"))
    g=load([o for o in objs if "ceramic_teapot" in o][0])
    gs,m=solidify(g);print("interior gaussians added:",m)
    ts=[0.0,0.5,1.0]
    fig,ax=plt.subplots(2,len(ts),figsize=(len(ts)*2.4,4.8))
    for j,t in enumerate(ts):
        ax[0,j].imshow(render(slice_cut(g,t)));ax[0,j].axis("off");ax[0,j].set_title(f"hollow shell t={t}",fontsize=8)
        ax[1,j].imshow(render(slice_cut(gs,t)));ax[1,j].axis("off");ax[1,j].set_title(f"solidified t={t}",fontsize=8)
    fig.suptitle(f"C2 interior-fill: slice a hollow shell (top) vs a solidified object ({m} interior gaussians, bottom)",fontsize=9)
    fig.tight_layout();fig.savefig("/scratch/vladimir_albrekht/projects/world-models/phystalk/c2_interior.png",dpi=110);print("saved c2_interior.png")
# ---------- C3: AGENT-AUTHORED programs (written by an LLM agent from the prompts
# "smash the teapot into pieces" and "cut the teapot in half so I can see inside") ----------
def author_smash(g):
    gf, _ = solidify(g)
    def prog(t):
        return shatter(gf, t, K=14, seed=0)
    return prog
def author_slice(g):
    gf, _ = solidify(g)
    def prog(t):
        return slice_cut(gf, t, axis=1, gap=0.7)
    return prog
def demo_c3():
    objs=sorted(glob.glob("/scratch/vladimir_albrekht/projects/world-models/TripoSplat_proj/objects_maxq/*.ply"))
    g=load([o for o in objs if "ceramic_teapot" in o][0])
    progs=[("agent: smash->pieces",author_smash(g)),("agent: slice->see inside",author_slice(g))]
    ts=[0.0,0.5,1.0]
    fig,ax=plt.subplots(2,len(ts),figsize=(len(ts)*2.4,4.8))
    for i,(name,prog) in enumerate(progs):
        for j,t in enumerate(ts):
            ax[i,j].imshow(render(prog(t)));ax[i,j].axis("off");ax[i,j].set_title((name+f"  t={t}") if j==0 else f"t={t}",fontsize=8)
    fig.suptitle("C3 PhysTalk core: an AGENT wrote these programs from text prompts (solidify->shatter / solidify->slice)",fontsize=9)
    fig.tight_layout();fig.savefig("/scratch/vladimir_albrekht/projects/world-models/phystalk/c3_authored.png",dpi=110);print("saved c3_authored.png")

def drop(g,t,dist=0.8):
    X=g["X"].clone();fall=min(0.5*4.0*t*t,dist);X[:,2]-=fall
    if 0.5*4.0*t*t>=dist:                      # landed on the floor -> settle squash (stays in frame)
        z=X[:,2];zmin=z.min();X[:,2]=zmin+(z-zmin)*0.85
    g2=dict(g);g2["X"]=X;return g2
def explode(g,t,seed=1):
    X=g["X"];oc=X.mean(0);d=X-oc;dn=d/(d.norm(dim=1,keepdim=True)+1e-6)
    rng=torch.Generator(device=DEV).manual_seed(seed);jit=0.3*torch.randn(X.shape,generator=rng,device=DEV)
    g2=dict(g);g2["X"]=X+(dn*1.4+jit)*t;return g2
def melt(g,t):
    X=g["X"].clone();z=X[:,2];zmin=z.min();zr=(z-zmin);X[:,2]=zmin+zr*(1-0.92*t)   # strong vertical collapse
    cx,cy=X[:,0].mean(),X[:,1].mean();spread=1+2.2*t*(zr/(zr.max()+1e-6))           # wide pool, top spreads most
    X[:,0]=cx+(X[:,0]-cx)*spread;X[:,1]=cy+(X[:,1]-cy)*spread;g2=dict(g);g2["X"]=X;return g2
def demo_c4():
    objs=sorted(glob.glob("/scratch/vladimir_albrekht/projects/world-models/TripoSplat_proj/objects_maxq/*.ply"))
    g=load([o for o in objs if "ceramic_teapot" in o][0]);gs,m=solidify(g);print("textured interior gaussians:",m)
    top=[("textured slice t=0",slice_cut(gs,0.0)),("textured slice t=0.5",slice_cut(gs,0.5)),("textured slice t=1",slice_cut(gs,1.0))]
    bot=[("drop t=1",drop(gs,1.0)),("explode t=1",explode(gs,1.0)),("melt t=1",melt(g,1.0))]
    fig,ax=plt.subplots(2,3,figsize=(3*2.4,4.8))
    for j,(n,gg) in enumerate(top):ax[0,j].imshow(render(gg));ax[0,j].axis("off");ax[0,j].set_title(n,fontsize=8)
    for j,(n,gg) in enumerate(bot):ax[1,j].imshow(render(gg));ax[1,j].axis("off");ax[1,j].set_title(n,fontsize=8)
    fig.suptitle("C4: textured interior (top, nearest-surface colour + depth) + new ops drop/explode/melt (bottom)",fontsize=9)
    fig.tight_layout();fig.savefig("/scratch/vladimir_albrekht/projects/world-models/phystalk/c4_ops.png",dpi=110);print("saved c4_ops.png")

def demo_c5():
    base="/scratch/vladimir_albrekht/projects/world-models/TripoSplat_proj/objects_maxq/"
    plan=[("ceramic_vase","shatter"),("dslr_camera","slice"),("sneaker_shoe","melt"),
          ("table_lamp","explode"),("acoustic_guitar","slice"),("ceramic_teapot","shatter")]
    ts=[0.0,0.5,1.0];rows=len(plan)
    fig,ax=plt.subplots(rows,3,figsize=(3*2.1,rows*2.1))
    for i,(name,op) in enumerate(plan):
        g=load(base+name+".ply")
        if op in("shatter","slice"):g,_=solidify(g)
        prog={"shatter":lambda t,g=g:shatter(g,t),"slice":lambda t,g=g:slice_cut(g,t),
              "melt":lambda t,g=g:melt(g,t),"explode":lambda t,g=g:explode(g,t)}[op]
        for j,t in enumerate(ts):
            ax[i,j].imshow(render(prog(t)));ax[i,j].axis("off")
            ax[i,j].set_title((f"{name}: {op}  t={t}") if j==0 else f"t={t}",fontsize=7)
        print("rendered",name,op,flush=True)
    fig.suptitle("C5 gallery: agent-authored, train-free physics on DISTINCT explicit-3DGS objects (shatter / slice-reveal-interior / melt / explode)",fontsize=9)
    fig.tight_layout();fig.savefig("/scratch/vladimir_albrekht/projects/world-models/phystalk/c5_gallery.png",dpi=108);print("saved c5_gallery.png")

def demo_hero():
    base="/scratch/vladimir_albrekht/projects/world-models/TripoSplat_proj/objects_maxq/"
    gt,_=solidify(load(base+"ceramic_teapot.ply"));gc,_=solidify(load(base+"dslr_camera.ply"));gl=load(base+"table_lamp.ply")
    rows=[("teapot: shatter",[shatter(gt,t) for t in (0.0,0.5,1.0)]),
          ("camera: slice -> interior",[slice_cut(gc,t) for t in (0.0,0.5,1.0)]),
          ("lamp: explode",[explode(gl,t) for t in (0.0,0.5,1.0)])]
    fig,ax=plt.subplots(3,3,figsize=(7.6,7.8))
    for i,(name,frames) in enumerate(rows):
        for j,fr in enumerate(frames):
            ax[i,j].imshow(render(fr));ax[i,j].axis("off");ax[i,j].set_title((name+f"   t={[0.0,0.5,1.0][j]}") if j==0 else f"t={[0.0,0.5,1.0][j]}",fontsize=8,loc="left")
    fig.suptitle("PhysTalk reimpl — train-free, agent-authored physics on explicit 3DGS\nthings surface-smearing can't do: shatter into SOLID pieces · slice to reveal INTERIOR · explode",fontsize=10)
    fig.tight_layout();fig.savefig("/scratch/vladimir_albrekht/projects/world-models/phystalk/hero.png",dpi=120);print("saved hero.png")

def demo_p2():
    base="/scratch/vladimir_albrekht/projects/world-models/TripoSplat_proj/objects_maxq/"
    g=load(base+"ceramic_teapot.ply")
    rows=[("drop -> land",[drop(g,t) for t in (0.0,0.5,1.0)]),("melt (stronger)",[melt(g,t) for t in (0.0,0.5,1.0)])]
    fig,ax=plt.subplots(2,3,figsize=(3*2.3,4.6))
    for i,(n,frames) in enumerate(rows):
        for j,fr in enumerate(frames):ax[i,j].imshow(render(fr));ax[i,j].axis("off");ax[i,j].set_title((n+f"  t={[0.0,0.5,1.0][j]}") if j==0 else f"t={[0.0,0.5,1.0][j]}",fontsize=8)
    fig.suptitle("P2: drop now lands in-frame; melt collapses + pools wider",fontsize=9)
    fig.tight_layout();fig.savefig("/scratch/vladimir_albrekht/projects/world-models/phystalk/p2_ops.png",dpi=110);print("saved p2_ops.png")

def demo_p3():
    base="/scratch/vladimir_albrekht/projects/world-models/TripoSplat_proj/objects_maxq/"
    g=load(base+"ceramic_vase.ply")
    g1,_=solidify(g,structured=False);g2,_=solidify(g,structured=True)
    fig,ax=plt.subplots(2,3,figsize=(3*2.3,4.6))
    for j,t in enumerate((0.0,0.5,1.0)):
        ax[0,j].imshow(render(slice_cut(g1,t)));ax[0,j].axis("off");ax[0,j].set_title(("flat interior  t=%.1f"%t) if j==0 else "t=%.1f"%t,fontsize=8)
        ax[1,j].imshow(render(slice_cut(g2,t)));ax[1,j].axis("off");ax[1,j].set_title(("structured: shell->core  t=%.1f"%t) if j==0 else "t=%.1f"%t,fontsize=8)
    fig.suptitle("P3 interior-grounding PROBE: flat (top) vs radial shell->core (bottom) -- still a GUESS, not the real inside (= the paper wedge)",fontsize=8)
    fig.tight_layout();fig.savefig("/scratch/vladimir_albrekht/projects/world-models/phystalk/interior_v2.png",dpi=110);print("saved interior_v2.png")

if __name__=="__main__":
    {"c2":demo_c2,"c3":demo_c3,"c4":demo_c4,"c5":demo_c5,"hero":demo_hero,"p2":demo_p2,"p3":demo_p3}.get(sys.argv[1] if len(sys.argv)>1 else "p3",demo_p3)()
