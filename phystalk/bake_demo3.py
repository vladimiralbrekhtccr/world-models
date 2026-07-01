import sys, json, os
sys.path.insert(0,"/scratch/vladimir_albrekht/projects/world-models/phystalk")
import phystalk as ph, interior as I
O="/scratch/vladimir_albrekht/projects/world-models/TripoSplat_proj/interior_objects"
D="/scratch/vladimir_albrekht/projects/world-models/TripoSplat_proj/explain/dwm/demo/splats"; os.makedirs(D,exist_ok=True)
sch=json.load(open(f"{O}/vlm_schemas.json")); AX=0
for name in ["watermelon","onion","apple"]:
    g=ph.load(f"{O}/{name}.ply")
    I.export_ply(g, f"{D}/{name}_whole.ply")
    I.export_ply(I.halve(I.arm_texture(g),axis=AX),               f"{D}/{name}_cutA.ply")
    I.export_ply(I.halve(I.arm_from_schema(g,sch[name]),axis=AX), f"{D}/{name}_cutB.ply")
    print("baked",name,flush=True)
print("DONE")
