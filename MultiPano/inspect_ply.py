import sys, numpy as np
from plyfile import PlyData
ply = PlyData.read(sys.argv[1])
v = ply['vertex']
xyz = np.stack([v['x'], v['y'], v['z']], axis=-1)
print(f"N = {len(xyz):,}")
for i, ax in enumerate("XYZ"):
    lo, hi = xyz[:, i].min(), xyz[:, i].max()
    print(f"bbox {ax}: [{lo:.2f}, {hi:.2f}]  span {hi-lo:.2f}")
print(f"centroid: ({xyz[:,0].mean():.2f}, {xyz[:,1].mean():.2f}, {xyz[:,2].mean():.2f})")
print(f"scale_0 mean: {v['scale_0'].mean():.3f}   exp = {np.exp(v['scale_0'].mean()):.3f} m")
print(f"opacity (pre-sigmoid) mean: {v['opacity'].mean():.3f}   sigmoid = {1/(1+np.exp(-v['opacity'].mean())):.3f}")
