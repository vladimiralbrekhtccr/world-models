"""Extract a textured triangle mesh from a splatfacto-trained PLY.

Approach: treat the gaussians' (means, colors) as a colored point cloud,
estimate normals from local neighbourhoods, run Poisson surface
reconstruction. Outputs GLB for browser viewing.
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData

SH_C0 = 0.28209479177387814


def load_splat(ply_path: Path, min_opacity: float):
    v = PlyData.read(str(ply_path))["vertex"].data
    op = 1.0 / (1.0 + np.exp(-v["opacity"]))
    mask = op > min_opacity
    v = v[mask]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float64)
    rgb = np.clip(fdc * SH_C0 + 0.5, 0, 1)
    print(f"  kept {len(xyz):,} gaussians (opacity>{min_opacity})")
    return xyz, rgb


def filter_floaters(xyz, rgb, radius_pct=99.0):
    """Drop the outer (radius_pct..100)% by distance from centroid."""
    c = np.median(xyz, axis=0)
    r = np.linalg.norm(xyz - c, axis=1)
    keep = r < np.percentile(r, radius_pct)
    print(f"  floater filter: kept {keep.sum():,}/{len(xyz):,} within {radius_pct}% radius")
    return xyz[keep], rgb[keep]


def reconstruct(xyz, rgb, depth, voxel):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(rgb)

    if voxel > 0:
        n0 = len(pc.points)
        pc = pc.voxel_down_sample(voxel)
        print(f"  voxel downsample @ {voxel}: {n0:,} → {len(pc.points):,}")

    print("  estimating normals…")
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3 or 0.05, max_nn=30))
    pc.orient_normals_consistent_tangent_plane(k=20)

    print(f"  Poisson reconstruction (depth={depth})…")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=depth, n_threads=-1,
    )
    # Trim low-density vertices (background hallucinations)
    densities = np.asarray(densities)
    cut = np.quantile(densities, 0.06)
    mesh.remove_vertices_by_mask(densities < cut)
    print(f"  mesh: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")
    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", type=Path)
    ap.add_argument("out", type=Path, help=".glb or .ply")
    ap.add_argument("--min_opacity", type=float, default=0.1)
    ap.add_argument("--radius_pct", type=float, default=99.0)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--depth", type=int, default=9)
    args = ap.parse_args()

    print(f"loading {args.ply}")
    xyz, rgb = load_splat(args.ply, args.min_opacity)
    xyz, rgb = filter_floaters(xyz, rgb, args.radius_pct)
    mesh = reconstruct(xyz, rgb, args.depth, args.voxel)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.out), mesh, write_ascii=False)
    size = args.out.stat().st_size / 1e6
    print(f"\nwrote {args.out}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
