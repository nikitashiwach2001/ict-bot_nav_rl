"""
Convert 2D occupancy map (black pixels = walls) to a USD mesh file.
Single merged mesh for all walls — required for Isaac Lab RayCaster.
Wall height: 1.0m. Ground plane included.
"""

import argparse
import numpy as np
import yaml
from PIL import Image
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Vt, Sdf
from scipy.ndimage import label as scipy_label

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--map-yaml",      default="data/maps/office_map_partcial.yaml")
parser.add_argument("--output",        default="data/maps/office_env.usd")
parser.add_argument("--clean-row-max", type=int,   default=73)
parser.add_argument("--wall-height",   type=float, default=1.0)
args = parser.parse_args()

# ── Load map ──────────────────────────────────────────────────────────────────
print(f"Loading map: {args.map_yaml}")
with open(args.map_yaml) as f:
    meta = yaml.safe_load(f)

img_full = np.array(Image.open(f"data/maps/{meta['image']}").convert("L"))
res      = meta["resolution"]
origin   = np.array(meta["origin"][:2])

# ── Crop to clean region ──────────────────────────────────────────────────────
img = img_full[0 : args.clean_row_max + 1, :]
print(f"Full map   : {img_full.shape}")
print(f"Cropped map: {img.shape}  (rows 0→{args.clean_row_max})")

# ── Clean wall mask — remove artifact clusters < 4px ─────────────────────────
wall_mask_raw = img <= 10
labeled_w, n_w = scipy_label(wall_mask_raw)
sizes_w = np.bincount(labeled_w.ravel())
clean_wall = np.zeros_like(wall_mask_raw)
for i in range(1, n_w + 1):
    if sizes_w[i] >= 4:
        clean_wall |= (labeled_w == i)

wall_pixels = np.argwhere(clean_wall)
print(f"Wall pixels (after cleaning): {len(wall_pixels)}")

# ── World coordinate helper ───────────────────────────────────────────────────
def px_to_world_centre(row, col):
    x = (col + 0.5) * res + origin[0]
    y = (img_full.shape[0] - row - 0.5) * res + origin[1]
    return x, y

# ── Compute environment bounds ────────────────────────────────────────────────
x_min = origin[0]
x_max = img.shape[1] * res + origin[0]
y_min = (img_full.shape[0] - args.clean_row_max) * res + origin[1]
y_max = img_full.shape[0] * res + origin[1]
cx    = (x_min + x_max) / 2.0
cy    = (y_min + y_max) / 2.0
w     = x_max - x_min
h_map = y_max - y_min

print(f"World bounds:")
print(f"  X : {x_min:.3f} → {x_max:.3f} m  (width  {w:.2f} m)")
print(f"  Y : {y_min:.3f} → {y_max:.3f} m  (height {h_map:.2f} m)")

# ── Create USD stage ──────────────────────────────────────────────────────────
print(f"\nCreating USD: {args.output}")
stage = Usd.Stage.CreateNew(args.output)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")

# ── Ground plane ──────────────────────────────────────────────────────────────
ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
half_w = w / 2.0 + 0.5
half_h = h_map / 2.0 + 0.5
ground.GetPointsAttr().Set(Vt.Vec3fArray([
    Gf.Vec3f(cx - half_w, cy - half_h, 0.0),
    Gf.Vec3f(cx + half_w, cy - half_h, 0.0),
    Gf.Vec3f(cx + half_w, cy + half_h, 0.0),
    Gf.Vec3f(cx - half_w, cy + half_h, 0.0),
]))
ground.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
ground.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(ground.GetPrim()).GetApproximationAttr().Set("none")
print("Ground plane added.")

# ── Single merged wall mesh ───────────────────────────────────────────────────
print(f"Building single merged wall mesh from {len(wall_pixels)} boxes...")

half_px = res / 2.0
all_points   = []
all_f_counts = []
all_f_idx    = []
v_offset     = 0

for row, col in wall_pixels:
    wx, wy = px_to_world_centre(row, col)
    x0, x1 = wx - half_px, wx + half_px
    y0, y1 = wy - half_px, wy + half_px
    z0, z1 = 0.0, args.wall_height

    # 8 vertices of this box
    all_points.extend([
        (x0, y0, z0),  # 0
        (x1, y0, z0),  # 1
        (x1, y1, z0),  # 2
        (x0, y1, z0),  # 3
        (x0, y0, z1),  # 4
        (x1, y0, z1),  # 5
        (x1, y1, z1),  # 6
        (x0, y1, z1),  # 7
    ])

    # 6 faces (quads)
    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 7, 6, 5],  # top
        [0, 4, 5, 1],  # front
        [1, 5, 6, 2],  # right
        [2, 6, 7, 3],  # back
        [3, 7, 4, 0],  # left
    ]
    for f in faces:
        all_f_counts.append(4)
        all_f_idx.extend([v + v_offset for v in f])
    v_offset += 8

# define single Mesh prim at /World/Walls
walls_mesh = UsdGeom.Mesh.Define(stage, "/World/Walls")
walls_mesh.GetPointsAttr().Set(
    Vt.Vec3fArray([Gf.Vec3f(p[0], p[1], p[2]) for p in all_points])
)
walls_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(all_f_counts))
walls_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(all_f_idx))

# physics collision
UsdPhysics.CollisionAPI.Apply(walls_mesh.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(walls_mesh.GetPrim()).GetApproximationAttr().Set("none")

# visual colour
vis_api = UsdGeom.PrimvarsAPI(walls_mesh)
color = vis_api.CreatePrimvar(
    "displayColor", Sdf.ValueTypeNames.Color3fArray,
    UsdGeom.Tokens.constant
)
color.Set(Vt.Vec3fArray([Gf.Vec3f(0.2, 0.2, 0.2)]))

print(f"Single merged wall mesh created at /World/Walls")
print(f"  Total vertices : {len(all_points)}")
print(f"  Total faces    : {len(all_f_counts)}")

# ── Save ──────────────────────────────────────────────────────────────────────
stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
stage.GetRootLayer().Save()
print(f"\nDone. Saved → {args.output}")
print(f"\nPrim structure:")
print(f"  /World/Ground  [Mesh] — ground plane")
print(f"  /World/Walls   [Mesh] — all {len(wall_pixels)} wall boxes merged")
print(f"\nFor RayCaster mesh_prim_paths use:")
print(f"  /World/OfficeEnv/World/Walls")