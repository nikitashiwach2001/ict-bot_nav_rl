"""
Reads office_map_partcial.pgm and prints Isaac Lab wall box definitions
to be pasted into scene.py.  Run once; output is static.

Usage:
    python scripts/map_to_scene.py
"""

import numpy as np
from PIL import Image
from scipy import ndimage

MAP_PGM  = "data/maps/office_map_partcial.pgm"
MAP_YAML_ORIGIN = [-1.34, -9.96]
RESOLUTION      = 0.05   # metres per pixel
WALL_HEIGHT     = 0.3    # metres
MIN_AREA_PX     = 2      # skip noise smaller than this

img    = np.array(Image.open(MAP_PGM).convert("L"))
H      = img.shape[0]
origin = np.array(MAP_YAML_ORIGIN)
wall   = img < 50

labeled, n_components = ndimage.label(wall)


def px_to_world(row, col):
    x = col * RESOLUTION + origin[0]
    y = (H - row) * RESOLUTION + origin[1]
    return x, y


boxes = []
for i in range(n_components):
    mask = labeled == (i + 1)
    if mask.sum() < MIN_AREA_PX:
        continue
    rows, cols = np.where(mask)
    x0, y0 = px_to_world(rows.max(), cols.min())
    x1, y1 = px_to_world(rows.min(), cols.max())
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    sx = max(abs(x1 - x0), RESOLUTION)
    sy = max(abs(y1 - y0), RESOLUTION)
    boxes.append((cx, cy, sx, sy, WALL_HEIGHT))

print(f"# {len(boxes)} wall boxes extracted from {MAP_PGM}\n")
for idx, (cx, cy, sx, sy, sz) in enumerate(boxes):
    name = f"wall_{idx:02d}"
    print(f'    {name} = AssetBaseCfg(')
    print(f'        prim_path="{{ENV_REGEX_NS}}/Walls/{name}",')
    print(f'        spawn=sim_utils.MeshCuboidCfg(')
    print(f'            size=({sx:.3f}, {sy:.3f}, {sz:.3f}),')
    print(f'            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=False),')
    print(f'            collision_props=sim_utils.CollisionPropertiesCfg(),')
    print(f'            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),')
    print(f'        ),')
    print(f'        init_state=AssetBaseCfg.InitialStateCfg(pos=({cx:.3f}, {cy:.3f}, {sz/2:.3f})),')
    print(f'    )')
