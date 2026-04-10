"""
Generate navigation paths strictly within the known (white) office region.
Ignores grey (unknown) and black (wall) pixels.
Clean region: rows 0-73 only.
"""

import argparse
import numpy as np
import yaml
import heapq
from PIL import Image
from scipy.ndimage import binary_dilation

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--map-yaml",      default="data/maps/office_map_partcial.yaml")
parser.add_argument("--num-paths",     type=int,   default=200)
parser.add_argument("--robot-radius",  type=float, default=0.3)
parser.add_argument("--max-waypoints", type=int,   default=100)
parser.add_argument("--output",        default="data/paths/paths.npy")
parser.add_argument("--clean-row-max", type=int,   default=73)
args = parser.parse_args()

# ── Load map ──────────────────────────────────────────────────────────────────
print(f"Loading map: {args.map_yaml}")
with open(args.map_yaml) as f:
    meta = yaml.safe_load(f)

img_full = np.array(Image.open(f"data/maps/{meta['image']}").convert("L"))
res      = meta["resolution"]
origin   = np.array(meta["origin"][:2])

# ── Crop to clean region only ─────────────────────────────────────────────────
img = img_full[0 : args.clean_row_max + 1, :]
print(f"Full map   : {img_full.shape}")
print(f"Cropped map: {img.shape}  (rows 0→{args.clean_row_max})")

# ── Free space mask (white only) ──────────────────────────────────────────────
free_mask  = img >= 250
radius_px  = int(np.ceil(args.robot_radius / res))
wall_mask  = ~free_mask
inflated   = binary_dilation(wall_mask, iterations=radius_px)
safe_mask  = ~inflated
free_pixels = np.argwhere(safe_mask)
print(f"Safe free pixels after inflation: {len(free_pixels)}")

if len(free_pixels) < 10:
    raise RuntimeError("Too few free pixels — reduce --robot-radius")

# ── World ↔ pixel ──────────────────────────────────────────────────────────────
def px_to_world(row, col):
    x = col * res + origin[0]
    y = (img_full.shape[0] - row) * res + origin[1]
    return np.array([x, y])

# ── A* ────────────────────────────────────────────────────────────────────────
def heuristic(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def astar(grid, start, goal):
    h, w   = grid.shape
    open_q = []
    heapq.heappush(open_q, (0, start))
    came_from = {start: None}
    g_cost    = {start: 0}
    while open_q:
        _, cur = heapq.heappop(open_q)
        if cur == goal:
            path = []
            while cur:
                path.append(cur)
                cur = came_from[cur]
            return path[::-1]
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),
                        (-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = cur[0]+dr, cur[1]+dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc]:
                step = 1.414 if dr!=0 and dc!=0 else 1.0
                ng   = g_cost[cur] + step
                nb   = (nr, nc)
                if nb not in g_cost or ng < g_cost[nb]:
                    g_cost[nb]    = ng
                    came_from[nb] = cur
                    heapq.heappush(open_q, (ng + heuristic(nb, goal), nb))
    return None

# ── Smooth ────────────────────────────────────────────────────────────────────
def smooth_path(pixel_path, max_wps):
    if len(pixel_path) <= max_wps:
        return pixel_path
    indices = np.round(np.linspace(0, len(pixel_path)-1, max_wps)).astype(int)
    return [pixel_path[i] for i in indices]

# ── Generate ──────────────────────────────────────────────────────────────────
print(f"\nGenerating {args.num_paths} paths...")
paths_world = []
attempts    = 0
min_dist_px = int(1.0 / res)   # min 1m between start and goal

while len(paths_world) < args.num_paths:
    attempts += 1
    if attempts > args.num_paths * 20:
        print(f"Warning: only {len(paths_world)} paths after {attempts} attempts")
        break

    idx_s, idx_g = np.random.choice(len(free_pixels), 2, replace=False)
    start = tuple(free_pixels[idx_s])
    goal  = tuple(free_pixels[idx_g])

    if heuristic(start, goal) < min_dist_px:
        continue

    pixel_path = astar(safe_mask, start, goal)
    if pixel_path is None:
        continue

    pixel_path  = smooth_path(pixel_path, args.max_waypoints)
    world_path  = np.array([px_to_world(r, c) for r, c in pixel_path])
    paths_world.append(world_path)

    if len(paths_world) % 20 == 0:
        print(f"  {len(paths_world)}/{args.num_paths}  (attempts: {attempts})")

# ── Pad & save ────────────────────────────────────────────────────────────────
out = np.zeros((len(paths_world), args.max_waypoints, 2), dtype=np.float32)
for i, p in enumerate(paths_world):
    n = min(len(p), args.max_waypoints)
    out[i, :n, :] = p[:n]
    out[i, n:, :] = p[-1]   # pad tail with final goal, not zeros

np.save(args.output, out)
print(f"\nSaved {len(paths_world)} paths → {args.output}")
print(f"Shape      : {out.shape}")
print(f"World X    : {out[:,:,0].min():.2f} → {out[:,:,0].max():.2f} m")
print(f"World Y    : {out[:,:,1].min():.2f} → {out[:,:,1].max():.2f} m")