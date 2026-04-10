"""
Visualise pre-generated paths overlaid on the office map.
Run after generate_paths.py.
"""

import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--map-yaml",     default="data/maps/office_map_partcial.yaml")
parser.add_argument("--paths",        default="data/paths/paths.npy")
parser.add_argument("--clean-row-max",type=int, default=73)
parser.add_argument("--mode",         default="overview",
                    choices=["overview", "single", "waypoints"],
                    help="overview=all paths | single=one path | waypoints=one path with WP table")
parser.add_argument("--path-idx",     type=int, default=0,
                    help="Which path to inspect (for single/waypoints mode)")
args = parser.parse_args()

# ── Load ──────────────────────────────────────────────────────────────────────
with open(args.map_yaml) as f:
    meta = yaml.safe_load(f)

img_full = np.array(Image.open(f"data/maps/{meta['image']}").convert("L"))
img      = img_full[0 : args.clean_row_max + 1, :]   # crop to clean region
res      = meta["resolution"]
origin   = np.array(meta["origin"][:2])
paths    = np.load(args.paths)   # (N, max_wp, 2)

print(f"Paths loaded : {paths.shape}  (num_paths, max_waypoints, 2)")
print(f"World X      : {paths[:,:,0].min():.2f} → {paths[:,:,0].max():.2f} m")
print(f"World Y      : {paths[:,:,1].min():.2f} → {paths[:,:,1].max():.2f} m")
print(f"Map size     : {img.shape[1]} x {img.shape[0]} px  @ {res} m/px")

# ── Helpers ───────────────────────────────────────────────────────────────────
def world_to_px(xy):
    """(N,2) world metres → (col, row) in cropped image."""
    col = (xy[:, 0] - origin[0]) / res
    row = img_full.shape[0] - (xy[:, 1] - origin[1]) / res
    return col, row

def get_path(idx):
    """Return path array, trimmed to last unique point (no padded duplicates)."""
    p = paths[idx]
    # find where waypoints stop changing
    diffs = np.linalg.norm(np.diff(p, axis=0), axis=1)
    last  = np.where(diffs > 1e-4)[0]
    end   = int(last[-1]) + 2 if len(last) > 0 else len(p)
    return p[:end]

# ── Mode: overview ────────────────────────────────────────────────────────────
if args.mode == "overview":
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(img, cmap="gray", origin="upper")
    colors = plt.cm.rainbow(np.linspace(0, 1, len(paths)))
    for i in range(len(paths)):
        p = get_path(i)
        c, r = world_to_px(p)
        ax.plot(c, r, color=colors[i], linewidth=0.8, alpha=0.6)
        ax.plot(c[0],  r[0],  "o", color="lime", markersize=3, alpha=0.8)
        ax.plot(c[-1], r[-1], "X", color="red",  markersize=4, alpha=0.8)
    ax.set_title(f"All {len(paths)} paths in clean office region\n"
                 f"(green=start, red=goal)")
    start_p = mpatches.Patch(color="lime", label="Start")
    goal_p  = mpatches.Patch(color="red",  label="Goal")
    ax.legend(handles=[start_p, goal_p], fontsize=9)
    plt.tight_layout()
    out = "data/paths/viz_overview.png"
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.show()

# ── Mode: single ──────────────────────────────────────────────────────────────
elif args.mode == "single":
    p    = get_path(args.path_idx)
    c, r = world_to_px(p)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(img, cmap="gray", origin="upper")
    ax.plot(c, r, color="dodgerblue", linewidth=2, label="path")
    ax.plot(c[0],  r[0],  "o", color="lime", markersize=12, label="start")
    ax.plot(c[-1], r[-1], "X", color="red",  markersize=14, label="goal")
    # arrows every 5 waypoints
    for i in range(0, len(p)-1, 5):
        ax.annotate("", xy=(c[i+1], r[i+1]), xytext=(c[i], r[i]),
                    arrowprops=dict(arrowstyle="->", color="cyan", lw=1.5))
    ax.set_title(f"Path {args.path_idx}  |  {len(p)} waypoints  |  "
                 f"start={p[0].round(2)} m  goal={p[-1].round(2)} m")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = f"data/paths/viz_path_{args.path_idx}.png"
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.show()

# ── Mode: waypoints ───────────────────────────────────────────────────────────
elif args.mode == "waypoints":
    p    = get_path(args.path_idx)
    c, r = world_to_px(p)

    # print waypoint table
    print(f"\nPath {args.path_idx}  →  {len(p)} waypoints\n")
    print(f"{'WP':>4}  {'X (m)':>8}  {'Y (m)':>8}  {'Dist to next (m)':>18}")
    print("─" * 50)
    for i, wp in enumerate(p):
        dist = np.linalg.norm(p[i+1] - wp) if i < len(p)-1 else 0.0
        tag  = " ← START" if i == 0 else (" ← GOAL" if i == len(p)-1 else "")
        print(f"{i:>4}  {wp[0]:>8.3f}  {wp[1]:>8.3f}  {dist:>18.3f}{tag}")
    total = sum(np.linalg.norm(p[i+1]-p[i]) for i in range(len(p)-1))
    print(f"\nTotal path length: {total:.2f} m")

    # plot with numbered waypoints
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(img, cmap="gray", origin="upper")
    ax.plot(c, r, color="dodgerblue", linewidth=2, zorder=2)
    for i, (ci, ri) in enumerate(zip(c, r)):
        if i == 0:
            ax.plot(ci, ri, "o", color="lime",   markersize=12, zorder=4)
            ax.annotate("START", (ci, ri), xytext=(6,6),
                        textcoords="offset points", fontsize=8,
                        color="lime", fontweight="bold")
        elif i == len(p)-1:
            ax.plot(ci, ri, "X", color="red",    markersize=14, zorder=4)
            ax.annotate("GOAL",  (ci, ri), xytext=(6,6),
                        textcoords="offset points", fontsize=8,
                        color="red",  fontweight="bold")
        else:
            ax.plot(ci, ri, "o", color="yellow", markersize=5,  zorder=3)
            ax.annotate(str(i), (ci, ri), xytext=(4,4),
                        textcoords="offset points", fontsize=6, color="white")
    ax.set_title(f"Path {args.path_idx}  |  {len(p)} waypoints  |  "
                 f"start={p[0].round(2)} m  goal={p[-1].round(2)} m")
    plt.tight_layout()
    out = f"data/paths/viz_path_{args.path_idx}_waypoints.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved → {out}")
    plt.show()