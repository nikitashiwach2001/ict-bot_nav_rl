"""
Transfer Phase 1 (8-dim obs) weights → Phase 2 (79-dim obs) warm-start checkpoint.

Input layer expanded:  (128, 8) → (128, 79)
  cols 0-7  : copied from Phase 1  (navigation knowledge fully preserved)
  cols 8-78 : zero-initialized     (LiDAR weights start silent; activated by gradients)

Hidden + output layers : copied exactly  (128→64→2, same shapes)
State preprocessor     : extended 8 → 79 (LiDAR dims start at mean=0, var=1)
"""

import os
import argparse
import torch

NAV_DIM   = 8
LIDAR_DIM = 71
NEW_DIM   = NAV_DIM + LIDAR_DIM   # 79

DEFAULT_SRC = (
    "logs/skrl/ict_bot_nav_plain/"
    "2026-04-17_07-16-50_ppo_torch_ppo_plain_finetune2/"
    "checkpoints/agent_100000.pt"
)
DEFAULT_DST = "logs/skrl/phase2_warmstart/agent_warmstart.pt"


def transfer(src_path: str, dst_path: str) -> None:
    src = torch.load(src_path, map_location="cpu")
    out: dict = {}

    for net in ("policy", "value"):
        net_src = src[net]
        net_dst: dict = {}

        # ── input layer: expand (128, 8) → (128, 80) ──────────────────────
        old_w = net_src["net_container.0.weight"]      # (128, 8)
        new_w = torch.zeros(128, NEW_DIM)
        new_w[:, :NAV_DIM] = old_w                     # preserve nav cols
        # LiDAR cols 8-79 stay at zero → network output unchanged at start
        net_dst["net_container.0.weight"] = new_w
        net_dst["net_container.0.bias"]   = net_src["net_container.0.bias"]

        # ── hidden + output layers: exact copy ────────────────────────────
        for key in (
            "net_container.2.weight", "net_container.2.bias",
            "policy_layer.weight",    "policy_layer.bias",
            "value_layer.weight",     "value_layer.bias",
            "log_std_parameter",
        ):
            if key in net_src:
                net_dst[key] = net_src[key]

        out[net] = net_dst

    # ── state preprocessor: extend 8 → 80 ────────────────────────────────
    sp = src["state_preprocessor"]
    new_mean = torch.zeros(NEW_DIM)
    new_var  = torch.ones(NEW_DIM)
    new_mean[:NAV_DIM] = sp["running_mean"]
    new_var[:NAV_DIM]  = sp["running_variance"]
    out["state_preprocessor"] = {
        "running_mean":     new_mean,
        "running_variance": new_var,
        "current_count":    sp["current_count"],
    }

    # value preprocessor unchanged (1-dim scalar)
    out["value_preprocessor"] = {
        "running_mean":     src["value_preprocessor"]["running_mean"],
        "running_variance": src["value_preprocessor"]["running_variance"],
        "current_count":    src["value_preprocessor"]["current_count"],
    }

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    torch.save(out, dst_path)

    print(f"[OK] Warm-start checkpoint saved → {dst_path}")
    print(f"     Input layer : (128, {NAV_DIM}) → (128, {NEW_DIM})  [{NAV_DIM} nav + {LIDAR_DIM} LiDAR]")
    print(f"     LiDAR cols  : {NAV_DIM}–{NEW_DIM - 1} zero-initialized")
    print(f"     Hidden/out  : copied exactly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 → Phase 2 weight transfer")
    parser.add_argument("--src", default=DEFAULT_SRC, help="Phase 1 checkpoint path")
    parser.add_argument("--dst", default=DEFAULT_DST, help="Output warm-start path")
    args = parser.parse_args()
    transfer(args.src, args.dst)
