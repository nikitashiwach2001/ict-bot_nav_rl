"""
Reward shaping for ICTBot waypoint navigation task.

Structure
---------
  Math helpers          get_yaw, wrap_angle
  Sensor helper         lidar_min_dist   (shared with terminations)
  Private helper        _rel_goal_local  (waypoint in body frame)
  Dense navigation      velocity_toward_target, reward_forward_speed,
                        reward_heading_alignment, penalize_backwards_movement
  Sparse terminal       goal_reached
  Safety penalty        collision

NOTE: This robot's forward direction is +X in the body (base_footprint) frame.
Positive wheel velocity → robot moves +Y in link_base → Rz(90°)*(0,1,0) = (+X) in base_footprint.
All velocity computations use  forward_speed = local_vel[:, 0].

Weights are set in RewardsCfg inside ict_bot_nav_rl_env_cfg.py.
"""

from __future__ import annotations

import math
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import quat_inv, quat_apply


# ── Math helpers ──────────────────────────────────────────────────────────────

def get_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle from (N, 4) quaternion tensor [w, x, y, z]."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap angle tensor to [-π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


# ── Sensor helper ─────────────────────────────────────────────────────────────

def lidar_min_dist(env: ManagerBasedEnv) -> torch.Tensor:
    """Per-environment minimum LiDAR ray distance (metres).

    Shared by wall_proximity / collision rewards and collision_termination.
    """
    sensor     = env.scene["lidar"]
    ray_hits   = sensor.data.ray_hits_w          # (N, n_rays, 3)
    sensor_pos = sensor.data.pos_w.unsqueeze(1)  # (N,      1, 3)
    return torch.norm(ray_hits - sensor_pos, dim=-1).min(dim=-1).values  # (N,)


# ── Private helper ────────────────────────────────────────────────────────────

def _rel_goal_local(env: ManagerBasedEnv):
    """Current waypoint sub-goal rotated into the robot's body frame.

    Returns
    -------
    local_2d : (N, 2)  — goal vector in body-frame XY
    dist     : (N,)    — Euclidean distance to goal
    """
    if not hasattr(env, "_goal_pos"):
        n = env.num_envs
        return (torch.zeros(n, 2, device=env.device),
                torch.ones(n,    device=env.device))

    robot  = env.scene["robot"]
    # Build a 3-D world-space vector (zero Z) from robot to goal
    diff_w        = torch.zeros(env.num_envs, 3, device=env.device)
    diff_w[:, :2] = env._goal_pos - robot.data.root_pos_w[:, :2]

    # Rotate into body frame using inverse of root quaternion
    local    = quat_apply(quat_inv(robot.data.root_quat_w), diff_w)  # (N, 3)
    local_2d = local[:, :2]                                           # (N, 2)
    dist     = torch.norm(local_2d, dim=-1)                           # (N,)
    return local_2d, dist


# ── Dense navigation rewards ──────────────────────────────────────────────────

def velocity_toward_target(env: ManagerBasedEnv) -> torch.Tensor:
    """Velocity component toward the current waypoint, gated by forward speed.

    Both the dot-product toward the goal AND forward motion must be positive
    to earn this reward.  This prevents the policy from backing into the goal
    or strafing sideways to collect it.
    """
    local_2d, dist = _rel_goal_local(env)
    target_dir     = local_2d / (dist.unsqueeze(-1) + 1e-6)   # unit vector (N,2)

    robot     = env.scene["robot"]
    local_vel = quat_apply(quat_inv(robot.data.root_quat_w),
                           robot.data.root_lin_vel_w)           # (N, 3) in body frame

    vel_toward    = (local_vel[:, :2] * target_dir).sum(dim=-1)
    forward_speed = local_vel[:, 0]                             # +X is forward

    return (torch.clamp(vel_toward,    min=0.0)
            * torch.clamp(forward_speed, min=0.0))


def reward_forward_speed(env: ManagerBasedEnv) -> torch.Tensor:
    """Squared velocity-toward-target — disproportionately rewards faster motion."""
    return velocity_toward_target(env) ** 2


def reward_heading_alignment(env: ManagerBasedEnv) -> torch.Tensor:
    """cos(heading_error) — continuous reward for facing the current waypoint.

    Heading error = 0 when the robot faces the goal (-Y body axis points at goal).
    Returns values in [0, 1]; negative heading errors yield 0 (clamped).
    """
    local_2d, _ = _rel_goal_local(env)
    # atan2(y, x): angle is 0 when +X points at goal (forward direction)
    angle = torch.atan2(local_2d[:, 1], local_2d[:, 0])
    return torch.clamp(torch.cos(angle), min=0.0)


def penalize_backwards_movement(env: ManagerBasedEnv) -> torch.Tensor:
    """Magnitude of reverse speed — weight in RewardsCfg must be negative."""
    robot     = env.scene["robot"]
    local_vel = quat_apply(quat_inv(robot.data.root_quat_w),
                           robot.data.root_lin_vel_w)
    forward_speed = local_vel[:, 0]                 # +X is forward
    return torch.clamp(-forward_speed, min=0.0)     # positive when reversing


# ── Sparse terminal reward ────────────────────────────────────────────────────

def goal_reached(env: ManagerBasedEnv) -> torch.Tensor:
    """Binary 0/1 reward when the robot reaches the FINAL waypoint.

    Fires on the same step as goal_reached_termination so the agent sees
    the bonus before the episode ends.
    """
    if not hasattr(env, "_final_goal_pos"):
        return torch.zeros(env.num_envs, device=env.device)
    root_pos = env.scene["robot"].data.root_pos_w[:, :2]
    dist     = torch.norm(env._final_goal_pos - root_pos, dim=-1)
    return (dist < env.cfg.goal_reach_threshold).float()


# ── Safety penalties ──────────────────────────────────────────────────────────

def fell_off(env: ManagerBasedEnv) -> torch.Tensor:
    """Returns -1 when the robot's Z drops below -0.5 m (fallen through/off).

    Weight in RewardsCfg should be POSITIVE (e.g. 200.0); the -1 carries the sign.
    """
    z = env.scene["robot"].data.root_pos_w[:, 2]
    return -(z < -0.5).float()


def collision(env: ManagerBasedEnv) -> torch.Tensor:
    """Returns -1 when lidar_min < collision_threshold, else 0.

    Weight in RewardsCfg should be POSITIVE (e.g. 50.0); the -1 carries
    the sign so the total contribution is negative.
    """
    return -(lidar_min_dist(env) < env.cfg.collision_threshold).float()
