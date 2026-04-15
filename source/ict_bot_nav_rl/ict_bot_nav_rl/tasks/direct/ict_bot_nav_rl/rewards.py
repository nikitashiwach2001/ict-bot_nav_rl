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
    """
    Current waypoint sub-goal rotated into the robot's body frame.

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
    """
    Velocity component toward the current waypoint, gated by forward speed.
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

def waypoint_reached(env: ManagerBasedEnv) -> torch.Tensor:
    """Binary 0/1 reward each time the robot reaches an intermediate waypoint.

    Fires on the same step that _advance_waypoints() will advance the index,
    i.e. when the robot enters waypoint_reach_threshold of _goal_pos.
    """
    if not hasattr(env, "_goal_pos"):
        return torch.zeros(env.num_envs, device=env.device)
    robot_xy = env.scene["robot"].data.root_pos_w[:, :2]
    dist = torch.norm(env._goal_pos - robot_xy, dim=-1)
    return (dist < env.cfg.waypoint_reach_threshold).float()


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


def wall_proximity(env: ManagerBasedEnv) -> torch.Tensor:
    """Continuous penalty that rises as the robot gets closer to any obstacle.

    Returns a value in [0, 1]:
      - 0.0 when min LiDAR dist >= proximity_threshold (safe zone)
      - 1.0 when touching (dist = 0)
    Weight in RewardsCfg must be NEGATIVE (e.g. -30.0).
    """
    min_dist = lidar_min_dist(env)
    safe     = env.cfg.proximity_threshold
    return torch.clamp(1.0 - min_dist / safe, min=0.0)


def collision(env: ManagerBasedEnv) -> torch.Tensor:
    """Returns -1 when lidar_min < collision_threshold, else 0.

    Weight in RewardsCfg should be POSITIVE (e.g. 100.0); the -1 carries
    the sign so the total contribution is negative.
    """
    return -(lidar_min_dist(env) < env.cfg.collision_threshold).float()

# extended reward functions:

def progress_delta(env: ManagerBasedEnv) -> torch.Tensor:
    """
    Reward for getting closer to the current waypoint since the last step.
 
    Gives the policy a step-level memory signal that it is making path-level
    progress toward the waypoint, independent of instantaneous velocity.
    Requires env._prev_goal_dist to be initialised in reset_robot_to_path_start.
 
    Returns values in [-0.5, 0.5] (clamped to prevent spike on waypoint advance).
    """
    if not hasattr(env, "_goal_pos") or not hasattr(env, "_prev_goal_dist"):
        return torch.zeros(env.num_envs, device=env.device)
    robot_xy  = env.scene["robot"].data.root_pos_w[:, :2]
    curr_dist = torch.norm(env._goal_pos - robot_xy, dim=-1)
    delta     = env._prev_goal_dist - curr_dist          # positive = getting closer
    env._prev_goal_dist = curr_dist.clone()
    return torch.clamp(delta, min=-0.5, max=0.5)


def forward_clearance(env: ManagerBasedEnv) -> torch.Tensor:
    """
    Reward open space directly ahead of the robot (within ±30° of forward axis).
 
    Encourages the policy to proactively seek clear corridors rather than
    hugging walls. Normalised to [0, 1] — full reward at 2m+ clearance ahead.
    """
    sensor   = env.scene["lidar"]
    ray_hits = sensor.data.ray_hits_w          # (N, n_rays, 3)
    pos      = sensor.data.pos_w               # (N, 3)
 
    ray_dirs = ray_hits - pos.unsqueeze(1)     # (N, n_rays, 3)
    ray_dist = torch.norm(ray_dirs, dim=-1)    # (N, n_rays)
 
    robot    = env.scene["robot"]
    fwd_body = torch.zeros(env.num_envs, 3, device=env.device)
    fwd_body[:, 0] = 1.0
    fwd_world = quat_apply(robot.data.root_quat_w, fwd_body)  # (N, 3)
 
    ray_dirs_norm = ray_dirs / (ray_dist.unsqueeze(-1) + 1e-6)
    cos_sim = (ray_dirs_norm * fwd_world.unsqueeze(1)).sum(dim=-1)  # (N, n_rays)
 
    # Front rays only: cos > cos(30°) ≈ 0.866
    front_mask = cos_sim > 0.866
    front_dist = torch.where(front_mask, ray_dist,
                             torch.full_like(ray_dist, 10.0))
    min_front  = front_dist.min(dim=-1).values  # (N,)
 
    return torch.clamp(min_front / 2.0, max=1.0)

def obstacle_avoidance(env: ManagerBasedEnv) -> torch.Tensor:
    """
    Graduated penalty across 3 zones that teaches the robot to steer
    around obstacles — not freeze or spin in place.
 
    Replaces the old binary collision() + collision_termination combo.
    NO termination on collision — the robot must learn to escape.
 
    Zone 1 (0.5m – 0.8m): Mild warning     → policy learns to steer early
    Zone 2 (0.2m – 0.5m): Strong pressure   → policy commits to avoidance
    Zone 3 (0.0m – 0.2m): Severe but alive  → policy must recover, not die
 
    Weight in RewardsCfg must be NEGATIVE (e.g. -8.0).
    """
    min_dist = lidar_min_dist(env)
 
    zone1 = torch.clamp(1.0 - min_dist / 0.8, min=0.0, max=1.0)
    zone2 = torch.clamp(1.0 - min_dist / 0.5, min=0.0, max=1.0) * 3.0
    zone3 = torch.clamp(1.0 - min_dist / 0.2, min=0.0, max=1.0) * 8.0
 
    return zone1 + zone2 + zone3  

def spinning_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """
    Penalise high angular velocity when forward speed is low.
 
    Spinning in place  = high |yaw_rate| + low forward_speed  → full penalty
    Turning while moving = high |yaw_rate| + high forward_speed → near-zero penalty
 
    Gate function: exp(-3.0 * forward_speed)
      forward_speed = 0.0  → multiplier = 1.00  (full penalty)
      forward_speed = 0.3  → multiplier ≈ 0.41
      forward_speed = 0.5  → multiplier ≈ 0.22
 
    This forces the policy to always combine rotation with forward motion —
    i.e. steer, not spin.
 
    Weight in RewardsCfg must be NEGATIVE (e.g. -5.0).
    """
    robot     = env.scene["robot"]
    local_vel = quat_apply(quat_inv(robot.data.root_quat_w),
                           robot.data.root_lin_vel_w)           # (N, 3) body frame
    ang_vel       = robot.data.root_ang_vel_w[:, 2]             # yaw rate (N,)
    forward_speed = torch.clamp(local_vel[:, 0], min=0.0)       # only positive fwd
 
    spin_cost = torch.abs(ang_vel) * torch.exp(-3.0 * forward_speed)
    return spin_cost


def lateral_clearance_reward(env: ManagerBasedEnv) -> torch.Tensor:
    """
    Reward the robot for turning toward the more open side when near an obstacle.
 
    Compares mean LiDAR distance on the left side (+Y body) vs right side (-Y body).
    If left is more open  → reward positive yaw rate (CCW, turning left).
    If right is more open → reward negative yaw rate (CW,  turning right).
 
    Only activates when min_dist < proximity_threshold so it does not interfere
    with open-space navigation where heading alignment should dominate.
 
    Weight in RewardsCfg must be POSITIVE (e.g. 6.0).
    """
    sensor   = env.scene["lidar"]
    ray_hits = sensor.data.ray_hits_w           # (N, n_rays, 3)
    pos      = sensor.data.pos_w                # (N, 3)
 
    ray_dirs = ray_hits - pos.unsqueeze(1)      # (N, n_rays, 3)
    ray_dist = torch.norm(ray_dirs, dim=-1)     # (N, n_rays)
 
    robot = env.scene["robot"]
    # Rotate all ray directions into body frame
    q_inv = quat_inv(robot.data.root_quat_w)                    # (N, 4)
    q_inv_exp = q_inv.unsqueeze(1).expand(
        env.num_envs, ray_dirs.shape[1], 4
    ).reshape(-1, 4)
    ray_dirs_flat  = ray_dirs.reshape(-1, 3)
    ray_body_flat  = quat_apply(q_inv_exp, ray_dirs_flat)
    ray_dirs_body  = ray_body_flat.reshape(env.num_envs, -1, 3) # (N, n_rays, 3)
 
    left_mask  = ray_dirs_body[:, :, 1] > 0.1   # +Y body = left
    right_mask = ray_dirs_body[:, :, 1] < -0.1  # -Y body = right
 
    left_dist  = torch.where(left_mask,  ray_dist, torch.full_like(ray_dist, 10.0))
    right_dist = torch.where(right_mask, ray_dist, torch.full_like(ray_dist, 10.0))
 
    left_mean  = left_dist.mean(dim=-1)          # (N,)
    right_mean = right_dist.mean(dim=-1)         # (N,)
 
    clearance_diff      = left_mean - right_mean # positive → left more open
    ang_vel             = robot.data.root_ang_vel_w[:, 2]  # positive = CCW = left turn
    turning_toward_open = clearance_diff * ang_vel
 
    # Gate: only active near obstacles
    min_dist      = lidar_min_dist(env)
    near_obstacle = torch.clamp(
        1.0 - min_dist / env.cfg.proximity_threshold, min=0.0
    )
 
    return torch.clamp(turning_toward_open, min=0.0) * near_obstacle
 