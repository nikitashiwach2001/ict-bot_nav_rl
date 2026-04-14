from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import quat_inv, quat_apply


def _world_to_body(env: ManagerBasedEnv, xy_world: torch.Tensor) -> torch.Tensor:
    """Convert world-frame XY positions to robot body-frame XY vectors. (N, 2)"""
    robot = env.scene["robot"]
    diff_w = torch.zeros(env.num_envs, 3, device=env.device)
    diff_w[:, :2] = xy_world - robot.data.root_pos_w[:, :2]
    return quat_apply(quat_inv(robot.data.root_quat_w), diff_w)[:, :2]


def lidar_ranges(env: ManagerBasedEnv) -> torch.Tensor:
    """72 LiDAR distances normalised to [0, 1]. Shape: (N, 72)"""
    sensor = env.scene["lidar"]
    dist = torch.norm(sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=-1)
    return torch.clamp(dist, 0.0, sensor.cfg.max_distance) / sensor.cfg.max_distance


def rel_goal_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """Current waypoint in robot body frame. Shape: (N, 2)"""
    if not hasattr(env, "_goal_pos"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return _world_to_body(env, env._goal_pos)


def next_wp_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """Next waypoint (wp+1) in robot body frame. Shape: (N, 2)"""
    if not hasattr(env, "_waypoint_idx"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    n_wps = env._paths.shape[1]
    next_idx = torch.clamp(env._waypoint_idx + 1, max=n_wps - 1)
    next_wp_local = env._paths[env._path_idx, next_idx]
    next_wp_world = next_wp_local + env.scene.env_origins[:, :2]
    return _world_to_body(env, next_wp_world)


def next_next_wp_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """Waypoint after next (wp+2) in robot body frame. Shape: (N, 2)"""
    if not hasattr(env, "_waypoint_idx"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    n_wps = env._paths.shape[1]
    next_idx = torch.clamp(env._waypoint_idx + 2, max=n_wps - 1)
    next_wp_local = env._paths[env._path_idx, next_idx]
    next_wp_world = next_wp_local + env.scene.env_origins[:, :2]
    return _world_to_body(env, next_wp_world)


def heading_error_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """Heading error to current waypoint as (sin θ, cos θ). Shape: (N, 2)
    θ = 0 when robot's +X body axis faces the goal (forward direction)."""
    if not hasattr(env, "_goal_pos"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    local_2d = _world_to_body(env, env._goal_pos)
    angle = torch.atan2(local_2d[:, 1], local_2d[:, 0])
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)


def wheel_velocities_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """Wheel joint velocities normalised to [-1, 1]. Shape: (N, 2)"""
    return env.scene["robot"].data.joint_vel / 6.0
