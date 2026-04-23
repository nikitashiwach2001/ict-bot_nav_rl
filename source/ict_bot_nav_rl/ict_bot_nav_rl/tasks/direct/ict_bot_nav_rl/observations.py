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


# ── Stage 1 observation functions ─────────────────────────────────────────────

def dtg_htg_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """[DTG / 10.0, cos(heading_error)] to final goal. Shape: (N, 2)

    DTG normalised by LiDAR max_distance (10 m) so it lives in [0, 1].
    cos(heading) is 1.0 when facing goal, -1.0 when facing away.
    """
    if not hasattr(env, "_final_goal_pos"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    robot_xy = env.scene["robot"].data.root_pos_w[:, :2]
    dist     = torch.norm(env._final_goal_pos - robot_xy, dim=-1, keepdim=True)
    local_2d = _world_to_body(env, env._final_goal_pos)
    angle    = torch.atan2(local_2d[:, 1], local_2d[:, 0])
    cos_h    = torch.cos(angle).unsqueeze(-1)
    return torch.cat([dist / 10.0, cos_h], dim=-1)


def robot_pos_vel_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """[x, y relative to env origin (÷5), forward_vel (÷0.3), ang_vel (÷2.0)]. Shape: (N, 4)

    Position normalised by arena half-extent (~5 m) so values are in [-1, 1].
    Forward speed normalised by max commanded speed (~0.3 m/s).
    Angular velocity normalised by max angular speed (2.0 rad/s) — robot knows how fast it is turning.
    """
    robot    = env.scene["robot"]
    pos_w    = robot.data.root_pos_w[:, :2]
    orig     = env.scene.env_origins[:, :2]
    pos_norm = (pos_w - orig) / 5.0
    local_vel = quat_apply(quat_inv(robot.data.root_quat_w), robot.data.root_lin_vel_w)
    fwd_vel   = local_vel[:, 0:1] / 0.3
    ang_vel   = robot.data.root_ang_vel_w[:, 2:3] / 2.0   # yaw rate normalised
    return torch.cat([pos_norm, fwd_vel, ang_vel], dim=-1)


def obstacle_placeholder_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """Zeros for K=1 obstacle slot — replaced by real obstacle obs in Stage 2. Shape: (N, 3)"""
    return torch.zeros(env.num_envs, 3, device=env.device)


def cp_obstacle_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """K=1 most dangerous obstacle in robot body frame + CP. Shape: (N, 3)

    Paper's full CP formula: CP = 0.5*Pc_ttc + 0.5*Pc_dto  [alpha=0.5]

    Pc_ttc = min(1, dt/TTC) if obstacle approaching, else 0
    Pc_dto = (lmax - dist) / (lmax - lmin) if dist < lmax, else 0
      lmax = 3.0m (detection range), lmin = collision_threshold

    Paper finding: K=1 performance == K=4,8 → single most dangerous obstacle is enough.
    Layout: [x/10, y/10, CP]

    Requires env._obstacle_pos_all (N, K, 2) and env._obstacle_vel_all (N, K, 2).
    """
    if not hasattr(env, "_obstacle_pos_all") or not hasattr(env, "_obstacle_vel_all"):
        return torch.zeros(env.num_envs, 3, device=env.device)

    robot    = env.scene["robot"]
    robot_xy = robot.data.root_pos_w[:, :2]       # (N, 2)
    robot_vw = robot.data.root_lin_vel_w[:, :2]   # (N, 2) world frame linear vel
    q_inv    = quat_inv(robot.data.root_quat_w)

    N    = env.num_envs
    dt   = env.cfg.sim.dt * env.cfg.decimation    # 0.05s Isaac Sim
    lmax = 3.0                                     # detection range (m)
    lmin = env.cfg.collision_threshold             # 0.4m

    delta_p = env._obstacle_pos_all - robot_xy.unsqueeze(1)   # (N, K, 2)
    delta_v = env._obstacle_vel_all - robot_vw.unsqueeze(1)   # (N, K, 2)
    dist    = torch.norm(delta_p, dim=-1)                      # (N, K)

    # Pc_ttc — TTC based
    dv_sq  = (delta_v ** 2).sum(dim=-1).clamp(min=1e-6)
    dp_dv  = (delta_p * delta_v).sum(dim=-1)
    ttc    = -dp_dv / dv_sq
    approaching = ttc > 0.0
    ttc_safe    = ttc.clamp(min=1e-6)
    pc_ttc = torch.where(approaching, torch.clamp(dt / ttc_safe, max=1.0),
                         torch.zeros_like(ttc))                # (N, K)

    # Pc_dto — distance based
    in_range = dist < lmax
    pc_dto   = torch.where(
        in_range,
        torch.clamp((lmax - dist) / (lmax - lmin + 1e-6), min=0.0, max=1.0),
        torch.zeros_like(dist)
    )                                                          # (N, K)

    # Combined CP
    cp = 0.5 * pc_ttc + 0.5 * pc_dto                          # (N, K)

    # K=1: pick most dangerous obstacle
    idx     = cp.argmax(dim=-1)                                # (N,)
    arange  = torch.arange(N, device=env.device)
    obs_pos = env._obstacle_pos_all[arange, idx]               # (N, 2)
    obs_cp  = cp[arange, idx]                                  # (N,)

    diff_w        = torch.zeros(N, 3, device=env.device)
    diff_w[:, :2] = obs_pos - robot_xy
    local = quat_apply(q_inv, diff_w)[:, :2]                  # (N, 2) body frame

    return torch.cat([local / 10.0, obs_cp.unsqueeze(-1)], dim=-1)  # (N, 3)


def nearest_obstacle_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """All K=4 moving obstacles in robot body frame + speed, sorted nearest-first. Shape: (N, 12)

    Layout: [x0/10, y0/10, spd0, x1/10, y1/10, spd1, x2/10, y2/10, spd2, x3/10, y3/10, spd3]
    Obstacles sorted by distance so index 0 is always the nearest.
    Replaces obstacle_placeholder_obs in Stage 2.
    Requires env._obstacle_pos_all (N, K, 2) and env._obstacle_speed_all (N, K).
    """
    if not hasattr(env, "_obstacle_pos_all") or not hasattr(env, "_obstacle_speed_all"):
        return torch.zeros(env.num_envs, 12, device=env.device)

    robot    = env.scene["robot"]
    robot_xy = robot.data.root_pos_w[:, :2]
    q_inv    = quat_inv(robot.data.root_quat_w)

    K    = env._obstacle_pos_all.shape[1]          # 4
    dists = torch.norm(
        env._obstacle_pos_all - robot_xy.unsqueeze(1), dim=-1
    )                                               # (N, K)
    order = torch.argsort(dists, dim=-1)            # (N, K) nearest first

    obs_parts = []
    for k in range(K):
        idx     = order[:, k]                                              # (N,)
        arange  = torch.arange(env.num_envs, device=env.device)
        obs_pos = env._obstacle_pos_all[arange, idx]                      # (N, 2)
        obs_spd = env._obstacle_speed_all[arange, idx]                    # (N,)

        diff_w        = torch.zeros(env.num_envs, 3, device=env.device)
        diff_w[:, :2] = obs_pos - robot_xy
        local = quat_apply(q_inv, diff_w)[:, :2]                          # (N, 2)

        obs_parts.append(torch.cat([local / 10.0, obs_spd.unsqueeze(-1)], dim=-1))  # (N, 3)

    return torch.cat(obs_parts, dim=-1)             # (N, 12)
