from __future__ import annotations

import math
import os
import torch
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedEnv
from isaaclab.utils import configclass
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)
import isaaclab.sim as sim_utils

from .robot_cfg import IctBotNavSceneCfg
from .actions import ActionsCfg
from .observations import (
    lidar_ranges, rel_goal_obs, next_wp_obs,
    next_next_wp_obs, heading_error_obs, wheel_velocities_obs,
)
from .rewards import (
    lidar_min_dist,
    velocity_toward_target, reward_forward_speed,
    reward_heading_alignment, penalize_backwards_movement,
    progress_delta, forward_clearance,
    waypoint_reached, goal_reached,
    wall_proximity, obstacle_avoidance,
    spinning_penalty, lateral_clearance_reward,
    fell_off,
)

_PATHS_FILE = os.path.join(os.path.dirname(__file__),
                            "../../../../../../data/paths/paths.npy")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        rel_goal     = ObsTerm(func=rel_goal_obs)
        next_wp      = ObsTerm(func=next_wp_obs)
        next_next_wp = ObsTerm(func=next_next_wp_obs)
        heading      = ObsTerm(func=heading_error_obs)
        wheel_vel    = ObsTerm(func=wheel_velocities_obs)
        last_action  = ObsTerm(func=mdp.last_action)
        lidar        = ObsTerm(func=lidar_ranges)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


def reset_robot_to_path_start(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    n         = len(env_ids)
    fixed     = getattr(env.cfg, "fixed_path_idx", None)
    if fixed is not None:
        path_ids = torch.full((n,), fixed, dtype=torch.long, device=env.device)
    else:
        path_ids = torch.randint(1, env._n_paths, (n,), device=env.device)
    env._path_idx[env_ids] = path_ids

    final_idx = env._paths.shape[1] - 1
    starts    = env._paths[path_ids, 0, :]

    dists_all  = torch.norm(env._paths[path_ids] - starts.unsqueeze(1), dim=-1)
    far_enough = dists_all >= 0.2
    has_far    = far_enough.any(dim=-1)
    first_far  = far_enough.long().argmax(dim=-1)
    first_far  = torch.where(has_far, first_far, torch.full_like(first_far, final_idx))
    ref_wp = env._paths[path_ids][
        torch.arange(n, device=env.device).unsqueeze(-1),
        first_far.unsqueeze(-1),
        torch.arange(2, device=env.device).unsqueeze(0),
    ].squeeze(1)

    yaw_to_ref     = torch.atan2(ref_wp[:, 1] - starts[:, 1], ref_wp[:, 0] - starts[:, 0])
    env_origins_xy = env.scene.env_origins[env_ids, :2]

    robot      = env.scene["robot"]
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, 0] = starts[:, 0] + env_origins_xy[:, 0]
    root_state[:, 1] = starts[:, 1] + env_origins_xy[:, 1]
    root_state[:, 2] = 0.1
    root_state[:, 3] = torch.cos(yaw_to_ref / 2.0)
    root_state[:, 4] = 0.0
    root_state[:, 5] = 0.0
    root_state[:, 6] = torch.sin(yaw_to_ref / 2.0)
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

    if len(env_ids) > 0:
        i = 0
        print(f"[SPAWN] env {env_ids[i].item():02d} | "
              f"start=({starts[i,0].item():.2f}, {starts[i,1].item():.2f}) | "
              f"yaw={math.degrees(yaw_to_ref[i].item()):+.1f}°")

    env._waypoint_idx[env_ids] = 1
    env._goal_pos[env_ids]     = env._paths[path_ids, 1, :] + env_origins_xy

    if hasattr(env, "_final_goal_pos"):
        env._final_goal_pos[env_ids] = env._paths[path_ids, final_idx, :] + env_origins_xy

    if hasattr(env, "_prev_goal_dist"):
        env._prev_goal_dist[env_ids] = torch.norm(
            env._goal_pos[env_ids] - (starts + env_origins_xy), dim=-1
        )


@configclass
class EventCfg:
    reset_robot = EventTerm(func=reset_robot_to_path_start, mode="reset")
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


def goal_reached_termination(env: ManagerBasedEnv) -> torch.Tensor:
    if not hasattr(env, "_final_goal_pos"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    dist = torch.norm(env._final_goal_pos - env.scene["robot"].data.root_pos_w[:, :2], dim=-1)
    return dist < env.cfg.goal_reach_threshold


def collision_termination(env: ManagerBasedEnv) -> torch.Tensor:
    return lidar_min_dist(env) < env.cfg.collision_threshold


def fell_off_termination(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_pos_w[:, 2] < -0.5


@configclass
class TerminationsCfg:
    time_out     = DoneTerm(func=mdp.time_out,             time_out=True)
    goal_reached = DoneTerm(func=goal_reached_termination, time_out=False)
    # collision    = DoneTerm(func=collision_termination,    time_out=False)
    fell_off     = DoneTerm(func=fell_off_termination,     time_out=False)


@configclass
class RewardsCfg:
    progress           = RewTerm(func=velocity_toward_target,      weight=3.0)
    progress_delta     = RewTerm(func=progress_delta,              weight=5.0)
    speed_bonus        = RewTerm(func=reward_forward_speed,        weight=3.0)
    heading            = RewTerm(func=reward_heading_alignment,    weight=4.0)
    forward_clearance  = RewTerm(func=forward_clearance,           weight=2.0)
    waypoint_reached   = RewTerm(func=waypoint_reached,            weight=20.0)
    goal_reached       = RewTerm(func=goal_reached,                weight=100.0)
    obstacle_avoidance = RewTerm(func=obstacle_avoidance,          weight=-6.0)   # was -1.5
    wall_proximity     = RewTerm(func=wall_proximity,              weight=-8.0)   # was -3.0
    spinning_penalty   = RewTerm(func=spinning_penalty,            weight=-5.0)
    lateral_clearance  = RewTerm(func=lateral_clearance_reward,    weight=12.0)  # was 6.0
    fell_off           = RewTerm(func=fell_off,                    weight=50.0)
    backward           = RewTerm(func=penalize_backwards_movement, weight=-3.0)
    action_rate        = RewTerm(func=mdp.action_rate_l2,          weight=-0.1)   # was -0.5 — allow fast obstacle-avoidance turns
    alive              = RewTerm(func=mdp.is_alive,                weight=-0.05)


@configclass
class IctBotNavRlEnvCfg(ManagerBasedRLEnvCfg):
    scene:        IctBotNavSceneCfg = IctBotNavSceneCfg(num_envs=4, env_spacing=15.0)
    observations: ObservationsCfg  = ObservationsCfg()
    actions:      ActionsCfg       = ActionsCfg()
    events:       EventCfg         = EventCfg()
    rewards:      RewardsCfg       = RewardsCfg()
    terminations: TerminationsCfg  = TerminationsCfg()

    paths_file:               str        = _PATHS_FILE
    fixed_path_idx:           int | None = None
    max_waypoints:            int        = 100
    waypoint_reach_threshold: float      = 0.4
    goal_reach_threshold:     float      = 0.5
    collision_threshold:      float      = 0.20
    proximity_threshold:      float      = 0.5
    log_interval_steps:       int        = 256

    def __post_init__(self):
        self.decimation          = 5
        self.episode_length_s    = 40.0  # was 25.0 — longer paths need more time
        self.sim.dt              = 1.0 / 100.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
