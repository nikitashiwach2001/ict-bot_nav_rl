from __future__ import annotations

import math
import os
import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)

from .actions import ActionsCfg
from .observations import (
    lidar_ranges, rel_goal_obs, next_wp_obs,
    next_next_wp_obs, heading_error_obs, wheel_velocities_obs,
)
from .rewards import (
    lidar_min_dist, velocity_toward_target, reward_forward_speed,
    reward_heading_alignment, penalize_backwards_movement,
    waypoint_reached, goal_reached, collision, fell_off,
)

_ICT_BOT_USD = "/home/user/Documents/ict_bot_nav_rl/urdf/ict_bot/ict_bot.usd"
_OFFICE_USD  = "/home/user/Documents/ict_bot_nav_rl/data/maps/office_env.usd"
_PATHS_FILE  = os.path.join(os.path.dirname(__file__),
                             "../../../../../../data/paths/single_path.npy")

ICT_BOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/ICTBot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_ICT_BOT_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=0.5,
            max_angular_velocity=6.25,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),
        joint_pos={"left_wheel_joint": 0.0, "right_wheel_joint": 0.0},
        joint_vel={"left_wheel_joint": 0.0, "right_wheel_joint": 0.0},
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=6.0,
            stiffness=0.0,
            damping=5.0,
        ),
    },
)

LIDAR_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/ICTBot/base_scan",
    mesh_prim_paths=["/World/OfficeEnv/Walls"],
    pattern_cfg=patterns.LidarPatternCfg(
        channels=1,
        vertical_fov_range=(0.0, 0.0),
        horizontal_fov_range=(-180.0, 180.0),
        horizontal_res=5.0,
    ),
    max_distance=10.0,
    drift_range=(-0.005, 0.005),
    debug_vis=False,
)


@configclass
class IctBotNavSceneCfg(InteractiveSceneCfg):
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.8, 0.8, 0.8)),
    )
    office_env = AssetBaseCfg(
        prim_path="/World/OfficeEnv",
        spawn=sim_utils.UsdFileCfg(usd_path=_OFFICE_USD),
    )
    robot: ArticulationCfg = ICT_BOT_CFG
    lidar: RayCasterCfg    = LIDAR_CFG


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
    path_ids  = torch.randint(0, env._n_paths, (n,), device=env.device)
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

    yaw_to_ref = torch.atan2(ref_wp[:, 1] - starts[:, 1], ref_wp[:, 0] - starts[:, 0])
    spawn_yaw  = yaw_to_ref

    robot      = env.scene["robot"]
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, 0] = starts[:, 0]
    root_state[:, 1] = starts[:, 1]
    root_state[:, 2] = 0.1
    root_state[:, 3] = torch.cos(spawn_yaw / 2.0)
    root_state[:, 4] = 0.0
    root_state[:, 5] = 0.0
    root_state[:, 6] = torch.sin(spawn_yaw / 2.0)
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

    if len(env_ids) > 0:
        i = 0
        print(f"[SPAWN] env {env_ids[i].item():02d} | "
              f"start=({starts[i,0].item():.2f}, {starts[i,1].item():.2f}) | "
              f"spawn_yaw={math.degrees(spawn_yaw[i].item()):+.1f}°")

    env._waypoint_idx[env_ids] = 1
    env._goal_pos[env_ids]     = env._paths[path_ids, 1, :]

    if hasattr(env, "_final_goal_pos"):
        env._final_goal_pos[env_ids] = env._paths[path_ids, final_idx, :]

    if hasattr(env, "_prev_goal_dist"):
        env._prev_goal_dist[env_ids] = torch.norm(env._goal_pos[env_ids] - starts, dim=-1)


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
    collision    = DoneTerm(func=collision_termination,    time_out=False)
    fell_off     = DoneTerm(func=fell_off_termination,     time_out=False)


@configclass
class RewardsCfg:
    progress          = RewTerm(func=velocity_toward_target,      weight=1.0)
    speed_bonus       = RewTerm(func=reward_forward_speed,        weight=50.0)
    heading           = RewTerm(func=reward_heading_alignment,    weight=2.0)
    backward          = RewTerm(func=penalize_backwards_movement, weight=-5.0)
    waypoint_reached  = RewTerm(func=waypoint_reached,            weight=100.0)
    goal_reached      = RewTerm(func=goal_reached,                weight=500.0)
    collision    = RewTerm(func=collision,                   weight=50.0)
    fell_off     = RewTerm(func=fell_off,                    weight=200.0)
    action_rate  = RewTerm(func=mdp.action_rate_l2,         weight=-0.5)
    alive        = RewTerm(func=mdp.is_alive,               weight=-1.0)


@configclass
class IctBotNavRlEnvCfg(ManagerBasedRLEnvCfg):
    scene:        IctBotNavSceneCfg = IctBotNavSceneCfg(num_envs=4, env_spacing=0.0)
    observations: ObservationsCfg  = ObservationsCfg()
    actions:      ActionsCfg       = ActionsCfg()
    events:       EventCfg         = EventCfg()
    rewards:      RewardsCfg       = RewardsCfg()
    terminations: TerminationsCfg  = TerminationsCfg()

    paths_file:               str   = _PATHS_FILE
    max_waypoints:            int   = 100
    waypoint_reach_threshold: float = 0.4
    goal_reach_threshold:     float = 0.5
    collision_threshold:      float = 0.15
    log_interval_steps:       int   = 256

    def __post_init__(self):
        self.decimation          = 5
        self.episode_length_s    = 25.0
        self.sim.dt              = 1.0 / 100.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
