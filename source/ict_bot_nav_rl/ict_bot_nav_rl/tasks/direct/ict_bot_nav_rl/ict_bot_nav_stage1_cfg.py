from __future__ import annotations

"""
Stage 1 — Plain ground with full obs architecture.

Goal: train basic navigation with the SAME obs dimensions as Stage 2 (moving
obstacles) so Stage 2 can fine-tune directly from Stage 1 weights.

Obs layout (89 dims):
  lidar_72          : 72  — LiDAR ranges normalised to [0,1], all ~1.0 (open space)
  dtg_htg_2         :  2  — [DTG/10, cos(heading_error)] to final goal
  robot_pos_vel_3   :  3  — [x/5, y/5, fwd_vel/0.3] relative to env origin
  obstacle_zeros_12 : 12  — zeros placeholder (K=4 × 3); Stage 2 fills with real obs
  ─────────────────────────
  Total             : 89

Rewards:
  R_step   -2.0  every step          (mdp.is_alive × -2)
  R_dtg    +1.0  if DTG improved     (r_dtg)
  R_htg    +1.0  if HTG improved     (r_htg)
  R_goal  +200   final goal reached  (goal_reached)
  R_wp    +200   local waypoint hit  (r_local_wp, waypoints every 0.6 m)

Goal: randomly placed 2–4 m from spawn each episode (re-randomised at reset).
"""

import math
import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedEnv
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_inv, quat_apply

from .robot_cfg import ICT_BOT_CFG
from .actions import ActionsCfg
from .observations import (
    lidar_ranges, dtg_htg_obs, robot_pos_vel_obs, obstacle_placeholder_obs,
)
from .rewards import (
    goal_reached, fell_off,
    r_dtg, r_htg, r_local_wp,
    spinning_penalty, penalize_backwards_movement,
)

# Local waypoint parameters
MAX_LOCAL_WPS = 15
WP_SPACING    = 0.6   # metres between successive waypoints along start→goal


# ── Scene ─────────────────────────────────────────────────────────────────────

@configclass
class Stage1SceneCfg(InteractiveSceneCfg):
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.8, 0.8, 0.8)),
    )
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    robot: ArticulationCfg = ICT_BOT_CFG

    # 72-ray horizontal LiDAR — same resolution as the office env.
    # In plain ground (no walls, no obstacles) all rays return max_distance.
    # The policy learns that all-ones = clear space; Stage 2 introduces real hits.
    lidar: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/ICTBot/base_scan",
        mesh_prim_paths=["/World/Ground"],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=5.0,   # 360 / 5 = 72 rays
        ),
        max_distance=10.0,
        debug_vis=False,
    )


# ── Observations (80 dims total) ──────────────────────────────────────────────

@configclass
class Stage1ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        lidar        = ObsTerm(func=lidar_ranges)             # (N, 72)
        dtg_htg      = ObsTerm(func=dtg_htg_obs)             # (N,  2)
        robot_state  = ObsTerm(func=robot_pos_vel_obs)       # (N,  3)
        obstacle_obs = ObsTerm(func=obstacle_placeholder_obs) # (N,  3)  zeros

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ── Reset ─────────────────────────────────────────────────────────────────────

def reset_stage1(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Spawn robot randomly; place goal randomly; generate local waypoints."""
    n           = len(env_ids)
    device      = env.device
    half        = env.cfg.spawn_half_extent
    goal_radius = env.cfg.goal_radius

    # ── Robot spawn ───────────────────────────────────────────────────────────
    xy  = (torch.rand(n, 2, device=device) - 0.5) * 2 * half
    yaw = (torch.rand(n, device=device) - 0.5) * 2 * math.pi

    robot       = env.scene["robot"]
    root_state  = robot.data.default_root_state[env_ids].clone()
    env_origins = env.scene.env_origins[env_ids, :2]

    root_state[:, 0] = xy[:, 0] + env_origins[:, 0]
    root_state[:, 1] = xy[:, 1] + env_origins[:, 1]
    root_state[:, 2] = 0.1
    root_state[:, 3] = torch.cos(yaw / 2)
    root_state[:, 4] = 0.0
    root_state[:, 5] = 0.0
    root_state[:, 6] = torch.sin(yaw / 2)
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

    # ── Goal ──────────────────────────────────────────────────────────────────
    angle      = torch.rand(n, device=device) * 2 * math.pi
    dist       = goal_radius * 0.5 + torch.rand(n, device=device) * goal_radius * 0.5
    start_w    = root_state[:, :2]                                 # world frame
    goal_w     = start_w + torch.stack(
        [dist * torch.cos(angle), dist * torch.sin(angle)], dim=-1
    )

    env._goal_pos[env_ids]       = goal_w
    env._final_goal_pos[env_ids] = goal_w
    env._prev_goal_dist[env_ids] = dist

    # ── Local waypoints every WP_SPACING metres along straight line ───────────
    direction  = goal_w - start_w                                  # (n, 2)
    total_dist = torch.norm(direction, dim=-1)                     # (n,)
    unit_dir   = direction / (total_dist.unsqueeze(-1) + 1e-6)    # (n, 2)

    wps = torch.zeros(n, MAX_LOCAL_WPS, 2, device=device)
    for i in range(MAX_LOCAL_WPS):
        d_i  = (i + 1) * WP_SPACING
        wp_i = start_w + unit_dir * d_i
        # If this spacing overshoots the goal, clamp to goal
        beyond = (d_i >= total_dist)
        wps[:, i, :] = torch.where(beyond.unsqueeze(-1), goal_w, wp_i)

    env._local_wps[env_ids]    = wps
    env._local_wp_idx[env_ids] = 0
    env._local_wp_pos[env_ids] = wps[:, 0, :]

    # ── Initialise heading error for r_htg ────────────────────────────────────
    diff_w        = torch.zeros(n, 3, device=device)
    diff_w[:, :2] = goal_w - start_w
    q_inv  = quat_inv(root_state[:, 3:7])
    local  = quat_apply(q_inv, diff_w)[:, :2]
    angle_h = torch.atan2(local[:, 1], local[:, 0])
    env._prev_heading_err[env_ids] = torch.abs(angle_h)


# ── Rewards ───────────────────────────────────────────────────────────────────

@configclass
class Stage1RewardsCfg:
    # Per-step existence penalty forces the robot to move efficiently
    step_alive  = RewTerm(func=mdp.is_alive,                weight=-2.0)
    # Dense progress signals
    dtg         = RewTerm(func=r_dtg,                       weight=1.0)
    htg         = RewTerm(func=r_htg,                       weight=1.0)
    # Sparse terminal rewards
    goal        = RewTerm(func=goal_reached,                weight=200.0)
    local_wp    = RewTerm(func=r_local_wp,                  weight=20.0)   # lower than goal to avoid farming
    # Behaviour shaping
    spinning    = RewTerm(func=spinning_penalty,            weight=-1.0)
    backward    = RewTerm(func=penalize_backwards_movement, weight=-2.0)
    action_rate = RewTerm(func=mdp.action_rate_l2,          weight=-0.1)
    fell_off    = RewTerm(func=fell_off,                    weight=50.0)


# ── Terminations ──────────────────────────────────────────────────────────────

def goal_reached_termination(env: ManagerBasedEnv) -> torch.Tensor:
    if not hasattr(env, "_final_goal_pos"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    dist = torch.norm(
        env._final_goal_pos - env.scene["robot"].data.root_pos_w[:, :2], dim=-1
    )
    return dist < env.cfg.goal_reach_threshold


def fell_off_termination(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_pos_w[:, 2] < -0.5


@configclass
class Stage1TerminationsCfg:
    time_out     = DoneTerm(func=mdp.time_out,             time_out=True)
    goal_reached = DoneTerm(func=goal_reached_termination, time_out=False)
    fell_off     = DoneTerm(func=fell_off_termination,     time_out=False)


# ── Events ────────────────────────────────────────────────────────────────────

@configclass
class Stage1EventCfg:
    reset_robot = EventTerm(func=reset_stage1, mode="reset")
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


# ── Main config ───────────────────────────────────────────────────────────────

@configclass
class IctBotNavStage1Cfg(ManagerBasedRLEnvCfg):
    scene:        Stage1SceneCfg        = Stage1SceneCfg(num_envs=4, env_spacing=12.0)
    observations: Stage1ObservationsCfg = Stage1ObservationsCfg()
    actions:      ActionsCfg            = ActionsCfg()
    events:       Stage1EventCfg        = Stage1EventCfg()
    rewards:      Stage1RewardsCfg      = Stage1RewardsCfg()
    terminations: Stage1TerminationsCfg = Stage1TerminationsCfg()

    # Task parameters
    spawn_half_extent:    float = 3.0   # robot spawns within ±3 m of env origin
    goal_radius:          float = 4.0   # goal placed 2–4 m from spawn
    goal_reach_threshold: float = 0.5
    wp_reach_threshold:   float = 0.5   # same threshold for local waypoints
    proximity_threshold:  float = 0.5   # kept for shared reward function signatures
    log_interval_steps:   int   = 1000

    def __post_init__(self):
        self.decimation          = 5
        self.episode_length_s    = 40.0
        self.sim.dt              = 1.0 / 100.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
