from __future__ import annotations

"""
Phase 1 — Plain ground goal navigation.

Flat ground, no walls, no obstacles.
Robot spawns at a random position and must reach a random goal.
Observation: goal direction + heading + wheel velocities (no waypoints, no LiDAR).
Rewards: purely navigation-focused — reach goal, make progress, face goal.

Train here first, then fine-tune on office + obstacles.
"""

import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg
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
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from .robot_cfg import ICT_BOT_CFG
from .actions import ActionsCfg
from .observations import rel_goal_obs, heading_error_obs, wheel_velocities_obs
from .rewards import (
    velocity_toward_target, reward_forward_speed,
    reward_heading_alignment, penalize_backwards_movement,
    progress_delta, goal_reached, goal_proximity, fell_off,
    spinning_penalty, penalize_away_from_goal, time_efficiency_bonus,
)


# ── Scene: flat ground only ────────────────────────────────────────────────────

@configclass
class PlainGroundSceneCfg(InteractiveSceneCfg):
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.8, 0.8, 0.8)),
    )
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    robot: ArticulationCfg = ICT_BOT_CFG


# ── Observations: no LiDAR, no waypoints ──────────────────────────────────────

@configclass
class PlainObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        rel_goal   = ObsTerm(func=rel_goal_obs)          # (N, 2) goal in body frame
        heading    = ObsTerm(func=heading_error_obs)     # (N, 2) sin/cos heading error
        wheel_vel  = ObsTerm(func=wheel_velocities_obs)  # (N, 2) wheel velocities
        last_action = ObsTerm(func=mdp.last_action)      # (N, 2) previous action

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ── Reset: random position + random goal ──────────────────────────────────────

def reset_robot_random(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Spawn robot at random position, set random goal within spawn_radius."""
    n           = len(env_ids)
    device      = env.device
    half        = env.cfg.spawn_half_extent
    goal_radius = env.cfg.goal_radius

    # Random spawn position (flat ground — z fixed at 0.1)
    xy          = (torch.rand(n, 2, device=device) - 0.5) * 2 * half
    yaw         = (torch.rand(n, device=device) - 0.5) * 2 * torch.pi

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

    # Random goal: placed at random angle + distance from spawn
    angle       = torch.rand(n, device=device) * 2 * torch.pi
    dist        = goal_radius * 0.5 + torch.rand(n, device=device) * goal_radius * 0.5
    goal_local  = torch.stack([dist * torch.cos(angle), dist * torch.sin(angle)], dim=-1)
    goal_world  = goal_local + root_state[:, :2]

    env._goal_pos[env_ids]       = goal_world
    env._final_goal_pos[env_ids] = goal_world
    env._prev_goal_dist[env_ids] = dist


# ── Rewards ────────────────────────────────────────────────────────────────────

@configclass
class PlainRewardsCfg:
    progress         = RewTerm(func=velocity_toward_target,      weight=5.0)
    progress_delta   = RewTerm(func=progress_delta,              weight=4.0)
    speed_bonus      = RewTerm(func=reward_forward_speed,        weight=3.0)
    heading          = RewTerm(func=reward_heading_alignment,    weight=3.0)
    goal_reached     = RewTerm(func=goal_reached,                weight=3000.0)
    time_efficiency  = RewTerm(func=time_efficiency_bonus,       weight=1500.0)  # bonus for reaching goal early
    goal_proximity   = RewTerm(func=goal_proximity,              weight=3.0)
    away_from_goal   = RewTerm(func=penalize_away_from_goal,     weight=-3.0)    # penalise driving away from goal
    spinning         = RewTerm(func=spinning_penalty,            weight=-1.0)
    backward         = RewTerm(func=penalize_backwards_movement, weight=-3.0)
    action_rate      = RewTerm(func=mdp.action_rate_l2,          weight=-0.2)    # smoother actions (finetune2)
    alive            = RewTerm(func=mdp.is_alive,                weight=-0.1)
    fell_off         = RewTerm(func=fell_off,                    weight=50.0)


# ── Terminations ───────────────────────────────────────────────────────────────

def goal_reached_termination(env: ManagerBasedEnv) -> torch.Tensor:
    if not hasattr(env, "_final_goal_pos"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    dist = torch.norm(env._final_goal_pos - env.scene["robot"].data.root_pos_w[:, :2], dim=-1)
    return dist < env.cfg.goal_reach_threshold


def fell_off_termination(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_pos_w[:, 2] < -0.5


@configclass
class PlainTerminationsCfg:
    time_out     = DoneTerm(func=mdp.time_out,             time_out=True)
    goal_reached = DoneTerm(func=goal_reached_termination, time_out=False)
    fell_off     = DoneTerm(func=fell_off_termination,     time_out=False)


# ── Events ─────────────────────────────────────────────────────────────────────

@configclass
class PlainEventCfg:
    reset_robot = EventTerm(func=reset_robot_random, mode="reset")
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


# ── Main config ────────────────────────────────────────────────────────────────

@configclass
class IctBotNavPlainCfg(ManagerBasedRLEnvCfg):
    scene:        PlainGroundSceneCfg  = PlainGroundSceneCfg(num_envs=4, env_spacing=10.0)
    observations: PlainObservationsCfg = PlainObservationsCfg()
    actions:      ActionsCfg           = ActionsCfg()
    events:       PlainEventCfg        = PlainEventCfg()
    rewards:      PlainRewardsCfg      = PlainRewardsCfg()
    terminations: PlainTerminationsCfg = PlainTerminationsCfg()

    # Task parameters
    spawn_half_extent:    float = 3.0   # robot spawns within ±3m of env origin
    goal_radius:          float = 4.0   # goal placed 2–4m from spawn
    goal_reach_threshold: float = 0.5
    proximity_threshold:  float = 0.5   # used by wall_proximity (not active here)
    log_interval_steps:   int   = 1000  # policy steps (per-env)

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
