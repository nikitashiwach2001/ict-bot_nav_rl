from __future__ import annotations

"""
Phase 2 — Office environment with static walls, LiDAR-based avoidance.

Observation : 79-dim = 8 (navigation) + 71 (LiDAR ranges)
Scene        : office USD + LiDAR raycaster
Reset        : spawn at path start, goal at path end (valid open-space positions)
New rewards  : graduated obstacle_avoidance + forward/lateral clearance
No collision termination — robot survives near-wall contacts to learn recovery.

Load warm-start checkpoint from scripts/transfer_weights_phase2.py before training.
"""

import os
import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedEnv
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)
from isaaclab.utils import configclass

from .robot_cfg import IctBotNavSceneCfg
from .actions import ActionsCfg
from .observations import (
    rel_goal_obs, heading_error_obs,
    wheel_velocities_obs, lidar_ranges,
)
from .rewards import (
    velocity_toward_target, reward_forward_speed,
    reward_heading_alignment, penalize_backwards_movement,
    progress_delta, goal_reached, goal_proximity, fell_off,
    spinning_penalty, penalize_away_from_goal, time_efficiency_bonus,
    obstacle_avoidance, forward_clearance, lateral_clearance_reward,
)

_PATHS_FILE = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../data/paths/paths.npy",
)


# ── Observations: 8 nav + 71 LiDAR = 79-dim ──────────────────────────────────

@configclass
class Phase2ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        rel_goal    = ObsTerm(func=rel_goal_obs)
        heading     = ObsTerm(func=heading_error_obs)
        wheel_vel   = ObsTerm(func=wheel_velocities_obs)
        last_action = ObsTerm(func=mdp.last_action)
        lidar       = ObsTerm(func=lidar_ranges)          # 71 rays at 5°/ray, 355° total

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ── Reset: spawn at path start, goal at path end ──────────────────────────────

def reset_robot_phase2(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Spawn at a random path start; goal = path end (valid office positions)."""
    n          = len(env_ids)
    device     = env.device
    path_ids   = torch.randint(0, env._n_paths, (n,), device=device)
    env._path_idx[env_ids] = path_ids

    starts = env._paths[path_ids, 0, :]                # (n, 2) local coords
    ends   = env._paths[path_ids, -1, :]               # (n, 2) local coords

    # Face toward second waypoint for natural spawn heading
    n_wp    = env._paths.shape[1]
    ref_idx = min(1, n_wp - 1)
    ref_wp  = env._paths[path_ids, ref_idx, :]
    yaw     = torch.atan2(ref_wp[:, 1] - starts[:, 1],
                          ref_wp[:, 0] - starts[:, 0])

    env_origins = env.scene.env_origins[env_ids, :2]
    robot       = env.scene["robot"]
    root_state  = robot.data.default_root_state[env_ids].clone()

    root_state[:, 0] = starts[:, 0] + env_origins[:, 0]
    root_state[:, 1] = starts[:, 1] + env_origins[:, 1]
    root_state[:, 2] = 0.1
    root_state[:, 3] = torch.cos(yaw / 2.0)
    root_state[:, 4] = 0.0
    root_state[:, 5] = 0.0
    root_state[:, 6] = torch.sin(yaw / 2.0)
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

    goal_world = ends + env_origins
    env._goal_pos[env_ids]       = goal_world
    env._final_goal_pos[env_ids] = goal_world
    env._prev_goal_dist[env_ids] = torch.norm(ends - starts, dim=-1)


# ── Rewards ───────────────────────────────────────────────────────────────────

@configclass
class Phase2RewardsCfg:
    # navigation (same as plain finetuned)
    progress        = RewTerm(func=velocity_toward_target,      weight=5.0)
    progress_delta  = RewTerm(func=progress_delta,              weight=4.0)
    speed_bonus     = RewTerm(func=reward_forward_speed,        weight=3.0)
    heading         = RewTerm(func=reward_heading_alignment,    weight=3.0)
    goal_reached    = RewTerm(func=goal_reached,                weight=3000.0)
    time_efficiency = RewTerm(func=time_efficiency_bonus,       weight=1500.0)
    goal_proximity  = RewTerm(func=goal_proximity,              weight=3.0)
    away_from_goal  = RewTerm(func=penalize_away_from_goal,     weight=-1.0)   # reduced: robot must detour around walls
    spinning        = RewTerm(func=spinning_penalty,            weight=-1.0)
    backward        = RewTerm(func=penalize_backwards_movement, weight=-3.0)
    action_rate     = RewTerm(func=mdp.action_rate_l2,          weight=-0.03)  # was -0.2; kept small so nav signal dominates
    alive           = RewTerm(func=mdp.is_alive,                weight=-0.1)
    fell_off        = RewTerm(func=fell_off,                    weight=50.0)
    # wall avoidance — graduated zones, no sudden-death termination
    obstacle_avoid  = RewTerm(func=obstacle_avoidance,          weight=-6.0)   # zones: warn(0.5-0.8m), pressure(0.2-0.5m), severe(<0.2m)
    fwd_clearance   = RewTerm(func=forward_clearance,           weight=2.0)    # reward open space directly ahead
    lat_clearance   = RewTerm(func=lateral_clearance_reward,    weight=5.0)    # steer toward more open side near walls


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
class Phase2TerminationsCfg:
    time_out     = DoneTerm(func=mdp.time_out,              time_out=True)
    goal_reached = DoneTerm(func=goal_reached_termination,  time_out=False)
    fell_off     = DoneTerm(func=fell_off_termination,      time_out=False)
    # collision_termination removed: graduated obstacle_avoidance replaces it.
    # Robot survives near-wall contacts so it can learn to escape narrow corridors.


# ── Events ────────────────────────────────────────────────────────────────────

@configclass
class Phase2EventCfg:
    reset_robot = EventTerm(func=reset_robot_phase2, mode="reset")
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
class IctBotNavPhase2Cfg(ManagerBasedRLEnvCfg):
    scene:        IctBotNavSceneCfg  = IctBotNavSceneCfg(num_envs=4, env_spacing=10.0)
    observations: Phase2ObservationsCfg = Phase2ObservationsCfg()
    actions:      ActionsCfg            = ActionsCfg()
    events:       Phase2EventCfg        = Phase2EventCfg()
    rewards:      Phase2RewardsCfg      = Phase2RewardsCfg()
    terminations: Phase2TerminationsCfg = Phase2TerminationsCfg()

    # task parameters
    goal_reach_threshold: float = 0.5
    proximity_threshold:  float = 0.8   # activates lateral_clearance_reward; matches obstacle_avoidance zone1
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
