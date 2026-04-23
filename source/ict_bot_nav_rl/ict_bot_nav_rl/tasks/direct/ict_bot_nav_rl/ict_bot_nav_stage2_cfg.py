from __future__ import annotations

"""
Stage 2 — Plain ground with 4 moving obstacles.

Fine-tunes directly from Stage 1 weights:
  - Same 89-dim obs layout (last 12 dims now real obstacle info, not zeros)
  - Same [256, 128] network — no architecture change
  - Same action space

Obstacle design:
  - 4 kinematic spheres (radius 0.25 m) per env
  - Moved programmatically each step (not driven by physics forces)
  - Speed 0.15–0.4 m/s, random direction, bounce off arena walls
  - Kinematic → analytical collision detection via _obstacle_pos_all

New reward vs Stage 1:
  - R_col  -200  any obstacle centre within collision_threshold of robot

Obs layout (81 dims):
  lidar_72          : 72
  dtg_htg_2         :  2
  robot_pos_vel_4   :  4  — [x/5, y/5, fwd_vel/0.3, ang_vel/2.0]
  obstacle_cp_3     :  3  — K=1 most dangerous [x/10, y/10, CP=0.5*Pc_ttc+0.5*Pc_dto]
"""

import math
import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
    lidar_ranges, dtg_htg_obs, robot_pos_vel_obs, cp_obstacle_obs,
)
from .rewards import (
    forward_clearance, goal_proximity, goal_reached, fell_off, lateral_clearance_reward, obstacle_avoidance, progress_delta,
    r_dtg, r_htg, r_local_wp, r_col,
    spinning_penalty, penalize_backwards_movement,
)

# Obstacle constants
N_OBSTACLES   = 4
OBS_RADIUS    = 0.25   # cylinder radius (m) — used for collision threshold
OBS_CYL_H    = 1.0    # cylinder visual height (m)
OBS_HEIGHT    = 0.0    # z of cylinder BASE — sits flat on ground
MAX_LOCAL_WPS = 15
WP_SPACING    = 0.6


# ── Scene ─────────────────────────────────────────────────────────────────────

def _obs_rigid_cfg(prim_path: str, init_x: float, init_y: float) -> RigidObjectCfg:
    """Helper: kinematic sphere obstacle at a given initial XY (env-relative)."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.SphereCfg(
            radius=OBS_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,         # we control position; not physics-driven
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.3, 0.1), opacity=0.85,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(init_x, init_y, OBS_HEIGHT),
        ),
    )


@configclass
class Stage2SceneCfg(InteractiveSceneCfg):
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.8, 0.8, 0.8)),
    )
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    robot: ArticulationCfg = ICT_BOT_CFG

    lidar: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/ICTBot/base_scan",
        mesh_prim_paths=["/World/Ground"],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=5.0,
        ),
        max_distance=10.0,
        debug_vis=False,
    )

    # 4 kinematic obstacles — initial positions spread around env origin
    obstacle_0: RigidObjectCfg = _obs_rigid_cfg("{ENV_REGEX_NS}/Obstacle0",  2.0,  0.0)
    obstacle_1: RigidObjectCfg = _obs_rigid_cfg("{ENV_REGEX_NS}/Obstacle1", -2.0,  0.0)
    obstacle_2: RigidObjectCfg = _obs_rigid_cfg("{ENV_REGEX_NS}/Obstacle2",  0.0,  2.0)
    obstacle_3: RigidObjectCfg = _obs_rigid_cfg("{ENV_REGEX_NS}/Obstacle3",  0.0, -2.0)


# ── Observations (80 dims — same layout as Stage 1) ───────────────────────────

@configclass
class Stage2ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        lidar        = ObsTerm(func=lidar_ranges)            # (N, 72)
        dtg_htg      = ObsTerm(func=dtg_htg_obs)            # (N,  2)
        robot_state  = ObsTerm(func=robot_pos_vel_obs)      # (N,  3)
        obstacle_obs = ObsTerm(func=cp_obstacle_obs)         # (N, 12) K=4 sorted by CP

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ── Reset ─────────────────────────────────────────────────────────────────────

_OBSTACLE_KEYS = ["obstacle_0", "obstacle_1", "obstacle_2", "obstacle_3"]


def reset_stage2(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Reset robot, goal, local waypoints, and obstacle positions/velocities."""
    n           = len(env_ids)
    device      = env.device
    half        = env.cfg.spawn_half_extent
    goal_radius = env.cfg.goal_radius

    # ── Robot ─────────────────────────────────────────────────────────────────
    xy  = (torch.rand(n, 2, device=device) - 0.5) * 2 * half
    yaw = (torch.rand(n, device=device) - 0.5) * 2 * math.pi

    robot       = env.scene["robot"]
    root_state  = robot.data.default_root_state[env_ids].clone()
    env_origins = env.scene.env_origins[env_ids, :2]

    root_state[:, 0] = xy[:, 0] + env_origins[:, 0]
    root_state[:, 1] = xy[:, 1] + env_origins[:, 1]
    root_state[:, 2] = 0.1
    root_state[:, 3] = torch.cos(yaw / 2)
    root_state[:, 4:7] = 0.0
    root_state[:, 6] = torch.sin(yaw / 2)
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

    # ── Goal ──────────────────────────────────────────────────────────────────
    angle   = torch.rand(n, device=device) * 2 * math.pi
    dist    = goal_radius * 0.5 + torch.rand(n, device=device) * goal_radius * 0.5
    start_w = root_state[:, :2]
    goal_w  = start_w + torch.stack(
        [dist * torch.cos(angle), dist * torch.sin(angle)], dim=-1
    )

    env._goal_pos[env_ids]       = goal_w
    env._final_goal_pos[env_ids] = goal_w
    env._prev_goal_dist[env_ids] = dist

    # ── Local waypoints ───────────────────────────────────────────────────────
    direction  = goal_w - start_w
    total_dist = torch.norm(direction, dim=-1)
    unit_dir   = direction / (total_dist.unsqueeze(-1) + 1e-6)

    wps = torch.zeros(n, MAX_LOCAL_WPS, 2, device=device)
    for i in range(MAX_LOCAL_WPS):
        d_i  = (i + 1) * WP_SPACING
        wp_i = start_w + unit_dir * d_i
        wps[:, i, :] = torch.where((d_i >= total_dist).unsqueeze(-1), goal_w, wp_i)

    env._local_wps[env_ids]    = wps
    env._local_wp_idx[env_ids] = 0
    env._local_wp_pos[env_ids] = wps[:, 0, :]

    # ── Heading error init ────────────────────────────────────────────────────
    diff_w        = torch.zeros(n, 3, device=device)
    diff_w[:, :2] = goal_w - start_w
    q_inv  = quat_inv(root_state[:, 3:7])
    local  = quat_apply(q_inv, diff_w)[:, :2]
    env._prev_heading_err[env_ids] = torch.abs(torch.atan2(local[:, 1], local[:, 0]))

    # ── Obstacles ─────────────────────────────────────────────────────────────
    bound = half - OBS_RADIUS - 0.2

    # direction robot → goal (compute once)
    start_w = root_state[:, :2]
    dir_vec = goal_w - start_w
    dist    = torch.norm(dir_vec, dim=-1, keepdim=True)
    unit_dir = dir_vec / (dist + 1e-6)

    # perpendicular direction
    perp_dir = torch.stack([-unit_dir[:, 1], unit_dir[:, 0]], dim=-1)

    base_fractions = torch.tensor([0.3, 0.5, 0.7, 0.9], device=device)

    for i, key in enumerate(_OBSTACLE_KEYS):

        # ── MIXED DIFFICULTY ──
        # use_path_blocking = torch.rand(n, device=device) > 0.3   # 70% path, 30% random
        use_path_blocking = torch.ones(n, dtype=torch.bool, device=device)

        pos = torch.zeros(n, 2, device=device)

        # PATH BLOCKING
        mask = use_path_blocking

        t = base_fractions[i] + (torch.rand(n, device=device) - 0.5) * 0.1
        t = torch.clamp(t, 0.2, 0.9)
        t = t.unsqueeze(-1)   # ✅ IMPORTANT

        center = start_w + unit_dir * (t * dist)
        offset = (torch.rand(n, device=device) - 0.5) * 2.0

        pos[mask] = center[mask] + perp_dir[mask] * offset[mask].unsqueeze(-1)

        # RANDOM
        pos[~mask] = env_origins[~mask] + (torch.rand((~mask).sum(), 2, device=device) - 0.5) * 2 * half # type: ignore

        # clamp to arena
        pos[:, 0] = torch.clamp(pos[:, 0],
                                env_origins[:, 0] - bound, env_origins[:, 0] + bound)
        pos[:, 1] = torch.clamp(pos[:, 1],
                                env_origins[:, 1] - bound, env_origins[:, 1] + bound)

        env._obs_pos[key][env_ids] = pos

        # ── VELOCITY DESIGN (Stage 2.5 CORE) ──
        speed = env.cfg.obs_speed_min + torch.rand(n, device=device) * (
            env.cfg.obs_speed_max - env.cfg.obs_speed_min
        )

        mode = torch.rand(n, device=device)

        vel = torch.zeros(n, 2, device=device)

        # CROSSING
        mask1 = mode < 0.5
        vel[mask1] = perp_dir[mask1] * speed[mask1].unsqueeze(-1)

        sign = torch.where(torch.rand(n, device=device) > 0.5, 1.0, -1.0)
        direction_flip = 1.0 if (i % 2 == 0) else -1.0
        vel[mask1] *= sign[mask1].unsqueeze(-1) * direction_flip

        # RANDOM
        mask2 = (mode >= 0.5) & (mode < 0.8)
        angle = torch.rand(n, device=device) * 2 * math.pi
        vel[mask2] = torch.stack(
            [torch.cos(angle[mask2]), torch.sin(angle[mask2])], dim=-1
        ) * speed[mask2].unsqueeze(-1)

        # GOAL DIRECTED
        mask3 = mode >= 0.8
        rand_goal = env_origins + (torch.rand(n, 2, device=device) - 0.5) * 2 * half
        dir_o = rand_goal - pos
        dir_o = dir_o / (torch.norm(dir_o, dim=-1, keepdim=True) + 1e-6)
        vel[mask3] = dir_o[mask3] * speed[mask3].unsqueeze(-1)

        env._obs_vel[key][env_ids] = vel

        # write pose to sim
        pose = torch.zeros(n, 7, device=device)
        pose[:, 0] = pos[:, 0]
        pose[:, 1] = pos[:, 1]
        pose[:, 2] = OBS_HEIGHT
        pose[:, 3] = 1.0
        env.scene[key].write_root_pose_to_sim(pose, env_ids)


# ── Rewards ───────────────────────────────────────────────────────────────────

# @configclass
# class Stage2RewardsCfg:
#     step_alive  = RewTerm(func=mdp.is_alive,                weight=-2.0)
#     dtg         = RewTerm(func=r_dtg,                       weight=1.0)
#     htg         = RewTerm(func=r_htg,                       weight=1.0)
#     goal        = RewTerm(func=goal_reached,                weight=200.0)
#     local_wp    = RewTerm(func=r_local_wp,                  weight=200.0)  # paper: R_wp=200 critical for convergence
#     collision   = RewTerm(func=r_col,                       weight=200.0)    # r_col returns -1 → 200×(-1) = -200 penalty
#     spinning    = RewTerm(func=spinning_penalty,            weight=-1.0)
#     backward    = RewTerm(func=penalize_backwards_movement, weight=-2.0)
#     fell_off    = RewTerm(func=fell_off,                    weight=50.0)
#     avoid     =   RewTerm(func=obstacle_avoidance, weight=-8.0)
#     clearance =   RewTerm(func=forward_clearance, weight=4.0)
#     lateral =     RewTerm(func=lateral_clearance_reward, weight=6.0)

class Stage2RewardsCfg:
    step_alive  = RewTerm(func=mdp.is_alive, weight=-2.0)

    # navigation
    progress    = RewTerm(func=progress_delta, weight=2.0)
    htg         = RewTerm(func=r_htg, weight=1.0)
    goal_dense  = RewTerm(func=goal_proximity, weight=5.0)

    # sparse
    goal        = RewTerm(func=goal_reached, weight=200.0)
    local_wp    = RewTerm(func=r_local_wp, weight=200.0)

    # obstacle intelligence (🔥 key)
    avoid       = RewTerm(func=obstacle_avoidance, weight=-12.0)
    clearance   = RewTerm(func=forward_clearance, weight=6.0)
    lateral     = RewTerm(func=lateral_clearance_reward, weight=6.0)

    # penalties
    collision   = RewTerm(func=r_col, weight=50.0)
    spinning    = RewTerm(func=spinning_penalty, weight=-3.0)
    backward    = RewTerm(func=penalize_backwards_movement, weight=-2.0)
    fell_off    = RewTerm(func=fell_off, weight=80.0)


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


def collision_termination(env: ManagerBasedEnv) -> torch.Tensor:
    """Terminate when any obstacle centre is within collision_threshold of the robot.

    Kinematic bodies moved via write_root_pose_to_sim don't always generate
    reliable PhysX contact impulses, so we use analytical distance instead.
    This gives the same training signal as a real collision: -200 reward + episode reset.
    """
    if not hasattr(env, "_obstacle_pos_all"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    robot_xy = env.scene["robot"].data.root_pos_w[:, :2]
    dists = torch.norm(env._obstacle_pos_all - robot_xy.unsqueeze(1), dim=-1)  # (N, K)
    return dists.min(dim=-1).values < env.cfg.collision_threshold


@configclass
class Stage2TerminationsCfg:
    time_out     = DoneTerm(func=mdp.time_out,              time_out=True)
    goal_reached = DoneTerm(func=goal_reached_termination,  time_out=False)
    fell_off     = DoneTerm(func=fell_off_termination,      time_out=False)
    collision    = DoneTerm(func=collision_termination,     time_out=False)


# ── Events ────────────────────────────────────────────────────────────────────

@configclass
class Stage2EventCfg:
    reset_all = EventTerm(func=reset_stage2, mode="reset")
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
class IctBotNavStage2Cfg(ManagerBasedRLEnvCfg):
    scene:        Stage2SceneCfg        = Stage2SceneCfg(num_envs=4, env_spacing=12.0)
    observations: Stage2ObservationsCfg = Stage2ObservationsCfg()
    actions:      ActionsCfg            = ActionsCfg()
    events:       Stage2EventCfg        = Stage2EventCfg()
    rewards:      Stage2RewardsCfg      = Stage2RewardsCfg()
    terminations: Stage2TerminationsCfg = Stage2TerminationsCfg()

    spawn_half_extent:    float = 3.0
    goal_radius:          float = 4.0
    goal_reach_threshold: float = 0.5
    wp_reach_threshold:   float = 0.5
    collision_threshold:  float = OBS_RADIUS + 0.15   # 0.45 m (obstacle radius + robot body)
    proximity_threshold:  float = 0.5
    obs_speed_min:        float = 0.1   # m/s
    obs_speed_max:        float = 0.35    # m/s — paper: max 0.2 m/s
    log_interval_steps:   int   = 2000

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
