from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils import configclass

_ICT_BOT_USD = "/home/user/Documents/ict_bot_nav_rl/urdf/ict_bot/ict_bot.usd"
_OFFICE_USD  = "/home/user/Documents/ict_bot_nav_rl/data/maps/office_env.usd"


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


LIDAR_CFG = MultiMeshRayCasterCfg(
    prim_path="{ENV_REGEX_NS}/ICTBot/base_scan",
    mesh_prim_paths=[
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/OfficeEnv",
            is_shared=False,
            merge_prim_meshes=True,
            track_mesh_transforms=False,
        )
    ],
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
        prim_path="{ENV_REGEX_NS}/OfficeEnv",
        spawn=sim_utils.UsdFileCfg(usd_path=_OFFICE_USD),
    )
    robot: ArticulationCfg = ICT_BOT_CFG
    lidar: RayCasterCfg    = LIDAR_CFG
