from __future__ import annotations

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils


def _wall(name: str, size: tuple, pos: tuple) -> AssetBaseCfg:
    # Global path so RayCaster can use a fixed mesh_prim_paths="/World/Walls"
    return AssetBaseCfg(
        prim_path=f"/World/Walls/{name}",
        spawn=sim_utils.MeshCuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


_H  = 0.30
_HH = _H / 2

# Wall boxes from office_map_partcial.pgm (world frame, metres)
_WALL_DATA = [
    ((6.000, 4.850, _H), ( 1.660, -1.735, _HH)),
    ((0.100, 0.050, _H), ( 3.260,  0.690, _HH)),
    ((0.100, 0.050, _H), ( 4.160,  0.690, _HH)),
    ((0.050, 7.800, _H), ( 6.310, -3.210, _HH)),
    ((0.050, 0.050, _H), (-0.165, -1.560, _HH)),
    ((0.050, 0.050, _H), ( 2.785, -1.560, _HH)),
    ((0.050, 0.050, _H), (-0.165, -2.760, _HH)),
    ((0.050, 0.050, _H), ( 2.785, -2.760, _HH)),
    ((0.050, 0.050, _H), ( 4.285, -2.760, _HH)),
    ((0.050, 0.150, _H), ( 4.660, -5.885, _HH)),
    ((0.050, 0.050, _H), ( 4.660, -6.135, _HH)),
    ((0.450, 0.050, _H), ( 4.335, -6.310, _HH)),
    ((0.050, 0.200, _H), ( 4.660, -6.810, _HH)),
    ((0.050, 0.650, _H), ( 6.310, -7.535, _HH)),
    ((0.050, 0.400, _H), ( 6.310, -8.160, _HH)),
    ((0.050, 0.200, _H), ( 6.310, -8.560, _HH)),
    ((0.050, 0.100, _H), ( 4.660, -8.610, _HH)),
    ((0.050, 0.100, _H), ( 6.310, -8.810, _HH)),
    ((0.050, 0.150, _H), ( 6.310, -9.085, _HH)),
    ((0.200, 0.050, _H), ( 4.560, -9.160, _HH)),
    ((0.050, 0.050, _H), ( 6.310, -9.335, _HH)),
    ((0.050, 0.050, _H), ( 4.660, -9.435, _HH)),
    ((0.050, 0.050, _H), ( 6.310, -9.585, _HH)),
    ((0.050, 0.050, _H), ( 6.310, -9.885, _HH)),
]


def make_scene_cfg(
    num_envs: int,
    robot_cfg: ArticulationCfg,
    lidar_cfg: RayCasterCfg,
    env_spacing: float = 0.0,
) -> IctBotNavSceneCfg:
    cfg = IctBotNavSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    cfg.robot = robot_cfg
    cfg.lidar  = lidar_cfg
    return cfg


@configclass
class IctBotNavSceneCfg(InteractiveSceneCfg):
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.8, 0.8, 0.8)),
    )
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    def __post_init__(self):
        super().__post_init__()
        for i, (size, pos) in enumerate(_WALL_DATA):
            setattr(self, f"wall_{i:02d}", _wall(f"wall_{i:02d}", size, pos))
