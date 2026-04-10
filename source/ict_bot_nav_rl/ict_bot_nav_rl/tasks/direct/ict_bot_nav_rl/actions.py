from isaaclab.envs.mdp import JointVelocityActionCfg
from isaaclab.utils import configclass


@configclass
class ActionsCfg:
    # forward (+X in base_footprint, lidar side leading). wheel_radius=0.05m,
    # max speed ≈ 0.25 m/s.
    wheel_action: JointVelocityActionCfg = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_wheel_joint", "right_wheel_joint"],
        scale=5.0,
    )
