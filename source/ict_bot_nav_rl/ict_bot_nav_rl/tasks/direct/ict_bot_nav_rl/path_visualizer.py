from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


def _disc(prim_path: str, radius: float, color: tuple) -> VisualizationMarkers:
    return VisualizationMarkers(VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "cylinder": sim_utils.CylinderCfg(
                radius=radius,
                height=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color, opacity=0.9,
                ),
            ),
        },
    ))


class PathVisualizer:
    def __init__(self, device: str):
        self.device = device
        self._waypoints  = _disc("/Visuals/PathWaypoints", radius=0.07, color=(1.0, 0.0, 0.0))
        self._subgoal    = _disc("/Visuals/SubGoal",       radius=0.12, color=(0.0, 1.0, 0.0))
        self._finalgoal  = _disc("/Visuals/FinalGoal",     radius=0.16, color=(1.0, 1.0, 0.0))

    def update(
        self,
        paths: torch.Tensor,
        path_idx: torch.Tensor,
        n_real_wps: torch.Tensor,
        goal_pos: torch.Tensor,
        final_goal_pos: torch.Tensor,
    ) -> None:
        # Paths are world frame (env_spacing=0, no origin offset needed)
        pid    = int(path_idx[0])
        n_real = int(n_real_wps[pid])
        wp_pos = torch.zeros(n_real, 3, device=self.device)
        wp_pos[:, :2] = paths[pid, :n_real, :]
        wp_pos[:, 2]  = 0.01
        self._waypoints.visualize(wp_pos)

        # Current sub-goal per env (green disc)
        sg_pos = torch.zeros(goal_pos.shape[0], 3, device=self.device)
        sg_pos[:, :2] = goal_pos
        sg_pos[:, 2]  = 0.01
        self._subgoal.visualize(sg_pos)

        # Final goal per env (yellow disc)
        fg_pos = torch.zeros(final_goal_pos.shape[0], 3, device=self.device)
        fg_pos[:, :2] = final_goal_pos
        fg_pos[:, 2]  = 0.01
        self._finalgoal.visualize(fg_pos)
