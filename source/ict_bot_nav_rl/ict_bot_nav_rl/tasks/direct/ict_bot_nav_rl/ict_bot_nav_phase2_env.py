from __future__ import annotations

"""
Phase 2 — Office environment with static walls, LiDAR-based avoidance.

Extends Phase 1 navigation with:
  - Office scene (walls + LiDAR raycaster)
  - 80-dim observations (8 nav + 72 LiDAR)
  - Collision termination and wall proximity penalties
  - Spawn/goal from pre-computed paths (valid open-space positions)
"""

import os
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from .ict_bot_nav_phase2_cfg import IctBotNavPhase2Cfg
from .episode_logger import EpisodeLogger

_PATHS_FILE = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../data/paths/paths.npy",
)


class IctBotNavPhase2Env(ManagerBasedRLEnv):
    cfg: IctBotNavPhase2Cfg

    def __init__(self, cfg: IctBotNavPhase2Cfg, render_mode: str | None = None, **kwargs):
        # Load paths before super().__init__ so reset_robot_phase2 can use them
        paths_np        = np.load(_PATHS_FILE)                          # (P, W, 2)
        self._paths     = torch.tensor(paths_np, dtype=torch.float32)   # loaded to CPU first
        self._n_paths   = self._paths.shape[0]

        super().__init__(cfg, render_mode, **kwargs)

        # Move paths to correct device after super().__init__ sets self.device
        self._paths = self._paths.to(self.device)

        n = self.num_envs
        self._goal_pos       = torch.zeros(n, 2, device=self.device)
        self._final_goal_pos = torch.zeros(n, 2, device=self.device)
        self._prev_goal_dist = torch.ones(n,    device=self.device)
        self._path_idx       = torch.zeros(n, dtype=torch.long, device=self.device)

        self._logger = EpisodeLogger(n, self.device, cfg.log_interval_steps,
                                     show_wp_stats=False)

        # Goal marker (GUI only)
        if self.sim.has_gui():
            self._goal_marker = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/Phase2Goal",
                markers={
                    "disc": sim_utils.CylinderCfg(
                        radius=0.25,
                        height=0.02,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 0.8, 1.0), opacity=0.9,
                        ),
                    ),
                },
            ))
        else:
            self._goal_marker = None

    def step(self, action):
        robot_xy = self.scene["robot"].data.root_pos_w[:, :2]
        new_ep   = self._logger.episode_steps == 0

        if new_ep.any():
            self._logger.record_step_start(
                new_ep, robot_xy,
                torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
                torch.zeros(1, 1, 2, device=self.device),
                self.cfg.goal_reach_threshold,
                torch.zeros(1, 1, dtype=torch.long, device=self.device),
                self.scene.env_origins[:, :2],
            )

        pre_step_xy    = robot_xy.clone()
        pre_final_goal = self._final_goal_pos.clone()   # save before reset overwrites

        obs, rew, terminated, truncated, extras = super().step(action)

        robot_xy = self.scene["robot"].data.root_pos_w[:, :2]
        self._logger.record_step(action, rew, robot_xy,
                                 self.scene["robot"].data, self.num_envs)

        done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            n_real   = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
            path_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._logger.record_done(
                done_ids, terminated, pre_step_xy,
                pre_final_goal, n_real, path_idx,
                self.cfg.goal_reach_threshold, extras,
            )

        if self._logger.should_log():
            self._logger.print_and_reset()

        if self._goal_marker is not None:
            marker_pos = torch.zeros(self.num_envs, 3, device=self.device)
            marker_pos[:, :2] = self._final_goal_pos
            marker_pos[:,  2] = 0.01
            self._goal_marker.visualize(marker_pos)

        return obs, rew, terminated, truncated, extras
