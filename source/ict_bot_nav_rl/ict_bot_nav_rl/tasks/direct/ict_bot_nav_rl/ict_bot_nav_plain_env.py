from __future__ import annotations

"""
Phase 1 — Plain ground goal navigation environment.

Minimal env: no paths file, no waypoints, no obstacles, no LiDAR.
Robot spawns randomly, learns to reach a random goal via forward motion + turning.
"""

import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from .ict_bot_nav_plain_cfg import IctBotNavPlainCfg
from .episode_logger import EpisodeLogger


class IctBotNavPlainEnv(ManagerBasedRLEnv):
    cfg: IctBotNavPlainCfg

    def __init__(self, cfg: IctBotNavPlainCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n = self.num_envs

        # Navigation state (same names as IctBotNavRlEnv so fine-tuning loads cleanly)
        self._goal_pos       = torch.zeros(n, 2, device=self.device)
        self._final_goal_pos = torch.zeros(n, 2, device=self.device)
        self._prev_goal_dist = torch.ones(n,    device=self.device)

        # Minimal logger — no waypoint stats
        self._logger = EpisodeLogger(n, self.device, cfg.log_interval_steps,
                                     show_wp_stats=False)

        # Goal marker — bright green disc visible on ground (GUI only)
        if self.sim.has_gui():
            self._goal_marker = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/PlainGoal",
                markers={
                    "disc": sim_utils.CylinderCfg(
                        radius=0.25,
                        height=0.02,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.0), opacity=0.9,
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
                torch.zeros(1, 1, 2, device=self.device),   # dummy paths
                self.cfg.goal_reach_threshold,
                torch.zeros(1, 1, dtype=torch.long, device=self.device),
                self.scene.env_origins[:, :2],
            )

        pre_step_xy   = robot_xy.clone()
        pre_final_goal = self._final_goal_pos.clone()   # save before reset overwrites it

        obs, rew, terminated, truncated, extras = super().step(action)
        # NOTE: super().step() resets done envs — _final_goal_pos is now the NEW goal for
        # those envs. We must use pre_final_goal for the done-episode distance check.

        robot_xy = self.scene["robot"].data.root_pos_w[:, :2]
        self._logger.record_step(action, rew, robot_xy, self.scene["robot"].data, self.num_envs)

        done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            # Dummy n_real_wps and path_idx for logger
            n_real = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
            path_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._logger.record_done(
                done_ids, terminated, pre_step_xy,
                pre_final_goal, n_real, path_idx,
                self.cfg.goal_reach_threshold, extras,
            )

        if self._logger.should_log():
            self._logger.print_and_reset()

        # Update goal marker position (GUI only)
        if self._goal_marker is not None:
            marker_pos = torch.zeros(self.num_envs, 3, device=self.device)
            marker_pos[:, :2] = self._final_goal_pos
            marker_pos[:,  2] = 0.01
            self._goal_marker.visualize(marker_pos)

        return obs, rew, terminated, truncated, extras
