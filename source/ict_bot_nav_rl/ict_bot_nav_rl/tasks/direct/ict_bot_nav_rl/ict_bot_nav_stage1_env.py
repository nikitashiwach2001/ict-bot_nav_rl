from __future__ import annotations

"""
Stage 1 environment — plain ground, full obs space, local waypoints.

Key responsibilities beyond the manager:
  - Allocate navigation state tensors (_goal_pos, _final_goal_pos,
    _prev_goal_dist, _prev_heading_err, _local_wps, _local_wp_idx,
    _local_wp_pos) so reward/obs functions can read them.
  - Advance local waypoints after each physics step (reward fires inside
    super().step(), waypoint index advances after — no double-counting).
"""

import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .ict_bot_nav_stage1_cfg import IctBotNavStage1Cfg, MAX_LOCAL_WPS
from .episode_logger import EpisodeLogger


class IctBotNavStage1Env(ManagerBasedRLEnv):
    cfg: IctBotNavStage1Cfg

    def __init__(self, cfg: IctBotNavStage1Cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n = self.num_envs

        # ── Navigation state (same names as other envs for reward fn compatibility)
        self._goal_pos         = torch.zeros(n, 2,              device=self.device)
        self._final_goal_pos   = torch.zeros(n, 2,              device=self.device)
        self._prev_goal_dist   = torch.ones(n,                  device=self.device)
        self._prev_heading_err = torch.zeros(n,                 device=self.device)

        # ── Local waypoints (0.6 m spaced along start→goal straight line)
        self._local_wps    = torch.zeros(n, MAX_LOCAL_WPS, 2,  device=self.device)
        self._local_wp_idx = torch.zeros(n, dtype=torch.long,  device=self.device)
        self._local_wp_pos = torch.zeros(n, 2,                 device=self.device)

        self._logger = EpisodeLogger(
            n, self.device, cfg.log_interval_steps, show_wp_stats=False
        )

        # Goal marker (GUI only)
        if self.sim.has_gui():
            self._goal_marker = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/Stage1Goal",
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

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        robot_xy = self.scene["robot"].data.root_pos_w[:, :2]
        new_ep   = self._logger.episode_steps == 0

        if new_ep.any():
            self._logger.record_step_start(
                new_ep, robot_xy,
                torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
                torch.zeros(1, 1, 2, device=self.device),    # dummy paths
                self.cfg.goal_reach_threshold,
                torch.zeros(1, 1, dtype=torch.long, device=self.device),
                self.scene.env_origins[:, :2],
            )

        pre_step_xy    = robot_xy.clone()
        pre_final_goal = self._final_goal_pos.clone()   # snapshot before reset overwrites

        # super().step() runs physics, computes rewards/obs/terms, resets done envs
        obs, rew, terminated, truncated, extras = super().step(action)

        robot_xy = self.scene["robot"].data.root_pos_w[:, :2]
        self._logger.record_step(action, rew, robot_xy, self.scene["robot"].data, self.num_envs)

        # Advance local waypoint index AFTER reward is computed (avoids double-count)
        self._advance_local_waypoints(robot_xy)

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

    # ── Local waypoint management ──────────────────────────────────────────────

    def _advance_local_waypoints(self, robot_xy: torch.Tensor) -> None:
        """Move each env's active waypoint to the next slot when the robot is close enough."""
        dist    = torch.norm(self._local_wp_pos - robot_xy, dim=-1)
        reached = dist < self.cfg.wp_reach_threshold
        if not reached.any():
            return
        next_idx = torch.clamp(self._local_wp_idx + 1, max=MAX_LOCAL_WPS - 1)
        self._local_wp_idx = torch.where(reached, next_idx, self._local_wp_idx)
        new_wp = self._local_wps[
            torch.arange(self.num_envs, device=self.device), self._local_wp_idx
        ]
        self._local_wp_pos = torch.where(reached.unsqueeze(-1), new_wp, self._local_wp_pos)
