from __future__ import annotations

"""
Stage 2 environment — plain ground with 3 kinematic moving obstacles.

Extends Stage 1 with:
  - Kinematic sphere obstacles that move at constant speed and bounce off walls
  - Real nearest-obstacle observation (replaces zero placeholder)
  - R_col penalty for getting too close to any obstacle

Fine-tuning entry point: load Stage 1 checkpoint, same 80-dim obs, same [256,128] net.
"""

import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .ict_bot_nav_stage2_cfg import (
    IctBotNavStage2Cfg, MAX_LOCAL_WPS, N_OBSTACLES, OBS_HEIGHT, OBS_CYL_H, OBS_RADIUS, _OBSTACLE_KEYS,
)
from .episode_logger import EpisodeLogger


class IctBotNavStage2Env(ManagerBasedRLEnv):
    cfg: IctBotNavStage2Cfg

    def __init__(self, cfg: IctBotNavStage2Cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n      = self.num_envs
        device = self.device

        # ── Navigation state ──────────────────────────────────────────────────
        self._goal_pos         = torch.zeros(n, 2, device=device)
        self._final_goal_pos   = torch.zeros(n, 2, device=device)
        self._prev_goal_dist   = torch.ones(n,    device=device)
        self._prev_heading_err = torch.zeros(n,   device=device)

        # ── Local waypoints ───────────────────────────────────────────────────
        self._local_wps    = torch.zeros(n, MAX_LOCAL_WPS, 2, device=device)
        self._local_wp_idx = torch.zeros(n, dtype=torch.long, device=device)
        self._local_wp_pos = torch.zeros(n, 2, device=device)

        # ── Obstacle state (tracked in Python; written to sim each step) ──────
        # _obs_pos[key]: (N, 2) world-frame XY of each obstacle
        # _obs_vel[key]: (N, 2) velocity m/s of each obstacle
        self._obs_pos: dict[str, torch.Tensor] = {
            k: torch.zeros(n, 2, device=device) for k in _OBSTACLE_KEYS
        }
        self._obs_vel: dict[str, torch.Tensor] = {
            k: torch.zeros(n, 2, device=device) for k in _OBSTACLE_KEYS
        }

        # Stacked views updated each step — used by reward and obs functions
        self._obstacle_pos_all = torch.zeros(n, N_OBSTACLES, 2, device=device)
        self._obstacle_vel_all = torch.zeros(n, N_OBSTACLES, 2, device=device)  # (N, K, 2) velocity vectors for CP

        self._step_dt = cfg.sim.dt * cfg.decimation   # seconds per policy step

        self._logger = EpisodeLogger(
            n, device, cfg.log_interval_steps, show_wp_stats=False
        )

        # Obstacle markers — red cylinders, one per env per obstacle (N * N_OBSTACLES total)
        if self.sim.has_gui():
            self._obs_marker = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/Stage2Obstacles",
                markers={
                    "cylinder": sim_utils.CylinderCfg(
                        radius=0.25,
                        height=OBS_CYL_H,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.0, 0.0), opacity=1.0,
                        ),
                    ),
                },
            ))
        else:
            self._obs_marker = None

        # Goal marker
        if self.sim.has_gui():
            self._goal_marker = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/Stage2Goal",
                markers={
                    "disc": sim_utils.CylinderCfg(
                        radius=0.25, height=0.02,
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
                torch.zeros(1, 1, 2, device=self.device),
                self.cfg.goal_reach_threshold,
                torch.zeros(1, 1, dtype=torch.long, device=self.device),
                self.scene.env_origins[:, :2],
            )

        # Move obstacles BEFORE physics so collision is detected this step
        self._step_obstacles()

        pre_step_xy    = robot_xy.clone()
        pre_final_goal = self._final_goal_pos.clone()

        obs, rew, terminated, truncated, extras = super().step(action)

        robot_xy = self.scene["robot"].data.root_pos_w[:, :2]
        self._logger.record_step(action, rew, robot_xy, self.scene["robot"].data, self.num_envs)

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

    # ── Obstacle movement ──────────────────────────────────────────────────────

    def _step_obstacles(self) -> None:
        """
        Advance kinematic obstacle positions, bounce off arena walls, write to sim.
        Also refreshes _obstacle_pos_all and _obstacle_speed_all for obs/reward fns.
        """
        bound    = self.cfg.spawn_half_extent - 0.1   # bounce before hitting edge
        orig     = self.scene.env_origins[:, :2]       # (N, 2)
        dt       = self._step_dt
        all_envs = torch.arange(self.num_envs, device=self.device)

        for i, key in enumerate(_OBSTACLE_KEYS):
            pos = self._obs_pos[key]   # (N, 2) world frame
            vel = self._obs_vel[key]   # (N, 2)

            # Local position relative to env origin
            pos_local = pos - orig     # (N, 2)

            # Bounce: if outside bound, flip the relevant velocity component
            out_x = pos_local[:, 0].abs() >= bound
            out_y = pos_local[:, 1].abs() >= bound
            vel[:, 0] = torch.where(out_x, -vel[:, 0], vel[:, 0])
            vel[:, 1] = torch.where(out_y, -vel[:, 1], vel[:, 1])

            # Advance position
            new_pos = pos + vel * dt

            # Hard clamp so they don't drift outside (edge case)
            new_pos[:, 0] = torch.clamp(new_pos[:, 0], orig[:, 0] - bound, orig[:, 0] + bound)
            new_pos[:, 1] = torch.clamp(new_pos[:, 1], orig[:, 1] - bound, orig[:, 1] + bound)

            self._obs_pos[key] = new_pos
            self._obs_vel[key] = vel

            # Write kinematic pose to sim (base on ground, no rotation)
            pose = torch.zeros(self.num_envs, 7, device=self.device)
            pose[:, 0] = new_pos[:, 0]
            pose[:, 1] = new_pos[:, 1]
            pose[:, 2] = OBS_RADIUS          # sphere centre = radius above ground → bottom at z=0
            pose[:, 3] = 1.0                # quat w
            self.scene[key].write_root_pose_to_sim(pose, all_envs)

            # Update stacked tensors used by obs/reward functions
            self._obstacle_pos_all[:, i, :] = new_pos
            self._obstacle_vel_all[:, i, :] = vel          # full velocity vector for CP computation

        # Update red cylinder markers so obstacles are visible in viewport
        if self._obs_marker is not None:
            all_pos = torch.stack(
                [self._obs_pos[k] for k in _OBSTACLE_KEYS], dim=1
            )   # (N, K, 2)
            marker_pos = torch.zeros(
                self.num_envs * N_OBSTACLES, 3, device=self.device
            )
            marker_pos[:, :2] = all_pos.reshape(-1, 2)
            marker_pos[:, 2]  = OBS_CYL_H / 2.0   # centre z
            self._obs_marker.visualize(marker_pos)

    # ── Local waypoint management ──────────────────────────────────────────────

    def _advance_local_waypoints(self, robot_xy: torch.Tensor) -> None:
        """Paper waypoint: always 0.6m ahead on line from robot to final goal.

        Unlike fixed pre-computed waypoints, this moves with the robot — so
        detours around obstacles don't cause waypoint misses.
        """
        direction  = self._final_goal_pos - robot_xy                      # (N, 2)
        dist_goal  = torch.norm(direction, dim=-1, keepdim=True)          # (N, 1)
        unit_dir   = direction / (dist_goal + 1e-6)                        # (N, 2)

        wp_dist    = torch.clamp(dist_goal.squeeze(-1),
                                 max=self.cfg.wp_reach_threshold * 2.0)    # at most 2× threshold ahead

        # Waypoint = robot_xy + 0.6m along line to goal (clamped at goal itself)
        candidate  = robot_xy + unit_dir * self.cfg.wp_reach_threshold * 2.0
        # If closer than 0.6m to goal, use goal directly
        near_goal  = dist_goal.squeeze(-1) < self.cfg.wp_reach_threshold * 2.0
        self._local_wp_pos = torch.where(
            near_goal.unsqueeze(-1), self._final_goal_pos, candidate
        )
