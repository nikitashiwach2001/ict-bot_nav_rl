from __future__ import annotations

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from .ict_bot_nav_rl_env_cfg import IctBotNavRlEnvCfg
from .episode_logger import EpisodeLogger
from .path_visualizer import PathVisualizer


class IctBotNavRlEnv(ManagerBasedRLEnv):
    cfg: IctBotNavRlEnvCfg

    def __init__(self, cfg: IctBotNavRlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        paths_np      = np.load(self.cfg.paths_file)
        self._paths   = torch.tensor(paths_np, dtype=torch.float32, device=self.device)
        self._n_paths = self._paths.shape[0]
        n_wps         = self._paths.shape[1]
        n             = self.num_envs

        # All positions are in world frame (env_spacing=0, origins all zero)
        self._path_idx       = torch.zeros(n, dtype=torch.long, device=self.device)
        self._waypoint_idx   = torch.zeros(n, dtype=torch.long, device=self.device)
        self._goal_pos       = torch.zeros(n, 2, device=self.device)
        self._final_goal_pos = torch.zeros(n, 2, device=self.device)
        self._prev_goal_dist = torch.zeros(n,    device=self.device)
        self._wp_indices     = torch.arange(n_wps, device=self.device).unsqueeze(0)

        diffs = torch.norm(self._paths[:, 1:] - self._paths[:, :-1], dim=-1)
        self._n_real_wps = (diffs > 1e-4).sum(dim=-1) + 1

        self._logger = EpisodeLogger(n, self.device, cfg.log_interval_steps)
        self._visualizer = PathVisualizer(self.device) if self.sim.has_gui() else None

    def step(self, action):
        robot_xy = self.scene["robot"].data.root_pos_w[:, :2]
        new_ep   = self._logger.episode_steps == 0

        self._logger.record_step_start(
            new_ep, robot_xy, self._path_idx, self._paths,
            self.cfg.waypoint_reach_threshold, self._wp_indices,
        )

        obs, rew, terminated, truncated, extras = super().step(action)

        robot_xy = self.scene["robot"].data.root_pos_w[:, :2]
        self._logger.record_step(action, rew, robot_xy, self.scene["robot"].data, self.num_envs)

        self._advance_waypoints(robot_xy)
        self._update_max_waypoint(robot_xy)

        done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        self._logger.record_done(
            done_ids, terminated, self.scene["robot"].data,
            self._final_goal_pos, self._n_real_wps, self._path_idx,
            self.cfg.goal_reach_threshold, extras,
        )

        if self._logger.should_log():
            self._logger.print_and_reset()

        if self._visualizer is not None:
            self._visualizer.update(
                self._paths, self._path_idx, self._n_real_wps,
                self._goal_pos, self._final_goal_pos,
            )

        return obs, rew, terminated, truncated, extras

    def _advance_waypoints(self, robot_xy: torch.Tensor) -> None:
        dist_to_wp  = torch.norm(self._goal_pos - robot_xy, dim=-1)
        n_wps       = self._paths.shape[1]
        next_idx    = self._waypoint_idx + 1
        can_advance = (dist_to_wp < self.cfg.waypoint_reach_threshold) & (next_idx < n_wps)
        if not can_advance.any():
            return
        self._logger.record_waypoint_advance(can_advance)
        self._waypoint_idx   = torch.where(can_advance, next_idx, self._waypoint_idx)
        new_goals            = self._paths[self._path_idx, self._waypoint_idx]
        self._goal_pos       = torch.where(can_advance.unsqueeze(-1), new_goals, self._goal_pos)
        self._prev_goal_dist = torch.where(
            can_advance,
            torch.norm(new_goals - robot_xy, dim=-1),
            self._prev_goal_dist,
        )

    def _update_max_waypoint(self, robot_xy: torch.Tensor) -> None:
        wp_dists    = torch.norm(self._paths[self._path_idx] - robot_xy.unsqueeze(1), dim=-1)
        max_reached = ((wp_dists < self.cfg.waypoint_reach_threshold).long()
                       * self._wp_indices).max(dim=-1).values
        self._logger.ep_max_waypoint = torch.maximum(
            self._logger.ep_max_waypoint, max_reached
        )
