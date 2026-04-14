from __future__ import annotations

import os
import time
import numpy as np
import torch

_LOG_PATH = os.path.join(os.path.dirname(__file__), "episode_log.txt")
_W = 62  # console width


class EpisodeLogger:
    def __init__(self, num_envs: int, device: str, log_interval_steps: int):
        n = num_envs
        self.device = device
        self.log_interval_steps = log_interval_steps

        # per-episode buffers (also read by env for navigation logic)
        self.episode_steps    = torch.zeros(n, dtype=torch.long,  device=device)
        self.episode_return   = torch.zeros(n,                    device=device)
        self.ep_start_pos     = torch.zeros(n, 2,                 device=device)
        self.ep_prev_xy       = torch.zeros(n, 2,                 device=device)
        self.ep_spawn_wp      = torch.zeros(n, dtype=torch.long,  device=device)
        self.ep_max_waypoint  = torch.zeros(n, dtype=torch.long,  device=device)
        self.ep_dist_traveled = torch.zeros(n,                    device=device)
        self.ep_action_sum    = torch.zeros(n,                    device=device)
        self.ep_fwd_vel_sum   = torch.zeros(n,                    device=device)
        self.ep_ang_vel_sum   = torch.zeros(n,                    device=device)
        self.ep_wps_reached   = torch.zeros(n, dtype=torch.long,  device=device)

        # iteration-window accumulators
        self.global_steps  = 0
        self._iter_steps   = 0
        self._iter_num     = 0
        self._iter_start_t = time.time()
        self._iter_rew_sum = 0.0
        self._iter_rew_n   = 0

        self._iter_ep_returns:  list[float] = []
        self._iter_ep_lengths:  list[int]   = []
        self._iter_ep_wp_pct:   list[float] = []
        self._iter_ep_wps_hit:  list[int]   = []

        self._n_timeout   = 0
        self._n_collision = 0
        self._n_success   = 0

        self._log_sum: dict[str, float] = {}
        self._log_n:   dict[str, int]   = {}

    def record_step_start(
        self,
        new_ep: torch.Tensor,
        robot_xy: torch.Tensor,
        path_idx: torch.Tensor,
        paths: torch.Tensor,
        wp_threshold: float,
        wp_indices: torch.Tensor,
        env_origins_xy: torch.Tensor,
    ) -> None:
        if not new_ep.any():
            return
        spawn_xy = robot_xy[new_ep]                              # world frame (for travel logging)
        self.ep_start_pos[new_ep] = spawn_xy.clone()
        self.ep_prev_xy[new_ep]   = spawn_xy.clone()

        spawn_xy_local = spawn_xy - env_origins_xy[new_ep]      # local frame (for WP comparison)
        spawn_wps = paths[path_idx[new_ep]]
        spawn_d   = torch.norm(spawn_wps - spawn_xy_local.unsqueeze(1), dim=-1)
        mask      = (spawn_d < wp_threshold).long()
        self.ep_spawn_wp[new_ep]     = (mask * wp_indices).max(dim=-1).values
        self.ep_max_waypoint[new_ep] = self.ep_spawn_wp[new_ep].clone()

    def record_step(
        self,
        action: torch.Tensor,
        rew: torch.Tensor,
        robot_xy: torch.Tensor,
        robot_data,
        num_envs: int,
    ) -> None:
        step_dist = torch.norm(robot_xy - self.ep_prev_xy, dim=-1)
        self.ep_dist_traveled += step_dist
        self.ep_prev_xy        = robot_xy.clone()

        self.ep_action_sum   += action.abs().mean(dim=-1)
        self.ep_fwd_vel_sum  += robot_data.root_lin_vel_b[:, 0]   # +X is forward
        self.ep_ang_vel_sum  += robot_data.root_ang_vel_w[:, 2]
        self.episode_return  += rew
        self.episode_steps   += 1

        self.global_steps    += num_envs
        self._iter_steps     += num_envs
        self._iter_rew_sum   += rew.sum().item()
        self._iter_rew_n     += num_envs

    def record_done(
        self,
        done_ids: torch.Tensor,
        terminated: torch.Tensor,
        robot_data,
        final_goal_pos: torch.Tensor,
        n_real_wps: torch.Tensor,
        path_idx: torch.Tensor,
        goal_reach_threshold: float,
        extras: dict,
    ) -> None:
        if done_ids.numel() == 0:
            return
        self._collect_stats(done_ids, terminated, robot_data, final_goal_pos,
                            n_real_wps, path_idx, goal_reach_threshold)
        self._accumulate_extras(extras, done_ids.numel())

    def record_waypoint_advance(self, can_advance: torch.Tensor) -> None:
        """Increment per-episode waypoint counter for envs that just advanced."""
        self.ep_wps_reached += can_advance.long()

    def should_log(self) -> bool:
        return self._iter_steps >= self.log_interval_steps

    def print_and_reset(self) -> None:
        self._iter_num += 1
        self._print_summary()
        self._reset_iter()

    # ── private ────────────────────────────────────────────────────────────────

    def _collect_stats(self, done_ids, terminated, robot_data, final_goal_pos,
                       n_real_wps, path_idx, goal_reach_threshold):
        root_pos   = robot_data.root_pos_w[:, :2]
        lidar      = None  # resolved lazily if needed
        file_lines = []

        for i in done_ids:
            n_steps   = max(int(self.episode_steps[i]), 1)
            goal_dist = torch.norm(final_goal_pos[i] - root_pos[i]).item()
            n_real    = int(n_real_wps[int(path_idx[i])].item())
            spawn_wp  = int(self.ep_spawn_wp[i])
            extra_wp  = max(0, min(int(self.ep_max_waypoint[i]), n_real - 1) - spawn_wp)
            max_poss  = max(0, n_real - 1 - spawn_wp)
            wp_pct    = (extra_wp / max_poss * 100.0) if max_poss > 0 else 0.0

            is_success = terminated[i].item() and goal_dist < goal_reach_threshold
            if not terminated[i].item():
                status = "timeout";   self._n_timeout   += 1
            elif is_success:
                status = "SUCCESS";   self._n_success   += 1
            else:
                status = "COLLISION"; self._n_collision += 1

            wps_hit = int(self.ep_wps_reached[i].item())
            self._iter_ep_returns.append(self.episode_return[i].item())
            self._iter_ep_lengths.append(n_steps)
            self._iter_ep_wp_pct.append(wp_pct)
            self._iter_ep_wps_hit.append(wps_hit)

            line = (
                f"[Env {i.item():02d}] [{status}] "
                f"steps={n_steps:4d} | "
                f"return={self.episode_return[i].item():8.3f} | "
                f"wp={extra_wp}/{max_poss} ({wp_pct:.0f}%) | "
                f"wps_reached={wps_hit} | "
                f"final_dist={goal_dist:.2f}m | "
                f"traveled={self.ep_dist_traveled[i].item():.2f}m | "
                f"avg_fwd={self.ep_fwd_vel_sum[i].item()/n_steps:+.3f}m/s | "
                f"avg_ang={self.ep_ang_vel_sum[i].item()/n_steps:+.3f}rad/s"
            )
            file_lines.append(line)
            print(line, flush=True)

        with open(_LOG_PATH, "a") as f:
            f.write("\n".join(file_lines) + "\n")

        self.episode_return[done_ids]   = 0.0
        self.episode_steps[done_ids]    = 0
        self.ep_action_sum[done_ids]    = 0.0
        self.ep_fwd_vel_sum[done_ids]   = 0.0
        self.ep_ang_vel_sum[done_ids]   = 0.0
        self.ep_max_waypoint[done_ids]  = 0
        self.ep_spawn_wp[done_ids]      = 0
        self.ep_dist_traveled[done_ids] = 0.0
        self.ep_wps_reached[done_ids]   = 0

    def _accumulate_extras(self, extras: dict, n_done: int) -> None:
        for k, v in extras.get("log", {}).items():
            if k == "time_outs":
                continue
            try:
                val = float(v.item() if torch.is_tensor(v) else v)
            except Exception:
                continue
            self._log_sum[k] = self._log_sum.get(k, 0.0) + val * n_done
            self._log_n[k]   = self._log_n.get(k, 0) + n_done

    def _print_summary(self) -> None:
        elapsed  = max(time.time() - self._iter_start_t, 1e-6)
        speed    = self._iter_steps / elapsed
        mean_rew = self._iter_rew_sum / max(self._iter_rew_n, 1)
        n_ep     = len(self._iter_ep_returns)
        mean_ret = float(np.mean(self._iter_ep_returns)) if n_ep else float("nan")
        mean_len = float(np.mean(self._iter_ep_lengths)) if n_ep else float("nan")
        mean_wp      = float(np.mean(self._iter_ep_wp_pct))   if n_ep else float("nan")
        mean_wps_hit = float(np.mean(self._iter_ep_wps_hit))  if n_ep else float("nan")
        total    = self._n_timeout + self._n_collision + self._n_success
        pct      = lambda x: 100.0 * x / max(total, 1)

        sep  = "─" * _W
        tsep = "━" * _W
        lines = [
            "", tsep,
            f"  Iteration {self._iter_num}  |  Total steps: {self.global_steps:,}",
            f"  Speed: {speed:,.0f} steps/s  |  Window: {self._iter_steps:,} steps ({elapsed:.1f}s)",
            sep,
            f"  Mean step reward   : {mean_rew:+.4f}",
        ]

        if n_ep > 0:
            lines += [
                f"  Mean episode return: {mean_ret:+.3f}  ({n_ep} episodes)",
                f"  Mean episode length: {mean_len:.1f} steps",
                f"  Mean wp progress   : {mean_wp:.1f}%",
                f"  Mean wps reached   : {mean_wps_hit:.1f}",
            ]
        else:
            lines.append("  (no episodes completed in this window)")

        rew_keys = {k: v for k, v in self._log_sum.items() if k.startswith("Episode_Reward/")}
        if rew_keys:
            lines += [sep, "  Episode_Reward/"]
            for k in sorted(rew_keys):
                mean = self._log_sum[k] / max(self._log_n[k], 1)
                lines.append(f"    {k.replace('Episode_Reward/', ''):<24s}: {mean:+.5f}")
        else:
            lines += [sep, "  Episode_Reward/  (awaiting completions...)"]

        lines += [
            sep, "  Episode_Termination/",
            f"    {'time_out':<24s}: {pct(self._n_timeout):5.1f}%  ({self._n_timeout})",
            f"    {'collision':<24s}: {pct(self._n_collision):5.1f}%  ({self._n_collision})",
            f"    {'goal_reached':<24s}: {pct(self._n_success):5.1f}%  ({self._n_success})",
            tsep,
        ]
        print("\n".join(lines), flush=True)

    def _reset_iter(self) -> None:
        self._iter_steps   = 0
        self._iter_start_t = time.time()
        self._iter_rew_sum = 0.0
        self._iter_rew_n   = 0
        self._iter_ep_returns.clear()
        self._iter_ep_lengths.clear()
        self._iter_ep_wp_pct.clear()
        self._iter_ep_wps_hit.clear()
        self._n_timeout   = 0
        self._n_collision = 0
        self._n_success   = 0
        self._log_sum.clear()
        self._log_n.clear()
