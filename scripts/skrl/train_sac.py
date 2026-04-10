"""Train ICTBot navigation with SAC (Soft Actor-Critic) via SKRL."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train ICTBot navigation with SAC.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task",     type=str, default="ICTBot-Nav-v0")
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint to resume from.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports after sim is up ────────────────────────────────────────────────────
import os
import time
from datetime import datetime

import gymnasium as gym
import torch

from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import ict_bot_nav_rl.tasks  # noqa: F401


# ── Model definitions ──────────────────────────────────────────────────────────
OBS_DIM = 78    # 72 lidar + 4 goal + 1 fwd_vel + 1 ang_vel
ACT_DIM = 2     # linear_vel, angular_vel


class Policy(GaussianMixin, Model):
    def __init__(self, obs_space, act_space, device, clip_actions=True):
        Model.__init__(self, obs_space, act_space, device)
        GaussianMixin.__init__(self, clip_actions=clip_actions,
                               clip_log_std=True, min_log_std=-20, max_log_std=2)
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256),     nn.ReLU(),
            nn.Linear(256, 128),     nn.ReLU(),
        )
        self.mean_layer = nn.Linear(128, ACT_DIM)
        self.log_std    = nn.Parameter(torch.zeros(ACT_DIM))

    def compute(self, inputs, role):
        x = self.net(inputs["states"])
        return self.mean_layer(x), self.log_std, {}


class Critic(DeterministicMixin, Model):
    def __init__(self, obs_space, act_space, device):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM + ACT_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256),                nn.ReLU(),
            nn.Linear(256, 128),                nn.ReLU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        x = torch.cat([inputs["states"], inputs["taken_actions"]], dim=-1)
        return self.net(x), {}


@hydra_task_config(args_cli.task, "skrl_sac_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    set_seed(args_cli.seed)

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device:
        env_cfg.sim.device = args_cli.device

    # ── logging ────────────────────────────────────────────────────────────────
    log_dir = os.path.join(
        "logs", "skrl", "ict_bot_nav",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_sac_torch_sac"
    )
    os.makedirs(log_dir, exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"),   env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    env_cfg.log_dir = log_dir

    # ── environment ────────────────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    device = env.device

    # ── models ─────────────────────────────────────────────────────────────────
    policy         = Policy(env.observation_space, env.action_space, device)
    critic_1       = Critic(env.observation_space, env.action_space, device)
    critic_2       = Critic(env.observation_space, env.action_space, device)
    target_critic_1 = Critic(env.observation_space, env.action_space, device)
    target_critic_2 = Critic(env.observation_space, env.action_space, device)

    models = {
        "policy":          policy,
        "critic_1":        critic_1,
        "critic_2":        critic_2,
        "target_critic_1": target_critic_1,
        "target_critic_2": target_critic_2,
    }

    # ── replay buffer ──────────────────────────────────────────────────────────
    memory = RandomMemory(memory_size=100_000, num_envs=env.num_envs, device=device)

    # ── SAC config ─────────────────────────────────────────────────────────────
    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg["gradient_steps"]          = 1
    cfg["batch_size"]              = 512
    cfg["discount_factor"]         = 0.99
    cfg["polyak"]                  = 0.005
    cfg["actor_learning_rate"]     = 3e-4
    cfg["critic_learning_rate"]    = 3e-4
    cfg["random_timesteps"]        = 1000
    cfg["learning_starts"]         = 1000
    cfg["grad_norm_clip"]          = 0.5
    cfg["learn_entropy"]           = True
    cfg["entropy_learning_rate"]   = 3e-4
    cfg["initial_entropy_value"]   = 1.0
    cfg["target_entropy"]          = -ACT_DIM   # standard SAC heuristic: -action_dim
    cfg["rewards_shaper"]          = lambda r, *_: r * 0.1
    cfg["state_preprocessor"]      = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"]      = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    cfg["experiment"]["directory"]        = os.path.abspath(log_dir)
    cfg["experiment"]["experiment_name"]  = ""
    cfg["experiment"]["write_interval"]   = 500
    cfg["experiment"]["checkpoint_interval"] = 5000

    # ── agent + trainer ────────────────────────────────────────────────────────
    agent = SAC(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    if args_cli.checkpoint:
        print(f"[INFO] Resuming from checkpoint: {args_cli.checkpoint}")
        agent.load(args_cli.checkpoint)

    trainer = SequentialTrainer(
        cfg={"timesteps": agent_cfg["trainer"]["timesteps"],
             "close_environment_at_exit": False},
        env=env,
        agents=agent,
    )

    start = time.time()
    trainer.train()
    print(f"Training time: {round(time.time() - start, 2)}s")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
