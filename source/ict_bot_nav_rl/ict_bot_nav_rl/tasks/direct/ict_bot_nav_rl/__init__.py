import gymnasium as gym

from . import agents

gym.register(
    id="ICTBot-Nav-v0",
    entry_point=f"{__name__}.ict_bot_nav_rl_env:IctBotNavRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ict_bot_nav_rl_env_cfg:IctBotNavRlEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)