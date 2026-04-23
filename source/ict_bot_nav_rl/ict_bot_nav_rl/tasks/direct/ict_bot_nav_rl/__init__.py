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

gym.register(
    id="ICTBot-Nav-Plain-v0",
    entry_point=f"{__name__}.ict_bot_nav_plain_env:IctBotNavPlainEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ict_bot_nav_plain_cfg:IctBotNavPlainCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_plain_cfg.yaml",
    },
)

gym.register(
    id="ICTBot-Nav-Stage1-v0",
    entry_point=f"{__name__}.ict_bot_nav_stage1_env:IctBotNavStage1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ict_bot_nav_stage1_cfg:IctBotNavStage1Cfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_stage1_cfg.yaml",
    },
)

gym.register(
    id="ICTBot-Nav-Stage2-v0",
    entry_point=f"{__name__}.ict_bot_nav_stage2_env:IctBotNavStage2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ict_bot_nav_stage2_cfg:IctBotNavStage2Cfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_stage2_cfg.yaml",
    },
)

gym.register(
    id="ICTBot-Nav-Stage2-TD3-v0",
    entry_point=f"{__name__}.ict_bot_nav_stage2_env:IctBotNavStage2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ict_bot_nav_stage2_cfg:IctBotNavStage2Cfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_td3_stage2_cfg.yaml",
    },
)

gym.register(
    id="ICTBot-Nav-Phase2-v0",
    entry_point=f"{__name__}.ict_bot_nav_phase2_env:IctBotNavPhase2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ict_bot_nav_phase2_cfg:IctBotNavPhase2Cfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_phase2_cfg.yaml",
    },
)
