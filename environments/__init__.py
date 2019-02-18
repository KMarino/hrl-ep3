from gym.envs.registration import register

register(
    id='ExplorerAnt-v2',
    entry_point='environments.explorer_ant:LowlevelAntEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
)

register(
    id='ExplorerAntLowGear-v2',
    entry_point='environments.explorer_ant:LowlevelAntLowGearEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
)

register(
    id='ExplorerProprioceptiveHumanoid-v2',
    entry_point='environments.explorer_humanoid:LowlevelProprioceptiveHumanoidEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
)

register(
    id='HierarchicalAnt-v2',
    entry_point='environments.explorer_ant:HierarchyAntEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
)

register(
    id='MyMazeDebugEnv-v2',
    entry_point='environments.maze_ant:ShowMazeEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
)

register(
    id='AntNavigateEnv-v2',
    entry_point='environments.maze_ant:AntNavigateEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
)

register(
    id='AntNavigateEnv10k-v2',
    entry_point='environments.maze_ant:AntNavigateEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
)

register(
    id='AntNavigateLowGearEnv-v2',
    entry_point='environments.maze_ant:AntNavigateLowGearEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
)

register(
    id='AntCrossMazeEnv-v2',
    entry_point='environments.maze_ant:AntCrossMazeEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
)

register(
    id='AntTMazeEnv-v2',
    entry_point='environments.maze_ant:AntTMazeEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 2000},
)

register(
    id='AntTMazeEnv10k-v2',
    entry_point='environments.maze_ant:AntTMazeEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
)

register(
    id='AntSkullMazeEnv-v2',
    entry_point='environments.maze_ant:AntSkullMazeEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
)

register(
    id='AntDebugMazeEnv-v2',
    entry_point='environments.maze_ant:DebugAntMazeEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
)

register(
    id='AntDebugMazeLeftEnv-v2',
    entry_point='environments.maze_ant:DebugAntMazeLeftEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
)

register(
    id='AntDebugMazeRightEnv-v2',
    entry_point='environments.maze_ant:DebugAntMazeRightEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
)

register(
    id='ProprioceptiveHumanoidSmallCrossMazeEnv10k-v2',
    entry_point='environments.maze_humanoid:ProprioceptiveHumanoidSmallCrossMazeEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
)

