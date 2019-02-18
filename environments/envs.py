import os
import collections
import pdb
import gym
import time
import csv
import json
import shutil
import numpy as np
import random
from . import ant_env
from . import explorer_ant
from . import maze_ant
from . import maze_humanoid
from gym.spaces.box import Box
from .ExplorationReward import ExplorationReward
from .LowLevelExplorationEnv import LowLevelExplorationEnv
from .SmartMonitor import SmartMonitor
from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass

def make_env(env_id, seed, rank, log_dir, opt, verbose, fixed_states=None):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)

        # If maze env, init with maze_opt
        if 'maze' in opt['env']:
            env.unwrapped.option_init(opt['env']['maze'])

        # Wrap with our known reset (if applicable)
        if opt['env']['known_reset']:
            assert isinstance(env.unwrapped, gym.envs.mujoco.MujocoEnv), "Known Reset probably won't work for non Mujoco envs"
            env = KnownReset(env)
            print("=================================================================")
            print(" WARNING - YOU ARE HARD RESETING MUJOCO TO A KNOWN INITIAL STATE")
            print("             ARE YOU SURE THIS IS WHAT YOU WANT?")
            print("=================================================================")

        # If we have a interp lowlevel, make sure to set fixed state
        if opt['model']['mode'] == 'phase_lowlevel' and opt['env']['theta_space_mode'] == 'pretrain_interp':
            opt['env']['fixed_states'] = fixed_states
            assert(fixed_states is not None)

        # Wrap with our exploration reward wrapper
        if opt['model']['mode'] == 'baseline':
            pass
        elif opt['model']['mode'] == 'baseline_reverse':
            # Wrapper reverses run direction
            env = ReverseDirection(env)
        elif opt['model']['mode'] == 'phasesimple' or opt['model']['mode'] == 'phasewstate':
            env = ExplorationReward(env, opt)
        elif opt['model']['mode'] in ['baseline_lowlevel', 'phase_lowlevel']:
            env = LowLevelExplorationEnv(env, opt)
        # Hierarchical wrapper
        elif opt['model']['mode'] in ['hierarchical','hierarchical_many']:
            env = HierarchicalWrapper(env)
            assert(not opt['env']['add_timestep'])
            assert(isinstance(env.unwrapped, maze_ant.HierarchyAntEnv) or isinstance(env.unwrapped, maze_ant.HierarchyAntLowGearEnv) or isinstance(env.unwrapped, maze_humanoid.HierarchyProprioceptiveHumanoidEnv))
        elif opt['model']['mode'] in ['maze_baseline', 'maze_baseline_wphase']:
            pass
        else:
            raise NotImplementedError

        # Optionally, add timestep to observations
        obs_shape = env.observation_space.shape
        if opt['env']['add_timestep']:
            assert len(obs_shape) == 1 and str(env).find('TimeLimit') > -1, "Cannot add timestep to this environment"
            env = AddTimestep(env)

        # Add the monitor
        if log_dir is not None:
            env = SmartMonitor(env, log_dir, rank, opt, verbose)
            
        # Add deepmind wrapper
        if is_atari:
            env = wrap_deepmind(env)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = WrapPyTorch(env)

        return env

    return _thunk

# Wrapper that makes reset go to a known position rather than slightly random
# This should only really be used to debug
class KnownReset(gym.Wrapper):
    def __init__(self, env=None):
        super(KnownReset, self).__init__(env)
        
    # This might only work for Mujoco so be careful
    # Also, I only can guarantee it works if it is put directly over your actual environment
    def reset(self):
        # Do reset to propogate previous hooks and such
        self.env.reset()        

        # Mostly copied from mujoco env reset
        self.env.unwrapped.sim.reset()
        obs = self.reset_model()
        if self.env.unwrapped.viewer is not None:
            self.env.unwrapped.viewer_setup() 
        return obs

    # Our version of reset_model (no random)
    def reset_model(self):
        qpos = self.env.unwrapped.init_qpos
        qvel = self.env.unwrapped.init_qvel
        self.env.unwrapped.set_state(qpos, qvel)
        return self.env.unwrapped._get_obs()

    # Implement step (doesn't change anything, just get rid of the warning)
    def step(self, action):
        return self.env.step(action)

    # Pass through _elapsed_steps
    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps

# Wrapper that adds timestep to observation
class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))

    # Pass through _elapsed_steps
    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps

# Wrapper that wraps up the high level / low level environments
# Makes obs be one box, but gives us the correct sizes for high level and low level
# Also gives helper functions to get low and high level obs from the raw
class HierarchicalWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(HierarchicalWrapper, self).__init__(env)

        # Get the correct observation sizes
        dummy_obs = env.unwrapped._get_obs()
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [dummy_obs.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))

    def _get_obs_mask(self):
        obs_mask = self.env.unwrapped._get_obs_mask()
        return np.concatenate([obs_mask, [1]])

    # Pass through _elapsed_steps
    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps

# Wrapper that fixes default ordering in pytorch
class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)
