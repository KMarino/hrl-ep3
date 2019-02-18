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
from . import geom_utils
from gym.spaces.box import Box
import math
import sys
sys.path.append('../')
from utils import RollingAverage

# Wrapper that defines our environment for our low level ant policy
# Everything is egocentric to the ant
class RewardCyclicEnv(gym.Wrapper):
    def __init__(self, env=None, opt=None):
        super(RewardCyclicEnv, self).__init__(env)

        # Should be in correct mode (can be baseline or phase, but should be theta version)
        self.mode = opt['model']['mode']
        assert(self.mode in ['cyclic'])

        # Make sure we're using the right environment (our ant for now)
        assert isinstance(env.unwrapped, ant_env.BaseAntEnv)

        # Keep memory
        # Figure out sizes of external and external
        self.ex_states = []
        self.pro_states = []
        self.actions = []

        # Phase period
        self.phase_k = opt['model']['phase_period']

        # Params of reward
        self.min_movement = opt['env']['min_movement']
        self.survive_reward = opt['env']['survive_reward']

    # Step function
    # Does step and updates our stored values and also calculates our exploration reward
    def step(self, action):
        # Do the original step and get the environment reward (will throw some of this out)
        obs, true_reward, done, info = self.env.step(action)

        # Get the new state and step
        new_state_pro, new_state_ex = self.unwrapped.get_intern_extern_state()

        # Update the states and actions in memory
        self.ex_states.append(np.array(new_state_ex))
        self.pro_states.append(np.array(new_state_pro))
        self.actions.append(np.array(action))
        new_count = self._elapsed_steps
        assert(len(self.ex_states) == new_count + 1)

        # Determine if there was enough movement
        min_movement_mult = float(np.linalg.norm(new_state_pro, 2) > self.min_movement)

        # Get cyclic penalty
        if len(self.pro_states) > self.phase_k + 1:
            # Get last/current cycle state and actions
            new_s = self.pro_states[-1]
            old_s = self.pro_states[-(self.phase_k+1)]
            new_a = self.actions[-1]
            old_a = self.actions[-(self.phase_k+1)]
            
            # Get cyclic reward for state
            state_diff = np.linalg.norm(new_s - old_s, 2)
            state_cycle_reward = -state_diff
        else:
            state_cycle_reward = 0

        # Update survive
        info['reward_survive'] = self.survive_reward
        info['reward_thresh'] = min_movement_mult
        info['reward_cycle'] = state_cycle_reward
        reward = info['reward_thresh'] * (info['reward_survive'] + info['reward_cycle']) 
        info['reward_env'] = reward
        #info['reward_env'] = info['reward_forward'] + info['reward_ctrl'] + info['reward_contact'] + info['reward_survive']        
       
        # Return
        return obs, reward, done, info

    # Reset
    # Pass through and reset data
    def reset(self):
        obs = self.env.reset()

        # Reset our storage structures
        self.ex_states = []
        self.pro_states = []
        self.actions = []
 
        # Update the states and actions in memory
        new_state_pro, new_state_ex = self.unwrapped.get_intern_extern_state()
        self.ex_states.append(np.array(new_state_ex))
        self.pro_states.append(np.array(new_state_pro))
        self.actions.append(np.zeros(self.action_space.shape))
        assert(self._elapsed_steps == 0)

        return obs

    # Pass through _elapsed_steps
    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps

