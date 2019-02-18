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
from gym.spaces.box import Box
from utils import RollingAverage

# Wrapper that changes the reward outputs to a new exploration reward
# This is also almost certainly hard-coded to Mujoco
class ExplorationReward(gym.Wrapper):
    def __init__(self, env=None, opt=None):
        super(ExplorationReward, self).__init__(env)

        # Set reward shape type and exploration direction
        # reward_shape_type
        # periodic - only reward at the beginning of each new cycle
        # step - reward every step, but need at least k steps already
        # ramp - same as step, but before k, give weighted reward for movement since 0
        # instant - movement since last step
        assert opt['env']['reward_shape_type'] in ['periodic', 'step', 'ramp', 'instant'], "Invalid reward shape option"
        self.reward_shape_type = opt['env']['reward_shape_type']

        # r_displ_dir
        # xpos - displacement in x direction (index 0 of qpos), direction doesn't matter
        # xypos - displacement in y direction (index :2 of qpos), direction doesn't matter
        # qpos - all position displacement, not recommended
        # Do manual switch
        if 'r_disp_dir' not in opt['env']:
            if isinstance(env.unwrapped, gym.envs.mujoco.HalfCheetahEnv) or isinstance(env.unwrapped, gym.envs.mujoco.HopperEnv):
                self.r_disp_dir = 'xpos'
            elif isinstance(env.unwrapped, gym.envs.mujoco.AntEnv):
                self.r_disp_dir = 'xypos'
            else:
                raise NotImplementedError
        else:
            self.r_disp_dir = opt['env']['r_disp_dir']
        assert self.r_disp_dir in ['xpos', 'xypos', 'qpos'], "Invalid reward displacement option"

        # Set state variables to conditionally add other rewards
        self.add_ctrl_r = opt['env']['add_ctrl_r']
        self.add_contact_r = opt['env']['add_contact_r']
        self.add_survive_r = opt['env']['add_survive_r']
        
        # Set signs
        self.sign_info = opt['env']['sign_info']
        assert(self.sign_info in ['unsigned', 'signed_positive', 'signed_negative'])

        # Initialize all the machinery for calculating exploration reward 
        # Get the state size for external state
        if self.r_disp_dir == 'xpos':
            self.ex_state_sz = 1
        elif self.r_disp_dir == 'xypos':
            self.ex_state_sz = 2
        elif self.r_disp_dir == 'qpos':
            qpos_shape = self.env.unwrapped.init_qpos.shape
            assert(len(qpos_shape) == 1)
            self.ex_state_sz = qpos_shape[0]

        # Get the proprioceptive state size (joint positions and velocities)
        s = env.unwrapped.state_vector()
        assert(len(s.shape) == 1)
        if isinstance(env.unwrapped, gym.envs.mujoco.HalfCheetahEnv):
            self.pro_state_sz = s.shape[0] - 2
        elif isinstance(env.unwrapped, gym.envs.mujoco.AntEnv):
            self.pro_state_sz = s.shape[0] - 3
        else:
            raise NotImplementedError

        # Whether we use dt and phase_k to normalize the reward
        self.norm_exp_reward = opt['env']['norm_exp_reward']

        # Weights for action and state cyclicality constraints
        self.state_cycle_weight = opt['env']['state_cycle_weight']
        self.action_cycle_weight = opt['env']['action_cycle_weight']
        self.cycle_startup = opt['env']['cycle_startup']
        
        # Storage
        self.phase_k = opt['model']['phase_period']
        self.ex_states = np.zeros([self.phase_k, self.ex_state_sz])
        self.pro_states = np.zeros([self.phase_k, self.pro_state_sz])
        self.actions = np.zeros([self.phase_k] + list(self.action_space.shape))

        # Part mask and part debug. Stores the timesteps (initally -1 when invalid)
        self.counts = np.ones(self.phase_k, dtype=np.int64)*-1

    # Step function
    # Does step and updates our stored values and also calculates our exploration reward
    def step(self, action):
        # Do the original step and get the environment reward
        obs, true_reward, done, info = self.env.step(action)

        # Get the new state and step
        new_state_ex, new_state_pro = self.get_state()
        new_count = self._elapsed_steps

        # Get the exploration reward 
        move_reward = 0
        assert(new_count > 0)
        if self.reward_shape_type == 'instant' and new_count > 0:
            # Get simple displacement reward since last step
            old_state_ex = self.ex_states[(new_count-1) % self.phase_k, :]
            assert(self.counts[(new_count-1) % self.phase_k] != -1)
            move_reward = self.displacement_reward(new_state_ex, old_state_ex)
            if self.norm_exp_reward:
                move_reward *= self.phase_k
        elif self.reward_shape_type == 'periodic' and new_count % self.phase_k == 0 or \
           (self.reward_shape_type == 'step' or self.reward_shape_type == 'ramp') and new_count >= self.phase_k:
            # Get simple displacement reward since last period
            old_state_ex = self.ex_states[new_count % self.phase_k, :]
            assert(self.counts[new_count % self.phase_k] != -1)
            move_reward = self.displacement_reward(new_state_ex, old_state_ex)
        elif self.reward_shape_type == 'ramp':
            # Get simple displacement reward since 0
            old_state_ex = self.ex_states[0, :]
            assert(self.counts[0] != -1)
            move_reward = self.displacement_reward(new_state_ex, old_state_ex)
        info['reward_move'] = move_reward
        exp_reward = move_reward

        # Get the cycle reward
        if self.state_cycle_weight > 0 or self.action_cycle_weight > 0:
            # Init rewards
            cycle_reward = 0
            s_cycle_reward = 0
            a_cycle_reward = 0

            # Get start cycle.
            # If cycle startup, don't punish until two cycles are done
            if self.cycle_startup > 0:
                mincount = (self.cycle_startup+1)*self.phase_k
            # Never start until phase_k + 1 (so we ignore the reset state/action)
            else:
                mincount = self.phase_k + 1
                
            # If count is sufficient, get cycle reward
            if new_count >= mincount:
                old_state_pro = self.pro_states[new_count % self.phase_k, :]
                old_action = self.actions[new_count % self.phase_k, :]
                assert(self.counts[new_count % self.phase_k] != -1)
                cycle_reward, s_cycle_reward, a_cycle_reward = self.cyclic_reward(new_state_pro, old_state_pro, action, old_action)
            info['reward_cycle'] = cycle_reward
            info['reward_cycle_s'] = s_cycle_reward
            info['reward_cycle_a'] = a_cycle_reward
            exp_reward += cycle_reward

        # Update storage
        self.ex_states[new_count % self.phase_k, :] = np.copy(new_state_ex)
        self.pro_states[new_count % self.phase_k, :] = np.copy(new_state_pro)
        self.actions[new_count % self.phase_k, :] = np.copy(action)
        self.counts[new_count % self.phase_k] = np.copy(new_count)

        # Add original env rewards optionally
        if self.add_ctrl_r and 'reward_ctrl' in info:
            exp_reward += info['reward_ctrl']
        if self.add_contact_r and 'reward_contact' in info:
            exp_reward += info['reward_contact']
        if self.add_survive_r and 'reward_survive' in info:
            exp_reward += info['reward_survive']

        # Update the info and add both the total true reward and the explore reward
        info['reward_env'] = true_reward
        info['reward_exp'] = exp_reward

        # Return
        return obs, exp_reward, done, info

    # Gets the extrinsic state we need for our exploration reward and the proprioceptive reward
    def get_state(self):
        # Get state
        state = self.env.unwrapped.state_vector()

        # Seperate external and proprioceptive states
        # This assumption might not be true for all envs
        s_ex = state[:self.ex_state_sz]
        s_pr = state[-self.pro_state_sz:]
        assert(len(s_ex) == self.ex_state_sz)
        assert(len(s_pr) == self.pro_state_sz)

        return s_ex, s_pr

    # Simple displacement reward
    def displacement_reward(self, new_s, old_s):
        # Get exploration reward
        if self.sign_info == 'unsigned':
            exp_reward = np.linalg.norm(new_s - old_s, 2)
        elif self.sign_info == 'signed_positive':
            assert(self.r_disp_dir == 'xpos')
            exp_reward = float(new_s - old_s)
        elif self.sign_info == 'signed_negative':
            assert(self.r_disp_dir == 'xpos')
            exp_reward = float(old_s - new_s)
        else:
            raise NotImplementedError

        # Optionally normalize
        if self.norm_exp_reward:
            exp_reward /= self.unwrapped.dt * self.phase_k

        return exp_reward

    # Cyclicality reward
    def cyclic_reward(self, new_s, old_s, new_a, old_a):
        # Get cyclic reward for state
        state_diff = np.linalg.norm(new_s - old_s, 2)
        state_cycle_reward = -self.state_cycle_weight*state_diff

        # Get cyclic reward for state
        action_diff = np.linalg.norm(new_a - old_a, 2)
        action_cycle_reward = -self.action_cycle_weight*action_diff

        # Normalize
        if self.norm_exp_reward:
            state_cycle_reward /= self.phase_k
            action_cycle_reward /= self.phase_k

        # Total
        cycle_reward = state_cycle_reward + action_cycle_reward
        return cycle_reward, state_cycle_reward, action_cycle_reward

    # Reset function
    # Resets our storage for new episode, sets initial state and count
    def reset(self):
        # Do the original reset
        obs = self.env.reset()

        # Reset our storage structures
        self.counts[:] = -1
        self.ex_states[:, :] = 0
        self.pro_states[:, :] = 0
        self.actions[:, :] = 0

        # Set initial state and count
        assert(self._elapsed_steps == 0)
        s_ex, s_pro = self.get_state()
        self.ex_states[0, :] = np.copy(s_ex)
        self.pro_states[0, :] = np.copy(s_pro)
        self.counts[0] = 0

        # Return initial obs
        return obs

    # Pass through _elapsed_steps
    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps
