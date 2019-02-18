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
from . import proprioceptive_humanoid_env
from . import explorer_ant
from . import explorer_humanoid
from . import geom_utils
from gym.spaces.box import Box
import math
import sys
sys.path.append('../')
from utils import RollingAverage

# Wrapper that defines our environment for our low level ant policy
# Everything is egocentric to the gent
class LowLevelExplorationEnv(gym.Wrapper):
    def __init__(self, env=None, opt=None):
        super(LowLevelExplorationEnv, self).__init__(env)

        # Change time limit
        time_limit = opt['env']['time_limit']
        assert(not env._max_episode_seconds)
        if self.env._max_episode_steps != time_limit:
            self.env._max_episode_steps = time_limit

        # Set theta mode
        self.theta_space_mode = opt['env']['theta_space_mode']
        assert(self.theta_space_mode in ['forward', 'pretrain_interp', 'pretrain_any', 'pretrain_any_far', 'pretrain_any_fromstart', 'pretrain_forward', 'pretrain_backward', 'pretrain_left', 'pretrain_right', 'simple_four', 'simple_eight', 'arbitrary', 'k_theta'])
        self.theta_reset_mode = opt['env']['theta_reset_mode']
        assert(self.theta_reset_mode in ['never', 'random_once'])
        self.theta_reward_mode = opt['env']['theta_reward_mode']
        assert(self.theta_reward_mode in ['lax', 'punish_dyaw'])
        self.theta_obs_mode = opt['env']['theta_obs_mode']
 
        # Add max norm
        if 'max_norm' in opt['env']:
            self.max_norm = opt['env']['max_norm']
        else:
            self.max_norm = float('inf')


        # Add backtrack multiplier
        if 'backtrack_mult' in opt['env']:
            self.backtrack_mult = opt['env']['backtrack_mult']
        else:
            self.backtrack_mult = 1

        # Get theta size for network
        assert(self.theta_obs_mode in ['ind', 'vector', 'pretrain'])
        if self.theta_obs_mode == 'ind': 
            if self.theta_space_mode == 'forward':
                self.theta_sz = 1
            elif self.theta_space_mode == 'simple_four':
                self.theta_sz = 4
            elif self.theta_space_mode == 'simple_eight':
                self.theta_sz = 8
            elif self.theta_space_mode == 'k_theta':
                self.theta_sz = opt['env']['num_theta']
                # Keeps the average movement for a particular theta
                self.direction_memory = [RollingAverage(opt['env']['replay_memory_sz'] * time_limit) for i in range(self.theta_sz)]
                self.dir_consistency = opt['env']['dir_consistency']   
            else:
                Exception("Ind not valid for space mode %s" % self.theta_space_mode)
        elif self.theta_obs_mode == 'vector':
            self.theta_sz = 2
        elif self.theta_obs_mode == 'pretrain':
            self.theta_sz = 0
        else:
            raise NotImplementedError

        # Add theta to observation_space size
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + self.theta_sz],
            dtype=self.observation_space.dtype)

        # Should be in correct mode (can be baseline or phase, but should be theta version)
        self.mode = opt['model']['mode']
        assert(self.mode in ['baseline_lowlevel', 'phase_lowlevel'])

        # Make sure we're using the right environment (our ant for now)
        if isinstance(env.unwrapped, ant_env.BaseAntEnv) or isinstance(env.unwrapped, ant_env.BaseAntLowGearEnv):
            self.env_base = 'ant'
            self.forward_mult = 1
        elif isinstance(env.unwrapped, proprioceptive_humanoid_env.BaseProprioceptiveHumanoidEnv):
            self.env_base = 'humanoid'
            if 'forward_mult' in opt['env']:
                self.forward_mult = opt['env']['forward_mult']
            else:
                self.forward_mult = 1
        else:
            raise NotImplementedError

        # Keep memory
        # Figure out sizes of external and external
        self.ex_states = []
        self.pro_states = []
        self.actions = []

        # Set reward shape type and exploration direction
        # reward_shape_type
        # periodic - only reward at the beginning of each new cycle
        # step - reward every step, but need at least k steps already
        # ramp - same as step, but before k, give weighted reward for movement since 0
        # instant - movement since last step
        assert opt['env']['reward_shape_type'] in ['ramp', 'instant'], "Invalid reward shape option"
        self.reward_shape_type = opt['env']['reward_shape_type']

        # Phase period
        self.phase_k = opt['model']['phase_period']

        # Options specific to phase
        if self.mode == 'phase_lowlevel':
            # Weights for action and state cyclicality constraints
            self.state_cycle_weight = opt['env']['state_cycle_weight']
            self.action_cycle_weight = opt['env']['action_cycle_weight']
            self.cycle_startup = opt['env']['cycle_startup']

        # How far back to look for consistent theta direction
        self.theta_memory_lookback = opt['env']['theta_memory_lookback']

        # If in the interp pretraining, load the states
        if self.theta_space_mode == 'pretrain_interp':
            self.fixed_states = opt['env']['fixed_states']
            assert(self.mode == 'phase_lowlevel')

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

        # Get the new "forward" reward
        forward_reward = self.forward_mult * self.forward_reward()
        info['reward_forward'] = forward_reward
        reward = forward_reward
        if self.env_base == 'humanoid' or self.env_base == 'simple_humanoid':
            info['reward_survive'] = info['reward_alive'] 
            info['reward_ctrl'] = info['reward_quadctrl']
            info['reward_contact'] = info['reward_impact']

        # Gew rewards specific to phase
        if self.mode == 'phase_lowlevel':
            # Get the cycle reward
            if self.state_cycle_weight > 0 or self.action_cycle_weight > 0:
                cycle_reward, s_cycle_reward, a_cycle_reward = self.cyclic_reward()
                info['reward_cycle'] = cycle_reward
                info['reward_cycle_s'] = s_cycle_reward
                info['reward_cycle_a'] = a_cycle_reward
                reward += cycle_reward

        # Calculate the new reward
        if 'reward_ctrl' in info:
            reward += info['reward_ctrl']
        if 'reward_contact' in info:
            reward += info['reward_contact']
        if 'reward_survive' in info:
            reward += info['reward_survive']

        # Set reward_exp for debug
        if self.mode == 'phase_lowlevel':
            info['reward_exp'] = reward
            
        # Calculate reward_env
        if self.env_base == 'ant':
            info['reward_env'] = info['reward_forward'] + info['reward_ctrl'] + info['reward_contact'] + info['reward_survive']
        elif self.env_base == 'swimmer':
            info['reward_env'] = info['reward_forward'] + info['reward_ctrl']
        elif self.env_base == 'humanoid' or self.env_base == 'simple_humanoid':
            info['reward_env'] = info['reward_forward'] + info['reward_ctrl'] + info['reward_survive'] + info['reward_contact']
        elif self.env_base == 'cheetah':
            info['reward_env'] = info['reward_forward'] + info['reward_ctrl']

        # Add theta to obs
        obs = self.add_theta_to_obs(obs)

        # Depending on mode, change theta
        if self.theta_reset_mode == 'never':
            pass
        elif self.theta_reset_mode == 'random_once':
            if new_count == self.reset_after_count:
                self.reset_theta()
        else:
            raise NotImplementedError

        # Return
        return obs, reward, done, info

    # Forward reward for baseline
    def forward_reward(self):
        # Get old and new states (depending on mode)
        assert(len(self.ex_states) > 0)
        new_state_ex = self.ex_states[-1]
        if self.reward_shape_type == 'instant':
            old_state_ex = self.ex_states[-2]
        elif self.reward_shape_type == 'ramp':
            if len(self.ex_states) <= self.phase_k:
                old_state_ex = self.ex_states[0]
            else:
                old_state_ex = self.ex_states[-(self.phase_k+1)]
            self.max_norm *= self.phase_k
        else:
            raise NotImplementedError

        # Extract xy
        # TODO - this is definitely hardcoded to our ant and swimmer (find better way to do this)
        old_xy = old_state_ex[0:2]
        new_xy = new_state_ex[0:2]
        delta_xy = new_xy - old_xy
        if self.env_base == 'ant' or self.env_base == 'humanoid' or self.env_base == 'simple_humanoid':
            yaw_ind = 3
        elif self.env_base == 'swimmer':
            yaw_ind = 2
        elif self.env_base == 'cheetah':
            yaw_ind = float('inf') # This value doesn't make sense for cheetah
        else:
            raise NotImplementedError

        # If the mode is anything other than pretrain_any (where we don't care about direction)
        if self.theta_space_mode not in ['pretrain_any', 'pretrain_any_far', 'pretrain_interp', 'pretrain_any_fromstart']:
            # Get the reference direction (get historic yaw from a few steps ago so we don't drift too far)
            # If memory lookback is a number, just look at that number
            if type(self.theta_memory_lookback) is int:  
                # Lookup old state
                if len(self.ex_states) <= self.theta_memory_lookback:
                    global_theta = self.ex_states[0][yaw_ind] 
                else:
                    global_theta = self.ex_states[-(self.theta_memory_lookback+1)][yaw_ind]
            elif self.theta_memory_lookback['mode'] == 'avg_window':
                lookback_start = self.theta_memory_lookback['start']
                lookback_end = self.theta_memory_lookback['end']
                lookback_window = lookback_start - lookback_end + 1
                assert(lookback_end < lookback_start)
                # Lookup old states
                if len(self.ex_states) <= lookback_window:
                    yaws = [s[yaw_ind] for s in self.ex_states]
                elif len(self.ex_states) <= lookback_start:
                    if lookback_end == 1:
                        yaws = [s[yaw_ind] for s in self.ex_states]
                    else:
                        yaws = [s[yaw_ind] for s in self.ex_states[0:-lookback_end+1]]
                    assert(len(yaws) == lookback_window)
                else:
                    if lookback_end == 1:
                        yaws = [s[yaw_ind] for s in self.ex_states[-lookback_start:]]
                    else:
                        yaws = self.ex_states[-lookback_start:-lookback_end+1]
                    assert(len(yaws) == lookback_window)
                global_theta = geom_utils.average_angles(yaws)

            # Get the egocentric xy
            delta_xy = geom_utils.convert_vector_to_egocentric(global_theta, delta_xy)

        # Calculate forward movement
        if self.theta_space_mode in ['pretrain_any_fromstart']:
            # Find delta movement from the first position
            delta_from_start = np.linalg.norm(self.ex_states[0][:2]-new_xy, 2)

            # Subtract the delta movement from the last time step
            old_delta = np.linalg.norm(self.ex_states[-2][:2]-self.ex_states[0][:2], 2)
            forward_reward = (delta_from_start - old_delta) / self.unwrapped.dt            

            #forward_reward = (self.ex_states[-1][0] - self.ex_states[-2][0]) / self.unwrapped.dt
        elif self.theta_space_mode in ['pretrain_any_far']:
            # Find sum delta movement from all previous external positions
            sum_delta = 0
            denom = 0
            for s_count, old_state in enumerate(self.ex_states[:-1]):
                mult_fact = math.pow(self.backtrack_mult,(len(self.ex_states) - s_count))
                denom += mult_fact
                sum_delta += mult_fact*np.linalg.norm(old_state[0:2]-new_xy, 2) / (len(self.ex_states) - s_count)
            forward_reward = sum_delta / (denom*self.unwrapped.dt)

        elif self.theta_space_mode in ['pretrain_any', 'pretrain_interp']:
            # Reward any movement in any direction
            forward_reward = np.linalg.norm(delta_xy, 2) / self.unwrapped.dt
        elif self.theta_space_mode == 'k_theta':
            # Figure out how far off we are from other thetas
            delta_xy_unit = delta_xy / np.linalg.norm(delta_xy, 2)
            assert(abs(np.linalg.norm(delta_xy_unit)-1) < 1e-6)  
             
            # Until we've gotten to a certain number of counts, start with just raw exploration reward
            if any(len(ra.data) < self.env._max_episode_steps * 5 for ra in self.direction_memory):  
                forward_reward = min(np.linalg.norm(delta_xy, 2), self.max_norm)
            else:
                # Get the unit direction for each k (except the current one)
                unit_vecs_k = [ra.average() / np.linalg.norm(ra.average(), 2) for ra in self.direction_memory]
                del unit_vecs_k[self.theta_ind]

                # Calculate the unit vector of our movement and get the similarities for each unit vec (except current)
                sims = [(np.dot(delta_xy_unit, unit_k) + 1)/2 for unit_k in unit_vecs_k]
                assert(all(sim >= 0 and sim <= 1 for sim in sims))

                # Reward is the norm of the movement weighted by the average similarity to other k's
                forward_reward = min(np.linalg.norm(delta_xy, 2), self.max_norm) * np.mean([(1-sim) for sim in sims])

            # Use direction memory from 10 steps ago and attenuate based on what direction we were in
            # TODO - this is super hard-coded
            if self.env._elapsed_steps > 10 and self.dir_consistency:
                old_delta_xy = self.direction_memory[self.theta_ind].data[-10]
                unit_old = old_delta_xy / np.linalg.norm(old_delta_xy)
                sim = (np.dot(delta_xy_unit, unit_old) + 1)/2
                forward_reward *= sim
            forward_reward /= self.unwrapped.dt

            # Update the memory with the delta state
            self.direction_memory[self.theta_ind].append(delta_xy)
        else:
            # Get the parallel part of the movement (dot product)
            parallel_movement = np.dot(delta_xy, self.theta) / self.unwrapped.dt
       
            # Get the perpendicular part of the movement (abs of cross product)
            perp_movement = abs(np.cross(delta_xy, self.theta)) / self.unwrapped.dt

            assert(self.theta_reward_mode in ['lax', 'punish_dyaw'])
            forward_reward = parallel_movement
            if self.theta_reward_mode == 'lax':
                pass
            elif self.theta_reward_mode == 'punish_dyaw':
                forward_reward -= 0.5 * np.square(self.ex_states[-1][7])
            elif self.theta_reward_mode == 'punish_perp':
                forward_reward -= perp_movement
            elif self.theta_reward_mode == 'punish_perp_clip':
                if forward_reward > 0:
                    forward_reward = max(0, forward_reward - perp_movement)
            elif self.theta_reward_mode == 'forward_window':
                forward_reward = max(0, forward_reward - perp_movement)
            else:
                raise NotImplementedError

        # Renormalize for phase (if not instant mode)
        if self.reward_shape_type != 'instant':
            forward_reward /= self.phase_k
            
        return forward_reward

    # Cyclicality reward
    def cyclic_reward(self):
        # Init rewards
        state_cycle_reward = 0
        action_cycle_reward = 0       

        # If we're in interp mode, we want to match to fixed states
        if self.theta_space_mode == 'pretrain_interp':
            new_count = self._elapsed_steps
            assert(len(self.ex_states) == new_count + 1)

            # Figure out where velocities start
            if self.env_base == 'ant':
                vel_start = 10
            elif self.env_base == 'swimmer':
                vel_start = 2
            else:
                raise NotImplementedError

            if new_count >= self.phase_k and new_count % self.phase_k/2 == 0:
                if new_count % self.phase_k == 0:
                    ref_state = self.fixed_states[0][:vel_start]
                else:
                    ref_state = self.fixed_states[1][:vel_start]
                
                # Compare (non-velocity states)
                cur_s = self.pro_states[-1][:vel_start]
                state_diff = np.linalg.norm(cur_s - ref_state, 2)
                state_cycle_reward = -state_diff*self.state_cycle_weight    # Set as state cycle reward
        else:
            # Check if we are far enough in episode 
            # If cycle startup, don't punish until two cycles are done
            if self.cycle_startup > 0:
                mincount = (self.cycle_startup+1)*self.phase_k
            # Never start until phase_k + 1 (so we ignore the reset state/action)
            else:
                mincount = self.phase_k + 1
            
            # If count is sufficient, get cycle reward
            if len(self.pro_states) > mincount:
                # Get last/current cycle state and actions
                new_s = np.array(self.pro_states[-1])
                old_s = np.array(self.pro_states[-(self.phase_k+1)])
                new_a = np.array(self.actions[-1])
                old_a = np.array(self.actions[-(self.phase_k+1)])

                # Do deg to rad conversions if necessary
                #if self.env_base == 'simple_humanoid':
                #    new_s *= math.pi / 180
                #    old_s *= math.pi / 180
                #    new_a *= math.pi / 180
                #    old_a *= math.pi / 180

                # Get cyclic reward for state
                state_diff = np.linalg.norm(new_s - old_s, 2)
                state_cycle_reward = -self.state_cycle_weight*state_diff

                # Get cyclic reward for state
                action_diff = np.linalg.norm(new_a - old_a, 2)
                action_cycle_reward = -self.action_cycle_weight*action_diff

        # Total
        cycle_reward = state_cycle_reward + action_cycle_reward
        return cycle_reward, state_cycle_reward, action_cycle_reward

    # Reset function
    # Resets our storage for new episode, sets initial state and count
    def reset(self):
        # Do the original reset
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

        # If in random once theta reset, randomly choose index to change after
        if self.theta_reset_mode in ['random_once']:
            # Choose between 0 and max (2 of those mean no reset effectively)
            self.reset_after_count = random.randint(0, self.env._max_episode_steps)

        # Reset theta
        self.reset_theta()
        
        # Add theta to obs
        obs = self.add_theta_to_obs(obs)
        
        # Return initial obs
        return obs

    # Reset theta
    def reset_theta(self):
        if self.theta_space_mode == 'arbitrary':
            # Choose a random unit vector in R2
            angle = np.pi * np.random.uniform(0, 2)
            x = np.cos(angle)
            y = np.sin(angle)
            self.theta = np.array([x, y])
        else: 
            # Choose theta index 
            if self.theta_space_mode == 'forward':
                self.theta_ind = 0
            elif self.theta_space_mode == 'pretrain_forward':
                self.theta_ind = 0
            elif self.theta_space_mode == 'pretrain_backward':
                self.theta_ind = 1
            elif self.theta_space_mode == 'pretrain_left':
                self.theta_ind = 2
            elif self.theta_space_mode == 'pretrain_right':
                self.theta_ind = 3
            elif self.theta_space_mode in ['pretrain_any','pretrain_any_far','pretrain_any_fromstart', 'pretrain_interp']:
                return
            elif self.theta_space_mode == 'simple_four':
                self.theta_ind = random.randint(0, 3)
            elif self.theta_space_mode == 'simple_eight':
                self.theta_ind = random.randint(0, 7)
            elif self.theta_space_mode == 'k_theta':
                self.theta_ind = random.randint(0, self.theta_sz-1)
                return
            else:
                raise NotImplementedError       
    
            # Get theta unit vector
            # 0 - x forward 
            if self.theta_ind == 0:
                self.theta = np.array([1, 0])
            # 1 - x backward
            elif self.theta_ind == 1:
                self.theta = np.array([-1, 0])
            # 2 - y forward 
            elif self.theta_ind == 2:
                self.theta = np.array([0, 1])
            # 3 - y backward
            elif self.theta_ind == 3:
                self.theta = np.array([0, -1])
            # 4 - forward and left
            elif self.theta_ind == 4:
                self.theta = np.array([math.sqrt(0.5), math.sqrt(0.5)])
            # 5 - backward and left
            elif self.theta_ind == 5:
                self.theta = np.array([-math.sqrt(0.5), math.sqrt(0.5)])
            # 6 - backward and right
            elif self.theta_ind == 6:
                self.theta = np.array([-math.sqrt(0.5), -math.sqrt(0.5)])
            # 7 - forward and right
            elif self.theta_ind == 7:
                self.theta = np.array([math.sqrt(0.5), -math.sqrt(0.5)])
            else:
                Exception("Theta is invalid")
           
    # Add theta to the observation
    def add_theta_to_obs(self, obs):
        # Get theta representation for obs
        if self.theta_obs_mode == 'pretrain':
            return obs
        elif self.theta_obs_mode in ['ind', 'k_theta']: 
            theta_obs = np.zeros(self.theta_sz)
            theta_obs[self.theta_ind] = 1 
        elif self.theta_obs_mode == 'vector':
            theta_obs = np.array(self.theta)
        else:
            raise NotImplementedError

        # Concat and return
        return np.concatenate((obs, theta_obs))

    # Pass through _elapsed_steps
    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps

