import os
import collections
import pdb
import gym
import gym.envs.mujoco
import time
import csv
import json
import shutil
import numpy as np
import random
from . import ant_env
from . import proprioceptive_humanoid_env
from . import maze_ant
from . import maze_humanoid

# Wrapper that records everything we might care about in our environment
# All rewards (clipped and raw), states, actions, time and steps
# Copied originally from https://github.com/openai/baselines/blob/master/baselines/bench/monitor.py
class SmartMonitor(gym.Wrapper):
    def __init__(self, env, log_dir, rank, opt, verbose=True, allow_early_resets=False):
        super(SmartMonitor, self).__init__(env)
        self.tstart = time.time()
        self.episode_count = -1
            
        # Get the rewards we want to log
        # Got to be a better way to get the names of the subpart rewards, but it seems to be hardcoded in the mujoco envs
        self.reward_list = ['reward_env']
        if opt['model']['mode'] in ['baseline', 'baseline_reverse', 'baselinewtheta', 'baseline_lowlevel']:
            self.baseline = True
        elif opt['model']['mode'] in ['phasesimple', 'phasewstate', 'phasewtheta', 'phase_lowlevel']:
            self.baseline = False
            self.reward_list.append('reward_exp')
            if opt['model']['mode'] != 'phase_lowlevel':
                self.reward_list.append('reward_move')
            if opt['env']['state_cycle_weight'] > 0 or opt['env']['action_cycle_weight'] > 0:
                self.reward_list.append('reward_cycle')
                self.reward_list.append('reward_cycle_s')
                self.reward_list.append('reward_cycle_a')
        elif opt['model']['mode'] == 'interpolate':
            self.baseline = False
            self.reward_list.append('reward_interpolate')
        elif opt['model']['mode'] == 'cyclic':
            self.baseline = False
            self.reward_list.append('reward_cycle')
            self.reward_list.append('reward_thresh')
        elif opt['model']['mode'] in ['hierarchical', 'hierarchical_many']:
            self.baseline = True
            self.reward_list.append('reward_velocity')
            self.reward_list.append('reward_goal')
        elif opt['model']['mode'] in [ 'maze_baseline', 'maze_baseline_wphase']:
            self.baseline = True
            self.reward_list.append('reward_velocity')
            self.reward_list.append('reward_goal')
        else:
            raise NotImplementedError

        # This is currently hardcoded to Mujoco envs
        if isinstance(env.unwrapped, ant_env.BaseAntEnv) or isinstance(env.unwrapped, ant_env.BaseAntLowGearEnv) or isinstance(env.unwrapped, proprioceptive_humanoid_env.BaseProprioceptiveHumanoidEnv):
            self.reward_list += ['reward_forward', 'reward_ctrl', 'reward_contact', 'reward_survive']   
        elif isinstance(env.unwrapped, gym.envs.mujoco.AntEnv):
            self.reward_list += ['reward_forward', 'reward_ctrl', 'reward_contact', 'reward_survive']
        else:
            raise NotImplementedError

        # Data structure that holds all the values we want to log
        self.episode_struct = collections.OrderedDict()
        all_keys = self.reward_list + ['obs', 'action', 'env_count', 'episode_count']
        if isinstance(env.unwrapped, ant_env.BaseAntEnv) or isinstance(env.unwrapped, ant_env.BaseAntLowGearEnv) or isinstance(env.unwrapped, proprioceptive_humanoid_env.BaseProprioceptiveHumanoidEnv) or isinstance(env.unwrapped, gym.envs.mujoco.MujocoEnv):
            all_keys += ['state']
        # Log the distances
        if opt['model']['mode'] in ['hierarchical', 'hierarchical_many', 'maze_baseline', 'maze_baseline_wphase']:
            if isinstance(env.unwrapped, maze_humanoid.ProprioceptiveHumanoidMazeEnv) or isinstance(env.unwrapped, maze_ant.AntMazeEnv):
                all_keys += ['goal_distance', 'goal_distance_radius']
        for key in all_keys:
            self.episode_struct[key] = []

        # Create and initialize our csv files
        # File to store entire episode information (rather than every single step)
        # Prints total reward (for all rewards), overall obs and state displacements, episode length, and episode time
        episode_filename = os.path.join(log_dir, str(rank) + '.Episode.Monitor.csv')
        self.ep_f = open(episode_filename, "wt")
        self.ep_f.write('# Episode Logging %s\n'%json.dumps({"t_start": self.tstart, 'env_id' : env.spec and env.spec.id, 'mode': opt['model']['mode'], 'name': opt['logs']['exp_name']}))
        ep_fields = self.reward_list + ['delta_obs', 'mean_action', 'episode_len', 'episode_dt', 'episode_count'] 
        if isinstance(env.unwrapped, ant_env.BaseAntEnv) or isinstance(env.unwrapped, ant_env.BaseAntLowGearEnv) or isinstance(env.unwrapped, proprioceptive_humanoid_env.BaseProprioceptiveHumanoidEnv) or isinstance(env.unwrapped, gym.envs.mujoco.MujocoEnv):
            ep_fields += ['delta_state']
        if opt['model']['mode'] in ['hierarchical', 'hierarchical_many', 'maze_baseline', 'maze_baseline_wphase']:
            if isinstance(env.unwrapped, maze_humanoid.ProprioceptiveHumanoidMazeEnv) or isinstance(env.unwrapped, maze_ant.AntMazeEnv):
                ep_fields += ['goal_distance', 'goal_distance_radius']
        self.ep_logger = csv.DictWriter(self.ep_f, fieldnames=ep_fields)
        self.ep_logger.writeheader()
        self.ep_f.flush()
            
        # If in super verbose mode
        if verbose:
            # File to store every step
            # Prints everything in episode_struct plus episode count
            step_filename = os.path.join(log_dir, str(rank) + '.Step.Monitor.csv')
            self.st_f = open(step_filename, "wt")
            self.st_f.write('# Episode Logging %s\n'%json.dumps({"t_start": self.tstart, 'env_id' : env.spec and env.spec.id, 'mode': opt['model']['mode'], 'name': opt['logs']['exp_name']}))
            st_fields = list(self.episode_struct.keys())
            self.st_logger = csv.DictWriter(self.st_f, fieldnames=st_fields)
            self.st_logger.writeheader()
            self.st_f.flush()
        else:
            self.st_f = None

        self.verbose = verbose
        self.rank = rank
        self.opt = opt
        self.log_dir = log_dir

        # Other bookkeeping 
        self.allow_early_resets = allow_early_resets                                                                                                                                              
        self.needs_reset = True
        self.total_steps = 0
        self.current_reset_info = {} # extra info about the current episode, that was passed in during reset()

    # Reset environment, record initial values
    def reset(self, **kwargs):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, wrap your env with Monitor(env, path, allow_early_resets=True)")

        # Reset all the values in self.episode_struct
        for key in self.episode_struct:
            self.episode_struct[key] = []
            
        # Update episode count
        self.episode_count += 1
        
        # Update values and return
        obs = self.env.reset(**kwargs)
        self.record_info(obs, 0)
        self.needs_reset = False
        return obs

    # Take a step, update all the values
    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")

        # Do step
        obs, rew, done, info = self.env.step(action)

        # Record new info
        self.record_info(obs, rew, action, info)

        # If done with episode, get summary info for episode and dump values to episode and step files
        if done:
            self.needs_reset = True

            # For rewards, get sums
            epinfo = {}
            for key in self.reward_list:
                reward_val = sum(self.episode_struct[key])
                epinfo[key] = reward_val

            # For obs and state, get delta change
            epinfo['delta_obs'] = self.episode_struct['obs'][-1] - self.episode_struct['obs'][0]
            if 'state' in self.episode_struct:
                epinfo['delta_state'] = self.episode_struct['state'][-1] - self.episode_struct['state'][0]

            # For action, get average value
            epinfo['mean_action'] = np.mean(self.episode_struct['action'], axis=0)

            # Update episode_len, episode_dt and episode_count
            epinfo['episode_len'] = len(self.episode_struct['env_count'])
            epinfo['episode_dt'] = round(time.time() - self.tstart, 6)
            epinfo['episode_count'] = self.episode_count

            # Update goal distances
            if 'goal_distance' in self.episode_struct:
                epinfo['goal_distance'] = self.episode_struct['goal_distance'][-1]
                epinfo['goal_distance_radius'] = self.episode_struct['goal_distance_radius'][-1]
            elif 'key_distance' in self.episode_struct:
                epinfo['key_distance'] = self.episode_struct['key_distance'][-1]
                epinfo['key_distance_radius'] = self.episode_struct['key_distance_radius'][-1]
                epinfo['lock_distance'] = self.episode_struct['lock_distance'][-1]
                epinfo['lock_distance_radius'] = self.episode_struct['lock_distance_radius'][-1] 

            # Do string conversion
            for k in epinfo:
                epinfo[k] = str(epinfo[k]).replace('\n', '')

            # Update episode file
            if self.ep_logger:
                self.ep_logger.writerow(epinfo)
                self.ep_f.flush()

            # If in super verbose mode
            if self.verbose:
                # Make and update a temp step file with just the last episode (and only rank 0, and only every 100)
                if self.rank == 0: #and self.episode_count % 100 == 0:
                    # Setup temp file
                    tmp_step_filename = os.path.join(self.log_dir, 'Tmp.Last.Step.Monitor.csv')
                    tmp_f = open(tmp_step_filename, "wt")
                    tmp_f.write('# Episode Logging %s\n'%json.dumps({"t_start": self.tstart, 'env_id' : self.env.spec and self.env.spec.id, 'mode': self.opt['model']['mode'], 'name': self.opt['logs']['exp_name']}))
                    st_fields = list(self.episode_struct.keys())
                    tmp_logger = csv.DictWriter(tmp_f, fieldnames=st_fields)
                    tmp_logger.writeheader()
                    tmp_f.flush()
                else:
                    tmp_f = None

                # Update step file
                assert(self.episode_struct['env_count'][-1]+1 == len(self.episode_struct['env_count']))
                for step in range(len(self.episode_struct['env_count'])):
                    stepinfo = {}
                    for key in self.episode_struct:
                        stepinfo[key] = self.episode_struct[key][step]

                    # Do string conversion
                    for k in stepinfo:
                        stepinfo[k] = str(stepinfo[k]).replace('\n', '')

                    # Update loggers
                    self.st_logger.writerow(stepinfo)
                    if tmp_f is not None:
                        tmp_logger.writerow(stepinfo)
                self.st_f.flush()
                
                # Write tmp file and close, copy tmp to last
                if tmp_f is not None:
                    tmp_f.flush()
                    tmp_f.close()

                    # Copy tmp to last
                    last_step_filename = os.path.join(self.log_dir, 'Last.Step.Monitor.csv')
                    shutil.copyfile(tmp_step_filename, last_step_filename)

            # Update info
            info['episode'] = epinfo
        self.total_steps += 1
        return (obs, rew, done, info)

    # Record step info
    def record_info(self, obs, rew, action=None, info=None):
        # Update all of our values
        # Reward values
        for key in self.reward_list:
            # If reset, all 0
            if info is None:
                self.episode_struct[key].append(0)
            else:
                # For baseline, reward_env is reward
                if key == 'reward_env' and self.baseline:
                    self.episode_struct[key].append(rew)
                else:
                    self.episode_struct[key].append(info[key])           

        # Observation values
        self.episode_struct['obs'].append(obs)

        # State values, right now just Mujoco
        if isinstance(self.env.unwrapped, ant_env.BaseAntEnv) or isinstance(self.env.unwrapped, ant_env.BaseAntLowGearEnv) or isinstance(self.env.unwrapped, proprioceptive_humanoid_env.BaseProprioceptiveHumanoidEnv)  or isinstance(self.env.unwrapped, gym.envs.mujoco.MujocoEnv):
            state = self.env.unwrapped.state_vector()
            self.episode_struct['state'].append(state)

        # Update actions
        if action is None:
            action = np.zeros(self.env.action_space.shape)
        self.episode_struct['action'].append(action)

        # Update step and episode counts
        env_count = self.env._elapsed_steps
        self.episode_struct['env_count'].append(env_count)
        self.episode_struct['episode_count'].append(self.episode_count)

        # Update distances
        if 'goal_distance' in self.episode_struct:
            if info is None:
                self.episode_struct['goal_distance'].append(0)
                self.episode_struct['goal_distance_radius'].append(0) 
            else:
                self.episode_struct['goal_distance'].append(info['goal_distance'])
                self.episode_struct['goal_distance_radius'].append(info['goal_distance_radius'])

    # Close file handles
    def close(self):
        if self.ep_f is not None:
            self.ep_f.close()
        if self.st_f is not None:
            self.st_f.close()

    # Get total number of steps
    def get_total_steps(self):
        return self.total_steps

