# By Kenneth Marino 
# This file loads model checkpoint(s) and performs evaluation and/or dumps video
# Modified from code from repo https://github.com/ikostrikov/pytorch-a2c-ppo-acktr 

import argparse
import yaml
from pprint import pprint
import click
import shutil
import copy
import glob
import os
import time
import scipy
import pdb

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from vec_env import DummyVecEnv, SubprocVecEnv
from wrappers import ObservationFilter
from environments.envs import make_env
from model import Policy
from storage import RolloutStorage
from visualize import Dashboard

import algo

# Get Input Arguments
parser = argparse.ArgumentParser(description='RL')

##################################################
# yaml options file contains all default choices #
parser.add_argument('--path_opt', default='options/baseline/default.yaml', type=str, 
                    help='path to a yaml options file')
parser.add_argument('--vis_path_opt', default='options/visualization/eval_default.yaml', type=str,
                    help='path to a yaml visualization options file')
##################################################
parser.add_argument('--algo', default='a2c',
                    help='algorithm to use: a2c | ppo | acktr') 
parser.add_argument('--env-name', default='Hopper-v2',
                    help='environment to train on (default: Hopper-v2)')
parser.add_argument('--ckpt', default='', type=str,
                    help='path to checkpoint we want to evaluate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-vis', action='store_true', default=False,
                    help='disables visdom visualization')
parser.add_argument('--port', type=int, default=8097,
                    help='port to run the server on (default: 8097)')
parser.add_argument('--no-verbose', action='store_true', default=False,
                    help='if true, print step logs')
parser.add_argument('--logdir', default='', type=str,
                    help='Where to store logs/video. Ideally, same dir as ckpt')
parser.add_argument('--num-ep', default=100, type=int,
		            help='How many episodes to run with this checkpoint')
parser.add_argument('--num-vid', default=0, type=int,
		            help='How many episodes to save a video for. 0 means no video')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--dump-obsdata', action='store_true', default=False,
                    help='Dump raw obs and action to file')

# Takes in one checkpointed mode (for multiple checkpoints, run more than once)
# Runs the environments forward for num_ep episodes
# Records everything like in training (but should dump to different location)
# Optionally dump num_vid of videos (0 by default)
def main():
    global args
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    # Set options
    if args.path_opt is not None:
        with open(args.path_opt, 'r') as handle:
            options = yaml.load(handle)
    if args.vis_path_opt is not None:
        with open(args.vis_path_opt, 'r') as handle:
            vis_options = yaml.load(handle)
    print('## args'); pprint(vars(args))
    print('## options'); pprint(options)

    # Put alg_%s and optim_%s to alg and optim depending on commandline
    options['use_cuda'] = args.cuda
    options['alg'] = options['alg_%s' % args.algo]
    options['optim'] = options['optim_%s' % args.algo]
    options['trial'] = 0 # Hard coded / doesn't matter
    alg_opt = options['alg']
    alg_opt['algo'] = args.algo
    model_opt = options['model']
    env_opt = options['env']
    env_opt['env-name'] = args.env_name
    log_opt = options['logs']
    optim_opt = options['optim']
    model_opt['time_scale'] = env_opt['time_scale']
    if model_opt['mode'] in ['baselinewtheta', 'phasewtheta']:
        model_opt['theta_space_mode'] = env_opt['theta_space_mode']
        model_opt['theta_sz'] = env_opt['theta_sz']
    elif model_opt['mode'] in ['baseline_lowlevel', 'phase_lowlevel']:
        model_opt['theta_space_mode'] = env_opt['theta_space_mode']

    # Check asserts
    assert(model_opt['mode'] in ['baseline', 'phasesimple', 'phasewstate', 'baselinewtheta', 'phasewtheta', 'baseline_lowlevel', 'phase_lowlevel', 'interpolate', 'cyclic'])
    assert(args.algo in ['a2c', 'ppo', 'acktr'])
    if model_opt['recurrent_policy']:
        assert args.algo in ['a2c', 'ppo'], 'Recurrent policy is not implemented for ACKTR'

    # Set seed 
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    torch.set_num_threads(1)

    # Set logging / load previous checkpoint
    logpath = args.logdir

    # Make directory, check before overwriting
    assert not os.path.isdir(logpath), "Give a new directory to save so we don't overwrite anything"
    os.system('mkdir -p ' + logpath)

    # Load checkpoint
    assert(os.path.isfile(args.ckpt))
    if args.cuda:
        ckpt = torch.load(args.ckpt)
    else:
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

    # Save options and args
    with open(os.path.join(logpath, os.path.basename(args.path_opt)), 'w') as f:
        yaml.dump(options, f, default_flow_style=False)
    with open(os.path.join(logpath, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    # Save git info as well
    os.system('git status > %s' % os.path.join(logpath, 'git_status.txt'))
    os.system('git diff > %s' % os.path.join(logpath, 'git_diff.txt'))
    os.system('git show > %s' % os.path.join(logpath, 'git_show.txt'))

    # Set up plotting dashboard
    dashboard = Dashboard(options, vis_options, logpath, vis=args.vis, port=args.port)

    # Create environment
    verbose = not args.no_verbose
    fixed_states = [np.zeros(20), np.zeros(20)]
    env = make_env(args.env_name, args.seed, 0, logpath, options, verbose, fixed_states)
    env = DummyVecEnv([env])
    if len(env.observation_space.shape) == 1:
        ignore_mask = np.zeros(env.observation_space.shape)
        if env_opt['add_timestep']:
            ignore_mask[-1] = 1
        if model_opt['mode'] in ['baselinewtheta', 'phasewtheta']:
            theta_sz = env_opt['theta_sz']
            if env_opt['add_timestep']:
                ignore_mask[-(theta_sz+1):] = 1
            else:
                ignore_mask[-theta_sz:] = 1
        env = ObservationFilter(env, ret=False, has_timestep=env_opt['add_timestep'], noclip=env_opt['step_plus_noclip'], ignore_mask=ignore_mask, time_scale=env_opt['time_scale'], gamma=env_opt['gamma'], train=False)
        env.ob_rms = ckpt['ob_rms'] 
        raw_env = env.venv.envs[0]
    else:
        raw_env = env.envs[0]

    # Get theta_sz for models (if applicable)
    if model_opt['mode'] == 'baseline_lowlevel':
        model_opt['theta_sz'] = env.venv.envs[0].env.theta_sz
    elif model_opt['mode'] == 'phase_lowlevel':
        model_opt['theta_sz'] = env.venv.envs[0].env.env.theta_sz   
    if 'theta_sz' in model_opt:
        env_opt['theta_sz'] = model_opt['theta_sz']

    # Init obs/state structures
    obs_shape = env.observation_space.shape
    obs_shape = (obs_shape[0] * env_opt['num_stack'], *obs_shape[1:])

    # Create the policy network
    actor_critic = Policy(obs_shape, env.action_space, model_opt)
    if args.cuda:
        actor_critic.cuda()

    # Load the checkpoint
    actor_critic.load_state_dict(ckpt['agent']['model'])
    if not args.cuda:
        actor_critic.base.cuda = False    

    # Inline define our helper function for updating obs
    def update_current_obs(obs):
        shape_dim0 = env.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if env_opt['num_stack'] > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    # Loop through episodes
    obs = env.reset()
    assert(args.num_vid <= args.num_ep) 
    episode_rewards = []
    tabbed = False
    raw_data = []
    for ep in range(args.num_ep):
        if ep < args.num_vid:
            record = True
        else:
            record = False

        # Reset env
        current_obs = torch.zeros(1, *obs_shape)
        states = torch.zeros(1, actor_critic.state_size)
        masks = torch.zeros(1, 1)
        update_current_obs(obs)

        # Complete episode
        done = False
        frames = []
        ep_total_reward = 0
        while not done:
            # Capture screenshot
            if record:
                raw_env.render()
                if not tabbed:
                    # GLFW TAB and RELEASE are hardcoded here
                    raw_env.unwrapped.viewer.key_callback(None, 258, None, 0, None)
                    tabbed = True
                frames.append(raw_env.unwrapped.viewer._read_pixels_as_in_window())
            
            # Determine action
            with torch.no_grad():
                value, action, _, states = actor_critic.act(current_obs, states, masks, deterministic=True)
            cpu_actions = action.squeeze(1).cpu().numpy()

            # Add to dataset (if applicable)
            if args.dump_obsdata:
                action_cp = np.array(cpu_actions)
                raw_obs_cp = np.array(env.raw_obs)
                raw_data.append([raw_obs_cp, action_cp])

            # Observe reward and next obs
            obs, reward, done, info = env.step(cpu_actions)
            ep_total_reward += reward            

            # Update obs
            masks.fill_(0.0 if done else 1.0)
            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks
            update_current_obs(obs)

        # Save video
        if record:
            for fr_ind, fr in enumerate(frames): 
                scipy.misc.imsave(os.path.join(logpath, 'tmp_fr_%d.jpg' % fr_ind), fr)
            os.system("ffmpeg -r 20 -i %s/" % logpath + "tmp_fr_%01d.jpg -y " + "%s/results_ep%d.mp4" % (logpath, ep))
            os.system("rm %s/tmp_fr*.jpg" % logpath)

        # Do dashboard logging for each epsiode
        try:
            dashboard.visdom_plot()
        except IOError:
            pass

        # Print / dump reward for episode
        # DEBUG for thetas
        #print("Theta %d" % env.venv.envs[0].env.env.theta)
        print("Total reward for episode %d: %f" % (ep, ep_total_reward))
        episode_rewards.append(ep_total_reward)

    # Dump episode data to file
    if args.dump_obsdata:
        torch.save(raw_data, logpath + '/raw_episode_data.tar')

    # Print average and variance of rewards
    avg_r = np.mean(episode_rewards)
    std_r = np.std(episode_rewards)
    print("Reward over episodes was %f+-%f" % (avg_r, std_r))

    # Do dashboard logging
    try:
        dashboard.visdom_plot()
    except IOError:
        pass

if __name__ == "__main__":
    main()
