# Modified by Kenneth Marino 
# This file trains the baseline actor critic methods
# Modified from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr 

import argparse
import yaml
import json
import csv
from pprint import pprint
import click
import shutil
import copy
import glob
import os
import time
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
from model import Policy, ModularPolicy, DQNPolicy
from storage import RolloutStorage, MaskingRolloutStorage
from visualize import Dashboard
from hier_utils import HierarchyUtils
from gym import spaces

import algo

# Get Input Arguments
parser = argparse.ArgumentParser(description='RL')

##################################################
# yaml options file contains all default choices #
parser.add_argument('--path_opt', default='options/baseline/default.yaml', type=str, 
                    help='path to a yaml options file')
# yaml option file containing the visdom plotting options
parser.add_argument('--vis_path_opt', default='options/visualization/default.yaml', type=str, 
                    help='path to a yaml visualization options file')
##################################################
parser.add_argument('--algo', default='a2c',
                    help='algorithm to use: a2c | ppo | acktr | dqn') 
parser.add_argument('--env-name', default='Hopper-v2',
                    help='environment to train on (default: Hopper-v2)')
parser.add_argument('--trial', type=int, default=0,
                    help='keep track of what trial you are on')
parser.add_argument('--save-every', default=1000000000, type=int,
                    help='how often to save our models permanently')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-vis', action='store_true', default=False,
                    help='disables visdom visualization')
parser.add_argument('--port', type=int, default=8097,
                    help='port to run the server on (default: 8097)')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='if true, print step logs')
parser.add_argument('--profile', action='store_true', default=False,
                    help='if true, print how long different parts of the code take')
parser.add_argument('--change-ll-offset', action='store_true', default=False,
                    help='if true, changes which lowlevel policy we choose based on offset val')

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

    # Load the lowlevel opt and 
    lowlevel_optfile = options['lowlevel']['optfile']
    with open(lowlevel_optfile, 'r') as handle:
        ll_opt = yaml.load(handle)

    # Whether we should set ll policy to be deterministic or not
    ll_deterministic = options['lowlevel']['deterministic']

    # Put alg_%s and optim_%s to alg and optim depending on commandline
    options['use_cuda'] = args.cuda
    options['trial'] = args.trial
    options['alg'] = options['alg_%s' % args.algo]
    options['optim'] = options['optim_%s' % args.algo]
    alg_opt = options['alg']
    alg_opt['algo'] = args.algo
    model_opt = options['model']
    env_opt = options['env']
    env_opt['env-name'] = args.env_name
    log_opt = options['logs']
    optim_opt = options['optim'] 
    options['lowlevel_opt'] = ll_opt    # Save low level options in option file (for logging purposes)

    # Pass necessary values in ll_opt
    assert(ll_opt['model']['mode'] in ['baseline_lowlevel', 'phase_lowlevel'])
    ll_opt['model']['theta_space_mode'] = ll_opt['env']['theta_space_mode']
    ll_opt['model']['time_scale'] = ll_opt['env']['time_scale']


    # If in many module mode, load the lowlevel policies we want
    if model_opt['mode'] == 'hierarchical_many':
        # Check asserts
        theta_obs_mode = ll_opt['env']['theta_obs_mode']
        theta_space_mode = ll_opt['env']['theta_space_mode']
        assert(theta_space_mode in ['pretrain_interp', 'pretrain_any', 'pretrain_any_far', 'pretrain_any_fromstart'])
        assert(theta_obs_mode == 'pretrain')

        # Get the theta size
        theta_sz = options['lowlevel']['num_load']
        ckpt_base = options['lowlevel']['ckpt']

        # Load checkpoints
        lowlevel_ckpts = []
        for ll_ind in range(theta_sz):
            if args.change_ll_offset:
                ll_offset = theta_sz * args.trial
            else:
                ll_offset = 0
            lowlevel_ckpt_file = ckpt_base + '/trial%d/ckpt.pth.tar' % (ll_ind+ll_offset)
            assert(os.path.isfile(lowlevel_ckpt_file))
            lowlevel_ckpts.append(torch.load(lowlevel_ckpt_file))

    # Otherwise it's one ll polciy to load
    else:
        # Get theta_sz for low level model
        theta_obs_mode = ll_opt['env']['theta_obs_mode']
        theta_space_mode = ll_opt['env']['theta_space_mode']
        assert(theta_obs_mode in ['ind', 'vector'])
        if theta_obs_mode == 'ind': 
            if theta_space_mode == 'forward':
                theta_sz = 1
            elif theta_space_mode == 'simple_four':
                theta_sz = 4
            elif theta_space_mode == 'simple_eight':
                theta_sz = 8
            elif theta_space_mode == 'k_theta':
                theta_sz = ll_opt['env']['num_theta']
            elif theta_obs_mode == 'vector':
                theta_sz = 2
            else:
                raise NotImplementedError            
        else:
            raise NotImplementedError
        ll_opt['model']['theta_sz'] = theta_sz
        ll_opt['env']['theta_sz'] = theta_sz

        # Load the low level policy params
        lowlevel_ckpt = options['lowlevel']['ckpt']            
        assert(os.path.isfile(lowlevel_ckpt))
        lowlevel_ckpt = torch.load(lowlevel_ckpt)
    hl_action_space = spaces.Discrete(theta_sz)

    # Check asserts
    assert(args.algo in ['a2c', 'ppo', 'acktr', 'dqn'])
    assert(optim_opt['hierarchical_mode'] in ['train_highlevel', 'train_both'])
    if model_opt['recurrent_policy']:
        assert args.algo in ['a2c', 'ppo'], 'Recurrent policy is not implemented for ACKTR'
    assert(model_opt['mode'] in ['hierarchical', 'hierarchical_many'])

    # Set seed - just make the seed the trial number
    seed = args.trial + 1000    # Make it different than lowlevel seed
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)

    # Initialization
    num_updates = int(optim_opt['num_frames']) // alg_opt['num_steps'] // alg_opt['num_processes'] // optim_opt['num_ll_steps']
    torch.set_num_threads(1)

    # Print warning
    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    # Set logging / load previous checkpoint
    logpath = os.path.join(log_opt['log_base'], model_opt['mode'], log_opt['exp_name'], args.algo, args.env_name, 'trial%d' % args.trial)
    if len(args.resume) > 0:
        assert(os.path.isfile(os.path.join(logpath, args.resume)))
        ckpt = torch.load(os.path.join(logpath, 'ckpt.pth.tar'))
        start_update = ckpt['update_count']
    else:
        # Make directory, check before overwriting
        if os.path.isdir(logpath):
            if click.confirm('Logs directory already exists in {}. Erase?'
                .format(logpath, default=False)):
                os.system('rm -rf ' + logpath)
            else:
                return
        os.system('mkdir -p ' + logpath)
        start_update = 0

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

    # Create environments
    envs = [make_env(args.env_name, seed, i, logpath, options, args.verbose) for i in range(alg_opt['num_processes'])]
    if alg_opt['num_processes'] > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    # Check if we use timestep in low level
    if 'baseline' in ll_opt['model']['mode']:
        add_timestep = False
    elif 'phase' in ll_opt['model']['mode']:
        add_timestep = True
    else:
        raise NotImplementedError

    # Get shapes
    dummy_env = make_env(args.env_name, seed, 0, logpath, options, args.verbose) 
    dummy_env = dummy_env()
    s_pro_dummy = dummy_env.unwrapped._get_pro_obs()
    s_ext_dummy = dummy_env.unwrapped._get_ext_obs()
    if add_timestep:
        ll_obs_shape = (s_pro_dummy.shape[0] + theta_sz + 1,)
        ll_raw_obs_shape =(s_pro_dummy.shape[0] + 1,)
    else:
        ll_obs_shape = (s_pro_dummy.shape[0] + theta_sz,)
        ll_raw_obs_shape = (s_pro_dummy.shape[0],)
    ll_obs_shape = (ll_obs_shape[0] * env_opt['num_stack'], *ll_obs_shape[1:])
    hl_obs_shape = (s_ext_dummy.shape[0],)
    hl_obs_shape = (hl_obs_shape[0] * env_opt['num_stack'], *hl_obs_shape[1:])
   
    # Do vec normalize, but mask out what we don't want altered
    # Also freeze all of the low level obs
    ignore_mask = dummy_env.env._get_obs_mask()
    freeze_mask, _ = dummy_env.unwrapped._get_pro_ext_mask()
    freeze_mask = np.concatenate([freeze_mask, [0]])
    if ('normalize' in env_opt and not env_opt['normalize']) or args.algo == 'dqn':
        ignore_mask = 1 - freeze_mask
    if model_opt['mode'] == 'hierarchical_many':
        # Actually ignore both ignored values and the low level values
        # That filtering will happen later
        ignore_mask = (ignore_mask + freeze_mask > 0).astype(float)
        envs = ObservationFilter(envs, ret=alg_opt['norm_ret'], has_timestep=True, noclip=env_opt['step_plus_noclip'], ignore_mask=ignore_mask, freeze_mask=freeze_mask, time_scale=env_opt['time_scale'], gamma=env_opt['gamma'])
    else:
        envs = ObservationFilter(envs, ret=alg_opt['norm_ret'], has_timestep=True, noclip=env_opt['step_plus_noclip'], ignore_mask=ignore_mask, freeze_mask=freeze_mask, time_scale=env_opt['time_scale'], gamma=env_opt['gamma'])

    # Make our helper object for dealing with hierarchical observations
    hier_utils = HierarchyUtils(ll_obs_shape, hl_obs_shape, hl_action_space, theta_sz, add_timestep)

    # Set up algo monitoring
    alg_filename = os.path.join(logpath, 'Alg.Monitor.csv')
    alg_f = open(alg_filename, "wt")
    alg_f.write('# Alg Logging %s\n'%json.dumps({"t_start": time.time(), 'env_id' : dummy_env.spec and dummy_env.spec.id, 'mode': options['model']['mode'], 'name': options['logs']['exp_name']}))
    alg_fields = ['value_loss', 'action_loss', 'dist_entropy']
    alg_logger = csv.DictWriter(alg_f, fieldnames=alg_fields)
    alg_logger.writeheader()
    alg_f.flush()
    ll_alg_filename = os.path.join(logpath, 'AlgLL.Monitor.csv')
    ll_alg_f = open(ll_alg_filename, "wt")
    ll_alg_f.write('# Alg Logging LL %s\n'%json.dumps({"t_start": time.time(), 'env_id' : dummy_env.spec and dummy_env.spec.id, 'mode': options['model']['mode'], 'name': options['logs']['exp_name']}))
    ll_alg_fields = ['value_loss', 'action_loss', 'dist_entropy']
    ll_alg_logger = csv.DictWriter(ll_alg_f, fieldnames=ll_alg_fields)
    ll_alg_logger.writeheader()
    ll_alg_f.flush()

    # Create the policy networks
    ll_action_space = envs.action_space 
    if args.algo == 'dqn':
        model_opt['eps_start'] = optim_opt['eps_start']
        model_opt['eps_end'] = optim_opt['eps_end']
        model_opt['eps_decay'] = optim_opt['eps_decay']
        hl_policy = DQNPolicy(hl_obs_shape, hl_action_space, model_opt)
    else:
        hl_policy = Policy(hl_obs_shape, hl_action_space, model_opt)
    if model_opt['mode'] == 'hierarchical_many':
        ll_policy = ModularPolicy(ll_raw_obs_shape, ll_action_space, theta_sz, ll_opt)
    else:
        ll_policy = Policy(ll_obs_shape, ll_action_space, ll_opt['model'])
    # Load the previous ones here?
    if args.cuda:
        hl_policy.cuda()
        ll_policy.cuda()

    # Create the high level agent
    if args.algo == 'a2c':
        hl_agent = algo.A2C_ACKTR(hl_policy, alg_opt['value_loss_coef'],
                               alg_opt['entropy_coef'], lr=optim_opt['lr'],
                               eps=optim_opt['eps'], alpha=optim_opt['alpha'],
                               max_grad_norm=optim_opt['max_grad_norm'])
    elif args.algo == 'ppo':
        hl_agent = algo.PPO(hl_policy, alg_opt['clip_param'], alg_opt['ppo_epoch'], alg_opt['num_mini_batch'],
                         alg_opt['value_loss_coef'], alg_opt['entropy_coef'], lr=optim_opt['lr'],
                         eps=optim_opt['eps'], max_grad_norm=optim_opt['max_grad_norm'])
    elif args.algo == 'acktr':
        hl_agent = algo.A2C_ACKTR(hl_policy, alg_opt['value_loss_coef'],
                               alg_opt['entropy_coef'], acktr=True)
    elif args.algo == 'dqn':
        hl_agent = algo.DQN(hl_policy, env_opt['gamma'], batch_size=alg_opt['batch_size'], target_update=alg_opt['target_update'], 
                        mem_capacity=alg_opt['mem_capacity'], lr=optim_opt['lr'], eps=optim_opt['eps'], max_grad_norm=optim_opt['max_grad_norm'])

    # Create the low level agent
    # If only training high level, make dummy agent (just does passthrough, doesn't change anything)
    if optim_opt['hierarchical_mode'] == 'train_highlevel':
        ll_agent = algo.Passthrough(ll_policy)
    elif optim_opt['hierarchical_mode'] == 'train_both':
        if args.algo == 'a2c':
            ll_agent = algo.A2C_ACKTR(ll_policy, alg_opt['value_loss_coef'],
                                      alg_opt['entropy_coef'], lr=optim_opt['ll_lr'],
                                      eps=optim_opt['eps'], alpha=optim_opt['alpha'],
                                      max_grad_norm=optim_opt['max_grad_norm'])
        elif args.algo == 'ppo':
            ll_agent = algo.PPO(ll_policy, alg_opt['clip_param'], alg_opt['ll_ppo_epoch'], alg_opt['num_mini_batch'],
                                alg_opt['value_loss_coef'], alg_opt['entropy_coef'], lr=optim_opt['ll_lr'],
                                eps=optim_opt['eps'], max_grad_norm=optim_opt['max_grad_norm'])
        elif args.algo == 'acktr':
            ll_agent = algo.A2C_ACKTR(ll_policy, alg_opt['value_loss_coef'],
                                      alg_opt['entropy_coef'], acktr=True)
    else:
        raise NotImplementedError

    # Make the rollout structures 
    hl_rollouts = RolloutStorage(alg_opt['num_steps'], alg_opt['num_processes'], hl_obs_shape, hl_action_space, hl_policy.state_size)
    ll_rollouts = MaskingRolloutStorage(alg_opt['num_steps'], alg_opt['num_processes'], ll_obs_shape, ll_action_space, ll_policy.state_size)
    hl_current_obs = torch.zeros(alg_opt['num_processes'], *hl_obs_shape)
    ll_current_obs = torch.zeros(alg_opt['num_processes'], *ll_obs_shape)

    # Helper functions to update the current obs
    def update_hl_current_obs(obs):
        shape_dim0 = hl_obs_shape[0]
        obs = torch.from_numpy(obs).float()
        if env_opt['num_stack'] > 1:
            hl_current_obs[:, :-shape_dim0] = hl_current_obs[:, shape_dim0:]
        hl_current_obs[:, -shape_dim0:] = obs
    def update_ll_current_obs(obs):
        shape_dim0 = ll_obs_shape[0]
        obs = torch.from_numpy(obs).float()
        if env_opt['num_stack'] > 1:
            ll_current_obs[:, :-shape_dim0] = ll_current_obs[:, shape_dim0:]
        ll_current_obs[:, -shape_dim0:] = obs

    # Update agent with loaded checkpoint
    if len(args.resume) > 0:
        # This should update both the policy network and the optimizer
        ll_agent.load_state_dict(ckpt['ll_agent'])
        hl_agent.load_state_dict(ckpt['hl_agent'])

        # Set ob_rms
        envs.ob_rms = ckpt['ob_rms']
    else:
        if model_opt['mode'] == 'hierarchical_many':
            ll_agent.load_pretrained_policies(lowlevel_ckpts)
        else:
            # Load low level agent 
            ll_agent.load_state_dict(lowlevel_ckpt['agent'])

            # Load ob_rms from low level (but need to reshape it)
            old_rms = lowlevel_ckpt['ob_rms']
            assert(old_rms.mean.shape[0] == ll_obs_shape[0])
            # Only copy the pro state part of it (not including thetas or count)
            envs.ob_rms.mean[:s_pro_dummy.shape[0]] = old_rms.mean[:s_pro_dummy.shape[0]] 
            envs.ob_rms.var[:s_pro_dummy.shape[0]] = old_rms.var[:s_pro_dummy.shape[0]] 

    # Reset our env and rollouts
    raw_obs = envs.reset()
    hl_obs, raw_ll_obs, step_counts = hier_utils.seperate_obs(raw_obs)
    ll_obs = hier_utils.placeholder_theta(raw_ll_obs, step_counts)
    update_hl_current_obs(hl_obs)
    update_ll_current_obs(ll_obs)
    hl_rollouts.observations[0].copy_(hl_current_obs)
    ll_rollouts.observations[0].copy_(ll_current_obs)
    ll_rollouts.recent_obs.copy_(ll_current_obs)
    if args.cuda:
        hl_current_obs = hl_current_obs.cuda()
        ll_current_obs = ll_current_obs.cuda()
        hl_rollouts.cuda()
        ll_rollouts.cuda()

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([alg_opt['num_processes'], 1])
    final_rewards = torch.zeros([alg_opt['num_processes'], 1])

    # Update loop
    start = time.time()
    for j in range(start_update, num_updates):
        for step in range(alg_opt['num_steps']):
            # Step through high level action
            start_time = time.time()
            with torch.no_grad():
                hl_value, hl_action, hl_action_log_prob, hl_states = hl_policy.act(hl_rollouts.observations[step], hl_rollouts.states[step], hl_rollouts.masks[step])
            hl_cpu_actions = hl_action.squeeze(1).cpu().numpy()
            if args.profile:
                print('hl act %f' % (time.time() - start_time))

            # Get values to use for Q learning
            hl_state_dqn = hl_rollouts.observations[step]
            hl_action_dqn = hl_action

            # Update last ll observation with new theta
            for proc in range(alg_opt['num_processes']):
                # Update last observations in memory
                last_obs = ll_rollouts.observations[ll_rollouts.steps[proc], proc]
                if hier_utils.has_placeholder(last_obs):         
                    new_last_obs = hier_utils.update_theta(last_obs, hl_cpu_actions[proc])
                    ll_rollouts.observations[ll_rollouts.steps[proc], proc].copy_(new_last_obs)

                # Update most recent observations (not necessarily the same)
                assert(hier_utils.has_placeholder(ll_rollouts.recent_obs[proc]))
                new_last_obs = hier_utils.update_theta(ll_rollouts.recent_obs[proc], hl_cpu_actions[proc])
                ll_rollouts.recent_obs[proc].copy_(new_last_obs)
            assert(ll_rollouts.observations.max().item() < float('inf') and ll_rollouts.recent_obs.max().item() < float('inf'))

            # Given high level action, step through the low level actions
            death_step_mask = np.ones([alg_opt['num_processes'], 1])    # 1 means still alive, 0 means dead  
            hl_reward = torch.zeros([alg_opt['num_processes'], 1])
            hl_obs = [None for i in range(alg_opt['num_processes'])]
            for ll_step in range(optim_opt['num_ll_steps']):
                # Sample actions
                start_time = time.time()
                with torch.no_grad():
                    ll_value, ll_action, ll_action_log_prob, ll_states = ll_policy.act(ll_rollouts.recent_obs, ll_rollouts.recent_s, ll_rollouts.recent_masks, deterministic=ll_deterministic)
                ll_cpu_actions = ll_action.squeeze(1).cpu().numpy()
                if args.profile:
                    print('ll act %f' % (time.time() - start_time))

                # Observe reward and next obs
                raw_obs, ll_reward, done, info = envs.step(ll_cpu_actions, death_step_mask)
                raw_hl_obs, raw_ll_obs, step_counts = hier_utils.seperate_obs(raw_obs)
                ll_obs = []
                for proc in range(alg_opt['num_processes']):
                    if (ll_step == optim_opt['num_ll_steps'] - 1) or done[proc]:
                        ll_obs.append(hier_utils.placeholder_theta(np.array([raw_ll_obs[proc]]), np.array([step_counts[proc]])))
                    else:
                        ll_obs.append(hier_utils.append_theta(np.array([raw_ll_obs[proc]]), np.array([hl_cpu_actions[proc]]), np.array([step_counts[proc]])))
                ll_obs = np.concatenate(ll_obs, 0) 
                ll_reward = torch.from_numpy(np.expand_dims(np.stack(ll_reward), 1)).float()
                episode_rewards += ll_reward
                hl_reward += ll_reward

                # Update values for Q learning and update replay memory
                time.time()
                hl_next_state_dqn = torch.from_numpy(raw_hl_obs)
                hl_reward_dqn = ll_reward
                hl_isdone_dqn = done
                if args.algo == 'dqn':
                    hl_agent.update_memory(hl_state_dqn, hl_action_dqn, hl_next_state_dqn, hl_reward_dqn, hl_isdone_dqn, death_step_mask)    
                hl_state_dqn = hl_next_state_dqn
                if args.profile:
                    print('dqn memory %f' % (time.time() - start_time))

                # Update high level observations (only take most recent obs if we haven't see a done before now and thus the value is valid)
                for proc, raw_hl in enumerate(raw_hl_obs):
                    if death_step_mask[proc].item() > 0:
                        hl_obs[proc] = np.array([raw_hl])

                # If done then clean the history of observations
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards  # TODO - actually not sure if I broke this logic, but this value is not used anywhere
                episode_rewards *= masks
                
                # TODO - I commented this out, which possibly breaks things if num_stack > 1. Fix later if necessary
                #if args.cuda:
                #    masks = masks.cuda()
                #if current_obs.dim() == 4:
                #    current_obs *= masks.unsqueeze(2).unsqueeze(2)
                #else:
                #    current_obs *= masks

                # Update low level observations
                update_ll_current_obs(ll_obs)

                # Update low level rollouts
                ll_rollouts.insert(ll_current_obs, ll_states, ll_action, ll_action_log_prob, ll_value, ll_reward, masks, death_step_mask) 

                # Update which ones have stepped to the end and shouldn't be updated next time in the loop
                death_step_mask *= masks    

            # Update high level rollouts
            hl_obs = np.concatenate(hl_obs, 0) 
            update_hl_current_obs(hl_obs) 
            hl_rollouts.insert(hl_current_obs, hl_states, hl_action, hl_action_log_prob, hl_value, hl_reward, masks)

            # Check if we want to update lowlevel policy
            if ll_rollouts.isfull and all([not hier_utils.has_placeholder(ll_rollouts.observations[ll_rollouts.steps[proc], proc]) for proc in range(alg_opt['num_processes'])]): 
                # Update low level policy
                assert(ll_rollouts.observations.max().item() < float('inf')) 
                if optim_opt['hierarchical_mode'] == 'train_both':
                    with torch.no_grad():
                        ll_next_value = ll_policy.get_value(ll_rollouts.observations[-1], ll_rollouts.states[-1], ll_rollouts.masks[-1]).detach()
                    ll_rollouts.compute_returns(ll_next_value, alg_opt['use_gae'], env_opt['gamma'], alg_opt['gae_tau'])
                    ll_value_loss, ll_action_loss, ll_dist_entropy = ll_agent.update(ll_rollouts)
                else:
                    ll_value_loss = 0
                    ll_action_loss = 0
                    ll_dist_entropy = 0
                ll_rollouts.after_update()

                # Update logger
                alg_info = {}
                alg_info['value_loss'] = ll_value_loss
                alg_info['action_loss'] = ll_action_loss
                alg_info['dist_entropy'] = ll_dist_entropy
                ll_alg_logger.writerow(alg_info)
                ll_alg_f.flush() 

        # Update high level policy
        start_time = time.time()
        assert(hl_rollouts.observations.max().item() < float('inf'))
        if args.algo == 'dqn':
            hl_value_loss, hl_action_loss, hl_dist_entropy = hl_agent.update(alg_opt['updates_per_step']) # TODO - maybe log this loss properly   
        else:
            with torch.no_grad():
                hl_next_value = hl_policy.get_value(hl_rollouts.observations[-1], hl_rollouts.states[-1], hl_rollouts.masks[-1]).detach()
            hl_rollouts.compute_returns(hl_next_value, alg_opt['use_gae'], env_opt['gamma'], alg_opt['gae_tau']) 
            hl_value_loss, hl_action_loss, hl_dist_entropy = hl_agent.update(hl_rollouts)
        hl_rollouts.after_update()
        if args.profile:
            print('hl update %f' % (time.time() - start_time))

        # Update alg monitor for high level
        alg_info = {}
        alg_info['value_loss'] = hl_value_loss
        alg_info['action_loss'] = hl_action_loss
        alg_info['dist_entropy'] = hl_dist_entropy
        alg_logger.writerow(alg_info)
        alg_f.flush() 

        # Save checkpoints 
        total_num_steps = (j + 1) * alg_opt['num_processes'] * alg_opt['num_steps'] * optim_opt['num_ll_steps']
        if 'save_interval' in alg_opt:
            save_interval = alg_opt['save_interval']
        else:
            save_interval = 100
        if j % save_interval == 0:
            # Save all of our important information
            start_time = time.time()
            save_checkpoint(logpath, ll_agent, hl_agent, envs, j, total_num_steps)
            if args.profile:
                print('save checkpoint %f' % (time.time() - start_time))

        # Print log
        log_interval = log_opt['log_interval'] * alg_opt['log_mult']
        if j % log_interval == 0:
            end = time.time()
            print("{}: Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(options['logs']['exp_name'], j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), hl_dist_entropy,
                       hl_value_loss, hl_action_loss))

        # Do dashboard logging
        vis_interval = log_opt['vis_interval'] * alg_opt['log_mult']
        if args.vis and j % vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                dashboard.visdom_plot()
            except IOError:
                pass

    # Save final checkpoint
    save_checkpoint(logpath, ll_agent, hl_agent, envs, j, total_num_steps)

    # Close logging file
    alg_f.close()
    ll_alg_f.close()

# Save a checkpoint
def save_checkpoint(logpath, ll_agent, hl_agent, envs, j, total_num_steps):
    # Get the checkpoint info
    ckpt = {}
    ckpt['update_count'] = j
    ckpt['total_num_steps'] = total_num_steps
    ckpt['ob_rms'] = envs.ob_rms
    ckpt['ll_agent'] = ll_agent.state_dict()   
    ckpt['hl_agent'] = hl_agent.state_dict()

    # Finally save the checkpoint
    ckpt_path = os.path.join(logpath, 'ckpt.pth.tar')
    torch.save(ckpt, ckpt_path)
    return

if __name__ == "__main__":
    main()
