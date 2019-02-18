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
import random

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
# yaml option file containing the visdom plotting options
parser.add_argument('--vis_path_opt', default='options/visualization/default.yaml', type=str, 
                    help='path to a yaml visualization options file')
##################################################
parser.add_argument('--algo', default='a2c',
                    help='algorithm to use: a2c | ppo | acktr') 
parser.add_argument('--env-name', default='Hopper-v2',
                    help='environment to train on (default: Hopper-v2)')
parser.add_argument('--trial', type=int, default=0,
                    help='keep track of what trial you are on')
parser.add_argument('--save-every', default=1000000000, type=int,
                    help='how often to save our models permanently')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint')
parser.add_argument('--other-resume', default='', type=str)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-vis', action='store_true', default=False,
                    help='disables visdom visualization')
parser.add_argument('--finetune-baseline', action='store_true', default=False,
                    help='resumes from a previously trained model')
parser.add_argument('--port', type=int, default=8097,
                    help='port to run the server on (default: 8097)')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='if true, print step logs')

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
    model_opt['time_scale'] = env_opt['time_scale']
    if model_opt['mode'] in ['baselinewtheta', 'phasewtheta']:
        model_opt['theta_space_mode'] = env_opt['theta_space_mode']
        model_opt['theta_sz'] = env_opt['theta_sz']
    elif model_opt['mode'] in ['baseline_lowlevel', 'phase_lowlevel']:
        model_opt['theta_space_mode'] = env_opt['theta_space_mode']

    # Check asserts
    assert(model_opt['mode'] in ['baseline', 'baseline_reverse', 'phasesimple', 'phasewstate', 'baselinewtheta', 'phasewtheta', 'baseline_lowlevel', 'phase_lowlevel', 'interpolate', 'cyclic', 'maze_baseline', 'maze_baseline_wphase'])
    assert(args.algo in ['a2c', 'ppo', 'acktr'])
    if model_opt['recurrent_policy']:
        assert args.algo in ['a2c', 'ppo'], 'Recurrent policy is not implemented for ACKTR'

    # Set seed - just make the seed the trial number
    seed = args.trial
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)

    # Initialization
    num_updates = int(optim_opt['num_frames']) // alg_opt['num_steps'] // alg_opt['num_processes']
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

    # If interpolate mode, choose states
    if options['model']['mode'] == 'phase_lowlevel' and options['env']['theta_space_mode'] == 'pretrain_interp':
        all_states = torch.load(env_opt['saved_state_file'])
        s1 = random.choice(all_states)
        s2 = random.choice(all_states)
        fixed_states = [s1, s2]
    elif model_opt['mode'] == 'interpolate':
        all_states = torch.load(env_opt['saved_state_file'])
        s1 = all_states[env_opt['s1_ind']]
        s2 = all_states[env_opt['s2_ind']]
        fixed_states = [s1, s2]
    else:
        fixed_states = None

    # Create environments
    dummy_env = make_env(args.env_name, seed, 0, logpath, options, args.verbose)
    dummy_env = dummy_env()
    envs = [make_env(args.env_name, seed, i, logpath, options, args.verbose, fixed_states) for i in range(alg_opt['num_processes'])]
    if alg_opt['num_processes'] > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    # Get theta_sz for models (if applicable)
    dummy_env.reset()    
    if model_opt['mode'] == 'baseline_lowlevel':
        model_opt['theta_sz'] = dummy_env.env.theta_sz
    elif model_opt['mode'] == 'phase_lowlevel':
        model_opt['theta_sz'] = dummy_env.env.env.theta_sz   
    if 'theta_sz' in model_opt:
        env_opt['theta_sz'] = model_opt['theta_sz']

    # Get observation shape
    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * env_opt['num_stack'], *obs_shape[1:])

    # Do vec normalize, but mask out what we don't want altered
    if len(envs.observation_space.shape) == 1:
        ignore_mask = np.zeros(envs.observation_space.shape)
        if env_opt['add_timestep']:
            ignore_mask[-1] = 1
        if model_opt['mode'] in ['baselinewtheta', 'phasewtheta', 'baseline_lowlevel', 'phase_lowlevel']:
            theta_sz = env_opt['theta_sz']
            if env_opt['add_timestep']:
                ignore_mask[-(theta_sz+1):] = 1
            else:
                ignore_mask[-theta_sz:] = 1
        if args.finetune_baseline:
            ignore_mask = dummy_env.unwrapped._get_obs_mask()
            freeze_mask, _ = dummy_env.unwrapped._get_pro_ext_mask()
            if env_opt['add_timestep']:
                ignore_mask = np.concatenate([ignore_mask, [1]])
                freeze_mask = np.concatenate([freeze_mask, [0]])
            ignore_mask = (ignore_mask + freeze_mask > 0).astype(float)
            envs = ObservationFilter(envs, ret=alg_opt['norm_ret'], has_timestep=True, noclip=env_opt['step_plus_noclip'], ignore_mask=ignore_mask, freeze_mask=freeze_mask, time_scale=env_opt['time_scale'], gamma=env_opt['gamma'])
        else:
            envs = ObservationFilter(envs, ret=alg_opt['norm_ret'], has_timestep=env_opt['add_timestep'], noclip=env_opt['step_plus_noclip'], ignore_mask=ignore_mask, time_scale=env_opt['time_scale'], gamma=env_opt['gamma'])

    # Set up algo monitoring
    alg_filename = os.path.join(logpath, 'Alg.Monitor.csv')
    alg_f = open(alg_filename, "wt")
    alg_f.write('# Alg Logging %s\n'%json.dumps({"t_start": time.time(), 'env_id' : dummy_env.spec and dummy_env.spec.id, 'mode': options['model']['mode'], 'name': options['logs']['exp_name']}))
    alg_fields = ['value_loss', 'action_loss', 'dist_entropy']
    alg_logger = csv.DictWriter(alg_f, fieldnames=alg_fields)
    alg_logger.writeheader()
    alg_f.flush()

    # Create the policy network
    actor_critic = Policy(obs_shape, envs.action_space, model_opt)
    if args.cuda:
        actor_critic.cuda()

    # Create the agent 
    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, alg_opt['value_loss_coef'],
                               alg_opt['entropy_coef'], lr=optim_opt['lr'],
                               eps=optim_opt['eps'], alpha=optim_opt['alpha'],
                               max_grad_norm=optim_opt['max_grad_norm'])
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, alg_opt['clip_param'], alg_opt['ppo_epoch'], alg_opt['num_mini_batch'],
                         alg_opt['value_loss_coef'], alg_opt['entropy_coef'], lr=optim_opt['lr'],
                         eps=optim_opt['eps'], max_grad_norm=optim_opt['max_grad_norm'])
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, alg_opt['value_loss_coef'],
                               alg_opt['entropy_coef'], acktr=True)
    rollouts = RolloutStorage(alg_opt['num_steps'], alg_opt['num_processes'], obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(alg_opt['num_processes'], *obs_shape)

    # Update agent with loaded checkpoint
    if len(args.resume) > 0:
        # This should update both the policy network and the optimizer
        agent.load_state_dict(ckpt['agent'])

        # Set ob_rms
        envs.ob_rms = ckpt['ob_rms']
    elif len(args.other_resume) > 0:
        ckpt = torch.load(args.other_resume)

        # This should update both the policy network
        agent.actor_critic.load_state_dict(ckpt['agent']['model'])

        # Set ob_rms
        envs.ob_rms = ckpt['ob_rms']
    elif args.finetune_baseline: 
        # Load the model based on the trial number
        ckpt_base = options['lowlevel']['ckpt'] 
        ckpt_file = ckpt_base + '/trial%d/ckpt.pth.tar' % args.trial
        ckpt = torch.load(ckpt_file)

        # Make "input mask" that tells the model which inputs were the same from before and should be copied
        oldinput_mask, _ = dummy_env.unwrapped._get_pro_ext_mask()

        # This should update both the policy network
        agent.actor_critic.load_state_dict_special(ckpt['agent']['model'], oldinput_mask)

        # Set ob_rms
        old_rms = ckpt['ob_rms']
        old_size = old_rms.mean.size
        if env_opt['add_timestep']:
            old_size -= 1

        # Only copy the pro state part of it 
        envs.ob_rms.mean[:old_size] = old_rms.mean[:old_size]
        envs.ob_rms.var[:old_size] = old_rms.var[:old_size]

    # Inline define our helper function for updating obs
    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if env_opt['num_stack'] > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    # Reset our env and rollouts
    obs = envs.reset()
    update_current_obs(obs)
    rollouts.observations[0].copy_(current_obs)
    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([alg_opt['num_processes'], 1])
    final_rewards = torch.zeros([alg_opt['num_processes'], 1])

    # Update loop
    start = time.time()
    for j in range(start_update, num_updates):
        for step in range(alg_opt['num_steps']):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, states = actor_critic.act(
                        rollouts.observations[step],
                        rollouts.states[step],
                        rollouts.masks[step])
            cpu_actions = action.squeeze(1).cpu().numpy()

            # Observe reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)
            #pdb.set_trace()
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs)
            rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks)

        # Update model and rollouts
        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.observations[-1],
                                                rollouts.states[-1],
                                                rollouts.masks[-1]).detach()
        rollouts.compute_returns(next_value, alg_opt['use_gae'], env_opt['gamma'], alg_opt['gae_tau'])
        value_loss, action_loss, dist_entropy = agent.update(rollouts)       
        rollouts.after_update()

        # Add algo updates here
        alg_info = {}
        alg_info['value_loss'] = value_loss
        alg_info['action_loss'] = action_loss
        alg_info['dist_entropy'] = dist_entropy
        alg_logger.writerow(alg_info)
        alg_f.flush() 

        # Save checkpoints 
        total_num_steps = (j + 1) * alg_opt['num_processes'] * alg_opt['num_steps']
        #save_interval = log_opt['save_interval'] * alg_opt['log_mult']
        save_interval = 100
        if j % save_interval == 0:
            # Save all of our important information
            save_checkpoint(logpath, agent, envs, j, total_num_steps, args.save_every, final=False)

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
                       final_rewards.max(), dist_entropy,
                       value_loss, action_loss))

        # Do dashboard logging
        vis_interval = log_opt['vis_interval'] * alg_opt['log_mult']
        if args.vis and j % vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                dashboard.visdom_plot()
            except IOError:
                pass

    # Save final checkpoint
    save_checkpoint(logpath, agent, envs, j, total_num_steps, args.save_every, final=False)

    # Close logging file
    alg_f.close()

def save_checkpoint(logpath, agent, envs, j, total_num_steps, save_every, final=False):
    # Get the checkpoint info
    ckpt = {}
    ckpt['update_count'] = j
    ckpt['total_num_steps'] = total_num_steps
    ckpt['ob_rms'] = envs.ob_rms
    ckpt['agent'] = agent.state_dict()   

    # Finally save the checkpoint
    ckpt_path = os.path.join(logpath, 'ckpt.pth.tar')
    torch.save(ckpt, ckpt_path)

    # Copy checkpoint if in save_every or final
    if final:
        final_path = os.path.join(logpath, 'final_ckpt.pth.tar')
        shutil.copyfile(ckpt_path, final_path)
    if j % save_every == 0:
        int_path = os.path.join(logpath, 'ckpt_%d.pth.tar' % j)
        shutil.copyfile(ckpt_path, int_path)
    return

if __name__ == "__main__":
    main()
