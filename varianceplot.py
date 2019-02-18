# File takes in one or more yaml files and creates the variance plot
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
import itertools
import pdb
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.signal import medfilt
matplotlib.rcParams.update({'font.size': 8})
from visdom import Visdom
from visualize import Dashboard

# Get Input Arguments
parser = argparse.ArgumentParser(description='RL')

##################################################
# yaml options file contains all default choices #
parser.add_argument('--path_opt', default=[], type=str, dest='path_opts', action='append',
                    help='list yaml options files we want to get variance plots for')
# yaml option file containing the visdom plotting options
parser.add_argument('--vis_path_opt', default='options/visualization/reward.yaml', type=str,
                    help='path to a yaml visualization options file')
##################################################
parser.add_argument('--eval-key', type=str, default='reward_env',
                    help='name of key in the Episode log that we actually want to evaluate on')
parser.add_argument('--no-vis', action='store_true', default=False,
                    help='disables visdom visualization')
parser.add_argument('--port', type=int, default=8097,
                    help='port to run the server on (default: 8097)')
parser.add_argument('--bin-size', type=int, default=100,
                    help='over how many episode to average final result value')
parser.add_argument('--smooth', type=int, default=1)
parser.add_argument('--num-trials', type=int, default=5)
parser.add_argument('--trial-offset', type=int, default=0)
parser.add_argument('--algo', default='a2c',
                    help='algorithm to use: a2c | ppo | acktr')
parser.add_argument('--env-name', default='Hopper-v2',
                    help='environment to train on (default: Hopper-v2)')
parser.add_argument('--legend', default=[], type=str, dest='legend_names', action='append',
                    help='name of each line in the variance plot (same order as path_opt)')
parser.add_argument('--mode', default='std', type=str,
                    help='variance plot mode. (minmax | variance | std | all)')
def main():
    global args
    args = parser.parse_args()

    # Get options
    opt_list = []
    for filename in args.path_opts:
        with open(filename, 'r') as f:
            opt = yaml.load(f)
        opt_list.append(opt)   
    with open(args.vis_path_opt, 'r') as handle:
        vis_options = yaml.load(handle)
    print('## args'); pprint(vars(args))

    # Get the relevant curves
    algo = args.algo
    env_name = args.env_name
    num_trials = args.num_trials
    trial_offset = args.trial_offset
    x_curves = []
    mean_curves = []
    bottom_curves = []
    top_curves = []
    for opt in opt_list:
        # Get the curves for all the trials for each yaml file (mean and variance)
        x_curve, mean_curve, top_curve, bottom_curve = get_eval_vals(opt, vis_options, args.eval_key, algo, env_name, num_trials, trial_offset, args.bin_size, args.smooth, args.mode)
        x_curves.append(x_curve)
        mean_curves.append(mean_curve)
        bottom_curves.append(bottom_curve)
        top_curves.append(top_curve) 
    assert(len(x_curves) == len(args.legend_names))  
    assert(len(mean_curves) == len(args.legend_names))
    assert(len(bottom_curves) == len(args.legend_names))
    assert(len(top_curves) == len(args.legend_names))

    # Plot the curves
    if args.no_vis:
        raise Exception("No, we need visdom to display")
    vis = Visdom(port=args.port)
    plt.figure()
    for leg_name, x, mean, bot, top in zip(args.legend_names, x_curves, mean_curves, bottom_curves, top_curves):
        # Subsample (because why not?)
        x = x[::100]
        mean = mean[::100]
        p = plt.plot(x, mean, label=leg_name)
        if args.mode == 'all':
            for curve in top:
                curve = curve[::100]
                plt.plot(x, curve)
        else:
            bot = bot[::100]
            top = top[::100]
            plt.fill_between(x, bot, top, alpha=0.2)
    plt.legend()
    plt.xlabel('Number of Timesteps')
    plt.ylabel(args.eval_key)
    plt.title(args.eval_key)
    plt.grid(True)
    vis.matplot(plt)
    plt.close()

# Get the last bucket values for the eval_key for each trial and return
def get_eval_vals(opt, vis_opt, eval_key, algo, env_name, num_trials, trial_offset, bin_size, smooth, mode='minmax'):
    # For each trial
    x_curves = []
    y_curves = []
    for trial in range(trial_offset, trial_offset+num_trials):
        # Get the logpath
        logpath = os.path.join(opt['logs']['log_base'], opt['model']['mode'], opt['logs']['exp_name'], algo, env_name, 'trial%d' % trial)
        print(logpath)
        assert(os.path.isdir(logpath))

        # Create the dashboard object
        opt['env']['env-name'] = env_name
        opt['alg'] = opt['alg_%s' % algo]
        opt['optim'] = opt['optim_%s' % algo]
        opt['alg']['algo'] = algo
        opt['trial'] = trial
        dash = Dashboard(opt, vis_opt, logpath, vis=True)

        # Get data
        dash.preload_data()
        x, y = dash.load_data('episode_monitor', 'scalar', eval_key)
        x = [float(i) for i in x]
        y = [float(i.replace('\x00','')) for i in y]

        # Smooth and bin
        if smooth == 1:
            x, y = dash.smooth_curve(x, y)
        elif smooth == 2:
            y = medfilt(y, kernel_size=9)
        x, y = dash.fix_point(x, y, bin_size)

        # Append
        x_curves.append(x)
        y_curves.append(y)

    # Interpolate the curves
    # Get the combined list of all x values
    union = set([])
    for x_curve in x_curves:
        union = union | set(x_curve)
    all_x = sorted(list(union))

    # Get interpolated y values of each list
    interp_y_curves = []
    for x_curve, y_curve in zip(x_curves, y_curves):
        interp_y = np.interp(all_x, x_curve, y_curve)
        interp_y_curves.append(interp_y)

    # Get mean and variance curves
    mean = np.mean(interp_y_curves, axis=0)
    y_middle = mean
    if mode == 'all':
        y_top = interp_y_curves
        y_bottom = None
    elif mode == 'minmax':
        y_bottom = np.min(interp_y_curves, axis=0)
        y_top = np.max(interp_y_curves, axis=0)
    elif mode == 'variance':
        var = np.var(interp_y_curves, axis=0)
        y_bottom = mean - var
        y_top = mean + var
    elif mode == 'std':    
        std = np.std(interp_y_curves, axis=0)
        y_bottom = mean - std
        y_top = mean + std

    # Return
    return np.array(all_x), y_middle, y_top, y_bottom

if __name__ == "__main__":
    main()

