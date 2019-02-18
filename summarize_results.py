# This file takes the completed slurm logs and summarizes the results
# Right now, just displays the average +- var of the final rewards (or whatever value we set in the ymal)
# Eventually should be able to plot variance?
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
from visualize import Dashboard

# Get Input Arguments
parser = argparse.ArgumentParser(description='RL')

##################################################
# yaml options file contains all default choices #
parser.add_argument('--batch_path_opt', default='options/batch/default.yaml', type=str, 
                    help='path to a yaml options file')
# yaml option file containing the visdom plotting options
parser.add_argument('--vis_path_opt', default='options/visualization/reward.yaml', type=str,
                    help='path to a yaml visualization options file')
##################################################
parser.add_argument('--eval-key', type=str, default='reward_env',
                    help='name of key in the Episode log that we actually want to evaluate on')
parser.add_argument('--outfile', type=str, default='',
                    help='where to dump these results (optional)')
parser.add_argument('--bin-size', type=int, default=100,
                    help='over how many episode to average final result value')
# These options only matter if we're batch_path_opt is actually just a single yaml (not a batch)
parser.add_argument('--num-trials', type=int, default=5)
parser.add_argument('--trial-offset', type=int, default=0)
parser.add_argument('--algo', default='a2c',
                    help='algorithm to use: a2c | ppo | acktr')
parser.add_argument('--env-name', default='Hopper-v2',
                    help='environment to train on (default: Hopper-v2)')
def main():
    global args
    args = parser.parse_args()

    # Set options
    if args.batch_path_opt is not None:
        with open(args.batch_path_opt, 'r') as handle:
            batch_options = yaml.load(handle)
    if args.vis_path_opt is not None:
        with open(args.vis_path_opt, 'r') as handle:
            vis_options = yaml.load(handle)
    print('## args'); pprint(vars(args))

    # Either use the slurm batch file or the single yaml file to get the values
    val_dict = {}
    if 'base_yaml' in batch_options:
        # Slurm version
        algo = batch_options['algo']
        env_name = batch_options['env_name']
        num_trials = batch_options['num_trials']
        trial_offset = batch_options['trial_offset']
        base_yaml_file = batch_options['base_yaml']

        # Get the list of yaml files
        # Copies logic from clusterrun to make these
        grid = batch_options['params']
        individual_options = [[{key: value} for value in values] for key, values in grid.items()]
        product_options = list(itertools.product(*individual_options))
        jobs = [{k: v for d in option_set for k, v in d.items()} for option_set in product_options]
        basenames = []
        yaml_files = []
        with open(base_yaml_file) as f:
            base_options = yaml.load(f)
        for job in jobs:
            new_unique_name = base_options['logs']['exp_name']
            for k, v in job.items():
                new_unique_name += "_" + str(k) + "_" + str(v)
            assert(len(base_yaml_file.split('.')) == 2)
            new_yaml_filename = base_yaml_file.split('.')[0]
            new_yaml_filename = os.path.join(new_yaml_filename, new_unique_name) + '.yaml'
            basenames.append(new_unique_name)
            yaml_files.append(new_yaml_filename)
        assert(len(yaml_files) == len(jobs))
        assert(len(basenames) == len(jobs))

        # Get the eval vals for each param set
        val_dict = {}
        for yaml_file, name in zip(yaml_files, basenames):
            with open(yaml_file, 'r') as handle:
                opt = yaml.load(handle)
            eval_vals = get_last_eval_vals(opt, vis_options, args.eval_key, algo, env_name, num_trials, trial_offset, args.bin_size) 
            if eval_vals is not None:
                val_dict[name] = eval_vals
    else:
        # Single yaml version
        algo = args.algo
        env_name = args.env_name
        opt = batch_options
        num_trials = args.num_trials
        trial_offset = args.trial_offset
    
        # Get the eval vals for this yaml
        eval_vals = get_last_eval_vals(opt, vis_options, args.eval_key, algo, env_name, num_trials, trial_offset, args.bin_size)

        # Save to dict
        name = opt['logs']['exp_name']
        val_dict[name] = eval_vals

    # Get the average values and std for each value in dict
    # Sort by average
    # Display / print each by decreasing average value
    avg_dict = {k: np.mean(v) for k, v in val_dict.items()}
    sorted_avg_dict = sorted(avg_dict.items(), reverse=True, key=lambda x: x[1])
    sorted_names = [x[0] for x in sorted_avg_dict]
    lines = []
    lines.append("Results for run of {yaml_name} on variable {var}".format(yaml_name=args.batch_path_opt.split('/')[-1], var=args.eval_key))
    for name in sorted_names:
        lines.append("{name}: {avg}+={std}".format(name=name, avg=np.mean(val_dict[name]), std=np.std(val_dict[name])))
    
    # Print results
    for line in lines:
        print(line)

    # Optionally print to file
    if len(args.outfile) > 0:
        with open(args.outfile, 'w') as f:
            for line in lines:
                f.write(line + '\n')

# Get the last bucket values for the eval_key for each trial and return
def get_last_eval_vals(opt, vis_opt, eval_key, algo, env_name, num_trials, trial_offset, bin_size):
    # For each trial
    eval_vals = []
    for trial in range(trial_offset, trial_offset+num_trials):
        # Get the logpath
        logpath = os.path.join(opt['logs']['log_base'], opt['model']['mode'], opt['logs']['exp_name'], algo, env_name, 'trial%d' % trial)
        if not os.path.isdir(logpath):
            return None


        # Create the dashboard object
        opt['env']['env-name'] = env_name
        opt['alg'] = opt['alg_%s' % algo]
        opt['optim'] = opt['optim_%s' % algo]
        opt['alg']['algo'] = algo
        opt['trial'] = trial
        dash = Dashboard(opt, vis_opt, logpath, vis=False)

        # Get data
        try:
            dash.preload_data()
            raw_x, raw_y = dash.load_data('episode_monitor', 'scalar', eval_key)
        except Exception:
            return None

        # Get data from last bin
        if not (len(raw_y) > bin_size):
            return None
        raw_vals = raw_y[-bin_size:]
        assert(len(raw_vals) == bin_size)
        raw_vals = [float(v) for v in raw_vals]
        raw_val = np.mean(raw_vals)
        eval_vals.append(raw_val)

    # Return
    return eval_vals

if __name__ == "__main__":
    main()

