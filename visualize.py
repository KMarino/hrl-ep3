# Edited by Kenneth Marino
# Copied from https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/visualize_atari.py
# and https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/load.py
# Thanks to the author and OpenAI team!

import glob
import json
import os
import pdb
import time
import argparse
import yaml
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from scipy.signal import medfilt
matplotlib.rcParams.update({'font.size': 8})
from visdom import Visdom
from environments import geom_utils

# Color defaults
color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

# Object that keeps track of plotting
class Dashboard(object):
    def __init__(self, opt, vis_opt, logpath, vis=True, port=8097, mode='realtime'):
        # Get relevant options
        exp_name = opt['logs']['exp_name']
        if 'env' in opt:
            self.game = opt['env']['env-name']
            self.alg_name = opt['alg']['algo']
            self.num_steps = opt['optim']['num_frames']
            trial = opt['trial']
            self.unique_name = exp_name + ' ' + self.game + ' ' + self.alg_name + ' trial %d' % trial    
        else:
            self.unique_name = exp_name + ' supervised'

        # Get strings for different monitors
        self.episode_monitor_str = vis_opt['episode_monitor_str'] if 'episode_monitor_str' in vis_opt else None
        self.step_monitor_str = vis_opt['step_monitor_str'] if 'step_monitor_str' in vis_opt else None
        self.alg_monitor_str = vis_opt['alg_monitor_str'] if 'alg_monitor_str' in vis_opt else None   
        self.ll_alg_monitor_str = vis_opt['ll_alg_monitor_str'] if 'll_alg_monitor_str' in vis_opt else None
        # TODO - do this later if we care
        self.logpath = logpath
       
        # Get keys we want to plot
        self.plot_keys = vis_opt['plot_keys']
 
        # Timing stuff
        # Keep track of last time we updated info for a specific episode
        # This lets us 
        self.time_since_update = {}
        
        # Set up Visdom server and windows
        if vis: 
            self.viz = Visdom(port=port)
            self.wins = {}

        # Save opt for special cases
        self.opt = opt        
       
    # Save the current windows to file
    def dump_plots(self):
        # TODO unsure of usage, but this seems good to add
        return

    # Do some smoothing on our curves
    def smooth_curve(self, x, y):
        # Halfwidth of our smoothing convolution
        halfwidth = min(31, int(np.ceil(len(x) / 30)))
        k = halfwidth
        xsmoo = x[k:-k]
        ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
            np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
        downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
        return xsmoo[::downsample], ysmoo[::downsample]

    # Does binning of x and y over the given interval
    def fix_point(self, x, y, interval):
        np.insert(x, 0, 0)
        np.insert(y, 0, 0)

        fx, fy = [], []
        pointer = 0

        ninterval = int(max(x) / interval + 1)

        for i in range(ninterval):
            tmpx = interval * i

            while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
                pointer += 1

            if pointer + 1 < len(x):
                alpha = (y[pointer + 1] - y[pointer]) / \
                    (x[pointer + 1] - x[pointer])
                tmpy = y[pointer] + alpha * (tmpx - x[pointer])
                fx.append(tmpx)
                fy.append(tmpy)

        return fx, fy

    # TODO - Make a function that outputs our variance plots over multiple trials
    def plot_trial_var_values(self):
        return
    
    # Given keys and a list of lists, return a dictionary which for each key contains a list
    def format_data(self, keys, data):
        # Make struct, and init each key with empty list
        ret_struct = {}
        for key in keys:
            ret_struct[key] = []
            
        # Go through our data, and populate the lists
        for data_row in data:
            if len(keys) != len(data_row):
                pdb.set_trace()
                # TODO - there is some weird error where there is a missing newline between the header row and first data row
                # Figure out how to deal with this elegantly, or just restart the job
            #assert(len(keys) == len(data_row))
            for key, val in zip(keys, data_row):
                ret_struct[key].append(val)
        
        # Check length makes sense
        for key in keys:
            assert(len(ret_struct[key]) == len(data))
            
        return ret_struct
    
    # Preload all the file data
    def preload_data(self):
        # Episode monitor (one for each thread)
        if self.episode_monitor_str:
            episode_monitor_data = []
            infiles = glob.glob(os.path.join(self.logpath, '*.' + self.episode_monitor_str))
            for inf in infiles:
                # Read all the lines at once quickly
                with open(inf, 'r') as f:
                    rawlines = f.readlines()
                rawlines = [l.rstrip('\n') for l in rawlines]           
 
                # Get the header with key names
                episode_monitor_keys = rawlines[1].split(',')
                time_index = episode_monitor_keys.index('episode_dt')
            
                # Get values line by line
                for line in rawlines[2:]:
                    line_vals = line.split(',')
                    t_time = float(line_vals[time_index])
                    tmp = [t_time, line_vals]
                    episode_monitor_data.append(tmp)
        
            # Resort episode monitor by timestamp
            episode_monitor_data = sorted(episode_monitor_data, key=lambda x: x[0])
            episode_monitor_data = [x[1] for x in episode_monitor_data]
            
            # Extract the lists for each key
            self.episode_monitor_data = self.format_data(episode_monitor_keys, episode_monitor_data)
            
            # Get the x axis (episode_len, but using cumulitive sum)
            self.episode_monitor_x = np.cumsum([int(x) for x in self.episode_monitor_data['episode_len']])
            self.episode_monitor_x = np.concatenate((np.array([0]), self.episode_monitor_x[:-1]))
                
        # Load the last episode's step monitor into memory (note, only one, even for multithreaded)
        if self.step_monitor_str:
            step_monitor_data = []
            # Read all the lines at once quickly
            with open(os.path.join(self.logpath, self.step_monitor_str), 'r') as f:
                rawlines = f.readlines()
            rawlines = [l.rstrip('\n') for l in rawlines]               
 
            # Get the header with key names
            step_monitor_keys = rawlines[1].split(',')
            
            # Get values line by line
            for line in rawlines[2:]:
                line_vals = line.split(',')
                step_monitor_data.append(line_vals)

            # Extract lists for each key
            self.step_monitor_data = self.format_data(step_monitor_keys, step_monitor_data)
            
            # Get the x axis (just the step count)
            self.step_monitor_x = [i for i in range(len(step_monitor_data))]
            
        # Load the algorithm monitor data
        if self.alg_monitor_str:
            alg_monitor_data = []
            # Read all the lines at once quickly
            with open(os.path.join(self.logpath, self.alg_monitor_str), 'r') as f:
                rawlines = f.readlines()
            rawlines = [l.rstrip('\n') for l in rawlines]
   
            # Get the header with key names
            alg_monitor_keys = rawlines[1].split(',')
            
            # Get values line by line
            for line in rawlines[2:]:
                line_vals = line.split(',')
                alg_monitor_data.append(line_vals)
                
            # Extract lists for each key
            self.alg_monitor_data = self.format_data(alg_monitor_keys, alg_monitor_data)
            
            # Get the x axis (just the update count)
            self.alg_monitor_x = [i for i in range(len(alg_monitor_data))]
    
    # Load the raw plotting data from source
    def load_data(self, data_src, data_type, log_name):
        # Load the raw values 
        if data_src == 'episode_monitor':
            if not self.episode_monitor_str:
                raise Exception("Episode monitor data not loaded")
            # Get the data for this log name from the right data source (should be already loaded)
            monitor = self.episode_monitor_data
            raw_data_x = self.episode_monitor_x
        elif data_src == 'step_monitor':
            if not self.step_monitor_str:
                raise Exception("Step monitor data not loaded")
            monitor = self.step_monitor_data
            raw_data_x = self.step_monitor_x
        elif data_src == 'alg_monitor':
            if not self.alg_monitor_str:
                raise Exception("Alg monitor data not loaded")
            monitor = self.alg_monitor_data
            raw_data_x = self.alg_monitor_x
        else:
            raise NotImplementedError

        # If multiscalar, load from all the compnent keys
        if data_type in ['multiscalar', 'special']:
            raw_data_y = {}
            for key in log_name:
                #pdb.set_trace()
                if log_name[key] or type(log_name[key]) is str:
                    assert(key in monitor)
                if key in monitor:
                    raw_data_y[key] = monitor[key]
        # Else, just get from monitor
        else:
            raw_data_y = monitor[log_name]        

        return raw_data_x, raw_data_y

    # Plot a trace of an agents x y movement
    def display_movement(self, xypos, plot_struct, thetas=None, yaws=None):
        # Do matplot for xy movement
        xlabel_name = 'X'
        ylabel_name = 'Y'
        plt.figure()
        x = [dat[0] for dat in xypos]
        y = [dat[1] for dat in xypos]
        plt.scatter(x, y, marker='*')
        plt.xlabel(xlabel_name)
        plt.ylabel(ylabel_name)
        plt.title(self.unique_name)

        # If applicable, draw theta direction
        if thetas is not None:
            last_theta = None
            for t, theta in enumerate(thetas):
                if len(thetas) > 100 and t % 10 > 0:
                    continue
                if last_theta is None or np.linalg.norm(last_theta-theta) > 1e-6: 
                    if np.linalg.norm(theta) > 1e-3:
                        x_offset = x[t]
                        y_offset = y[t]                  
                        plt.arrow(x_offset, y_offset, theta[0], theta[1], head_width=0.05, head_length=0.1, fc='k', ec='k')
                last_theta = theta

        if yaws is not None:
            last_yaw = None
            for t, yaw in enumerate(yaws):
                if len(yaws) > 100 and t % 10 > 0:
                    continue
                if last_yaw is None or np.linalg.norm(last_yaw-yaw) > 1e-6:
                    if np.linalg.norm(yaw) > 1e-3:
                        x_offset = x[t]
                        y_offset = y[t]
                        plt.arrow(x_offset, y_offset, yaw[0], yaw[1], head_width=0.05, head_length=0.1, fc='g', ec='g')
                last_yaw = yaw

        # Update window
        # TODO - is there a draw over option maybe?
        if plot_struct['window_once']:
            win = self.viz.matplot(plt)
        elif keyname in self.wins:
            self.wins[keyname] = self.viz.matplot(plt, win=self.wins[keyname])
        else:
            self.wins[keyname] = self.viz.matplot(plt)
        plt.close()

    # Do simple x, y plot
    def simple_plot(self, x, y, keyname, plot_struct):
        # Check we've reached bin_size
        bin_size = plot_struct['bin_size']
        if len(x) < bin_size:
            return

        # Time subset (if applicable)
        if 'time_start' in plot_struct:
            start = plot_struct['time_start']
            end = plot_struct['time_end']
            x = x[start:end+1]
            y = y[start:end+1]

        # Do smoothing or any other work for x and y
        if plot_struct['smooth'] == 1:
            x, y = self.smooth_curve(x, y)
        elif plot_struct['smooth'] == 2:
            y = medfilt(y, kernel_size=9)
        if bin_size > 1:
            x, y = self.fix_point(x, y, bin_size)
       
        # Do matplot
        if plot_struct['data_src'] == 'episode_monitor':
            xlabel_name = 'Number of Timesteps'
        elif plot_struct['data_src'] == 'step_monitor':
            xlabel_name = 'Number of Steps'
        elif plot_struct['data_src'] == 'alg_monitor':
            xlabel_name = 'Number of Updates'
        else:
            raise NotImplementedError 
        plt.figure()
        plt.plot(x, y)
        plt.xlabel(xlabel_name)
        plt.ylabel(keyname)
        plt.title(self.unique_name)
        plt.grid(True)

        # Update window
        if keyname in self.wins:
            self.wins[keyname] = self.viz.matplot(plt, win=self.wins[keyname])
        else:
            self.wins[keyname] = self.viz.matplot(plt)
        plt.close()

    # Plot multiple y values
    def multi_plot(self, x, y_dict, keyname, plot_struct):
        # Check we've reached bin_size
        bin_size = plot_struct['bin_size']
        if len(x) < bin_size:
            return

        # Convert values to float
        for k in y_dict:
            y_dict[k] = [float(val.replace('\x00','')) for val in y_dict[k]]

        # Do smoothing or any other work for x and y
        orig_x = x
        if plot_struct['smooth'] == 1:
            for key in y_dict:
                y_orig = y_dict[key]
                x = np.copy(orig_x)
                x, y = self.smooth_curve(x, y_orig)
                y_dict[key] = y
        elif plot_struct['smooth'] == 2:
            for key in y_dict:
                y_dict[key] = medfilt(y_dict[key], kernel_size=9)
        orig_x = x
        for key in y_dict:
            x = np.copy(orig_x)
            y_orig = y_dict[key]
            x, y = self.fix_point(x, y_orig, bin_size)
            y_dict[key] = y

        # Do matplot
        if plot_struct['data_src'] == 'episode_monitor':
            xlabel_name = 'Number of Timesteps'
        elif plot_struct['data_src'] == 'step_monitor':
            xlabel_name = 'Number of Steps'
        elif plot_struct['data_src'] == 'alg_monitor':
            xlabel_name = 'Number of Updates'
        else:
            raise NotImplementedError 
        fig = plt.figure()

        # Put each y value on
        legend_handles = []
        legend_labels = []
        for key in y_dict:  # Kind of hack. First few values often plot garbage
            plt.plot(x, y_dict[key], label=key)
        plt.legend()
        plt.xlabel(xlabel_name)
        plt.ylabel(keyname)
        plt.title(self.unique_name)
        plt.grid(True)

        # Update window
        if keyname in self.wins:
            self.wins[keyname] = self.viz.matplot(plt, win=self.wins[keyname])
        else:
            self.wins[keyname] = self.viz.matplot(plt)
        plt.close()

    # Do simple value display
    def display_simple_value(self, value_str, keyname, plot_struct):
        display_text = self.unique_name + '\n' + keyname + ': ' + value_str
        if keyname in self.wins:
            self.wins[keyname] = self.viz.text(display_text, win=self.wins[keyname])
        else:
            self.wins[keyname] = self.viz.text(display_text)
 
    # Update visdom plot
    def update_display(self, raw_data_x, raw_data_y, keyname, plot_struct):
        # First split on data type
        # If scalar, pretty simple plotting
        data_type = plot_struct['data_type']
        # If scalar, call simple plot
        if data_type == 'scalar':
            raw_data_y = [float(y) for y in raw_data_y]
            self.simple_plot(raw_data_x, raw_data_y, keyname, plot_struct)
        # If multiscalar, do more complex plot with legend
        elif data_type == 'multiscalar':
            self.multi_plot(raw_data_x, raw_data_y, keyname, plot_struct)
        # If array, either do elementwise or get norm
        elif data_type == 'array':
            # Clean up and convert arrays
            conv_array = lambda s: np.fromstring(s.replace('[', '').replace(']', '').replace('"', ''), dtype=float, sep=' ') 
            raw_data_y = [conv_array(y) for y in raw_data_y]
            display_type = plot_struct['display_type']

            # If norm, take np norm and do simple plot
            if display_type == 'norm':
                raw_data_y = [np.linalg.norm(y) for y in raw_data_y]
                self.simple_plot(raw_data_x, raw_data_y, keyname, plot_struct)
            # If elementwise, plot for each element
            elif display_type == 'elementwise':
                for i in range(len(raw_data_y[0])):
                    new_key = keyname + '[%d]' % i
                    data_y = [y[i] for y in raw_data_y]
                    self.simple_plot(raw_data_x, data_y, new_key, plot_struct)
            # Same as above, but only show subset of indices
            elif display_type == 'elementwise_subset':
                # Display for [start_ind, end_ind)
                start_ind = plot_struct['start_ind']
                end_ind = plot_struct['end_ind']
                for i in range(start_ind, end_ind):
                    new_key = keyname + '[%d]' % i 
                    data_y = [y[i] for y in raw_data_y]
                    self.simple_plot(raw_data_x, data_y, new_key, plot_struct)
            else:
                raise NotImplementedError
        # If single value, display as text
        elif data_type == 'single_value':
            dat = raw_data_y[0]
            self.display_simple_value(dat, keyname, plot_struct)
        elif data_type == 'special':
            if keyname == 'theta_xy_plot':
                # Get xy and get theta 
                conv_array = lambda s: np.fromstring(s.replace('[', '').replace(']', '').replace('"', ''), dtype=float, sep=' ') 
                state_raw = [conv_array(y) for y in raw_data_y['state']]
                obs_raw = [conv_array(y) for y in raw_data_y['obs']]
                xypos = [y[:2] for y in state_raw]
                quats = [y[3:7] for y in state_raw]
                yaws = []
                for q in quats:
                    _, _, yaw = geom_utils.quaternion_to_euler_angle(q)
                    yaws.append(yaw)
                if 'theta_sz' in self.opt['env']:
                    theta_sz = self.opt['env']['theta_sz']
                    if self.opt['env']['add_timestep']:
                        start_ind = -(theta_sz+1)
                        end_ind = -2
                    else:
                        start_ind = -theta_sz
                        end_ind = -1
                    theta_space_mode = self.opt['env']['theta_space_mode']
                    yaws_2d = None
                    if theta_space_mode in ['arbitrary', 'arbitrary_stop']:
                        thetas = [np.array([y[start_ind], y[end_ind]]) for y in obs_raw]
                        assert(abs(np.linalg.norm(thetas[0]) - 1) < 1e-6)
                    elif theta_space_mode in ['simple_four']:
                        thetas = []
                        yaws_2d = []
                        for obs, yaw in zip(obs_raw, yaws):
                            if obs[start_ind] == 1:
                                theta = np.array([1, 0])
                            elif obs[start_ind+1] == 1:
                                theta = np.array([-1, 0])
                            elif obs[start_ind+2] == 1:
                                theta = np.array([0, 1])
                            elif obs[start_ind+3] == 1:
                                theta = np.array([0, -1])
                            else:
                                raise Exception("Something is wrong")
                            # Turn with yaw
                            #pdb.set_trace()
                            theta = geom_utils.convert_vector_to_egocentric(-yaw, theta)
                            thetas.append(theta)
                            yaws_2d.append(np.array([math.cos(yaw), math.sin(yaw)]))
                    elif theta_space_mode in ['simple_eight']:
                        thetas = []
                        yaws_2d = []
                        for obs, yaw in zip(obs_raw, yaws):
                            if obs[start_ind] == 1:
                                theta = np.array([1, 0])
                            elif obs[start_ind+1] == 1:
                                theta = np.array([-1, 0])
                            elif obs[start_ind+2] == 1:
                                theta = np.array([0, 1])
                            elif obs[start_ind+3] == 1:
                                theta = np.array([0, -1])
                            elif obs[start_ind+4] == 1:
                                theta = np.array([math.sqrt(0.5), math.sqrt(0.5)])
                            elif obs[start_ind+5] == 1:
                                theta = np.array([-math.sqrt(0.5), math.sqrt(0.5)])
                            elif obs[start_ind+6] == 1:
                                theta = np.array([-math.sqrt(0.5), -math.sqrt(0.5)])
                            elif obs[start_ind+7] == 1:
                                theta = np.array([math.sqrt(0.5), -math.sqrt(0.5)])
                            else:
                                raise Exception("Something is wrong")
                            # Turn with yaw
                            #pdb.set_trace()
                            theta = geom_utils.convert_vector_to_egocentric(-yaw, theta)
                            thetas.append(theta)
                        yaws_2d = None
                    elif theta_space_mode in ['k_theta']:
                        yaws_2d = []
                        for yaw in yaws:
                            yaws_2d.append(np.array([math.cos(yaw), math.sin(yaw)])) 
                        thetas = None
                    else:
                        thetas = None
                else:
                    thetas = None
                    yaws_2d = None
                #pdb.set_trace()
                self.display_movement(xypos, plot_struct, thetas, yaws_2d)
        else:
            raise NotImplementedError

    # Plot all values in visdom
    def visdom_plot(self):
        # Preload the monitor data
        self.preload_data() 
        
        # For each plot key
        for key in self.plot_keys:
            # Get values
            plot_struct = self.plot_keys[key]
            data_src = plot_struct['data_src']
            data_type = plot_struct['data_type']
            log_name = plot_struct['log_name']       

            # Check if we want to wait until a certain delay
            if 'update_delay' in plot_struct:                
                update_delay = plot_struct['update_delay']
                # Skip this if it hasn't been long enough since last update
                if key in self.time_since_update and time.time() - self.time_since_update[key] < update_delay:
                    continue
                else:
                    # Update time delay value
                    self.time_since_update[key] = time.time()
                            
            # Get the data from the correct source
            data_x, data_y = self.load_data(data_src, data_type, log_name)
            
            # Display the data
            self.update_display(data_x, data_y, key, plot_struct)


# Main

# Get Input Arguments
parser = argparse.ArgumentParser(description='RL')

##################################################
# yaml options file contains all default choices #
parser.add_argument('--path_opt', default='options/baseline/default.yaml', type=str,
                    help='path to a yaml options file')
# yaml option file containing the visdom plotting options
parser.add_argument('--vis_path_opt', default='options/visualization/default.yaml', type=str,
                    help='path to a yaml visualization options file')
parser.add_argument('--algo', default='a2c',
                    help='algorithm to use: a2c | ppo | acktr')
parser.add_argument('--env-name', default='Hopper-v2',
                    help='environment to train on (default: Hopper-v2)')
parser.add_argument('--trial', type=int, default=0,
                    help='keep track of what trial you are on')
parser.add_argument('--save-every', default=100, type=int,
                    help='how often to save our models permanently')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-vis', action='store_true', default=False,
                    help='disables visdom visualization')
parser.add_argument('--port', type=int, default=8097,
                    help='port to run the server on (default: 8097)')

def main():
    global args
    args = parser.parse_args() 
    args.vis = not args.no_vis

    # Set options
    if args.path_opt is not None:
        with open(args.path_opt, 'r') as handle:
            options = yaml.load(handle)
    if args.vis_path_opt is not None:
        with open(args.vis_path_opt, 'r') as handle:
            vis_options = yaml.load(handle)

    # Put alg_%s and optim_%s to alg and optim depending on commandline
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

    # Get log path
    logpath = os.path.join(log_opt['log_base'], model_opt['mode'], log_opt['exp_name'], args.algo, args.env_name, 'trial%d' % args.trial)
    assert(os.path.isdir(logpath))    

    # Set up plotting dashboard
    dashboard = Dashboard(options, vis_options, logpath, vis=args.vis, port=args.port)

    # Do visdom logging
    dashboard.visdom_plot()

    # TODO - later want to add way of dumping videos here    

# Should be used to get plots for finished trials
if __name__ == "__main__":
    main()

