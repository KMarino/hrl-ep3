# Modified by Kenneth Marino                    
# Impliments the models for the RL algos
# Modified from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from distributions import Categorical, DiagGaussian
from baselines.common.running_mean_std import RunningMeanStd
from utils import init, init_normc_
import pdb
import time
import random
import numpy as np

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# Defines the policy network for DQN
# The network is really a value function, but we can call act on it like we can with a policy network
class DQNPolicy(nn.Module):
    def __init__(self, obs_shape, action_space, opt):
        super(DQNPolicy, self).__init__()
        assert(action_space.__class__.__name__ == "Discrete")

        # Define the network
        # Baselines networks
        if opt['mode'] in ['hierarchical', 'hierarchical_many']:
            self.policy_net = DQNMLPBase(obs_shape[0], action_space.n, opt)
            self.target_net = DQNMLPBase(obs_shape[0], action_space.n, opt)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
        else:
            raise NotImplementedError

        # Define output space and sampling dist
        assert(action_space.__class__.__name__ == "Discrete")
        self.state_size = self.policy_net.state_size

        # Define number of update steps
        self.total_num_updates = 0
        self.action_space = action_space

        # Get eps info
        self.eps_start = opt['eps_start']
        self.eps_end = opt['eps_end']
        self.eps_decay = opt['eps_decay']
        self.steps_done = 0

    # Not implemented. self.base is invoked directly, look at that forward
    def forward(self, inputs, states, masks):
        raise NotImplementedError

    # Get the resulting actions from policy
    def act(self, inputs, states, masks, deterministic=False):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        # Select action
        if sample > eps_threshold:
            with torch.no_grad():
                action = self.policy_net(inputs).max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(self.action_space.n)] for i in range(inputs.size(0))], dtype=torch.long)
        value = torch.zeros(inputs.size(0), 1)
        action_log_probs = torch.zeros(inputs.size(0), 1)

        return value, action, action_log_probs, states

# Defines the policy network
class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, opt):
        super(Policy, self).__init__()

        # Define the network
        # Baselines networks
        if opt['mode'] in ['baseline', 'baseline_reverse', 'baselinewtheta', 'hierarchical', 'hierarchical_many']:
            if len(obs_shape) == 3:
                self.base = CNNBase(obs_shape[0], opt)
            elif len(obs_shape) == 1:
                assert not opt['recurrent_policy'], "Recurrent policy is not implemented for the MLP controller"
                # Add theta mult network option
                if opt['model_type'] == 'Module' and opt['mode'] == 'baselinewtheta':
                    self.base = SwitchModuleBaseWithTheta(obs_shape[0], opt) 
                else:
                    self.base = MLPBase(obs_shape[0], opt)
            else:
                raise NotImplementedError
        elif opt['mode'] == 'phasesimple':
            if len(obs_shape) == 3:
                raise NotImplementedError
            elif len(obs_shape) == 1:
                assert not opt['recurrent_policy'], "Recurrent policy is not implemented for the MLP controller"
                self.base = MLPBaseSimplePhase(obs_shape[0], opt)
            else:
                raise NotImplementedError
        elif opt['mode'] in ['phasewstate', 'phasewtheta', 'interpolate', 'cyclic']:
            if len(obs_shape) == 3:
                raise NotImplementedError
            elif len(obs_shape) == 1:
                assert not opt['recurrent_policy'], "Recurrent policy is not implemented for the MLP controller"
                if opt['model_type'] == 'MLP':
                    self.base = MLPBasePhaseWithState(obs_shape[0], opt)
                elif opt['model_type'] == 'Mult':
                    self.base = MultMLPBasePhaseWithState(obs_shape[0], opt)
                elif opt['model_type'] == 'Module' and opt['mode'] == 'phasewtheta':
                    self.base = SwitchModuleBasePhaseWithStateTheta(obs_shape[0], opt) 
                elif opt['model_type'] == 'Module':
                    self.base = SwitchModuleBasePhaseWithState(obs_shape[0], opt)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        elif opt['mode'] in ['baseline_lowlevel', 'phase_lowlevel', 'supervised']:
            if len(obs_shape) == 3:
                raise NotImplementedError
            elif len(obs_shape) == 1:
                assert not opt['recurrent_policy'], "Recurrent policy is not implemented for the MLP controller"
                if opt['model_type'] == 'MLPSimpleDebug':
                    self.base = MLPBase(obs_shape[0], opt)
                elif opt['model_type'] == 'MLPPretrain':
                    self.base = MLPBasePretrain(obs_shape[0], opt)
                elif opt['model_type'] == 'MLPDeepMindPretrain':
                    self.base = MLPBaseDeepMindPretrain(obs_shape[0], opt)
                elif opt['model_type'] == 'MLP':
                    self.base = MLPBaseTheta(obs_shape[0], opt)
                elif opt['model_type'] == 'Bilinear':
                    self.base = MLPBaseBilinearTheta(obs_shape[0], opt)
                elif opt['model_type'] == 'Module':
                    self.base = SwitchModuleBaseTheta(obs_shape[0], opt)
                else:
                    raise NotImplementedError 
        # Baseline will use the MLPBasePretrain
        elif opt['mode'] in ['maze_baseline', 'maze_baseline_wphase']:
            if len(obs_shape) == 3:
                raise NotImplementedError
            elif len(obs_shape) == 1:
                if opt['mode'] == 'maze_baseline':
                    opt['mode'] = 'baseline_lowlevel'
                    self.base = MLPBase(obs_shape[0], opt)
                elif opt['mode'] == 'maze_baseline_wphase':
                    opt['mode'] = 'phase_lowlevel'
                    self.base = MLPBasePretrain(obs_shape[0], opt)               
        else:
            raise NotImplementedError

        # Define output space and sampling dist
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError
        self.state_size = self.base.state_size
        self.is_recurrent = hasattr(self.base, 'gru')

    # Do special loading
    def load_state_dict_special(self, ckpt, input_mask):
        ckpt = self.base.update_state_dict_special(ckpt, input_mask)
        self.load_state_dict(ckpt)

    # Not implemented. self.base is invoked directly, look at that forward
    def forward(self, inputs, states, masks):
        raise NotImplementedError

    # Get the resulting values and actions from policy
    def act(self, inputs, states, masks, deterministic=False):
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, states

    # Same as act, but only gives the value
    def get_value(self, inputs, states, masks):
        value, _, _ = self.base(inputs, states, masks)
        return value

    # Evaluate actions (similar to act, but don't sample actions)
    def evaluate_actions(self, inputs, states, masks, action):
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, states

# Defines the policy network
# This version actually just contains a set of low level policies and chooses which
# based on passed in theta
class ModularPolicy(nn.Module):
    def __init__(self, raw_obs_shape, action_space, num_modules, opt):
        super(ModularPolicy, self).__init__()

        # Create the seperate policies
        self.opt = opt
        self.sub_policies = []
        self.filter_means = []
        self.filter_vars = []
        for mod_num in range(num_modules):
            # Create the policy
            sub_policy = Policy(raw_obs_shape, action_space, opt['model'])
            self.add_module('policy_%d' % mod_num, sub_policy)
            self.sub_policies.append(sub_policy)

            # Make the ob_rms
            ob_rms = RunningMeanStd(shape=raw_obs_shape)
            ob_mean = torch.from_numpy(ob_rms.mean).float()
            ob_var = torch.from_numpy(ob_rms.var).float()
            self.filter_means.append(ob_mean.unsqueeze(0))
            self.filter_vars.append(ob_var.unsqueeze(0))
        self.filter_means = torch.cat(self.filter_means)
        self.filter_vars = torch.cat(self.filter_vars)

        # Ignore mask (don't filter count)
        self.ignore_mask = torch.zeros(raw_obs_shape)
        if opt['env']['add_timestep']:
            self.ignore_mask[-1] = 1

        # Other random things we need
        self.state_size = self.sub_policies[0].state_size
        self.is_recurrent = False

    # Extract which policy we should use given the observation
    def get_subpolicy_index(self, obs):
        # Extract theta 
        if self.opt['env']['add_timestep']:
            step = obs[-1]
            obs = obs[:-1]
        theta_input = obs[-len(self.sub_policies):]

        # Get the index value
        theta_ind = -1
        for i in range(len(self.sub_policies)):
            if theta_input[i] > 0.5:
                theta_ind = i     
        assert(theta_ind >= 0)
        return theta_ind

    # Filter obs using the filters used by each policy originally
    def filter_obs(self, obs, theta_ind):
        # Get mean and variance filters
        ob_mean = self.filter_means[theta_ind]
        ob_var = self.filter_vars[theta_ind]

        # Get rid of the theta part of obs
        if self.opt['env']['add_timestep']:
            raw_obs = obs[:, :-1]
            raw_obs = raw_obs[:, :-len(self.sub_policies)]
            counts = obs[:, -1].unsqueeze(0)
            obs = torch.cat([raw_obs, counts], 1)
        else:
            obs = obs[:, :-len(self.sub_policies)] 


        # Do mean/var filtering
        obs_orig = obs.clone()
        obs = (obs - ob_mean) / (3*ob_var.sqrt() + 0.1)       
        obs = (1 - self.ignore_mask) * obs + self.ignore_mask * obs_orig

        return obs

    # Not implemented. self.base is invoked directly, look at that forward
    def forward(self, inputs, states, masks):
        raise NotImplementedError

    # The methods below all just figure out subpolicy ind and call the correct policies version
    # Deals with multiple inputs by calling them in a loop (which is kind of dumb)
    # This doesn't matter much if we're not optimizing ll policy
    # If we are, this is massively inefficient
    def act(self, inputs, states, masks, deterministic=False):
        # Hardcoded for single actions right now (revisit for A2C or if we update ll policy)
        if inputs.size(0) == 1:
            policy_ind = self.get_subpolicy_index(inputs[0]) 
            inputs = self.filter_obs(inputs, policy_ind)
            return self.sub_policies[policy_ind].act(inputs, states, masks, deterministic)
        # Or just iterate through (possibly inefficient)
        else:
            values = [] 
            actions = []
            action_log_probs = []
            new_states = []
            for obs, state, mask in zip(inputs, states, masks):
                policy_ind = self.get_subpolicy_index(obs)
                obs = self.filter_obs(obs.unsqueeze(0), policy_ind)
                state = state.unsqueeze(0)
                mask = mask.unsqueeze(0)
                value, action, action_log_prob, state = self.sub_policies[policy_ind].act(obs, state, mask, deterministic)
                values.append(value)
                actions.append(action)
                action_log_probs.append(action_log_prob)
                new_states.append(state)
            values = torch.cat(values, 0)
            actions = torch.cat(actions, 0)
            action_log_probs = torch.cat(action_log_probs, 0)
            new_states = torch.cat(new_states, 0)
            return values, actions, action_log_probs, new_states

    def get_value(self, inputs, states, masks):
        # Hardcoded for single actions right now (revisit for A2C or if we update ll policy) 
        #assert(inputs.size(0) == 1) 
        if inputs.size(0) == 1:
            policy_ind = self.get_subpolicy_index(inputs[0])
            inputs = self.filter_obs(inputs, policy_ind)
            return self.sub_policies[policy_ind].get_value(inputs, states, masks)
        else:
            values = []
            for obs, state, mask in zip(inputs, states, masks):
                policy_ind = self.get_subpolicy_index(obs)
                obs = self.filter_obs(obs.unsqueeze(0), policy_ind)
                state = state.unsqueeze(0)
                mask = mask.unsqueeze(0)
                value = self.sub_policies[policy_ind].get_value(obs, state, mask)
                values.append(value)
            values = torch.cat(values, 0)
            return values

    def evaluate_actions(self, inputs, states, masks, actions):
        # Hardcoded for single actions right now (revisit for A2C or if we update ll policy) 
        #assert(inputs.size(0) == 1) 
        if inputs.size(0) == 1:
            policy_ind = self.get_subpolicy_index(inputs[0])
            inputs = self.filter_obs(inputs, policy_ind)
            return self.sub_policies[policy_ind].evaluate_actions(inputs, states, masks, actions)
        else:
            values = []
            dist_entropies = []
            new_states = []
            action_log_probs = []
            for obs, state, mask, action in zip(inputs, states, masks, actions):
                policy_ind = self.get_subpolicy_index(obs)
                obs = self.filter_obs(obs.unsqueeze(0), policy_ind)
                state = state.unsqueeze(0)
                mask = mask.unsqueeze(0)
                action = action.unsqueeze(0)
                value, action_log_prob, dist_entropy, state = self.sub_policies[policy_ind].evaluate_actions(obs, state, mask, action)
                values.append(value)
                action_log_probs.append(action_log_prob)
                dist_entropies.append(dist_entropy)
                new_states.append(state)
            values = torch.cat(values, 0)
            action_log_probs = torch.cat(action_log_probs, 0)
            new_states = torch.cat(new_states, 0)
            dist_entropy = sum(dist_entropies) / len(actions)
            return values, action_log_probs, dist_entropy, new_states

    # Load policies that were pretrained from somewhere else
    def load_pretrained_policies(self, ckpts):
        assert(len(ckpts) == len(self.sub_policies))
        for mod_num, ckpt in enumerate(ckpts):
            policy_state_dict = ckpt['agent']['model']
            self.sub_policies[mod_num].load_state_dict(policy_state_dict)
            ob_rms = ckpt['ob_rms']
            self.filter_means[mod_num].copy_(torch.from_numpy(ob_rms.mean))
            self.filter_vars[mod_num].copy_(torch.from_numpy(ob_rms.var))

    # Overwrite load and save state dicts so we are sure to save everything
    def state_dict(self):
        ckpt = {}
        ckpt['models'] = []
        ckpt['rms_means'] = []
        ckpt['rms_vars'] = []
        for policy, filter_mean, filter_var in zip(self.sub_policies, self.filter_means, self.filter_vars):
            ckpt['models'].append(policy.state_dict())
            ckpt['rms_means'].append(filter_mean.cpu())
            ckpt['rms_vars'].append(filter_var.cpu())
        assert(len(self.sub_policies) == len(self.filter_means))
        assert(len(self.sub_policies) == len(self.filter_vars))
        return ckpt
    def load_state_dict(self, ckpt):
        assert(len(self.sub_policies) == len(ckpt['models']))
        assert(len(self.sub_policies) == len(ckpt['rms_means']))
        assert(len(self.sub_policies) == len(ckpt['rms_vars']))
        for mod_num, model_ckpt, rms_mean, rms_var in zip(range(len(self.sub_policies)), ckpt['models'], ckpt['rms_means'], ckpt['rms_vars']):
            self.sub_policies[mod_num].load_state_dict(model_ckpt)
            self.filter_means[mod_num].copy_(rms_mean)
            self.filter_vars[mod_num].copy_(rms_var)

    # Overwrite cuda
    def cuda(self):
        self.filter_means = self.filter_means.cuda()
        self.filter_vars = self.filter_vars.cuda()
        self.ignore_mask = self.ignore_mask.cuda()
        for policy in self.sub_policies:
            policy.cuda()

# Simple DQN MLP module
class DQNMLPBase(nn.Module):
    def __init__(self, input_sz, num_actions, opt):
        super(DQNMLPBase, self).__init__()
        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        # Get params
        hid_sz = opt['hid_sz']
        num_layer = opt['num_layer']

        # Make network
        self.tanh = nn.Tanh()
        self.in_fc = init_(nn.Linear(input_sz, hid_sz))
        assert(num_layer >= 1) 
        hid_layers = []
        for i in range(0, num_layer-1):
            hid_fc = init_(nn.Linear(hid_sz, hid_sz))
            hid_layers.append(hid_fc)
        self.hid_layers = ListModule(*hid_layers)
        self.out_fc = init_(nn.Linear(hid_sz, num_actions))

    def forward(self, input):
        x = self.in_fc(input) 
        x = self.tanh(x)
        for hid_fc in self.hid_layers:
            x = hid_fc(x)
            x = self.tanh(x)
        x = self.out_fc(x)
        return x

    # Here to make code happy (does nothing essentially)
    @property
    def state_size(self):
        return 1

# Base CNN architecture for actor/critic networks
# For envs that have image inaput
class CNNBase(nn.Module):
    def __init__(self, num_inputs, opt):
        super(CNNBase, self).__init__()
        use_gru = opt['recurrent_policy']

        init_ = lambda m: init(m,
                      nn.init.orthogonal_,
                      lambda x: nn.init.constant_(x, 0),
                      nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, 512)),
            nn.ReLU()
        )

        if use_gru:
            self.gru = nn.GRUCell(512, 512)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        init_ = lambda m: init(m,
          nn.init.orthogonal_,
          lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(512, 1))

        self.train()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    @property
    def output_size(self):
        return 512

    def forward(self, inputs, states, masks):
        x = self.main(inputs / 255.0)

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)

        return self.critic_linear(x), x, states

# Base MLP architecture for actor/critic networks
# For envs that have 1D inputs (like most MuJoCo locomotion tasks)
class MLPBase(nn.Module):
    def __init__(self, num_inputs, opt):
        super(MLPBase, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        # Actor output always 64 because the linear to actual size is in distribution
        self.hid_sz = opt['hid_sz']
        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, self.hid_sz)),
            nn.Tanh(),
            init_(nn.Linear(self.hid_sz, self.hid_sz)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, self.hid_sz)),
            nn.Tanh(),
            init_(nn.Linear(self.hid_sz, self.hid_sz)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(self.hid_sz, 1))

        self.train()

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self.hid_sz

    def forward(self, inputs, states, masks):
        st = time.time()
        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)
        hidden_critic = self.critic_linear(hidden_critic)
        #if inputs.size(0) > 1:
        #    print(inputs.size(0))
        #    print(time.time()-st)

        return hidden_critic, hidden_actor, states

    # Load special (copy with different input weights)
    def update_state_dict_special(self, ckpt, oldinput_mask):
        # Edit 'base.actor.0.weight' and 'base.critic.0.weight'
        new_actor_in = torch.Tensor(self.actor[0].weight)
        new_critic_in = torch.Tensor(self.critic[0].weight)

        # Splice in corresponding part of checkpoint
        oldinput_mask = torch.Tensor(oldinput_mask.astype(int)) == True
        new_actor_in[:, oldinput_mask].copy_(ckpt['base.actor.0.weight'])
        new_critic_in[:, oldinput_mask].copy_(ckpt['base.critic.0.weight'])
        ckpt['base.actor.0.weight'] = new_actor_in
        ckpt['base.critic.0.weight'] = new_critic_in 

        # Return new ckpt
        return ckpt

# Base MLP architecture for actor/critic networks
# For envs that have 1D inputs (like most MuJoCo locomotion tasks)
# This version also takes the phase counter as an input and has learned params for each count
class MLPBaseSimplePhase(nn.Module):
    def __init__(self, num_inputs, opt):
        super(MLPBaseSimplePhase, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        # Define phase parameter list
        self.param_sz = opt['phase_hid_sz']
        self.k = opt['phase_period']
        self.phase_param = nn.Embedding(self.k, self.param_sz)

        # Define actor network modules
        self.hid_sz = opt['hid_sz']
        self.tanh = nn.Tanh()
        self.actor_in_fc = init_(nn.Linear(self.param_sz, self.hid_sz))
        self.actor_hid_fc = init_(nn.Linear(self.hid_sz, self.hid_sz))

        # Define critic network modules
        self.critic_in_fc = init_(nn.Linear(self.param_sz, self.hid_sz))
        self.critic_hid_fc = init_(nn.Linear(self.hid_sz, self.hid_sz))
        self.critic_out_fc = init_(nn.Linear(self.hid_sz, 1))

        # Set to training mode by default
        self.train()

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self.hid_sz

    # Forward through actor critic
    def forward(self, inputs, states, masks):
        # Seperate out the raw observation input from the step count
        obs_input = inputs[:, :-1]
        step_count = torch.round(inputs[:, -1])
        assert(abs(round(step_count[0].item()) - step_count[0].item()) < 1e-6)
        step_count = torch.LongTensor(step_count.size()).copy_(step_count)
        if obs_input.is_cuda:
            step_count = step_count.cuda()

        # Get phase input (do mod)
        phase_input = step_count % self.k

        # Forward through actor and critic models
        # We actually don't use obs_input
        hidden_critic = self.critic_forward(phase_input)
        hidden_actor = self.actor_forward(phase_input)

        return hidden_critic, hidden_actor, states

    # Forward through actor network
    def critic_forward(self, phase_input):
        x = self.phase_param(phase_input)
        x = self.critic_in_fc(x)
        x = self.tanh(x)
        x = self.critic_hid_fc(x)
        x = self.tanh(x)
        x = self.critic_out_fc(x)
        return x

    # Forward through actor network
    def actor_forward(self, phase_input):
        x = self.phase_param(phase_input)
        x = self.actor_in_fc(x)
        x = self.tanh(x)
        x = self.actor_hid_fc(x)
        x = self.tanh(x)
        return x

# Base MLP architecture for actor/critic networks
# For envs that have 1D inputs (like most MuJoCo locomotion tasks)
# This version also takes the phase counter as an input and has learned params for each count
# This version uses both state and phase
class MLPBasePhaseWithState(nn.Module):
    def __init__(self, num_inputs, opt):
        super(MLPBasePhaseWithState, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        # Define phase parameter list
        self.param_sz = opt['phase_hid_sz']
        self.k = opt['phase_period']
        self.phase_param = nn.Embedding(self.k, self.param_sz)

        # Figure out if you want to use timestep as obs
        self.use_timestep = opt['use_timestep']
        self.time_scale = opt['time_scale']
        if self.use_timestep:
            input_sz = self.param_sz + num_inputs
        else:
            input_sz = self.param_sz + num_inputs - 1 

        # Get number of layers
        self.num_layer = opt['num_layer']

        # Define actor network modules
        self.hid_sz = opt['hid_sz']
        self.tanh = nn.Tanh()
        self.actor_in_fc = init_(nn.Linear(input_sz, self.hid_sz))

        # Define critic network modules
        self.critic_in_fc = init_(nn.Linear(input_sz, self.hid_sz))
        self.critic_out_fc = init_(nn.Linear(self.hid_sz, 1))

        # Depending on num_layer, add more layers
        assert(self.num_layer >= 2)
        actor_hid_layers = []
        critic_hid_layers = []
        for i in range(0, self.num_layer-1):
            actor_hid_fc = init_(nn.Linear(self.hid_sz, self.hid_sz))
            actor_hid_layers.append(actor_hid_fc)
            critic_hid_fc = init_(nn.Linear(self.hid_sz, self.hid_sz))
            critic_hid_layers.append(critic_hid_fc)
        self.actor_hid_layers = ListModule(*actor_hid_layers)     
        self.critic_hid_layers = ListModule(*critic_hid_layers) 

        # Set to training mode by default
        self.train()

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self.hid_sz

    # Forward through actor critic
    def forward(self, inputs, states, masks):
        # Seperate out the raw observation input from the step count
        if self.use_timestep:
            obs_input = inputs
        else:
            obs_input = inputs[:, :-1]
        time_mult = 1 / self.time_scale
        step_count = torch.round(inputs[:, -1]*time_mult)
        assert(abs(round(step_count[0].item()) - step_count[0].item()) < 1e-6)
        step_count = torch.LongTensor(step_count.size()).copy_(step_count)
        if obs_input.is_cuda:
            step_count = step_count.cuda()

        # Get phase input (do mod)
        phase_input = step_count % self.k

        # Forward through actor and critic models
        hidden_critic = self.critic_forward(phase_input, obs_input)
        hidden_actor = self.actor_forward(phase_input, obs_input)

        return hidden_critic, hidden_actor, states

    # Forward through actor network
    def critic_forward(self, phase_input, obs_input):
        phase_input = self.phase_param(phase_input) 
        x = torch.cat([phase_input, obs_input], 1)
        x = self.critic_in_fc(x)
        x = self.tanh(x)
        for critic_hid_fc in self.critic_hid_layers:
            x = critic_hid_fc(x)
            x = self.tanh(x)
        x = self.critic_out_fc(x)
        return x

    # Forward through actor network
    def actor_forward(self, phase_input, obs_input):
        phase_input = self.phase_param(phase_input) 
        x = torch.cat([phase_input, obs_input], 1) 
        x = self.actor_in_fc(x)
        x = self.tanh(x)
        for actor_hid_fc in self.actor_hid_layers:
            x = actor_hid_fc(x)
            x = self.tanh(x)
        return x

# Base MLP architecture for actor/critic networks
# For envs that have 1D inputs (like most MuJoCo locomotion tasks)
# This version also takes the phase counter as an input and has learned params for each count
# This version uses both state and phase
class MultMLPBasePhaseWithState(nn.Module):
    def __init__(self, num_inputs, opt):
        super(MultMLPBasePhaseWithState, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        # Define phase parameter list
        self.param_sz = opt['phase_hid_sz']
        self.k = opt['phase_period']
        self.phase_param = nn.Embedding(self.k, self.param_sz)

        # Figure out if you want to use timestep as obs
        self.use_timestep = opt['use_timestep']
        self.time_scale = opt['time_scale']
        if not self.use_timestep: 
            num_inputs -= 1

        # Get number of layers
        self.num_layer = opt['num_layer']

        # Define actor network modules
        self.hid_sz = opt['hid_sz']
        self.tanh = nn.Tanh()
        self.actor_obs_in_fc = init_(nn.Linear(num_inputs, self.hid_sz))
        self.actor_phase_in_fc = init_(nn.Linear(self.param_sz, self.hid_sz))

        # Define critic network modules
        self.critic_obs_in_fc = init_(nn.Linear(num_inputs, self.hid_sz))
        self.critic_phase_in_fc = init_(nn.Linear(self.param_sz, self.hid_sz))
        self.critic_out_fc = init_(nn.Linear(self.hid_sz, 1))

        # Depending on num_layer, add more layers
        assert(self.num_layer >= 2)
        actor_hid_layers = []
        critic_hid_layers = []
        for i in range(0, self.num_layer-1):
            actor_hid_fc = init_(nn.Linear(self.hid_sz, self.hid_sz))
            actor_hid_layers.append(actor_hid_fc)
            critic_hid_fc = init_(nn.Linear(self.hid_sz, self.hid_sz))
            critic_hid_layers.append(critic_hid_fc)
        self.actor_hid_layers = ListModule(*actor_hid_layers)     
        self.critic_hid_layers = ListModule(*critic_hid_layers) 

        # Set to training mode by default
        self.train()

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self.hid_sz

    # Forward through actor critic
    def forward(self, inputs, states, masks):
        # Seperate out the raw observation input from the step count
        if self.use_timestep:
            obs_input = inputs
        else:
            obs_input = inputs[:, :-1]
        time_mult = 1 / self.time_scale
        step_count = torch.round(inputs[:, -1]*time_mult)
        assert(abs(round(step_count[0].item()) - step_count[0].item()) < 1e-6)
        step_count = torch.LongTensor(step_count.size()).copy_(step_count)
        if obs_input.is_cuda:
            step_count = step_count.cuda()

        # Get phase input (do mod)
        phase_input = step_count % self.k

        # Forward through actor and critic models
        hidden_critic = self.critic_forward(phase_input, obs_input)
        hidden_actor = self.actor_forward(phase_input, obs_input)

        return hidden_critic, hidden_actor, states

    # Forward through actor network
    def critic_forward(self, phase_input, obs_input):
        obs_input = self.critic_obs_in_fc(obs_input)
        phase_input = self.phase_param(phase_input)
        phase_input = self.critic_phase_in_fc(phase_input)
        x = obs_input * phase_input
        x = self.tanh(x)
        for critic_hid_fc in self.critic_hid_layers:
            x = critic_hid_fc(x)
            x = self.tanh(x)
        x = self.critic_out_fc(x)
        return x

    # Forward through actor network
    def actor_forward(self, phase_input, obs_input):
        obs_input = self.actor_obs_in_fc(obs_input)
        phase_input = self.phase_param(phase_input)
        phase_input = self.actor_phase_in_fc(phase_input) 
        x = obs_input * phase_input
        x = self.tanh(x)
        for actor_hid_fc in self.actor_hid_layers:
            x = actor_hid_fc(x)
            x = self.tanh(x)
        return x

# BMLP architecture for actor/critic networks
# For envs that have 1D inputs (like most MuJoCo locomotion tasks)
# This version also takes the phase counter as an input and has learned params for each count
# This version uses the SwitchModule
class SwitchModuleBasePhaseWithState(nn.Module):
    def __init__(self, num_inputs, opt):
        super(SwitchModuleBasePhaseWithState, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        # Define phase parameter list
        self.k = opt['phase_period']
        
        # If hard, using phase as index, so num_modules is simply period
        if opt['switch_mode'] == 'hard':
            opt['num_modules'] = self.k
        elif opt['switch_mode'] == 'soft':
            opt['switch_sz'] = self.k
        self.switch_mode = opt['switch_mode']

        # Figure out if you want to use timestep as obs
        self.use_timestep = opt['use_timestep']
        self.time_scale = opt['time_scale']
        if not self.use_timestep: 
            num_inputs -= 1

        # Figure out hidden size
        self.hid_sz = opt['hid_sz']

        # Define actor and critic network modules
        self.tanh = nn.Tanh()
        self.actor_module_net = SwitchModule(num_inputs, opt)
        self.critic_module_net = SwitchModule(num_inputs, opt)
        self.critic_out_fc = init_(nn.Linear(self.hid_sz, 1))      
 
        # Set to training mode by default
        self.train()

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self.hid_sz

    # Forward through actor critic
    def forward(self, inputs, states, masks):
        # Seperate out the raw observation input from the step count
        if self.use_timestep:
            obs_input = inputs
        else:
            obs_input = inputs[:, :-1]
        step_count = torch.round(inputs[:, -1]/self.time_scale)
        assert(abs(round(step_count[0].item()) - step_count[0].item()) < 1e-6)
        step_count = torch.LongTensor(step_count.size()).copy_(step_count)
        if obs_input.is_cuda:
            step_count = step_count.cuda()

        # Get phase input (do mod)
        phase_input = step_count % self.k

        if self.switch_mode == 'soft':
            phase_input = phase_input.reshape([-1, 1])
            phase_onehot = torch.zeros(phase_input.size(0), self.k)
            if obs_inputs.is_cuda:
                phase_onehot = phase_onehot.cuda()
            phase_onehot.scatter_(1, phase_input, 1)
            phase_input = phase_onehot

        # Forward through actor and critic models
        hidden_critic = self.critic_module_net([obs_input, phase_input])
        hidden_critic = self.critic_out_fc(self.tanh(hidden_critic))
        hidden_actor = self.actor_module_net([obs_input, phase_input])
        hidden_actor = self.tanh(hidden_actor)

        return hidden_critic, hidden_actor, states

# BMLP architecture for actor/critic networks
# For envs that have 1D inputs (like most MuJoCo locomotion tasks)
# This version also takes the theta as an input
# This version uses the SwitchModule
class SwitchModuleBaseWithTheta(nn.Module):
    def __init__(self, num_inputs, opt):
        super(SwitchModuleBaseWithTheta, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        assert(opt['switch_mode'] == 'soft') # I'm actually not sure how to do the hard mode anyway

        # Remove theta from state
        self.theta_sz = opt['theta_sz']
        num_inputs -= self.theta_sz

        # Choose theta mode
        self.theta_space_mode = opt['theta_space_mode']  
        if self.theta_space_mode == 'simple_four':     
            opt['switch_sz'] = 4 
        elif self.theta_space_mode in ['arbitrary', 'arbitrary_stop', 'k_theta', 'k_theta_stop']:
            opt['switch_sz'] = self.theta_sz
        #assert(opt['theta_space_mode'] == 'simple_four')    # Hardcode to simple four mode
        self.switch_mode = opt['switch_mode']

        # Figure out hidden size
        self.hid_sz = opt['hid_sz']

        # Define actor and critic network modules
        self.tanh = nn.Tanh()
        self.actor_module_net = SwitchModule(num_inputs, opt)
        self.critic_module_net = SwitchModule(num_inputs, opt)
        self.critic_out_fc = init_(nn.Linear(self.hid_sz, 1))      
 
        # Set to training mode by default
        self.train()

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self.hid_sz

    # Forward through actor critic
    def forward(self, inputs, states, masks):
        # Seperate theta and obs
        theta_input = inputs[:, -self.theta_sz:]
        obs_input = inputs[:, :-self.theta_sz]
       
        # Get theta input (unhardcode the four encoding)
        if self.theta_space_mode == 'simple_four':
            theta_onehot = torch.zeros(theta_input.size(0), 4)
            if obs_input.is_cuda:
                theta_onehot = theta_onehot.cuda()
            for i in range(theta_input.size(0)): 
                if theta_input[i, 0] == 1:
                    theta_onehot[i, 0] = 1
                elif theta_input[i, 0] == -1:
                    theta_onehot[i, 1] = 1
                elif theta_input[i, -1] == 1:
                    theta_onehot[i, 2] = 1
                elif theta_input[i, -1] == -1:
                    theta_onehot[i, 3] = 1
                else:
                    raise Exception("Theta encoding is wrong somehow")
            ind_input = theta_onehot
        elif self.theta_space_mode in ['arbitrary', 'arbitrary_stop', 'k_theta', 'k_theta_stop']:
            ind_input = theta_input

        # Forward through actor and critic models
        st = time.time()
        hidden_critic = self.critic_module_net([obs_input, ind_input])
        hidden_critic = self.critic_out_fc(self.tanh(hidden_critic))
        hidden_actor = self.actor_module_net([obs_input, ind_input])
        hidden_actor = self.tanh(hidden_actor)
        #if theta_input.size(0) > 1:
        #    print(theta_input.size(0))
        #    print(time.time()-st)

        return hidden_critic, hidden_actor, states

# BMLP architecture for actor/critic networks
# For envs that have 1D inputs (like most MuJoCo locomotion tasks)
# This version also takes the phase counter and theta as an input
# This version uses the SwitchModule
class SwitchModuleBasePhaseWithStateTheta(nn.Module):
    def __init__(self, num_inputs, opt):
        super(SwitchModuleBasePhaseWithStateTheta, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        # Define phase parameter list
        self.k = opt['phase_period'] 
        assert(opt['switch_mode'] == 'soft') # I'm actually not sure how to do the hard mode anyway
        #assert(opt['theta_space_mode'] == 'simple_four')    # Hardcode to simple four mode
        self.switch_mode = opt['switch_mode']

        # Remove theta from obs input
        self.theta_sz = opt['theta_sz']
        num_inputs -= self.theta_sz
    
        # Decide where inputs belong
        self.theta_space_mode = opt['theta_space_mode']
        if self.theta_space_mode == 'simple_four':
            opt['switch_sz'] = 4
        elif self.theta_space_mode in ['arbitrary', 'arbitrary_stop', 'k_theta', 'k_theta_stop']:
            opt['switch_sz'] = self.theta_sz
        if opt['phase_input_mode'] == 'switch':
            opt['switch_sz'] += self.k 
        elif opt['phase_input_mode'] == 'obs_input':
            self.param_sz = opt['phase_hid_sz']
            self.phase_param = nn.Embedding(self.k, self.param_sz)
            num_inputs += self.param_sz
        else:
            raise NotImplementedError
        self.phase_input_mode = opt['phase_input_mode']

        # Figure out if you want to use timestep as obs
        self.use_timestep = opt['use_timestep']
        self.time_scale = opt['time_scale']
        if not self.use_timestep:
            num_inputs -= 1

        # Figure out hidden size
        self.hid_sz = opt['hid_sz']

        # Define actor and critic network modules
        self.tanh = nn.Tanh()
        self.actor_module_net = SwitchModule(num_inputs, opt)
        self.critic_module_net = SwitchModule(num_inputs, opt)
        self.critic_out_fc = init_(nn.Linear(self.hid_sz, 1))      
 
        # Set to training mode by default
        self.train()

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self.hid_sz

    # Forward through actor critic
    def forward(self, inputs, states, masks):
        # Seperate out the raw observation input from the step count
        if self.use_timestep:
            obs_input = inputs
        else:
            obs_input = inputs[:, :-1]

        # Seperate theta and obs
        theta_input = obs_input[:, -self.theta_sz:]
        obs_input = obs_input[:, :-self.theta_sz]

        # Get the step count
        step_count = torch.round(inputs[:, -1]/self.time_scale)
        assert(abs(round(step_count[0].item()) - step_count[0].item()) < 1e-6)
        step_count = torch.LongTensor(step_count.size()).copy_(step_count)
        if obs_input.is_cuda:
            step_count = step_count.cuda()

        # Get phase input (do mod)
        phase_input = step_count % self.k

        # Get theta input (unhardcode the four encoding)
        if self.theta_space_mode == 'simple_four':
            theta_onehot = torch.zeros(theta_input.size(0), 4)
            if obs_input.is_cuda:
                theta_onehot = theta_onehot.cuda()
            for i in range(theta_input.size(0)):    
                if theta_input[i, 0] == 1:
                    theta_onehot[i, 0] = 1
                elif theta_input[i, 0] == -1:
                    theta_onehot[i, 1] = 1
                elif theta_input[i, -1] == 1:
                    theta_onehot[i, 2] = 1
                elif theta_input[i, -1] == -1:
                    theta_onehot[i, 3] = 1
                else:
                    raise Exception("Theta encoding is wrong somehow")
            theta_input = theta_onehot

        # Depending on mode, make phase_input either the parameter embedding or onehot and add to appropriate place
        assert(self.switch_mode == 'soft')
        # Put phase as onehot and append to ind_input
        if self.phase_input_mode == 'switch':
            phase_input = phase_input.reshape([-1, 1])
            phase_onehot = torch.zeros(phase_input.size(0), self.k)
            if obs_input.is_cuda:
                phase_onehot = phase_onehot.cuda()
            phase_onehot.scatter_(1, phase_input, 1)
            phase_input = phase_onehot
            ind_input = torch.cat([phase_input, theta_input], 1)
            net_input = obs_input
        # Get lookup table for theta and append to net_input
        elif self.phase_input_mode == 'obs_input':
            phase_input = self.phase_param(phase_input)
            ind_input = theta_input
            net_input = torch.cat([obs_input, phase_input], 1)
        else:
            raise NotImplementedError

        # Forward through actor and critic models
        hidden_critic = self.critic_module_net([net_input, ind_input])
        hidden_critic = self.critic_out_fc(self.tanh(hidden_critic))
        hidden_actor = self.actor_module_net([net_input, ind_input])
        hidden_actor = self.tanh(hidden_actor)

        return hidden_critic, hidden_actor, states

# This module implements a hard or soft network switch
# It takes a network input and a hard or soft selection input
# Then it runs the input through a network(s) depending on the selection
# And finally (for soft) combined the output
class SwitchModule(nn.Module):
    def __init__(self, input_sz, opt):
        super(SwitchModule, self).__init__()

        # Hard or soft mode
        self.mode = opt['switch_mode']        

        # Make whatever our modules are
        self.input_sz = input_sz
        self.num_modules = opt['num_modules']
        assert(opt['module']['module_type'] == 'MLP')   # Right now it's an MLP only
        self.hid_sz = opt['hid_sz']
        num_layer = opt['module']['num_layer']
        self.batch_modules = BatchMLP(self.input_sz, self.hid_sz, num_layer, self.num_modules)

        # Make soft attention network components (if applicible)
        if self.mode == 'soft':
            self.switch_sz = opt['switch_sz']
            self.att_in = nn.Linear(self.switch_sz, self.num_modules)
            self.softmax = nn.Softmax()

    # Forward (mainly switch between soft and hard modes)
    def forward(self, inputs):
        net_inputs = inputs[0]

        # Compute batch module output
        st = time.time()
        batch_inputs = net_inputs.unsqueeze(1).expand([-1, self.num_modules, -1])
        module_outs = self.batch_modules(batch_inputs) # module_outs is bs x nm x out_sz

        # Either soft or hard mask these
        st = time.time()
        if self.mode == 'hard':
            assert(False)
            switch_idx = inputs[1]
            switch_idx = switch_idx.reshape([-1, 1]).unsqueeze(2)
            mask = torch.zeros(module_outs.size(0), 1, self.num_modules)
            if module_outs.is_cuda:
                mask = mask.cuda()
            mask.scatter_(1, switch_idx, 1)
            mask = mask.repeat([1, 1, module_outs.size(2)])
            module_outs *= mask
            module_outs = module_outs.sum(1)
        elif self.mode == 'soft':
            switch_input = inputs[1]
            selection = self.softmax(self.att_in(switch_input))
            selection = selection.unsqueeze(2)
            selection = selection.repeat([1, 1, module_outs.size(2)])
            module_outs *= selection
            module_outs = module_outs.sum(1)
        else:
            raise NotImplementedError

        return module_outs

# Simple MLP module
class SimpleMLP(nn.Module):
    def __init__(self, input_sz, opt):
        super(SimpleMLP, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        # Get params
        hid_sz = opt['hid_sz']
        num_layer = opt['num_layer']

        # Make network
        self.tanh = nn.Tanh()
        self.in_fc = init_(nn.Linear(input_sz, hid_sz))
        assert(num_layer >= 2) # If num_layer is 2, actually no hidden layers technically
        hid_layers = []
        for i in range(0, num_layer-2):
            hid_fc = init_(nn.Linear(hid_sz, hid_sz))
            hid_layers.append(hid_fc)
        self.hid_layers = ListModule(*hid_layers)
        self.out_fc = init_(nn.Linear(hid_sz, hid_sz))

    def forward(self, input):
        x = self.in_fc(input) 
        x = self.tanh(x)
        for hid_fc in self.hid_layers:
            x = hid_fc(x)
            x = self.tanh(x)
        x = self.out_fc(x)
        return x

# Base bilinear architecture for actor/critic networks
# For envs that have 1D inputs (like most MuJoCo locomotion tasks)
class MLPBaseBilinearTheta(nn.Module):
    def __init__(self, num_inputs, opt):
        super(MLPBaseBilinearTheta, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        # Get sizes 
        self.theta_sz = opt['theta_sz']
        self.obs_sz = num_inputs - self.theta_sz
        self.hid_sz = opt['hid_sz']
        self.time_scale = opt['time_scale']

        # If in a phase mode, add phase params
        self.mode = opt['mode']  
        assert(self.mode in ['baseline_lowlevel', 'phase_lowlevel'])  
        if self.mode == 'phase_lowlevel':
            # Define phase parameter list
            self.obs_sz -= 1 # Remove step from input count
            self.k = opt['phase_period']
            self.phase_param = nn.Embedding(self.k, self.obs_sz)
            self.cat_input_sz = 2*self.obs_sz
        else:
            self.cat_input_sz = self.obs_sz
        
        # Create linear recombination layer for theta
        self.theta_emb = nn.Linear(self.theta_sz, self.obs_sz)

        # Get number of layers
        self.num_layer = opt['num_layer']

        # Define actor network modules
        self.tanh = nn.Tanh()
        self.actor_in_fc = init_(nn.Bilinear(self.obs_sz, self.cat_input_sz, self.hid_sz))

        # Define critic network modules
        self.critic_in_fc = init_(nn.Bilinear(self.obs_sz, self.cat_input_sz, self.hid_sz))
        self.critic_out_fc = init_(nn.Linear(self.hid_sz+self.cat_input_sz, 1))

        # Depending on num_layer, add more layers
        assert(self.num_layer >= 2)
        actor_hid_layers = []
        critic_hid_layers = []
        for i in range(0, self.num_layer-1):
            actor_hid_fc = init_(nn.Bilinear(self.hid_sz, self.cat_input_sz, self.hid_sz))
            actor_hid_layers.append(actor_hid_fc)
            critic_hid_fc = init_(nn.Bilinear(self.hid_sz, self.cat_input_sz, self.hid_sz))
            critic_hid_layers.append(critic_hid_fc)
        self.actor_hid_layers = ListModule(*actor_hid_layers)
        self.critic_hid_layers = ListModule(*critic_hid_layers)

        # Special case for forward (always make theta input zeros)
        if opt['theta_space_mode'] == 'forward':
            self.set_theta_zero = True
        else:
            self.set_theta_zero = False

        # Set to training mode by default
        self.train()

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self.hid_sz + self.cat_input_sz

    # Forward through actor critic
    def forward(self, inputs, states, masks):
        # Seperate out the raw observation input, step count, and theta
        ac_inputs = []
        if self.mode == 'phase_lowlevel':
            # Seperate out obs and step count
            obs_input = inputs[:, :-1]
        
            # Get step count
            time_mult = 1 / self.time_scale
            step_count = torch.round(inputs[:, -1]*time_mult)
            assert(abs(round(step_count[0].item()) - step_count[0].item()) < 1e-6)
            step_count = torch.LongTensor(step_count.size()).copy_(step_count)
            if obs_input.is_cuda:
                step_count = step_count.cuda()

            # Get phase input
            phase_input = step_count % self.k
            phase_input = self.phase_param(phase_input)
            ac_inputs.append(phase_input)
        else:
            obs_input = inputs

        # Seperate out theta
        theta_input = obs_input[:, -self.theta_sz:]
        theta_input = self.theta_emb(theta_input)
        if self.set_theta_zero:    
            theta_input.zero_()
        obs_input = obs_input[:, :-self.theta_sz]
        ac_inputs.append(theta_input)
        ac_inputs.append(obs_input)

        # Forward through actor and critic models
        hidden_critic = self.critic_forward(ac_inputs)
        hidden_actor = self.actor_forward(ac_inputs)

        return hidden_critic, hidden_actor, states

    # Forward through actor network
    def critic_forward(self, ac_inputs):
        # Seperate obs and extra inputs
        extra_inputs = torch.cat(ac_inputs[:-1], 1)
        obs_input = ac_inputs[-1]

        # Get initial input and forward through input layer
        #x = torch.cat(ac_inputs, 1)
        x = self.critic_in_fc(obs_input, extra_inputs)
        x = self.tanh(x)

        # Go through each hidden layer
        for critic_hid_fc in self.critic_hid_layers:
            x = critic_hid_fc(x, extra_inputs)
            x = self.tanh(x)
        x = torch.cat([x, extra_inputs], 1)
        x = self.critic_out_fc(x)
        return x

    # Forward through actor network
    def actor_forward(self, ac_inputs):
        # Seperate obs and extra inputs
        extra_inputs = torch.cat(ac_inputs[:-1], 1)
        obs_input = ac_inputs[-1]

        # Get initial input and forward through input layer
        x = self.actor_in_fc(obs_input, extra_inputs)
        x = self.tanh(x)

        # Go through each hidden layer 
        for actor_hid_fc in self.actor_hid_layers:
            x = actor_hid_fc(x, extra_inputs)
            x = self.tanh(x)
        x = torch.cat([x, extra_inputs], 1) 
        return x

# BMLP architecture for actor/critic networks
# For envs that have 1D inputs (like most MuJoCo locomotion tasks)
# This version also takes the theta as an input
# This version uses the SwitchModule
class SwitchModuleBaseTheta(nn.Module):
    def __init__(self, num_inputs, opt):
        super(SwitchModuleBaseTheta, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        # Figure out sizes
        self.theta_sz = opt['theta_sz']
        self.obs_sz = num_inputs - self.theta_sz
        self.hid_sz = opt['hid_sz']
        opt['switch_sz'] = self.theta_sz
        opt['switch_mode'] = 'soft'
        self.time_scale = opt['time_scale']

        # If in a phase mode, add phase params
        self.mode = opt['mode']  
        assert(self.mode in ['baseline_lowlevel', 'phase_lowlevel'])  
        if self.mode == 'phase_lowlevel':
            # Define phase parameter list
            self.obs_sz -= 1 # Remove step from input count
            self.k = opt['phase_period']
            self.phase_param = nn.Embedding(self.k, self.obs_sz)
            self.input_sz = 2*self.obs_sz
        else:
            self.input_sz = self.obs_sz

        # Figure out hidden size
        # Define actor and critic network modules
        self.tanh = nn.Tanh()
        self.actor_module_net = SwitchModule(self.input_sz, opt)
        self.critic_module_net = SwitchModule(self.input_sz, opt)
        self.critic_out_fc = init_(nn.Linear(self.hid_sz, 1))      

        # Set to training mode by default
        self.train()

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self.hid_sz

    # Forward through actor critic
    def forward(self, inputs, states, masks):
        # Seperate out the raw observation input, step count, and theta
        if self.mode == 'phase_lowlevel':
            # Seperate out obs and step count
            obs_input = inputs[:, :-1]
        
            # Get step count
            time_mult = 1 / self.time_scale
            step_count = torch.round(inputs[:, -1]*time_mult)
            assert(abs(round(step_count[0].item()) - step_count[0].item()) < 1e-6)
            step_count = torch.LongTensor(step_count.size()).copy_(step_count)
            if obs_input.is_cuda:
                step_count = step_count.cuda()

            # Get phase input
            phase_input = step_count % self.k
            phase_input = self.phase_param(phase_input)            
        else:
            obs_input = inputs

        # Seperate out theta
        theta_input = obs_input[:, -self.theta_sz:]
        obs_input = obs_input[:, :-self.theta_sz]
      
        # Get net input
        if self.mode == 'phase_lowlevel':
            net_input = torch.cat([obs_input, phase_input], 1)
        else:
            net_input = obs_input

        # Forward through actor and critic models
        st = time.time()
        hidden_critic = self.critic_module_net([net_input, theta_input])
        hidden_critic = self.critic_out_fc(self.tanh(hidden_critic))
        hidden_actor = self.actor_module_net([net_input, theta_input])
        hidden_actor = self.tanh(hidden_actor)

        return hidden_critic, hidden_actor, states

# Base MLP architecture for actor/critic networks
# For envs that have 1D inputs (like most MuJoCo locomotion tasks)
# This network is hardcoded for Deepmind's humanoid network with hidden sizes
# 300, 200, 100
class MLPBaseDeepMindPretrain(nn.Module):
    def __init__(self, num_inputs, opt):
        super(MLPBaseDeepMindPretrain, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        # Get sizes 
        self.obs_sz = num_inputs
        self.time_scale = opt['time_scale']

        # If in a phase mode, add phase params
        self.mode = opt['mode']  
        assert(self.mode in ['baseline_lowlevel', 'phase_lowlevel'])  
        if self.mode == 'phase_lowlevel':
            # Define phase parameter list
            self.obs_sz -= 1 # Remove step from input count
            self.k = opt['phase_period']
            self.phase_hid_sz = opt['phase_hid_sz']
            self.phase_param = nn.Embedding(self.k, self.phase_hid_sz)
            self.input_sz = self.phase_hid_sz + self.obs_sz
        else:
            self.input_sz = self.obs_sz

        # Define actor network modules
        self.tanh = nn.Tanh()
        self.actor_in_fc = init_(nn.Linear(self.input_sz, 300))
        self.actor_hid1_fc = init_(nn.Linear(300, 200))
        self.actor_hid2_fc = init_(nn.Linear(200, 100))

        # Define critic network modules
        self.critic_in_fc = init_(nn.Linear(self.input_sz, 300))
        self.critic_hid1_fc = init_(nn.Linear(300, 200))
        self.critic_hid2_fc = init_(nn.Linear(200, 100))
        self.critic_out_fc = init_(nn.Linear(100, 1))

        # Set to training mode by default
        self.train()

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 100

    # Forward through actor critic
    def forward(self, inputs, states, masks):
        # Seperate out the raw observation input, step count, and theta
        ac_inputs = []
        if self.mode == 'phase_lowlevel':
            # Seperate out obs and step count
            obs_input = inputs[:, :-1]
        
            # Get step count
            time_mult = 1 / self.time_scale
            step_count = torch.round(inputs[:, -1]*time_mult)
            assert(abs(round(step_count[0].item()) - step_count[0].item()) < 1e-6)
            step_count = torch.LongTensor(step_count.size()).copy_(step_count)
            if obs_input.is_cuda:
                step_count = step_count.cuda()

            # Get phase input
            phase_input = step_count % self.k
            phase_input = self.phase_param(phase_input)
            ac_inputs.append(phase_input)
        else:
            obs_input = inputs
        ac_inputs.append(obs_input)

        # Forward through actor and critic models
        hidden_critic = self.critic_forward(ac_inputs)
        hidden_actor = self.actor_forward(ac_inputs)

        return hidden_critic, hidden_actor, states

    # Forward through actor network
    def critic_forward(self, ac_inputs):
        #pdb.set_trace()
        # Get initial input and forward through input layer
        x = torch.cat(ac_inputs, 1)
        x = self.critic_in_fc(x)
        x = self.tanh(x)
        x = self.critic_hid1_fc(x)
        x = self.tanh(x)
        x = self.critic_hid2_fc(x)
        x = self.tanh(x)        
        x = self.critic_out_fc(x)
        return x

    # Forward through actor network
    def actor_forward(self, ac_inputs):
        #pdb.set_trace()
        # Get initial input and forward through input layer
        x = torch.cat(ac_inputs, 1)
        x = self.actor_in_fc(x)
        x = self.tanh(x)
        x = self.actor_hid1_fc(x)
        x = self.tanh(x)
        x = self.actor_hid2_fc(x)
        x = self.tanh(x)
        return x

# Base MLP architecture for actor/critic networks
# For envs that have 1D inputs (like most MuJoCo locomotion tasks)
class MLPBasePretrain(nn.Module):
    def __init__(self, num_inputs, opt):
        super(MLPBasePretrain, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        # Get sizes 
        self.obs_sz = num_inputs
        self.hid_sz = opt['hid_sz']
        self.time_scale = opt['time_scale']
        self.skip_layer = opt['skip_layer']
        self.num_layer = opt['num_layer']

        # If in a phase mode, add phase params
        self.mode = opt['mode']  
        assert(self.mode in ['baseline_lowlevel', 'phase_lowlevel'])  
        if self.mode == 'phase_lowlevel':
            # Define phase parameter list
            self.obs_sz -= 1 # Remove step from input count
            self.k = opt['phase_period']
            self.phase_hid_sz = opt['phase_hid_sz']
            self.phase_param = nn.Embedding(self.k, self.phase_hid_sz)
            self.input_sz = self.phase_hid_sz + self.obs_sz
        else:
            self.input_sz = self.obs_sz

        # Define actor network modules
        self.tanh = nn.Tanh()
        self.actor_in_fc = init_(nn.Linear(self.input_sz, self.hid_sz))

        # Define critic network modules
        self.critic_in_fc = init_(nn.Linear(self.input_sz, self.hid_sz))
        self.critic_out_fc = init_(nn.Linear(self.hid_sz, 1))

        # Depending on num_layer, add more layers
        if self.num_layer >= 2:
            actor_hid_layers = []
            critic_hid_layers = []
            for i in range(0, self.num_layer-1):
                actor_hid_fc = init_(nn.Linear(self.hid_sz, self.hid_sz))
                actor_hid_layers.append(actor_hid_fc)
                critic_hid_fc = init_(nn.Linear(self.hid_sz, self.hid_sz))
                critic_hid_layers.append(critic_hid_fc)
            self.actor_hid_layers = ListModule(*actor_hid_layers)
            self.critic_hid_layers = ListModule(*critic_hid_layers)
        else:
            assert(self.num_layer == 1)
            self.actor_hid_layers = []
            self.critic_hid_layers = []

        # Set to training mode by default
        self.train()

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self.hid_sz

    # Forward through actor critic
    def forward(self, inputs, states, masks):
        # Seperate out the raw observation input, step count, and theta
        ac_inputs = []
        if self.mode == 'phase_lowlevel':
            # Seperate out obs and step count
            obs_input = inputs[:, :-1]
        
            # Get step count
            time_mult = 1 / self.time_scale
            step_count = torch.round(inputs[:, -1]*time_mult)
            assert(abs(round(step_count[0].item()) - step_count[0].item()) < 1e-6)
            step_count = torch.LongTensor(step_count.size()).copy_(step_count)
            if obs_input.is_cuda:
                step_count = step_count.cuda()

            # Get phase input
            phase_input = step_count % self.k
            phase_input = self.phase_param(phase_input)
            ac_inputs.append(phase_input)
        else:
            obs_input = inputs
        ac_inputs.append(obs_input)

        # Forward through actor and critic models
        hidden_critic = self.critic_forward(ac_inputs)
        hidden_actor = self.actor_forward(ac_inputs)

        return hidden_critic, hidden_actor, states

    # Forward through actor network
    def critic_forward(self, ac_inputs):
        #pdb.set_trace()
        # Get initial input and forward through input layer
        x = torch.cat(ac_inputs, 1)
        x = self.critic_in_fc(x)
        x = self.tanh(x)
        all_x = [x]

        # Go through each hidden layer
        for critic_hid_fc in self.critic_hid_layers:
            if self.skip_layer:
                x = sum(all_x)
            x = critic_hid_fc(x)
            x = self.tanh(x)
            all_x.append(x)
        if self.skip_layer:
            x = sum(all_x)
        x = self.critic_out_fc(x)
        return x

    # Forward through actor network
    def actor_forward(self, ac_inputs):
        #pdb.set_trace()
        # Get initial input and forward through input layer
        x = torch.cat(ac_inputs, 1)
        x = self.actor_in_fc(x)
        x = self.tanh(x)
        all_x = [x]

        # Go through each hidden layer 
        for actor_hid_fc in self.actor_hid_layers:
            if self.skip_layer:
                x = sum(all_x)
            x = actor_hid_fc(x)
            x = self.tanh(x)
            all_x.append(x)
        if self.skip_layer:
            x = sum(all_x)
        return x

    # Load special (copy with different input weights)
    def update_state_dict_special(self, ckpt, oldinput_mask):
        # Edit 'base.actor.0.weight' and 'base.critic.0.weight'
        new_actor_in = torch.Tensor(self.actor_in_fc.weight)
        new_critic_in = torch.Tensor(self.critic_in_fc.weight)

        # Splice in corresponding part of checkpoint
        oldinput_mask = torch.cat([torch.Tensor(oldinput_mask.astype(int)), torch.Tensor(np.ones(self.phase_hid_sz).astype(int))]) == True
        new_actor_in[:, oldinput_mask].copy_(ckpt['base.actor_in_fc.weight'])
        new_critic_in[:, oldinput_mask].copy_(ckpt['base.critic_in_fc.weight'])
        ckpt['base.actor_in_fc.weight'] = new_actor_in
        ckpt['base.critic_in_fc.weight'] = new_critic_in 

        # Return new ckpt
        return ckpt



# Fix or remove before release
# Base MLP architecture for actor/critic networks
# For envs that have 1D inputs (like most MuJoCo locomotion tasks)
class MLPBaseTheta(nn.Module):
    def __init__(self, num_inputs, opt):
        super(MLPBaseTheta, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        # Get sizes 
        self.theta_sz = opt['theta_sz']
        self.obs_sz = num_inputs - self.theta_sz
        self.hid_sz = opt['hid_sz']
        self.time_scale = opt['time_scale']

        # If in a phase mode, add phase params
        self.mode = opt['mode']  
        assert(self.mode in ['baseline_lowlevel', 'phase_lowlevel'])  
        if self.mode == 'phase_lowlevel':
            # Define phase parameter list
            self.obs_sz -= 1 # Remove step from input count
            self.k = opt['phase_period']
            self.phase_param = nn.Embedding(self.k, self.obs_sz)
            self.cat_input_sz = 2*self.obs_sz
        else:
            self.cat_input_sz = self.obs_sz
        self.input_sz = self.obs_sz + self.cat_input_sz 

        # Create linear recombination layer for theta
        self.theta_emb = nn.Linear(self.theta_sz, self.obs_sz)

        # Get number of layers
        self.num_layer = opt['num_layer']

        # Define actor network modules
        self.tanh = nn.Tanh()
        self.actor_in_fc = init_(nn.Linear(self.input_sz, self.hid_sz))

        # Define critic network modules
        self.critic_in_fc = init_(nn.Linear(self.input_sz, self.hid_sz))
        self.critic_out_fc = init_(nn.Linear(self.hid_sz+self.cat_input_sz, 1))

        # Depending on num_layer, add more layers
        assert(self.num_layer >= 2)
        actor_hid_layers = []
        critic_hid_layers = []
        for i in range(0, self.num_layer-1):
            actor_hid_fc = init_(nn.Linear(self.hid_sz+self.cat_input_sz, self.hid_sz))
            actor_hid_layers.append(actor_hid_fc)
            critic_hid_fc = init_(nn.Linear(self.hid_sz+self.cat_input_sz, self.hid_sz))
            critic_hid_layers.append(critic_hid_fc)
        self.actor_hid_layers = ListModule(*actor_hid_layers)
        self.critic_hid_layers = ListModule(*critic_hid_layers)

        # Special case for forward (always make theta input zeros)
        if opt['theta_space_mode'] == 'forward':
            self.set_theta_zero = True
        else:
            self.set_theta_zero = False

        # Set to training mode by default
        self.train()

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self.hid_sz + self.cat_input_sz

    # Forward through actor critic
    def forward(self, inputs, states, masks):
        # Seperate out the raw observation input, step count, and theta
        ac_inputs = []
        if self.mode == 'phase_lowlevel':
            # Seperate out obs and step count
            obs_input = inputs[:, :-1]
        
            # Get step count
            time_mult = 1 / self.time_scale
            step_count = torch.round(inputs[:, -1]*time_mult)
            assert(abs(round(step_count[0].item()) - step_count[0].item()) < 1e-6)
            step_count = torch.LongTensor(step_count.size()).copy_(step_count)
            if obs_input.is_cuda:
                step_count = step_count.cuda()

            # Get phase input
            phase_input = step_count % self.k
            phase_input = self.phase_param(phase_input)
            ac_inputs.append(phase_input)
        else:
            obs_input = inputs

        # Seperate out theta
        theta_input = obs_input[:, -self.theta_sz:]
        theta_input = self.theta_emb(theta_input)
        if self.set_theta_zero:    
            theta_input.zero_()
        obs_input = obs_input[:, :-self.theta_sz]
        ac_inputs.append(theta_input)
        ac_inputs.append(obs_input)

        # Forward through actor and critic models
        hidden_critic = self.critic_forward(ac_inputs)
        hidden_actor = self.actor_forward(ac_inputs)

        return hidden_critic, hidden_actor, states

    # Forward through actor network
    def critic_forward(self, ac_inputs):
        # Seperate obs and extra inputs
        extra_inputs = ac_inputs[:-1]
        obs_input = ac_inputs[-1]

        # Get initial input and forward through input layer
        x = torch.cat(ac_inputs, 1)
        x = self.critic_in_fc(x)
        x = self.tanh(x)

        # Go through each hidden layer
        for critic_hid_fc in self.critic_hid_layers:
            x = torch.cat([x] + extra_inputs, 1)    
            x = critic_hid_fc(x)
            x = self.tanh(x)
        x = torch.cat([x] + extra_inputs, 1)
        x = self.critic_out_fc(x)
        return x

    # Forward through actor network
    def actor_forward(self, ac_inputs):
        # Seperate obs and extra inputs
        extra_inputs = ac_inputs[:-1]
        obs_input = ac_inputs[-1]

        # Get initial input and forward through input layer
        x = torch.cat(ac_inputs, 1)
        x = self.actor_in_fc(x)
        x = self.tanh(x)

        # Go through each hidden layer 
        for actor_hid_fc in self.actor_hid_layers:
            x = torch.cat([x] + extra_inputs, 1)    
            x = actor_hid_fc(x)
            x = self.tanh(x)
        x = torch.cat([x] + extra_inputs, 1) 
        return x

# Batch MLP module
# Basically does a for loop over MLPs, but does this efficiently using bmm
class BatchMLP(nn.Module):
    def __init__(self, input_sz, hid_sz, num_layer, num_modules):
        super(BatchMLP, self).__init__()
        
        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))
       
        # Make network values
        self.tanh = nn.Tanh()
        self.in_fc = init_(BatchLinear(input_sz, hid_sz, num_modules))
        assert(num_layer >= 2) # If num_layer is 2, actually no hidden layers technically
        hid_layers = []
        for i in range(0, num_layer-2):
            hid_fc = init_(BatchLinear(hid_sz, hid_sz, num_modules))
            hid_layers.append(hid_fc)
        self.hid_layers = ListModule(*hid_layers)
        self.out_fc = init_(BatchLinear(hid_sz, hid_sz, num_modules))

    # Input batch_size x num_modules x input_sz
    # Output batch_size x num_modules x output_sz
    def forward(self, input):
        x = self.in_fc(input)
        x = self.tanh(x)
        for hid_fc in self.hid_layers:
            x = hid_fc(x)
            x = self.tanh(x)
        x = self.out_fc(x)
        return x

# BatchLinear module
# Same as nn.Linear, but it takes list of inputs x and outputs list of outputs y
# Equivalent to same operation if we did a for loop and did nn.Linear for each
class BatchLinear(nn.Module):
    def __init__(self, in_features, out_features, num_modules, bias=True):
        super(BatchLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_modules = num_modules
        self.weight = nn.Parameter(torch.Tensor(num_modules, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_modules, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.W = None
        self.b = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # Input batch_size x num_module x input_sz
    # Output batch_size x num_module x output_sz
    def forward(self, input):
        # Get sizes
        bs = input.size(0)
        nm = input.size(1)
        assert(input.size(2) == self.in_features)

        # Reshape to proper matrices
        if self.W is None or self.W.size(0) != bs:
            self.W = self.weight.unsqueeze(0).expand([bs, -1, -1, -1]).contiguous().view(nm*bs, self.out_features, self.in_features)  
        input = input.contiguous().view(nm*bs, self.in_features, 1)

        # Compute matrix multiply and add bias (if applicable)
        output = torch.bmm(self.W, input) 
        if self.bias is not None:
            if self.b is None or self.b.size(0) != bs:
                self.b = self.bias.unsqueeze(0).expand([bs, -1, -1]).contiguous().view(nm*bs, self.out_features, 1)
            output += self.b 

        # Reshape output
        output = output.view(bs, nm, self.out_features)
        return output 

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
       
# Use ListModule code from fmassa (shouldn't this be in pytorch already?)
class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

# Helper function that copies from list of nn.Linear to BatchLinear
def copy_linear(list_linear, batch_linear):
    list_weights = torch.cat([l.weight.unsqueeze(0) for l in list_linear], 0)
    list_biases = torch.cat([l.bias.unsqueeze(0) for l in list_linear], 0)
    batch_linear.weight.data.copy_(list_weights)
    batch_linear.bias.data.copy_(list_biases)

# Main function - does a sanity check that BatchMLP and BatchLinear behave the same as SimpleMLP and nn.Linear in a for loop
def main():
    # Test BatchLinear
    in_sz = 10
    out_sz = 10
    num_modules = 4
    list_linear = [nn.Linear(in_sz, out_sz) for i in range(num_modules)]
    batch_linear = BatchLinear(in_sz, out_sz, num_modules)
    copy_linear(list_linear, batch_linear)

    # Forward pass
    x = torch.Tensor(2, 10).normal_()
    batch_x = x.unsqueeze(1).expand([-1, num_modules, -1])
    list_outputs = torch.cat([lin(x).unsqueeze(1) for lin in list_linear], 1)
    batch_outputs = batch_linear(batch_x)
    print(torch.abs(list_outputs - batch_outputs).sum()) 
    assert(torch.abs(list_outputs - batch_outputs).sum() < 1e-5)
    
    # Test BatchMLP
    num_layer = 2
    opt = {}
    opt['hid_sz'] = out_sz
    opt['num_layer'] = num_layer
    list_mlp = [SimpleMLP(in_sz, opt) for i in range(num_modules)]
    batch_mlp = BatchMLP(in_sz, out_sz, num_layer, num_modules)
    copy_linear([mlp.in_fc for mlp in list_mlp], batch_mlp.in_fc)
    copy_linear([mlp.out_fc for mlp in list_mlp], batch_mlp.out_fc)
    list_outputs = torch.cat([mlp(x).unsqueeze(1) for mlp in list_mlp], 1)
    batch_outputs = batch_mlp(batch_x)
    print(torch.abs(list_outputs - batch_outputs).sum()) 
    assert(torch.abs(list_outputs - batch_outputs).sum() < 1e-5)

if __name__ == "__main__":
    main() 
