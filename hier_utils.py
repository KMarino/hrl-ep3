import torch
import torch.nn as nn
import collections
import numpy as np
import math
import pdb

# Class that handles all the messy hierarchical observation stuff
class HierarchyUtils(object):
    def __init__(self, ll_obs_sz, hl_obs_sz, hl_action_space, theta_sz, add_count):
        self.ll_obs_sz = ll_obs_sz
        if add_count:
            self.ll_raw_obs_sz = [self.ll_obs_sz[0] - theta_sz - 1]
        else:
            self.ll_raw_obs_sz = [self.ll_obs_sz[0] - theta_sz]
        self.hl_obs_sz = hl_obs_sz
        self.theta_sz = theta_sz
        self.hl_action_space = hl_action_space
        self.add_count = add_count

    # Seperate out highlevel, lowlevel and counts
    def seperate_obs(self, obs):
        ll_raw_obs = obs[:, :self.ll_raw_obs_sz[0]]
        assert(ll_raw_obs.shape[-1] == self.ll_raw_obs_sz[0])
        hl_obs = obs[:, self.ll_raw_obs_sz[0]:-1]
        assert(hl_obs.shape[-1] == self.hl_obs_sz[0])
        count = obs[:, -1]
        return hl_obs, ll_raw_obs, count
    
    # Append theta and count to ll obs
    def append_theta(self, ll_raw_obs, hl_action, counts):
        # Get theta
        if self.hl_action_space.__class__.__name__ == 'Discrete':
            assert(self.theta_sz == self.hl_action_space.n)
            thetas = np.zeros([len(hl_action), self.theta_sz])
            for e, act in enumerate(hl_action):
                thetas[e, act] = 1
        else:
            thetas = hl_action

        # Concanetate
        if self.add_count:
            if len(counts.shape) != len(ll_raw_obs.shape):
                counts = np.expand_dims(counts, axis=1)
            ll_obs = np.concatenate([ll_raw_obs, thetas, counts], 1)
        else:
            ll_obs = np.concatenate([ll_raw_obs, thetas], 1)
        assert(ll_obs.shape[-1] == self.ll_obs_sz[0])

        return ll_obs

    # Append placeholder theta and count to ll obs
    def placeholder_theta(self, ll_raw_obs, counts):
        thetas = float('inf') * np.ones([len(ll_raw_obs), self.theta_sz])

        # Concanetate
        if self.add_count:
            if len(counts.shape) != len(ll_raw_obs.shape):
                counts = np.expand_dims(counts, axis=1)
            ll_obs = np.concatenate([ll_raw_obs, thetas, counts], 1)
        else:
            ll_obs = np.concatenate([ll_raw_obs, thetas], 1)
        assert(ll_obs.shape[-1] == self.ll_obs_sz[0])

        return ll_obs

    # Update ll_obs to remove placeholders
    def update_theta(self, ll_obs, hl_action):
        # Take in single obs and high level action and update away the placehodler
        assert(self.has_placeholder(ll_obs))
        assert(ll_obs.shape == self.ll_obs_sz)
        
        # Get theta
        if self.hl_action_space.__class__.__name__ == 'Discrete':
            assert(self.theta_sz == self.hl_action_space.n)
            theta = torch.zeros(self.theta_sz)
            theta[hl_action] = 1
        else:
            theta = torch.from_numpy(hl_action)

        # Update observation with theta
        if self.add_count:
            ll_obs[self.ll_raw_obs_sz[0]:-1] = theta
        else:
            ll_obs[self.ll_raw_obs_sz[0]:] = theta
        assert(not self.has_placeholder(ll_obs))
        return ll_obs

    # Check if ll_obs has a placeholder
    def has_placeholder(self, ll_obs):
        if float('inf') in ll_obs:
            return True
        else:
            return False

