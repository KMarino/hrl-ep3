import torch
import torch.nn as nn
import torch.optim as optim

# Does no updates, just does forward passes
class Passthrough(object):
    def __init__(self, actor_critic): 
        self.actor_critic = actor_critic

    # Generate a state_dict object
    def state_dict(self):
        ckpt = {}
        ckpt['model'] = self.actor_critic.state_dict()
        return ckpt

    # Load from a state dict
    def load_state_dict(self, ckpt):
        self.actor_critic.load_state_dict(ckpt['model'])

    # Load from pretrained (ModularPolicy)
    def load_pretrained_policies(self, ckpts):
        self.actor_critic.load_pretrained_policies(ckpts)

    # Update our policy network
    def update(self, rollouts):
        return 0, 0, 0

