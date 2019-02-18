import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pdb
from storage import ReplayMemory, Transition

# Code copied and adapted from pytorch Q learning tutorial https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class DQN(object):
    def __init__(self,
                 dqn,
                 gamma,
                 batch_size=128,
                 target_update=100,
                 mem_capacity=10000000,
                 lr=None,
                 eps=None,
                 max_grad_norm=1):

        self.gamma = gamma
        self.dqn = dqn
        self.batch_size = batch_size
        self.target_update = target_update
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(self.dqn.policy_net.parameters(), lr=lr, eps=eps)
        self.num_updates = 0
        self.replay_memory = ReplayMemory(mem_capacity)

    # Generate a state_dict object
    def state_dict(self):
        ckpt = {}
        ckpt['model'] = [self.dqn.policy_net.state_dict(), self.dqn.target_net.state_dict()]
        ckpt['optim'] = self.optimizer.state_dict()
        ckpt['steps_done'] = self.dqn.steps_done
        ckpt['num_updates'] = self.num_updates
        ckpt['memory'] = self.replay_memory
        return ckpt

    # Load from a state dict
    def load_state_dict(self, ckpt):
        self.dqn.policy_net.load_state_dict(ckpt['model'][0])
        self.dqn.target_net.load_state_dict(ckpt['model'][1])
        self.optimizer.load_state_dict(ckpt['optim'])
        self.dqn.steps_done = ckpt['steps_done']
        self.num_updates = ckpt['num_updates']
        self.replay_memory = ckpt['memory']

    # Update the replay memory
    def update_memory(self, states, actions, next_states, rewards, done_mask, step_masks):
        # Go through each index (corresponding to different environment steps)
        for state, action, next_state, reward, done, step_mask in zip(states, actions, next_states, rewards, done_mask, step_masks):
            # If in zombie step mask state, do nothing
            if step_mask > 0:
                # Make deep copies, convert to numpy and append to replay memory
                state = np.array(state.cpu().numpy())
                action = np.array(action.cpu().numpy())
                reward = np.array(reward)
                if done:
                    next_state = None
                else:
                    next_state = np.array(next_state.cpu().numpy())

                # Push into replay memory
                self.replay_memory.push(state, action, next_state, reward)

    # Update our policy network
    def update(self, num_updates):
        # Replay memory needs to at least be the batch size
        if len(self.replay_memory) < self.batch_size:
            return 0, 0, 0
        assert(len(self.replay_memory) >= self.batch_size)

        # Do updates
        dqn_loss = 0
        for update in range(num_updates):
            # Get batch values
            transitions = self.replay_memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.uint8)
            non_final_mask = non_final_mask.unsqueeze(1)
            non_final_next_states = torch.cat([torch.from_numpy(s).unsqueeze(0) for s in batch.next_state
                                                if s is not None])
            state_batch = torch.cat([torch.from_numpy(s).unsqueeze(0) for s in batch.state])
            action_batch = torch.cat([torch.from_numpy(a).unsqueeze(0) for a in batch.action])
            reward_batch = torch.cat([torch.from_numpy(r).unsqueeze(0) for r in batch.reward])
            next_state_values = torch.zeros(self.batch_size, 1)
            
            # Convert to cuda 
            if self.dqn.target_net.in_fc.weight.is_cuda:
                non_final_mask = non_final_mask.cuda()
                non_final_next_states = non_final_next_states.cuda()
                state_batch = state_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                next_state_values = next_state_values.cuda()

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken
            state_action_values = self.dqn.policy_net(state_batch).gather(1, action_batch)
            
            # Compute V(s_{t+1}) for all next states.
            next_state_values[non_final_mask] = self.dqn.target_net(non_final_next_states).max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
            dqn_loss += loss           
 
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.dqn.policy_net.parameters():
                param.grad.data.clamp_(-self.max_grad_norm, self.max_grad_norm)
            self.optimizer.step()

            self.num_updates += 1

            # Update target network
            if self.num_updates % self.target_update == 0:
                self.dqn.target_net.load_state_dict(self.dqn.policy_net.state_dict())
    
        dqn_loss /= num_updates
        return dqn_loss, 0, 0 
