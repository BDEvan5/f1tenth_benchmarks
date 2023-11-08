import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NN_LAYER_1 = 100
NN_LAYER_2 = 100
# NN_LAYER_1 = 400
# NN_LAYER_2 = 300


MEMORY_SIZE = 100000

class OffPolicyBuffer(object):
    def __init__(self, state_dim, action_dim):     
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0

        self.states = np.empty((MEMORY_SIZE, state_dim))
        self.actions = np.empty((MEMORY_SIZE, action_dim))
        self.next_states = np.empty((MEMORY_SIZE, state_dim))
        self.rewards = np.empty((MEMORY_SIZE, 1))
        self.dones = np.empty((MEMORY_SIZE, 1))

    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr += 1
        
        if self.ptr == MEMORY_SIZE: self.ptr = 0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.ptr-1, size=batch_size)
        states = np.empty((batch_size, self.state_dim))
        actions = np.empty((batch_size, self.action_dim))
        next_states = np.empty((batch_size, self.state_dim))
        rewards = np.empty((batch_size, 1))
        dones = np.empty((batch_size, 1))

        for i, j in enumerate(ind): 
            states[i] = self.states[j]
            actions[i] = self.actions[j]
            next_states[i] = self.next_states[j]
            rewards[i] = self.rewards[j]
            dones[i] = self.dones[j]
            
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(1- dones)

        return states, actions, next_states, rewards, dones

    def size(self):
        return self.ptr
   

class DoublePolicyNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(DoublePolicyNet, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_mu = nn.Linear(NN_LAYER_2, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) 
        return mu


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6

class PolicyNetworkSAC(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PolicyNetworkSAC, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, NN_LAYER_1)
        self.linear2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.mean_linear = nn.Linear(NN_LAYER_2, num_actions)
        self.log_std_linear = nn.Linear(NN_LAYER_2, num_actions)
    
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(0, 1) # assumes actions have been normalized to (0,1)
        
        z = mean + std * normal.sample().requires_grad_()
        action = torch.tanh(z)
        log_prob = torch.distributions.Normal(mean, std).log_prob(z) - torch.log(1 - action * action + EPSILON) 
        log_prob = log_prob.sum(-1, keepdim=True)
            
        return action, log_prob
   

class DoubleQNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(DoubleQNet, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + act_dim, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_out = nn.Linear(NN_LAYER_2, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x2 = F.relu(self.fc1(x))
        x3 = F.relu(self.fc2(x2))
        q = self.fc_out(x3)
        return q




if __name__ == "__main__":
    print("Hello World!")
    
    