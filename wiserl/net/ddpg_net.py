import torch
import torch.nn as nn
import numpy as np

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, action_dim)
        self.action_bound = action_bound

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.tanh(self.fc2(x)) * self.action_bound
        return x

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], 1)))
        x = self.fc2(x)
        return x