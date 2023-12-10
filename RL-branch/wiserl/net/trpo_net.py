import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return F.softmax(self.fc2(x), dim=1)

# Critic network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)