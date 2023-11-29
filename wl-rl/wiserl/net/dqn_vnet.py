import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
# 2. Define the network used in both target net and the net for training
class DQNVNet(nn.Module):
    def __init__(self,n_states,n_actions):
        super(DQNVNet, self).__init__()
        # Define the structure of fully connected network
        self.hidden_size = 512
        self.fc1 = nn.Linear(n_states, self.hidden_size)  # layer 1
        self.out_a = nn.Linear(self.hidden_size, n_actions) # layer 2
        self.out_v = nn.Linear(self.hidden_size, 1)
        
    def forward(self, x):
        # Define how the input data pass inside the network
        x = self.fc1(x)
        x = F.relu(x)
        A = self.out_a(x)
        V = self.out_v(x)
        Q = V + A - A.mean(1).view(-1, 1)
        return Q