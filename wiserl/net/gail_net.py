import torch.nn as nn
import torch
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, 128)
        self.fc2 = torch.nn.Linear(128, 1)
    
    def forward(self, state, action):
        cat = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(cat))
        return torch.sigmoid(self.fc2(x))
