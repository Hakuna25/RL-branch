import torch.nn as nn
import torch.nn.functional as F

# 2. Define the network used in both target net and the net for training
class PolicyNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super(PolicyNet, self).__init__()
        # Define the structure of fully connected network
        self.hidden_size = 128
        self.fc1 = nn.Linear(n_states, self.hidden_size)  # layer 1
        self.out = nn.Linear(self.hidden_size, n_actions) # layer 2
          
    def forward(self, x):
        # Define how the input data pass inside the network
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return F.softmax(actions_value, dim=1)


class ValueNet(nn.Module):
    def __init__(self, n_states):
        super(ValueNet, self).__init__()
        # Define the structure of fully connected network
        self.hidden_size = 128
        self.fc1 = nn.Linear(n_states, self.hidden_size)  # layer 1
        self.out = nn.Linear(self.hidden_size, 1) # layer 2
        
    def forward(self, x):
        # Define how the input data pass inside the network
        x = self.fc1(x)
        x = F.relu(x)
        value = self.out(x)
        return value