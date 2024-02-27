import torch
import torch.nn as nn
import numpy as np

class PolicyNet(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax()
        )

    def forward(self, x):
        x = x.view(1, -1)
        return self.fc(x)

class Baseline(nn.Module):

    def __init__(self, input_shape):
        super(Baseline, self).__init__()
        shape = 1
        for dim in input_shape:
            shape *= dim
        self.fc = nn.Sequential(
            nn.Linear(shape, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = x.view(1, -1)
        return self.fc(x)