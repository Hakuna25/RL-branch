import numpy as np
import torch
import collections
import random
def compute_returns(next_value, rewards, dones, gamma=0.997):
    R = next_value.detach().cpu().numpy()
    returns = []
    for step in reversed(range(len(rewards))):
        rate = 1 if dones[step] == False else 0
        R = rewards[step] + gamma * R * rate
        returns.insert(0, R)
    return returns

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    advantage_list = np.array(advantage_list)
    return torch.tensor(advantage_list, dtype=torch.float)