import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal
import numpy as np
from wiserl.core.agent import Agent
import wiserl.agent.bc_agent.config as cfg
from wiserl.store.mem_store import ReplayBuffer
from wiserl.core.rl_utils import compute_advantage
import torch.nn.functional as F

import os

device = torch.device(' cuda' if torch.cuda.is_available() else 'cpu')

class BCAgent(Agent):
    def __init__(self, Actor, state_dim, action_dim, config=None, sync=True):
        super(BCAgent, self).__init__(sync)
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.lr)
    
    def update(self, states, actions):
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions).view(-1,1).to(device)
        log_probs = torch.log(self.actor(states).gather(1, actions))
        bc_loss = torch.mean(-log_probs) # 最大似然估计

        self.actor_optimizer.zero_grad()
        bc_loss.backward()
        self.actor_optimizer.step()

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()


