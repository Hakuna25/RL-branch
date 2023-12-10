import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal
import numpy as np
from wiserl.core.agent import Agent
import wiserl.agent.gail_agent.config as cfg
from wiserl.store.mem_store import ReplayBuffer
from wiserl.core.rl_utils import compute_advantage
import torch.nn.functional as F

import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GAILAgent(Agent):
    def __init__(self, Disc, state_dim, action_dim, config=None, sync=True):
        super(GAILAgent, self).__init__(sync)
        self.discriminator = Disc(state_dim, action_dim).to(device)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=cfg.lr_d)

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s, dones):
        expert_states = torch.tensor(expert_s, dtype=torch.float).to(device)
        expert_actions = torch.tensor(expert_a).to(device)
        agent_states = torch.tensor(agent_s, dtype=torch.float).to(device)
        agent_actions = torch.tensor(agent_a).to(device)
        expert_actions = F.one_hot(expert_actions, num_classes=2).float()
        agent_actions = F.one_hot(agent_actions, num_classes=2).float()

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)
        discriminator_loss = nn.BCELoss()(agent_prob, torch.ones_like(agent_prob)) + nn.BCELoss()(expert_prob, torch.zeros_like(expert_prob))
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        rewards = -torch.log(agent_prob).detach().cpu().numpy()
        transition_dict = {'states': agent_s, 'actions': agent_a, 'rewards': rewards, 'next_states': next_s, 'dones': dones}
        return transition_dict

        
        


