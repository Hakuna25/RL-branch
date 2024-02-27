import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal
import numpy as np
from wiserl.core.agent import Agent
import wiserl.agent.ppo_agent.config as cfg
from wiserl.store.mem_store import ReplayBuffer
from wiserl.core.rl_utils import compute_advantage
import torch.nn.functional as F

import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PPOAgent(Agent):
    def __init__(self, Actor, Critic, state_dim, action_dim, config=None, sync=True):
        super(PPOAgent, self).__init__(sync)
        # self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.gamma = cfg.gamma
        self.lmbda = cfg.lmbda
        self.epochs = cfg.epochs
        self.eps = cfg.eps
        
    def take_action(self, state):
        state = torch.tensor([state],dtype=torch.float).to(device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(device)
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))                  # PPO loss-function
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


        
