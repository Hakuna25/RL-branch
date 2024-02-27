import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal
import numpy as np
from wiserl.core.agent import Agent
import wiserl.agent.td3_agent.config as cfg
from wiserl.store.mem_store import ReplayBuffer
import os
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TD3Agent(Agent):
    def __init__(self, Actor, Critic, state_dim, action_dim, action_bound, config=None, sync=True):
        super(TD3Agent, self).__init__(sync)
        self.config = cfg
        if config != None:
            self.config=config
        self.actor, self.actor_target = Actor(state_dim,action_dim,action_bound).to(device), Actor(state_dim,action_dim,action_bound).to(device)
        self.critic, self.critic_target = Critic(state_dim,action_dim).to(device), Critic(state_dim,action_dim).to(device)

        self.replay_buffer = ReplayBuffer(cfg.MEMORY_CAPACITY, state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.num_transition = 0  # pointer of replay buffer
        self.num_training = 1
        self.tau = cfg.tau
        self.policy_noise = cfg.policy_noise
        self.max_action = action_bound
        self.noise_clip = cfg.noise_clip
        self.policy_freq = cfg.policy_freq
        self.discount = cfg.discount

        self.total_it = 0
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        os.makedirs('./TD3_model/', exist_ok=True)

    def choseAction(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state)
        return action.detach().numpy()[0]

    def update(self, s, a, r, s_, d):
        self.replay_buffer.store(s, a, r, s_, d)

        if self.num_training % 500 == 0:
            print("Training ... {} ".format(self.num_training))

        # Sample a random minibatch from the replay buffer
        buffer = self.replay_buffer.sample(cfg.batch_size)
        state = buffer['state'].to(torch.float32).to(device)
        action = buffer['action'].to(torch.float32).to(device)
        reward = buffer['reward'].to(torch.float32).to(device)
        next_state = buffer['next_state'].to(torch.float32).to(device)
        done = buffer['done'].to(torch.float32).to(device)

        with torch.no_grad():
			# Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            a = (~torch.tensor(done, dtype=torch.bool)).numpy()

            target_Q = reward + (~torch.tensor(done, dtype=torch.bool)) * self.discount * target_Q

		# Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # print('critic_loss:', critic_loss)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

		# Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # q1, q2 = self.critic(state, self.actor(state))
			# Compute actor loss
            q1,_ = self.critic(state, self.actor(state))
            actor_loss = -q1.mean()
        
			# Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            with torch.no_grad():
                # Update the frozen target models
                self.update_target(self.actor, self.actor_target, cfg.tau)
                self.update_target(self.critic, self.critic_target, cfg.tau)
                    
        self.num_training += 1
    
    def update_target(self, current_model, target_model, tau):
        for param, target_param in zip(current_model.parameters(), target_model.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)

    def save(self):
        torch.save(self.actor.state_dict(), './TD3_model/actor_net.pth')
        torch.save(self.actor_target.state_dict(), './TD3_model/actor_target_net.pth')
        torch.save(self.critic.state_dict(), './TD3_model/critic_net.pth')
        torch.save(self.critic_target.state_dict(), './TD3_model/critic_target_net.pth')
        print("====================================")
        print("Model has been saved.")
        print("====================================")

    def load(self):
        torch.load(self.actor.state_dict(), './TD3_model/actor_net.pth')
        torch.load(self.actor_target.state_dict(), './TD3_model/actor_target_net.pth')
        torch.load(self.critic.state_dict(), './TD3_model/critic_net.pth')
        torch.load(self.critic_target.state_dict(), './TD3_model/critic_target_net.pth')
        print("====================================")
        print("Model has been loaded.")
        print("====================================")