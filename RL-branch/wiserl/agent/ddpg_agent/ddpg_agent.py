import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal
import numpy as np
from wiserl.core.agent import Agent
import wiserl.agent.ddpg_agent.config as cfg
from wiserl.store.mem_store import ReplayBuffer
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DDPGAgent(Agent):
    def __init__(self, Actor, Critic, state_dim, action_dim, action_bound, config=None, sync=True):
        super(DDPGAgent, self).__init__(sync)
        self.config = config
        if config != None:
            self.config=config
        self.actor, self.actor_target = Actor(state_dim,action_dim,action_bound).to(device), Actor(state_dim,action_dim,action_bound).to(device)
        self.critic, self.critic_target = Critic(state_dim,action_dim).to(device), Critic(state_dim,action_dim).to(device)

        self.replay_buffer = ReplayBuffer(cfg.MEMORY_CAPACITY, state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        
        self.num_transition = 0  # pointer of replay buffer
        self.num_training = 1

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        os.makedirs('./DDPG_model/', exist_ok=True)

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
        state_batch = buffer['state'].to(torch.float32).to(device)
        action_batch = buffer['action'].to(torch.float32).to(device)
        reward_batch = buffer['reward'].to(torch.float32).to(device)
        next_state_batch = buffer['next_state'].to(torch.float32).to(device)
        done_batch = buffer['done'].to(torch.float32).to(device)

        # Compute Q targets
        next_action = self.actor_target(next_state_batch)
        target_Q = self.critic_target(next_state_batch, next_action)
        target_Q = reward_batch + (1 - done_batch) * cfg.gamma * target_Q.detach()

        # Update critic
        current_Q = self.critic(state_batch, action_batch)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        predicted_action = self.actor(state_batch)
        actor_loss = -self.critic(state_batch, predicted_action).mean()
        print('!!!!!!!!!!actor_loss:',actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.update_target(self.actor, self.actor_target, cfg.tau)
        self.update_target(self.critic, self.critic_target, cfg.tau)

        self.num_training += 1

    def update_target(self, current_model, target_model, tau):
        for param, target_param in zip(current_model.parameters(), target_model.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)

    def save(self):
        torch.save(self.actor.state_dict(), './DDPG_model/actor_net.pth')
        torch.save(self.actor_target.state_dict(), './DDPG_model/actor_target_net.pth')
        torch.save(self.critic.state_dict(), './DDPG_model/critic_net.pth')
        torch.save(self.critic_target.state_dict(), './DDPG_model/critic_target_net.pth')
        print("====================================")
        print("Model has been saved.")
        print("====================================")

    def load(self):
        torch.load(self.actor.state_dict(), './DDPG_model/actor_net.pth')
        torch.load(self.actor_target.state_dict(), './DDPG_model/actor_target_net.pth')
        torch.load(self.critic.state_dict(), './DDPG_model/critic_net.pth')
        torch.load(self.critic_target.state_dict(), './DDPG_model/critic_target_net.pth')
        print("====================================")
        print("Model has been loaded.")
        print("====================================")