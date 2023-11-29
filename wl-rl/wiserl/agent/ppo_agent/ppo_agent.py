import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from wiserl.core.agent import Agent
import wiserl.agent.ppo_agent.config as cfg
from wiserl.core.wise_util import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PPOAgent(Agent):
    def __init__(self, actor_class, critic_class, n_states, n_actions, config=None, sync=True, use_ray=True):
        super().__init__(sync)
        self.config = cfg
        self.use_ray = use_ray
        if config != None:
            self.config=config
        self.n_actions = n_actions
        self.n_states = n_states
        self.actor = actor_class(n_states, n_actions).to(device)
        self.critic = critic_class(n_states).to(device)
        self.learn_step_counter = 0
        #------- Define the optimizer------#
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.ACTOR_LR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.CRITIC_LR)
        # ------Define the loss function-----#
        self.gamma = self.config.GAMMA
        self.lamda = self.config.LAMDA
        self.eps = self.config.PPO_EPS
        self.num_eps = 1e-9
        self.epoches = self.config.EPOCHS

    def update(self,  transition_dict):
        # update the target network every fixed steps
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(device)
        actions = torch.tensor(np.array(transition_dict['actions'])).view(-1, 1).to(device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(device)
        self.learn_step_counter += 1
        actions = actions.to(device)
        states = states.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        # calculate the Q value of state-action pair
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lamda, td_delta.cpu()).to(device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions) + self.num_eps).detach()
        for _ in range(self.epoches):
            log_probs = torch.log(self.actor(states).gather(1, actions) + self.num_eps)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        if self.sync == False and self.use_ray:
            self._syncModel()


    def choseAction(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state).to(device), 0) # add 1 dimension to input state x
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()    

    def _syncModel(self):
        actor_param = self.actor.state_dict()
        critic_param = self.critic.state_dict()
        if device.type != "cpu":
            for name, mm in actor_param.items():
                actor_param[name] = mm.cpu()
        self._fire(actor_param)
        if device.type != "cpu":
            for name, mm in critic_param.items():
                actor_param[name] = mm.cpu()
        self._fire(critic_param)
    
    def _updateModel(self,param):
        self.actor.load_state_dict(param)
        self.critic.load_state_dict(param)

