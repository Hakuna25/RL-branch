import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Normal
import numpy as np
from wiserl.core.agent import Agent
import wiserl.agent.ddpg_agent.config as cfg
from wiserl.store.mem_store import ReplayBuffer
import os
import copy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TRPOAgent(Agent):
    """
    TRPO Agent
    """
    def __init__(self, Actor, Critic, state_dim, action_dim, lmbda, kl_constraint, alpha, critic_lr, gamma, config=None, sync=True):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.gamma = gamma
        self.lmbda = lmbda # GAE参数
        self.kl_constraint = kl_constraint # KL 距离最大限制
        self.alpha = alpha # 线性搜索参数
        os.makedirs('./TRPO_model/', exist_ok=True)
    
    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    def take_action(self, state):
        state = torch.tensor([state],dtype=torch.float).to(device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        # 计算海森矩阵和一个向量的乘积
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists)) # 计算平均KL距离
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # KL距离的梯度先和向量进行点积运算
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for _ in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            alpha = rdotr / (torch.dot(p, Hp) + 1e-8)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_probs, old_action_dists, max_vec):# 线性搜索
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)

        for i in range(15):
            coef = self.alpha ** i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())
            new_action_dists = torch.distributions.Categorical(new_actor(states))
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage): # 更新策略函数
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # 共轭梯度法计算 x = H^(-1)g
        descent_direction = self.conjugate_gradient(obj_grad, states, old_action_dists)
        Hd = self.hessian_matrix_vector_product(states, old_action_dists, descent_direction)
        max_coef = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, Hd) + 1e-8))
        new_para = self.line_search(states, actions, advantage, old_log_probs, old_action_dists, descent_direction * max_coef) # 线性搜索
        nn.utils. convert_parameters.vector_to_parameters(new_para, self.actor.parameters())

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(device)
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta.cpu().to(device))
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        old_action_dists = torch.distributions.Categorical(self.actor(states).detach())
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step() # 更新价值函数
        # 更新策略函数
        self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)

    def choseAction(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state)
        return action.detach().numpy()[0]

    def save(self):
        torch.save(self.actor.state_dict(), './TRPO_model/actor_net.pth')
        torch.save(self.critic.state_dict(), './TRPO_model/critic_net.pth')
        print("====================================")
        print("Model has been saved.")
        print("====================================")

    def load(self):
        torch.load(self.actor.state_dict(), './TRPO_model/actor_net.pth')
        torch.load(self.critic.state_dict(), './TRPO_model/critic_net.pth')
        print("====================================")
        print("Model has been loaded.")
        print("====================================")