import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from wiserl.core.agent import Agent
from wiserl.store.prioritised import Memory
from wiserl.net.dqn_net import DQN
from wiserl.net.dqn_net import NoisyVAnet
from wiserl.net.dqn_net import VAnet
import torch.optim as optim
import wiserl.agent.dqn_agent.config as cfg
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and prioritized experience replay memory & target q network
class DQN_PERAgent(Agent):
    def __init__(self, state_size, action_size, sync=True):
        super().__init__(sync)
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.memory_size = 20000
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 5000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.batch_size = 32
        self.train_start = 1000

        # create prioritized replay memory using SumTree
        self.memory = Memory(self.memory_size)

        # create main model and target model
        # self.model = DQN(state_size, action_size)
        self.model = NoisyVAnet(state_size, action_size)
        # self.model.apply(self.weights_init)
        # self.target_model = DQN(state_size, action_size)
        self.target_model = NoisyVAnet(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model = torch.load('save_model/cartpole_dqn')

    # weight xavier initialize
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # get action from model using epsilon-greedy policy
    def choseAction(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.from_numpy(state)
            state = Variable(state).float().cpu()
            q_value = self.model(state)
            _, action = torch.max(q_value, 1)
            return int(action)

    # save sample (error,<s,a,r,s'>) to the replay memory
    def store_experience(self, state, action, reward, next_state, done):
        target = self.model(Variable(torch.FloatTensor(state))).data
        old_val = target[0][action]
        target_val = self.target_model(Variable(torch.FloatTensor(next_state))).data
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.discount_factor * torch.max(target_val)

        error = abs(old_val - target[0][action])

        self.memory.add(error, (state, action, reward, next_state, done))
    
    # Function to compute multi-step returns
    def compute_n_step_returns(self, rewards, gamma, n_steps):
        returns = []
        for t in range(len(rewards)):
            G = 0
            for i in range(n_steps):
                if t + i < len(rewards):
                    G += (gamma**i) * rewards[t + i]
            returns.append(G)
        return torch.tensor(returns)

    # pick samples from prioritized replay memory (with batch_size)
    def update(self):
        if self.memory.tree.n_entries >= self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay

            mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
            states = np.vstack([sublist[0] for sublist in mini_batch])
            actions = [sublist[1] for sublist in mini_batch]
            rewards = [sublist[2] for sublist in mini_batch]
            next_states = np.vstack([sublist[3] for sublist in mini_batch])
            dones = np.array([sublist[4] for sublist in mini_batch])

            # bool to binary
            dones = dones.astype(int)

            # Q function of current state
            states = torch.Tensor(states)
            states = Variable(states).float()
            pred = self.model(states)

            # one-hot encoding
            a = torch.LongTensor(actions).view(-1, 1)
            one_hot_action = torch.FloatTensor(self.batch_size, self.action_size).zero_()
            one_hot_action.scatter_(1, a, 1)

            pred = torch.sum(pred.mul(Variable(one_hot_action)), dim=1)

            # Q function of next state
            next_states = torch.Tensor(next_states)
            next_states = Variable(next_states).float()
            next_pred = self.target_model(next_states).data

            rewards = torch.FloatTensor(rewards)
            # Multi-step
            # rewards = self.compute_n_step_returns(rewards, 0.99, 4)

            dones = torch.FloatTensor(dones)
            
            # Q Learning: get maximum Q value at s' from target model
            target = rewards + (1 - dones) * self.discount_factor * next_pred.max(1)[0]
            
            #####################################
            # For Double DQN
            one_hot_action_ = torch.FloatTensor(self.batch_size, self.action_size).zero_()
            a_ = torch.argmax(self.model(next_states), dim=1).unsqueeze(1)
            one_hot_action_.scatter_(1, a_, 1)
            target = rewards + (1 - dones) * self.discount_factor * torch.sum(next_pred.mul(Variable(one_hot_action_)), dim=1)
            #####################################

            target = Variable(target)

            errors = torch.abs(pred - target).data.numpy()

            # update priority
            for i in range(self.batch_size):
                idx = idxs[i]
                self.memory.update(idx, errors[i])

            self.optimizer.zero_grad()

            # MSE Loss function
            loss = (torch.FloatTensor(is_weights) * F.mse_loss(pred, target)).mean()
            loss.backward()

            # and train
            self.optimizer.step()
            
            