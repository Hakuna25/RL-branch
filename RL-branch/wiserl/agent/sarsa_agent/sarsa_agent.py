import numpy as np
from wiserl.core.agent import Agent
import wiserl.agent.sarsa_agent.config as cfg

class SARSAAgent(Agent):
    '''Q-Learning algorithm'''
    def __init__(self, ncol, nrow, n_action, config=None, sync=True):
        super(SARSAAgent, self).__init__(sync)
        self.config = cfg
        if config is not None:
            self.config = config

        self.Q_table = np.zeros([nrow * ncol, n_action]) # Initialize Q(s,a) table
        self.n_action = n_action
        self.alpha = cfg.alpha
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon
    
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error
