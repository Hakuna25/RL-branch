from wiserl.core.runner import Runner
from wiserl.agent.dqn_agent.prdqn_agent import DQN_PERAgent
from wiserl.net.dqn_net import DQNNet
from wiserl.core.wise_rl import WiseRL
import wiserl.agent.dqn_agent.config as cfg
import gym
import time
import numpy as np
import ray
import sys
import torch
import pylab
import random
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

use_ray = True
if use_ray:
    wise_rl = WiseRL()

class GymRunner(Runner):
    def __init__(self, local_rank=0):
        self.env = gym.make("CartPole-v1")
        print("rank=", local_rank)
        self.action_dim = self.env.action_space.n  # 2 actions
        self.state_dim = self.env.observation_space.shape[0]  # 4 states
        self.local_rank = local_rank
        if use_ray:
            if local_rank == 0:
                # net = DQNNet(self.state_dim, self.action_dim)    
                wise_rl.makePERDQNAgent(name="prdqn_agent", agent_class=DQN_PERAgent, state_dim=self.state_dim,
                                action_dim=self.action_dim, sync=False)
            self.agent = wise_rl.getAgent("prdqn_agent")
        else:
            self.agent = DQN_PERAgent(self.state_dim, self.action_dim, sync=False)

    def run(self):
        # In case of CartPole-v1, maximum length of episode is 500
        scores, episodes = [], []

        for e in range(500):
            done = False
            score = 0
            
            state = self.env.reset()[0]
            state = np.reshape(state, [1, self.state_dim])

            while not done:
                if self.agent.render:
                    self.env.render()

                # get action for the current state and go one step in environment
                action = self.agent.choseAction(state)
                
                next_state, reward, done, info, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_dim])
                # if an action make the episode end, then gives penalty of -100
                reward = reward if not done or score == 499 else -10

                # save the sample <s, a, r, s'> to the replay memory
                self.agent.store_experience(state, action, reward, next_state, done)
                # every time step do the training
                # if self.agent.memory.tree.n_entries >= self.agent.train_start:
                self.agent.update()

                score += reward
                state = next_state

                if done:
                    # every episode update the target model to be same with model
                    self.agent.update_target_model()

                    # every episode, plot the play time
                    score = score if score == 500 else score + 10
                    scores.append(score)
                    episodes.append(e)
                    pylab.plot(episodes, scores, 'b')
                    pylab.savefig("./save_graph/cartpole_dqn.png")
                    print("episode:", e, "score:", score)

                    # if the mean of scores of last 10 episode is bigger than 490
                    # stop training
                    if np.mean(scores[-min(10, len(scores)):]) > 490:
                        torch.save(self.agent.model, "./save_model/cartpole_dqn")
                        sys.exit()
        
if __name__ == '__main__':
    if use_ray:
        runners = wise_rl.makeRunner("runner", GymRunner, num=1)
        wise_rl.startAllRunner(runners)
    else:
        ray.shutdown()
        print('is initialized?',ray.is_initialized())
        runners = GymRunner()
        runners.run()
    
