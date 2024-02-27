import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from wiserl.core.rl_utils import CliffWalkingEnv
from wiserl.core.runner import Runner
from wiserl.agent.sarsa_agent.sarsa_agent import SARSAAgent
from wiserl.core.wise_rl import WiseRL
import wiserl.agent.sarsa_agent.config as cfg
import gym
import time
import ray

wise_rl = WiseRL()

np.random.seed(0)

class GymRunner(Runner):
    def __init__(self, local_rank=0):
        self.env = CliffWalkingEnv(cfg.ncol, cfg.nrow)
        print("rank=", local_rank)
        self.rank = local_rank
        if self.rank == 0:
            wise_rl.makeSARSAAgent(name='sarsa_agent', agent_class=SARSAAgent, \
            ncol=cfg.ncol, nrow=cfg.nrow, n_action=4, sync=True)
            self.agent = wise_rl.getAgent('sarsa_agent')
        else:
            self.agent = wise_rl.getAgent('sarsa_agent')
        
    def run(self):
        return_list = []
        for i in range(10):
            with tqdm(total=int(cfg.num_episodes), desc="Iteration %d" % i) as pbar:
                for i_episode in range(int(cfg.num_episodes/10)):
                    episode_return = 0
                    state = self.env.reset()
                    done = False
                    while not done:
                        action = self.agent.take_action(state)
                        next_state, reward, done = self.env.step(action)
                        next_action = self.agent.take_action(next_state)
                        episode_return += reward
                        self.agent.update(state, action, reward, next_state, next_action)
                        state = next_state
                    return_list.append(episode_return)
                    if (i_episode+1) % 10 == 0:
                        pbar.set_postfix({'episode':'%d' % (cfg.num_episodes / 10 * i + i_episode + 1), \
                                          'return':'%.3f' % np.mean(return_list[-10:])})
                    pbar.update(1)
        episode_list = list(range(len(return_list)))
        plt.plot(episode_list, return_list)
        plt.xlabel('episode')
        plt.ylabel('return')
        plt.savefig('SARSAg Cliff.png')

if __name__ == '__main__':
    print("use GPU to train" if torch.cuda.is_available() else "use CPU to train")
    # ray.init(address="auto")
    # ray.init(local_mode=True)
    
    runners = wise_rl.makeRunner("runner", GymRunner, num=3)
    wise_rl.startAllRunner(runners)