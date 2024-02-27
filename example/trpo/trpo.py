import torch
from wiserl.core.runner import Runner
from wiserl.agent.trpo_agent.trpo_agent import TRPOAgent
from wiserl.net.trpo_net import Actor, Critic
from wiserl.core.wise_rl import WiseRL
from wiserl.core.rl_utils import train_on_policy_agent
from wiserl.core.rl_utils import moving_average
from gym.wrappers import RescaleAction
import wiserl.agent.trpo_agent.config as cfg
import gym
import time
import ray
import numpy as np
import matplotlib.pyplot as plt

wise_rl = WiseRL()

class GymRunner(Runner):
    def __init__(self, local_rank=0):
        self.env = gym.make('CartPole-v1')  
        # self.env = RescaleAction(self.env, min_action=-1.0, max_action=1.0)
        self.env.reset(seed=0)
        torch.manual_seed(0)
        print("rank=", local_rank)
        N_ACTIONS = self.env.action_space.n
        N_STATES = self.env.observation_space.shape[0]
        print('action_dim:', N_ACTIONS, 'state_dim:', N_STATES)
        self.rank = local_rank
        if self.rank == 0:
            wise_rl.makeTRPOAgent(name='trpo_agent', agent_class=TRPOAgent,
            actor_net=Actor, critic_net=Critic, n_states=N_STATES,
            n_actions=N_ACTIONS, lmbda=0.95, kl_constraint=0.0005, alpha=0.5, critic_lr=1e-2, gamma=0.98, sync=True)
            self.agent = wise_rl.getAgent('trpo_agent')
        else:
            self.agent = wise_rl.getAgent('trpo_agent')
        
    def run(self):
        return_list = train_on_policy_agent(self.env, self.agent, 500)
        episodes_list = list(range(len(return_list)))
        

        with open('5*return_list.txt', 'w') as file:
            for item in return_list:
                file.write(str(item) + '\n')
        with open('5*episodes_list.txt', 'w') as file:
            for item in episodes_list:
                file.write(str(item) + '\n')
        

        plt.plot(episodes_list, return_list)
        plt.xlabel('episodes')
        plt.ylabel('returns')
        plt.title('TRPO on CartPole-v1')
        plt.savefig('5*TRPO_CartPole-v1.png')

        mv_return = moving_average(return_list, 9)
        plt.plot(episodes_list, mv_return)
        plt.xlabel('episodes')
        plt.ylabel('returns')
        plt.title('TRPO on CartPole-v1')
        plt.savefig('5*movingTRPO_CartPole-v1.png')
        print('Ending.')


if __name__ == '__main__':
    print("use GPU to train" if torch.cuda.is_available() else "use CPU to train")
    # ray.init(address="auto")
    # ray.init(local_mode=True)
    
    runners = wise_rl.makeRunner("runner", GymRunner, num=5)
    wise_rl.startAllRunner(runners)