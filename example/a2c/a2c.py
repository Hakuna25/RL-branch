import torch
from wiserl.core.runner import Runner
from wiserl.agent.a2c_agent.a2c_agent import A2CAgent
from wiserl.net.a2c_net import PolicyNet, Baseline
from wiserl.core.wise_rl import WiseRL
import matplotlib.pyplot as plt
import gym
import time
import ray

use_ray = False
if use_ray:
    wise_rl = WiseRL(use_ray=True)

class GymRunner(Runner):
    def __init__(self, local_rank=0):
        self.env = gym.make('CartPole-v1')
        print("rank=", local_rank)
        N_ACTIONS = self.env.action_space.n  # 2 actions
        N_STATES = self.env.observation_space.shape[0]  # 4 states
        self.rank = local_rank
        if use_ray:
            if self.rank == 0:
                wise_rl.makeA2CAgent(name='ddpg_agent', agent_class=A2CAgent,
                policy_net=PolicyNet(self.env.observation_space.shape, self.env.action_space.n).to(torch.device('cuda')), \
                baseline_net=Baseline(self.env.observation_space.shape).to(torch.device('cuda')),\
                sync=True)
                self.agent = wise_rl.getAgent('ddpg_agent')
            else:
                self.agent = wise_rl.getAgent('ddpg_agent')
        else:
            self.agent = A2CAgent(policy_net=PolicyNet(self.env.observation_space.shape, self.
                                                       env.action_space.n).to(torch.device('cuda')),
                                  baseline_net=Baseline(self.env.observation_space.shape).to(torch.device('cuda')))


    def run(self):

        rews = self.agent.train(self.env, 32, 200, 0.99, 5)
        plt.plot(rews)
        plt.savefig('a2c_cartpole.png')

if __name__ == '__main__':
    if use_ray:
        runners = wise_rl.makeRunner("runner", GymRunner, num=1)
        wise_rl.startAllRunner(runners)
    else:
        ray.shutdown()
        print('is initialized?',ray.is_initialized())
        runners = GymRunner()
        runners.run()