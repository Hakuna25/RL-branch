from wiserl.core.runner import Runner
from wiserl.agent.ac_agent.a2c_agent import A2CAgent
from wiserl.net.a2c_net import ActorCritic
from wiserl.core.wise_rl import WiseRL
from wiserl.core.wise_util import *
import gym
import time
import torch
import matplotlib.pyplot as plt
import wiserl.agent.ppo_agent.config as config
use_ray = True
if use_ray:
    wise_rl = WiseRL()

torch.manual_seed(2)# 为CPU设置随机种子

class GymRunner(Runner):
    def __init__(self, local_rank=0):
        self.env_name = "CartPole-v1"
        self.env = gym.make(self.env_name)
        self.max_length = 300
        print("rank=", local_rank)
        N_ACTIONS = self.env.action_space.n  # 2 actions
        N_STATES = self.env.observation_space.shape[0]  # 4 states
        self.local_rank = local_rank
        if use_ray:
            if local_rank == 0:
                 wise_rl.makeA2CAgent(name="a2c_agent", agent_class=A2CAgent, actor_net=ActorCritic,
                                n_states=N_STATES, n_actions=N_ACTIONS, sync=True)
            self.agent = wise_rl.getAgent("a2c_agent")
        else:
            self.agent = A2CAgent(ActorCritic, N_STATES, N_ACTIONS, config, sync=False, use_ray=use_ray)

    def run(self):
        print("run")
        start = time.time()
        return_list = []
        for i_episode in range(1000):
            state = self.env.reset()[0]
            ep_r = 0
            total_entropy = 0
            now_length = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], "total_entropy": [],
                               'dones': [], "log_probs": [], "returns": [], "values": []}
            while True:
                now_length += 1
                action, log_prob, value, entropy = self.agent.choseAction(state)
                total_entropy += entropy
                next_state, reward, done, info, _ = self.env.step(action)
                x, x_dot, theta, theta_dot = next_state
                r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
                r = r1 + r2
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                transition_dict['log_probs'].append(log_prob)
                transition_dict['values'].append(value.detach().cpu().numpy())
                ep_r += reward
                if done or (now_length >= self.max_length):
                    return_list.append(ep_r)
                    r = -10
                    end = time.time()
                    print(self.local_rank, 'time', round((end - start), 2), ' Ep: ', i_episode, ' |', 'Ep_r: ',
                          round(ep_r, 2))
                    break
                state = next_state
            action, log_prob, next_value, entropy = self.agent.choseAction(next_state)
            returns = compute_returns(next_value, transition_dict['rewards'], transition_dict['dones'])
            transition_dict['returns'].append(returns)
            transition_dict['total_entropy'].append(total_entropy)
            self.agent.update(transition_dict)

        end = time.time()
        print('time spend:', end-start)
        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list, return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('A2C on {}'.format(self.env_name))
        plt.savefig('../../result/A2C.png')
        plt.show()


if __name__ == '__main__':
    if use_ray:
        runners = wise_rl.makeRunner("runner", GymRunner, num=1)
        wise_rl.startAllRunner(runners)
    else:
        runners = GymRunner()
        runners.run()
