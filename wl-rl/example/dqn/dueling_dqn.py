from wiserl.core.runner import Runner
from wiserl.agent.dqn_agent.dueling_dqn_agent import DuelingDQNAgent
from wiserl.net.dqn_vnet import DQNVNet
from wiserl.core.wise_rl import WiseRL
import gym
import time
import matplotlib.pyplot as plt
use_ray = True
if use_ray:
    wise_rl = WiseRL()
else:
    import wiserl.agent.dqn_agent.config as config

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
                wise_rl.makeAgent(name="dueling_dqn_agent", agent_class=DuelingDQNAgent, net_class=DQNVNet, n_states=N_STATES,
                                n_actions=N_ACTIONS, sync=False, use_ray=use_ray)
                self.agent = wise_rl.getAgent("dueling_dqn_agent")
            else:
                self.agent = wise_rl.getAgent("dueling_dqn_agent")
        else:
            self.agent = DuelingDQNAgent(DQNVNet,N_STATES,N_ACTIONS, config, sync=False, use_ray=use_ray)

    def run(self):
        print("run")
        start = time.time()
        return_list = []
        for i_episode in range(500):
            s = self.env.reset()[0]
            ep_r = 0
            now_length = 0
            while True:
                now_length += 1
                a = self.agent.choseAction(s)
                s_, r, done, info, _ = self.env.step(a)
                x, x_dot, theta, theta_dot = s_
                r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
                r = r1 + r2
                self.agent.update(s, a, r, s_, done)
                ep_r += r
                if done or (now_length >= self.max_length):
                    return_list.append(ep_r)
                    r = -10
                    end = time.time()
                    print(self.local_rank, 'time', round((end - start), 2), ' Ep: ', i_episode, ' |', 'Ep_r: ',
                          round(ep_r, 2))
                    break
                s = s_
        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list, return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DuelingDQN on {}'.format(self.env_name))
        plt.savefig('../../result/DuelingDQN.png')
        plt.show()


if __name__ == '__main__':
    if use_ray:
        runners = wise_rl.makeRunner("runner", GymRunner, num=2)
        wise_rl.startAllRunner(runners)
    else:
        runners = GymRunner()
        runners.run()
