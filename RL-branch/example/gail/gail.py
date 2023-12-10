import torch
from wiserl.core.runner import Runner
from wiserl.agent.ppo_agent.ppo_agent import PPOAgent
from wiserl.agent.gail_agent.gail_agent import GAILAgent
from wiserl.net.ppo_net import Actor, Critic
from wiserl.net.gail_net import Discriminator
from wiserl.core.wise_rl import WiseRL
import wiserl.agent.ppo_agent.config as ppocfg
import wiserl.agent.gail_agent.config as gailcfg
import wiserl.core.rl_utils as rl_utils
import numpy as np
import gym
import time
import ray
import random
from tqdm import tqdm

wise_rl = WiseRL()

class GymRunner(Runner):
    def __init__(self, local_rank=0):
        self.env = gym.make('CartPole-v1')
        self.env.reset(seed=0)
        torch.manual_seed(0)
        print("rank=", local_rank)
        N_ACTIONS = self.env.action_space.n
        N_STATES = self.env.observation_space.shape[0]

        self.rank = local_rank
        if self.rank == 0:
            wise_rl.makePPOAgent(name='ppo_agent', agent_class=PPOAgent,
            actor_net=Actor, critic_net=Critic, n_states=N_STATES,
            n_actions=N_ACTIONS, sync=True)
            wise_rl.makeGAILAgent(name='gail_agent', agent_class=GAILAgent,
            d_net=Discriminator, n_states=N_STATES,
            n_actions=N_ACTIONS, sync=True)
            self.agent = wise_rl.getAgent('ppo_agent')
            self.gailagent = wise_rl.getAgent('gail_agent')
        else:
            self.agent = wise_rl.getAgent('ppo_agent')
            self.gailagent = wise_rl.getAgent('gail_agent')

    def sample_expert_data(self, n_episode):
        states = []
        actions = []
        for episode in range(n_episode):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.agent.take_action(state)
                states.append(state)
                actions.append(action)
                next_state, reward, done, _, _ = self.env.step(action)
                state = next_state
        return np.array(states), np.array(actions)

    def test_agent(self, agent, env, n_episodes=10):
        return_list = []
        for episode in range(n_episodes):
            episode_return = 0
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _, _ = env.step(action)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
        return np.mean(return_list)

    def run(self):
        return_list = rl_utils.train_on_policy_agent(self.env, self.agent, ppocfg.num_episodes)
        self.env.reset(seed=0)
        torch.manual_seed(0)
        random.seed(0)
        expert_s, expert_a = self.sample_expert_data(ppocfg.n_episode)

        random_index = random.sample(range(expert_s.shape[0]), ppocfg.n_samples)
        expert_s = expert_s[random_index]
        expert_a = expert_a[random_index]
        return_list = []
        with tqdm(total=gailcfg.n_episode, desc='进度条') as pbar:
            for i in range(gailcfg.n_episode):
                episode_return = 0
                state, _ = self.env.reset()
                done = False
                state_list = []
                action_list = []
                next_state_list = []
                done_list = []
                while not done:
                    action = self.agent.take_action(state)
                    next_state, reward, done, _, _ = self.env.step(action)
                    state_list.append(state)
                    action_list.append(action)
                    next_state_list.append(next_state)
                    done_list.append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                transition_dict = self.gailagent.learn(expert_s, expert_a, state_list, action_list, next_state_list, done_list)
                self.agent.update(transition_dict)
                if (i + 1) % 10 == 0:
                    pbar.set_postfix({'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

if __name__ == '__main__':
    print("use GPU to train" if torch.cuda.is_available() else "use CPU to train")
    # ray.init(address="auto")
    # ray.init(local_mode=True)
    
    runners = wise_rl.makeRunner("runner", GymRunner, num=1)
    wise_rl.startAllRunner(runners)