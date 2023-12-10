import torch
from wiserl.core.runner import Runner
from wiserl.agent.td3_agent.td3_agent import TD3Agent
from wiserl.net.td3_net import Actor, Critic
from wiserl.core.wise_rl import WiseRL
from gym.wrappers import RescaleAction
import wiserl.agent.td3_agent.config as cfg
import gym
import time
import ray

wise_rl = WiseRL()

class GymRunner(Runner):
    def __init__(self, local_rank=0):
        self.env = gym.make('Pendulum-v1')
        
        self.env = RescaleAction(self.env, min_action=-1.0, max_action=1.0)
        print("rank=", local_rank)
        N_ACTIONS = self.env.action_space.shape[0]
        N_STATES = self.env.observation_space.shape[0]
        ACTION_BOUND = self.env.action_space.high[0]
        self.rank = local_rank
        if self.rank == 0:
            wise_rl.makeTD3Agent(name='td3_agent', agent_class=TD3Agent,
            actor_net=Actor, critic_net=Critic, n_states=N_STATES,
            n_actions=N_ACTIONS, action_bound=ACTION_BOUND, sync=True)
            self.agent = wise_rl.getAgent('td3_agent')
        else:
            self.agent = wise_rl.getAgent('td3_agent')
        
    def run(self):

        print("====================================")
        print("Collection Experience...")
        print("====================================")
        start_time = time.time()
        ep_r = 0
        for i in range(cfg.MAX_EPOCH):
            state, _ = self.env.reset()
            for t in range(cfg.max_steps):
                action = self.agent.choseAction(state)
                # print('!!!!!!!!!!!!!action,',action)
                # next_state, reward, done, _, _ = env.step(np.float32(action))
                next_state, reward, done, _, _ = self.env.step(action)
                ep_r += reward
                self.agent.update(state, action, reward, next_state, done)
                state = next_state
                if done or t == 199:
                    if self.rank == 0:
                        print("Ep_i: {},  ep_r: {}, time_step: {}".format(i, ep_r, t))
                    # print("Ep_i: {},  ep_r: {}, time_step: {}".format(i, ep_r, t))
                    break
                # print(step)
            if ep_r > -10:
                if self.rank == 0:
                    self.agent.save()
                    print("train finish")
                    print("training time= ", time.time() - start_time)
                break
            if i % cfg.log_interval == 0:
                self.agent.save()
            ep_r = 0

if __name__ == '__main__':
    print("use GPU to train" if torch.cuda.is_available() else "use CPU to train")
    # ray.init(address="auto")
    # ray.init(local_mode=True)
    
    runners = wise_rl.makeRunner("runner", GymRunner, num=2)
    wise_rl.startAllRunner(runners)