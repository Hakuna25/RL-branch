from wiserl.core.agent import Agent
import wiserl.agent.ppo_agent.config as cfg
from wiserl.core.wise_util import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class A2CAgent(Agent):
    def __init__(self, actor_class, n_states, n_actions, config=None, sync=True, use_ray=True):
        super().__init__(sync)
        self.config = cfg
        self.use_ray = use_ray
        if config is not None:
            self.config = config
        self.n_actions = n_actions
        self.n_states = n_states
        self.actor = actor_class(n_states, n_actions).to(device)
        self.learn_step_counter = 0
        #------- Define the optimizer------#
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.ACTOR_LR)
        # ------Define the loss function-----#
        self.gamma = self.config.GAMMA
        self.num_eps = 1e-9
        self.entropy = 0

    def update(self,  transition_dict):
        # update the target network every fixed steps
        self.learn_step_counter += 1
        log_probs = torch.cat(transition_dict['log_probs'])
        returns = torch.tensor(np.array(transition_dict['returns'][0]), dtype=torch.float).to(device)
        entropy = transition_dict['total_entropy'][0]
        returns = torch.squeeze(torch.squeeze(returns))
        values = torch.tensor(np.array(transition_dict['values']), dtype=torch.float).to(device)
        values = torch.squeeze(torch.squeeze(values))
        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean() * 0.5
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print("更新/////////////////////")
        if self.sync == False and self.use_ray:
            self._syncModel()

    def choseAction(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state).to(device), 0)# add 1 dimension to input state x
        action_dist, value = self.actor(state)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy().mean()
        return action.item(), log_prob, value, entropy

    def _syncModel(self):
        actor_param = self.actor.state_dict()
        if device.type != "cpu":
            for name, mm in actor_param.items():
                actor_param[name] = mm.cpu()
        self._fire(actor_param)
    
    def _updateModel(self, param):
        self.actor.load_state_dict(param)

