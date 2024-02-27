import numpy as np
from torch.autograd import Variable
import torch
import random
import ray

class MemoryStore:
    def __init__(self, capacity, size):
        self.memory = np.zeros((capacity, size))
        self.memory_counter = 0
        self.capacity = capacity
        self.size = size

    def push(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))  # horizontally stack these vectors
        index = self.memory_counter % self.capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def sample(self, batch_size, N_STATES):
        sample_index = np.random.choice(self.capacity, batch_size)  # randomly select some data from buffer
        # extract experiences of batch size from buffer.
        b_memory = self.memory[sample_index, :]
        # extract vectors or matrices s,a,r,s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        # convert long int type to tensor
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))
        return b_s, b_a, b_r, b_s_

    def sampleppo(self):
        l_s, l_a, l_r, l_s_, l_done = [], [], [], [], []
        for n in self.memory_counter:
            s, a, r, s_, done = self.memory[n]
            l_s.append(torch.tensor([s], dtype=torch.float))
            l_a.append(torch.tensor([[a]], dtype=torch.float))
            l_r.append(torch.tensor([[r]], dtype=torch.float))
            l_s_.append(torch.tensor([s_], dtype=torch.float))
            l_done.append(torch.tensor([[done]], dtype=torch.float))
        s = torch.cat(l_s, dim=0)
        a = torch.cat(l_a, dim=0)
        r = torch.cat(l_r, dim=0)
        s_ = torch.cat(l_s_, dim=0)
        done = torch.cat(l_done, dim=0)
        return s, a, r, s_, done

    def put(self):
        return ray.put(self.memory)


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim=1, log_prob_dim=1, reward_dim=1):
        self.capacity = capacity
        self.memory_counter = 0

        self.state = np.zeros((capacity, state_dim))
        self.next_state = np.zeros((capacity, state_dim))
        self.action = np.zeros((capacity, action_dim))
        self.log_prob = np.zeros((capacity, log_prob_dim))
        self.reward = np.zeros((capacity, reward_dim))
        self.done = np.zeros((capacity, 1),dtype=np.bool_)

    def store(self, s, a, r, s_, log_prob=None, done=None):
        index = self.memory_counter % self.capacity

        self.state[index] = s
        self.action[index] = a
        self.reward[index] = r
        self.next_state[index] = s_

        if log_prob is not None:
            self.log_prob[index] = log_prob

        if done is not None:
            self.done[index] = done

        self.memory_counter += 1

    def sample(self, batch_size):
        sample_index = np.random.choice(min(self.capacity, self.memory_counter), batch_size)
        return {
            'state': torch.tensor(self.state[sample_index]),
            'action': torch.tensor(self.action[sample_index]),
            'reward': torch.tensor(self.reward[sample_index]),
            'next_state': torch.tensor(self.next_state[sample_index]),
            'done': torch.tensor(self.done[sample_index]),
            'log_prob': torch.tensor(self.log_prob[sample_index])
        }


class MAReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.buffer = None
        self.reset_buffer()
        # create a buffer (dictionary)

    def reset_buffer(self):
        self.buffer = {'obs_n': np.empty([self.batch_size, self.episode_limit, self.N, self.obs_dim]),
                       's': np.empty([self.batch_size, self.episode_limit, self.state_dim]),
                       'v_n': np.empty([self.batch_size, self.episode_limit + 1, self.N]),
                       'a_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'a_logprob_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'r_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'done_n': np.empty([self.batch_size, self.episode_limit, self.N])
                       }
        self.episode_num = 0

    def store_transition(self, episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['a_logprob_n'][self.episode_num][episode_step] = a_logprob_n
        self.buffer['r_n'][self.episode_num][episode_step] = r_n
        self.buffer['done_n'][self.episode_num][episode_step] = done_n

    def store_last_value(self, episode_step, v_n):
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.episode_num += 1

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key == 'a_n':
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)
        return batch

# SumTree
class SumTree:
    write = 0
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
    
    # update upwards to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample relying on priority s
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    
    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class Memory_Buffer_PER(object):
    # store as (s,a,r,s_) in SumTree
    def __init__(self, memory_size = 1000, a = 0.6, e = 0.01):
        self.tree = SumTree(memory_size)
        self.memory_size = memory_size
        self.prio_max = 0.1
        self.a = a
        self.e = e

    def push(self, s, a, r, s_):
        data = (s, a, r, s_)
        p = (np.abs(self.prio_max) + self.e) ** self.a
        self.tree.add(p, data)

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)

            state, action, reward, next_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            idxs.append(idx)
            priorities.append(p)
        return idxs, np.concatenate(states), actions, rewards, np.concatenate(next_states), dones

    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p)
        
    def size(self):
        return self.tree.n_entries
    
    

