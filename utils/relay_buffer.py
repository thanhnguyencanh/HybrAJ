import numpy as np
from numpy.random import default_rng

# Expects tuples of (id, state, next_state, action, possible_action, suction, reward, done)
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.mem_cnt = 0
        self.seed = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)
        self.mem_cnt += 1

    def pack_action(self, action):
        ac, ad = action
        return np.concatenate([ac, ad])

    def ready(self, batch_size):
        if self.mem_cnt > (batch_size * 100):
            return True
        else:
            return False

    def sample(self, batch_size):
        rng = np.random.default_rng(self.seed)
        ind = rng.integers(len(self.storage), size=batch_size)
        self.seed += 1

        states, next_states, actions, rewards, dones = [], [], [], [], []
        # (state, next_state, (action, possible_action), reward, done)
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            states.append(np.asarray(state))
            next_states.append(np.asarray(next_state))
            actions.append(self.pack_action(action))
            rewards.append(np.asarray(reward))
            dones.append(np.asarray(done))

        return [np.array(states), np.array(next_states), np.array(actions),
                np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1)]