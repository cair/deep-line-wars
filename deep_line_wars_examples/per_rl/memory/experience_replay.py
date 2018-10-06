import torch
import numpy as np

class ExperienceReplay(object):

    def __init__(self, spec):
        self.memory_batch_size = spec["memory"]["batch"]
        self.memory_capacity = spec["memory"]["capacity"]
        self.memory_size = 0
        self._memory_pointer = 0

        self.memory_state = torch.empty((self.memory_capacity, ) + (4, ), dtype=torch.float)  # from spec
        self.memory_state1 = torch.empty((self.memory_capacity, ) + (4, ),  dtype=torch.float)  # from spec
        self.memory_rewards = torch.empty((self.memory_capacity, ),  dtype=torch.float)
        self.memory_actions = torch.empty((self.memory_capacity, ) + (1, ),  dtype=torch.long)  # from spec
        self.memory_terminal = torch.empty((self.memory_capacity, ),  dtype=torch.float)

    def add_memory(self, s, a, r, s1, t):
        """Saves a transition."""
        self.memory_state[self._memory_pointer] = s
        self.memory_state1[self._memory_pointer] = s1
        self.memory_rewards[self._memory_pointer] = r
        self.memory_actions[self._memory_pointer] = a
        self.memory_terminal[self._memory_pointer] = 1 if t else 0
        self._memory_pointer = (self._memory_pointer + 1) % self.memory_capacity
        self.memory_size = min(self.memory_size + 1, self.memory_capacity)

    def sample_memory(self):
        sampling_area = torch.tensor(np.random.randint(0, self.memory_size, self.memory_batch_size))

        return (self.memory_state.index_select(0, sampling_area),
               self.memory_actions.index_select(0, sampling_area),
               self.memory_rewards.index_select(0, sampling_area),
               self.memory_state1.index_select(0, sampling_area),
               self.memory_terminal.index_select(0, sampling_area))

