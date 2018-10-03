import random
import torch


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

        state_batch = torch.empty((self.memory_batch_size, ) + (4, ), dtype=torch.float)  # from spec
        state1_batch = torch.empty((self.memory_batch_size, ) + (4, ), dtype=torch.float)  # from spec
        reward_batch = torch.empty((self.memory_batch_size, ), dtype=torch.float)
        action_batch = torch.empty((self.memory_batch_size, ) + (1, ), dtype=torch.long)  # from spec
        terminal_batch = torch.empty((self.memory_batch_size, ), dtype=torch.float)

        for j, i in enumerate(random.sample(range(0, self.memory_size - 1), self.memory_batch_size)):
            state_batch[j] = self.memory_state[i]
            state1_batch[j] = self.memory_state1[i]
            reward_batch[j] = self.memory_rewards[i]
            action_batch[j] = self.memory_actions[i]
            terminal_batch[j] = self.memory_terminal[i]

        return state_batch, action_batch, reward_batch, state1_batch, terminal_batch

