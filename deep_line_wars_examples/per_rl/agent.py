import torch
import numpy as np
from deep_line_wars_examples.per_rl.base import BaseObject
from deep_line_wars_examples.per_rl.memory import Memory
from deep_line_wars_examples.per_rl.models import Model


class Agent(BaseObject):

    def __init__(self, spec):
        super().__init__()
        self.depends_on_mixin([Model])

        self.has_memory = self.has_mixin(Memory)
        self.steps = 0

        # Initialize Sampler
        _sampler_class = spec["sampling"]["model"] if "sampling" in spec and "model" in  spec["sampling"] else None
        _sampler_args = spec["sampling"]["options"] if "sampling" in spec and "options" in spec["sampling"] else dict()
        self._sampler_algorithm = spec["sampling"]["algorithm"] if "sampling" in spec and "algorithm" in spec["sampling"] else random.randint
        self.sampler = _sampler_class(**_sampler_args)
        self.latest_state = None
        self.latest_action = None
        self.latest_reward = None

        self.optimization_iterator = 0

        # Insert options to the agent
        for k, v in spec["options"].items():
            setattr(self, k, v)

    def _train(self):
        raise NotImplementedError("Agents must inherit the _train function to qualify as an agent!")

    def train(self):

        if self.has_memory and self.memory_size > self.memory_batch_size:
            minibatch = self.sample_memory()
            self._train(minibatch)
            self.optimization_iterator += 1

    def observe(self, s, r, t):
        #########
        # State - Preprocess
        # TODO
        s = torch.tensor(s, dtype=torch.float)  # TODO - maybe should be another place?
        # Reward Shaping
        # TODO
        #######
        r = torch.tensor(r, dtype=torch.float)



        # Save in replay if specifid
        if self.has_memory and self.latest_state is not None and self.latest_action is not None:

            self.add_memory(self.latest_state, self.latest_action, r, s, t)

        self.latest_state = s
        self.latest_reward = r
        self.steps += 1

    def act(self):
        if self.sampler.eval() or self.latest_state is None:
            # Draw Random
            a = self._sampler_algorithm(0, np.prod(self.output_shape) - 1)

        else:
            # Forward pass
            out = self.forward(self.latest_state)
            a = self._act(out)

        # TODO if sentence?
        a_dist = np.zeros(self.output_shape) # TODO
        a_dist[a] = 1 # TODO
        self.log_histogram("action_distribution", a_dist, self.steps) # TODO

        self.latest_action = torch.tensor([a])
        return self.latest_action

