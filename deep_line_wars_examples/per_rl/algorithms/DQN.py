import torch
import torch.nn.functional as F
import torch.optim

from deep_line_wars_examples.per_rl.agent import Agent
from deep_line_wars_examples.per_rl.callbacks import Callbacks
from deep_line_wars_examples.per_rl.environment import Environment
from deep_line_wars_examples.per_rl.logger import Logger
from deep_line_wars_examples.per_rl.memory import Memory
from deep_line_wars_examples.per_rl.models import Model





class DQN(Agent, Model, Memory, Environment, Logger, Callbacks):

    def __init__(self, spec):
        Model.__init__(self, spec)
        Memory.__init__(self, spec)
        Agent.__init__(self, spec)
        Environment.__init__(self, spec)
        Callbacks.__init__(self)
        Logger.__init__(self, spec)

    def _train(self, batches):

        state_batch, action_batch, reward_batch, state1_batch, terminal_batch = batches

        # Predict Q-Values for s

        s_values = self(state_batch).gather(1, action_batch)

        # Predict Q-Values for s1
        s1_values = self.forward(state1_batch).max(1)[0].detach()

        """target = reward_batch + (self.gamma * s1_values)

        bellman_error = target.unsqueeze(1) - s_values

        clipped_bellman_error = bellman_error.clamp(-1, 1)

        delta_error = clipped_bellman_error * -1.0

        self.optimizer.zero_grad()

        s_values.backward(delta_error.data)
        self.log_scalar("loss", delta_error.sum(), self.optimization_iterator)

        self.optimizer.step()"""



        # Compute the expected Q values
        expected_state_action_values = ((s1_values * self.gamma) + reward_batch)

        # Compute Huber loss
        loss = F.mse_loss(s_values, expected_state_action_values.unsqueeze(1))
        self.log_scalar("loss", loss, self.optimization_iterator)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def _act(self, tensor):
        return int(torch.argmax(tensor).numpy())


