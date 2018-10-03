import random

import torch
import torch.nn.functional as F
import torch.optim
from deep_line_wars_examples.per_rl import memory, sampling, models
from deep_line_wars_examples.per_rl.models import dqn
from deep_line_wars_examples.per_rl.memory import Memory
from deep_line_wars_examples.per_rl.models import Model


class BaseObject:

    def __init__(self):
        pass

    def depends_on_mixin(self, dependencies):
        for dependency in dependencies:
            if dependency not in self.__class__.__bases__:
                raise BaseException("Dependency not met: %s " % (dependency))

    def has_mixin(self, mixin):
        return mixin in self.__class__.__bases__


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

    def _train(self):
        raise NotImplementedError("Agents must inherit the _train function to qualify as an agent!")

    def train(self):

        if self.has_memory and self.memory_size > self.memory_batch_size:
            minibatch = self.sample_memory()
            self._train(minibatch)

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

        self.latest_action = torch.tensor([a])
        return self.latest_action


class DQN(Agent, Model, Memory):

    def __init__(self, spec):
        Model.__init__(self, spec)
        Memory.__init__(self, spec)
        Agent.__init__(self, spec)

    def _train(self, batches):

        state_batch, action_batch, reward_batch, state1_batch, terminal_batch = batches

        # Predict Q-Values for s

        s_values = self(state_batch).gather(1, action_batch)

        # Predict Q-Values for s1
        s1_values = self.forward(state1_batch).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = ((s1_values * 0.90) + reward_batch)

        # Compute Huber loss
        loss = F.smooth_l1_loss(s_values, expected_state_action_values.unsqueeze(1))


        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

#target = r + 0.99 * torch.a

    def _act(self, tensor):
        return int(torch.argmax(tensor).numpy())



if __name__ == "__main__":
    import numpy as np
    import gym

    # Create the Cart-Pole game environment
    env = gym.make('CartPole-v0')

    agent = DQN(spec=dict(
        input=dict(shape=env.observation_space.shape),
        output=dict(shape=(env.action_space.n, )),
        model=models.dqn.MLP,
        memory=dict(
            model=memory.experience_replay,
            capacity=10000,
            batch=512
        ),
        optimizer=dict(
            model=torch.optim.RMSprop,
            options=dict(
                lr=0.0001
            )
        ),
        sampling=dict(
            model=sampling.epsilon_decay,
            algorithm=random.randint,
            options=dict(
                start=0.5,
                end=0.01,
                steps=10000
            )
        )
    ))

    EPISODES = 40000
    for i in range(EPISODES):
        t = False
        s = env.reset()
        agent.observe(s, 0, t)  # Initial State observation
        steps = 0
        while not t:

            a = agent.act()
            s1, r, t, _ = env.step(a.item())

            agent.train()
            steps += 1

        print(steps)



#print(agent.model.forward(np.zeros((80, 80, 3))))