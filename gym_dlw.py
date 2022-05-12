import gym

from .deep_line_wars.config import Config
from .deep_line_wars.game import Game
from .deep_line_wars.gui import pygame, dummy


class DeepLineWarsEnvMultiAgentWrapper:
    # TODO Have not tested this.
    def __init__(self, env):
        self.env = env

    def step(self, actions):
        assert len(actions) == 2, "Must have two items in dict"
        agent_data = {}

        for agent_id, action in actions.items():
            agent_data[agent_id] = self.env.step(action)
            self.env.flip_player()
        return agent_data


class DeepLineWarsEnv(gym.Env):
    id = "deeplinewars-random-v0"
    metadata = {'render.modes': ['human']}

    def __init__(self, ai="random", width=10, height=10, config=dict(), **kwargs):
        self.env_config = kwargs["env_config"] if "env_config" in kwargs else {}
        self.window = self.env_config["window"] if "window" in self.env_config else False
        self.gui = self.env_config["gui"] if "gui" in self.env_config else False

        config_cls = Config(
            gui=Config.GUI(
                engine=dummy.GUI if not self.gui else pygame.GUI
            ),
            mechanics=Config.Mechanics(
                ups=-1,
                fps=-1
            ),
            **config
        )

        self.env = Game(width=width, height=height, config=config_cls)
        self.player = self.env.players[0]
        opponent = self.env.players[1]

        self.observation_space = self.env.get_state().shape
        self.action_space = gym.spaces.Discrete(self.env.get_action_space())

    def set_representation(self, rep):
        self.env.representation = rep
        self.observation_space = self.env.get_state().shape

    def step(self, action):
        data = self.env.step(action)
        self.env.update()
        return data

    def _seed(self):
        return None

    def reset(self):
        return self.env.reset()

    def render(self, mode='human', close=False):
        if close:
            self.env.quit()
            return

        state = self.env.render()
        if self.window:
            self.env.render_window()
        return state


class DeepLineWarsDeterministic11x11Env(DeepLineWarsEnv):
    id = "deeplinewars-deterministic-11x11-v0"

    def __init__(self, **kwargs):
        super(DeepLineWarsDeterministic11x11Env, self).__init__(
            ai="hard_code_1",
            width=11,
            height=13,
            **kwargs
        )


class DeepLineWarsDeterministic13x13Env(DeepLineWarsEnv):
    id = "deeplinewars-deterministic-13x13-v0"

    def __init__(self):
        super(DeepLineWarsDeterministic13x13Env, self).__init__(ai="hard_code_1", width=13, height=13)


class DeepLineWarsDeterministic15x15Env(DeepLineWarsEnv):
    id = "deeplinewars-deterministic-15x15-v0"

    def __init__(self):
        super(DeepLineWarsDeterministic15x15Env, self).__init__(ai="hard_code_1",
                                                                width=15, height=15)


class DeepLineWarsDeterministic17x17Env(DeepLineWarsEnv):
    id = "deeplinewars-deterministic-17x17-v0"

    def __init__(self):
        super(DeepLineWarsDeterministic17x17Env, self).__init__(ai="hard_code_1",
                                                                width=17, height=17)


class DeepLineWarsStochastic11x11Env(DeepLineWarsEnv):
    id = "deeplinewars-stochastic-11x11-v0"

    def __init__(self):
        super(DeepLineWarsStochastic11x11Env, self).__init__(ai="random", width=11, height=11)


class DeepLineWarsStochastic13x13Env(DeepLineWarsEnv):
    id = "deeplinewars-stochastic-13x13-v0"

    def __init__(self):
        super(DeepLineWarsStochastic13x13Env, self).__init__(ai="random", width=13, height=13)


class DeepLineWarsStochastic15x15Env(DeepLineWarsEnv):
    id = "deeplinewars-stochastic-15x15-v0"

    def __init__(self):
        super(DeepLineWarsStochastic15x15Env, self).__init__(ai="random", width=15, height=15)


class DeepLineWarsStochastic17x17Env(DeepLineWarsEnv):
    id = "deeplinewars-stochastic-17x17-v0"

    def __init__(self):
        super(DeepLineWarsStochastic17x17Env, self).__init__(ai="random", width=17, height=17)


class DeepLineWarsShuffle11x11Env(DeepLineWarsEnv):
    id = "deeplinewars-shuffle-11x11-v0"

    def __init__(self):
        super(DeepLineWarsShuffle11x11Env, self).__init__(ai="shuffle", width=11, height=11)


class DeepLineWarsShuffle13x13Env(DeepLineWarsEnv):
    id = "deeplinewars-shuffle-13x13-v0"

    def __init__(self):
        super(DeepLineWarsShuffle13x13Env, self).__init__(ai="shuffle",width=13, height=13)


class DeepLineWarsShuffle15x15Env(DeepLineWarsEnv):
    id = "deeplinewars-shuffle-15x15-v0"

    def __init__(self):
        super(DeepLineWarsShuffle15x15Env, self).__init__(ai="shuffle", width=15, height=15)


class DeepLineWarsShuffle17x17Env(DeepLineWarsEnv):
    id = "deeplinewars-shuffle-17x17-v0"

    def __init__(self):
        super(DeepLineWarsShuffle17x17Env, self).__init__(ai="shuffle", width=17, height=17)

