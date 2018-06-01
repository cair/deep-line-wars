import gym
from gym import error, spaces, utils
from gym.utils import seeding

from DeepLineWars.Game import Game


class DeepLineWarsEnv(gym.Env):
    id = "deeplinewars-random-v0"
    metadata = {'render.modes': ['human']}

    def __init__(self, ai="random", config={}):
        self.env = Game(config_override=config)
        self.player = self.env.players[0]
        opponent = self.env.players[1]

        self.observation_space = self.env.get_state(self.player).shape
        self.action_space = len(self.player.action_space)

    def set_representation(self, rep):
        self.env.representation = rep
        self.observation_space = self.env.get_state(self.player).shape

    def _step(self, action):
        data = self.env.step(self.player, action)
        self.env.update()
        return data

    def _reset(self):
        return self.env.reset(self.player)

    def _render(self, mode='human', close=False):
        if close:
            self.env.quit()
            return

        return self.env.render()


class DeepLineWarsDeterministic11x11Env(DeepLineWarsEnv):
    id = "deeplinewars-deterministic-11x11-v0"

    def __init__(self):
        super(DeepLineWarsDeterministic11x11Env, self).__init__(ai="hard_code_1", config={"game": {"width": 11, "height": 11}})


class DeepLineWarsDeterministic13x13Env(DeepLineWarsEnv):
    id = "deeplinewars-deterministic-13x13-v0"

    def __init__(self):
        super(DeepLineWarsDeterministic13x13Env, self).__init__(ai="hard_code_1", config={"game": {"width": 13, "height": 13}})


class DeepLineWarsDeterministic15x15Env(DeepLineWarsEnv):
    id = "deeplinewars-deterministic-15x15-v0"

    def __init__(self):
        super(DeepLineWarsDeterministic15x15Env, self).__init__(ai="hard_code_1", config={"game": {"width": 15, "height": 15}})


class DeepLineWarsDeterministic17x17Env(DeepLineWarsEnv):
    id = "deeplinewars-deterministic-17x17-v0"

    def __init__(self):
        super(DeepLineWarsDeterministic17x17Env, self).__init__(ai="hard_code_1", config={"game": {"width": 17, "height": 17}})


class DeepLineWarsStochastic11x11Env(DeepLineWarsEnv):
    id = "deeplinewars-stochastic-11x11-v0"

    def __init__(self):
        super(DeepLineWarsStochastic11x11Env, self).__init__(ai="random", config={"game": {"width": 11, "height": 11}})


class DeepLineWarsStochastic13x13Env(DeepLineWarsEnv):
    id = "deeplinewars-stochastic-13x13-v0"

    def __init__(self):
        super(DeepLineWarsStochastic13x13Env, self).__init__(ai="random", config={"game": {"width": 13, "height": 13}})


class DeepLineWarsStochastic15x15Env(DeepLineWarsEnv):
    id = "deeplinewars-stochastic-15x15-v0"

    def __init__(self):
        super(DeepLineWarsStochastic15x15Env, self).__init__(ai="random", config={"game": {"width": 15, "height": 15}})


class DeepLineWarsStochastic17x17Env(DeepLineWarsEnv):
    id = "deeplinewars-stochastic-17x17-v0"

    def __init__(self):
        super(DeepLineWarsStochastic17x17Env, self).__init__(ai="random", config={"game": {"width": 17, "height": 17}})


class DeepLineWarsShuffle11x11Env(DeepLineWarsEnv):
    id = "deeplinewars-shuffle-11x11-v0"

    def __init__(self):
        super(DeepLineWarsShuffle11x11Env, self).__init__(ai="shuffle", config={"game": {"width": 11, "height": 11}})


class DeepLineWarsShuffle13x13Env(DeepLineWarsEnv):
    id = "deeplinewars-shuffle-13x13-v0"

    def __init__(self):
        super(DeepLineWarsShuffle13x13Env, self).__init__(ai="shuffle", config={"game": {"width": 13, "height": 13}})


class DeepLineWarsShuffle15x15Env(DeepLineWarsEnv):
    id = "deeplinewars-shuffle-15x15-v0"

    def __init__(self):
        super(DeepLineWarsShuffle15x15Env, self).__init__(ai="shuffle", config={"game": {"width": 15, "height": 15}})


class DeepLineWarsShuffle17x17Env(DeepLineWarsEnv):
    id = "deeplinewars-shuffle-17x17-v0"

    def __init__(self):
        super(DeepLineWarsShuffle17x17Env, self).__init__(ai="shuffle", config={"game": {"width": 17, "height": 17}})


