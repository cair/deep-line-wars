import uuid
import numpy as np

from os.path import realpath, dirname

import time

from .config import Config
from .player import Player
from .shop import Shop
from .state import State

dir_path = dirname(realpath(__file__))


class Game:

    def __init__(self, width, height, config: Config = None):
        # Create
        self.id = uuid.uuid4()

        self.config = config if config else Config()
        self.config.set_size(width, height)
        self.config.validate()

        self.width = self.config.width
        self.height = self.config.height

        self.ticks = 0
        self.running = False

        self.state = State(self, width, height)

        self.winner = None


        p1 = Player(1, self)
        p2 = Player(2, self)
        p1.opponent = p2
        p2.opponent = p1
        self.players = [p1, p2]
        self.selected_player = p1
        self.flipped = False

        self.gui = self.config.gui.engine(self)
        self.shop = Shop(self)

        self.ticks_per_second = self.config.mechanics.ticks_per_second

    def is_terminal(self):
        return True if self.winner else False

    def step(self, action):

        # Perform Action
        self.selected_player.action_space.perform(action)

        # Update state
        self.update()

        # Evaluate terminal state
        terminal = self.is_terminal()

        # Adjust reward according to terminal value
        if terminal:
            reward = -1 if self.winner != self.selected_player else 1
        else:
            reward = -1 if self.selected_player.health < self.selected_player.opponent.health else 0.001
        return self.get_state(), reward, terminal, {}

    def render_interval(self):
        return 1.0 / self.config.mechanics.fps if self.config.mechanics.fps > 0 else 0

    def update_interval(self):
        return 1.0 / self.config.mechanics.ups if self.config.mechanics.ups > 0 else 0

    def stat_interval(self):
        return 1.0 / self.config.mechanics.statps if self.config.mechanics.statps > 0 else 0

    def set_running(self, value):
        self.running = value

    def game_time(self):
        return self.ticks / self.ticks_per_second

    def reset(self):
        self.state.reset()

        for player in self.players:
            agent = player.agents.get()
            if agent:
                agent.reset()
            player.reset()
            player.agents.next()

        self.winner = None
        self.ticks = 0

        return self.get_state()

    def _get_raw_state(self, flip=False):
        state = np.reshape(self.state.grid, (self.state.grid.shape[2], self.state.grid.shape[1], self.state.grid.shape[0]))
        if flip:
            state = np.fliplr(state)
        return state

    def get_state(self):

        if self.config.gui.state_representation == "RAW":
            return self._get_raw_state(flip=self.flipped)
        elif self.config.gui.state_representation == "RGB":
            self.render()
            return self.gui.get_state(grayscale=False, flip=self.state.flipped)
        elif self.config.gui.state_representation == "L":
            self.render()
            return self.gui.get_state(grayscale=True, flip=self.flipped)
        else:
            raise NotImplementedError("representation must be RAW, RGB, or L")

    def update(self):

        if self.winner:
            return

        self.ticks += 1

        for player in self.players:
            player.update()

            if player.health <= 0:
                self.winner = player.opponent
                break

        if self.config.mechanics.ups > 0:
            time.sleep(self.update_interval())

    def render(self):
        self.gui.event()
        self.gui.draw()

    def render_window(self):
        self.gui.draw_screen()

    def quit(self):
        self.gui.quit()

    def caption(self):
        self.gui.caption()

    def flip_player(self):
        self.selected_player = self.selected_player.opponent
        self.state.flipped = not self.state.flipped

    def get_action_space(self):
        return self.selected_player.action_space.size
