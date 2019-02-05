import uuid
import numpy as np

from os.path import realpath, dirname, join

from shop import Shop
from .building import Building
from .player import Player
from .unit import Unit
from .utils import dict_to_object, update
from deep_line_wars import config as conf
dir_path = dirname(realpath(__file__))


class Game:

    def __init__(self,
                 config=None,
                 unit_config=conf.unit,
                 building_config=conf.building,
                 levelup_config=conf.level_up
                 ):
        # Game ID
        self.id = uuid.uuid4()

        # Load configuration
        self.unit_data = unit_config
        self.building_data = building_config
        self.player_levels = levelup_config

        # Load Configuration
        # Apply customizations
        # Transform to Object
        self.config = conf.default_config
        if type(config) == dict:
            update(self.config, config)

        self.config = dict_to_object(self.config)

        self.width = self.config.game.width
        self.height = self.config.game.height

        self.ticks = 0
        self.running = False

        #self.setup_environment()

        self.winner = None

        p1 = Player(1, self)
        p2 = Player(2, self)
        p1.opponent = p2
        p2.opponent = p1
        self.players = [p1, p2]
        self.selected_player = p1

        self.gui = self.config.gui(self)

        self.ticks_per_second = self.config.mechanics.ticks_per_second

        self.shop = Shop(self)

    def is_terminal(self):
        return True if self.winner else False

    def step(self, action):

        pass
        
        # Perform Action
        """self.selected_player.action_space.perform(action)

        # Update state
        self.update()

        # Evaluate terminal state
        terminal = self.is_terminal()

        # Adjust reward according to terminal value
        if terminal:
            reward = -1 if self.winner != self.selected_player else 1
        else:
            reward = -1 if self.selected_player.health < self.selected_player.opponent.health else 0.001
        return self.get_state(), reward, terminal, {}"""

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
        for player in self.players:
            agent = player.agents.get()
            if agent:
                agent.reset()
            player.reset()
            player.agents.next()

        self.winner = None
        self.ticks = 0

        return self.get_state()

    def get_state(self):
        pass

    def update(self):

        if self.winner:
            return

        self.ticks += 1

        for player in self.players:
            player.update()

            if player.health <= 0:
                self.winner = player.opponent
                break

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

    def get_action_space(self):
        return self.selected_player.action_space.size
