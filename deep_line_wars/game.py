import uuid
import numpy as np

from os.path import realpath, dirname, join
from .building import Building
from .player import Player
from .unit import Unit
from .utils import dict_to_object, update
from deep_line_wars import config as conf
dir_path = dirname(realpath(__file__))


class Game:

    def __init__(self, config=dict(), unit_config=conf.unit, building_config=conf.building, levelup_config=conf.levelup):
        # Create
        self.id = uuid.uuid4()

        # Load configuration
        self.unit_data = unit_config
        self.building_data = building_config
        self.player_levels = levelup_config

        # Load Configuration
        # Apply customizations
        # Transform to Object
        self.config = conf.default_config
        update(self.config, config)
        self.config = dict_to_object(self.config)

        self.width = self.config.game.width
        self.height = self.config.game.height

        self.ticks = 0
        self.running = False

        # Z = 0 - Environmental Layer
        # Z = 1 - Unit Layer
        # Z = 2 - Unit Player Layer
        # Z = 3 - Building Layer
        # Z = 4 - Building Player Layer
        self.map = np.zeros((5, self.width, self.height), dtype=np.uint8)
        self.center_area = None
        self.setup_environment()
        self.action_space = 2  # 0 = Build, 1 = Spawn

        self.winner = None

        p1 = Player(1, self)
        p2 = Player(2, self)
        p1.opponent = p2
        p2.opponent = p1
        self.players = [p1, p2]
        self.selected_player = p1

        self.gui = self.config.gui(self)
        self.unit_shop = [Unit(data) for data in self.unit_data]
        self.building_shop = [Building(data) for data in self.building_data]

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
        reward = 1 if terminal and self.winner != self.selected_player else 0

        return self.get_state(), reward, terminal, {}

    def setup_environment(self):
        env_map = self.map[0]
        edges = [0, env_map.shape[0] - 1]

        # Set edges to "goal type"
        for edge in edges:
            for i in range(env_map[edge].shape[0]):
                env_map[edge][i] = 1

        # Set mid to "mid type"
        center = env_map.shape[0] / 2
        center_area = [int(center), int(center - 1)] if center.is_integer() else [int(center)]

        for center_item in center_area:
            for i in range(env_map[center_item].shape[0]):
                env_map[center_item][i] = 2

        self.center_area = center_area

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
        self.map[1].fill(0)
        self.map[2].fill(0)
        self.map[3].fill(0)
        self.map[4].fill(0)

        for player in self.players:
            agent = player.agents.get()
            if agent:
                agent.reset()
            player.reset()
            player.agents.next()

        self.winner = None
        self.ticks = 0

        return self.get_state()

    def _get_raw_state(self):
        return np.reshape(self.map, (self.map.shape[2], self.map.shape[1], self.map.shape[0]))

    def get_state(self):

        if self.config.state_representation == "RAW":
            return self._get_raw_state()
        elif self.config.state_representation == "RGB":
            return self.gui.get_state(grayscale=False)
        elif self.config.state_representation == "L":
            return self.gui.get_state(grayscale=True)
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
