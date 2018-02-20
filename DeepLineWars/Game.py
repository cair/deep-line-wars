import uuid
import json
import pygame
import importlib
import scipy.misc
import numpy as np
from os.path import realpath, dirname, join

from .Building import Building
from .GUI import GUI, NoGUI
from .Player import Player
from .Unit import Unit
from .utils import dict_to_object

dir_path = dirname(realpath(__file__))


class Game:

    def __init__(self, config_override=None):
        if config_override is None:
            config_override = {}

        # Game ID
        self.id = uuid.uuid4()

        # Load configuration
        self.unit_data = json.load(open(join(dir_path, "config/units.json"), "r"))
        self.building_data = json.load(open(join(dir_path, "config/buildings.json"), "r"))
        self.config = json.load(open(join(dir_path, "config/config.json"), "r"))
        self.config.update(config_override)
        self.config = dict_to_object(self.config)

        self.width = self.config.game.width
        self.height = self.config.game.height
        self.representation = self.config.representation

        self.ticks = 0
        self.running = False

        # Z = 0 - Environmental Layer
        # Z = 1 - Unit Layer
        # Z = 2 - Unit Player Layer
        # Z = 3 - Building Layer
        # Z = 4 - Building Player Layer
        self.map = np.zeros((5, self.width, self.height))
        self.center_area = None
        self.setup_environment()
        self.action_space = 2  # 0 = Build, 1 = Spawn

        self.winner = None

        p1 = Player(1, self)
        p2 = Player(2, self)
        p1.opponent = p2
        p2.opponent = p1
        self.players = [p1, p2]

        self.gui = GUI(self) if self.config.gui.enabled else NoGUI(self)

        self.unit_shop = [Unit(data) for data in self.unit_data]
        self.building_shop = [Building(data) for data in self.building_data]

        self.ticks_per_second = self.config.mechanics.ticks_per_second

    def is_terminal(self):
        return True if self.winner else False

    def step(self, player, action):

        reward = player.do_action(action)
        is_terminal = self.is_terminal()
        if is_terminal:
            if self.winner != player:
                reward = -1
            else:
                reward = 1

        reward = ((self.players[0].health - self.players[1].health) / 50) + 0.01
        return self.get_state(player), reward, is_terminal, {},

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

    def reset(self, _player=None):
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

        if _player:
            return self.get_state(_player)

        return None

    def get_state(self, player):

        if self.representation == "raw":
            return np.expand_dims(self.map, 0)
        elif self.representation == "raw_enemy":
            arr = np.zeros(shape=(1, 2, self.width, self.height))

            for u in player.buildings:
                arr[0, 0, u.x, u.y] = u.id

            for u in player.opponent.units:
                arr[0, 1, u.x, u.y] = u.id

            return arr

        elif self.representation == "raw_unit":
            return np.expand_dims(self.map[1], 0)
        elif self.representation == "image":
            image = np.array(pygame.surfarray.array3d(self.gui.surface_game))
            scaled = scipy.misc.imresize(image, (84, 84), 'nearest')
            scaled = np.swapaxes(scaled, 0, 2)
            return np.expand_dims(scaled, 0)
        elif self.representation == "image_grayscale":
            image = np.array(pygame.surfarray.array3d(self.gui.surface_game))
            scaled = scipy.misc.imresize(image, (84, 84), 'nearest')
            scaled = np.dot(scaled[..., :3], [0.299, 0.587, 0.114])
            scaled /= 255
            scaled = np.expand_dims(scaled, axis=0)
            return np.expand_dims(scaled, 0)
        else:
            raise NotImplementedError("representation must be image, raw, heatmap, raw_unit, image_grayscale")

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

    def quit(self):
        self.gui.quit()

    def caption(self):
        self.gui.caption()
