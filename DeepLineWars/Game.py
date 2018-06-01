import uuid
import json
import pygame
import numpy as np
from scipy.misc import imresize, imsave
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
        self.primary_player = p1

        self.gui = GUI(self) if self.config.gui.enabled else NoGUI(self)
        self.unit_shop = [Unit(data) for data in self.unit_data]
        self.building_shop = [Building(data) for data in self.building_data]

        self.ticks_per_second = self.config.mechanics.ticks_per_second

    def is_terminal(self):
        return True if self.winner else False

    def step(self, action, representation="image"):
        reward = self.primary_player.do_action(action[0], action[1]) # TODO fix return val
        reward = 0
        self.update()
        is_terminal = self.is_terminal()
        if is_terminal:
            if self.winner != self.primary_player:
                reward = -1
            else:
                reward = 1
        #reward = ((self.players[0].health - self.players[1].health) / 50) + 0.01
        return self.get_state(representation), reward, is_terminal, {},

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

    def reset(self, representation="image"):
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

        return self.get_state(representation=representation)

    def rgb2gray(self, rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def get_state(self, representation="image"):

        if representation == "raw":
            return np.reshape(self.map, (self.map.shape[2], self.map.shape[1], self.map.shape[0]))

        elif representation == "image":
            self.render()
            image = np.transpose(np.array(pygame.surfarray.pixels3d(self.gui.surface_game), dtype=np.uint8))
            image = imresize(image, (80, 80))
            #image = exposure.rescale_intensity(image, out_range=(0, 255))
            return image

            """elif representation == "image_grayscaled":
            image = np.array(pygame.surfarray.pixels3d(self.gui.surface_game), dtype=np.uint8)
            image = transform.resize(image, (80, 80), mode="constant")
            image = color.rgb2gray(image)
            return image
            """
        else:
            raise NotImplementedError("representation must be raw, image, or image_grayscaled")

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
