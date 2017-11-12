import json
import time
import numpy as np
import pygame
from multiprocessing import Process
from .Building import Building
from .GUI import GUI, NoGUI
from .Player import Player
from .Unit import Unit
from .utils import json_file_to_object
import importlib
import scipy.misc
import uuid
from os.path import realpath, dirname, join
dir_path = dirname(realpath(__file__))


class Game:

    def __init__(self, config_override={}, config_path=join(dir_path, "config/config.json")):
        self.id = uuid.uuid4()

        # Load configuration
        self.unit_data = json.load(open(join(dir_path, "config/units.json"), "r"))
        self.building_data = json.load(open(join(dir_path, "config/buildings.json"), "r"))
        self.config = json_file_to_object(config_path)

        self.width = self.config.game.width
        self.height = self.config.game.height

        if "game" in config_override:
            if "width" in config_override["game"]:
                self.width = config_override["game"]["width"]

            if "height" in config_override["game"]:
                self.height = config_override["game"]["height"]

        self.representation = "image"


        # Heatmap
        """import matplotlib.pyplot as plt
        self.hm_color_nothing = plt.cm.jet(0.0)[0:3]
        self.hm_color_building = plt.cm.jet(1.0)[0:3]
        self.hm_color_enemy_unit = plt.cm.jet(0.5)[0:3]
        self.hm_color_cursor = plt.cm.jet(0.2)[0:3]"""

        self.ticks = 0
        self.running = False

        # Z = 0 - Environmental Layer
        # Z = 1 - Unit Layer
        # Z = 2 - Unit Player Layer
        # Z = 3 - Building Layer
        # Z = 4 - Building Player Layer
        self.map = np.zeros((5, self.width, self.height))
        self.mid = None
        self.setup_environment()
        self.action_space = 2  # 0 = Build, 1 = Spawn

        self.winner = None

        p1 = Player(1, self)
        p2 = Player(2, self)
        p1.opponent = p2
        p2.opponent = p1
        self.players = [p1, p2]
        self.gui = GUI(self) if self.config.gui.enabled else NoGUI(self)
        self.load_ai(p1, p2)

        self.statistics = {
            p1.id: 0,
            p2.id: 0
        }

        self.unit_shop = [Unit(data) for data in self.unit_data]
        self.building_shop = [Building(data) for data in self.building_data]

        self.ticks_per_second = self.config.mechanics.ticks_per_second

        self.frame_counter = 0
        self.allow_ai_update = False

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

    def is_terminal(self):
        return True if self.winner else False

    def load_ai(self, player_1, player_2):
        players = [player_1, player_2]

        for idx, player in enumerate(players):
            ai_list = self.config.ai.agents[idx]

            for ai in ai_list:
                module_name = ai.package
                module_class_name = ai.clazz
                state_representation = ai.representation

                loaded_module = importlib.import_module(module_name)
                agent_instance = getattr(loaded_module, module_class_name)(self, player, state_representation)
                player.agents.append(agent_instance)

    def setup_environment(self):
        env_map = self.map[0]
        edges = [0, env_map.shape[0] - 1]

        # Set edges to "goal type"
        for edge in edges:
            for i in range(env_map[edge].shape[0]):
                env_map[edge][i] = 1

        # Set mid to "mid type"
        mid = env_map.shape[0] / 2
        mids = []
        if mid.is_integer():
            mids = [int(mid), int(mid) - 1]
        else:
            mids = [int(mid)]

        for mid in mids:
            for i in range(env_map[mid].shape[0]):
                env_map[mid][i] = 2
        self.mid = mids

    def render_interval(self):
        return 1.0 / self.config.mechanics.fps if self.config.mechanics.fps > 0 else 0

    def update_interval(self):
        return 1.0 / self.config.mechanics.ups if self.config.mechanics.ups > 0 else 0

    def stat_interval(self):
        return 1.0 / self.config.mechanics.statps if self.config.mechanics.statps > 0 else 0

    def apm_interval(self):
        return self.config.mechanics.max_aps
        #return 1.0 / self.config["max_aps"] if self.config["max_aps"] > 0 else 0

    def set_running(self, value):
        self.running = value

    def summary(self):
        print(self.statistics)

    def game_time(self):
        return self.ticks / self.ticks_per_second

    def update_statistics(self):
        self.statistics[self.winner.id] += 1

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

    def generate_heatmap(self, player):

        # Start with fully exposed map
        m = np.zeros(shape=(self.height, self.width, 3))

        for y in range(m.shape[0]):
            for x in range(m.shape[1]):
                # Set initial color to 0
                m[y, x] = self.hm_color_nothing

        # Computer for buildings
        for b in player.buildings:
            m[b.y, b.x] = self.hm_color_building

        for u in player.opponent.units:
            m[u.y, u.x] = self.hm_color_enemy_unit

        # cursor
        m[player.virtual_cursor_y, player.virtual_cursor_x] = self.hm_color_cursor

        return m

    def get_state(self, player):

        if self.representation == "heatmap":
            return np.expand_dims(self.generate_heatmap(player), 0)
        elif self.representation == "raw":
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

            if player.agents.has_agent():
                player.agents.get().update(self.ticks / self.ticks_per_second)

            if player.health <= 0:
                self.winner = player.opponent
                self.update_statistics()
                break

    def render(self):
        self.gui.event()
        self.gui.draw()

    def quit(self):
        self.gui.quit()

    def caption(self):
        self.gui.caption()
