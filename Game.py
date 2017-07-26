import io
import json
import time
from itertools import cycle

import numpy as np
import pygame
from PIL import Image
import base64
from Building import Building
from GUI import GUI
from Player import Player
from Unit import Unit
from web.Server import Webserver
import importlib
import matplotlib.pyplot as plt
import scipy.misc


class Game:

    def __init__(self):

        # Load configuration
        self.config = json.load(open("./config.json", "r"))
        self.unit_data = json.load(open("./units.json", "r"))
        self.building_data = json.load(open("./buildings.json", "r"))

        # Start web-server
        self.web_server = Webserver() if self.config["web"]["enabled"] else None


        # Heatmap
        self.hm_color_nothing = plt.cm.jet(0.0)[0:3]
        self.hm_color_building = plt.cm.jet(1.0)[0:3]
        self.hm_color_enemy_unit = plt.cm.jet(0.5)[0:3]
        self.hm_color_cursor = plt.cm.jet(0.2)[0:3]

        self.ticks = 0
        self.running = False

        # Z = 0 - Environmental Layer
        # Z = 1 - Unit Layer
        # Z = 2 - Unit Player Layer
        # Z = 3 - Building Layer
        # Z = 4 - Building Player Layer
        self.map = np.zeros((5, self.config["width"], self.config["height"]))
        self.mid = None
        self.setup_environment()
        self.action_space = 2 # 0 = Build, 1 = Spawn

        self.winner = None

        p1 = Player(1, self)
        p2 = Player(2, self)
        p1.opponent = p2
        p2.opponent = p1
        self.players = [p1, p2]
        self.gui = GUI(self)
        self.load_ai(p1, p2)





        self.statistics = {
            p1.id: 0,
            p2.id: 0
        }

        self.unit_shop = [Unit(data) for data in self.unit_data]
        self.building_shop = [Building(data) for data in self.building_data]

        self.ticks_per_second = self.config["ticks_per_second"]

        self.frame_counter = 0
        self.update_counter = 0
        self.allow_ai_update = False




    def step(self, player, action):

        reward = player.do_action(action)
        is_terminal = self.is_terminal()
        if is_terminal:
            if self.winner != player:
                reward = -100
            else:
                reward = 1000

        return self.get_state(player), reward, is_terminal, None,


    def is_terminal(self):
        return True if self.winner else False

    def load_ai(self, player_1, player_2):
        players = [player_1, player_2]

        for player_idx, ai_list in enumerate(self.config["ai"]["agents"]):
            for ai in ai_list:
                mod_name = ai[0]
                mod_loaded = importlib.import_module(mod_name)
                agent_instance = getattr(mod_loaded, ai[1])(self)
                agent_instance.init(players[player_idx])
                players[player_idx].agents.append(agent_instance)

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
        return 1.0 / self.config["fps"] if self.config["fps"] > 0 else 0

    def update_interval(self):
        return 1.0 / self.config["ups"] if self.config["ups"] > 0 else 0

    def stat_interval(self):
        return 1.0 / self.config["statps"] if self.config["statps"] > 0 else 0

    def apm_interval(self):
        return self.config["max_aps"]
        #return 1.0 / self.config["max_aps"] if self.config["max_aps"] > 0 else 0

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def summary(self):
        print(self.statistics)

    def loop(self):
        update_ratio = self.update_interval()
        render_ratio = self.render_interval()
        stat_ratio = self.stat_interval()
        apm_ratio = self.apm_interval()
        next_update = time.time()
        next_render = time.time()
        next_stat = time.time()
        #next_apm = time.time()

        while self.running:
            now = time.time()

            if now >= next_update:
                self.update()
                next_update = now + update_ratio
                self.update_counter += 1
                self.ticks += 1
                self.allow_ai_update = True

            if self.ticks % apm_ratio == 0 and self.allow_ai_update:
                self.ai_update()
                self.allow_ai_update = False
                #next_apm = now + apm_ratio
                # stats

            if now >= next_render:
                self.render()
                next_render = now + render_ratio
                self.frame_counter += 1

            if now >= next_stat:
                self.gui.caption()
                self.frame_counter = 0
                self.update_counter = 0
                next_stat = now + stat_ratio

        self.summary()

    def game_time(self):
        return self.ticks / self.ticks_per_second

    def update_statistics(self):
        self.statistics[self.winner.id] += 1

    def reset(self):
        self.map[1].fill(0)
        self.map[2].fill(0)
        self.map[3].fill(0)
        self.map[4].fill(0)

        for player in self.players:
            player.agents.get().reset()
            player.reset()
            player.agents.next()

        self.winner = None
        self.ticks = 0

    def generate_heatmap(self, player):

        # Start with fully exposed map
        m = np.zeros(shape=(self.config["height"], self.config["width"], 3))

        for y in range(m.shape[0]):
            for x in range(m.shape[1]):
                # Set initial color to 0
                m[y, x] = self.hm_color_nothing

        # Computer for buildings
        for b in player.buildings:
            m[b.y, b.x] = self.hm_color_building
            #for vis_y in range(max(0, b.y - b.attack_range), min(self.config["height"], b.y + b.attack_range)):
            #    for vis_x in range(max(0, b.x - b.attack_range), min(self.config["width"], b.x + b.attack_range)):
            #m[vis_y, vis_x] = self.hm_color_building

        for u in player.opponent.units:
            m[u.y, u.x] = self.hm_color_enemy_unit

        # cursor
        m[player.virtual_cursor_y, player.virtual_cursor_x] = self.hm_color_cursor

        #img = Image.fromarray(m, "RGBA")
        #img.save("heatmap_%s_%s.jpg" % (player.id, 0))
        #scipy.misc.toimage(m, cmin=0.0).save("heatmap_%s_%s.png" % (player.id, self.ticks))

        if self.web_server:
            image = scipy.misc.toimage(m, cmin=0.0)
            in_mem_file = io.BytesIO()
            image.save(in_mem_file, format="PNG")
            base64_encoded_result_bytes = base64.b64encode(in_mem_file.getvalue())
            base64_encoded_result_str = base64_encoded_result_bytes.decode('ascii')

            self.web_server.emit('heatmap', {"data": base64_encoded_result_str, "player": player.id})

        return m

    def get_state(self, player):
        if self.config["state_repr"] == "heatmap":
            return self.generate_heatmap(player)
        elif self.config["state_repr"] == "raw":
            return self.map
        elif self.config["state_repr"] == "image":
            # Get fullsize image
            image = np.array(pygame.surfarray.array3d(self.gui.surface_game))
            image = np.resize(image, (int(image.shape[0]/2), image.shape[1], image.shape[2]))
            scaled = scipy.misc.imresize(image, (40, 40), 'nearest')
            return scaled
        else:
            print("Error! MUSt choose state_repr as either heatmap or raw")
            exit(0)

    def update(self):

        if self.ticks / self.ticks_per_second > 600:
            self.reset()
            return

        if self.winner:
            self.update_statistics()
            self.reset()

        for player in self.players:

            # Check for winner
            if player.health <= 0:
                self.winner = player.opponent
                break

            player.update()

    def ai_update(self):
        if not self.config["ai"]["enabled"]:
            return

        for player in self.players:
            if player.agents.has_agent():
                player.agents.get().update(self.ticks / self.ticks_per_second)

    def render(self):
        self.gui.event()
        self.gui.draw()
