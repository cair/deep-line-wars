import json
import numpy as np
import time

from Building import Building
from GUI import GUI
from Player import Player
from Unit import Unit


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


class Game:

    def __init__(self):
        self.running = False
        self.config = json.load(open("./config.json", "r"))
        self.unit_data = json.load(open("./units.json", "r"))
        self.building_data = json.load(open("./buildings.json", "r"))

        # Z = 0 - Environmental Layer
        # Z = 1 - Unit Layer
        # Z = 2 - Unit Player Layer
        # Z = 3 - Building Layer
        # Z = 4 - Building Player Layer
        self.map = np.zeros((5, self.config["width"], self.config["height"]))
        self.mid = None
        self.setup_environment()
        self.gui = GUI(self)
        self.winner = None
        p1 = Player(1, self)
        p2 = Player(2, self)
        p1.opponent = p2
        p2.opponent = p1
        self.players = [p1, p2]

        self.statistics = {
            p1.id: 0,
            p2.id: 0
        }

        self.unit_shop = [Unit(data) for data in self.unit_data]
        self.building_shop = [Building(data) for data in self.building_data]
        self.ticks = 0
        self.ticks_per_second = self.config["ticks_per_second"]

        self.frame_counter = 0
        self.update_counter = 0

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
        print(update_ratio, render_ratio)
        next_update = time.time()
        next_render = time.time()
        next_stat = time.time()

        while self.running:
            now = time.time()

            if now >= next_update:
                self.update()
                next_update = now + update_ratio
                self.update_counter += 1
                self.ticks += 1

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

        self.winner = None
        p1 = Player(1, self)
        p2 = Player(2, self)
        p1.opponent = p2
        p2.opponent = p1
        self.players = [p1, p2]
        self.map[1].fill(0)
        self.map[2].fill(0)
        self.ticks = 0




    def update(self):
        if self.winner:
            self.update_statistics()
            self.reset()

        for player in self.players:

            # Check for winner
            if player.health <= 0:
                self.winner = player.opponent
                break

            player.update()

    def render(self):
        self.gui.event()
        self.gui.draw()
