import json
import copy
import numpy as np
import math


class Player:

    def __init__(self, i, game):
        self.health = game.config["start_health"]
        self.gold = game.config["start_gold"]
        self.lumber = game.config["start_lumber"]
        self.income = game.config["start_income"]
        self.levels = json.load(open("./levelup.json", "r"))
        self.id = i
        self.level = 0
        self.game = game
        self.units = []
        self.buildings = []
        self.spawn_queue = []
        self.opponent = None

        self.income_frequency = game.config["income_frequency"] * game.config["ticks_per_second"]
        self.income_counter = self.income_frequency

        self.spawn_x = 0 if i is 1 else game.map[0].shape[0]-1
        self.goal_x = game.map[0].shape[0]-1 if i is 1 else 0

        if i == 1:
            self.direction = 1
        elif i == 2:
            self.direction = -1

    def update(self):

        # Income Logic
        # Decrements counter each tick and fires income event when counter reaches 0
        self.income_counter -= 1
        if self.income_counter is 0:
            self.gold += self.income
            self.income_counter = self.income_frequency

        # Spawn Queue Logic
        # Spawn queued units
        if self.spawn_queue:
            self.spawn(self.spawn_queue.pop())

        # Process buildings
        for building in self.buildings:
            for opp_unit in self.opponent.units:
                success = building.shoot(opp_unit)
                if success:
                    break

        # Process units
        for unit in self.units:
            if unit.despawn:
                self.units.remove(unit)

            unit.move()

    def increase_gold(self, amount):
        self.gold += amount

    def levelup(self):
        #Lvelup logic
        next_level = self.levels[0]
        if self.gold >= next_level[0] and self.lumber >= next_level[1]:
            self.gold -= next_level[0]
            self.lumber -= next_level[1]
            self.level += 1
            self.levels.pop(0)
        else:
            print("Cannot afford levelup!")

    def build(self, x, y, building):

        # Restrict players from placing towers on mid area and on opponents side
        if self.direction == 1 and not all(i > x for i in self.game.mid):
            return False
        elif self.direction == -1 and not all(i < x for i in self.game.mid):
            return False

        # Ensure that there is no building already on this tile (using layer 4 (Building Player Layer)
        if self.game.map[4][x, y] != 0:
            return False

        # Check if can afford
        if self.gold >= building.gold_cost:
            self.gold -= building.gold_cost
        else:
            return False

        building = copy.copy(building)
        building.setup(self)
        building.x = x
        building.y = y

        # Update game state
        self.game.map[4][x, y] = building.id
        self.game.map[3][x, y] = self.id

        self.buildings.append(building)

    def spawn(self, data):
        idx = data[0] + 1
        type = data[1]

        # Check if can afford
        if self.gold >= type.gold_cost:
            self.gold -= type.gold_cost
        else:
            return False

        # Update income
        self.income += type.gold_cost * self.game.config["income_ratio"]

        spawn_x = self.spawn_x
        open_spawn_points = [np.where(self.game.map[1][spawn_x] == 0)[0]][0]

        if open_spawn_points.size == 0:
            #print("No spawn location for unit!")
            self.spawn_queue.append(data)
            return False

        spawn_y = np.random.choice(open_spawn_points)

        unit = copy.copy(type)
        unit.id = idx
        unit.setup(self)
        unit.x = spawn_x
        unit.y = spawn_y

        # Update game state
        self.game.map[1][spawn_x, spawn_y] = idx
        self.game.map[2][spawn_x, spawn_y] = self.id

        self.units.append(unit)

        return True


