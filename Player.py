import json
import copy
import random

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
        self.ai = None

        self.stat_spawn_counter = 0

        self.income_frequency = game.config["income_frequency"] * game.config["ticks_per_second"]
        self.income_counter = self.income_frequency

        self.spawn_x = 0 if i is 1 else game.map[0].shape[0]-1
        self.goal_x = game.map[0].shape[0]-1 if i is 1 else 0

        self.action_space = [
            #{"action": "Select Building 0", "type": "building_select", "value": 0},
            #{"action": "Select Building 1", "type": "building_select", "value": 1},
            #{"action": "Select Building 2", "type": "building_select", "value": 2},
            #{"action": "Select Unit 0", "type": "unit_select", "value": 0},
            #{"action": "Select Unit 1", "type": "unit_select", "value": 1},
            #{"action": "Select Unit 2", "type": "unit_select", "value": 2},
            #{"action": "Select Unit 3", "type": "unit_select", "value": 3},
            {"action": "Move Cursor Up", "type": "cursor_y", "value": -1},
            {"action": "Move Cursor Down", "type": "cursor_y", "value": 1},
            {"action": "Move Cursor Right", "type": "cursor_x", "value": 1},
            {"action": "Move Cursor Left", "type": "cursor_x", "value": -1},
            {"action": "Purchase Unit 0", "type": "purchase_unit", "value": 0},
            {"action": "Purchase Unit 1", "type": "purchase_unit", "value": 1},
            {"action": "Purchase Unit 2", "type": "purchase_unit", "value": 2},
            {"action": "Purchase Unit 3", "type": "purchase_unit", "value": 3},
            {"action": "Purchase Building 0", "type": "purchase_building", "value": 0},
            {"action": "Purchase Building 1", "type": "purchase_building", "value": 1},
            {"action": "Purchase Building 2", "type": "purchase_building", "value": 2},
            {"action": "Purchase Building 3", "type": "purchase_building", "value": 3},
        ]

        self.virtual_cursor_x = self.spawn_x
        self.virtual_cursor_y = 0


        if i == 1:
            self.direction = 1
        elif i == 2:
            self.direction = -1


    def available_buildings(self):
        return [b for b in self.game.building_shop if b.level <= self.level]

    def rel_pos_to_abs(self, x, y):
        if self.direction == -1:
            return self.game.config["width"] - x - 1, y

        return x, y

    def get_building_idx(self, idx, last_if_error=True):
        avb = self.available_buildings()
        try:
            return avb[idx]
        except IndexError as e:
            if last_if_error:
                return avb.pop()

    def get_actions(self, idx):
        return self.action_space[idx]

    def get_random_action(self):
        return random.choice(self.action_space)

    def get_score(self):
        return (self.stat_spawn_counter + self.income) * (100 * self.level)

    def can_afford_unit(self, u):
        if u.gold_cost <= self.gold:
            return True
        return False


    def available_units(self):
        return [u for u in self.game.unit_shop if u.level <= self.level]

    def can_afford_idx(self, idx):
        available_buildings = self.available_buildings()
        if idx >= len(available_buildings):
            if available_buildings.pop().gold_cost > self.gold:
                return False
            return True

        if available_buildings[idx].gold_cost > self.gold:
            return False

        return True

    def do_generic_action(self, a):
        # 0 = spawn
        # 1 = build

        if a == 0:
            # Spawn random unit
            available = [u for u in self.available_units() if self.can_afford_unit(u)]
            if len(available) == 0:
                return -0.1

            ridx = random.randint(0, len(available) - 1)
            self.spawn((ridx, available[ridx]))
            return 1
        elif a == 1:
            # build building at random loc
            available = [b for idx, b in enumerate(self.available_buildings()) if self.can_afford_idx(idx)]
            if len(available) == 0:
                return -0.1

            ridx = random.randint(0, len(available) - 1)
            r_x = random.randint(0, (self.game.config["width"] / 2) - 1)
            r_y = random.randint(0, self.game.config["height"] - 1)

            r_x, r_y = self.rel_pos_to_abs(r_x, r_y)
            self.build(r_x, r_y, available[ridx])
            return 1





    def do_action(self, aidx):
        a = self.action_space[aidx]
        #if a["type"] == "building_select":
        #    self.game.gui.selected_building = a["value"]
        #if a["type"] == "unit_select":
        #    self.game.gui.selected_unit = a["value"]
        try:
            if a["type"] == "cursor_y":
                prev = self.virtual_cursor_y
                self.virtual_cursor_y = max(min(self.game.config["height"] - 1, self.virtual_cursor_y + a["value"]), 0)
                if self.virtual_cursor_y == prev:
                    return -1
                return 0.1
            elif a["type"] == "cursor_x":
                prev = self.virtual_cursor_x
                self.virtual_cursor_x = max(min(self.game.config["width"] - 1, self.virtual_cursor_x + a["value"]), 1)
                if self.virtual_cursor_x == prev:
                    return -1

                return 0.1
            elif a["type"] == "purchase_unit":
                succ = self.spawn(
                    (a["value"], self.game.unit_shop[a["value"]])
                )
                return 0.1 if succ else -1
            elif a["type"] == "purchase_building":
                succ = self.build(
                    self.virtual_cursor_x,
                    self.virtual_cursor_y,
                    self.game.building_shop[a["value"]]
                )
                return 0.1 if succ else -1
        except IndexError as e:
            return -1  # Punish invalid move

        return None

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

    def enemy_unit_reached_base(self, u):

        if self.direction == 1 and u.x < self.game.mid[0]:
            return True
        elif self.direction == -1 and u.x > self.game.mid[0]:

            return True
        else:
            return False

    def build(self, x, y, building):

        # Restrict players from placing towers on mid area and on opponents side
        if self.direction == 1 and not all(i > x for i in self.game.mid) and not self.game.config["build_anywhere"]:
            return False
        elif self.direction == -1 and not all(i < x for i in self.game.mid) and not self.game.config["build_anywhere"]:
            return False
        elif x == 0 or x == self.game.config["width"] - 1:
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

        return True

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


