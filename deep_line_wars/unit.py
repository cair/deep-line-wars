import pygame
import cv2
from os.path import realpath, dirname, join
dir_path = dirname(realpath(__file__))


class Unit:

    def __init__(self, data):
        self.name = data["name"]
        self.health = data["health"]
        self.armor = data["armor"]
        self.speed = data["speed"]
        self.type = data["type"]
        self.icon_name = data["icon"]
        self.id = None
        self.gold_cost = data["gold_cost"]

        self.tick_speed = None
        self.tick_counter = None
        self.player = None
        self.despawn = False

        self.icon_image = cv2.imread(join(dir_path, "sprites/units/%s") % self.icon_name)
        self.icon_image = cv2.cvtColor(self.icon_image, cv2.COLOR_BGR2RGB)
        self.icon_image = cv2.resize(self.icon_image, (32, 32))
        self.icon_image = cv2.rotate(self.icon_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        self.level = data["level"]

        self.x = None
        self.y = None

        self.target_x = None
        self.target_y = None

        self.move_failures = 0

    def setup(self, player):
        self.player = player

        # Draw outline with correct color
        cv2.rectangle(self.icon_image, (0, 0), (32, 32), (255, 255, 255), 3)

        # Flip other player
        if self.player.direction == 1:
            self.icon_image = cv2.flip(self.icon_image, 0)

        self.tick_speed = self.player.game.ticks_per_second / self.speed
        self.tick_counter = self.tick_speed

    def damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            # Increase opponents gold with a ratio of what the unit was worth.
            self.player.opponent.increase_gold(self.gold_cost * self.player.game.config.mechanics.kill_gold_ratio)
            self.despawn = True

    def move(self):

        self.tick_counter -= 1
        if self.tick_counter <= 0:


            # Identify next position
            next_x = self.x + self.player.direction

            # If tile is occupied by friendly, try to find a path around it

            # If tile is occupied by enemy, try to find a path around it

            # If tile is occupied and there is not way around, destroy it!

            # Update position of the unit
            self.player.game.map[1][self.x, self.y] = 0
            self.player.game.map[2][self.x, self.y] = 0
            self.x += self.player.direction
            self.player.game.map[1][self.x, self.y] = self.id
            self.player.game.map[2][self.x, self.y] = self.player.id

            # If unit has reached its final destination
            if self.x == self.player.goal_x:
                # Unit has reached goal
                self.player.opponent.health -= 1
                self.despawn = True

            self.tick_counter = self.tick_speed

    def find_closest_gap(self, x, y):
        the_map = self.player.game.map[3][x]
        the_map_len = len(the_map)
        expand = 1

        for i in range(self.player.game.height):
            neg_expand = y - expand
            pos_expand = y + expand
            if neg_expand > 0 and the_map[neg_expand] == 0:
                return x, neg_expand
            elif pos_expand <= the_map_len and the_map[pos_expand] == 0:
                return x, pos_expand
            expand += 1

    def remove(self):
        # Unit has reached goal
        self.despawn = True
        self.player.game.map[1][self.x, self.y] = 0
        self.player.game.map[2][self.x, self.y] = 0


