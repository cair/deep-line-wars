import cv2

import pygame
import math
import random
from os.path import realpath, dirname, join
dir_path = dirname(realpath(__file__))


class Building:

    def __init__(self, data):
        self.name = data["name"]
        self.icon = data["icon"]
        self.health = data["health"]
        self.attack_min = data["attack_min"]
        self.attack_max = data["attack_max"]
        self.attack_pen = data["attack_pen"]
        self.attack_speed = data["attack_speed"]
        self.attack_range = data["attack_range"]
        self.id = data["id"]
        self.gold_cost = data["gold_cost"]

        self.icon_image = cv2.imread(join(dir_path, "sprites/buildings/%s") % self.icon)
        self.icon_image = cv2.cvtColor(self.icon_image, cv2.COLOR_BGR2RGB)
        self.icon_image = cv2.resize(self.icon_image, (32, 32))
        self.icon_image = cv2.rotate(self.icon_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        self.level = data["level"]

        self.player = None
        self.tick_speed = None
        self.tick_counter = None
        self.x = None
        self.y = None

    def setup(self, player):
        self.player = player

        # Draw outline with correct color
        cv2.rectangle(self.icon_image, (0, 0), (32, 32), (255, 255, 255), 2)

        self.tick_speed = self.player.game.ticks_per_second / self.attack_speed
        self.tick_counter = self.tick_speed

    def shoot(self, unit):
            # Wait for reload (shooting speed)
            self.tick_counter -= 1
            if self.tick_counter > 0:
                return True # Cannot shoot at all this tick, so its done

            self.tick_counter = self.tick_speed
            x2 = self.x
            x1 = unit.x
            y2 = self.y
            y1 = unit.y
            distance = math.hypot(x2 - x1, y2 - y1)

            if self.attack_range >= distance:
                # Can shoot
                # Attack-dmg - min(0, (armor - attack_pen))
                damage = random.randint(self.attack_min, self.attack_max) - min(0, unit.armor - self.attack_pen)
                unit.damage(damage)
                return True

            return False
