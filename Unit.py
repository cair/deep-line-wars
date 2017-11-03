import pygame
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

        self.icon_image = pygame.image.load(join(dir_path, "sprites/units/%s") % self.icon_name)
        self.level = data["level"]

        self.x = None
        self.y = None

    def setup(self, player):
        self.player = player
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
            # TODO pathfinding
            # Update game state
            self.player.game.map[1][self.x, self.y] = 0
            self.player.game.map[2][self.x, self.y] = 0

            self.x += self.player.direction

            # Update game state (again)
            self.player.game.map[1][self.x, self.y] = self.id
            self.player.game.map[2][self.x, self.y] = self.player.id

            if self.x == self.player.goal_x:
                # Unit has reached goal
                self.player.opponent.health -= 1
                self.despawn = True

            self.tick_counter = self.tick_speed

    def remove(self):
        # Unit has reached goal
        #self.player.opponent.health -= 1
        self.despawn = True
        self.player.game.map[1][self.x, self.y] = 0
        self.player.game.map[2][self.x, self.y] = 0


