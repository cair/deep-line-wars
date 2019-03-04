import math
import random

from deep_line_wars.config import default_config
import numpy as np
import pygame


class Game:

    def __init__(self, config: dict):
        self.config = default_config.copy()
        self.config.update(config)  # TODO is this deep update?

        self.map = Map(self)

        self.players = [Player(self, player_side=_) for _ in range(2)]
        for p in self.players:
            p.post_init()

        self.graphics = Graphics(self) if self.config["gui"] else None

        self._fps_interval = None
        self._ups_interval = None
        self._terminal = None

        self._tick = None
        self._ticks_per_second = self.config["mechanics"]["ticks_per_second"]
        self._tick_income_frequency = self.config["mechanics"]["income_frequency"] * self._ticks_per_second

    @property
    def terminal(self):
        if self._terminal is True:
            return True

        self._terminal = self._is_terminal()
        return self._terminal

    def _is_terminal(self):
        any_terminal = False
        for p in self.players:
            if p.terminal:
                any_terminal = True
                break

        return any_terminal

    def get_winner(self):
        if not self.terminal:
            return None

        return next(x for x in self.players if not x.terminal)

    def reset(self):
        self._terminal = False
        self.map.reset()
        for p in self.players:
            p.reset()

        self._fps_interval = self.config["mechanics"]["fps"]
        self._ups_interval = self.config["mechanics"]["ups"]

        self._tick = 0

    def render(self):
        if not self.graphics:
            return

        self.graphics.update()

    def update(self):

        for player in self.players:

            """Yield income if its time."""
            if self._tick % self._tick_income_frequency == 0:
                player.update_income()

            """Update player state."""
            player.update()

        self._tick += 1

class Graphics:

    def __init__(self, game: Game):
        self.game = game

        self.g_width = self.game.map.width
        self.g_height = self.game.map.height

        self.tile_size = self.game.config["game"]["tile_size"]

        self.canvas_shape = (
            self.g_height * self.tile_size,
            self.g_width * self.tile_size,
        )

        self.has_window = self.game.config["gui"]["window"]

        self.updates_cells = []
        self.updates_rects = []  # TODO need this?

        self.rectangles = []

        for y in range(self.g_height):
            for x in range(self.g_width):
                self.rectangles.append(
                    pygame.Rect(
                        (x * self.tile_size, y * self.tile_size),
                        (self.tile_size, self.tile_size)
                    )
                )

        self.game.map.cb_on_cell_change.append(self.on_cell_change)

        if self.has_window:
            pygame.display.init()
            self.canvas = pygame.display.set_mode(
                (self.canvas_shape[1], self.canvas_shape[0]),
                0  # TODO NOFRAME OPENGL, HWSURFACE? DOUBLEBUF?
            )
        else:
            self.canvas = pygame.Surface((self.canvas_shape[1], self.canvas_shape[0]))

        # TODO
        self.sprites = {
            Ground: Type.setup(Ground, self),
            RedZone: Type.setup(RedZone, self),
            Center: Type.setup(Center, self)
        }

        self._init_canvas()

    def _init_canvas(self):
        """Construct grid."""

        for cell in self.game.map.data:
            self.draw_sprite(cell)

        if self.has_window:
            pygame.display.update()

    def draw_sprite(self, cell):
        sprite = self.sprites[cell.type]
        self.canvas.blit(sprite, self.rectangles[cell.i])


    def on_cell_change(self):
        pass

    def update(self):
        pass


class Type:
    color = None

    @staticmethod
    def setup(t, g: Graphics, borders=True):
        #  TODO game can request to use sprites instead of colors
        rect = pygame.Rect((0, 0, g.tile_size, g.tile_size))
        surf = pygame.Surface((g.tile_size, g.tile_size))

        if borders:
            surf.fill(t.border_color, rect)
            surf.fill(t.color, rect.inflate(-t.border_width*2, -t.border_width*2))
        else:
            surf.fill(t.color, rect)

        return surf


class Ground(Type):
    color = (0, 255, 0)
    border_color = (0, 0, 0)
    border_width = 1


class Center(Type):
    color = (0, 0, 255)
    border_color = (0, 0, 0)
    border_width = 1


class RedZone(Type):
    color = (255, 0, 0)
    border_color = (0, 0, 0)
    border_width = 1


class Map:

    def __init__(self, game: Game):
        self.game = game
        self.width = self.game.config["game"]["width"]
        self.height = self.game.config["game"]["height"]

        """Compute center area."""
        center = self.width / 2
        self.center_area = [int(center), int(center - 1)] if center.is_integer() else [int(center)]

        self.cb_on_cell_change = []

        self.data = np.empty(shape=(self.height * self.width,), dtype=Cell)
        self.setup()

    def reset(self):
        for cell in self.data:
            cell.occupants.clear()

    def setup(self):

        """Construct center area."""
        for y in self.center_area:
            for x in range(self.height):
                self._set_cell(x, y, Cell(self, x=x, y=y, type=Center))

        """Compute edges"""
        p_0_base = list(range(0, self.width))[0:self.game.config["game"]["base_size"]]
        p_1_base = list(range(0, self.width))[-self.game.config["game"]["base_size"]:]

        """Construct edges. (Red Zone)."""
        for y in p_0_base + p_1_base:
            for x in range(self.height):
                self._set_cell(x, y, Cell(self, x=x, y=y, type=RedZone))

        """Construct remaining cells to normal ground."""
        normal_type = set(range(0, self.width)) - set(p_0_base + p_1_base + self.center_area)
        for y in normal_type:
            for x in range(self.height):
                self._set_cell(x, y, Cell(self, x=x, y=y, type=Ground))

    def _set_cell(self, x, y, val):
        self.data[(x * self.width) + y] = val

    def cell(self, x, y):
        return self.data[y, x]

    def update(self):
        pass


class Cell:

    def __init__(self, map: Map, x: int, y: int, type):
        self.map = map
        self.x = x
        self.y = y
        self.i = (self.x * self.map.width) + self.y
        self.type = type

        self.occupants = []

    def has_occupants(self):
        return len(self.occupants) > 0


class Cursor:

    def __init__(self, player):
        self.x = np.array([player.territory.start_x, player.territory.end_x]).mean()
        self.y = np.array([player.territory.start_y, player.territory.end_y]).mean()

        self.map = player.game.map

    def set_relative(self, xt, yt):
        new_x = self.x + xt
        new_y = self.y + yt

        self.set_position(new_x, new_y)

    def set_position(self, x, y):
        if x > self.map.width:
            self.x = 0
        elif x < 0:
            self.x = self.map.width - 1
        else:
            self.x = x

        if y > self.map.height:
            self.y = 0
        elif y < 0:
            self.y = self.map.height - 1
        else:
            self.y = 0


class Territory:

    def __init__(self, player):
        self._player_side = player.player_side
        self.map = player.game.map

        if self._player_side == 0:
            self.start_x = 0
            self.end_x = min(self.map.center_area)
        elif self._player_side == 1:
            self.start_x = max(self.map.center_area) + 1
            self.end_x = self.map.width - 1
        else:
            raise NotImplementedError("Player with size id 2 should not exists. "
                                      "This may happen when there are more then 2 players")

        self.start_y = 0
        self.end_y = self.map.height - 1

        self.territory_rect = (
            self.start_x, self.end_x,
            self.start_y, self.end_y
        )


class Player:

    def __init__(self, game: Game, player_side):
        self.game = game
        self.units = []
        self.health = None
        self.income = None
        self.gold = None
        self.lumber = None
        self._terminal = None
        self.player_side = player_side

        self.opponent = None

        self._income_ratio = game.config["mechanics"]["income_ratio"]
        self._kill_gold_ratio = game.config["mechanics"]["kill_gold_ratio"]

        self.territory = Territory(self)
        self.cursor = Cursor(self)

    def post_init(self):
        self.opponent = self.game.players[0] if self.player_side == 1 else self.game.players[1]

    @property
    def terminal(self):
        if self._terminal:
            return True

        return self.health == 0

    def update(self):


        for unit in self.units:
            unit.update()

        if self.spawn_queue:
            self.spawn(self.spawn_queue.pop())

        for unit in self.units:

            if unit.stationary:
                for opp_unit in self.opponent.units:
                    success = unit.shoot(opp_unit)
                    if success:
                        break

            if unit.despawn:
                unit.remove()
                self.units.remove(unit)
            else:
                unit.move()

    def update_income(self):
        self.gold += self.income

    def reset(self):
        self.units.clear()
        self.health = self.game.config["mechanics"]["start_health"]
        self.income = self.game.config["mechanics"]["start_income"]
        self.gold = self.game.config["mechanics"]["start_gold"]
        self.lumber = self.game.config["mechanics"]["start_lumber"]
        self._terminal = False

    def do_action(self, action_index):
        pass

    def sample_action(self):
        """Samples a random index from uniform distribution."""
        """
        0 - Mouse Left
        1 - Mouse Right
        2 - Mouse Up
        3 - Mouse Down
        4 - Build Tower 0
        5 - Build Tower 1
        6 - Build Tower 2
        7 - Build Tower 3
        8 - Build Unit 0
        9 - Build Unit 1
        10 - Build Unit 2
        11 - Build Unit 3
        12 - * Reserved / NOOP *
        """
        return random.randint(0, 12)


class Unit:

    def __init__(self, spec):
        self.cell = None

        self.id = None

        self.type = None

        self.player = None

        self.health = None
        self.armor = None

        self.attack_min = None
        self.attack_max = None

        self.attack_speed = None
        self.attack_pen = None
        self.attack_range = None

        self.speed = None

        self.stationary = None

        self.level = None
        self.gold_cost = None
        self.lumber_cost = None

        self.despawn = False
        self.attack_ticks = None
        self._tick_counter_attack = 0

        self.parse_spec(spec)

    def assign(self, player):
        self.player = player
        self.despawn = False

    def parse_spec(self, spec):
        for k, v in spec.items():
            getattr(self, k, v)

        self.attack_ticks = self.player.game.ticks_per_second / self.attack_speed

    def damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            # Increase opponents gold with a ratio of what the unit was worth.
            self.player.opponent.increase_gold(self.gold_cost * self.player.game.config.mechanics.kill_gold_ratio)
            self.despawn = True

    def shoot(self, unit):
        # Wait for reload (shooting speed)
        self._tick_counter_attack -= 1
        if self._tick_counter_attack > 0:
            return True # Cannot shoot at all this tick, so its done

        self._tick_counter_attack = self.attack_ticks

        distance = math.hypot(self.cell.x - unit.cell.x, self.cell.y - unit.cell.y)

        if self.attack_range >= distance:
            damage = random.randint(self.attack_min, self.attack_max) - min(0, unit.armor - self.attack_pen)
            unit.damage(damage)
            return True

        return False

    def update(self):

        self.tick_counter -= 1
        if self.tick_counter <= 0:
            # Identify next position
            next_x = self.x + self.player.direction


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




    def destroy(self):
        self.cell.occupants.remove(self)
        self.player.shop.add(self)
        self.player = None
        self.despawn = True


class Shop:

    def __init__(self, player: Player):
        self.player = player

    def purchase(self, unit: Unit):
        pass


if __name__ == "__main__":
    g = Game(config=dict())

    g.reset()
    while not g.terminal:

        for p in g.players:
            p.do_action(p.sample_action())

        g.update()
        g.render()
