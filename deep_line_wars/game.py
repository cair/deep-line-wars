from deep_line_wars.config import default_config
import numpy as np
import pygame


class Game:

    def __init__(self, config: dict):
        self.config = default_config.copy()
        self.config.update(config)  # TODO is this deep update?

        self.map = Map(self)

        self.players = []

        self.graphics = Graphics(self) if self.config["gui"] else None

        self._fps_interval = None
        self._ups_interval = None
        self._terminal = None

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
        #self.map.reset()
        self._fps_interval = self.config["mechanics"]["fps"]
        self._ups_interval = self.config["mechanics"]["ups"]

    def render(self):
        if not self.graphics:
            return

        self.graphics.update()

    def update(self):
        pass


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

        self.cb_on_cell_change = []

        self.data = np.empty(shape=(self.height * self.width,), dtype=Cell)
        self.setup()

    def reset(self):
        for cell in self.data:
            cell.occupants.clear()

    def setup(self):
        """Compute center area."""
        center = self.width / 2
        center_area = [int(center), int(center - 1)] if center.is_integer() else [int(center)]

        """Construct center area."""
        for y in center_area:
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
        normal_type = set(range(0, self.width)) - set(p_0_base + p_1_base + center_area)
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


class Player:

    def __init__(self, game: Game):
        self.game = game

        self.units = []

    def update(self):
        for unit in self.units:
            unit.update()


class Unit:

    def __init__(self, spec):
        pass

    def update(self):
        pass


class Shop:

    def __init__(self, player: Player):
        self.player = player

    def purchase(self, unit: Unit):
        pass


if __name__ == "__main__":
    g = Game(config=dict())

    g.reset()
    while not g.terminal:
        g.update()
        g.render()
