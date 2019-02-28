from deep_line_wars.config import default_config
import numpy as np
import pygame


class Game:

    def __init__(self, config: dict):
        self.config = default_config.copy()
        self.config.update(config)  # TODO is this deep update?

        self.map = Map(self)

        self.fps_interval = self.config["mechanics"]["fps"]
        self.ups_interval = self.config["mechanics"]["ups"]

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

        self.has_window = None

        self.updates_cells = []
        self.updates_rects = [] # TODO need this?

        self.rectangles = []

        for x in range(self.g_width):
            for y in range(self.g_height):
                self.rectangles.append(
                    pygame.Rect(
                        (x*self.tile_size, y*self.tile_size),
                        (self.tile_size, self.tile_size)
                    )
                )

        self.game.map.cb_on_cell_change.append(self.on_cell_change)

        if self.has_window:
            pygame.display.init()
            self.canvas = pygame.display.set_mode(
                (self.canvas_shape[1], self.canvas_shape[0]),
                0 # TODO NOFRAME OPENGL, HWSURFACE? DOUBLEBUF?
            )
        else:
            self.canvas = pygame.Surface((self.canvas_shape[1], self.canvas_shape[0]))

        # TODO
        self.sprites = {
            Type.Ground: None,
            Type.RedZone: None,
            Type.Center: None
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

    Ground = dict(
        color=(0, 255, 0)
    )

    Center = dict(
        color=(0, 0, 255)
    )

    RedZone = dict(
        color=(255, 0, 0)
    )


class Map:

    def __init__(self, game: Game):
        self.game = game
        self.width = self.game.config["game"]["width"]
        self.height = self.game.config["game"]["height"]

        self.cb_on_cell_change = []

        self.data = np.empty(shape=(self.height * self.width, ), dtype=Cell)
        self.setup()

    def setup(self):
        """Compute center area."""
        center = self.width / 2
        center_area = [int(center), int(center - 1)] if center.is_integer() else [int(center)]

        """Construct center area."""
        for y in center_area:
            for x in range(self.width):
                self._set_cell(x, y, Cell(self, x=y, y=y, type=Type.Center))

        """Compute edges"""
        p_0_base = list(range(0, self.width))[0:self.game.config["game"]["base_size"]]
        p_1_base = list(range(0, self.width))[-self.game.config["game"]["base_size"]:]

        """Construct edges. (Red Zone)."""
        for y in p_0_base + p_1_base:
                for x in range(self.width):
                    self._set_cell(x, y, Cell(self, x=x, y=y, type=Type.RedZone))

        """Construct remaining cells to normal ground."""
        normal_type = set(range(0, self.width)) - set(p_0_base + p_1_base + center_area)
        for y in normal_type:
            for x in range(self.width):
                self._set_cell(x, y, Cell(self, x=x, y=y, type=Type.Ground))

    def _set_cell(self, x, y, val):
        self.data[x * self.height + y] = val

    def cell(self, x, y):
        return self.data[y, x]

    def update(self):
        pass


class Cell:

    def __init__(self, map: Map, x: int, y: int, type: dict):
        self.map = map
        self.x = x
        self.y = y
        self.i = self.x * self.map.height + self.y
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
    g.update()
