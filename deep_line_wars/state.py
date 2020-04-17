import itertools
import numpy as np

from deep_line_wars.entity import Entity, Building


class State:

    SPAWN_AREA = 0x1
    CENTER_AREA = 0x2

    def __init__(self, game, width, height):
        self.game = game
        self.height = height
        self.width = width
        # Z = 0 - Environmental Layer
        # Z = 1 - Unit Layer
        # Z = 2 - Unit Player Layer
        # Z = 3 - Building Layer
        # Z = 4 - Building Player Layer
        self.grid = np.zeros((5, width, height), dtype=np.uint8)
        self.center_area = None
        self.spawn_area = [
            list(range(0, game.config.map.spawn_area_size)),
            list(range(width - game.config.map.spawn_area_size, width))
        ]

        center = width / 2
        self.center_area = [int(center), int(center - 1)] if center.is_integer() else [int(center)]

        self.static_tiles = []

        self.setup_environment()
        self.flipped = False  # Whether the state is flipped or not

    def setup_environment(self):

        for x in list(itertools.chain(*self.spawn_area)):
            self.grid[0, x] = State.SPAWN_AREA
            self.static_tiles.extend([(x, y, State.SPAWN_AREA) for y in range(0, self.height)])

        for x in self.center_area:
            self.grid[0, x] = State.CENTER_AREA
            self.static_tiles.extend([(x, y, State.CENTER_AREA) for y in range(0, self.height)])

    def update(self, entity, x, y):
        a, b = (3, 4) if entity.entity_type == Building else (0, 1)

        if entity.x is not None and entity.y is not None:
            self.grid[a, entity.x, entity.y] = 0
            self.grid[b, entity.x, entity.y] = 0
            entity.x = -1
            entity.y = -1

        if x is not None and y is not None:
            self.grid[a, x, y] = entity.id
            self.grid[b, x, y] = entity.player.id
            entity.x = x
            entity.y = y

    def free_spawn_points(self, player):
        items = []
        for x in self.spawn_area[player.id - 1]:
            items.extend([(x, y[0]) for y in np.argwhere(self.grid[1, x] == 0)])

        return items

    def reset(self):
        self.grid[1:] = 0
