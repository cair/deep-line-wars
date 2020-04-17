from deep_line_wars.gui import dummy, pygame


class Config:

    class Map:
        def __init__(
                self,
                spawn_area_size=1,
                mid_area_size=1
        ):
            self.spawn_area_size = spawn_area_size
            self.mid_area_size = mid_area_size

    class Mechanics:
        def __init__(self,
                     build_anywhere: bool = True,
                     start_health: int = 50,
                     start_gold: int = 50,
                     start_lumber: int = 0,
                     start_income: int = 20,
                     income_frequency: int = 10,
                     ticks_per_second: int = 10,
                     fps: int = 10,
                     ups: int = 10008000,
                     income_ratio: int = 0.20,
                     kill_gold_ratio: int = 0.10,
                     enemy_territory_decay=.10,
                     friendly_territory_decay=.0001
                     ):
            self.build_anywhere = build_anywhere
            self.start_health = start_health
            self.start_gold = start_gold
            self.start_lumber = start_lumber
            self.start_income = start_income
            self.income_frequency = income_frequency
            self.ticks_per_second = ticks_per_second
            self.fps = fps
            self.ups = ups
            self.income_ratio = income_ratio
            self.kill_gold_ratio = kill_gold_ratio
            self.enemy_territory_decay = enemy_territory_decay
            self.friendly_territory_decay = friendly_territory_decay

    class Game:
        def __init__(self,
                     width: int = None,
                     height: int = None,
                     tile_width=32,
                     tile_height=32
                     ):
            self.width = width
            self.height = height
            self.tile_width = tile_width
            self.tile_height = tile_height

    class GUI:

        def __init__(self,
                     engine=pygame.GUI,
                     draw_friendly: bool = True,
                     state_representation="RGB"  # RAW, RGB, L
                     ):
            self.engine = engine
            self.draw_friendly = draw_friendly
            self.state_representation = state_representation

    def __init__(self,
                 game: 'Game' = Game(),
                 mechanics: 'Mechanics' = Mechanics(),
                 gui: 'GUI' = GUI(),
                 map: 'Map' = Map()
                 ):

        self.game: 'Game' = game
        self.mechanics: 'Mechanics' = mechanics
        self.gui: 'GUI' = gui
        self.map: 'Map' = map
        self.width = None
        self.height = None
        self._size_is_set = False

    def set_size(self, w, h):
        self._size_is_set = True
        self.width = w
        self.height = h

    def validate(self):
        assert self._size_is_set, "Size must be set with the set_size function"





level_up = [
    [100, 0],
    [1000, 0],
    [10000, 0],
    [100000, 1]
]

upgrades = [
    dict()
]
