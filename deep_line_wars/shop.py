import typing
import logging

from deep_line_wars import entity
from deep_line_wars.entity import \
    Entity, \
    Militia, \
    Footman, \
    Grunt, \
    ArmoredGrunt, \
    BasicTower, \
    FastTower, \
    FasterTower

_LOGGER = logging.getLogger(__name__)


class Shop:
    MILITIA = 0x1
    FOOTMAN = 0x2
    GRUNT = 0x4
    ARMORED_GRUNT = 0x8

    BASIC_TOWER = 0x1
    FAST_TOWER = 0x2
    FASTER_TOWER = 0x4

    def __init__(self, game: 'Game'):
        self.game = game

        self.units: typing.Dict[int, typing.Type[Entity]] = {
            Shop.MILITIA: Militia,
            Shop.FOOTMAN: Footman,
            Shop.GRUNT: Grunt,
            Shop.ARMORED_GRUNT: ArmoredGrunt
        }

        self.buildings: typing.Dict[int, typing.Type[Entity]] = {
            Shop.BASIC_TOWER: BasicTower,
            Shop.FAST_TOWER: FastTower,
            Shop.FASTER_TOWER: FasterTower
        }

    def buy(self,
            player: 'Player',
            index,
            entity_type: typing.Union[entity.Ground, entity.Flying, entity.Building]
            ):

        if entity_type == entity.Ground or entity_type == entity.Flying:
            # Unit
            unit = self.units[index]
            return unit(player) if unit.can_afford(player) else Entity
        elif entity_type == entity.Building:
            # Building
            building = self.buildings[index]
            return building(player) if building.can_afford(player) else Entity
        else:
            _LOGGER.error("Invalid entity_type %s", entity_type)
