import random

import math
import typing
import cv2


from .utils import get_icon


class Entity:

    class Attack:

        def __init__(self, a_min, a_max, a_pen, a_speed, a_range):
            self.min = a_min
            self.max = a_max
            self.pen = a_pen
            self.speed = a_speed
            self.range = a_range

    id: int = None
    health: int = None
    armor: int = None
    cost_gold: int = None
    drop_gold: int = None
    attack: typing.Union[None, 'Attack'] = None
    entity_type: typing.Union['Flying', 'Ground', 'Building'] = None
    speed: int = None
    name: str = None
    icon_template: str = None
    level: str = None

    def __init__(self, player: 'Player'):
        self.player: 'Player' = player
        self.x = None
        self.y = None

        self.tick_speed = 0 if self.speed == 0 else self.player.game.ticks_per_second / self.speed
        self.tick_counter = self.tick_speed
        self.despawn: bool = False

        # Draw outline with correct color
        self.icon_image = cv2.rectangle(self.icon_template, (0, 0), (32, 32), self.player.color, 3)

        # Flip other player
        if self.player.direction == 1:
            self.icon_image = cv2.flip(self.icon_image, 0)

    def update(self):
        pass

    def remove(self):
        # Unit has reached goal
        self.despawn = True
        self.player.game.state.update(self, x=None, y=None)

    def damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            # Increase opponents gold with a ratio of what the unit was worth.
            self.player.opponent.increase_gold(self.cost_gold * self.player.game.config.mechanics.kill_gold_ratio)
            self.despawn = True

    @classmethod
    def can_afford(cls, player: 'Player') -> bool:
        return player.gold > cls.cost_gold

    @classmethod
    def spawn(cls, player: 'Player', x: int = None, y: int = None):
        if cls == Entity:
            return False

        # Check if can afford
        if player.gold < cls.cost_gold:
            return False

        player.gold -= cls.cost_gold

        # Update income
        player.income += cls.cost_gold * player.game.config.mechanics.income_ratio

        if not x and not y:
            free_spawn_points = player.game.state.free_spawn_points(player)

            if len(free_spawn_points) == 0:
                player.spawn_queue.append(cls)
                return False

            spawn_x, spawn_y = random.choice(free_spawn_points)
        else:
            spawn_x = x
            spawn_y = y

        entity = cls(player)
        player.game.state.update(entity, x=spawn_x, y=spawn_y)

        entity.x = spawn_x
        entity.y = spawn_y

        player.units.append(entity)

        return entity


class Ground(Entity):

    def update(self):
        self.move()

    def move(self):

        if self.tick_counter > 0:
            self.tick_counter -= 1
            return
        else:
            # Reset tick-counter
            self.tick_counter = self.tick_speed

        # If tile is occupied by friendly, try to find a path around it

        # If tile is occupied by enemy, try to find a path around it

        # If tile is occupied and there is not way around, destroy it!

        # If unit has reached its final destination
        next_x = self.x + self.player.direction

        if next_x == self.player.goal_x:
            # Unit has reached goal
            self.player.opponent.health -= 1
            self.despawn = True

        # Update position of the unit
        self.player.game.state.update(self, x=next_x, y=self.y)

    def shoot(self):
        pass


class Flying(Entity):
    pass


class Building(Entity):

    def __init__(self, player: 'Player'):
        super().__init__(player)
        self.enemy_territory: bool = False

    def update(self):
        # Process buildings
        for opp_unit in self.player.opponent.units:
            success = self.shoot(opp_unit)
            if success:
                break

        # Decay health for buildings on enemy territory
        self.decay()


    def move(self):
        pass

    @classmethod
    def spawn(cls, player: 'Player', x: int = None, y: int = None):

        # Ensure that there is no building already on this tile (using layer 4 (Building Player Layer)
        if player.game.state.grid[4, x, y] != 0 or (x == 0 or x == player.game.width - 1):
            return False

        over_center = (player.direction == 1 and not all(i > x for i in player.game.state.center_area) ) or (
                player.direction == -1 and not all(i < x for i in player.game.state.center_area)
        )

        # Restrict players from placing towers on mid area and on opponents sides
        if over_center and not player.game.config.mechanics.build_anywhere:
            return False

        entity = super().spawn(player, x, y)

        if entity and over_center:
            entity.enemy_territory = True

        return entity

    def decay(self):
        if self.enemy_territory:
            self.health -= (self.__class__.health * self.player.game.config.mechanics.enemy_territory_decay)
        else:
            self.health -= (self.__class__.health * self.player.game.config.mechanics.friendly_territory_decay)

        if self.health <= 0:
            self.despawn = True

    def shoot(self, unit):
        # Wait for reload (shooting speed)
        self.tick_counter -= 1
        if self.tick_counter > 0:
            return True  # Cannot shoot at all this tick, so its done

        self.tick_counter = self.tick_speed
        distance = math.hypot(self.x - unit.x, self.y - unit.y)

        if self.attack.range >= distance:
            # Can shoot
            # Attack-dmg - min(0, (armor - attack_pen))
            damage = random.randint(self.attack.min, self.attack.max) - min(0, unit.armor - self.attack.pen)
            unit.damage(damage)
            return True

        return False


class BasicTower(Building):
    entity_type = Building
    id = 1
    name = "Basic-Tower"
    icon_template = get_icon("sprites/buildings/tower_1.png")
    health = 100
    speed = 0
    armor = 0
    attack = Entity.Attack(
        a_min=2,
        a_max=4,
        a_pen=2,
        a_speed=3,
        a_range=3
    )
    level = 0
    cost_gold = 10
    drop_gold = 0


class FastTower(Building):
    entity_type = Building
    id = 2
    name = "Fast-Tower"
    icon_template = get_icon("sprites/buildings/tower_2.png")
    health = 100
    speed = 0
    armor = 0
    attack = Entity.Attack(
        a_min=2,
        a_max=4,
        a_pen=2,
        a_speed=4,
        a_range=3
    )
    level = 0
    cost_gold = 20
    drop_gold = 0


class FasterTower(Building):
    entity_type = Building
    id = 3
    name = "Faster-Tower"
    icon_template = get_icon("sprites/buildings/lazer_tower.png")
    health = 100
    speed = 0
    armor = 0
    attack = Entity.Attack(
        a_min=4,
        a_max=6,
        a_pen=3,
        a_speed=5,
        a_range=3
    )
    level = 1
    cost_gold = 30
    drop_gold = 0


class Militia(Ground):
    entity_type = Ground
    id = 1
    name = "Militia"
    icon_template = get_icon("sprites/units/militia.png")
    health = 40
    speed = 1
    armor = 2
    attack = None
    level = 0
    cost_gold = 10
    drop_gold = 1


class Footman(Ground):
    entity_type = Ground
    id = 2
    name = "Footman"
    icon_template = get_icon("sprites/units/footman.png")
    health = 80
    speed = 1
    armor = 4
    attack = None
    level = 0
    cost_gold = 20
    drop_gold = 1


class Grunt(Ground):
    entity_type = Ground
    id = 3
    name = "Grunt"
    icon_template = get_icon("sprites/units/grunt.png")
    health = 140
    speed = 1
    armor = 4
    attack = None
    level = 0
    cost_gold = 40
    drop_gold = 1


class ArmoredGrunt(Ground):
    entity_type = Ground
    id = 4
    name = "Armored Grunt"
    icon_template = get_icon("sprites/units/armored_grunt.png")
    health = 190
    speed = 1.2
    armor = 6
    attack = None
    level = 0
    cost_gold = 100
    drop_gold = 1

