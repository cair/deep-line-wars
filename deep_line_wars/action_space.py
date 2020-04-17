from __future__ import annotations

from deep_line_wars import entity
from deep_line_wars.shop import Shop


class BaseActionSpace:

    def __init__(self, game: 'Game'):
        self.game: 'Game' = game
        self.size = None
        self.actions = []

    def perform(self, a):
        if a < 0 or a >= self.size:
            raise ValueError("Out of bounds action %s when size of the action-space is %s" % (a, self.size))

        self.actions[a]()

    def build(self):
        method_list = [func for func in dir(self.__class__) if callable(getattr(self.__class__, func))]
        method_list = [getattr(self, x) for x in method_list if "__" not in str(x) and str(x) not in ["perform", "build"]]
        self.actions.extend(method_list)
        self.size = len(self.actions)


class StandardActionSpace(BaseActionSpace):

    def __init__(self, game: 'Game'):
        super().__init__(game)

        self.build()

    def cursor_left(self):
        self.game.selected_player.set_cursor(-1, 0)

    def cursor_right(self):
        self.game.selected_player.set_cursor(1, 0)

    def cursor_up(self):
        self.game.selected_player.set_cursor(0, -1)

    def cursor_down(self):
        self.game.selected_player.set_cursor(0, 1)

    def send_militia(self):
        self.game.shop.buy(
            self.game.selected_player,
            Shop.MILITIA,
            entity.Ground
        ).spawn(
            player=self.game.selected_player
        )

    def send_footman(self):
        self.game.shop.buy(
            self.game.selected_player,
            Shop.FOOTMAN,
            entity.Ground
        ).spawn(
            player=self.game.selected_player
        )

    def send_grunt(self):
        self.game.shop.buy(
            self.game.selected_player,
            Shop.GRUNT,
            entity.Ground
        ).spawn(
            player=self.game.selected_player
        )

    def send_armored_grunt(self):
        self.game.shop.buy(
            self.game.selected_player,
            Shop.ARMORED_GRUNT,
            entity.Ground
        ).spawn(
            player=self.game.selected_player
        )

    def build_basic_tower(self):
        self.game.shop.buy(
            self.game.selected_player,
            Shop.BASIC_TOWER,
            entity.Building
        ).spawn(
            player=self.game.selected_player,
            x=self.game.selected_player.virtual_cursor_x,
            y=self.game.selected_player.virtual_cursor_y
        )

    def build_fast_tower(self):
        self.game.shop.buy(
            self.game.selected_player,
            Shop.FAST_TOWER,
            entity.Building
        ).spawn(
            player=self.game.selected_player,
            x=self.game.selected_player.virtual_cursor_x,
            y=self.game.selected_player.virtual_cursor_y
        )

    def build_faster_tower(self):
        self.game.shop.buy(
            self.game.selected_player,
            Shop.FASTER_TOWER,
            entity.Building
        ).spawn(
            player=self.game.selected_player,
            x=self.game.selected_player.virtual_cursor_x,
            y=self.game.selected_player.virtual_cursor_y
        )

    def no_action(self):
        pass


class ContinousActionSpace(BaseActionSpace):

    def __init__(self, game):
        super().__init__(game)
        """
         action = a_idx
        action_intensity = a_intensity
        if action == 0:
            # Move Mouse X

            clipped_intensity = max(0, min(action_intensity, 1))
            self.virtual_cursor_x = int(self.game.width * clipped_intensity)

        elif action == 1:
            # Move Mouse Y
            clipped_intensity = max(0, min(action_intensity, 1))
            self.virtual_cursor_y = int(self.game.height * clipped_intensity)

        elif action == 2:
            # Unit send
            clipped_intensity = int(max(0, min(action_intensity, 3)))
            success = self.spawn(
                (clipped_intensity, self.game.unit_shop[clipped_intensity])
            )

        elif action == 3:
            # Building build
            clipped_intensity = int(3 * action_intensity)

            success = self.build(
                self.virtual_cursor_x,
                self.virtual_cursor_y,
                self.game.building_shop[clipped_intensity]
            )
        else:
            raise RuntimeError("Action %s is not part of the action-space" % action)

        
        """
