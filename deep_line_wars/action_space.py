

class BaseActionSpace:

    def __init__(self, game):
        self.game = game
        self.size = None
        self.actions = []

    @property
    def shape(self):
        return self.size,

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

    def __init__(self, game):
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
        self.game.selected_player.spawn((0, self.game.unit_shop[0]))

    def send_footman(self):
        self.game.selected_player.spawn((1, self.game.unit_shop[1]))

    def send_grunt(self):
        self.game.selected_player.spawn((2, self.game.unit_shop[2]))

    def send_armored_grunt(self):
        self.game.selected_player.spawn((3, self.game.unit_shop[3]))

    def build_basic_tower(self):
        self.game.selected_player.build(
            self.game.selected_player.virtual_cursor_x,
            self.game.selected_player.virtual_cursor_y,
            self.game.building_shop[0]
        )

    def build_fast_tower(self):
        self.game.selected_player.build(
            self.game.selected_player.virtual_cursor_x,
            self.game.selected_player.virtual_cursor_y,
            self.game.building_shop[1]
        )

    def build_faster_tower(self):
        self.game.selected_player.build(
            self.game.selected_player.virtual_cursor_x,
            self.game.selected_player.virtual_cursor_y,
            self.game.building_shop[2]
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
