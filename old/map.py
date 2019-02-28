class Cell:
    RED_ZONE = 1
    GROUND = 2
    CENTER = 3

    def __init__(self, map, x, y):
        self.map = map
        self.x = x
        self.y = y
        self.i = self.x * self.map.height + self.y
        self.type = None
        self.occupants = []

    def has_occupants(self):
        return len(self.occupants) > 0


class Map:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data = [
            Cell(self, x, y)
            for x in range(self.width)
            for y in range(self.height)
        ]
        self.on_change_callbacks = []

    def setup(self):
        # TODO --
        env_map = self.map[0]
        edges = [0, env_map.shape[0] - 1]

        # Set edges to "goal type"
        for edge in edges:
            for i in range(env_map[edge].shape[0]):
                env_map[edge][i] = 1

        # Set mid to "mid type"
        center = env_map.shape[0] / 2
        center_area = [int(center), int(center - 1)] if center.is_integer() else [int(center)]

        for center_item in center_area:
            for i in range(env_map[center_item].shape[0]):
                env_map[center_item][i] = 2

        self.center_area = center_area


def add_on_change_callback(self, cb):
        self.on_change_callbacks.append(cb)

    def cell(self, x, y):
        return self.data[x * self.height + y]

    def move_relative(self, agent, x, y):
        return self.move(agent, agent.cell.x + x, agent.cell.y + y)

    def move(self, agent, x, y):
        pass
        # TODO
        """if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return Grid.MOVE_WALL_COLLISION

        cell = self.grid[y, x]

        if cell.occupant and cell.occupant != agent:
            return Grid.MOVE_AGENT_COLLISION
        else:
            agent.cell = cell
            cell.occupant = agent
            return Grid.MOVE_OK"""
