

class Unit:

    def __init__(self, data):
        self.name = data["name"]
        self.health = data["health"]
        self.armor = data["armor"]
        self.speed = data["speed"]
        self.type = data["type"]