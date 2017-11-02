import random


class DecisionTree:


    def __init__(self):
        pass


class DecisionNode:

    def __init__(self, desc):
        self.description = desc
        self.children = []
        self.f = [DecisionNode.noop, self]
        self._next = None
        self.player = None
        self.game = None

    def has_next(self):
        return False if not self._next else True

    def next(self):
        return self._next

    def set_next(self, nxt):
        self._next = nxt

    @staticmethod
    def noop(node):
        node.set_next(node.random_decision())

    def process(self):
        self.f[0](self)

    def random_decision(self):
        try:
            choice = random.choice(self.children)
            return choice
        except:
            return None

    def info(self):
        print(self.description)






