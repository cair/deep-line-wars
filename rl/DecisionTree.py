


class DecisionTree:


    def __init__(self):
        pass


class DecisionNode:

    def __init__(self, desc):
        self.description = desc
        self.children = []
        self.f = None

    def noop(self):
        print(self.description, " NO ACTION")





