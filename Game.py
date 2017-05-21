import json
import numpy as np

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


class Game:

    def __init__(self):
        self.running = False
        self.config = json.load(open("./config.json", "r"))
        self.map = np.zeros((3, self.config["height"], self.config["width"]))

        print(matprint(self.map[0]))
        print(self.config)



    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def summary(self):
        print("TODO, Summary")

    def loop(self):
        while self.running:
            self.update()
            self.render()

        self.summary()

    def update(self):
        pass

    def render(self):
        pass