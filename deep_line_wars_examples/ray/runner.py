import cv2
import os
import random
import time

import ray
import threading

from flatbuffers.encode import np

from deep_line_wars.game import Game

ray.init()
from deep_line_wars.gui import opencv, pygame


@ray.remote
class RayGame:

    def __init__(self, demonstration=False):
        os.environ["MKL_NUM_THREADS"] = "1"
        self.env = Game(dict(gui=opencv.GUI))
        self.ups = 0

        self.demonstration = demonstration
        if demonstration:
            self.demo_gui = pygame.GUI(self.env)
            self.orginal_gui = self.env.gui

        t1 = threading.Thread(target=self.ups_counter)
        t1.start()

    def ups_counter(self):
        while True:
            time.sleep(1.0)
            #print("UPS: ", self.ups)
            self.ups = 0

    def step(self, action):
        state, reward, terminal, info = self.env.step(action=action)
        self.env.flip_player()
        self.ups += 1

        if self.demonstration:
            self.env.gui = self.demo_gui
            self.env.render()
            self.env.render_window()
            self.env.gui = self.orginal_gui

        if terminal:
            self.reset()

    def reset(self):
        self.env.reset()


if __name__ == "__main__":

    actors = [RayGame.remote(demonstration=False) for _ in range(16)]
    actors.append(RayGame.remote(demonstration=True))

    info_env = Game(dict(gui=pygame.GUI))

    while True:
        for actor in actors:
            a = random.randint(0, info_env.get_action_space()-1)
            actor.step.remote(a)


