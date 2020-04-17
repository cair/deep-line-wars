import random

import time

from deep_line_wars.config import Config
from deep_line_wars.game import Game
from deep_line_wars.gui import pygame, dummy, opencv

if __name__ == "__main__":

    g = Game(20, 5, Config(
        gui=Config.GUI(
            engine=pygame.GUI
        ),
        mechanics=Config.Mechanics(
            ups=-1,
            fps=-1
        )
    ))

    # Time: 18.240913152694702 - OpenCV
    # Time: 0.30150508880615234 - Dummy
    # Time: 38.14940595626831 - PyGame

    s = time.time()
    for i in range(10):
        g.reset()
        while not g.is_terminal():

            # Perform Action
            a1 = random.randint(0, g.get_action_space()-1)
            g.step(a1)
            g.flip_player()

            g.render()
            g.render_window()
            """
            a2 = random.randint(0, g.get_action_space()-1)
            g.step(a2)
            g.flip_player()

            g.render()
            g.render_window()"""







    print("Time: %s" % (time.time() - s))

    """
    g = Game(dict(gui=pygame.GUI))

    s = time.time()
    for i in range(10):
        g.reset()
        while not g.is_terminal():

            # Perform Action
            a1 = random.randint(0, g.get_action_space()-1)
            g.step(a1)
            g.flip_player()

            a2 = random.randint(0, g.get_action_space()-1)
            g.step(a2)
            g.flip_player()

            g.render()

    print("Time: %s" % (time.time() - s))
    """
