import multiprocessing
import time

import Game
from utils import set_thread_name



if __name__ == "__main__":
    set_thread_name("Main Thread")


    processes = 8

    for i in range(processes):
        g = Game.Game()
        g.running = True
        g.start()

