import Game
from utils import set_thread_name
import os

if __name__ == "__main__":
    if os.name == 'posix':
        set_thread_name("Main Thread")


    processes = 8

    for i in range(processes):
        g = Game.Game()
        g.running = True
        g.loop()

