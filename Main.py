import Game
from utils import set_thread_name


if __name__ == "__main__":
    set_thread_name("Main Thread")
    processes = 8

    g = Game.Game()
    g.running = True
    g.loop()

    """for i in range(processes):
        g = Game.Game()
        g.running = True
        g.start()
    """

