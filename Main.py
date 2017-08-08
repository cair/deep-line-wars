
import Game
from utils import set_thread_name


if __name__ == "__main__":
    set_thread_name("Main Thread")
    g = Game.Game()
    g.running = True
    g.loop()
