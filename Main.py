
import Game
from utils import set_thread_name
set_thread_name("Main Thread")



g = Game.Game()
g.running = True
g.loop()