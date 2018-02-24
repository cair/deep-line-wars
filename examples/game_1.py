import random

import numpy as np
from PIL import Image
from DeepLineWars.Game import Game
import uuid

class GameInstance:

    @staticmethod
    def start(data_queue):
        g = GameInstance(data_queue)
        g.loop()
        return True

    def get_stacked_state(self, swapaxes=False):
        if len(self.states) > self.stack:
            if swapaxes:
                return np.swapaxes(np.array(self.states[-1 * self.stack:]), 0, 2)
            else:
                return np.array(self.states[-1 * self.stack:])

        return None

    def __init__(self, data_queue):
        self.id = uuid.uuid4()
        print("Game %s - Start" % self.id)
        self.data_queue = data_queue
        self.game = Game({
            "game": {
                "width": 11,
                "height": 11,
                "tile_width": 32,
                "tile_height": 32
            },
            "mechanics": {
                "complexity": {
                    "build_anywhere": False
                },
                "start_health": 50,
                "start_gold": 100,
                "start_lumber": 0,
                "start_income": 20,
                "income_frequency": 10,
                "ticks_per_second": 20,
                "fps": 10,
                "ups": 10008000,
                "income_ratio": 0.20,
                "kill_gold_ratio": 0.10
            },
            "gui": {
                "enabled": True,
                "draw_friendly": True,
                "minimal": True
            }
        })

        self.states = list()
        self.experience_replay = list()
        self.s0 = None
        self.player_1 = self.game.players[0]
        self.player_2 = self.game.players[1]
        self.episode = 1
        self.representation = "image_grayscaled"
        self.running = False
        self.stack = 4
        self.num_ticks = 10
        self.tick_limit = 30000

    def loop(self):
        self.running = True
        t = 0
        while self.running:

            # Do action
            self.player_1.do_action(random.randint(0, 12))
            self.player_2.do_action(random.randint(0, 12))

            # Process game
            for i in range(self.num_ticks):
                self.game.update()
                t += 1

            # Update image state
            self.game.render()

            # Retrieve state, add to list of states,
            s1 = self.game.get_state(representation=self.representation)
            self.states.append(s1)
            self.s0 = s1

            # Terminal State, Reset Game
            if self.game.is_terminal() or t >= self.tick_limit:
                self.game.reset()
                print("Game %s - %s#%s" % (self.id, self.episode, t))
                self.episode += 1

                if t < self.tick_limit:
                    self.data_queue.put(self.states)
                self.states.clear()
                t = 0


if __name__ == "__main__":
    import multiprocessing
    import threading




    n_proc = 10
    processes = []
    data_queue = multiprocessing.Queue()

    def on_data():
        while True:
            data = data_queue.get(block=True)
            #print(data)

    t = threading.Thread(target=on_data, args=())
    t.start()

    g = GameInstance(data_queue)
    g.loop()

    with multiprocessing.Pool(processes=n_proc):
        for n in range(n_proc):
            p = multiprocessing.Process(target=GameInstance.start, args=(data_queue, ))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()




