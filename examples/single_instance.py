import time

from DeepLineWars.Game import Game
import random
if __name__ == "__main__":

    game = Game({
        "game": {
            "width": 15,
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
            "ups": 10,
            "income_ratio": 0.20,
            "kill_gold_ratio": 0.10
        },
        "gui": {
            "enabled": True,
            "draw_friendly": True,
            "minimal": True
        }
    })

    while True:
        for p in game.players:
            p.do_action(random.randint(0, 12))

        game.update()
        game.render()
        game.gui.draw_screen()


        time.sleep(.1)


