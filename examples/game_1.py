import random

import time

from DeepLineWars.Game import Game

g = Game({
    "game": {
        "width": 11,
        "height": 11,
        "tile_width": 32,
        "tile_height": 32
    },
    "representation": "image",
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



while True:
    g.render()
    g.update()

    for p in g.players:
        p.do_action(random.randint(0, 12))

    if g.is_terminal():
        g.reset()



