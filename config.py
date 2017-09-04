
game = {
"width": 30,
"height": 11,
"tile_width": 32,
"tile_height": 32
}

gui = {
    "enabled": True,
}


mechanics = {
  "complexity": {
      "build_anywhere": True,
      "draw_friendly": False
  },
  "start_health": 50,
  "start_gold": 100,
  "start_lumber": 0,
  "start_income": 20,
  "income_frequency": 10,
  "ticks_per_second": 10,
  "fps": 10,
  "ups": 1000000,
  "max_aps": 10,
  "statps": 1,
  "income_ratio": 0.20,
  "kill_gold_ratio": 0.10,
}

ai = {
    "enabled": True,
    "agents": [
        [{"package": "rl.hard_code_1.Main", "class": "Algorithm", "representation": "image"}],
        [{"package": "rl.hard_code_1.Main", "class": "Algorithm", "representation": "image"}]
    ]
}


web = {
    "enabled": True,
    "update_interval": 5,
    "state_images": True,
    "nn_images": True
}
