import random
from DeepLineWars.Game import Game
import tensorflow as tf

def proximal_policy_optimization_loss(actual_value, predicted_value, old_prediction):
    advantage = actual_value - predicted_value

    def loss(y_true, y_pred):
        prob = K.sum(y_true * y_pred)
        old_prob = K.sum(y_true * old_prediction)
        r = prob/(old_prob + 1e-10)

        return -K.log(prob + 1e-10) * K.mean(K.minimum(r * advantage, K.clip(r, min_value=0.8, max_value=1.2) * advantage))
    return loss






if __name__ == "__main__":
    pass

    env = Game({
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
            "ticks_per_second": 5,
            "fps": 1,
            "ups": 1,
            "income_ratio": 0.20,
            "kill_gold_ratio": 0.10
        },
        "gui": {
            "enabled": True,
            "draw_friendly": True,
            "minimal": True
        }
    })


    s = env.reset()

    while True:
        # Which action
        # [0-4]

        # X velocity, Y velocity, Unit velocity, Building Velocity
        # [0-1, 0-1, 0-4, 0-4]


        a = random.randint(0, 3)
        a_intensities = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 4), random.uniform(0, 4)]
        a_intensity = a_intensities[a]
        s1, r, t, _  = env.step([a, a_intensity])

        env.render()
        env.render_window()

        if t:
            env.reset()

        s = s1