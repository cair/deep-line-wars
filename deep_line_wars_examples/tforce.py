import cv2

import tensorforce

import random

import time

from tensorforce.agents import PPOAgent, DQNAgent

from deep_line_wars.game import Game
from deep_line_wars.gui import pygame, dummy, opencv

if __name__ == "__main__":

    g = Game(dict(gui=pygame.GUI))

    dqn = dict(
        states=dict(type='float', shape=(80, 80, 3)),
        actions=dict(type='int', num_actions=g.selected_player.action_space.size),
        network=[
            dict(
                type="conv2d",
                size=32,
                window=8,
                stride=4
            ),
            dict(
                type="conv2d",
                size=64,
                window=4,
                stride=2
            ),
            dict(
                type="conv2d",
                size=64,
                window=3,
                stride=1
            ),
            dict(
                type="flatten"
            ),
            dict(
                type="dense",
                size=512
            )
        ],
        update_mode=dict(
            unit="timesteps",
            batch_size=64,
            frequency=4
        ),
        memory=dict(
            type="replay",
            capacity=10000,
            include_next_states=True
        ),
        optimizer=dict(
            type="clipped_step",
            clipping_value=0.1,
            optimizer=dict(
                type="adam",
                learning_rate=1e-3
            )
        ),
        discount=0.99,
        entropy_regularization=None,
        double_q_model=True,

        target_sync_frequency=1000,
        target_update_weight=1.0,

        actions_exploration=dict(
            type="epsilon_anneal",
            initial_epsilon=1.0,
            final_epsilon=0.0,
            timesteps=1000000
        )
    )

    ppo = dict(
        states=dict(type='float', shape=(80, 80, 3)),
        actions=dict(type='int', num_actions=g.selected_player.action_space.size),
        network=[
            {
                "type": "conv2d",
                "size": 32,
                "window": 8,
                "stride": 4
            },
            {
                "type": "conv2d",
                "size": 64,
                "window": 4,
                "stride": 2
            },
            {
                "type": "conv2d",
                "size": 64,
                "window": 3,
                "stride": 1
            },
            {
                "type": "flatten"
            },
            {
                "type": "dense",
                "size": 512
            }
        ],
        update_mode=dict(
            unit="episodes",
            batch_size=10,
            frequency=10
        ),
        memory=dict(
            type="latest",
            include_next_states=False,
            capacity=5000
        ),
        step_optimizer=dict(
            type="adam",
            learning_rate=1e-3
        ),
        subsampling_fraction=0.1,
        optimization_steps=50,

        discount=0.99,
        entropy_regularization=0.01,
        gae_lambda=None,
        likelihood_ratio_clipping=0.2,

        baseline_mode="states",
        baseline=dict(
            type="cnn",
            conv_sizes=[32, 32],
            dense_sizes=[32]
        ),
        baseline_optimizer=dict(
            type="multi_step",
            optimizer=dict(
                type="adam",
                learning_rate=1e-3
            ),
            num_steps=5
        )
    )

    agent = DQNAgent(**dqn)
    #agent = PPOAgent(**ppo)
    statistics = {}
    s = time.time()
    skip_steps = 4
    for i in range(100000):
        state = g.reset()
        g.render()
        state = g.get_state()
        while not g.is_terminal():
            state = cv2.resize(state, (80, 80))
            # Perform Action
            action = agent.act(state)
            s1, r, t, _ = g.step(action)

            g.render()
            s1 = g.get_state()

            # Add experience, agent automatically updates model according to batch size
            agent.observe(reward=r, terminal=t)

            g.flip_player()
            a2 = random.randint(0, g.get_action_space() - 1)
            _, _, t2, _ = g.step(a2)
            g.flip_player()

            for _ in range(skip_steps):
                g.update()
            # Re render and get state
            g.render()
            s1 = g.get_state()


            if t is True or t2 is True:
                if g.winner.id not in statistics:
                    statistics[g.winner.id] = 1
                else:
                    statistics[g.winner.id] += 1
                print(statistics)
                g.render_window()

            state = s1

    print("Time: %s" % (time.time() - s))
