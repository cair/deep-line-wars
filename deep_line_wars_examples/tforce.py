import cv2

import tensorforce

import random

import time

from tensorforce.agents import PPOAgent, DQNAgent

from deep_line_wars.game import Game
from deep_line_wars.gui import  dummy, opencv
# https://github.com/reinforceio/tensorforce/blob/master/FAQ.md
if __name__ == "__main__":

    g = Game(dict(gui=opencv.GUI))

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
            batch_size=32,
            frequency=4
        ),
        memory=dict(
            type="replay",
            capacity=10000,
            include_next_states=True
        ),
        optimizer=dict(
            type="clipped_step",
            clipping_value=1.0,
            optimizer=dict(
                type="adam",
                learning_rate=1e-4
            )
        ),
        discount=0.94,
        entropy_regularization=None,
        double_q_model=True,

        target_sync_frequency=1000,
        target_update_weight=0.5,

        actions_exploration=dict(
            type="epsilon_anneal",
            initial_epsilon=1.0,
            final_epsilon=0.0,
            timesteps=10000
        ),
        #saver=dict(
        #    directory=None,
        #    seconds=600
        #),
        summarizer=dict(
            directory="./board",
            labels=["graph", "total-loss", "gradients",
                    'gradients_scalar',
                    'regularization',
                    'states', 'actions', 'rewards',
                    'losses',
                    'variables']
        ),
        execution=dict(
            type="single",
            session_config=None,
            distributed_spec=None
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
    actions = [0 for x in range(16)]
    s = time.time()
    skip_steps = 8
    g.flip_player()
    for i in range(100000):
        state = g.reset()

        while not g.is_terminal():
            state = cv2.resize(state, (80, 80))
            # Perform Action
            action = agent.act(state)
            actions[action] += 1
            _, r, t, _ = g.step(action)


            # Add experience, agent automatically updates model according to batch size
            agent.observe(reward=r, terminal=t)
            g.flip_player()
            a2 = random.randint(0, g.get_action_space() - 1)
            _, _, t2, _ = g.step(a2)
            g.flip_player()

            for _ in range(skip_steps):
                g.update()
            # Re render and get state
            s1 = g.get_state()


            if t is True or t2 is True:
                if g.winner.id not in statistics:
                    statistics[g.winner.id] = 1
                else:
                    statistics[g.winner.id] += 1
                print(statistics, actions)
                g.render_window()

            state = s1

    print("Time: %s" % (time.time() - s))
