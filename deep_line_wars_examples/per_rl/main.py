import random
import torch
from deep_line_wars_examples.per_rl import sampling, memory, models
from deep_line_wars_examples.per_rl.models import dqn
from deep_line_wars_examples.per_rl.algorithms.DQN import DQN
import gym

if __name__ == "__main__":

    # Create the Cart-Pole game environment
    env = gym.make('CartPole-v0')

    agent = DQN(spec=dict(
        input=dict(shape=env.observation_space.shape),
        output=dict(shape=(env.action_space.n, )),
        model=models.dqn.MLP,
        options=dict(
          gamma=0.90
        ),
        memory=dict(
            model=memory.experience_replay,
            capacity=10000,
            batch=512
        ),
        optimizer=dict(
            model=torch.optim.Adam,
            options=dict(
                lr=0.00001,
            )
        ),
        sampling=dict(
            model=sampling.epsilon_decay,
            algorithm=random.randint,
            options=dict(
                start=1.0,
                end=0.1,
                steps=1000000
            )
        ),
        environment=dict(
            model=env,
            episodes=1000000,
        ),
        summary=dict(
            destination="./board/",
            save_interval=60,
            frequency=4,  # Frequency at which the log writes a record (n steps)
            models=[
                'accumulative_reward',
                'action_distribution',
                'loss'
            ]
        )

    ))

    agent.run()


    """
    opts = dict(

        states=dict(type='float', shape=env.observation_space.shape),
        actions=dict(type='int', num_actions=env.action_space.n),

        network=[
            dict(type="dense",size=64),
            dict(type="dense",size=64),
            dict(type="dense",size=64),
            dict(type="dense",size=64),
            dict(type="dense",size=64)
        ],
        update_mode=dict(
            unit="timesteps",
            batch_size=128,
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
                type="rmsprop",
                learning_rate=1e-6
            )
        ),
        discount=0.90,
        entropy_regularization=None,
        double_q_model=True,

        target_sync_frequency=1000,
        target_update_weight=0.50,

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
            directory="./runs",
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
    dqn = DQNAgent(**opts)

    for i in range(1000000):
        t = False
        s = env.reset()
        steps = 0
        while not t:

            a = dqn.act(s)
            s1, r, t, _ = env.step(a)
            dqn.observe(reward=r, terminal=t)
            steps += 1

            s = s1
        print(steps)
    """

#print(agent.model.forward(np.zeros((80, 80, 3))))