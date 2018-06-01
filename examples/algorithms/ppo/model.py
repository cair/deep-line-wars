import random

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense
from tensorflow.python.keras.activations import relu, softmax, linear
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Adam
import numpy as np

from DeepLineWars.Game import Game
from examples.algorithms.ppo.NoisyDense import NoisyDense

def proximal_policy_optimization_loss(actual_value, predicted_value, old_prediction):
    advantage = actual_value - predicted_value

    def loss(y_true, y_pred):
        prob = K.sum(y_true * y_pred)
        old_prob = K.sum(y_true * old_prediction)
        r = prob / (old_prob + 1e-10)

        return -K.log(prob + 1e-10) * K.mean(
            K.minimum(r * advantage, K.clip(r, min_value=0.8, max_value=1.2) * advantage))

    return loss



class RLModel:

    def __init__(self):
        self.frames = 3
        self.action_space_a = 13  # Linear
        self.action_space_m = 2  # Softmax

        self.DUMMY_ACTION, self.DUMMY_VALUE = np.zeros((1, self.action_space_a)), np.zeros((1,1))
        self.NUM_STATE = 84*84*3
        self.GAMMA = 0.99
        self.BATCH_SIZE = 32
        self.episode = 0
        self.EPISODES = 10000
        self.reward = []
        self.reward_over_time = []
        self.LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
        self.EPOCHS = 10

        self.env = Game({
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
        self.env.reset()
        self.env.render()
        self.observation = self.env.get_state("image")

        self.actor_discrete = self.build_actor(discrete=True, action_space=self.action_space_a, activation=linear)
        #self.actor_continous = self.build_actor(discrete=False, action_space=self.action_space_m, activation=softmax)
        self.critic = self.build_critic()
        # Actor 1
        # -------
        # Send Unit 0
        # Send Unit 1
        # Send unit 2

        # Build 0
        # Build 1
        # Build 2

        # Mouse_on_off

        # Actor 2
        # -------
        # Mouse_vel_x
        # Mouse_vel_y

    def build_actor(self, discrete=True, action_space=None, activation=None):
        print(action_space)
        input_image = Input(shape=(84, 84, self.frames))
        actual_value = Input(shape=(1, ))
        predicted_value = Input(shape=(1, ))
        old_prediction = Input(shape=(action_space, ))

        x = Conv2D(32, (8, 8), (2, 2), 'same', activation=relu)(input_image)
        x = Conv2D(64, (4, 4), (2, 2), 'same', activation=relu)(x)
        x = Conv2D(128, (2, 2), (2, 2), 'same', activation=relu)(x)
        x = Conv2D(256, (1, 1), (2, 2), 'same', activation=relu)(x)
        x = Flatten()(x)

        x = Dense(512, activation=relu)(x)

        out_actions = Dense(action_space, activation=softmax, name='output')(x)
        #out_actions = NoisyDense(action_space, activation=softmax, sigma_init=0.02, name='output')(x)

        model = Model(inputs=[input_image, actual_value, predicted_value, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=10e-4),
                      loss=[proximal_policy_optimization_loss(
                          actual_value=actual_value,
                          old_prediction=old_prediction,
                          predicted_value=predicted_value)])
        model.summary()
        return model

    def build_critic(self):
        input_image = Input(shape=(84, 84, self.frames))
        x = Conv2D(32, (8, 8), (2, 2), 'same', activation=relu)(input_image)
        x = Conv2D(64, (4, 4), (2, 2), 'same', activation=relu)(x)
        x = Conv2D(128, (2, 2), (2, 2), 'same', activation=relu)(x)
        x = Conv2D(256, (1, 1), (2, 2), 'same', activation=relu)(x)
        x = Flatten()(x)

        x = Dense(512, activation=relu)(x)

        out_value = Dense(1, activation=linear)(x)

        model = Model(inputs=[input_image], outputs=[out_value])
        model.compile(optimizer=Adam(lr=10e-4), loss='mse')
        model.summary()
        return model

    def get_action(self):
        p = self.actor_discrete.predict(
            [
                np.reshape(self.observation, (1, ) + self.observation.shape),
                self.DUMMY_VALUE,
                self.DUMMY_VALUE,
                self.DUMMY_ACTION
            ])
        action = np.random.choice(self.action_space_a, p=np.nan_to_num(p[0]))
        action_matrix = np.zeros(p[0].shape)
        action_matrix[action] = 1
        return action, action_matrix, p

    def get_batch(self):
        batch = [
            [],  # Observations
            [],  # Actions
            [],  # Predicted
            []   # Reward
        ]

        tmp_batch = [
            [],
            [],
            []
        ]

        while len(batch[0]) < self.BATCH_SIZE:
            action, action_matrix, predicted_action = self.get_action()

            self.env.primary_player.opponent.do_action(random.randint(0, 12))
            observation, reward, done, info = self.env.step(action)

            self.reward.append(reward)

            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)


            self.observation = observation

            if done:
                self.env.render_window()
                self.transform_reward()
                for i in range(len(tmp_batch[0])):
                    obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                    r = self.reward[i]
                    batch[0].append(obs)
                    batch[1].append(action)
                    batch[2].append(pred)
                    batch[3].append(r)
                tmp_batch = [[], [], []]
                self.reset_env()

        obs, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]), (len(batch[3]), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward

    def reset_env(self):
        self.episode += 1
        self.observation = self.env.reset()
        self.reward = []

    def run(self):

        while self.episode < self.EPISODES:
            obs, action, pred, reward = self.get_batch()

            old_prediction = pred
            pred_values = self.critic.predict(obs)

            # Train Actor
            for e in range(self.EPOCHS):
                self.actor_discrete.train_on_batch([obs, reward, pred_values, old_prediction], [action])

            # Train Critic
            for e in range(self.EPOCHS):
                self.critic.train_on_batch([obs], [reward])

    def transform_reward(self):

        print('Episode #', self.episode, '\tfinished with reward', np.array(self.reward).sum(),
                  '\tAverage Noisy Weights', np.mean(self.actor_discrete.get_layer('output').get_weights()[1]))
        self.reward_over_time.append(np.array(self.reward).sum())
        for j in range(len(self.reward)):
            reward = self.reward[j]
            for k in range(j + 1, len(self.reward)):
                reward += self.reward[k] * self.GAMMA ** k
            self.reward[j] = reward



if __name__ == "__main__":
    agent = RLModel()
    agent.run()




