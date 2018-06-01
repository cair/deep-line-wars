import random
from collections import deque
import pygame
import os

from PIL import Image
from scipy.misc import imsave
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, Concatenate, Lambda, Conv2DTranspose, Reshape
from tensorflow.python.keras.activations import relu, softmax, linear
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop
import numpy as np

from DeepLineWars.Game import Game

class ReplayMemory():
    """
    Memory Replay Buffer
    """
    """
    def __init__(self, buffer_size=10000):
        self
        self.S = np.zeros(shape=(buffer_size, 80, 80), dtype=np.uint8)
        self.A = np.zeros(shape=(buffer_size, 1), dtype=np.uint8)
        self.R = np.zeros(shape=(buffer_size, 1), dtype=np.uint8)
        self.T = np.zeros(shape=(buffer_size, 1), dtype=np.uint8)
        
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, episode_experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []
        self.buffer.append(episode_experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode)+1-trace_length)
            sampledTraces.append(episode[point:point+trace_length])
        sampledTraces = np.array(sampledTraces)
        return sampledTraces
    """

class StateEncoder:


    def __init__(self, state_shape, memory, batch_size):
        self.memory = memory
        self.batch_size = batch_size
        self.state_shape = state_shape
        self.vae, self.decoder = self._model()

    def train(self):
        batch = np.dstack(self.memory)
        X = np.array([batch])
        X /= 255
        self.vae.fit(X, X,
                       epochs=1,
                       batch_size=1,
                       verbose=1
                     )


        Y = self.vae.predict(X)
        z = self.decoder.predict(X)

        X *= 255
        Y *= 255

        img = np.concatenate((X[0], Y[0]))
        imsave("Memory.png", img)

    def _model(self):
        state = Input(shape=self.state_shape, name="State")
        
        conv_1 = Conv2D(32, (8, 8), (4, 4), 'same', activation=relu)(state)
        conv_2 = Conv2D(64, (4, 4), (2, 2), 'same', activation=relu)(conv_1)
        conv_3 = Conv2D(64, (3, 3), (1, 1), 'same', activation=relu)(conv_2)
        #conv_4 = Conv2D(256, (2, 2), (2, 2), 'same', activation=relu)(conv_3)

        ###################################
        #
        # Encoder
        #
        ###################################
        flatten = Flatten()(conv_3)

        fc_1 = Dense(256, activation=relu)(flatten)  # Fist encoding layer

        """z_mean = Dense(2)(fc_1)
        z_log_var = Dense(2)(fc_1)

        def sampling(args):
            epsilon_std = 1.0
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 2),
                                      mean=0., stddev=epsilon_std)
            return z_mean + K.exp(z_log_var) * epsilon

        z = Lambda(sampling, output_shape=(2,))([z_mean, z_log_var])

        decoder_hid = Dense(256, activation=relu)(z)"""
        fc_3 = Dense(int(flatten.shape[1]), activation=relu)(fc_1)

        ###################################
        #
        # Decoder
        #
        ###################################
        x = Reshape(conv_3.shape[1:])(fc_3)

        #deconv_4 = Conv2DTranspose(128, (2, 2), (2, 2), 'same', activation=relu)(x)
        deconv_3 = Conv2DTranspose(64, (3, 3), (1, 1), 'same',activation=relu)(x)
        deconv_2 = Conv2DTranspose(32, (4, 4), (2, 2), 'same',activation=relu)(deconv_3)
        decoded = Conv2DTranspose(self.state_shape[-1], (8, 8), (4, 4), 'same', activation=relu)(deconv_2)

        # Create model
        vae = Model(
            inputs=[state],
            outputs=[decoded],
        )
        vae.compile(
            optimizer=Adam(lr=1e-3),
            loss='mse'
        )

        decoder = Model(
            inputs=[state],
            outputs=[fc_1]
        )

        vae.summary()
        return vae, decoder




class RewardPredictor:
    pass


class Environment:

    def __init__(self):
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
        self.updates = 10  # Updates per actions

    def step(self, a):
        data = self.env.step(a, representation="image")
        for i in range(self.updates):
            self.env.update()
        return data

    def reset(self):
        return self.env.reset()

class Agent:
    def __init__(self, exploration_steps, episodes, batch_size):
        self.exploration_steps = exploration_steps

        self.BATCH_SIZE = batch_size
        self.EPISODES = episodes

        self.env = Environment()
        self.state_shape = self.env.reset().shape[:-1] + (3, )

        self.memory = deque(maxlen=3) #ReplayMemory()
        self.vision = StateEncoder( state_shape=self.state_shape, memory=self.memory, batch_size=batch_size)

    def run(self):

        for episode in range(self.EPISODES):

            t = False
            _s = self.env.reset()
            _s = self.env.env.rgb2gray(_s)
            step = 0
            while t is False:
                self.memory.append(_s)

                # Do action
                a = random.randint(0, 12)
                _s1, r, t, _ = self.env.step(a)
                _s1 = self.env.env.rgb2gray(_s1)
                _s = _s1

                step += 1
                if step < 20:
                    continue

                #self.memory.add((s, a, r, s1, t))

                # Do not proceed if replay memory has low experience count
                #if len(self.memory.buffer) < self.exploration_steps:
                #    continue


                self.vision.train()






if __name__ == "__main__":

    agent = Agent(exploration_steps=150,
                  episodes=100,
                  batch_size=16)
    agent.run()




"""
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


if __name__ == "__main__":
    agent = RLModel()
    agent.run()
"""



