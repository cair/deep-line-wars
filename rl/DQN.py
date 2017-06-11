import random
import numpy as np

from tensorflow.contrib.keras.python.keras.layers import Activation, Dense, Flatten
from tensorflow.contrib.keras.python.keras.layers.convolutional import Convolution2D
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.optimizers import Adam


class Memory:

    def __init__(self, memory_size):
        self.buffer = []
        self.count = 0
        self.max_memory_size = memory_size

    def add(self, memory):
        self.buffer.append(memory)
        self.count += 1

        if self.count > self.max_memory_size:
            self.buffer.pop(0)
            self.count -= 1

    def get(self, batch_size=1):
        if self.count <= batch_size:
            return self.buffer

        return random.sample(self.buffer, batch_size)


class DQN:

    def __init__(self, memory_size=50000, learning_rate=1e-4, channels=5, width=30, height=11):
        self.memory = Memory(memory_size)
        self.model = None
        self.LEARNING_RATE = learning_rate
        self.CHANNELS = channels
        self.WIDTH = width
        self.HEIGHT = height



        self.build_model()

    def build_model(self):
        self.model = Sequential()

        # subsample=(4, 4)
        self.model.add(
            Convolution2D(32, 8, 8,
                          padding='same',
                          input_shape=(self.CHANNELS, self.WIDTH, self.HEIGHT),
                          activation='relu'
                          ))
        self.model.add(Convolution2D(64, 4, 4, padding='same', activation='relu'))
        self.model.add(Convolution2D(64, 3, 3, padding='same', activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(2))

        adam = Adam(lr=self.LEARNING_RATE)
        self.model.compile(loss='mse', optimizer=adam)

    def observe(self):
        pass

    def remember(self, state):
        pass

    def step(self, state):

        loss = self.model.train_on_batch(
            np.expand_dims(state, 0),
            np.expand_dims([1, 0], 0)
        )

        print(loss)


