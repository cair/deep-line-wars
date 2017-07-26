import random

import keras
from keras import backend as K
import numpy as np
from keras import Input
from keras.engine import Model
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D
from tensorflow.contrib.keras.python.keras.layers.core import Activation, Flatten, Dense
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt

class PlotLosses(keras.callbacks.Callback):

    def __init__(self):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        #self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        if self.i % 1000 == 0:
            plt.clf()
            plt.plot(self.x, self.losses, label="loss")
            #plt.plot(self.x, self.val_losses, label="val_loss")
            plt.legend()
            plt.savefig("fig.png")




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


class DivideAndWin:

    def __init__(self,
                 game,
                 memory_size=50000,
                 learning_rate=1e-4,
                 epsilon=1.0,
                 epsilon_end=0.10,
                 epsilon_steps=10000,
                 exploration_steps=10000):

        self.game = game
        self.memory = Memory(memory_size)

        self.primary_model = None
        self.secondary_model = None

        # Static Variables
        self.LEARNING_RATE = learning_rate
        self.EPSILON_START = epsilon
        self.EPSILON_END = epsilon_end
        self.EPSILON_DECAY = (self.EPSILON_END - self.EPSILON_START) / epsilon_steps
        self.EXPLORATION_STEPS = exploration_steps
        self.BATCH_SIZE = 8
        self.GAMMA = 0.99
        self.player = None
        self.state_size = None
        self.action_size = None
        self.state = None

        # Variables
        self.epsilon = self.EPSILON_START
        self.iteration = 0
        self.episode = 0
        self.loss_sum = 0
        self.rewards = 0

        self.plot_losses = PlotLosses()

    def reset(self):
        self.episode += 1
        self.state = np.expand_dims(self.game.get_state(self.player), 0)


    def init(self):
        self.state = self.game.get_state(self.player)
        self.state_size = self.state.shape
        self.reset()
        self.action_size = len(self.player.action_space)

        self.primary_model = self.build_model()
        self.secondary_model = self.build_model()


        try:
            self.load("./save/dqn_3_p%s.h5" % self.player.id)
            print("Loaded ./save/dqn_3_p%s.h5" % self.player.id)
        except:
            pass

        print("DQNRUnner inited!")
        print("State size is: %s,%s,%s" % self.state_size)
        print("Action size is: %s" % self.action_size)
        print("Batch size is: %s " % self.BATCH_SIZE)

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        input_layer = Input(shape=self.state_size, name='image_input')
        conv1 = Conv2D(32, (8, 8), strides=(1, 1), activation='relu')(input_layer)
        conv2 = Conv2D(64, (2, 2), strides=(1, 1), activation='relu')(conv1)
        conv3 = Conv2D(64, (2, 2), strides=(1, 1), activation='relu')(conv2)
        conv_flatten = Flatten()(conv3)
        fc1 = Dense(512)(conv_flatten)
        advantage = Dense(self.action_size)(fc1)
        fc2 = Dense(512)(conv_flatten)
        value = Dense(1)(fc2)
        policy = keras.layers.merge([advantage, value], mode=lambda x: x[0]-K.mean(x[0])+x[1], output_shape=(self.action_size,))
        model = Model(inputs=[input_layer], outputs=[policy])

        optimizer = keras.optimizers.adam(self.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])



        """model.add(Conv2D(32, (8, 8), strides=(1, 1), data_format='channels_first', padding='same', input_shape=self.state_size))
        model.add(Activation('relu'))
        #model.add(Conv2D(64, (4, 4), strides=(1, 1), padding='same'))
        #model.add(Activation('relu'))
        model.add(Conv2D(64, (2, 2), strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))

        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)"""
        return model

    def load(self, name):
        self.primary_model.load_weights(name)

    def save(self, name):
        self.primary_model.save_weights(name)

    def train(self):
        if self.memory.count < self.BATCH_SIZE:
            return

        loss = 0
        memories = self.memory.get(self.BATCH_SIZE)
        for (s, a, r, s1, terminal) in memories:

            target = self.primary_model.predict(s)
            Q_sa = self.primary_model.predict(s1)[0]

            if terminal:
                target[0, a] = r
            else:
                target[0, a] = r + self.GAMMA * np.max(Q_sa)

            self.primary_model.fit(s, target, epochs=1, batch_size=1, callbacks=[self.plot_losses], verbose=0)

    def act(self):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Exploit Q-Knowledge
        act_values = self.primary_model.predict(self.state)
        return np.argmax(act_values[0])  # returns action

    def defense_reward(self):
        score = 1
        for u in self.player.opponent.units:
            if self.player.enemy_unit_reached_base(u):
                score -= 1

        return score

    def attack_reward(self):
        l_u = len(self.player.units)
        return -1 if l_u <= 0 else l_u

    def update(self, seconds):

        # 1. Do action
        # 2. Observe
        # 3. Train
        # 4. set state+1 to state
        action = self.act()

        s, a, s1, r, terminal, _ = self.state, action, *self.game.step(self.player, action)
        s1 = np.expand_dims(s1, 0)

        def_r = self.defense_reward()
        attk_r = self.attack_reward()

        self.memory.add([s, a, def_r, s1, terminal])

        self.train()

        self.epsilon += self.EPSILON_DECAY
        self.iteration += 1
        self.state = s1
        self.rewards += def_r

        if self.iteration % 100 == 0:
            print("I: %s, Epsilon: %s, def_r: %s, Loss: %s" % (self.iteration, self.epsilon, self.rewards, self.loss_sum / self.iteration))
            self.save("./save/dqn_3_p%s.h5" % self.player.id)