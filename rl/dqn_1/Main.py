import os
import time
import random

import keras
import numpy as np
from keras import backend as K
from keras.utils import plot_model
from tensorflow.contrib.keras.python.keras import initializers
from tensorflow.contrib.keras.python.keras.optimizers import rmsprop, adam
from tensorflow.contrib.keras.python.keras.engine import Input, Model
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D
from tensorflow.contrib.keras.python.keras.layers.core import Activation, Flatten, Dense
from rl.dqn_1.settings import settings

from rl.dqn_1.Memory import Memory
from rl.dqn_1.PlotEngine import PlotEngine


class Algorithm:

    def __init__(self, game):

        self.game = game
        self.player = None
        self.memory = Memory(settings["memory_size"])
        self.use_training_data = settings["use_training_data"]

        self.target_model = None
        self.model = None

        # Parameters
        self.LEARNING_RATE = settings["learning_rate"]
        self.BATCH_SIZE = settings["batch_size"]
        self.GAMMA = settings["discount_factor"]

        # Epsilon decent
        self.EPSILON_START = settings["epsilon_start"]
        self.EPSILON_END = settings["epsilon_end"]
        self.EPSILON_DECAY = (self.EPSILON_END - self.EPSILON_START) / settings["epsilon_steps"]
        self.epsilon = self.EPSILON_START

        # Exploration parameters (fully random play)
        self.EXPLORATION_WINS = settings["exploration_wins"]
        self.EXPLORATION_WINS_COUNTER = 0

        # Episode data
        self.episode = 0            # Episode Count
        self.episode_loss = 0       # Loss sum of a episode
        self.episode_reward = 0     # Reward sum of a episode
        self.iteration = 0          # Iteration counter
        self.loss_list = []

        # State data
        self.state_size = None
        self.action_size = None
        self.state = None
        self.state_vec = None
        self.action_distribution = None
        self.q_values = []
        self.loss_average = []

        self.plot_losses = PlotEngine(self.game, self)
        self.plot_losses.start()

    def reset(self):

        # Get new initial state
        self.state = np.expand_dims(self.game.get_state(self.player), 0)

        #self.loss_list = []

        # Reset plot
        self.plot_losses.new_game()

        # Update target model
        if self.target_model:
            self.update_target_model()

        # Check if game was a victory
        if self.game.winner == self.player:
            # Add to counter
            self.EXPLORATION_WINS_COUNTER += 1
            items = np.array(self.memory.buffer[-1*self.iteration:])
            np.save("./training_data/win_%s_%s" % (self.EXPLORATION_WINS_COUNTER, time.time()), items)

        else:
            pass
            # Lost the round, delete memories
            #self.memory.remove_n(self.iteration)

        # Print output
        print("Episode: %s, Epsilon: %s, Reward: %s, Loss: %s, Memory: %s" % (self.episode, self.epsilon, self.episode_reward, self.episode_loss / (self.iteration+1), self.memory.count))

        self.iteration = 0

        # Reset loss sum
        self.episode_loss = 0

        # Reset episode reward
        self.episode_reward = 0

        # Increase episode
        self.episode += 1

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def init(self, player):
        self.player = player

        self.state = self.game.get_state(self.player)
        self.state_size = self.state.shape

        self.reset()
        self.action_size = len(self.player.action_space)
        self.action_distribution = [0 for _ in range(self.action_size)]

        #self.model = self.build_model()
        #self.target_model = self.build_model()
        self.model = self.build_dense_model()
        self.target_model = self.build_dense_model()

        try:
            # Load training data
            if self.use_training_data:
                for file in os.listdir("./training_data/"):
                    file = os.path.join("./training_data/", file)
                    training_items = np.load(file)
                    for training_item in training_items:
                        self.memory.add(training_item)

            self.load("./save/dqn_3_p%s.h5" % self.player.id)
            print("Loaded ./save/dqn_3_p%s.h5" % self.player.id)

        except:
            pass

        print("DQNRUnner inited!")
        print("State size is: %s,%s,%s" % self.state_size)
        print("Action size is: %s" % self.action_size)
        print("Batch size is: %s " % self.BATCH_SIZE)

    def build_dense_model(self):
        initializer = initializers.random_normal(stddev=0.5)
        #initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.5)
        input_layer = Input(shape=self.state_size, name='image_input')
        flatt = Flatten()(input_layer)
        d_1 = Dense(1024, activation="relu", kernel_initializer=initializer)(flatt)
        d_2 = Dense(1024, activation="relu", kernel_initializer=initializer)(d_1)
        d_3 = Dense(1024, activation="relu", kernel_initializer=initializer)(d_2)
        d_4 = Dense(1024, activation="relu", kernel_initializer=initializer)(d_3)
        d_5 = Dense(1024, activation="relu", kernel_initializer=initializer)(d_4)
        d_6 = Dense(1024, activation="relu", kernel_initializer=initializer)(d_5)
        policy = Dense(self.action_size, activation="linear")(d_6)

        model = Model(inputs=[input_layer], outputs=[policy])
        optimizer = adam(self.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss="mse")
        plot_model(model, to_file='./model.png', show_shapes=True, show_layer_names=True)
        return model

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        initializer = initializers.random_normal(stddev=0.01)
        #initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.5)

        # Image input


        conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', kernel_initializer=initializer)(input_layer)
        conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', kernel_initializer=initializer)(conv1)
        conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', kernel_initializer=initializer)(conv2)
        conv_flatten = Flatten()(conv3)

        # Vector state input
        """input_layer_2 = Input(shape=(2, ), name="vector_input")
        vec_dense = Dense(512, activation='relu')(input_layer_2)
        #concat_layer = keras.layers.concatenate([vec_dense, dense_first])
        concat_layer = keras.layers.merge([vec_dense, dense_first], 
        input_layer = Input(shape=self.state_size, name='image_input')mode=lambda x: x[0]-K.mean(x[0])+x[1] , output_shape=(512,))
        dense_1 = Dense(512, activation='relu')(concat_layer)"""

        # Stream split
        fc1 = Dense(512, kernel_initializer=initializer, activation="relu")(conv_flatten)
        fc2 = Dense(512, kernel_initializer=initializer, activation="relu")(conv_flatten)

        advantage = Dense(self.action_size)(fc1)
        value = Dense(1)(fc2)

        policy = keras.layers.merge([advantage, value], mode=lambda x: x[0]-K.mean(x[0])+x[1],  output_shape=(self.action_size,))
        #policy = Dense(self.action_size, activation="linear")(fc1)

        model = Model(inputs=[input_layer], outputs=[policy])
        #model = Model(inputs=[input_layer, input_layer_2], outputs=[policy])
        optimizer = adam(self.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss="mse")
        plot_model(model, to_file='./model.png', show_shapes=True, show_layer_names=True)

        return model

    @staticmethod
    def huber(y_true, y_pred):
        cosh = lambda x: (K.exp(x)+K.exp(-x))/2
        return K.mean(K.log(cosh(y_pred - y_true)), axis=-1)

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)

    def save(self, name):
        self.target_model.save_weights(name)

    def train(self):
        if self.memory.count < self.BATCH_SIZE:
            return

        loss = 0
        memories = self.memory.get(self.BATCH_SIZE)
        for (s, s_vec, a, r, s1, s1_vec, terminal) in memories:
            #target = self.model.predict([s, s_vec])
            target = self.model.predict(s)

            if terminal:
                target[0, a] = r
            else:
                pred_a = self.model.predict(s)
                pred_t = self.target_model.predict(s1)[0]

                #pred_a = self.model.predict([s, s_vec])
                #pred_t = self.target_model.predict([s1, s1_vec])[0]
                target[0, a] = r + self.GAMMA * pred_t[np.argmax(pred_a)]

            history = self.model.fit(s, target, epochs=1, batch_size=1, callbacks=[], verbose=0)
            #history = self.target_model.fit([s, s_vec], target, epochs=1, batch_size=1, callbacks=[self.plot_losses], verbose=0)
            loss += history.history["loss"][0]

            self.loss_list.append(history.history["loss"][0])
            #self.average_loss[len(self.episode_loss_list)] = (self.average_loss[len(self.episode_loss_list)] + history.history["loss"][0]) / 2

        self.episode_loss += loss

    def act(self):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Exploit Q-Knowledge
        act_values = self.target_model.predict(self.state)
        self.q_values = act_values[0]
        return np.argmax(act_values[0])  # returns action

    def reward_fn(self):
        reached_red = self.player.enemy_unit_reached_red()
        if reached_red:
            score = -1
        else:
            score = .1

        score = ((self.player.health - self.player.opponent.health) / 50) + 0.01
        return score

    def update(self, seconds):

        # 1. Do action
        # 2. Observe
        # 3. Train
        # 4. set state+1 to state
        action = self.act()
        self.action_distribution[action] += 1
        s_vec = np.array([[self.player.health, self.player.opponent.health]])
        s, a, s1, r, terminal, _ = self.state, action, *self.game.step(self.player, action)
        s1_vec = np.array([[self.player.health, self.player.opponent.health]])
        s1 = np.expand_dims(s1, 0)

        reward = self.reward_fn()

        self.memory.add([s, s_vec, a, reward, s1, s1_vec, terminal])

        self.iteration += 1
        self.state = s1
        self.state_vec = s1_vec
        self.episode_reward += reward

        # Exploration wins must be counter up before training. This is so we have a prepopulated memory with GOOD memories
        if self.EXPLORATION_WINS_COUNTER < self.EXPLORATION_WINS:
            return

        self.train()

        self.epsilon += self.EPSILON_DECAY

        if self.iteration % 1000 == 0:
            self.save("./save/dqn_3_p%s.h5" % self.player.id)