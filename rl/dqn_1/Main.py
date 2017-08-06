import os
import time
import random
from threading import Thread

import keras
import numpy as np

import tensorflow as tf
from keras import backend as K
from keras.utils import plot_model
from tensorflow.contrib.keras.python.keras import initializers
from tensorflow.contrib.keras.python.keras.optimizers import rmsprop, adam
from tensorflow.contrib.keras.python.keras.engine import Input, Model
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D
from tensorflow.contrib.keras.python.keras.layers.core import Flatten, Dense, Lambda

from rl.dqn_1.ModelViz import ModelViz
from rl.dqn_1.settings import settings

from rl.dqn_1.Memory import Memory
from rl.dqn_1.PlotEngine import PlotEngine
from utils import set_thread_name


class ThreadedTrainer(Thread):

    def __init__(self, train_func):
        Thread.__init__(self, name="ThreadedTrainer")
        self.train_f = train_func
        set_thread_name("ThreadedTrainer")

    def run(self):
        self.train()

    def train(self):
        while True:
            self.train_f()


class Algorithm:
    def __init__(self, game, player):

        self.game = game
        self.player = player

        self.memory = Memory(settings["memory_size"])

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
        self.episode = 0  # Episode Count
        self.episode_loss = 0  # Loss sum of a episode
        self.episode_reward = 0  # Reward sum of a episode
        self.frame = 0  # Frame counter
        self.loss_list = []

        # State data
        self.state = self.game.get_state(self.player)
        self.state_size = self.state[0].shape

        # Action data
        self.action_size = len(self.player.action_space)
        self.action_distribution = [0 for _ in range(self.action_size)]


        self.q_values = [0 for _ in range(self.action_size)]
        self.loss_average = []
        self.train_count = 0

        # Construct Models
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.graph = tf.get_default_graph()

        # Plotting
        self.plot_losses = PlotEngine(self.game, self)
        self.plot_losses.start()

        # DeepDream
        #self.model_viz = ModelViz(self)
        #self.model_viz.start()

        # Training is done in a separate thread
        if settings["threaded_training"]:
            threaded_training = ThreadedTrainer(self.train)
            threaded_training.setDaemon(True)
            threaded_training.start()


        # Load training data
        if settings["use_training_data"]:
            for file in os.listdir("./training_data/"):
                file = os.path.join("./training_data/", file)
                training_items = np.load(file)
                for training_item in training_items:
                    self.memory.add(training_item)



        # Load existing model
        #self.load("./save/dqn_3_p%s.h5" % self.player.id)
        #print("Loaded ./save/dqn_3_p%s.h5" % self.player.id)

        print("DQNRUnner inited!")
        print("State size is: %s,%s,%s" % self.state_size)
        print("Action size is: %s" % self.action_size)
        print("Batch size is: %s " % self.BATCH_SIZE)

    def reset(self):

        # Get new initial state
        self.state = self.game.get_state(self.player)

        # Reset plot
        self.plot_losses.new_game()

        # Update target model
        if self.target_model:
            self.update_target_model()

            # Save target model
            self.save("./save/dqn_p%s_%s.h5" % (self.player.id, time.time()))


        # Check if game was a victory
        if self.game.winner == self.player:
            # Add to counter
            self.EXPLORATION_WINS_COUNTER += 1
            items = np.array(self.memory.buffer[-1 * self.frame:])
            np.save("./training_data/win_%s_%s" % (self.EXPLORATION_WINS_COUNTER, time.time()), items)

        else:
            pass
            # Lost the round, delete memories
            # self.memory.remove_n(self.iteration)

        # Print output
        print("Episode: %s, Epsilon: %s, Reward: %s, Loss: %s, Memory: %s" % (
        self.episode, self.epsilon, self.episode_reward, self.episode_loss / (self.frame + 1), self.memory.count))

        self.frame = 0

        # Reset loss sum
        self.episode_loss = 0

        # Reset episode reward
        self.episode_reward = 0

        # Increase episode
        self.episode += 1

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def build_dense_model(self):
        initializer = initializers.random_normal(stddev=0.5)
        # initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.5)
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

        # Image input
        input_layer = Input(shape=self.state_size, name='image_input')
        x = Conv2D(16, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='uniform', trainable=True)(
            input_layer)
        x = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='uniform', trainable=True)(x)
        x = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='uniform', trainable=True)(x)
        #x = Conv2D(128, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='uniform')(x)
        #x = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='uniform')(x)
        x = Flatten()(x)

        # Value Stream
        vs = Dense(512, activation="relu", kernel_initializer='uniform')(x)
        vs = Dense(1, kernel_initializer='uniform')(vs)

        # Advantage Stream
        ad = Dense(512, activation="relu", kernel_initializer='uniform')(x)
        ad = Dense(self.action_size, kernel_initializer='uniform')(ad)

        policy = Lambda(lambda w: w[0] - K.mean(w[0]) + w[1])([vs, ad])
        #policy = keras.layers.merge([vs, ad], mode=lambda x: x[0] - K.mean(x[0]) + x[1], output_shape=(self.action_size,))

        model = Model(inputs=[input_layer], outputs=[policy])
        optimizer = adam(self.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss="mse")
        plot_model(model, to_file='./output/model.png', show_shapes=True, show_layer_names=True)

        return model

    @staticmethod
    def huber(y_true, y_pred):
        cosh = lambda x: (K.exp(x) + K.exp(-x)) / 2
        return K.mean(K.log(cosh(y_pred - y_true)), axis=-1)

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)

    def save(self, name):
        self.target_model.save_weights(name)

    def train(self):

        if self.memory.count < self.BATCH_SIZE:
            return

        with self.graph.as_default():
            batch_loss = 0
            memories = self.memory.get(self.BATCH_SIZE)
            for s, a, r, s1, terminal in memories:
                # Model = We train on
                # Target = Draw actions from

                target = r

                tar_s = self.target_model.predict(s)
                if not terminal:
                    tar_s1 = self.target_model.predict(s1)
                    target = r + self.GAMMA * np.amax(tar_s1[0])

                tar_s[0][a] = target
                loss = (r + (self.GAMMA * np.amax(tar_s1[0]) - np.amax(tar_s[0]))) ** 2

                history = self.model.fit(s, tar_s, epochs=1, batch_size=1, callbacks=[], verbose=0)
                batch_loss += loss
                self.episode_loss += loss
                self.train_count += 1
            self.loss_list.append(batch_loss)

    def act(self):
        if np.random.uniform() <= self.epsilon:
            return random.randrange(self.action_size)

        # Exploit Q-Knowledge
        act_values = self.target_model.predict(self.state)
        self.q_values = act_values[0]
        return np.argmax(act_values[0])  # returns action

    def reward_fn(self):
        score = ((self.player.health - self.player.opponent.health) / 50) + 0.01
        return score

    def update(self, seconds):

        # 1. Do action
        # 2. Observe
        # 3. Train
        # 4. set state+1 to state
        action = self.act()
        self.action_distribution[action] += 1
        s, a, s1, r, terminal, _ = self.state, action, *self.game.step(self.player, action)


        reward = self.reward_fn()

        self.memory.add([s, a, reward, s1, terminal])

        self.frame += 1
        self.state = s1
        self.episode_reward += reward

        if self.EXPLORATION_WINS_COUNTER < self.EXPLORATION_WINS:
            return

        if not settings["threaded_training"]:
            self.train()

        self.epsilon += self.EPSILON_DECAY

