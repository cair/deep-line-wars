# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Convolution2D, Activation, Flatten
from keras.optimizers import Adam
from keras import backend as K
from tensorflow.contrib.keras.python.keras.initializers import normal

EPISODES = 5000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000000) # 1 million
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99998
        self.learning_rate = 0.001
        self.pre_training_iterations = 1000000  # 1 Million pre training frames
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.action_count = 0
        self.loss = 0

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()

        model.add(Conv2D(32, (8, 8), strides=(4, 4), data_format='channels_first', padding='same', input_shape=self.state_size))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def update_target_external(self, weights):
        # COpy weights from external target model
        self.target_model.set_weights(weights)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Exploit Q-Knowledge
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        loss = 0
        for s, a, r, s1, terminal in minibatch:

            target = self.model.predict(s)

            Q_sa = self.model.predict(s1)[0]
            Q_sa_t = self.target_model.predict(s1)[0]

            if terminal:
                target[0, a] = r
            else:
                target[0][a] = r + self.gamma * Q_sa_t[np.argmax(Q_sa)]

            loss += self.model.train_on_batch(s, target)
            #self.model.fit(s, target, epochs=1, verbose=0)

        self.loss = (self.loss + loss) / 2
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class DQNRUnner:

    def __init__(self, game):
        self.episode = 0
        self.iteration = 0
        self.game = game
        self.state_size = None
        self.action_size = None
        self.agent = None
        self.done = None
        self.batch_size = None
        self.state = None
        self.player = None

    def reset(self):
        print("episode: {}/{}, score: {}, e: {:.2}, loss: {:.3}".format(self.episode, EPISODES, self.iteration, self.agent.epsilon, self.agent.loss))
        self.episode += 1
        self.state = np.expand_dims(self.game.get_state(self.player), 0)
        self.iteration = 0

    def init(self):
        self.state_size = self.game.get_state(self.player).shape
        self.action_size = len(self.player.action_space) #self.game.action_space

        self.agent = DQNAgent(self.state_size, self.action_size)
        self.done = False
        self.batch_size = 4

        try:
            self.agent.load("./save/dqn_2_p%s.h5" % self.player.id)
            print("Loaded ./save/dqn_2_p%s.h5" % self.player.id)
        except:
            pass

        print("DQNRUnner inited!")
        print("State size is: %s,%s,%s" % self.state_size)
        print("Action size is: %s" % self.action_size)
        print("Batch size is: %s " % self.batch_size)

    def update(self, seconds):
        action = self.agent.act(self.state)

        next_state, reward, done, _ = self.game.step(self.player, action)
        next_state = np.expand_dims(next_state, 0)

        #print(self.player.action_space[action]["action"], reward, done)

        if self.state is not None:
            self.agent.remember(self.state, action, reward, next_state, done)

        self.state = next_state

        # Training
        if len(self.agent.memory) > self.batch_size:
            self.agent.replay(self.batch_size)

        # If game is done or every 10000 steps
        if done or self.iteration % 10000 == 0:
            self.agent.update_target_model()
            print("episode: {}/{}, score: {}, e: {:.2}".format(self.episode, EPISODES, self.iteration, self.agent.epsilon))
            if self.episode % 10 == 0:
                print("Saving network at episode %s" % self.episode)
                self.agent.save("./save/dqn_2_p%s.h5" % self.player.id)

        self.iteration += 1


