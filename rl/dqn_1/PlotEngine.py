from threading import Thread

import keras
import time
import matplotlib
import pygame
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


class PlotEngine(Thread):
    def __init__(self, game, algorithm):
        Thread.__init__(self)
        self.game = game
        self.algorithm = algorithm

        self.i = 0
        self.x = []
        self.update_interval = 1
        self.action_names = [x["short"] for x in self.game.players[0].action_space]
        self.refresh_rate = 1

        matplotlib.rcParams.update({'font.size': 12})

        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.plot_loss = self.figure.add_subplot(2, 2, 1)
        self.plot_action = self.figure.add_subplot(2, 2, 2)
        self.plot_state = self.figure.add_subplot(2, 2, 3)
        self.plot_q = self.figure.add_subplot(2, 2, 4)
        self.canvas = FigureCanvasAgg(self.figure)
        self.canvas.draw()
        self.renderer = self.canvas.get_renderer()

    def run(self):
        while True:
            try:
                self.loss()
                self.action_distribution()
                self.state_representation()
                self.q_values()
                self.draw()
            except:
                pass
            time.sleep(self.refresh_rate)

    def on_train_begin(self, logs={}):
        pass

    def new_game(self):
        self.i = 0

    def draw(self):
        self.canvas.draw()
        surface = pygame.image.fromstring(self.renderer.tostring_rgb(), self.canvas.get_width_height(), "RGB")
        self.game.gui.surface_plot.plots["loss"] = surface

    def loss(self):

        self.plot_loss.cla()
        self.plot_loss.plot(self.algorithm.loss_list, label="loss")
        # self.plot_loss.plot(self.x, self.average_losses, label="average_loss")
        self.plot_loss.set_ylabel('Percent')
        self.plot_loss.set_title('Time-step')

    def action_distribution(self):

        s = np.sum(np.array(self.algorithm.action_distribution))
        y = np.arange(len(self.algorithm.action_distribution))

        self.plot_action.cla()
        self.plot_action.bar(y, [(x / s) * 100 for x in self.algorithm.action_distribution], align='center', alpha=0.5)
        self.plot_action.set_xticks(y, self.action_names)
        self.plot_action.set_ylabel('Frequency')
        self.plot_action.set_title('Action Distribution')

    def q_values(self):
        y = np.arange(len(self.algorithm.action_distribution))
        self.plot_q.cla()
        self.plot_q.bar(y, self.algorithm.q_values, align='center', alpha=0.5)
        self.plot_q.set_xticks(y, self.action_names)
        self.plot_q.set_ylabel('Value')
        self.plot_q.set_ylim([-1, 1])
        self.plot_q.set_title('Action')

    def state_representation(self):
        state = self.game.get_state(self.algorithm.game.gui.surface_interaction.selected_player, True)
        state = state[:, :, 0]
        self.plot_state.cla()
        self.plot_state.axis('off')
        self.plot_state.imshow(state, interpolation='nearest')

    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')

        if len(self.x) <= self.i:
            self.x.append(self.i)
            self.losses.append(0)
            self.average_losses.append(loss)

        self.losses[self.i] = loss
        self.average_losses[self.i] = (self.average_losses[self.i] + loss) / 2

        if self.i % self.update_interval == 0:
            self.loss()
            self.action_distribution()
            self.state_representation()
            self.q_values()

        self.canvas.draw()
        surface = pygame.image.fromstring(self.renderer.tostring_rgb(), self.canvas.get_width_height(), "RGB")
        self.game.gui.surface_plot.plots["loss"] = surface

        self.i += 1
