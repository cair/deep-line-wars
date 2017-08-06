import random

from rl.dqn_1.PlotEngine import PlotEngine
from rl.mcqt.Node import Node


class Agent:
    def __init__(self, game):

        self.game = game
        self.player = None

        self.action_size = None

        self.max_length = 20000
        self.num_games = 10000

        self.e_start = 1
        self.e_end = .1
        self.e_steps = 10000000
        self.e_decay = (self.e_end - self.e_start) / self.e_steps
        self.e = self.e_start

        #self.summary = []
        #self.loss_list = []
        #self.q_values = None
        self.action_distribution = None

        #self.plot_engine = PlotEngine(self.game, self)
        #self.plot_engine.start()

        self.i = 0
        self.game_num = 0

    def init(self, player):
        self.player = player

        self.action_size = len(self.player.action_space)
        self.action_distribution = [0 for _ in range(self.action_size)]
        self.root = Node(0, None, None, None, self.action_size, True)
        self.node = self.root

    def reset(self):
        #self.plot_engine.new_game()

        if self.game.winner is not self.player or self.game.winner is not None:
            self.node.q = -10000000
            self.node.backprop_to_root()
        else:
            print('+'.join(self.summary))
        #print("Game %s" % self.game_num, "State: %s " % self.node.s, "Epsilon: %s" % self.e, "Sequence: %s" % len(self.summary), "Root-Q: %s" % self.root.get_q_values())
        #print(self.summary)
        self.i = 0
        self.loss_list.append(self.player.opponent.health)
        self.summary = []
        self.node = self.root
        self.game_num += 1


    def reward_fn(self):
        #reached_red = self.player.enemy_unit_reached_red()
        #if reached_red:
        #    score = -1
        #else:
        #    score = .1

        #score = ((self.player.health - self.player.opponent.health) / 50) + 0.01
        score = self.player.health
        return score

    def update(self, seconds):

        # 1. Do action
        # 2. Observe
        # 3. Get reward
        # 4. Transition
        # 3. Train
        # 4. set state+1 to state

        # Do action
        if random.uniform(0, 1) >= self.e:
            a = self.node.argmax_a()
        else:
            a = self.node.random_a()

        self.action_distribution[a] += 1
        self.q_values = self.node.get_q_values()
        self.summary.append(str(a))

        if self.e > 0:
            self.e += self.e_decay

        # Observe
        s1, r, terminal, _ = self.game.step(self.player, a)
        r = self.reward_fn()

        # Create new node
        self.node = self.node.transition(a, r)

        # Train
        self.node.backprop()

        # IF TERMINAL
        if terminal:
            print("Game %s" % self.i, "State: %s " % self.node.s, "Epsilon: %s" % self.e)
            print('+'.join(self.summary))

        self.i += 1
