import random
from rl.mcqt.Node import Node


class MonteCarloSequence:

    def __init__(self, action_size):
        self.action_size = action_size
        self.sequence = []
        self.add_sequence()
        self.cursor = 0
        self.current_q = 0


    def add_sequence(self):
        self.sequence.append([0 for _ in range(self.action_size)])

    def next(self):
        self.cursor += 1

    def previous(self):
        self.cursor += 1

    def set(self, i):
        self.cursor = i

    def get(self):
        return self.sequence[self.cursor]

    def argmax_a(self):
        return self.sequence[self.cursor].index(max(self.sequence[self.cursor]))

    def random_a(self):
        return random.randint(0, self.action_size - 1)


actions = [x for x in range(-32, 32)]
mcs = MonteCarloSequence(len(actions))
max_length = 2000
num_games = 10000

e_start = 1
e_end = .1
e_steps = 1000000
e_decay = (e_end - e_start) / e_steps
e = e_start

i = 0
while True:
    state_val = 0
    state_goal = 100
    s = 0
    summary = []
    mcs.set(0)
    mcs.current_q = 0


    # Episode in the game
    while True:

        # Do action
        if random.uniform(0, 1) >= e:
            a = mcs.argmax_a()
        else:
            a = mcs.random_a()

            e = max(0, e + e_decay)

        # Observe
        state_val += actions[a]
        summary.append(str(actions[a]))

        # Get reward
        r = abs(state_val - state_goal) * -1

        # Create new node
        mcs.add_sequence()  # Add next sequence

        # Train
        new_q = mcs.current_q + 0.70 * (r + (0.99 * mcs.get()[a]) - mcs.current_q)
        mcs.get()[a] = new_q
        mcs.current_q

        # IF TERMINAL
        if state_val == state_goal:
            print("Game %s" % i, "State: %s " % mcs.cursor, "Epsilon: %s" % e)
            print('+'.join(summary))

            for x in range(0, mcs.cursor):
                print(mcs.sequence[x])

            max_length = mcs.cursor
            break

        if mcs.cursor >= max_length:
            print(":O")
            mcs.get()[a] = -1000
            break

        mcs.next()
    i += 1

    #if i >= num_games:
    #    break