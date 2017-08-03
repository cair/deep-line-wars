import random
from rl.mcqt.Node import Node

actions = [x for x in range(-32, 32)]
node = Node(0, None, None, None, len(actions), True)
root = node
max_length = 20000
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
    node = root
    summary = []


    # Episode in the game
    while True:

        # Do action
        if random.uniform(0, 1) >= e:
            a = node.argmax_a()
        else:
            a = node.random_a()

        if e > 0:
            e += e_decay


        # Observe
        state_val += actions[a]
        summary.append(str(actions[a]))

        # Get reward
        r = abs(state_val - state_goal) * -1

        # Create new node
        node = node.transition(a, r)

        # Train
        node.backprop()

        # IF TERMINAL
        if state_val == state_goal:
            print("Game %s" % i, "State: %s " % node.s, "Epsilon: %s" %e)
            print('+'.join(summary))
            max_length = node.s
            break

        if node.s >= max_length:
            node.q = -10000000
            break


    i += 1

    #if i >= num_games:
    #    break