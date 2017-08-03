import random


class Node:
    def __init__(self, s, a, r, parent, n_actions, init_children=False):
        self.s = s
        self.a = a
        self.r = r
        self.q = 0
        self.n_actions = n_actions
        self.has_init_children = False

        if init_children:
            self.init_children()
        else:
            self.children_nodes = [None for a in range(n_actions)]  # Not inited

        self.parent = parent

    def init_children(self):
        self.children_nodes = [Node(self.s + 1, self.a, None, self, self.n_actions) for a in range(self.n_actions)]
        self.has_init_children = True

    def backprop_to_root(self):

        current = self
        while current.parent is not None:
            current.parent.q = current.q * 0.95
            current = current.parent

    def backprop(self):
        s = self.parent
        s1 = self
        r = self.r
        lr = .0004
        df = 0.99
        new_q = s.q + lr * (r + df * s1.q - s.q)
        s.q = new_q
        #print("Backprop: s:%s, s1:%s, r:%s, a:%s" % (s.s, s1.s, r, s1.a), " | s.q <= %s + %s * (%s + %s * %s - %s)" % (s.q, lr, r, df, s1.q, s.q), "Old-Q: %s, New-Q: %s" % (s.q, new_q))

    def random_a(self):
        if not self.has_init_children:
            self.init_children()

        return random.randint(0, len(self.children_nodes) - 1)

    def argmax_a(self):
        if not self.has_init_children:
            self.init_children()
        return self.children_nodes.index(max(self.children_nodes))

    def get_q_values(self):
        return [round(n.q, 2) if n is not None else 0 for n in self.children_nodes]

    def new(self, a, r):
        new_node = Node(self.s + 1, a, r, self, self.n_actions)
        return new_node

    def transition(self, a, r):
        next_node = self.children_nodes[a]
        next_node.parent = self
        next_node.a = a
        next_node.r = r
        return next_node

    def __lt__(self, other):
        return self.q < other.q

    def __eq__(self, other):
        return self.q == other.q
