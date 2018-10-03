import random


class EpsilonDecay:

    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.e = self.start
        self.steps = steps
        self.decay = -((self.start - self.end) / self.steps)

    def eval(self):
        t = False
        if random.random() < self.e:
            t = True

        self.e = max(self.end, self.e + self.decay)
        return t