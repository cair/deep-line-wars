class Callbacks:

    def __init__(self):
        self._on_episode_end = []
        self._on_episode_start = []

    def add_on_episode_start(self, fn):
        self._on_episode_start.append(fn)

    def add_on_episode_end(self, fn):
        self._on_episode_end.append(fn)

    def on_episode_start(self):
        for fn in self._on_episode_start:
            fn()

    def on_episode_end(self):
        for fn in self._on_episode_end:
            fn()

