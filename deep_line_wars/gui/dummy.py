
class GUI:
    def __init__(self, game):
        self.game = game

    def caption(self):
        print("%s - DeepLineWars v1.0 [%sfps|%sups]" % (self.game.id, self.game.frame_counter, self.game.update_counter))

    def event(self):
        pass

    def draw(self):
        pass

    def quit(self):
        pass

    def get_state(self, grayscale=False, flip=False):
        return self.game._get_raw_state()

    def draw_screen(self):
        pass
