
import pygame


class PyGameViewWrapper:

    def __init__(self, game):
        pygame.init()
        self.screen = pygame.display.set_mode((500, 500))
        self.gui = None
        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill((0, 0, 0))

    @staticmethod
    def wrap(guicls):
        return lambda game: guicls()

    def draw_screen(self):
        self.background.blit(self.gui.get_state(), (0, 0))
        self.screen.blit(self.background, (0, 0))
        pygame.display.flip()

