import cv2
import numpy as np
#import pygame

class GUI:

    def __init__(self, game):
        #SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
        #self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.game = game
        self.tile_colors = {
            0: (0, 123, 12),  # Grass
            1: (255, 20, 147),  # Goal / Hot zone
            2: (0, 128, 255),  # Mid Zone
            3: (0, 255, 255)
        }
        self.tile_size = 32


        self.config_draw_friendly = self.game.config.gui_draw_friendly
        self.canvas = np.zeros((
            self.game.width * self.tile_size,
            self.game.height * self.tile_size,
            3
        ), dtype=np.uint8)

    def caption(self):
        print("%s - DeepLineWars v1.0 [%sfps|%sups]" % (self.game.id, self.game.frame_counter, self.game.update_counter))

    def event(self):
        pass

    def get_health_color(self, n):
        R = (255 * n)
        G = (255 * (1 - n))
        B = 0
        return R, G, B

    def draw(self, ignore=False):
        if ignore:
            return False


        # DRAW background
        for player in self.game.players:
            health_percent = 1 - max(0, player.health / 50)
            color = self.get_health_color(health_percent)

            self.canvas[
                player.territory[0]:player.territory[0]+player.territory[2],
                player.territory[1]:player.territory[1]+player.territory[3]] = color

            self.canvas[
                player.spawn_x*self.tile_size:(player.spawn_x*self.tile_size)+self.tile_size,
                0:self.game.height*self.tile_size] = self.tile_colors[1]

        for center in self.game.center_area:
            c = center * self.tile_size

            self.canvas[
            c:c+self.tile_size,
            0:self.game.height*self.tile_size] = self.tile_colors[2]


        # Draw Units
        for player in self.game.players:

            if not self.config_draw_friendly:
                if self.game.selected_player == player:
                    continue

            for unit in player.units:
                x = int((unit.x * 32) + (32 * (1 - (unit.tick_counter / unit.tick_speed))) * unit.player.direction)
                y = int((unit.y * 32))

                self.canvas[
                x:x+self.tile_size,
                y:y+self.tile_size] = unit.icon_image

        # Draw Buildings
        for player in self.game.players:
            for building in player.buildings:
                x = int(building.x * self.tile_size)
                y = int(building.y * self.tile_size)
                self.canvas[
                x:x+self.tile_size,
                y:y+self.tile_size] = building.icon_image

        # Player cursors
        for player in self.game.players:
            x = player.virtual_cursor_x * self.tile_size
            y = player.virtual_cursor_y * self.tile_size

            self.canvas[
            x:x+self.tile_size,
            y:y+self.tile_size] = player.cursor_colors  # cursor = np.tile(player.cursor_colors, (32, 32, 1))

        #self.screen.blit(pygame.surfarray.make_surface(self.canvas), (0,0))
        #pygame.display.flip()
        #cv2.imwrite("/home/per/img.png", self.canvas)


    def quit(self):
        pass

    def get_state(self, grayscale=False):
        image = np.array(self.canvas)
        if grayscale:
            return cv2.cvtColor(self.canvas, cv2.COLOR_RGB2GRAY)
        return image

    def draw_screen(self):
        pass