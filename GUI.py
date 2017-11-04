import pygame
import numpy as np
from PIL import Image


class TopSurface(pygame.Surface):
    def __init__(self, size, game):
        pygame.Surface.__init__(self, size=size)
        self.game = game
        self.font = pygame.font.SysFont("Comic Sans MS", 25)
        self.b_font = pygame.font.SysFont("Comic Sans MS", 40)

    def draw_game_clock(self):
        # Draw Game-Clock
        clock = self.font.render("%ss" % self.game.game_time(), 1, (0, 128, 255))
        self.blit(clock, ((self.get_width() / 2) - 35, 10))

    def draw_player_data(self):
        x_pos = [10, self.get_width() - (self.get_width() / 2) + self.game.config.game.tile_width + 30]
        for i, player in enumerate(self.game.players):
            p_name = self.b_font.render("Player %s" % player.id, 1, (255, 255, 0))
            p_health = self.font.render("Health: %s" % player.health, 1, (0, 255, 0))
            p_gold = self.font.render("Gold: %s" % player.gold, 1, (255, 255, 0))
            p_lumber = self.font.render("Lumber: %s" % player.lumber, 1, (255, 255, 0))
            p_income = self.b_font.render("Income: %s" % player.income, 1, (255, 255, 0))
            p_winner = self.font.render("W: %s | %f" % (self.game.statistics[player.id], (self.game.statistics[player.id] / max(1, sum(self.game.statistics.values())))), 1, (255, 255, 0))

            self.blit(p_name, (x_pos[i], 10))
            self.blit(p_health, (x_pos[i] + 120, 5))
            self.blit(p_gold, (x_pos[i] + 120, 20))
            self.blit(p_lumber, (x_pos[i] + 120, 35))
            self.blit(p_income, (x_pos[i] + 240, 10))
            self.blit(p_winner, (x_pos[i] + 240, 35))

    def draw(self):
        self.fill((80,80,80))
        self.draw_game_clock()
        self.draw_player_data()


class PlotSurface(pygame.Surface):
    def __init__(self, size, game):
        pygame.Surface.__init__(self, size=size)
        self.game = game
        self.plots = {}

    def draw(self):
        self.fill((255, 255, 0))

        for idx, (key, plot) in enumerate(self.plots.items()):
            if plot:
                n_plots = len(self.plots)
                plot_w = int(self.game.gui.game_width / n_plots)
                plot_h = min(plot_w, self.game.gui.surface_plot_h)
                scaled = pygame.transform.scale(plot, (plot_w, plot_h))
                self.blit(scaled, (idx * plot_w, 0))


    def handle(self):
        pass


class GameSurface(pygame.Surface):
    def __init__(self, size, game):
        pygame.Surface.__init__(self, size=size)
        self.game = game
        self.tiles = {   #TODO make file
            0: (0, 123, 12),  # Grass
            1: (255, 20, 147),  # Goal / Hot zone
            2: (0, 128, 255),  # Mid Zone
            3: (0, 255, 255)
        }

        self.config_draw_friendly = self.game.config.gui.draw_friendly

        # Create rectangles for map
        self.map_rects = [[] for x in range(self.game.map[0].shape[0])]
        for x in range(self.game.map[0].shape[0]):
            for y in range(self.game.map[0].shape[1]):
                rect = pygame.Rect(x * 32, (y*32), 32, 32)
                owned_by = 0 if x < int(self.game.map[0].shape[0] / 2) else 1
                item = (rect, x, y, self.game.map[0][x][y], owned_by)
                self.map_rects[x].append(item)


    def get_health_color(self, n):
        R = (255 * n)
        G = (255 * (1 - n))
        B = 0
        return R, G, B

    def draw_map(self):
        # Draw Z = 0 - Environmental Layer
        for (x, y), v in np.ndenumerate(self.game.map[0]):
            data = self.map_rects[x][y]
            owned_by = data[4]
            health_val = 1 - (self.game.players[owned_by].health / 50)
            color = self.tiles[data[3]] if data[3] != 0 else self.get_health_color(health_val)
            try:
                pygame.draw.rect(self, color, self.map_rects[x][y][0])
            except:
                pass
            #pygame.draw.rect(self, (0, 0, 0), self.map_rects[x][y][0], 1)   # Grid Effect

    def draw_units(self):
        # Get all units on map
        for player in self.game.players:

            if not self.config_draw_friendly:
                if self.game.gui.surface_interaction.selected_player == player:
                    continue

            for unit in player.units:
                pos = (unit.x * 32, unit.y * 32)
                unit_rect = pygame.Rect(pos[0], pos[1], 32, 32)
                self.blit(pygame.transform.scale(unit.icon_image, (32, 32)), unit_rect)
                pygame.draw.rect(self, player.player_color, (pos[0], pos[1], 32, 32), 2)   # Grid Effect

    def draw_buildings(self):
        # Get all units on map
        for player in self.game.players:
            for building in player.buildings:
                pos = (building.x * 32, building.y * 32)
                unit_rect = pygame.Rect(pos[0], pos[1], 32, 32)
                self.blit(pygame.transform.scale(building.icon_image, (32, 32)), unit_rect)

    def draw_cursor(self):
        for player in self.game.players:
            pygame.draw.rect(self, (255, 255, 0), [
                player.virtual_cursor_x * 32,(player.virtual_cursor_y * 32), 32, 32
            ])

    def draw(self):
        self.fill((0, 0, 0))
        self.draw_map()
        self.draw_units()
        self.draw_buildings()
        self.draw_cursor()


class InteractionSurface(pygame.Surface):
    def __init__(self, size, game):
        pygame.Surface.__init__(self, size=size)
        self.game = game
        self.font = pygame.font.SysFont("Comic Sans MS", 25)
        self.selected_player = game.players[0]  # Select first player available #
        self.selected_unit = None
        self.icon_size = (32, 32)
        self.unit_buttons = [[], None]

        self.btn_p1 = pygame.Rect(self.get_width() - 100, 0, 100, 100)
        self.btn_p2 = pygame.Rect(self.get_width()- 201, 0, 100, 100)
        self.btn_level_up = pygame.Rect(self.get_width() - 302, 0, 100, 100)

    def get_unit_buttons(self):
        return self.unit_buttons

    def draw_unit_select(self):
        self.unit_buttons = [[], None]

        available_units = self.selected_player.available_units()
        for idx, unit in enumerate(available_units):

            scaled = pygame.transform.scale(unit.icon_image, self.icon_size)
            button_pos = pygame.Rect(idx * self.icon_size[0], 0, self.icon_size[0], self.icon_size[1])
            self.unit_buttons[0].append(button_pos)
            self.blit(scaled, button_pos)

            if self.selected_unit and self.selected_unit == unit:
                pygame.draw.rect(self, (255, 125, 0), (idx * self.icon_size[0], 0, self.icon_size[0], self.icon_size[1]), 3)

        if self.selected_unit:
            spawn_button = pygame.Rect(len(available_units) * self.icon_size[0], 0, 128, 64)
            pygame.draw.rect(self, (255, 125, 0), spawn_button, 3)
            txt_purchase = self.font.render("Buy %s" % self.selected_unit.name, 1, (0, 128, 255))
            self.blit(txt_purchase, (len(available_units) * self.icon_size[0] + 10, 22))
            self.unit_buttons[1] = spawn_button

    def draw_building_select(self):
        self.building_list.clear()

        i = 0
        for building in self.game.building_shop:
            p_level = self.game.players[self.selected_player].level
            if p_level < building.level:
                continue

            rect = building.icon_image.get_rect()
            rect[0] = i * rect[2]
            rect[1] = self.stat_panel_height + self.game_grid_height + rect[3]

            self.building_list.append([rect, building])
            self.screen.blit(building.icon_image, rect)

            if i == self.selected_building:
                pygame.draw.rect(self.screen, (255, 125, 0), rect, 4)

            i += 1

    def draw_player_select(self):
        # Draws buttons which indicates which player is selected
        p1_color = (192, 192, 192) if self.selected_player is 0 else (0, 125, 255)
        p2_color = (192, 192, 192) if self.selected_player is 1 else (0, 125, 255)

        pygame.draw.rect(self, p1_color, self.btn_p1)
        pygame.draw.rect(self, p2_color, self.btn_p2)

        p1 = self.font.render("Player 1", 1, (255, 0, 0))
        p2 = self.font.render("Player 2", 1, (0, 0, 255))
        self.blit(p1, (self.get_width() - 180, 40, 100, 100))
        self.blit(p2, (self.get_width() - 80, 40, 100, 100))

    def draw(self):
        self.fill((23, 23, 0))
        self.draw_unit_select()
        self.draw_player_select()
        #self.draw_building_select() # TODO must implement in v2

    def event(self, event, pos):
        if self.unit_buttons[1] and self.unit_buttons[1].collidepoint(pos):
            print(self.selected_unit.type)
            self.selected_player.spawn((0, self.selected_unit))
            return

        for idx, unit_button in enumerate(self.unit_buttons[0]):
            if unit_button.collidepoint(pos):
                self.selected_unit = self.selected_player.available_units()[idx]
                return




class GUI:

    def __init__(self, game):
        pygame.init()

        # Game variables
        self.game = game

        # Font definition
        self.font = pygame.font.SysFont("Comic Sans MS", 25)
        self.bfont = pygame.font.SysFont("Comic Sans MS", 40)

        # GUI Interaction variables
        self.selected_player = 0
        self.selected_unit = 0
        self.selected_unit_type = None
        self.selected_building = 0


        # Size definitions
        self.stat_panel_height = 50
        self.bot_panel_height = 256
        self.game_grid_height = self.game.config.game.height * self.game.config.game.tile_height  # Height of game graphics
        self.game_width = self.game.config.game.width * self.game.config.game.tile_width
        self.plot_panel_height = int(self.game_width / 3)
        self.game_height = self.bot_panel_height + self.stat_panel_height + self.game_grid_height + self.plot_panel_height

        # Position definitions
        self.plot_surface_x = 0
        self.plot_surface_y = self.stat_panel_height + self.game_grid_height + self.bot_panel_height

        # PYGame variables
        self.screen = pygame.display.set_mode((self.game_width, self.game_height))
        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill((0, 0, 0))

        # [Surface] Game Area
        self.game_surface = pygame.Surface((self.game_width, self.game_grid_height))

        pygame.display.set_caption("DeepLineWars v1.0")

        self.surface_top_h = 55
        self.surface_game_h = self.game.config.game.height * self.game.config.game.tile_height
        self.surface_interaction_h = 100
        self.surface_plot_h = 400

        self.surface_top = TopSurface((self.game_width, self.stat_panel_height), game)
        self.surface_game = GameSurface((self.game_width, self.game_grid_height), game)
        self.surface_interaction = InteractionSurface((self.game_width, self.surface_interaction_h), game)
        self.surface_plot = PlotSurface((self.game_width, self.surface_plot_h), game)

        self.surface_top_y = 0
        self.surface_game_y = self.surface_top_h
        self.surface_interaction_y = self.surface_game_y + self.surface_game_h
        self.surface_plot_y = self.surface_interaction_y + self.surface_interaction_h

        self.i = 0

    def player(self):
        return self.game.players[self.selected_player]

    def caption(self):
        pygame.display.set_caption("DeepLineWars v1.0 [%sfps|%sups]" % (self.game.frame_counter, self.game.update_counter))

    def draw_level_up(self):
        pygame.draw.rect(self.screen, (0, 255, 255), self.btn_level_up)
        level = self.font.render("^ Level %s ^" % (self.player().level + 1), 1, (255, 0, 0))
        gold = self.font.render("Gold: %s" % (self.player().levels[0][0]), 1, (255, 0, 0))
        lumber = self.font.render("Lumber: %s" % (self.player().levels[0][1]), 1, (255, 0, 0))

        self.screen.blit(level, (self.game_width - 295, self.stat_panel_height + self.game_grid_height + 10, 100, 100))
        self.screen.blit(gold, (self.game_width - 295, self.stat_panel_height + self.game_grid_height + 40, 100, 100))
        self.screen.blit(lumber, (self.game_width - 295, self.stat_panel_height + self.game_grid_height + 60, 100, 100))

    def draw_heat_maps(self):
        for i, p in enumerate(self.game.players):
            data = self.game.generate_heatmap(p)
            data *= 255
            tmp_surf = pygame.Surface((data.shape[0], data.shape[1]))
            pygame.surfarray.blit_array(tmp_surf, data)
            tmp_surf = pygame.transform.scale(tmp_surf, (100, 100))
            tmp_surf = pygame.transform.rotate(tmp_surf, 90)
            self.screen.blit(tmp_surf, (759 + (i * 100) + i, 510))






    def draw(self):
        #self.draw_player_select()
        #self.draw_unit_select()
        #self.draw_building_select()
        #self.draw_level_up()
        #self.draw_heat_maps()

        self.surface_top.draw()
        self.screen.blit(self.surface_top, (0, 0))

        self.surface_game.draw()
        self.screen.blit(self.surface_game, (0, self.surface_game_y))

        self.surface_interaction.draw()
        self.screen.blit(self.surface_interaction, (0, self.surface_interaction_y))

        self.surface_plot.draw()
        self.screen.blit(self.surface_plot, (0, self.surface_plot_y))



        #self.screen.blit(self.plot_surface, (self.plot_surface_x, self.plot_surface_y))

        pygame.display.flip()

    def quit(self):
        pygame.display.quit()
        pygame.quit()

    def event(self):
        keybind_units = [
            pygame.K_1, pygame.K_2,
            pygame.K_3, pygame.K_4, pygame.K_5,
            pygame.K_6, pygame.K_7, pygame.K_8,
            pygame.K_9, pygame.K_0
        ]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game.running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    self.player().spawn(self.selected_unit_type)

                # Keybind 0-9 for selecting units
                for idx, k in enumerate(keybind_units):
                    if event.key == k:
                        self.selected_unit = self.unit_list[idx][1]
                        self.selected_unit_type = (idx, self.game.unit_shop[idx])

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = pygame.mouse.get_pos()

                if self.surface_interaction.get_rect(y=self.surface_interaction_y).collidepoint(pos):
                    self.surface_interaction.event(event, (pos[0], pos[1] - self.surface_interaction_y))
                    return


                """if self.btn_p1.collidepoint(pos):
                    self.selected_player = 1

                elif self.btn_p2.collidepoint(pos):
                    self.selected_player = 0

                elif self.btn_level_up.collidepoint(pos):
                    self.player().levelup()"""



                """for idx, building in enumerate(self.building_list):
                    if building[0].collidepoint(pos):
                        self.selected_building = idx

                for tile_row in self.map_rects:
                    for tile in tile_row:
                        if tile[0].collidepoint(pos):

                            building_data = self.building_list[self.selected_building][1]
                            self.player().build(tile[1], tile[2], building_data)"""





class NoGUI():
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