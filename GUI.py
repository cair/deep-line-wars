import pygame
import numpy as np


class GUI:

    def __init__(self, game):
        pygame.init()
        self.font = pygame.font.SysFont("Comic Sans MS", 25)
        self.bfont = pygame.font.SysFont("Comic Sans MS", 40)

        self.game = game
        self.selected_player = 0
        self.selected_unit = 0
        self.selected_unit_type = None
        self.selected_building = 0
        self.top_height = 50
        self.bot_height = 128

        self.gw = self.game.config["width"] * self.game.config["tile_width"]
        self.game_height = (self.game.config["height"] * self.game.config["tile_height"]) # Height of game graphics
        self.gh = self.bot_height + self.top_height + self.game_height
        self.screen = pygame.display.set_mode((self.gw, self.gh))
        self.background = pygame.Surface(self.screen.get_size())
        self.background =  self.background.convert()
        self.background.fill((0, 0, 0))

        pygame.display.set_caption("DeepLineWars v1.0")

        self.tiles = {
            0: (0, 123, 12),  # Grass
            1: (255, 25, 0),  # Goal / Hot zone
            2: (0, 128, 255),  # Mid Zone
            3: (0, 255, 255)
        }

        self.btn_p1 = pygame.Rect(self.gw - 100, self.top_height + self.game_height, 100, 100)
        self.btn_p2 = pygame.Rect(self.gw - 201, self.top_height + self.game_height, 100, 100)
        self.btn_level_up = pygame.Rect(self.gw - 302, self.top_height + self.game_height, 100, 100)
        self.btn_spawn = None
        self.unit_list = []
        self.building_list = []

        # Create rectangles for map
        self.map_rects = [[] for x in range(self.game.map[0].shape[0])]
        for x in range(self.game.map[0].shape[0]):
            for y in range(self.game.map[0].shape[1]):
                rect = pygame.Rect(x*32, (y*32) + self.top_height, 32, 32)
                self.map_rects[x].append((rect, x, y, self.game.map[0][x][y]))

    def player(self):
        return self.game.players[self.selected_player]

    def caption(self):
        pygame.display.set_caption("DeepLineWars v1.0 [%sfps|%sups]" % (self.game.frame_counter, self.game.update_counter))

    def draw_level_up(self):
        pygame.draw.rect(self.screen, (0, 255, 255), self.btn_level_up)
        level = self.font.render("^ Level %s ^" % (self.player().level + 1), 1, (255, 0, 0))
        gold = self.font.render("Gold: %s" % (self.player().levels[0][0]), 1, (255, 0, 0))
        lumber = self.font.render("Lumber: %s" % (self.player().levels[0][1]), 1, (255, 0, 0))

        self.screen.blit(level, (self.gw - 295, self.top_height + self.game_height + 10, 100, 100))
        self.screen.blit(gold, (self.gw - 295, self.top_height + self.game_height + 40, 100, 100))
        self.screen.blit(lumber, (self.gw - 295, self.top_height + self.game_height + 60, 100, 100))

    def draw_unit_select(self):
        self.unit_list.clear()

        i = 0
        selected_unit = None
        for idx, unit in enumerate(self.game.unit_shop):
            p_level = self.game.players[self.selected_player].level
            if p_level < unit.level:
                continue

            rect = unit.icon_image.get_rect()
            rect[0] = i * rect[2]
            rect[1] = self.top_height + self.game_height
            self.unit_list.append([rect, i])

            self.screen.blit(unit.icon_image, rect)
            if i == self.selected_unit:
                selected_unit = unit
                self.selected_unit_type = (idx, unit)

                pygame.draw.rect(self.screen, (255, 125, 0), rect, 4)

            i += 1
        self.btn_spawn = pygame.Rect(i * rect[2], self.top_height + self.game_height, 128, 64)
        pygame.draw.rect(self.screen, (255, 125, 0), self.btn_spawn, 3)
        txt_purchase = self.font.render("Buy %s" % selected_unit.name, 1, (0, 128, 255))
        self.screen.blit(txt_purchase, (i * rect[2] + 10,  self.top_height + self.game_height + 22))

    def draw_building_select(self):
        self.building_list.clear()

        i = 0
        for building in self.game.building_shop:
            p_level = self.game.players[self.selected_player].level
            if p_level < building.level:
                continue

            rect = building.icon_image.get_rect()
            rect[0] = i * rect[2]
            rect[1] = self.top_height + self.game_height + rect[3]

            self.building_list.append([rect, building])
            self.screen.blit(building.icon_image, rect)

            if i == self.selected_building:
                pygame.draw.rect(self.screen, (255, 125, 0), rect, 4)

            i += 1

    def draw_player_select(self):
        # Draws buttons which indicates which player is selected
        p1_color = (192, 192, 192) if self.selected_player is 0 else (0, 125, 255)
        p2_color = (192, 192, 192) if self.selected_player is 1 else (0, 125, 255)

        pygame.draw.rect(self.screen, p1_color, self.btn_p1)
        pygame.draw.rect(self.screen, p2_color, self.btn_p2)

        p1 = self.font.render("Player 1", 1, (255, 0, 0))
        p2 = self.font.render("Player 2", 1, (0, 0, 255))
        self.screen.blit(p1, (self.gw - 180, self.top_height + self.game_height + 40, 100, 100))
        self.screen.blit(p2, (self.gw - 80, self.top_height + self.game_height + 40, 100, 100))

    def draw_units(self):
        # Get all units on map
        for player in self.game.players:
            for unit in player.units:
                pos = (unit.x * 32, self.top_height + unit.y * 32)
                unit_rect = pygame.Rect(pos[0], pos[1], 32, 32)
                self.screen.blit(pygame.transform.scale(unit.icon_image, (32, 32)), unit_rect)

    def draw_buildings(self):
        # Get all units on map
        for player in self.game.players:
            for building in player.buildings:
                pos = (building.x * 32, self.top_height + building.y * 32)
                unit_rect = pygame.Rect(pos[0], pos[1], 32, 32)
                self.screen.blit(pygame.transform.scale(building.icon_image, (32, 32)), unit_rect)

    def draw_map(self):
        self.screen.blit(self.background, (0, 0))

        # Draw Z = 0 - Environmental Layer

        for (x, y), v in np.ndenumerate(self.game.map[0]):
            pygame.draw.rect(self.screen, self.tiles[v], self.map_rects[x][y][0])
            pygame.draw.rect(self.screen, (0, 0, 0), self.map_rects[x][y][0], 1)

        x_pos = [10, self.gw - (self.gw / 2) + self.game.config["tile_width"] + 30]
        for i, player in enumerate(self.game.players):
            p_name = self.bfont.render("Player %s" % player.id, 1, (255, 255, 0))
            p_health = self.font.render("Health: %s" % player.health, 1, (0, 255, 0))
            p_gold = self.font.render("Gold: %s" % player.gold, 1, (255, 255, 0))
            p_lumber = self.font.render("Lumber: %s" % player.lumber, 1, (255, 255, 0))
            p_income = self.bfont.render("Income: %s" % player.income, 1, (255, 255, 0))

            self.screen.blit(p_name, (x_pos[i], 10))
            self.screen.blit(p_health, (x_pos[i] + 120, 5))
            self.screen.blit(p_gold, (x_pos[i] + 120, 20))
            self.screen.blit(p_lumber, (x_pos[i] + 120, 35))
            self.screen.blit(p_income, (x_pos[i] + 240, 10))

    def draw_game_clock(self):
        # Draw Game-Clock
        clock = self.font.render("%ss" % self.game.game_time(), 1, (0, 128, 255))
        self.screen.blit(clock, ((self.gw / 2) - 35,  10))

    def draw(self):
        self.draw_map()
        self.draw_game_clock()
        self.draw_units()
        self.draw_buildings()
        self.draw_player_select()
        self.draw_unit_select()
        self.draw_building_select()
        self.draw_level_up()

        pygame.display.flip()

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

                if self.btn_p1.collidepoint(pos):
                    self.selected_player = 1

                elif self.btn_p2.collidepoint(pos):
                    self.selected_player = 0

                elif self.btn_level_up.collidepoint(pos):
                    self.player().levelup()

                elif self.btn_spawn.collidepoint(pos):
                    print("Spawn!")
                    self.player().spawn(self.selected_unit_type)

                for unit in self.unit_list:
                    if unit[0].collidepoint(pos):
                        self.selected_unit = unit[1]

                for idx, building in enumerate(self.building_list):
                    if building[0].collidepoint(pos):
                        self.selected_building = idx

                for tile_row in self.map_rects:
                    for tile in tile_row:
                        if tile[0].collidepoint(pos):

                            building_data = self.building_list[self.selected_building][1]
                            self.player().build(tile[1], tile[2], building_data)

