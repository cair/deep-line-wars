import math
import config

class Algorithm:


    def __init__(self, game, player, representation):
        self.representation = representation
        self.game = game
        self.player = player
        self.build_instruction_x_offset_start = 1
        self.build_instruction_x_offset_end = 10 # TODO
        self.build_instruction_y_offset_start = 0
        self.build_instruction_y_offset_end = config.game["height"] - 1
        self.buildings_per_level = 10


        self.spawn_instructions = [
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            3,
            3,
        ]

        # Step 0 = Do build instruction
        # Step 1 = Do spawn instruction
        self.step = 0
        self.counter = 0
        self.build_successes = 0
        self.spawn_successes = 0
        self.spawn_per_unit_type = 10
        self.spawn_counter_before_no_money = 0

        self.spawn_delay = 4
        self.iteration = 0

    def reset(self):
        self.step = 0
        self.counter = 0
        self.build_successes = 0
        self.spawn_successes = 0
        self.spawn_counter_before_no_money = 0
        self.spawn_delay = 4
        self.iteration = 0

    def update(self, seconds):

        if self.step == 0:
            # Build instruction

            build_idx = math.floor(self.build_successes / self.buildings_per_level)
            can_afford = self.player.can_afford_idx(build_idx)

            if can_afford:
                sign = 1
                x = self.build_instruction_x_offset_start
                y = self.build_instruction_y_offset_start
                for i in range(self.build_successes):
                    y += sign

                    if y >= self.build_instruction_y_offset_end or y <= self.build_instruction_y_offset_start:
                        x += 2
                        sign *= -1

                rel_x, rel_y = self.player.rel_pos_to_abs(x, y)

                succ = self.player.build(rel_x, rel_y, self.player.get_building_idx(build_idx))
                #print(self.player.id, rel_x, rel_y, succ)
                if succ:
                    self.build_successes += 1

                self.step = 1

        elif self.step == 1:
            # Spam remaining gold

            if self.iteration % self.spawn_delay == 0:

                i = math.floor(self.spawn_counter_before_no_money / self.spawn_per_unit_type)
                us = self.player.available_units()
                u = us[min(len(us) - 1, i)]

                if not self.player.can_afford_unit(u):
                    self.step = 0
                    self.spawn_counter_before_no_money = 0
                    return True

                self.spawn_counter_before_no_money += 1
                self.player.spawn([i, u])

                if self.spawn_counter_before_no_money % 5 == 0:
                    self.step = 0


        self.iteration += 1











        pass

    def on_defeat(self):
        pass

    def on_victory(self):
        pass