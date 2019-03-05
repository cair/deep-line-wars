from .gui import dummy, pygame
default_config = dict(
    game=dict(
        width=11,
        height=11,
        tile_width=32,
        tile_height=32
    ),
    mechanics=dict(
        complexity=dict(
            build_anywhere=False
        ),
        start_health=50,
        start_gold=50,
        start_lumber=0,
        start_income=20,
        income_frequency=10,
        ticks_per_second=10,
        fps=10,
        ups=10008000,
        income_ratio=0.20,
        kill_gold_ratio=0.10
    ),
    gui=pygame.GUI,
    gui_window=True,
    gui_draw_friendly=True,
    state_representation="RGB"  # RAW, RGB, L
)

building = [
    dict(
        id=1,
        name="Basic-Tower",
        icon="tower_1.png",
        health=100,
        attack_min=2,
        attack_max=4,
        attack_pen=2,
        attack_speed=3,
        attack_range=3,
        level=0,
        gold_cost=10

    ),
    dict(
        id=2,
        name="Fast-Tower",
        icon="tower_2.png",
        health=100,
        attack_min=2,
        attack_max=4,
        attack_pen=2,
        attack_speed=4,
        attack_range=3,
        level=0,
        gold_cost=20
    ),
    dict(
        id=3,
        name="Faster-Tower",
        icon="lazer_tower.png",
        health=100,
        attack_min=4,
        attack_max=6,
        attack_pen=3,
        attack_speed=5,
        attack_range=3,
        level=1,
        gold_cost=30
    )
]

level_up = [
    [100, 0],
    [1000, 0],
    [10000, 0],
    [100000, 1]
]

unit_type = [
    dict(
        id=0,
        name="Ground",
        collision=True
    ),
    dict(
        id=1,
        name="Flying",
        collision=False
    )
]

unit = [
    dict(
        name="Militia",
        icon="militia.png",
        health=40,
        armor=2,
        speed=1,
        type=0,
        level=0,
        gold_drop=1,
        gold_cost=10
    ),
    dict(
        name="Footman",
        icon="footman.png",
        health=80,
        armor=4,
        speed=1,
        type=0,
        level=0,
        gold_drop=1,
        gold_cost=20
    ),
    dict(
        name="Grunt",
        icon="grunt.png",
        health=140,
        armor=4,
        speed=1,
        type=0,
        level=0,
        gold_drop=1,
        gold_cost=40
    ),
    dict(
        name="Armored Grunt",
        icon="armored_grunt.png",
        health=190,
        armor=6,
        speed=1.2,
        type=0,
        level=0,
        gold_drop=1,
        gold_cost=100
    )

]

upgrades = [
    dict()
]
