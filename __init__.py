

try:
    import gym
    from . import gym_dlw

    all_classes = dir(gym_dlw)
    all_classes = [x for x in all_classes if "DeepLineWars" in x]
    all_classes = [getattr(gym_dlw, x) for x in all_classes]

    for cls in all_classes:
        if not hasattr(cls, "id"):
            continue

        gym.envs.register(
            id=cls.id,
            entry_point=f'deep_line_wars.gym_dlw:{cls.__name__}',
            max_episode_steps=1000,
        )

except ImportError as e:
    print(e)
    pass
