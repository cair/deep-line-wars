from deep_line_wars_examples.per_rl.memory.experience_replay import ExperienceReplay

experience_replay = ExperienceReplay


class Memory:

    def __init__(self, spec):

        getattr(spec["memory"]["model"], "__init__")(self, spec)
        for k, fn in spec["memory"]["model"].__dict__.items():
            if "__" in k:
                continue
            setattr(self, k, getattr(spec["memory"]["model"], k).__get__(self, Memory))
