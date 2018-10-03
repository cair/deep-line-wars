from torch import nn


class Model(nn.Module):

    def __init__(self, spec):
        super(Model, self).__init__()

        if not all(x in spec for x in ["model"]):
            raise AttributeError("Missing required spec parameters!")

        self.input_shape  = spec["input"]["shape"]
        self.output_shape = spec["output"]["shape"]

        self.build = spec["model"].build.__get__(self, Model)
        self.forward = spec["model"].forward.__get__(self, Model)

        self.build()

        self.optimizer = spec["optimizer"]["model"](self.parameters(), **spec["optimizer"]["options"])
