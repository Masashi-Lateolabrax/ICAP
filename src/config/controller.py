import torch

from framework.prelude import *


class Controller(torch.nn.Module):
    def __init__(self, parameters: Individual = None):
        super(Controller, self).__init__()

        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(6, 12),
            torch.nn.Mish(),
            torch.nn.Linear(12, 4),
            torch.nn.Mish(),
            torch.nn.Linear(4, 2),
            torch.nn.Sigmoid()
        )

        if parameters is not None:
            assert len(parameters) == self.dim, "Parameter length does not match the network's parameter count."
            torch.nn.utils.vector_to_parameters(
                torch.tensor(parameters, dtype=torch.float32),
                self.parameters()
            )

    @property
    def dim(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_):
        x = self.sequential(input_)

        # clip_to_one_indexes = x > 0.99
        # clip_to_zero_indexes = x < 0.01
        # x[clip_to_one_indexes] = 0.99
        # x[clip_to_zero_indexes] = 0.01

        return x
