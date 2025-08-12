import torch

from framework.prelude import *


class Controller(torch.nn.Module):
    def __init__(self, parameters: Individual = None):
        super(Controller, self).__init__()

        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(6, 12),
            torch.nn.Mish(),
            torch.nn.Linear(12, 12),
            torch.nn.Mish(),
            torch.nn.Linear(12, 12),
            torch.nn.Mish(),
            torch.nn.Linear(12, 6),
            torch.nn.Mish(),
            torch.nn.Linear(6, 2),
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
        return self.sequential(input_)
