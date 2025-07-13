import torch

from framework.prelude import Individual


class RobotNeuralNetwork(torch.nn.Module):
    def __init__(self, parameters: Individual = None):
        super(RobotNeuralNetwork, self).__init__()

        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(6, 3),
            torch.nn.Tanhshrink(),
            torch.nn.Linear(3, 2),
            torch.nn.Tanh()
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
