import torch
from torch import nn

from .hyper_parameters import StaticParameters


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.sequence = nn.Sequential(
            nn.Flatten(0, -1),
            nn.Linear(StaticParameters.input_size(), 4),
            nn.Tanh(),
            nn.Linear(4, 2),
            nn.Sigmoid(),
        )

    def forward(self, input_) -> torch.Tensor:
        return self.sequence.forward(input_)

    def load_para(self, para):
        with torch.no_grad():
            s = 0
            for p in self.parameters():
                if not p.requires_grad:
                    continue
                n = p.numel()
                p.data.copy_(torch.tensor(para[s:s + n]).view(p.size()))
                s += n

    def num_dim(self):
        dim = 0
        for p in self.parameters():
            if not p.requires_grad:
                continue
            dim += p.numel()
        return dim
