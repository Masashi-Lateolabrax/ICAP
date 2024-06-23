import torch
from torch import nn

from lib.analizer.neural_network import Debugger

from .hyper_parameters import StaticParameters


class NeuralNetwork(nn.Module):
    def __init__(self, debug=False):
        super(NeuralNetwork, self).__init__()

        self.debugger = Debugger(debug)

        self.sequence = nn.Sequential(
            nn.Flatten(0, -1),
            self.debugger.create_investigator("l0"),
            nn.Linear(StaticParameters.input_size(), 4),
            self.debugger.create_investigator("l1"),
            nn.Tanh(),
            self.debugger.create_investigator("l2"),
            nn.Linear(4, 2),
            self.debugger.create_investigator("l3"),
            nn.Sigmoid(),
            self.debugger.create_investigator("l4"),
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
