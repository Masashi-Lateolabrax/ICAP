import torch
from torch import nn

from lib.analizer.neural_network import Debugger


class NeuralNetwork(nn.Module):
    def __init__(self, debug=False):
        super(NeuralNetwork, self).__init__()

        self.debugger = Debugger(debug)

        self.sequence1 = nn.Sequential(
            nn.Flatten(0, -1),
            self.debugger.create_investigator("s1l0"),

            nn.Linear(6, 12),
            self.debugger.create_investigator("s1l1"),
            nn.Tanh(),

            nn.Linear(12, 6),
            self.debugger.create_investigator("s1l2"),
            nn.Tanh(),

            nn.Linear(6, 3),
            self.debugger.create_investigator("s1l3"),
            nn.Tanh(),

            self.debugger.create_investigator("s1l4"),
        )
        self.sequence2 = nn.Sequential(
            self.debugger.create_investigator("s2l0"),

            nn.Linear(1, 5),
            self.debugger.create_investigator("s2l1"),
            nn.Tanh(),

            nn.Linear(5, 5),
            self.debugger.create_investigator("s2l2"),
            nn.Tanh(),

            nn.Linear(5, 3),
            self.debugger.create_investigator("s2l3"),
            nn.Tanh(),

            self.debugger.create_investigator("s2l4"),
        )
        self.sequence3 = nn.Sequential(
            self.debugger.create_investigator("s3l0"),
            nn.Sigmoid(),
            self.debugger.create_investigator("s3l1")
        )

    def forward(self, sight, pheromone) -> torch.Tensor:
        x1 = self.sequence1.forward(sight)
        x2 = self.sequence2.forward(pheromone)
        return self.sequence3.forward(x1 + x2)

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
