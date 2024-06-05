import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # the number of node is 2.1179^(an index of layer)
        self.sequence1 = nn.Sequential(
            nn.Flatten(0, -1),
            nn.Linear(6, 4),
            nn.Tanh(),
            nn.Linear(4, 3),
            nn.Tanh(),
        )
        self.sequence2 = nn.Sequential(
            nn.Linear(1, 3),
            nn.Tanh(),
            nn.Linear(3, 3),
            nn.Tanh(),
        )

    def forward(self, sight, pheromone) -> torch.Tensor:
        x1 = self.sequence1.forward(sight)
        x2 = self.sequence2.forward(pheromone)
        return nn.functional.tanh((x1 + x2) * 0.7)

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
