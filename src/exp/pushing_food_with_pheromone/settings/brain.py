import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.sequence1 = nn.Sequential(
            nn.Flatten(0, -1),
            nn.Linear(6, 12),
            nn.Tanh(),
            nn.Linear(12, 6),
            nn.Tanh(),
            nn.Linear(6, 3),
            nn.Tanh(),
        )
        self.sequence2 = nn.Sequential(
            nn.Linear(1, 5),
            nn.Tanh(),
            nn.Linear(5, 3),
            nn.Tanh(),
        )

    def forward(self, sight, pheromone) -> torch.Tensor:
        x1 = self.sequence1.forward(sight)
        x2 = self.sequence2.forward(pheromone)
        return nn.functional.sigmoid(x1 + x2)

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
