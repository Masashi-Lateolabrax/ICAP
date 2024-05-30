import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # the number of node is 2.1179^(an index of layer)
        self.sequence = nn.Sequential(
            nn.Flatten(0, -1),
            nn.Linear(6, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
            nn.Tanh(),
        )

    def forward(self, x) -> torch.Tensor:
        x = self.sequence.forward(x)
        return x

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
