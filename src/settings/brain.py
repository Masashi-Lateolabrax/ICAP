import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(65, 30),
            nn.Tanh(),
            nn.Linear(30, 4),
        )
        self.af1 = nn.Softmax(dim=0)
        self.af2 = nn.Sigmoid()

    def forward(self, x):
        x = self.sequence.forward(x)
        x1 = self.af1.forward(x[0:3])
        x2 = self.af2.forward(x[3])
        x2 = x2.unsqueeze(0)
        return torch.cat([x1, x2])

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
