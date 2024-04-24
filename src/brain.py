import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv1d(3, 10, 8),  # out[10,58]
            nn.MaxPool1d(4),  # out[20,14]
            nn.Tanh(),
            nn.Conv1d(10, 10, 4),  # out[10,11]
            nn.MaxPool1d(2),  # out[10,5]
            nn.Tanh(),
            nn.Flatten(0, -1),  # out[50]
            nn.Linear(50, 25),
            nn.Tanh(),
            nn.Linear(25, 10),
            nn.Tanh(),
            nn.Linear(10, 5),
            nn.Tanh(),
            nn.Linear(5, 4),
        )
        self.af1 = nn.Softmax()
        self.af2 = nn.Sigmoid()

    def forward(self, x):
        for layer in self.sequence:
            x = layer.forward(x)
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
