import torch
import torch.nn as nn


class NormalNoize(nn.Module):
    def __init__(self, precision: float):
        super(NormalNoize, self).__init__()
        self.register_buffer('precision', torch.tensor(precision))

    def forward(self, x):
        std = self.precision * torch.abs(x)
        return x + torch.normal(0, std / 3)
