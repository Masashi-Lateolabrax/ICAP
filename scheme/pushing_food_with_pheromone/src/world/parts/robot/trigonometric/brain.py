import torch
from torch import nn

from libs.torch_utils import Debugger, NormalNoize


class _UnsqueezeLayer(nn.Module):
    def __init__(self, dim):
        super(_UnsqueezeLayer, self).__init__()
        self.requires_grad_(False)
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(self.dim)


class _IndexFilter(nn.Module):
    def __init__(self, i):
        super(_IndexFilter, self).__init__()
        self.requires_grad_(False)
        self.i = i

    def forward(self, x):
        x = x[self.i]
        return x


class Brain(nn.Module):
    def __init__(self, debug=False):
        super(Brain, self).__init__()

        self.debugger = Debugger(debug)

        self.seq = nn.Sequential(
            self.debugger.create_investigator("l0"),

            NormalNoize(0.01),
            self.debugger.create_investigator("l0n"),

            _UnsqueezeLayer(0),
            nn.RNN(9, 20),
            _IndexFilter(0),
            _IndexFilter(0),
            self.debugger.create_investigator("l1"),
            NormalNoize(0.01),
            self.debugger.create_investigator("l1n"),

            nn.Linear(20, 3),
            nn.Sigmoid(),
            self.debugger.create_investigator("l2"),
        )

    def forward(
            self,
            sight: torch.Tensor,
            nest: torch.Tensor,
            pheromone: torch.Tensor,
            speed: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.concat([sight, nest, pheromone, speed])
        x = self.seq.forward(x)
        return x

    def load_para(self, para):
        with torch.no_grad():
            idx = 0
            for p in self.parameters():
                if not p.requires_grad:
                    continue
                n = p.numel()
                p.data.copy_(torch.tensor(para[idx:idx + n]).view(p.size()))
                idx += n
        return idx

    def num_dim(self):
        dim = 0
        for p in self.parameters():
            if not p.requires_grad:
                continue
            dim += p.numel()
        return dim
