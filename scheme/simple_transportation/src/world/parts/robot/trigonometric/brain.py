import torch
from torch import nn

from libs.torch_utils import Debugger

from .....settings import Settings


class Brain(nn.Module):
    def __init__(self, debug=False):
        super(Brain, self).__init__()

        self.debugger = Debugger(debug)

        self.sequence1 = nn.Sequential(
            nn.Flatten(0, -1),
            self.debugger.create_investigator("s1l0"),

            nn.Linear(6, 12),
            nn.Tanh(),
            self.debugger.create_investigator("s1l1"),

            nn.Linear(12, 6),
            nn.Tanh(),
            self.debugger.create_investigator("s1l2"),

            nn.Linear(6, 3),
            nn.Sigmoid(),
            self.debugger.create_investigator("s1l3"),

        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.sequence1.forward(input_)

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
