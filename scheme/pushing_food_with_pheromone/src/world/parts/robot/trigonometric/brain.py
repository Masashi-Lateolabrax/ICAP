import torch
from torch import nn

from libs.torch_utils import Debugger, NormalNoize


class MyGRU(nn.Module):
    def __init__(
            self, input_size, hidden_size, bias=True, device=None, dtype=None
    ):
        super(MyGRU, self).__init__()

        self.gru = nn.GRUCell(input_size, hidden_size, bias, device, dtype)
        self.noise = NormalNoize(0.01)
        self.hidden: torch.Tensor = None

    def forward(self, input_):
        y = self.gru.forward(input_, self.hidden)
        # y = self.noise.forward(y)
        self.hidden = y.detach()
        return y


class Brain(nn.Module):
    def __init__(self, debug=False):
        super(Brain, self).__init__()

        self.debugger = Debugger(debug)

        self.seq = nn.Sequential(
            self.debugger.create_investigator("l0"),

            NormalNoize(0.01),
            self.debugger.create_investigator("l0n"),

            nn.Linear(9, 8),
            nn.Tanh(),
            self.debugger.create_investigator("l1"),

            # NormalNoize(0.01),
            # self.debugger.create_investigator("l1n"),

            nn.Linear(8, 3),
            nn.Sigmoid(),
            self.debugger.create_investigator("l2"),

            # NormalNoize(0.001),
            # self.debugger.create_investigator("l2n"),
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
