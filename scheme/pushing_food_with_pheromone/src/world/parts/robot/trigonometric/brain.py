import torch
from torch import nn

from libs.torch_utils import Debugger


class Brain(nn.Module):
    def __init__(self, debug=False):
        super(Brain, self).__init__()

        self.debugger = Debugger(debug)

        self.layer_0_to_2 = nn.Sequential(
            self.debugger.create_investigator("l0"),
            nn.RNN(10, 5),
            self.debugger.create_investigator("l1"),
        )

        self.layer_2_to_3 = nn.Sequential(
            nn.Linear(5, 5),
            nn.Tanh(),
            self.debugger.create_investigator("l2"),

            nn.Linear(5, 3),
            nn.Sigmoid(),
            self.debugger.create_investigator("l3"),
        )

    def forward(
            self,
            sight: torch.Tensor,
            nest: torch.Tensor,
            pheromone: torch.Tensor,
            speed: torch.Tensor,
            remaining_pheromone: float,
    ) -> torch.Tensor:
        x = torch.concat([sight, nest, pheromone, speed, remaining_pheromone])
        x = x.unsqueeze(0)
        x, _ = self.layer_0_to_2.forward(x)
        x = self.layer_2_to_3.forward(x[0])
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
