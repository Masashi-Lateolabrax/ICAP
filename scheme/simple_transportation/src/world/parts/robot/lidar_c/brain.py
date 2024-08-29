import torch
from torch import nn

from lib.analizer.neural_network import Debugger

from ....settings import Settings


class Brain(nn.Module):
    def __init__(self, debug=False):
        super(Brain, self).__init__()

        self.debugger = Debugger(debug)

        input_dim = Settings.Environment.Robot.Sensor.Lidar.DownScale.DIM_OUTPUT

        self.sequence1 = nn.Sequential(
            nn.Flatten(0, -1),
            self.debugger.create_investigator("s1l0"),

            nn.Linear(input_dim, 12),
            nn.Tanh(),
            self.debugger.create_investigator("s1l1"),

            nn.Linear(12, 6),
            nn.Tanh(),
            self.debugger.create_investigator("s1l2"),

            nn.Linear(6, 3),
            nn.Softmax(0),
            self.debugger.create_investigator("s1l3"),

        )
        self.sequence2 = nn.Sequential(
            self.debugger.create_investigator("s2l0"),

            nn.Linear(4, 5),
            nn.Tanh(),
            self.debugger.create_investigator("s2l1"),

            nn.Linear(5, 5),
            nn.Tanh(),
            self.debugger.create_investigator("s2l2"),

            nn.Linear(5, 3),
            nn.Sigmoid(),
            self.debugger.create_investigator("s2l3"),
        )

    def forward(self, sight: torch.Tensor, pheromone: torch.Tensor) -> torch.Tensor:
        x1 = self.sequence1.forward(sight)
        x2 = torch.concatenate([x1, pheromone])
        x3 = self.sequence2.forward(x2)
        return x3

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
