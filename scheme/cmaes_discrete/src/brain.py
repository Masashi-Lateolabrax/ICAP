import random
import torch

import framework
from libs.optimizer import Individual


class Timer:
    def __init__(self, interval):
        self.interval = interval
        self.time = 0

    def tick(self):
        self.time += 1
        if self.time >= self.interval:
            self.time = 0
            return True
        return False


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.sequence = torch.nn.Sequential(
            torch.nn.Linear(6, 6),
            torch.nn.Tanh(),
            torch.nn.Linear(6, 6),
            torch.nn.GELU(),
            torch.nn.Linear(6, 5),
            torch.nn.Softmax(dim=0)
        )

    def forward(self, x: framework.simulator.objects.RobotInput):
        x = x.get()
        y = self.sequence(x)
        return y

    def set_para(self, para):
        i = 0
        for p in self.parameters():
            n = p.numel()
            p.data = torch.tensor(
                para[i:i + n], dtype=torch.float32
            ).reshape(p.data.shape)
            i += n

    def num_dim(self):
        res = 0
        for p in self.parameters():
            res += p.numel()
        return res


class Brain(framework.interfaceis.BrainInterface):
    def __init__(self, settings: framework.Settings, para: Individual):
        self.timer = Timer(settings.Robot.THINK_INTERVAL / settings.Simulation.TIME_STEP)
        self.state = framework.BrainJudgement.STOP

        self.neural_network = NeuralNetwork()
        self.neural_network.set_para(para)

    def think(self, input_: framework.simulator.objects.RobotInput) -> framework.BrainJudgement:
        r = random.uniform(0, 4 * 200)
        if 0 <= r < 5:
            self.state = framework.BrainJudgement(int(r))
        return self.state


class BrainBuilder(framework.interfaceis.BrainBuilder):
    @staticmethod
    def get_dim() -> int:
        return NeuralNetwork().num_dim()

    def __init__(self, settings):
        self.settings = settings

    def build(self, para: Individual) -> framework.interfaceis.BrainInterface:
        return Brain(self.settings, para)
