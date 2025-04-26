import torch

from libs.optimizer import Individual

import framework
from framework.simulator.objects.robot import RobotInput

from settings import Settings


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
            torch.nn.Linear(6, 4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 2),
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


class Brain(framework.interfaces.BrainInterface):
    def __init__(self, settings: Settings, nn: NeuralNetwork):
        self.settings = settings

        self.timer = Timer(settings.Robot.THINK_INTERVAL / settings.Simulation.TIME_STEP)
        self.state = framework.BrainJudgement.STOP

        self.neural_network = nn
        self.output = torch.zeros(2, dtype=torch.float32, requires_grad=False)
        self.output_is_updated = False

    def think(self, input_: RobotInput) -> torch.Tensor:
        if self.timer.tick():
            self.output[:] = self.neural_network(input_)
            self.output_is_updated = True
        return self.output


class BrainBuilder(framework.interfaces.BrainBuilder):
    @staticmethod
    def get_dim() -> int:
        return NeuralNetwork().num_dim()

    def __init__(self, settings):
        self.settings = settings
        self.buf_net: NeuralNetwork = NeuralNetwork()

    def build(self, para: Individual) -> framework.interfaces.BrainInterface:
        self.buf_net.set_para(para)
        return Brain(self.settings, self.buf_net)
