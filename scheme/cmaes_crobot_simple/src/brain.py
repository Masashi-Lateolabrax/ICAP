import torch

import framework
from framework.simulator.objects import RobotInput
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
            torch.nn.Linear(6, 5),
            torch.nn.Tanhshrink(),
            torch.nn.Linear(5, 4),
            torch.nn.Tanhshrink(),
            torch.nn.Linear(4, 2),
            torch.nn.Tanh(),
        )

    def forward(self, x: framework.simulator.objects.RobotInput):
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
    def __init__(self, settings: framework.Settings, nn: NeuralNetwork):
        self.settings = settings

        self.timer = Timer(settings.Robot.THINK_INTERVAL / settings.Simulation.TIME_STEP)

        self.neural_network = nn

        self._output_buf = torch.zeros(2, dtype=torch.float32, requires_grad=False)

    def think(self, input_: RobotInput) -> torch.Tensor:
        if self.timer.tick():
            x = input_.get()
            if torch.any(torch.isnan(x) | torch.isinf(x)):
                print("The input tensor for robots contains invalid values (NaN or Inf).")
            output = self.neural_network(x)
            self._output_buf[:] = output[:2]

        return self._output_buf


class BrainBuilder(framework.interfaces.BrainBuilder):
    @staticmethod
    def get_dim() -> int:
        return NeuralNetwork().num_dim()

    def __init__(self, settings):
        self.settings = settings
        self.nn = NeuralNetwork()

    def build(self, para: Individual) -> framework.interfaces.BrainInterface:
        self.nn.set_para(para)
        return Brain(self.settings, self.nn)
