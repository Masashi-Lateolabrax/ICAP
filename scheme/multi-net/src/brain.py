import torch

import framework
from framework.simulator.objects import RobotInput
from libs.optimizer import Individual

from neural_net import NeuralNetwork

NUM_SUBNET = 3


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


class Brain(framework.interfaces.BrainInterface):
    def __init__(self, settings: framework.Settings, nn: NeuralNetwork):
        self.settings = settings

        self.timer = Timer(settings.Robot.THINK_INTERVAL / settings.Simulation.TIME_STEP)
        self.state = framework.BrainJudgement.STOP

        self.neural_network = nn
        self.output = torch.zeros(2, dtype=torch.float32, requires_grad=False)

    def think(self, input_: RobotInput) -> torch.Tensor:
        if self.timer.tick():
            x = input_.get()[2:6]
            if torch.any(torch.isnan(x) | torch.isinf(x)):
                print("The input tensor for robots contains invalid values (NaN or Inf).")
            self.output[:] = self.neural_network(x)
        return self.output


class BrainBuilder(framework.interfaces.BrainBuilder):
    @staticmethod
    def get_dim() -> int:
        return NeuralNetwork(NUM_SUBNET).num_dim()

    def __init__(self, settings):
        self.settings = settings
        self.buf_net: NeuralNetwork = NeuralNetwork(NUM_SUBNET)

    def build(self, para: Individual) -> framework.interfaces.BrainInterface:
        self.buf_net.set_para(para)
        return Brain(self.settings, self.buf_net)
