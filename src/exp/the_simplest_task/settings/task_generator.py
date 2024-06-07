import lib.optimizer as opt

from .brain import NeuralNetwork
from .task import Task
from .hyper_parameters import HyperParameters


class TaskGenerator(opt.TaskGenerator):
    def __init__(self):
        import random

        p = HyperParameters.Environment.BOT_POS
        self.bot_pos = [
            (p[0], p[1], 360 * random.random()) for _ in range(HyperParameters.Simulator.TRY_COUNT)
        ]
        self.brain = NeuralNetwork()

    @staticmethod
    def get_dim():
        return NeuralNetwork().num_dim()

    def generate(self, para) -> Task:
        self.brain.load_para(para)
        return Task(self.bot_pos, self.brain)
