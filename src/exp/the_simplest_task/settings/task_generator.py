import lib.optimizer as opt

from .brain import NeuralNetwork
from .task import Task
from .hyper_parameters import HyperParameters


class TaskGenerator(opt.TaskGenerator):
    def __init__(self, bot_pos: tuple[float, float, float] | None = None):
        if bot_pos is None:
            self.bot_pos = HyperParameters.Environment.BOT_POS
        else:
            self.bot_pos = [bot_pos]

    @staticmethod
    def get_dim():
        return NeuralNetwork().num_dim()

    def generate(self, para, debug=False) -> Task:
        brain = NeuralNetwork(debug)
        brain.load_para(para)
        return Task(self.bot_pos, brain)
