import lib.optimizer as opt

from .brain import NeuralNetwork
from .task import Task
from .hyper_parameters import HyperParameters


class TaskGenerator(opt.TaskGenerator):
    def __init__(self):
        pass

    @staticmethod
    def get_dim():
        return NeuralNetwork().num_dim()

    def generate(self, para, debug=False) -> Task:
        brain = NeuralNetwork(debug)
        brain.load_para(para)
        return Task(HyperParameters.Environment.BOT_POS, brain)
