import libs.optimizer as opt

from .neural_network import NeuralNetwork
from .task import Task


class TaskGenerator(opt.TaskGenerator):
    @staticmethod
    def get_dim():
        return NeuralNetwork().num_dim()

    def generate(self, para, debug=False) -> Task:
        return Task(para)
