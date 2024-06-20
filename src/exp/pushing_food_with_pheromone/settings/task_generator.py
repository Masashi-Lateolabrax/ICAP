import lib.optimizer as opt
from .brain import NeuralNetwork
from .task import Task


class TaskGenerator(opt.TaskGenerator):
    @staticmethod
    def get_dim():
        return NeuralNetwork().num_dim()

    def generate(self, para, debug=False) -> Task:
        brain = NeuralNetwork(debug)
        brain.load_para(para)
        return Task(brain)
