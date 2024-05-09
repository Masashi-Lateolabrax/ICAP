import pickle

import src.optimizer as opt
from .utils.brain import NeuralNetwork
from .task import Task
from .hyper_parameters import HyperParameters


class TaskGenerator(opt.TaskGenerator):
    def __init__(self):
        import random

        self._brain = NeuralNetwork()

        self.bot_pos = [
            [(0, -5, 360 * random.random())] for _ in range(HyperParameters.TRY_COUNT)
        ]
        self.goal_pos = [
            [(0, 5, 0)] for _ in range(HyperParameters.TRY_COUNT)
        ]

    def get_dim(self):
        return self._brain.num_dim()

    def save(self) -> bytes:
        return pickle.dumps(self)

    def load(self, byte_data: bytes) -> int:
        new_instance = pickle.loads(byte_data)
        self.__dict__.update(new_instance.__dict__)
        return len(byte_data)

    def generate(self, para) -> Task:
        self._brain.load_para(para)
        return Task(
            self.bot_pos, self.goal_pos, self._brain,
        )
