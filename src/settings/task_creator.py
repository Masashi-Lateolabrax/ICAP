import pickle

import src.optimizer as opt
from .brain import NeuralNetwork
from .task import Task


class TaskGenerator(opt.TaskGenerator):
    TIMESTEP = 0.01
    NUM_BOT = 1
    NUM_GOAL = 1
    TRY_COUNT = 3

    def __init__(self):
        import random

        self._brain = NeuralNetwork()

        self.bot_pos = [
            [(0, -5, 360 * random.random())] for _ in range(TaskGenerator.TRY_COUNT)
        ]
        self.goal_pos = [
            [(0, 5, 0)] for _ in range(TaskGenerator.TRY_COUNT)
        ]

    def save(self) -> bytes:
        return pickle.dumps(self)

    def load(self, byte_data: bytes) -> int:
        new_instance = pickle.loads(byte_data)
        self.__dict__.update(new_instance.__dict__)
        return len(byte_data)

    def generate(self, para) -> Task:
        self._brain.load_para(para)
        return Task(self.bot_pos, self.goal_pos, self._brain, TaskGenerator.TIMESTEP)
