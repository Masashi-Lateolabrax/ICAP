import random

import libs.optimizer as opt

from .settings import Settings
from .task import Task
from .world import World


class TaskGenerator(opt.TaskGenerator):
    def __init__(self, sigma):
        self.bot_pos = [
            (p[0], p[1], 90 + sigma * 180 * (2 * random.random() - 1)) for p in Settings.Task.Robot.POSITIONS
        ]

    @staticmethod
    def get_dim():
        return World.get_dim()

    def generate(self, para, debug=False) -> Task:
        return Task(para, self.bot_pos, debug)
