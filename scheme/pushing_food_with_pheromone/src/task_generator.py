import random

import libs.optimizer as opt

from .settings import Settings
from .task import Task
from .world import World


class TaskGenerator(opt.TaskGenerator):
    def __init__(self, sigma, panel: bool):
        self.panel = panel
        self.bot_pos = Settings.Task.Robot.POSITIONS(sigma)
        self.food_pos = Settings.Task.Food.POSITIONS()

    @staticmethod
    def get_dim():
        return World.get_dim()

    def generate(self, para, debug=False) -> Task:
        return Task(para, self.bot_pos, self.food_pos, self.panel, debug)
