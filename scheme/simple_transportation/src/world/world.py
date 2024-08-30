from dataclasses import dataclass
from array import array

import torch

from ..settings import Settings
from .parts import Environment, Food
from .parts.robot import trigonometric as tr


@dataclass
class _Buffer:
    sight = torch.zeros((6, 1))
    direction: torch.Tensor = torch.zeros((3, 1))


class World:
    def __init__(self, para: array):
        self._buf = _Buffer()

        self.env = Environment()

        factory = tr.RobotFactory(True, True)
        self.bots: list[tr.RobotTR] = [
            factory.manufacture(self.env, i, para) for i in range(len(Settings.Task.Robot.POSITIONS))
        ]

        self.food = [
            Food(self.env, i) for i in range(len(Settings.Task.Food.POSITIONS))
        ]

    def calc_step(self):
        self.env.calc_step()
        for bot in self.bots:
            bot.act(self.env)