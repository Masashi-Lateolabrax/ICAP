from abc import ABC

from ..prelude import *
from .basic_environment import BasicEnvironment


class BasicSimulator(BasicEnvironment, ABC):
    def __init__(self, settings, render: bool = False):
        super().__init__(settings, render)

        self.robot_values = [
            RobotValues(settings.Robot.DISTANCE_BETWEEN_WHEELS, settings.Robot.MAX_SPEED, self.data, s)
            for s in self.robot_specs
        ]
        self.food_values = [FoodValues(self.data, s) for s in self.food_specs]
