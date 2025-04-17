import numpy as np

import framework
from framework.simulator.objects.robot import ROBOT_SIZE
from framework.simulator.objects.food import FOOD_SIZE
from framework.simulator.objects.nest import NEST_SIZE


def _calc_loss_sigma(point, value):
    return -(point ** 2) / np.log(value)


class Loss(framework.interfaces.Loss):
    def calc_elements(self, nest_pos: np.ndarray, robot_pos: np.ndarray, food_pos: np.ndarray) -> np.ndarray:
        return np.linalg.norm(robot_pos[:, :2] - food_pos, axis=1)

    @staticmethod
    def num_elements():
        return 1

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
