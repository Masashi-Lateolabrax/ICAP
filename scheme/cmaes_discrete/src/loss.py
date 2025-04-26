import numpy as np

import framework
from framework.simulator.objects.robot import ROBOT_SIZE
from framework.simulator.objects.food import FOOD_SIZE
from framework.simulator.objects.nest import NEST_SIZE


def _calc_loss_sigma(point, value):
    return -(point ** 2) / np.log(value)


class Loss(framework.interfaces.Loss):
    def __init__(self):
        self.offset_nest_and_food = NEST_SIZE + FOOD_SIZE
        self.sigma_nest_and_food = _calc_loss_sigma(4, 0.01)
        self.GAIN_NEST_AND_FOOD = 1

        self.offset_robot_and_food = ROBOT_SIZE + FOOD_SIZE
        self.sigma_robot_and_food = _calc_loss_sigma(1, 0.3)
        self.GAIN_ROBOT_AND_FOOD = 0.01

    def calc_r_loss(self, robot_pos: np.ndarray, food_pos: np.ndarray) -> float:
        distance = np.max(
            np.linalg.norm(robot_pos[:, :2] - food_pos, axis=1) - self.offset_robot_and_food,
            0
        )
        loss = -np.average(np.exp(-(distance ** 2) / self.sigma_robot_and_food))
        return self.GAIN_ROBOT_AND_FOOD * loss

    def calc_n_loss(self, nest_pos: np.ndarray, food_pos: np.ndarray) -> float:
        distance = np.max(
            np.linalg.norm(food_pos - nest_pos, axis=1) - self.offset_nest_and_food,
            0
        )
        loss = -np.average(np.exp(-(distance ** 2) / self.sigma_nest_and_food))
        return self.GAIN_NEST_AND_FOOD * loss

    def calc_loss(self, nest_pos: np.ndarray, robot_pos: np.ndarray, food_pos: np.ndarray) -> float:
        loss_r = self.calc_r_loss(robot_pos, food_pos)
        loss_n = self.calc_n_loss(nest_pos, food_pos)
        return loss_r + loss_n

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
