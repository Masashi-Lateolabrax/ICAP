import numpy as np

import framework
from framework.simulator.objects.robot import ROBOT_SIZE
from framework.simulator.objects.food import FOOD_SIZE
from framework.simulator.objects.nest import NEST_SIZE


def _calc_loss_sigma(point, value):
    return -(point ** 2) / np.log(value)


class Loss(framework.interfaces.Loss):
    offset_nest_and_food = NEST_SIZE + FOOD_SIZE
    sigma_nest_and_food = _calc_loss_sigma(4, 0.01)
    GAIN_NEST_AND_FOOD = 1

    offset_robot_and_food = ROBOT_SIZE + FOOD_SIZE
    sigma_robot_and_food = _calc_loss_sigma(1, 0.3)
    GAIN_ROBOT_AND_FOOD = 0.01

    def _calc_r_loss(self, robot_pos: np.ndarray, food_pos: np.ndarray) -> float:
        distance = np.max(
            np.linalg.norm(robot_pos[:, :2] - food_pos, axis=1) - self.offset_robot_and_food,
            0
        )
        loss = -np.average(np.exp(-(distance ** 2) / self.sigma_robot_and_food))
        return self.GAIN_ROBOT_AND_FOOD * loss

    def _calc_n_loss(self, nest_pos: np.ndarray, food_pos: np.ndarray) -> float:
        distance = np.max(
            np.linalg.norm(food_pos - nest_pos, axis=1) - self.offset_nest_and_food,
            0
        )
        loss = -np.average(np.exp(-(distance ** 2) / self.sigma_nest_and_food))
        return self.GAIN_NEST_AND_FOOD * loss

    def calc_elements(self, nest_pos: np.ndarray, robot_pos: np.ndarray, food_pos: np.ndarray) -> np.ndarray:
        return np.array([
            self._calc_r_loss(robot_pos, food_pos),
            self._calc_n_loss(nest_pos, food_pos)
        ])

    @staticmethod
    def num_elements():
        return 2
