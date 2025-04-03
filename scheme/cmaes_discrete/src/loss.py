import numpy as np

import framework


def _calc_loss_sigma(point, value):
    return np.sqrt(-(point ** 2) / np.log(value))


class Loss(framework.interfaceis.Loss):
    def __init__(self):
        self.sigma_nest_and_food = 1
        self.sigma_robot_and_food = _calc_loss_sigma(3, 0.1)

        self.gain_nest_and_food = 0.001
        self.gain_robot_and_food = _calc_loss_sigma(1, 0.5)

    def calc_loss(self, nest_pos: np.ndarray, robot_pos: np.ndarray, food_pos: np.ndarray) -> float:
        dist_r = np.linalg.norm(robot_pos[:, :2] - food_pos, axis=1)
        loss_r = np.average(np.exp(-dist_r / self.sigma_robot_and_food))

        dist_n = np.linalg.norm(nest_pos - food_pos, axis=1)
        loss_n = np.average(np.exp(-dist_n / self.sigma_nest_and_food))

        return self.gain_robot_and_food * loss_r + self.gain_nest_and_food * loss_n

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
