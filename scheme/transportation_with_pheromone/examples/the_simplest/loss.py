import numpy as np

from prelude import *


class Loss(framework.interfaceis.Loss):
    def __init__(self):
        self.sigma_nest_and_food = 1
        self.sigma_robot_and_food = 1

        self.gain_nest_and_food = 1
        self.gain_robot_and_food = 1

    def calc_loss(self, nest_pos: np.ndarray, robot_pos: np.ndarray, food_pos: np.ndarray) -> float:
        dist_r = np.linalg.norm(robot_pos[:, :2] - food_pos, axis=1)
        loss_r = np.average(np.exp(-dist_r / self.sigma_robot_and_food))

        dist_n = np.linalg.norm(nest_pos - food_pos, axis=1)
        loss_n = np.average(np.exp(-dist_n / self.sigma_nest_and_food))

        return self.gain_robot_and_food * loss_r + self.gain_nest_and_food * loss_n
