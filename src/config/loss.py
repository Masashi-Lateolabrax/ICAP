import numpy as np

from framework.prelude import *


class Loss:
    def __init__(self, settings: Settings, robot_positions, food_positions, nest_position):
        self.settings = settings
        robot_pos_array = np.array(robot_positions)
        food_pos_array = np.array(food_positions)
        nest_pos_array = nest_position[0:2]
        self.r_loss = self._calc_r_loss(robot_pos_array, food_pos_array)
        self.n_loss = self._calc_n_loss(nest_pos_array, food_pos_array)

    def _calc_r_loss(self, robot_pos: np.ndarray, food_pos: np.ndarray) -> float:
        subs = (robot_pos[:, None, :] - food_pos[None, :, :]).reshape(-1, 2)
        distance = np.clip(
            np.linalg.norm(subs, axis=1) - self.settings.Loss.OFFSET_ROBOT_AND_FOOD,
            a_min=0,
            a_max=None
        )
        loss = -np.sum(np.exp(-(distance ** 2) / self.settings.Loss.SIGMA_ROBOT_AND_FOOD))
        return self.settings.Loss.GAIN_ROBOT_AND_FOOD * loss

    def _calc_n_loss(self, nest_pos: np.ndarray, food_pos: np.ndarray) -> float:
        subs = food_pos - nest_pos[None, :]
        distance = np.clip(
            np.linalg.norm(subs, axis=1) - self.settings.Loss.OFFSET_NEST_AND_FOOD,
            a_min=0,
            a_max=None
        )
        loss = -np.sum(np.exp(-(distance ** 2) / self.settings.Loss.SIGMA_NEST_AND_FOOD))
        return self.settings.Loss.GAIN_NEST_AND_FOOD * loss

    def as_float(self) -> float:
        return self.r_loss + self.n_loss
