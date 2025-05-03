import abc

import numpy as np


class Loss(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calc_elements(
            self, para: np.ndarray, nest_pos: np.ndarray, robot_pos: np.ndarray, food_pos: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def num_elements():
        raise NotImplementedError

    def calc_loss(
            self, para: np.ndarray, nest_pos: np.ndarray, robot_pos: np.ndarray, food_pos: np.ndarray
    ) -> float:
        elements = self.calc_elements(para, nest_pos, robot_pos, food_pos)
        return np.sum(elements)

    def __call__(
            self, para: np.ndarray, nest_pos: np.ndarray, robot_pos: np.ndarray, food_pos: np.ndarray
    ) -> float:
        return self.calc_loss(para, nest_pos, robot_pos, food_pos)
