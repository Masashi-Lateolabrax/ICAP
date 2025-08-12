import abc

import numpy as np


class SimulatorBackend(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self):
        raise NotImplementedError

    @abc.abstractmethod
    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_scores(self) -> list[float]:
        raise NotImplementedError

    @abc.abstractmethod
    def calc_total_score(self) -> float:
        raise NotImplementedError
