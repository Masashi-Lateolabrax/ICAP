import abc

import numpy as np


class SimulatorBackend(metaclass=abc.ABCMeta):
    def step(self):
        raise NotImplementedError

    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def scores(self) -> list[float]:
        raise NotImplementedError

    def total_score(self) -> float:
        raise NotImplementedError
