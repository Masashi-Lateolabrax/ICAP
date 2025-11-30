import abc

import numpy as np

from framework.prelude import *

from .debug_data import DebugData


class SimulatorForDebugInterface(abc.ABC):
    @abc.abstractmethod
    def __init__(self, settings: Settings, parameters: Individual, render: bool = False):
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        raise NotImplementedError()

    @abc.abstractmethod
    def debug_data(self) -> list[DebugData]:
        raise NotImplementedError()

    @abc.abstractmethod
    def loss(self) -> float:
        raise NotImplementedError()
