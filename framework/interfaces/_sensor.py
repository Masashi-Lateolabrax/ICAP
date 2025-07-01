import abc
from abc import ABC

import numpy as np


class SensorInterface(ABC):
    @abc.abstractmethod
    def get(self) -> np.ndarray:
        raise NotImplementedError
