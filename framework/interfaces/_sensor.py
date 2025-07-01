import abc

import numpy as np


class SensorInterface(metaclass=abc.ABC):
    def get(self) -> np.ndarray:
        raise NotImplementedError
