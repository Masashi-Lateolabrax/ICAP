from typing import Callable

import numpy as np

_BLUE = np.array([[0], [0], [255]])
_RED = np.array([[255], [0], [0]])
_TRANSITION = _RED - _BLUE


class TrigonoOmniSensorBuf:
    def __init__(self):
        self.result = np.zeros((2,))
        self.trigono_components = np.zeros(0)
        self.img_result = np.zeros((2, 3))

    def reallocate(self, n: int):
        if self.trigono_components.shape != (2, n):
            self.trigono_components = np.zeros((2, n))


class TrigonoOmniSensor:
    def __init__(self, gain: Callable[[np.ndarray], np.ndarray], buf: TrigonoOmniSensorBuf = None):
        """
        Parameters
        ----------
        gain: functional(distances: n-dim ndarray) -> n-dim ndarray, default lambda d: d .

        buf: TrigonoOmniSensorBuf, optional
        """

        self.gain = gain
        self._buf = TrigonoOmniSensorBuf() if buf is None else buf

    def get_buf(self):
        return self._buf

    def measure(self, position: np.ndarray, direction: np.ndarray, targets: np.ndarray):
        """
        Parameters
        ----------
        position: 2-dim ndarray

        direction: 2-dim ndarray

        targets: (n,2) shape ndarray

        Returns
        ------
        result: 2-dim ndarray
        """

        if position.shape != (2,):
            raise "Invalid argument. the shape of 'center' must be (2,)."
        elif direction.shape != (2,):
            raise "Invalid argument. the shape of 'direction' must be (2,)."
        elif targets.ndim != 2 or targets.shape[1] != 2:
            raise "Invalid argument. the shape of 'targets' must be (n,2)."

        if targets.shape[0] == 0:
            return np.array([0, 0])

        self._buf.reallocate(targets.shape[0])

        direction = np.array([
            [direction[1], -direction[0]],
            [direction[0], direction[1]],
        ])

        sub = targets - position
        distance = np.linalg.norm(sub, axis=1)
        sub /= distance
        np.dot(direction, sub.T * self.gain(distance), out=self._buf.trigono_components)
        np.dot(
            self.gain(distance),
            self._buf.trigono_components.T,
            out=self._buf.result
        )
        self._buf.result /= targets.shape[0]

        return self._buf.result

    def convert_to_img(self, result: np.ndarray):
        self._buf.img_result[:, :] = ((result + 1) * 0.5 * _TRANSITION + _BLUE).T
        return self._buf.img_result

    def measure_with_img(
            self, position: np.ndarray, direction: np.ndarray, targets: np.ndarray,
    ):
        """
        Parameters
        ----------
        position: 2-dim ndarray

        direction: 2-dim ndarray

        targets: (n,2) shape ndarray

        Returns
        ------
        result: (1,2) shape ndarray
        """

        result = self.measure(position, direction, targets)
        return self.convert_to_img(result)


def trigono_omni_sensor(
        center: np.ndarray, direction: np.ndarray, targets: np.ndarray, f=lambda d: d
) -> (float, float):
    if center.shape != (2,):
        raise "Invalid argument. the shape of 'center' must be (2,)."
    elif direction.shape != (2,):
        raise "Invalid argument. the shape of 'direction' must be (2,)."
    elif targets.ndim != 2 or targets.shape[1] != 2:
        raise "Invalid argument. the shape of 'targets' must be (n,2)."

    if targets.shape[0] == 0:
        return 0, 0

    f = np.frompyfunc(f, 1, 1)

    direction = np.array([
        [direction[1], -direction[0]],
        [direction[0], direction[1]],
    ])

    sub = targets - center
    distance = np.linalg.norm(sub, axis=1)
    trigonal_components = np.dot(direction, sub.T / distance).T
    result = np.dot(
        f(distance),
        trigonal_components
    ) / targets.shape[0]

    return result[0], result[1]


def _test():
    center = np.array([0, 0])
    direction = np.array([3 / 5, 4 / 5])
    targets = np.array([
        [1, 1],
        [2, 3],
        [-1, 1]
    ])
    result = trigono_omni_sensor(center, direction, targets, lambda d: 1 / (0.001 * d + 1))
    print(result)


if __name__ == '__main__':
    _test()
