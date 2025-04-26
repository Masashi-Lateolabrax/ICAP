import numpy as np
from mujoco._structs import _MjDataSiteViews


class DirectionSensor:
    def __init__(self, r: float, target_site: _MjDataSiteViews):
        self.r = r
        self.target = target_site

    def get(self, direction: np.ndarray, center: np.ndarray) -> np.ndarray:
        if center.shape != (2,):
            raise ValueError("Invalid argument. the shape of 'center' must be (2,).")
        elif direction.shape != (2,):
            raise ValueError("Invalid argument. the shape of 'direction' must be (2,).")

        direction_matrix = np.array([
            [direction[1], -direction[0]],
            [direction[0], direction[1]],
        ])

        target_position = self.target.xpos[0:2]
        sub = target_position - center
        distance = np.linalg.norm(sub)
        k = 1 / max(distance, self.r)
        result = np.dot(direction_matrix, sub * k)

        return result
