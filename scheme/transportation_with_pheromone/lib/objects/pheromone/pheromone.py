import colorsys

import numpy as np

from mujoco_xml_generator.utils import DummyGeom

from ._dynamic_numpy_array import DynamicNumpyArray as _DynamicNumpyArray


class Pheromone:
    def __init__(
            self, evaporation_speed: float, diffusion_speed: float, near: float = 0, dummy: list[DummyGeom] = None
    ):
        """
        Constructor of Pheromone class

        :param evaporation_speed: Speed of evaporation
        :param diffusion_speed: Speed of diffusion
        :param near: Distance to consider as a neighbor
        :param dummy: List of dummy geometries
        """

        self.evaporation_speed = evaporation_speed
        self.diffusion_speed = diffusion_speed
        self.near = near

        self.points = _DynamicNumpyArray(5)

        self._dummy = dummy if dummy is not None else []

    def _nearest_point(self, position: np.ndarray) -> tuple[int, float, np.ndarray] | None:
        if len(self.points) == 0:
            return None
        distance = np.linalg.norm(self.points.public[:, 0:2] - position, axis=1)

        idx = np.argmin(distance)
        d = distance[idx]
        point = self.points[idx]

        return idx, d, point

    def add_liquid(self, x_pos: float, y_pos: float, value: float):
        new_point = np.array([x_pos, y_pos, value, 0.0, 0.0])

        if len(self.points) == 0:
            self.points.insert(new_point)
            return

        _, d, p = self._nearest_point(new_point[0:2])
        if d < self.near:
            p[2] += value
            return

    def evaporation_model(self, point: np.ndarray, dt: float):
        evaporation = np.minimum(point[2], self.evaporation_speed * dt)
        point[2] -= evaporation
        point[3] += evaporation

    def diffusion_model(self, point: np.ndarray, dt: float):
        dh = np.minimum(point[3], self.diffusion_speed * dt)
        ds = self.diffusion_speed * dt
        point[3] -= dh
        point[4] += ds

    def get(self, x_pos: float, y_pos: float) -> float:
        if len(self.points) == 0:
            return 0

        distance = np.sum((self.points.public[:, 0:2] - np.array([x_pos, y_pos])) ** 2, axis=1)
        value = self.points[:, 3] * np.exp(-distance / self.points[:, 4])
        return np.sum(value)

    def _update_dummies(self):
        for d in self._dummy:
            pos = d.get_pos()
            value = self.get(pos[0], pos[1])
            pheromone = np.clip(value, 0, 1)
            c = colorsys.hsv_to_rgb(0.66 * (1.0 - pheromone), 1.0, 1.0)
            d.set_rgba((c[0], c[1], c[2], 1.0))

    def calc_step(self, dt):
        self.evaporation_model(self.points.public, dt)
        self.diffusion_model(self.points.public, dt)
        self._update_dummies()
