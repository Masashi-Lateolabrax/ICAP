import numpy as np


class PheromoneField:
    def __init__(
            self,
            evaporation_speed: float,
            diffusion_speed: float,
            near: float = 0,
    ):
        # Structure of the ndarray in self.points: [x_pos, y_pos, value, evaporated_pheromone, spread_size]
        # x_pos: Position along the x-axis
        # y_pos: Position along the y-axis
        # value: Amount of liquid pheromone at the position
        # evaporated_pheromone: Amount of evaporated pheromone at the position
        # spread_size: Size of spread pheromone
        self.points: list[np.ndarray] = []

        self.evaporation_speed = evaporation_speed
        self.diffusion_speed = diffusion_speed
        self.near = near

    def add_liquid(self, x_pos: float, y_pos: float, value: float):
        position = np.array([x_pos, y_pos])

        if len(self.points) > 0:
            points = np.array(self.points)
            distance = np.linalg.norm(position - points[:, 0:2], axis=1)
            neighbor = points[distance < self.near, :]
            if neighbor.shape[0] > 0:
                neighbor[:, 2] += value / neighbor.shape[0]
                return

        self.points.append(np.array([
            x_pos,
            y_pos,
            value,
            0.0,
            0.0
        ]))

    def get_all_liquid(self):
        return sum(map(lambda x: x[2], self.points))

    def get_all_gas(self):
        return sum(map(lambda x: x[3], self.points))

    def get_gas(self, x_pos: float, y_pos: float) -> float:
        if len(self.points) == 0:
            return 0

        points = np.array(self.points)

        height = points[:, 3]
        size = points[:, 4]
        valid_points = np.logical_and(height > 0, size > 0)

        position = np.array([x_pos, y_pos])
        distance = np.linalg.norm(points[valid_points, 0:2] - position, axis=1)

        return np.sum(
            height[valid_points] * np.exp(
                -((distance ** 2) / (2 * (size[valid_points] ** 2)))
            )
        )

    def _del_empty_points(self):
        idx = 0
        while idx < len(self.points):
            if self.points[idx][2] <= 0 and self.points[idx][3] <= 0:
                self.points.pop(idx)
            else:
                idx += 1

    def update(self, dt: float, iteration: int = 1):
        self._del_empty_points()

        if len(self.points) == 0:
            return
        dt = dt / iteration

        points = np.array(self.points)
        liquid = points[:, 2]
        height = points[:, 3]
        size = points[:, 4]

        for _ in range(iteration):
            evaporation = np.ones(liquid.shape[0]) * self.evaporation_speed
            evaporation = np.min([evaporation, liquid], axis=0)

            decrease_h = np.ones(height.shape[0]) * self.diffusion_speed
            decrease_h = np.min([decrease_h, height], axis=0)

            increase_s = decrease_h

            liquid[:] += (-evaporation) * dt
            height[:] += (evaporation - decrease_h) * dt
            size[:] += (increase_s) * dt
