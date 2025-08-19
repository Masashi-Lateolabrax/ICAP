from abc import ABC

import numpy as np

from ..prelude import *
from ..pheromone import PheromoneFieldCell
from .basic_environment import BasicEnvironment


class BasicSimulator(BasicEnvironment, ABC):
    def __init__(self, settings, render: bool = False):
        super().__init__(settings, render)

        self.nest_site = self.data.site(self.nest_spec.name)
        self.robot_values = [
            RobotValues(settings.Robot.DISTANCE_BETWEEN_WHEELS, settings.Robot.MAX_SPEED, self.data, s)
            for s in self.robot_specs
        ]
        self.food_values = [FoodValues(self.data, s) for s in self.food_specs]
        self._pheromone_cells: list[PheromoneFieldCell] = [s.get_cell(self.model) for s in self.pheromone_cell_specs]

    def add_pheromone(self, x: float, y: float, value: float):
        if len(self._pheromone_cells) == 0:
            return

        lu_cell = self._pheromone_cells[0]
        rd_cell = self._pheromone_cells[-1]
        pos = np.array([x, y])
        rpos = (pos - lu_cell.pos[:2]) / (rd_cell.pos[:2] - lu_cell.pos[:2])
        index_x = rd_cell.index_x * rpos[0]
        index_y = rd_cell.index_y * rpos[1]
        index_x = np.clip(index_x, 0, rd_cell.index_x)
        index_y = np.clip(index_y, 0, rd_cell.index_y)

        cell = self._pheromone_cells[int(index_x * rd_cell.index_y + index_y)]
        cell.add_value += value

    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        if not self._do_render:
            return

        if self._pheromone_field:
            color_max = 1.0
            pheromone: np.ndarray = self._pheromone_field.get_gas_all()
            for cell in self._pheromone_cells:
                pheromone_value = float(pheromone[cell.index_y, cell.index_x])
                rgba: tuple[float, float, float] = (pheromone_value / color_max, 0.0, 1 - pheromone_value / color_max)
                cell.set_color(*rgba, 0.5)

        super().render(img_buf, pos, lookat)
