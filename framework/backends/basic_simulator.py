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
        self.pheromone_cells: list[PheromoneFieldCell] = [s.get_cell(self.model) for s in self.pheromone_cell_specs]

    def add_pheromone(self, x: float, y: float, value: float):
        if len(self.pheromone_cells) == 0:
            return

        lu_cell = self.pheromone_cells[0]
        rd_cell = self.pheromone_cells[-1]
        pos = np.array([x, y])
        rpos = (pos - lu_cell.pos[:2]) / (rd_cell.pos[:2] - lu_cell.pos[:2])
        index_x = rd_cell.index_x * rpos[0]
        index_y = rd_cell.index_y * rpos[1]
        index_x = np.clip(index_x, 0, rd_cell.index_x)
        index_y = np.clip(index_y, 0, rd_cell.index_y)

        cell = self.pheromone_cells[int(index_x * rd_cell.index_y + index_y)]
        cell.add_value += value
