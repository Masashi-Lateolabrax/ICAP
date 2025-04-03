import colorsys

import mujoco
import numpy as np

from mujoco_xml_generator.utils import DummyGeom

from libs.pheromone.gaussian import PheromoneField
from libs.mujoco_utils.objects import BodyObject


class PheromoneFieldWithDummies2:
    def __init__(
            self,
            pheromone_field: PheromoneField,
            nx, ny,
            cell_size_for_mujoco,
            create_dummies: bool
    ):
        self._pheromone = pheromone_field
        pheromone_field_size = (nx, ny)

        self._panels: list[DummyGeom] = []
        if create_dummies:
            for xi in range(pheromone_field_size[0]):
                x = (xi - (pheromone_field_size[0] - 1) * 0.5) * cell_size_for_mujoco
                for yi in range(pheromone_field_size[1]):
                    y = ((pheromone_field_size[1] - 1) * 0.5 - yi) * cell_size_for_mujoco
                    obj = DummyGeom(mujoco.mjtGeom.mjGEOM_PLANE)
                    obj.set_size([cell_size_for_mujoco * 0.5, cell_size_for_mujoco * 0.5, 1])
                    obj.set_pos([x, y, 0.001])
                    self._panels.append(obj)

    def get_dummy_panels(self):
        return self._panels

    def add_liquid(self, body: BodyObject, amount):
        pos = body.get_body().xpos[0:2]
        self._pheromone.add_liquid(pos[0], pos[1], amount)

    def get_gas(self, body: BodyObject):
        pos = body.get_body().xpos[0:2]
        return self._pheromone.get_gas(pos[0], pos[1])

    def update(self, timestep, iteration, max_, dummies=True):
        self._pheromone.update(timestep, iteration)
        if dummies:
            for obj in self._panels:
                p = np.clip(
                    self._pheromone.get_gas(obj.pos[0], obj.pos[1]) / max_,
                    0, 1
                )
                c = colorsys.hsv_to_rgb(0.66 * (1.0 - p), 1.0, 1.0)
                obj.set_rgba((c[0], c[1], c[2], 1.0))
