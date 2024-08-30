import colorsys

import mujoco
import numpy as np

from mujoco_xml_generator.utils import DummyGeom

from libs.pheromone import PheromoneField, PheromoneCell
from libs.mujoco_utils.objects import BodyObject


class PheromoneFieldWithDummies:
    def __init__(
            self,
            pheromone_field: PheromoneField,
            cell_size_for_mujoco,
            create_dummies: bool
    ):
        self._pheromone = pheromone_field
        pheromone_field_size = self._pheromone.get_field_size()

        self._cell_size_for_mujoco = cell_size_for_mujoco

        self._panels = []
        if create_dummies:
            for xi in range(pheromone_field_size[0]):
                x = (xi - (pheromone_field_size[0] - 1) * 0.5) * cell_size_for_mujoco
                for yi in range(pheromone_field_size[1]):
                    y = (yi - (pheromone_field_size[1] - 1) * 0.5) * cell_size_for_mujoco
                    create_dummies = DummyGeom(mujoco.mjtGeom.mjGEOM_PLANE)
                    create_dummies.set_size([cell_size_for_mujoco * 0.5, cell_size_for_mujoco * 0.5, 1])
                    create_dummies.set_pos([x, y, 0.001])
                    self._panels.append((xi, yi, create_dummies))

        self._offset = np.array(pheromone_field_size, dtype=np.float32) * self._cell_size_for_mujoco * 0.5

    def get_dummy_panels(self):
        return list(map(lambda x: x[2], self._panels))

    def _mujoco_pos_to_pheromone_field_pos(self, m_pos: np.ndarray) -> np.ndarray:
        m_pos[1] *= -1
        return (m_pos + self._offset) / self._cell_size_for_mujoco

    def get_sv(self):
        return self._pheromone.get_sv()

    def get_cell(self, xi: int, yi: int) -> PheromoneCell:
        return self._pheromone.get_cell(xi, yi)

    def get_all_liquid(self):
        return self._pheromone.get_all_liquid()

    def add_liquid(self, body: BodyObject, amount):
        m_pos = body.get_body().xpos[0:2]
        p_pos = self._mujoco_pos_to_pheromone_field_pos(m_pos)
        self._pheromone.add_liquid(p_pos[0], p_pos[1], amount)

    def get_all_gas(self):
        return self._pheromone.get_all_gas()

    def get_gas(self, body: BodyObject):
        m_pos = body.get_body().xpos[0:2]
        p_pos = self._mujoco_pos_to_pheromone_field_pos(m_pos)
        return self._pheromone.get_gas(p_pos[0], p_pos[1])

    def update(self, timestep, iteration, dummies=True):
        self._pheromone.update(timestep, iteration)
        sv = self._pheromone.get_sv()
        if dummies:
            for xi, yi, p in self._panels:
                _, gas_cell = self._pheromone.get_cell(xi, yi)
                pheromone = np.clip(gas_cell[0, 0] / sv, 0, 1)
                c = colorsys.hsv_to_rgb(0.66 * (1.0 - pheromone), 1.0, 1.0)
                p.set_rgba((c[0], c[1], c[2], 1.0))
