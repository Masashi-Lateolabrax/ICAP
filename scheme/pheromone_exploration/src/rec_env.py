import colorsys
import warnings

import mujoco
import numpy as np

from mujoco_xml_generator import common as mjc_cmn
from mujoco_xml_generator.utils import DummyGeom

from mujoco_xml_generator import Generator, Option
from mujoco_xml_generator import Visual, visual
from mujoco_xml_generator import WorldBody, body

from libs.optimizer import MjcTaskInterface

from .settings import Settings
from .utils import convert_para


def gen_xml():
    resolution = Settings.Display.RESOLUTION
    xml = Generator().add_children([
        Option(
            timestep=Settings.Simulation.TIMESTEP,
            integrator=mjc_cmn.IntegratorType.IMPLICITFACT
        ),
        Visual().add_children([
            visual.Global(offwidth=resolution[0], offheight=resolution[1])
        ]),
        WorldBody().add_children([
            body.Geom(
                type_=mjc_cmn.GeomType.PLANE, pos=(0, 0, 0), size=(0, 0, 1)
            ),
        ]),
    ]).build()
    return xml


class RecEnv(MjcTaskInterface):
    def __init__(self, gas: np.ndarray, para):
        self._para = convert_para(para)
        self._gas = gas
        self._time = 0

        xml = gen_xml()
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)

        cell_size = Settings.Pheromone.CELL_SIZE_FOR_MUJOCO
        pheromone_field_size = Settings.Pheromone.NUM_CELL

        self._panels = []
        for xi in range(pheromone_field_size[0]):
            x = (xi - (pheromone_field_size[0] - 1) * 0.5) * cell_size
            for yi in range(pheromone_field_size[1]):
                y = (yi - (pheromone_field_size[1] - 1) * 0.5) * cell_size

                create_dummies = DummyGeom(mujoco.mjtGeom.mjGEOM_PLANE)
                create_dummies.set_size([cell_size * 0.5, cell_size * 0.5, 1])
                create_dummies.set_pos([x, y, 0.001])
                self._panels.append((xi, yi, create_dummies))

    def get_model(self) -> mujoco.MjModel:
        return self.m

    def get_data(self) -> mujoco.MjData:
        return self.d

    def calc_step(self) -> float:
        mujoco.mj_step(self.m, self.d)

        for xi, yi, dg in self._panels:
            value = self._gas[self._time, xi, yi]
            value = np.clip(value / self._para["sv"], 0, 1)
            c = colorsys.hsv_to_rgb(0.66 * (1.0 - value), 1.0, 1.0)
            dg.set_rgba((c[0], c[1], c[2], 1.0))

        self._time += 1
        if self._time >= self._gas.shape[0]:
            warnings.warn("Reset time")
            self._time = 0

        return 0.0

    def run(self) -> float:
        total_step = int(Settings.Simulation.EPISODE_LENGTH / Settings.Simulation.TIMESTEP + 0.5)
        for _ in range(total_step):
            self.calc_step()
        return 0.0

    def get_dummies(self) -> list[DummyGeom]:
        return list(map(lambda x: x[2], self._panels))
