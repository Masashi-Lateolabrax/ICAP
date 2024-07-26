import mujoco
import numpy as np

from mujoco_xml_generator.utils import DummyGeom

from lib.optimizer import MjcTaskInterface
from lib.pheromone import PheromoneField2
from lib.mujoco_utils import PheromoneFieldWithDummies

from .settings import Settings

from ._xml_setting import gen_xml


class AnalysisEnvironment(MjcTaskInterface):
    def __init__(self, liquid):
        self.time = 0

        self.center_liquid = liquid
        self.dif_liquid = 0

        xml = gen_xml()
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)

        self.pheromone = PheromoneFieldWithDummies(
            PheromoneField2(
                nx=Settings.World.NUM_CELL[0],
                ny=Settings.World.NUM_CELL[1],
                d=Settings.Pheromone.CELL_SIZE_FOR_CALCULATION,
                sv=Settings.Pheromone.SATURATION_VAPOR,
                evaporate=Settings.Pheromone.EVAPORATION,
                diffusion=Settings.Pheromone.DIFFUSION,
                decrease=Settings.Pheromone.DECREASE
            ),
            Settings.Pheromone.CELL_SIZE_FOR_MUJOCO,
            True
        )

        self.center_cell, _ = self.pheromone.get_cell(
            Settings.World.CENTER_INDEX[0], Settings.World.CENTER_INDEX[1]
        )

    def get_model(self) -> mujoco.MjModel:
        return self.m

    def get_data(self) -> mujoco.MjData:
        return self.d

    def calc_step(self) -> float:
        self.time += Settings.Simulation.TIMESTEP
        self.dif_liquid = 0

        for self.time in range(Settings.Simulation.PHEROMONE_ITER):
            if self.time < Settings.Task.SECRETION_PERIOD:
                self.center_cell[0, 0] = self.center_liquid
            else:
                self.center_cell[0, 0] = 0

            prev_liquid = self.pheromone.get_all_liquid()
            self.pheromone.update(Settings.Simulation.PHEROMONE_TIMESTEP, 1, dummies=True)
            current_liquid = self.pheromone.get_all_liquid()

            self.dif_liquid += np.sum(current_liquid - prev_liquid)

        return 0.0

    def run(self) -> float:
        total_step = int(Settings.Task.EPISODE_LENGTH / Settings.Simulation.TIMESTEP + 0.5)
        for _ in range(total_step):
            self.calc_step()
        return 0.0

    def get_dummies(self) -> list[DummyGeom]:
        return self.pheromone.get_dummy_panels()
