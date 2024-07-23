import numpy as np
import mujoco

from lib.pheromone import PheromoneField2
from lib.mujoco_utils import PheromoneFieldWithDummies

from ..settings import Settings
from .._xml_setting import gen_xml


class World:
    def __init__(
            self,
            bot_pos: list[tuple[float, float, float]], safezone: list[tuple[float, float]], create_dummies: bool
    ):
        xml = gen_xml(
            Settings.Simulation.TIMESTEP, safezone, bot_pos
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.bot_pos = np.zeros((Settings.Task.NUM_ROBOT, 2))
        self.safezone_pos = np.array(safezone)

        self.pheromone = None
        if Settings.Environment.Pheromone.ENABLE_PHEROMONE:
            pheromone = PheromoneField2(
                Settings.Environment.Pheromone.PHEROMONE_FIELD_SIZE[0],
                Settings.Environment.Pheromone.PHEROMONE_FIELD_SIZE[1],
                Settings.Characteristics.Pheromone.CELL_SIZE_FOR_CALCULATION,
                Settings.Characteristics.Pheromone.SATURATION_VAPOR,
                Settings.Characteristics.Pheromone.EVAPORATION,
                Settings.Characteristics.Pheromone.DIFFUSION,
                Settings.Characteristics.Pheromone.DECREASE
            )
            self.pheromone = PheromoneFieldWithDummies(
                pheromone,
                Settings.Characteristics.Pheromone.CELL_SIZE_FOR_MUJOCO,
                create_dummies
            )
