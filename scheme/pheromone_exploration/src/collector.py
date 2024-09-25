import numpy as np

from libs.pheromone import PheromoneField

from .settings import Settings
from .utils import init_pheromone_field


class IncreaseData:
    def __init__(self, para):
        total_step = int(Settings.Simulation.EPISODE_LENGTH / Settings.Simulation.TIMESTEP + 0.5)

        self._unstable = False
        self.pheromone = init_pheromone_field(para)
        self.gas: np.ndarray = np.zeros((total_step, *self.pheromone.get_all_gas().shape))
        self.dif_liquid: np.ndarray = np.zeros(total_step)

        for t in range(total_step):
            center_cell = self.pheromone.get_cell(
                xi=Settings.Pheromone.CENTER_INDEX[0],
                yi=Settings.Pheromone.CENTER_INDEX[1],
            )
            center_cell.set_liquid(Settings.Pheromone.LIQUID)

            self.gas[t] = gas = self.pheromone.get_all_gas()

            self.pheromone.update(Settings.Simulation.TIMESTEP)

            self.dif_liquid[t] = center_cell.get_liquid() - Settings.Pheromone.LIQUID

            a = np.min(gas)
            if a < 0.0:
                self._unstable = True
                break

    def is_unstable(self):
        return self._unstable

    def is_invalid(self):
        return self.pheromone is None


class IncreaseData2:
    def __init__(self, pheromone_parameter):
        self.pheromone = PheromoneField(
            nx=Settings.Environment.NUM_CELL[0],
            ny=Settings.Environment.NUM_CELL[1],
            d=Settings.Environment.CELL_SIZE,
            **pheromone_parameter
        )
        self.sv = self.pheromone.get_sv()
        self.gas: np.ndarray = np.zeros((Settings.Simulation.TOTAL_STEP, *self.pheromone.get_all_gas().shape))
        self.evaporation: np.ndarray = np.zeros(Settings.Simulation.TOTAL_STEP)

        for t in range(Settings.Simulation.TOTAL_STEP):
            center_cell = self.pheromone.get_cell(
                xi=Settings.Environment.CENTER_INDEX[0],
                yi=Settings.Environment.CENTER_INDEX[1],
            )
            center_cell.set_liquid(Settings.Environment.LIQUID)

            self.gas[t] = self.pheromone.get_all_gas()

            self.pheromone.update(Settings.Simulation.TIMESTEP)

            self.evaporation[t] = center_cell.get_liquid() - Settings.Environment.LIQUID


class DecreaseData:
    def __init__(self, data: IncreaseData):
        if data.is_invalid():
            return

        total_step = int(Settings.Simulation.EPISODE_LENGTH / Settings.Simulation.TIMESTEP + 0.5)

        pheromone = data.pheromone
        data.pheromone = None

        self._unstable = False
        self.gas = np.zeros((total_step, *pheromone.get_all_gas().shape))

        center_cell = pheromone.get_cell(
            xi=Settings.Pheromone.CENTER_INDEX[0],
            yi=Settings.Pheromone.CENTER_INDEX[1],
        )
        center_cell.set_liquid(0)

        for t in range(total_step):
            self.gas[t] = gas = pheromone.get_all_gas()

            pheromone.update(Settings.Simulation.TIMESTEP)

            a = np.min(gas)
            if a < 0.0:
                self._unstable = True
                break

    def is_unstable(self):
        return self._unstable


class DecreaseData2:
    def __init__(self, data: IncreaseData2):
        pheromone = data.pheromone
        data.pheromone = None

        self.sv = pheromone.get_sv()
        self.gas = np.zeros(data.gas.shape)

        center_cell = pheromone.get_cell(
            xi=Settings.Environment.CENTER_INDEX[0],
            yi=Settings.Environment.CENTER_INDEX[1],
        )
        center_cell.set_liquid(0)

        for t in range(Settings.Simulation.TOTAL_STEP):
            self.gas[t] = pheromone.get_all_gas()
            pheromone.update(Settings.Simulation.TIMESTEP)
