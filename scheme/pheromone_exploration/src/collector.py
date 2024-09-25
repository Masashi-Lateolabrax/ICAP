import numpy as np

from libs.pheromone import PheromoneField

from .settings import Settings


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
