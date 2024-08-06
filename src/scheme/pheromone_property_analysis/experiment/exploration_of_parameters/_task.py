import warnings

import numpy as np

from lib.optimizer import TaskInterface, TaskGenerator
from lib.pheromone import PheromoneField2

from .settings import Settings


class Task(TaskInterface):
    def __init__(self, para):
        para = np.log(1 + np.exp(para))

        self.pheromone = PheromoneField2(
            nx=Settings.Pheromone.NUM_CELL[0],
            ny=Settings.Pheromone.NUM_CELL[1],
            d=Settings.Pheromone.CELL_SIZE_FOR_CALCULATION,
            sv=para[0],
            evaporate=para[1],
            diffusion=para[2],
            decrease=para[4]
        )

        self._target = np.array([
            Settings.Optimization.Loss.STABLE_STATE_TIME,
            Settings.Optimization.Loss.DECREASED_STATE_TIME,
            Settings.Optimization.Loss.EVAPORATION_SPEED,
            Settings.Optimization.Loss.RELATIVE_STABLE_GAS_VOLUME,
            Settings.Optimization.Loss.FIELD_SIZE,
        ])

        self.gas_buf = np.zeros(0)

    def run(self) -> float:
        total_step = int(Settings.Simulation.EPISODE_LENGTH / Settings.Simulation.TIMESTEP + 0.5)

        dif_liquid = np.zeros(total_step)
        gas_buf1 = np.zeros((total_step, *self.pheromone.get_all_gas().shape))
        gas_buf2 = np.zeros(gas_buf1.shape)

        for t in range(total_step):
            center_cell = self.pheromone.get_cell_v2(
                xi=Settings.Pheromone.CENTER_INDEX[0],
                yi=Settings.Pheromone.CENTER_INDEX[1],
            )
            center_cell.set_liquid(Settings.Pheromone.LIQUID)

            before_liquid = self.pheromone.get_all_liquid()
            self.pheromone.update(Settings.Simulation.TIMESTEP)

            dif_liquid[t] = np.sum(self.pheromone.get_all_liquid() - before_liquid) / Settings.Simulation.TIMESTEP
            gas_buf1[t] = self.pheromone.get_all_gas()

        self.pheromone.get_cell_v2(
            xi=Settings.Pheromone.CENTER_INDEX[0],
            yi=Settings.Pheromone.CENTER_INDEX[1],
        ).set_liquid(0)
        for t in range(total_step):
            self.pheromone.update(Settings.Simulation.TIMESTEP)
            gas_buf2[t] = self.pheromone.get_all_gas()

        stability = np.linalg.norm(gas_buf1[1:] - gas_buf1[:-1], axis=(1, 2))

        if np.max(np.abs(stability[1:] - stability[:-1])) > Settings.Evaluation.EPS:
            warnings.warn("The pheromone field calculation fell into an unstable state.")
            return 100000000.0

        a = np.where(np.exp(-stability) > Settings.Evaluation.STABILITY_THRESHOLD)[0]
        if a.size == 0:
            warnings.warn("The pheromone field calculation is very slow. [STABLE]")
            return 100000000.0

        stable_state_index = np.min(a)
        stable_state_time = stable_state_index * Settings.Simulation.TIMESTEP

        evaporation_speed = dif_liquid[stable_state_index]
        stable_gas_volume = np.max(gas_buf1[stable_state_index])
        relative_stable_gas_volume = stable_gas_volume / self.pheromone.get_sv()

        distances = np.linalg.norm(
            np.array(
                np.where(gas_buf1[stable_state_index] >= stable_gas_volume * 0.5)
            ).T - np.array(Settings.Pheromone.CENTER_INDEX),
            axis=1
        )
        field_size = np.max(distances) * Settings.Pheromone.CELL_SIZE_FOR_CALCULATION

        a = np.where(np.max(gas_buf2, axis=(1, 2)) <= stable_gas_volume * 0.5)[0]
        if a.size == 0:
            warnings.warn("The pheromone field calculation is very slow. [DECREASE]")
            return 100000000.0
        decreased_state_index = np.min(a)
        decreased_state_time = decreased_state_index * Settings.Simulation.TIMESTEP

        result = np.array([
            stable_state_time,
            decreased_state_time,
            evaporation_speed,
            relative_stable_gas_volume,
            field_size
        ])

        return float(np.sum((result - self._target) ** 2))


class Generator(TaskGenerator):
    def generate(self, para, debug=False) -> TaskInterface:
        return Task(para)
