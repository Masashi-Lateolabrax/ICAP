import warnings

import numpy as np

from lib.optimizer import TaskInterface, TaskGenerator
from lib.pheromone import PheromoneField2

from .settings import Settings


class Task(TaskInterface):
    def __init__(self, para):
        para = np.tanh(para) + 1

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

        stability = np.zeros(gas_buf1.shape[0] - 2)
        for t in range(2, gas_buf1.shape[0]):
            current_gas = (gas_buf1[t - 2] - gas_buf1[t - 1]).ravel()
            next_gas = (gas_buf1[t - 1] - gas_buf1[t]).ravel()
            d = np.linalg.norm(next_gas) * np.linalg.norm(current_gas)
            stability[t - 2] = np.dot(current_gas, next_gas)
            stability[t - 2] = stability[t - 2] / d if d > 0.00000001 else 0.0

        a = np.where(stability > Settings.Evaluation.STABILITY_THRESHOLD)[0]
        if a.size == 0:
            warnings.warn("The pheromone field calculation is very slow. [STABLE]")
            return 100000000.0

        stable_state_index = np.min(a)
        stable_state_time = stable_state_index * Settings.Simulation.TIMESTEP

        evaporation_speed = dif_liquid[stable_state_index]
        stable_gas_volume = np.max(gas_buf1[stable_state_index])
        relative_stable_gas_volume = stable_gas_volume / self.pheromone.get_sv()

        distances = np.ones(total_step) * Settings.Pheromone.CENTER_INDEX[1]
        for t in range(total_step):
            max_gas = gas_buf1[t, Settings.Pheromone.CENTER_INDEX[0], Settings.Pheromone.CENTER_INDEX[1]]
            sub_gas = gas_buf1[t, Settings.Pheromone.CENTER_INDEX[0], Settings.Pheromone.CENTER_INDEX[1]:]
            s1 = np.max(np.where(sub_gas >= max_gas * 0.5)[0])
            if s1 == sub_gas.shape[0] - 1:
                break
            g1 = sub_gas[s1]
            g2 = sub_gas[s1 + 1]
            distances[t] = (max_gas * 0.5 - g1) / (g2 - g1) + s1
        distances *= Settings.Pheromone.CELL_SIZE_FOR_CALCULATION
        field_size = np.max(distances)

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
        loss = float(np.sum(((result - self._target) ** 2) * np.array(Settings.Optimization.Loss.WEIGHT)))
        loss += (1 - float(np.min(stability))) * 0.5 * Settings.Optimization.Loss.UNSTABLE_WEIGHT

        return loss


class Generator(TaskGenerator):
    def generate(self, para, debug=False) -> TaskInterface:
        return Task(para)
