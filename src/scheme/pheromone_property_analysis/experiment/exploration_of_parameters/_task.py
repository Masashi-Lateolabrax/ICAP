import warnings

import numpy as np

from lib.optimizer import TaskInterface, TaskGenerator

from ._utils import calc_consistency, init_pheromone_field

from .settings import Settings


class Task(TaskInterface):
    def __init__(self, para):
        self.pheromone = init_pheromone_field(para)
        self.para = np.array(para)

    def run(self) -> float:
        total_step = int(Settings.Simulation.EPISODE_LENGTH / Settings.Simulation.TIMESTEP + 0.5)

        gas_buf1 = np.zeros((total_step, *self.pheromone.get_all_gas().shape))
        gas_buf2 = np.zeros(gas_buf1.shape)

        for t in range(total_step):
            center_cell = self.pheromone.get_cell_v2(
                xi=Settings.Pheromone.CENTER_INDEX[0],
                yi=Settings.Pheromone.CENTER_INDEX[1],
            )
            center_cell.set_liquid(Settings.Pheromone.LIQUID)

            self.pheromone.update(Settings.Simulation.TIMESTEP)

            gas_buf1[t] = self.pheromone.get_all_gas()

            a = np.min(gas_buf1[t])
            if a < 0.0:
                warnings.warn("Pheromone calculation is unstable.")
                return 100000000

        self.pheromone.get_cell_v2(
            xi=Settings.Pheromone.CENTER_INDEX[0],
            yi=Settings.Pheromone.CENTER_INDEX[1],
        ).set_liquid(0)
        for t in range(total_step):
            self.pheromone.update(Settings.Simulation.TIMESTEP)
            gas_buf2[t] = self.pheromone.get_all_gas()

        sv = self.pheromone.get_sv()
        if sv < Settings.Pheromone.MIN_SV:
            warnings.warn("The saturated vapor is nealy zero.")
            return 100000000
        relative_gas = np.max(gas_buf1, axis=(1, 2)) / sv

        consistency = calc_consistency(gas_buf1)

        # increase
        increase_idx = int(Settings.Optimization.Loss.INCREASE_TIME / Settings.Simulation.TIMESTEP + 0.5)
        increase_err = np.max(relative_gas[increase_idx]) - Settings.Optimization.Loss.RELATIVE_STABLE_GAS_VOLUME

        # sv_err
        sv_err = np.abs(relative_gas[increase_idx:] - Settings.Optimization.Loss.RELATIVE_STABLE_GAS_VOLUME)
        sv_err = np.sum(sv_err * Settings.Simulation.TIMESTEP) / Settings.Simulation.EPISODE_LENGTH

        # field_size
        size_idx = int(Settings.Optimization.Loss.SIZE_TIME / Settings.Simulation.TIMESTEP + 0.5)
        center_idx = Settings.Pheromone.CENTER_INDEX
        g = gas_buf1[size_idx, center_idx[0], center_idx[1] + Settings.Optimization.Loss.FIELD_SIZE] / sv
        size_err = g - Settings.Optimization.Loss.RELATIVE_STABLE_GAS_VOLUME * 0.5

        # decreased_err
        decrease_idx = int(Settings.Optimization.Loss.DECREASE_TIME / Settings.Simulation.TIMESTEP + 0.5)
        decrease_err = np.max(gas_buf2[decrease_idx]) - Settings.Optimization.Loss.RELATIVE_STABLE_GAS_VOLUME * 0.5

        err = np.abs(np.array([
            sv_err,
            increase_err,
            decrease_err,
            size_err,
        ]))

        loss = float(np.sum(err * np.array(Settings.Optimization.Loss.WEIGHT)))
        loss += (float(np.min(consistency)) - 1) * 0.5 * Settings.Optimization.Loss.CONSISTENCY_WEIGHT
        loss += np.sum(self.para ** 2) * Settings.Optimization.Loss.L2

        return loss


class Generator(TaskGenerator):
    def generate(self, para, debug=False) -> TaskInterface:
        return Task(para)
