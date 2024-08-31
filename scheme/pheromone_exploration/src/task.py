import warnings

import numpy as np

from libs.optimizer import TaskInterface, TaskGenerator

from .settings import Settings
from .utils import init_pheromone_field, convert_para, calc_consistency
from .collector import IncreaseData, DecreaseData


def calc_gas_volume_err(gas, time, target, sv):
    idx = int(time / Settings.Simulation.TIMESTEP + 0.5)
    volume_err = np.max(gas[idx] / sv) - target
    return volume_err


def calc_sv_err(gas, time, target, sv):
    idx = int(time / Settings.Simulation.TIMESTEP + 0.5)
    sv_err = (gas[idx:] / sv - target) ** 2
    sv_err = np.sum(sv_err * Settings.Simulation.TIMESTEP) / Settings.Simulation.EPISODE_LENGTH
    return sv_err


def calc_size_err(gas, time, size: float, target, sv):
    idx = int(time / Settings.Simulation.TIMESTEP + 0.5)
    center_idx = Settings.Pheromone.CENTER_INDEX
    size = size / Settings.Pheromone.CELL_SIZE_FOR_MUJOCO

    s1 = int(size)
    s2 = s1 + 1

    g1 = gas[idx, center_idx[0], center_idx[1] + s1]
    g2 = gas[idx, center_idx[0], center_idx[1] + s2]

    g = (g2 - g1) * (size - s1) + g1

    size_err = g / sv - target
    return size_err


class Task(TaskInterface):
    def __init__(self, para):
        self.pheromone = init_pheromone_field(para)
        self.para = np.array(para)

    def run(self) -> float:
        sv = convert_para(self.para)["sv"]

        if sv < Settings.Pheromone.MIN_SV:
            warnings.warn("The saturated vapor is nealy zero.")
            return 100000000

        data_inc = IncreaseData(self.para)
        data_dec = DecreaseData(data_inc)

        if data_inc.is_unstable() or data_dec.is_unstable():
            warnings.warn("Pheromone calculation is unstable")
            return 100000000

        consistency = calc_consistency(data_inc.gas)

        sv_err = calc_sv_err(
            data_inc.gas,
            Settings.Optimization.Target.INCREASE_TIME,
            Settings.Optimization.Target.RELATIVE_STABLE_GAS_VOLUME,
            sv,
        )

        dec_volume_err = calc_gas_volume_err(
            data_dec.gas,
            Settings.Optimization.Target.INCREASE_TIME,
            Settings.Optimization.Target.RELATIVE_STABLE_GAS_VOLUME,
            sv
        )

        size_err = calc_size_err(
            data_inc.gas,
            Settings.Optimization.Target.INCREASE_TIME,
            Settings.Optimization.Target.FIELD_SIZE,
            Settings.Optimization.Target.RELATIVE_STABLE_GAS_VOLUME * 0.5,
            sv
        )

        err = np.abs(np.array([
            sv_err,
            dec_volume_err,
            size_err,
        ]))

        loss = float(np.sum(err * np.array(Settings.Optimization.Weight.WEIGHT)))
        loss += (float(np.min(consistency)) - 1) * 0.5 * Settings.Optimization.Weight.CONSISTENCY_WEIGHT
        loss += np.sum(self.para ** 2) * Settings.Optimization.Weight.L2

        return loss


class Generator(TaskGenerator):
    def generate(self, para, debug=False) -> TaskInterface:
        return Task(para)
