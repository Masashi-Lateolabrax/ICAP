import warnings

import numpy as np

from .settings import Settings


def calc_consistency(gas):
    consistency = np.zeros(gas.shape[0] - 2)
    for t in range(2, gas.shape[0]):
        current_gas = (gas[t - 1] - gas[t - 2]).ravel()
        cd = np.linalg.norm(current_gas)
        next_gas = (gas[t] - gas[t - 1]).ravel()
        nd = np.linalg.norm(next_gas)
        if np.isinf(cd) or np.isinf(nd):
            warnings.warn("The pheromone field calculation fell into an unstable state.")
            return 100000000.0
        if cd < 1e-30:
            current_gas.fill(0)
        else:
            current_gas /= cd
        if nd < 1e-30:
            next_gas.fill(0)
        else:
            next_gas /= nd
        consistency[t - 2] = np.dot(current_gas, next_gas)
    return consistency


def calc_stability(gas, sv):
    return np.max(np.abs(gas[1:] - gas[:-1]), axis=(1, 2)) / sv


def calc_size(gas, total_step, pheromone_cell_size):
    distances = np.ones(total_step) * Settings.Pheromone.CENTER_INDEX[1]
    for t in range(total_step):
        max_gas = gas[t, Settings.Pheromone.CENTER_INDEX[0], Settings.Pheromone.CENTER_INDEX[1]]
        sub_gas = gas[t, Settings.Pheromone.CENTER_INDEX[0], Settings.Pheromone.CENTER_INDEX[1]:]
        s1 = np.max(np.where(sub_gas >= max_gas * 0.5)[0])
        if s1 == sub_gas.shape[0] - 1:
            break
        g1 = sub_gas[s1]
        g2 = sub_gas[s1 + 1]
        distances[t] = (max_gas * 0.5 - g1) / (g2 - g1) + s1
    distances *= pheromone_cell_size
    return distances
