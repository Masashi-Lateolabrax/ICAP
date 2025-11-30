import matplotlib.pyplot as plt

import numpy as np

from framework.prelude import *
from src.analysis_mod.structure.debug_data import DebugData


def plot_total_gas_pheromone(settings: Settings, debug_data: list[DebugData], file_path: str):
    time_steps = [i * settings.Simulation.TIME_STEP for i in range(len(debug_data))]
    pheromone = np.array([di.total_gas_pheromone for di in debug_data])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(time_steps, pheromone)
    ax.set_xlabel('Time (s)')

    ax.set_ylabel('Total Gas Pheromone')
    ax.set_title('Total Gas Pheromone Over Time')

    fig.savefig(file_path)


def plot_total_liquid_pheromone(settings: Settings, debug_data: list[DebugData], file_path: str):
    time_steps = [i * settings.Simulation.TIME_STEP for i in range(len(debug_data))]
    pheromone = np.array([di.total_liquid_pheromone for di in debug_data])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(time_steps, pheromone)
    ax.set_xlabel('Time (s)')

    ax.set_ylabel('Total Liquid Pheromone')
    ax.set_title('Total Liquid Pheromone Over Time')

    fig.savefig(file_path)
