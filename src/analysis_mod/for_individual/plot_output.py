import matplotlib.pyplot as plt

import numpy as np

from framework.prelude import *
from src.analysis_mod.structure.debug_data import DebugData


def _plot(ax: plt.Axes, settings: Settings, debug_data: list[DebugData], idx: int):
    time_steps = [i * settings.Simulation.TIME_STEP for i in range(len(debug_data))]
    num_robots = debug_data[0].robot_outputs.shape[0]
    powers = np.array([di.robot_outputs[:, idx] for di in debug_data])

    for ri in range(num_robots):
        ax.plot(time_steps, powers[:, ri], label=f'Robot {ri}')

    ax.set_xlabel('Time (s)')


def plot_left_wheel_act(settings: Settings, debug_data: list[DebugData], file_path: str):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    _plot(ax, settings, debug_data, idx=1)

    ax.set_ylabel('Left Wheel Actuation')
    ax.set_title('Left Wheel Actuation Over Time')
    ax.legend()

    fig.savefig(file_path)


def plot_right_wheel_act(settings: Settings, debug_data: list[DebugData], file_path: str):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    _plot(ax, settings, debug_data, idx=0)

    ax.set_ylabel('Right Wheel Actuation')
    ax.set_title('Right Wheel Actuation Over Time')
    ax.legend()

    fig.savefig(file_path)


def plot_pheromone_act(settings: Settings, debug_data: list[DebugData], file_path: str):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    _plot(ax, settings, debug_data, idx=2)

    ax.set_ylabel('Pheromone Actuation')
    ax.set_title('Pheromone Actuation Over Time')
    ax.legend()

    fig.savefig(file_path)
