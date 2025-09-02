import matplotlib.pyplot as plt

import numpy as np

from framework.prelude import *
from src.analysis_mod.structure.debug_data import DebugData


def plot_robot_sensor(settings: Settings, debug_data: list[DebugData], file_path: str):
    time_steps = [i * settings.Simulation.TIME_STEP for i in range(len(debug_data))]
    num_robots = debug_data[0].robot_inputs.shape[0]
    powers = np.array([np.linalg.norm(di.robot_inputs[:, 0:2], axis=1) for di in debug_data])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for ri in range(num_robots):
        ax.plot(time_steps, powers[:, ri], label=f'Robot {ri}')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Sensor Power')
    ax.set_title('Robot Sensor Power Over Time')
    ax.legend()

    fig.savefig(file_path)


def plot_food_sensor(settings: Settings, debug_data: list[DebugData], file_path: str):
    time_steps = [i * settings.Simulation.TIME_STEP for i in range(len(debug_data))]
    num_robots = debug_data[0].robot_inputs.shape[0]
    powers = np.array([np.linalg.norm(di.robot_inputs[:, 2:4], axis=1) for di in debug_data])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for ri in range(num_robots):
        ax.plot(time_steps, powers[:, ri], label=f'Robot {ri}')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Sensor Power')
    ax.set_title('Food Sensor Power Over Time')
    ax.legend()

    fig.savefig(file_path)


def plot_pheromone_sensor(settings: Settings, debug_data: list[DebugData], file_path: str):
    time_steps = [i * settings.Simulation.TIME_STEP for i in range(len(debug_data))]
    num_robots = debug_data[0].robot_inputs.shape[0]
    powers = np.array([di.robot_inputs[:, 6] for di in debug_data])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for ri in range(num_robots):
        ax.plot(time_steps, powers[:, ri], label=f'Robot {ri}')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Sensor Power')
    ax.set_title('Pheromone Sensor Power Over Time')
    ax.legend()

    fig.savefig(file_path)
