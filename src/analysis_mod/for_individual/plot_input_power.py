import matplotlib.pyplot as plt

import numpy as np

from framework.prelude import *
from src.analysis_mod.structure.debug_data import DebugData


def plot_velocity_sensor(settings: Settings, debug_data: list[DebugData], file_path: str):
    time_steps = [i * settings.Simulation.TIME_STEP for i in range(len(debug_data))]
    num_robots = debug_data[0].robot_inputs.shape[0]

    # VelocitySensor outputs: v_x (index 2), v_y (index 3), ω_z (index 4)
    v_x = np.array([di.robot_inputs[:, 2] for di in debug_data])
    v_y = np.array([di.robot_inputs[:, 3] for di in debug_data])
    omega_z = np.array([di.robot_inputs[:, 4] for di in debug_data])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Plot v_x
    for ri in range(num_robots):
        axes[0].plot(time_steps, v_x[:, ri], label=f'Robot {ri}')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('v_x')
    axes[0].set_title('Linear Velocity X')
    axes[0].legend()

    # Plot v_y
    for ri in range(num_robots):
        axes[1].plot(time_steps, v_y[:, ri], label=f'Robot {ri}')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('v_y')
    axes[1].set_title('Linear Velocity Y')
    axes[1].legend()

    # Plot ω_z
    for ri in range(num_robots):
        axes[2].plot(time_steps, omega_z[:, ri], label=f'Robot {ri}')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('ω_z')
    axes[2].set_title('Angular Velocity Z')
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(file_path)
    plt.close(fig)


def plot_pheromone_sensor(settings: Settings, debug_data: list[DebugData], file_path: str):
    time_steps = [i * settings.Simulation.TIME_STEP for i in range(len(debug_data))]
    num_robots = debug_data[0].robot_inputs.shape[0]

    # Pheromone index: 5 + DEPTH_SENSOR_NUM_RAYS
    pheromone_idx = 5 + settings.Robot.DEPTH_SENSOR_NUM_RAYS
    powers = np.array([di.robot_inputs[:, pheromone_idx] for di in debug_data])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for ri in range(num_robots):
        ax.plot(time_steps, powers[:, ri], label=f'Robot {ri}')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Sensor Power')
    ax.set_title('Pheromone Sensor Power Over Time')
    ax.legend()

    fig.savefig(file_path)
    plt.close(fig)
