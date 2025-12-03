"""
Simulation utilities for Shapley value analysis of cluster transitions.
"""

import numpy as np
import torch
import mujoco

from src.config import Simulator


def extract_sensor_features(simulator: Simulator) -> np.ndarray:
    """Extract 6D sensor features from robot inputs.

    Args:
        simulator: Simulator instance

    Returns:
        (n_robots, 6) array: [robot_sensor_x, robot_sensor_y, food_sensor_x, food_sensor_y,
                              direction_sensor_x, direction_sensor_y] for all robots
    """
    return simulator.input_ndarray[:, 0:6]


def predict_cluster(kmeans, scaler, sensor_features: np.ndarray) -> np.ndarray:
    """Predict cluster labels from sensor features.

    Args:
        kmeans: Trained KMeans model
        scaler: Trained StandardScaler
        sensor_features: (n_robots, 6) array of sensor features

    Returns:
        (n_robots,) array: Cluster labels for all robots
    """
    features_scaled = scaler.transform(sensor_features)
    return kmeans.predict(features_scaled)


def step(simulator: Simulator, kmeans, scaler):
    """Execute one step and detect cluster transitions for all robots.

    Args:
        simulator: Simulator instance
        kmeans: Trained KMeans model
        scaler: Trained StandardScaler

    Returns:
        (n_robots,) bool array: True if cluster transition occurred for each robot
    """
    # Get current clusters
    current_features = extract_sensor_features(simulator)  # (n_robots, 6)
    current_clusters = predict_cluster(kmeans, scaler, current_features)  # (n_robots,)

    # Execute actual step
    simulator.step()

    # Get next clusters
    next_features = extract_sensor_features(simulator)  # (n_robots, 6)
    next_clusters = predict_cluster(kmeans, scaler, next_features)  # (n_robots,)

    # Check for transitions
    return current_clusters != next_clusters  # (n_robots,)


def step_baseline(simulator: Simulator) -> np.ndarray:
    """Execute one step with baseline (pheromone=0) and return next input.

    This function executes Simulator.step() but with pheromone input zeroed out.
    Unlike the normal step(), this returns the next step's input features.

    Args:
        simulator: Simulator instance

    Returns:
        (n_robots, 9) array: Next step's input features
    """

    # Use pre-allocated arrays to avoid memory allocation overhead
    for i, robot in enumerate(simulator.robot_values):
        simulator._robot_positions[i] = robot.xpos  # (n_robots, 2)
        simulator._robot_v_direction[i] = robot.xdirection  # (n_robots, 2)
        simulator._robot_h_direction[i, 0] = robot.xdirection[1]  # (n_robots, 2)
        simulator._robot_h_direction[i, 1] = -robot.xdirection[0]

    if simulator.timer.tick():
        with torch.no_grad():
            input_ = simulator.create_input_for_controller()  # (n_robots, 9)
            input_[:, 6:9] = 0.0  # (n_robots, 3) <- 0

            output = simulator.controller.forward(input_)  # (n_robots, 3)
            simulator.output_ndarray = output.numpy()  # (n_robots, 3)

    for i, robot in enumerate(simulator.robot_values):
        robot.act(
            right_wheel=simulator.output_ndarray[i, 0],  # scalar
            left_wheel=simulator.output_ndarray[i, 1]  # scalar
        )

    if simulator._pheromone_field is not None:
        simulator.add_pheromone(
            simulator._robot_positions,  # (n_robots, 2)
            simulator.output_ndarray[:, 2] * simulator.settings.Robot.MAX_PHEROMONE_SECRETION  # (n_robots,)
        )
        simulator._pheromone_field.add_liquid_by_cell(simulator._pheromone_cells)
        simulator._pheromone_field.step()

    mujoco.mj_step(simulator.model, simulator.data)

    return simulator.create_input_for_controller().numpy()  # (n_robots, 9)
