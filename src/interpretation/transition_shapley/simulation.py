"""
Simulation utilities for Shapley value analysis of cluster transitions.
"""

import numpy as np
import torch
import mujoco

from src.config import Simulator


def extract_sensor_features(simulator: Simulator, robot_idx: int) -> np.ndarray:
    """Extract 6D sensor features from robot inputs.

    Args:
        simulator: Simulator instance
        robot_idx: Index of robot to extract features for

    Returns:
        (6,) array: [robot_sensor_x, robot_sensor_y, food_sensor_x, food_sensor_y,
                     direction_sensor_x, direction_sensor_y]
    """
    return simulator.input_ndarray[robot_idx, 0:6]


def predict_cluster(kmeans, scaler, sensor_features: np.ndarray) -> int:
    """Predict cluster label from sensor features.

    Args:
        kmeans: Trained KMeans model
        scaler: Trained StandardScaler
        sensor_features: (6,) array of sensor features

    Returns:
        Cluster label (int)
    """
    features_scaled = scaler.transform(sensor_features.reshape(1, -1))
    return kmeans.predict(features_scaled)[0]


def step(simulator: Simulator, kmeans, scaler, robot_idx: int) -> bool:
    """Execute one step and detect cluster transition for specified robot.

    Args:
        simulator: Simulator instance
        kmeans: Trained KMeans model
        scaler: Trained StandardScaler
        robot_idx: Index of robot to track

    Returns:
        True if cluster transition occurred, False otherwise
    """
    # Get current cluster
    current_features = extract_sensor_features(simulator, robot_idx)
    current_cluster = predict_cluster(kmeans, scaler, current_features)

    # Execute actual step
    simulator.step()

    # Get next cluster
    next_features = extract_sensor_features(simulator, robot_idx)
    next_cluster = predict_cluster(kmeans, scaler, next_features)

    # Check for transition
    return current_cluster != next_cluster


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
        simulator._robot_positions[i] = robot.xpos
        simulator._robot_v_direction[i] = robot.xdirection
        simulator._robot_h_direction[i, 0] = robot.xdirection[1]
        simulator._robot_h_direction[i, 1] = -robot.xdirection[0]

    if simulator.timer.tick():
        with torch.no_grad():
            input_ = simulator.create_input_for_controller()
            # Zero out pheromone features for baseline
            input_[:, 6:9] = 0.0

            output = simulator.controller.forward(input_)
            simulator.output_ndarray = output.numpy()

    for i, robot in enumerate(simulator.robot_values):
        robot.act(
            right_wheel=simulator.output_ndarray[i, 0],
            left_wheel=simulator.output_ndarray[i, 1]
        )

    if simulator._pheromone_field is not None:
        simulator.add_pheromone(
            simulator._robot_positions,
            simulator.output_ndarray[:, 2] * simulator.settings.Robot.MAX_PHEROMONE_SECRETION
        )
        simulator._pheromone_field.add_liquid_by_cell(simulator._pheromone_cells)
        simulator._pheromone_field.step()

        max_pheromone = simulator._pheromone_field.get_max_value()
        simulator._max_pheromone = max(simulator._max_pheromone, max_pheromone)

    for food in simulator.food_values:
        if np.linalg.norm(food.xpos - simulator.nest_site.xpos[0:2]) <= simulator.settings.Nest.RADIUS:
            simulator._respawn_food(food)

    mujoco.mj_step(simulator.model, simulator.data)

    return simulator.create_input_for_controller().numpy()
