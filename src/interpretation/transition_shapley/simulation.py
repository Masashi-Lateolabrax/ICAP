"""
Simulation utilities for Shapley value analysis of cluster transitions.
"""

import numpy as np

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
