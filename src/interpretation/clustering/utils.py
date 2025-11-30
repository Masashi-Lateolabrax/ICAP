"""
Utility functions for clustering analysis.

This module provides common utility functions used across clustering modules.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pickle
from pathlib import Path

from src.analysis_mod.structure.debug_data import DebugData


@dataclass
class RobotSensorSample:
    """Single sample of robot sensor data for clustering analysis.

    This captures robot sensor inputs at a specific timestep for analyzing
    behavioral patterns through clustering.
    """
    # Time information
    timestep: int
    time_seconds: float

    # Robot identification
    robot_index: int

    # Full input vector (9 dimensions)
    full_input: np.ndarray  # Shape: (9,)

    # Decomposed sensor inputs
    robot_sensor: np.ndarray       # Shape: (2,) - PreprocessedOmniSensor for robots
    food_sensor: np.ndarray        # Shape: (2,) - PreprocessedOmniSensor for food
    direction_sensor: np.ndarray   # Shape: (2,) - DirectionSensor to nest
    pheromone_magnitude: float     # Scalar - normalized pheromone concentration
    pheromone_grad_forward: float  # Scalar - gradient in forward direction
    pheromone_grad_side: float     # Scalar - gradient in sideways direction

    # Network output for context
    network_output: np.ndarray     # Shape: (3,) - [right_wheel, left_wheel, pheromone_secretion]

    # Additional context
    robot_position: np.ndarray     # Shape: (2,) - (x, y)
    robot_direction: np.ndarray    # Shape: (2,) - unit vector


def convert_debug_data_to_dataset(
    debug_data: list[DebugData],
    experiment_id: str = "debug_analysis",
    generation: int = 0,
    timestep_offset: int = 0,
    time_step: float = 0.01
) -> list[RobotSensorSample]:
    """
    Convert DebugData list to list of RobotSensorSample for clustering analysis.

    Args:
        debug_data: List of DebugData from analysis run
        experiment_id: Experiment identifier (not used, kept for compatibility)
        generation: Generation number (not used, kept for compatibility)
        timestep_offset: Starting timestep number (default: 0)
        time_step: Simulation time step in seconds (default: 0.01)

    Returns:
        List of RobotSensorSample ready for clustering

    Note:
        DebugData.robot_inputs format (9 dimensions per robot):
        [0-1]: robot_sensor (magnitude, angle)
        [2-3]: food_sensor (magnitude, angle)
        [4-5]: direction_sensor (magnitude, angle)
        [6-8]: pheromone (magnitude, grad_forward, grad_side)
    """
    samples = []

    for frame_idx, frame in enumerate(debug_data):
        timestep = timestep_offset + frame_idx
        time_seconds = frame.time

        # robot_inputs shape: (n_robots, 9)
        n_robots = frame.robot_inputs.shape[0]

        for robot_idx in range(n_robots):
            # Extract input vector for this robot
            full_input = frame.robot_inputs[robot_idx]  # Shape: (9,)

            # Decompose input vector
            robot_sensor = full_input[0:2]
            food_sensor = full_input[2:4]
            direction_sensor = full_input[4:6]
            pheromone_magnitude = float(full_input[6])
            pheromone_grad_forward = float(full_input[7])
            pheromone_grad_side = float(full_input[8])

            # Get network output for this robot
            network_output = frame.robot_outputs[robot_idx]  # Shape: (3,)

            # Get position and direction
            robot_position = frame.robot_positions[robot_idx][:2]  # (x, y)
            robot_direction = frame.robot_directions[robot_idx][:2]  # unit vector

            # Create sample
            sample = RobotSensorSample(
                timestep=timestep,
                time_seconds=time_seconds,
                robot_index=robot_idx,
                full_input=full_input,
                robot_sensor=robot_sensor,
                food_sensor=food_sensor,
                direction_sensor=direction_sensor,
                pheromone_magnitude=pheromone_magnitude,
                pheromone_grad_forward=pheromone_grad_forward,
                pheromone_grad_side=pheromone_grad_side,
                network_output=network_output,
                robot_position=robot_position,
                robot_direction=robot_direction,
            )

            samples.append(sample)

    return samples


def convert_debug_data_to_dataset_filtered(
    debug_data: list[DebugData],
    experiment_id: str = "debug_analysis",
    generation: int = 0,
    timestep_offset: int = 0,
    time_step: float = 0.01,
    pheromone_threshold: float = 0.0,
    sample_interval: int = 1
) -> list[RobotSensorSample]:
    """
    Convert DebugData to list of RobotSensorSample with filtering options.

    Args:
        debug_data: List of DebugData from analysis run
        experiment_id: Experiment identifier (not used, kept for compatibility)
        generation: Generation number (not used, kept for compatibility)
        timestep_offset: Starting timestep number
        time_step: Simulation time step in seconds (not used, kept for compatibility)
        pheromone_threshold: Only include samples with pheromone >= threshold
        sample_interval: Only include every Nth timestep (1 = all, 2 = every other, etc.)

    Returns:
        List of RobotSensorSample with filtered samples
    """
    samples = []

    for frame_idx, frame in enumerate(debug_data):
        # Skip frames based on sample interval
        if frame_idx % sample_interval != 0:
            continue

        timestep = timestep_offset + frame_idx
        time_seconds = frame.time

        n_robots = frame.robot_inputs.shape[0]

        for robot_idx in range(n_robots):
            full_input = frame.robot_inputs[robot_idx]
            pheromone_magnitude = float(full_input[6])

            # Apply pheromone threshold filter
            if pheromone_magnitude < pheromone_threshold:
                continue

            # Decompose input vector
            robot_sensor = full_input[0:2]
            food_sensor = full_input[2:4]
            direction_sensor = full_input[4:6]
            pheromone_grad_forward = float(full_input[7])
            pheromone_grad_side = float(full_input[8])

            network_output = frame.robot_outputs[robot_idx]
            robot_position = frame.robot_positions[robot_idx][:2]
            robot_direction = frame.robot_directions[robot_idx][:2]

            sample = RobotSensorSample(
                timestep=timestep,
                time_seconds=time_seconds,
                robot_index=robot_idx,
                full_input=full_input,
                robot_sensor=robot_sensor,
                food_sensor=food_sensor,
                direction_sensor=direction_sensor,
                pheromone_magnitude=pheromone_magnitude,
                pheromone_grad_forward=pheromone_grad_forward,
                pheromone_grad_side=pheromone_grad_side,
                network_output=network_output,
                robot_position=robot_position,
                robot_direction=robot_direction,
            )

            samples.append(sample)

    return samples


def save_debug_data_as_dataset(
    debug_data: list[DebugData],
    output_path: str,
    experiment_id: str = "debug_analysis",
    generation: int = 0,
    **kwargs
):
    """
    Convert DebugData and save as list of RobotSensorSample pickle file.

    Args:
        debug_data: List of DebugData from analysis run
        output_path: Path to save pickle file
        experiment_id: Experiment identifier (not used, kept for compatibility)
        generation: Generation number (not used, kept for compatibility)
        **kwargs: Additional arguments passed to convert_debug_data_to_dataset_filtered
    """
    # Convert to samples list
    samples = convert_debug_data_to_dataset_filtered(
        debug_data,
        experiment_id=experiment_id,
        generation=generation,
        **kwargs
    )

    # Save to pickle
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)

    print(f"Saved dataset to: {output_path}")
    print(f"  Total samples: {len(samples)}")
