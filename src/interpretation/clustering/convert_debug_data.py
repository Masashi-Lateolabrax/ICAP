"""
Convert DebugData from analysis to ShapleyDataset for clustering.

This module provides conversion utilities to transform debug data collected
during analysis runs into the ShapleyDataset format used for clustering.

Usage:
    from src.analysis_mod.for_individual.run import record
    from src.interpretation.clustering.convert_debug_data import convert_debug_data_to_dataset

    # Run analysis and collect debug data
    debug_data = record(settings, simulator, "output.mp4")

    # Convert to clustering format
    dataset = convert_debug_data_to_dataset(debug_data, experiment_id="test", generation=0)

    # Run clustering
    result = cluster_sensor_states(dataset, n_clusters=9)
"""

from typing import Optional
import numpy as np

from src.analysis_mod.structure.debug_data import DebugData
from src.interpretation.data_collection.io_sample_definition import (
    ShapleyInputSample,
    ShapleyDataset,
)


def convert_debug_data_to_dataset(
    debug_data: list[DebugData],
    experiment_id: str = "debug_analysis",
    generation: int = 0,
    timestep_offset: int = 0,
    time_step: float = 0.01
) -> ShapleyDataset:
    """
    Convert DebugData list to ShapleyDataset for clustering analysis.

    Args:
        debug_data: List of DebugData from analysis run
        experiment_id: Experiment identifier (default: "debug_analysis")
        generation: Generation number (default: 0)
        timestep_offset: Starting timestep number (default: 0)
        time_step: Simulation time step in seconds (default: 0.01)

    Returns:
        ShapleyDataset ready for clustering

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
            sample = ShapleyInputSample(
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

    # Create dataset
    dataset = ShapleyDataset(
        experiment_id=experiment_id,
        generation=generation,
        samples=samples,
    )

    return dataset


def convert_debug_data_to_dataset_filtered(
    debug_data: list[DebugData],
    experiment_id: str = "debug_analysis",
    generation: int = 0,
    timestep_offset: int = 0,
    time_step: float = 0.01,
    pheromone_threshold: float = 0.0,
    sample_interval: int = 1
) -> ShapleyDataset:
    """
    Convert DebugData to ShapleyDataset with filtering options.

    Args:
        debug_data: List of DebugData from analysis run
        experiment_id: Experiment identifier
        generation: Generation number
        timestep_offset: Starting timestep number
        time_step: Simulation time step in seconds
        pheromone_threshold: Only include samples with pheromone >= threshold
        sample_interval: Only include every Nth timestep (1 = all, 2 = every other, etc.)

    Returns:
        ShapleyDataset with filtered samples
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

            sample = ShapleyInputSample(
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

    dataset = ShapleyDataset(
        experiment_id=experiment_id,
        generation=generation,
        samples=samples,
    )

    return dataset


def save_debug_data_as_dataset(
    debug_data: list[DebugData],
    output_path: str,
    experiment_id: str = "debug_analysis",
    generation: int = 0,
    **kwargs
):
    """
    Convert DebugData and save as ShapleyDataset pickle file.

    Args:
        debug_data: List of DebugData from analysis run
        output_path: Path to save pickle file
        experiment_id: Experiment identifier
        generation: Generation number
        **kwargs: Additional arguments passed to convert_debug_data_to_dataset_filtered
    """
    import pickle
    from pathlib import Path

    # Convert to dataset (use filtered version for flexibility)
    dataset = convert_debug_data_to_dataset_filtered(
        debug_data,
        experiment_id=experiment_id,
        generation=generation,
        **kwargs
    )

    # Save to pickle
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"Saved dataset to: {output_path}")
    print(f"  Total samples: {len(dataset.samples)}")
    print(f"  Experiment: {dataset.experiment_id}")
    print(f"  Generation: {dataset.generation}")
