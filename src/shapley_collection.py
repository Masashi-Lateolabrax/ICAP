"""
Shapley Value Data Collection Script

This script loads a trained neural network from a completed experiment and runs
long-term simulations to collect robot input data when pheromone is detected.
The collected data is used for Shapley value analysis to quantify sensor contributions.

Usage:
    PYTHONPATH=. uv run src/shapley_collection.py --extra cu128 [options]

Example:
    PYTHONPATH=. uv run src/shapley_collection.py --extra cu128 \
        --experiment-id 20251027-024639_7bb53c8b \
        --generation 499 \
        --duration 300 \
        --threshold 0.01
"""

import argparse
import pickle
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

from framework.prelude import Settings
from framework.types import IndividualRecorder
from config.simulator import Simulator
from shapley_data import ShapleyInputSample, ShapleyDataset
from settings import MySettings


def collect_shapley_data(
    experiment_id: str,
    generation: int,
    duration: float,
    pheromone_threshold: float,
    render: bool = False
) -> ShapleyDataset:
    """
    Collect Shapley input data from a trained model.

    Args:
        experiment_id: Experiment ID (directory name in results/)
        generation: Generation number to use
        duration: Simulation duration in seconds
        pheromone_threshold: Minimum pheromone value to record (normalized)
        render: Whether to render the simulation

    Returns:
        ShapleyDataset containing collected samples
    """
    # Load settings
    settings = MySettings()

    # Load trained model
    results_dir = Path("results") / experiment_id
    optimization_log_path = results_dir / "optimization_log.pkl"

    if not optimization_log_path.exists():
        raise FileNotFoundError(f"Optimization log not found: {optimization_log_path}")

    print(f"Loading trained model from: {optimization_log_path}")
    recorder = IndividualRecorder.load(str(optimization_log_path))

    if generation not in recorder.recs:
        available_gens = sorted(recorder.recs.keys())
        raise ValueError(
            f"Generation {generation} not found. Available generations: "
            f"{available_gens[0]}-{available_gens[-1]}"
        )

    # Get best individual from specified generation
    rec = recorder[generation]
    individual = rec.best_individual
    individual_id = f"{hex(id(individual))[2:]}_{generation}"

    print(f"Loaded individual from generation {generation}")
    print(f"  Fitness: {individual.get_fitness():.2f}")
    print(f"  Individual ID: {individual_id}")
    print(f"  Parameters shape: {individual.shape}")

    # Create simulator
    print(f"\nInitializing simulator...")
    print(f"  Duration: {duration}s")
    print(f"  Timestep: {settings.Simulation.TIME_STEP}s")
    print(f"  Total steps: {int(duration / settings.Simulation.TIME_STEP)}")

    simulator = Simulator(settings, individual, render=render)

    # Prepare data collection
    samples = []
    total_steps = int(duration / settings.Simulation.TIME_STEP)
    detection_count = 0

    print(f"\nStarting simulation with pheromone threshold: {pheromone_threshold}")
    print("Progress: ", end="", flush=True)

    # Run simulation and collect data
    for step in range(total_steps):
        # Display progress every 10%
        if step % (total_steps // 10) == 0:
            print(f"{100 * step // total_steps}%...", end="", flush=True)

        # Get current state before stepping
        # We need to check after timer.tick() is called in simulator.step()
        time_seconds = step * settings.Simulation.TIME_STEP

        # Step simulation
        simulator.step()

        # Get input/output data after step
        # The simulator only updates controller on timer tick intervals
        input_data = simulator.input_ndarray.copy()
        output_data = simulator.output_ndarray.copy()

        # Check each robot for pheromone detection
        for robot_idx in range(settings.Robot.NUM):
            pheromone_magnitude = input_data[robot_idx, 6]

            # Only record if pheromone is detected above threshold
            if pheromone_magnitude < pheromone_threshold:
                continue

            detection_count += 1

            # Get robot state
            robot = simulator.robot_values[robot_idx]

            # Create sample
            sample = ShapleyInputSample(
                timestep=step,
                time_seconds=time_seconds,
                robot_index=robot_idx,
                full_input=input_data[robot_idx].copy(),
                robot_sensor=input_data[robot_idx, 0:2].copy(),
                food_sensor=input_data[robot_idx, 2:4].copy(),
                direction_sensor=input_data[robot_idx, 4:6].copy(),
                pheromone_magnitude=float(input_data[robot_idx, 6]),
                pheromone_grad_forward=float(input_data[robot_idx, 7]),
                pheromone_grad_side=float(input_data[robot_idx, 8]),
                network_output=output_data[robot_idx].copy(),
                robot_position=robot.xpos.copy(),
                robot_direction=robot.xdirection.copy(),
            )

            samples.append(sample)

    print("100% Complete!")
    print(f"\nSimulation complete!")
    print(f"  Total detections: {detection_count}")
    print(f"  Samples collected: {len(samples)}")

    # Create dataset
    dataset = ShapleyDataset(
        experiment_id=experiment_id,
        generation=generation,
        individual_id=individual_id,
        simulation_duration=duration,
        pheromone_threshold=pheromone_threshold,
        collection_timestamp=datetime.now().isoformat(),
        samples=samples,
    )

    return dataset


def save_dataset(dataset: ShapleyDataset, output_dir: Path):
    """Save dataset to disk with metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as pickle
    pickle_path = output_dir / "samples.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"\nSaved dataset to: {pickle_path}")

    # Save metadata as text
    metadata_path = output_dir / "metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write(dataset.get_summary())
    print(f"Saved metadata to: {metadata_path}")

    # Save as dictionary (for potential JSON export)
    dict_path = output_dir / "samples_dict.pkl"
    with open(dict_path, 'wb') as f:
        pickle.dump(dataset.to_dict(), f)
    print(f"Saved dictionary format to: {dict_path}")

    print(f"\nDataset Summary:")
    print(dataset.get_summary())


def main():
    parser = argparse.ArgumentParser(
        description="Collect Shapley value data from trained ICAP models"
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="20251027-024639_7bb53c8b",
        help="Experiment ID (directory name in results/)"
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=499,
        help="Generation number to use (default: 499)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="Simulation duration in seconds (default: 300)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Minimum pheromone threshold for data collection (default: 0.01)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering (slower)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/{experiment_id}/shapley_data/)"
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        output_dir = Path("results") / args.experiment_id / "shapley_data"
    else:
        output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Shapley Value Data Collection")
    print("=" * 60)
    print(f"Experiment: {args.experiment_id}")
    print(f"Generation: {args.generation}")
    print(f"Duration: {args.duration}s")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Collect data
    dataset = collect_shapley_data(
        experiment_id=args.experiment_id,
        generation=args.generation,
        duration=args.duration,
        pheromone_threshold=args.threshold,
        render=args.render,
    )

    # Save dataset
    save_dataset(dataset, output_dir)

    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
