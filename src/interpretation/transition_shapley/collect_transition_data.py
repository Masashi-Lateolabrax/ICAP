"""
Collect transition data for Shapley value analysis of cluster transitions.

Usage:
    PYTHONPATH=. uv run --extra cpu src/interpretation/transition_shapley/collect_transition_data.py \
        --kmeans-model results/clustering/kmeans_model.joblib \
        --optimization-log results/optimization_log.pkl \
        --generation 499 \
        --seed 100 \
        --time-length 60.0 \
        --pheromone-threshold 0.1 \
        --output-path results/transition_shapley/transition_dataset.pkl

Note:
    - If --seed is specified, it overrides Individual.generation for RNG initialization
    - This ensures reproducible simulations matching specific debug_data
"""

import argparse
import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import joblib

from src.settings import MySettings
from src.config import Simulator
from src.interpretation.transition_shapley.data_structures import SimulatorState
from src.interpretation.transition_shapley import simulation
from framework.prelude import Individual


@dataclass
class TransitionMoment:
    """Data captured at a single transition moment."""
    time_step: int
    robot_indices: np.ndarray  # (n_transitioned,) indices of robots that transitioned
    current_features: np.ndarray  # (n_transitioned, 6) sensor features before transition
    actual_next_features: np.ndarray  # (n_transitioned, 6) features after actual step
    baseline_next_features: np.ndarray  # (n_transitioned, 6) features after baseline step
    current_clusters: np.ndarray  # (n_transitioned,) cluster labels before transition
    actual_next_clusters: np.ndarray  # (n_transitioned,) cluster labels after actual step
    baseline_next_clusters: np.ndarray  # (n_transitioned,) cluster labels after baseline step


@dataclass
class TransitionDataset:
    """Complete dataset of transition moments."""
    moments: list[TransitionMoment]
    n_robots: int
    n_clusters: int
    total_steps: int


class TransitionCollector:
    def __init__(self, settings, individual, kmeans, scaler, pheromone_threshold: float = 0.0):
        self.simulator = Simulator(settings, individual, render=False)
        self.simulator_baseline = Simulator(settings, individual, render=False)
        self.kmeans = kmeans
        self.scaler = scaler
        self.pheromone_threshold = pheromone_threshold

    def step(self):
        """Execute step and collect transition data if transitions occur.

        Returns:
            Tuple of (current_input, actual_next_input, baseline_next_input, transition_mask) where:
            - current_input: (n_robots, 9) array of inputs before transition
            - actual_next_input: (n_robots, 9) array of next inputs with actual pheromone
            - baseline_next_input: (n_robots, 9) array of next inputs with pheromone=0
            - transition_mask: (n_robots,) bool array indicating which robots transitioned

            Or None if no transitions occurred.
        """
        # Get current input
        current_input = self.simulator.input_ndarray.copy()  # (n_robots, 9)

        # Backup current state
        backup = SimulatorState(self.simulator)

        # Execute actual step and check for transitions (with pheromone filter)
        transitions = simulation.step(self.simulator, self.kmeans, self.scaler,
                                     self.pheromone_threshold)  # (n_robots,)

        if transitions.any():
            # At least one transition occurred! Get actual next input
            actual_next_input = self.simulator.input_ndarray.copy()  # (n_robots, 9)

            # Restore state to baseline simulator and execute baseline step
            backup.restore(self.simulator_baseline)
            baseline_next_input = simulation.step_baseline(self.simulator_baseline)  # (n_robots, 9)

            return current_input, actual_next_input, baseline_next_input, transitions

        return None


def collect_transitions(collector: TransitionCollector, max_steps: int) -> list[TransitionMoment]:
    """Run simulation and collect all transition moments.

    Args:
        collector: TransitionCollector instance
        max_steps: Maximum number of simulation steps

    Returns:
        List of TransitionMoment instances
    """
    moments = []

    for step_idx in range(max_steps):
        result = collector.step()

        if result is not None:
            current_input, actual_next_input, baseline_next_input, transitions = result
            robot_indices = np.where(transitions)[0]  # (n_transitioned,)

            # Extract features from inputs
            current_features = current_input[:, 0:6]  # (n_robots, 6)
            current_clusters = simulation.predict_cluster(
                collector.kmeans, collector.scaler, current_features
            )  # (n_robots,)

            moment = TransitionMoment(
                time_step=step_idx,
                robot_indices=robot_indices,
                current_features=current_features[robot_indices],  # (n_transitioned, 6)
                actual_next_features=actual_next_input[robot_indices, 0:6],  # (n_transitioned, 6)
                baseline_next_features=baseline_next_input[robot_indices, 0:6],  # (n_transitioned, 6)
                current_clusters=current_clusters[robot_indices],  # (n_transitioned,)
                actual_next_clusters=simulation.predict_cluster(
                    collector.kmeans, collector.scaler, actual_next_input[robot_indices, 0:6]
                ),  # (n_transitioned,)
                baseline_next_clusters=simulation.predict_cluster(
                    collector.kmeans, collector.scaler, baseline_next_input[robot_indices, 0:6]
                )  # (n_transitioned,)
            )

            moments.append(moment)

            if len(moments) % 10 == 0:
                print(f"  Step {step_idx}: {len(moments)} transitions collected")

        if (step_idx + 1) % 1000 == 0:
            print(f"  Progress: {step_idx + 1}/{max_steps} steps")

    return moments


def load_kmeans_model(model_path: Path):
    """Load KMeans model and scaler from joblib file."""
    kmeans, scaler = joblib.load(model_path)
    print(f"Loaded KMeans model from: {model_path}")
    print(f"  Number of clusters: {kmeans.n_clusters}")
    return kmeans, scaler


def load_individual_from_log(log_path: Path, generation: int) -> Individual:
    """Load Individual from optimization log."""
    from framework.types.utils import IndividualRecorder

    recorder = IndividualRecorder.load(str(log_path))
    rec = recorder[generation]
    individual = rec.best_individual

    print(f"Loaded Individual from generation {generation}")
    print(f"  Fitness: {individual.get_fitness()}")

    return individual


def main():
    parser = argparse.ArgumentParser(description='Collect transition data for Shapley analysis')
    parser.add_argument('--kmeans-model', type=Path, required=True,
                        help='Path to KMeans model file (joblib)')
    parser.add_argument('--optimization-log', type=Path, required=True,
                        help='Path to optimization log (pkl)')
    parser.add_argument('--generation', type=int, required=True,
                        help='Generation to analyze')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for simulation (default: use generation as seed)')
    parser.add_argument('--time-length', type=float, default=60.0,
                        help='Simulation time length in seconds (default: 60.0)')
    parser.add_argument('--pheromone-threshold', type=float, default=0.0,
                        help='Minimum pheromone value to collect transitions (default: 0.0)')
    parser.add_argument('--output-path', type=Path, required=True,
                        help='Path to save transition dataset (pkl)')

    args = parser.parse_args()

    # Load models and data
    kmeans, scaler = load_kmeans_model(args.kmeans_model)
    individual = load_individual_from_log(args.optimization_log, args.generation)

    # Override generation with seed if specified (for deterministic simulation)
    if args.seed is not None:
        print(f"Overriding Individual.generation ({individual.generation}) with seed: {args.seed}")
        individual._generation = args.seed  # Direct access to internal field
    else:
        print(f"Using Individual.generation as seed: {individual.generation}")

    # Create settings
    settings = MySettings()

    # Convert time to steps
    max_steps = int(args.time_length / settings.Simulation.TIME_STEP)
    print(f"Time length: {args.time_length}s = {max_steps} steps")

    # Create collector
    print(f"Pheromone threshold: {args.pheromone_threshold}")
    collector = TransitionCollector(settings, individual, kmeans, scaler,
                                   pheromone_threshold=args.pheromone_threshold)

    # Collect transitions
    print(f"\nCollecting transition data for {max_steps} steps...")
    moments = collect_transitions(collector, max_steps)

    # Create dataset
    dataset = TransitionDataset(
        moments=moments,
        n_robots=settings.Robot.NUM,
        n_clusters=kmeans.n_clusters,
        total_steps=max_steps
    )

    # Save dataset
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to: {args.output_path}")
    print(f"  Transition moments: {len(dataset.moments)}")
    print(f"  Total robots: {dataset.n_robots}")
    print(f"  Total clusters: {dataset.n_clusters}")


if __name__ == '__main__':
    main()