"""Shapley value calculation for robot sensor contributions.

This script calculates Shapley values to quantify the contribution of each
sensor type to the robot's decision-making when pheromone is detected.

Feature Groups:
    1. robot_sensor (2D): Preprocessed omni-sensor for other robots
    2. food_sensor (2D): Preprocessed omni-sensor for food items
    3. direction_sensor (2D): Direction to nest
    4. pheromone_magnitude (1D): Normalized pheromone concentration
    5. pheromone_grad_forward (1D): Pheromone gradient in forward direction
    6. pheromone_grad_side (1D): Pheromone gradient in sideways direction

Usage:
    PYTHONPATH=. uv run --extra cpu src/shapley_calculation.py \
        --data-dir results/20251027-024639_7bb53c8b/shapley_data \
        --n-samples 10000

Parameters:
    --data-dir: Directory containing collected Shapley data
    --n-samples: Number of samples to use for calculation (default: 10000)
    --output-dir: Custom output directory (default: same as data-dir)
"""

import argparse
import pickle
from pathlib import Path
from typing import Optional
import math
import numpy as np
import torch
import torch.nn as nn
from itertools import combinations
from dataclasses import dataclass
from datetime import datetime

from src.interpretation.data_collection.io_sample_definition import ShapleyDataset, ShapleyInputSample
from framework.types import IndividualRecorder
from config.controller import Controller


@dataclass
class ShapleyValues:
    """Shapley values for each feature group."""
    robot_sensor: float
    food_sensor: float
    direction_sensor: float
    pheromone_magnitude: float
    pheromone_grad_forward: float
    pheromone_grad_side: float

    def to_dict(self) -> dict:
        return {
            'robot_sensor': self.robot_sensor,
            'food_sensor': self.food_sensor,
            'direction_sensor': self.direction_sensor,
            'pheromone_magnitude': self.pheromone_magnitude,
            'pheromone_grad_forward': self.pheromone_grad_forward,
            'pheromone_grad_side': self.pheromone_grad_side,
        }


class FeatureGroupMasker:
    """Handles masking of feature groups for Shapley value calculation."""

    FEATURE_GROUPS = {
        'robot_sensor': (0, 2),
        'food_sensor': (2, 4),
        'direction_sensor': (4, 6),
        'pheromone_magnitude': (6, 7),
        'pheromone_grad_forward': (7, 8),
        'pheromone_grad_side': (8, 9),
    }

    @classmethod
    def get_group_names(cls) -> list[str]:
        """Get list of feature group names."""
        return list(cls.FEATURE_GROUPS.keys())

    @classmethod
    def mask_features(cls, input_vector: np.ndarray, mask_groups: set[str],
                     baseline: np.ndarray) -> np.ndarray:
        """Mask specified feature groups with baseline values.

        Args:
            input_vector: Original input vector (9,)
            mask_groups: Set of feature group names to mask
            baseline: Baseline values to use for masked features (9,)

        Returns:
            Masked input vector (9,)
        """
        masked = input_vector.copy()
        for group_name in mask_groups:
            start, end = cls.FEATURE_GROUPS[group_name]
            masked[start:end] = baseline[start:end]
        return masked


class ShapleyCalculator:
    """Calculates Shapley values using neural network model."""

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """Initialize calculator with trained model.

        Args:
            model: Trained neural network model
            device: Device to run calculations on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Run model prediction on input batch.

        Args:
            inputs: Input array of shape (batch_size, 9)

        Returns:
            Predictions of shape (batch_size, 3)
        """
        with torch.no_grad():
            inputs_tensor = torch.from_numpy(inputs).float().to(self.device)
            outputs = self.model(inputs_tensor)
            return outputs.cpu().numpy()

    def calculate_shapley_values(self, samples: list[ShapleyInputSample],
                                 n_samples: Optional[int] = None) -> ShapleyValues:
        """Calculate Shapley values for feature groups.

        Args:
            samples: List of collected samples
            n_samples: Number of samples to use (None = use all)

        Returns:
            Shapley values for each feature group
        """
        # Subsample if requested
        if n_samples is not None and n_samples < len(samples):
            indices = np.random.choice(len(samples), n_samples, replace=False)
            samples = [samples[i] for i in indices]

        print(f"Calculating Shapley values using {len(samples)} samples...")

        # Calculate baseline as mean input across all samples
        all_inputs = np.array([s.full_input for s in samples])
        baseline = np.mean(all_inputs, axis=0)

        print(f"Baseline values: {baseline}")

        # Get feature group names
        feature_names = FeatureGroupMasker.get_group_names()
        n_features = len(feature_names)

        # Initialize Shapley values
        shapley_values = {name: 0.0 for name in feature_names}

        # For each feature, calculate its marginal contribution across all coalitions
        for i, feature in enumerate(feature_names):
            print(f"\nCalculating Shapley value for: {feature}")

            # Get all other features (excluding current feature)
            other_features = [f for f in feature_names if f != feature]

            # Iterate over all possible coalitions of other features
            total_contribution = 0.0
            n_coalitions = 0

            for coalition_size in range(len(other_features) + 1):
                for coalition in combinations(other_features, coalition_size):
                    coalition_set = set(coalition)
                    coalition_with_feature = coalition_set | {feature}

                    # Calculate coalition weights
                    weight = 1.0 / (n_features * math.comb(n_features - 1, coalition_size))

                    # Prepare masked inputs for both coalitions
                    # Coalition without feature
                    masked_without = np.array([
                        FeatureGroupMasker.mask_features(
                            s.full_input,
                            set(feature_names) - coalition_set,
                            baseline
                        ) for s in samples
                    ])

                    # Coalition with feature
                    masked_with = np.array([
                        FeatureGroupMasker.mask_features(
                            s.full_input,
                            set(feature_names) - coalition_with_feature,
                            baseline
                        ) for s in samples
                    ])

                    # Get predictions
                    pred_without = self.predict(masked_without)
                    pred_with = self.predict(masked_with)

                    # Calculate marginal contribution (using L2 norm of output difference)
                    diff = np.linalg.norm(pred_with - pred_without, axis=1)
                    marginal_contribution = np.mean(diff)

                    # Weighted contribution
                    weighted_contribution = weight * marginal_contribution
                    total_contribution += weighted_contribution
                    n_coalitions += 1

            shapley_values[feature] = total_contribution
            print(f"  Shapley value: {total_contribution:.6f} (from {n_coalitions} coalitions)")

        return ShapleyValues(**shapley_values)


def load_dataset(data_dir: Path) -> ShapleyDataset:
    """Load Shapley dataset from directory.

    Args:
        data_dir: Directory containing shapley data files

    Returns:
        Loaded ShapleyDataset
    """
    # Try dict version first (smaller file size, faster loading)
    dict_path = data_dir / "samples_dict.pkl"
    samples_path = data_dir / "samples.pkl"

    if dict_path.exists():
        print(f"Loading dataset from: {dict_path}")
        with open(dict_path, 'rb') as f:
            data_dict = pickle.load(f)
        dataset = ShapleyDataset.from_dict(data_dict)
        print(f"Loaded {len(dataset)} samples")
        return dataset
    elif samples_path.exists():
        print(f"Loading dataset from: {samples_path}")
        with open(samples_path, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Loaded {len(dataset)} samples")
        return dataset
    else:
        raise FileNotFoundError(f"Data file not found in: {data_dir}")


def load_model(experiment_dir: Path, generation: int) -> nn.Module:
    """Load trained neural network model.

    Args:
        experiment_dir: Experiment directory
        generation: Generation number

    Returns:
        Loaded neural network model
    """
    optimization_log_path = experiment_dir / "optimization_log.pkl"

    if not optimization_log_path.exists():
        raise FileNotFoundError(f"Optimization log not found: {optimization_log_path}")

    print(f"Loading model from generation {generation}...")
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

    # Build neural network from individual parameters
    model = Controller(individual)
    print(f"Loaded model from generation {generation}")
    print(f"  Fitness: {individual.get_fitness():.2f}")
    print(f"  Parameters shape: {individual.shape}")
    print(f"  Model: {model}")

    return model


def save_results(shapley_values: ShapleyValues, output_dir: Path,
                dataset: ShapleyDataset, n_samples: int):
    """Save Shapley value calculation results.

    Args:
        shapley_values: Calculated Shapley values
        output_dir: Output directory
        dataset: Original dataset
        n_samples: Number of samples used
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as pickle
    results = {
        'shapley_values': shapley_values.to_dict(),
        'metadata': {
            'experiment_id': dataset.experiment_id,
            'generation': dataset.generation,
            'individual_id': dataset.individual_id,
            'n_samples_used': n_samples,
            'total_samples': len(dataset),
            'calculation_timestamp': datetime.now().isoformat(),
        }
    }

    pickle_path = output_dir / "shapley_results.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to: {pickle_path}")

    # Save as text
    text_path = output_dir / "shapley_results.txt"
    with open(text_path, 'w') as f:
        f.write("Shapley Value Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Experiment: {dataset.experiment_id}\n")
        f.write(f"Generation: {dataset.generation}\n")
        f.write(f"Individual: {dataset.individual_id}\n")
        f.write(f"Samples Used: {n_samples} / {len(dataset)}\n")
        f.write(f"Calculation Time: {results['metadata']['calculation_timestamp']}\n\n")

        f.write("Shapley Values by Feature Group:\n")
        f.write("-" * 50 + "\n")
        for name, value in shapley_values.to_dict().items():
            f.write(f"{name:30s}: {value:10.6f}\n")

        f.write("\n")
        total = sum(shapley_values.to_dict().values())
        f.write(f"{'Total':30s}: {total:10.6f}\n\n")

        f.write("Normalized Contributions (%):\n")
        f.write("-" * 50 + "\n")
        if total > 0:
            for name, value in shapley_values.to_dict().items():
                percentage = (value / total) * 100
                f.write(f"{name:30s}: {percentage:6.2f}%\n")

    print(f"Saved summary to: {text_path}")


def main():
    parser = argparse.ArgumentParser(description="Calculate Shapley values for robot sensors")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing Shapley data")
    parser.add_argument("--n-samples", type=int, default=10000,
                       help="Number of samples to use (default: 10000)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: same as data-dir)")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"],
                       help="Device to use for calculations")

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    experiment_dir = data_dir.parent
    output_dir = Path(args.output_dir) if args.output_dir else data_dir

    print(f"Data directory: {data_dir}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print()

    # Load dataset (with early subsampling if needed)
    print(f"Loading dataset with subsampling (n_samples={args.n_samples})...")
    dataset = load_dataset(data_dir)

    # Subsample early to save memory and time
    if args.n_samples < len(dataset.samples):
        print(f"Subsampling {args.n_samples} from {len(dataset.samples)} samples...")
        indices = np.random.choice(len(dataset.samples), args.n_samples, replace=False)
        dataset.samples = [dataset.samples[i] for i in indices]
        print(f"Subsampled dataset size: {len(dataset.samples)}")

    # Load model
    model = load_model(experiment_dir, dataset.generation)

    # Calculate Shapley values
    calculator = ShapleyCalculator(model, device=args.device)
    shapley_values = calculator.calculate_shapley_values(
        dataset.samples,
        n_samples=args.n_samples
    )

    # Display results
    print("\n" + "=" * 60)
    print("Shapley Value Results")
    print("=" * 60)
    for name, value in shapley_values.to_dict().items():
        print(f"{name:30s}: {value:10.6f}")

    total = sum(shapley_values.to_dict().values())
    print(f"{'Total':30s}: {total:10.6f}")

    print("\nNormalized Contributions:")
    if total > 0:
        for name, value in shapley_values.to_dict().items():
            percentage = (value / total) * 100
            print(f"{name:30s}: {percentage:6.2f}%")

    # Save results
    save_results(shapley_values, output_dir, dataset,
                min(args.n_samples, len(dataset)))

    print("\nCalculation complete!")


if __name__ == "__main__":
    main()
