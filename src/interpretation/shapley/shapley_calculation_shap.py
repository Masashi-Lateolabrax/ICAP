"""Shapley value calculation using SHAP library.

This script calculates SHAP values to quantify the contribution of each
sensor type to the robot's decision-making when pheromone is detected.

Feature Groups:
    1. robot_sensor (2D): Preprocessed omni-sensor for other robots
    2. food_sensor (2D): Preprocessed omni-sensor for food items
    3. direction_sensor (2D): Direction to nest
    4. pheromone_magnitude (1D): Normalized pheromone concentration
    5. pheromone_grad_forward (1D): Pheromone gradient in forward direction
    6. pheromone_grad_side (1D): Pheromone gradient in sideways direction

Usage:
    PYTHONPATH=. uv run --extra cpu src/shapley_calculation_shap.py \
        --data-dir results/20251027-024639_7bb53c8b/shapley_data_subset_10000 \
        --n-samples 1000 \
        --n-background 100

Parameters:
    --data-dir: Directory containing collected Shapley data
    --n-samples: Number of samples to explain (default: 1000)
    --n-background: Number of background samples for SHAP (default: 100)
    --output-dir: Custom output directory (default: same as data-dir)
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt

import shap

from src.interpretation.data_collection.io_sample_definition import ShapleyDataset, ShapleyInputSample
from framework.types import IndividualRecorder
from config.controller import Controller


# Feature group definitions
FEATURE_GROUPS = {
    'robot_sensor': (0, 2),
    'food_sensor': (2, 4),
    'direction_sensor': (4, 6),
    'pheromone_magnitude': (6, 7),
    'pheromone_grad_forward': (7, 8),
    'pheromone_grad_side': (8, 9),
}

FEATURE_NAMES = [
    'robot_x', 'robot_y',
    'food_x', 'food_y',
    'direction_x', 'direction_y',
    'pheromone_mag',
    'pheromone_grad_fwd',
    'pheromone_grad_side'
]


def load_dataset(data_dir: Path) -> ShapleyDataset:
    """Load Shapley dataset from directory."""
    dict_path = data_dir / "samples_dict.pkl"
    samples_path = data_dir / "samples.pkl"

    if dict_path.exists():
        print(f"Loading dataset from: {dict_path}")
        with open(dict_path, 'rb') as f:
            data_dict = pickle.load(f)
        dataset = ShapleyDataset.from_dict(data_dict)
    elif samples_path.exists():
        print(f"Loading dataset from: {samples_path}")
        with open(samples_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        raise FileNotFoundError(f"No data files found in: {data_dir}")

    print(f"Loaded {len(dataset)} samples")
    return dataset


def load_model(experiment_dir: Path, generation: int) -> nn.Module:
    """Load trained neural network model."""
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

    rec = recorder[generation]
    individual = rec.best_individual

    model = Controller(individual)
    model.eval()

    print(f"Loaded model from generation {generation}")
    print(f"  Fitness: {individual.get_fitness():.2f}")
    print(f"  Parameters shape: {individual.shape}")

    return model


def aggregate_shap_values_by_group(shap_values: np.ndarray) -> dict[str, float]:
    """Aggregate SHAP values by feature groups.

    Args:
        shap_values: SHAP values array of shape (n_samples, n_features)

    Returns:
        Dictionary mapping group names to aggregated SHAP values
    """
    # Take absolute values and average across samples
    abs_shap = np.abs(shap_values).mean(axis=0)

    group_values = {}
    for group_name, (start, end) in FEATURE_GROUPS.items():
        # Sum absolute SHAP values within group
        group_values[group_name] = abs_shap[start:end].sum()

    return group_values


def save_results(shap_values: np.ndarray,
                group_values: dict[str, float],
                output_dir: Path,
                dataset: ShapleyDataset,
                n_samples: int,
                samples_array: np.ndarray):
    """Save SHAP analysis results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw SHAP values
    shap_path = output_dir / "shap_values.npy"
    np.save(shap_path, shap_values)
    print(f"Saved SHAP values to: {shap_path}")

    # Save group aggregations
    results = {
        'group_shap_values': group_values,
        'metadata': {
            'experiment_id': dataset.experiment_id,
            'generation': dataset.generation,
            'individual_id': dataset.individual_id,
            'n_samples_used': n_samples,
            'total_samples': len(dataset),
            'calculation_timestamp': datetime.now().isoformat(),
        }
    }

    pickle_path = output_dir / "shap_results.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to: {pickle_path}")

    # Save text summary
    text_path = output_dir / "shap_results.txt"
    total = sum(group_values.values())

    with open(text_path, 'w') as f:
        f.write("SHAP Value Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Experiment: {dataset.experiment_id}\n")
        f.write(f"Generation: {dataset.generation}\n")
        f.write(f"Individual: {dataset.individual_id}\n")
        f.write(f"Samples Used: {n_samples}\n")
        f.write(f"Calculation Time: {results['metadata']['calculation_timestamp']}\n\n")

        f.write("SHAP Values by Feature Group:\n")
        f.write("-" * 50 + "\n")
        for name, value in sorted(group_values.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{name:30s}: {value:10.6f}\n")

        f.write("\n")
        f.write(f"{'Total':30s}: {total:10.6f}\n\n")

        f.write("Normalized Contributions (%):\n")
        f.write("-" * 50 + "\n")
        if total > 0:
            for name, value in sorted(group_values.items(), key=lambda x: x[1], reverse=True):
                percentage = (value / total) * 100
                f.write(f"{name:30s}: {percentage:6.2f}%\n")

    print(f"Saved summary to: {text_path}")

    # Create visualizations
    create_visualizations(shap_values, samples_array, group_values, output_dir)


def create_visualizations(shap_values: np.ndarray,
                         samples: np.ndarray,
                         group_values: dict[str, float],
                         output_dir: Path):
    """Create SHAP visualization plots."""

    # 1. Summary plot (beeswarm)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, samples, feature_names=FEATURE_NAMES, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_beeswarm.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved beeswarm plot to: {output_dir / 'shap_summary_beeswarm.png'}")

    # 2. Summary plot (bar)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, samples, feature_names=FEATURE_NAMES,
                     plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved bar plot to: {output_dir / 'shap_summary_bar.png'}")

    # 3. Feature group importance (custom bar plot)
    plt.figure(figsize=(10, 6))
    sorted_groups = sorted(group_values.items(), key=lambda x: x[1], reverse=True)
    names = [name for name, _ in sorted_groups]
    values = [value for _, value in sorted_groups]

    colors = ['#ff7f0e' if 'pheromone' in name else '#1f77b4' for name in names]

    plt.barh(names, values, color=colors)
    plt.xlabel('Mean |SHAP value|', fontsize=12)
    plt.title('Feature Group Importance (SHAP)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)

    # Add percentage labels
    total = sum(values)
    for i, (name, value) in enumerate(sorted_groups):
        percentage = (value / total) * 100
        plt.text(value, i, f'  {percentage:.1f}%',
                va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "shap_group_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved group importance plot to: {output_dir / 'shap_group_importance.png'}")


def main():
    parser = argparse.ArgumentParser(description="Calculate SHAP values for robot sensors")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing Shapley data")
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="Number of samples to explain (default: 1000)")
    parser.add_argument("--n-background", type=int, default=100,
                       help="Number of background samples for SHAP (default: 100)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: same as data-dir)")

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    experiment_dir = data_dir.parent
    output_dir = Path(args.output_dir) if args.output_dir else data_dir

    print(f"Data directory: {data_dir}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Output directory: {output_dir}")
    print(f"N samples: {args.n_samples}")
    print(f"N background: {args.n_background}")
    print()

    # Load dataset
    dataset = load_dataset(data_dir)

    # Subsample if needed
    if args.n_samples < len(dataset.samples):
        print(f"Subsampling {args.n_samples} from {len(dataset.samples)} samples...")
        indices = np.random.choice(len(dataset.samples), args.n_samples, replace=False)
        samples = [dataset.samples[i] for i in indices]
    else:
        samples = dataset.samples

    # Convert to numpy arrays
    X = np.array([s.full_input for s in samples])
    print(f"Input shape: {X.shape}")

    # Load model
    model = load_model(experiment_dir, dataset.generation)

    # Create model wrapper for SHAP
    def model_predict(x):
        """Wrapper for SHAP - returns model output."""
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float()
            outputs = model(x_tensor)
            return outputs.numpy()

    # Select background samples
    # Pheromone features (indices 6, 7, 8) are set to 0
    # Other features are sampled from data
    if args.n_background < len(X):
        background_indices = np.random.choice(len(X), args.n_background, replace=False)
        background = X[background_indices].copy()
    else:
        background = X.copy()

    # Set pheromone-related features to 0
    # pheromone_magnitude (6), pheromone_grad_forward (7), pheromone_grad_side (8)
    pheromone_indices = [6, 7, 8]
    background[:, pheromone_indices] = 0.0
    print(f"Background data: pheromone features set to 0, others sampled from data")

    print(f"\nCreating SHAP explainer with {len(background)} background samples...")
    explainer = shap.KernelExplainer(model_predict, background)

    # Calculate SHAP values
    print(f"Calculating SHAP values for {len(X)} samples...")
    print("This may take several minutes...")
    shap_values = explainer.shap_values(X)

    # Handle multi-output case (take first output - right wheel)
    if isinstance(shap_values, list):
        print(f"Multi-output model detected ({len(shap_values)} outputs)")
        print("Using first output (right wheel) for analysis")
        shap_values = shap_values[0]

    print(f"SHAP values shape: {shap_values.shape}")

    # Aggregate by feature groups
    print("\nAggregating SHAP values by feature groups...")
    group_values = aggregate_shap_values_by_group(shap_values)

    # Display results
    print("\n" + "=" * 60)
    print("SHAP Value Results (Feature Groups)")
    print("=" * 60)

    sorted_groups = sorted(group_values.items(), key=lambda x: x[1], reverse=True)
    for name, value in sorted_groups:
        print(f"{name:30s}: {value:10.6f}")

    total = sum(group_values.values())
    print(f"{'Total':30s}: {total:10.6f}")

    print("\nNormalized Contributions:")
    for name, value in sorted_groups:
        percentage = (value / total) * 100
        print(f"{name:30s}: {percentage:6.2f}%")

    # Save results
    save_results(shap_values, group_values, output_dir, dataset, len(X), X)

    print("\nCalculation complete!")


if __name__ == "__main__":
    main()
