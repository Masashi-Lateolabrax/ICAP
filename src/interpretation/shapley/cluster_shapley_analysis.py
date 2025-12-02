"""Calculate Shapley values for each cluster to understand situation-dependent feature importance.

This script:
1. Loads existing clustering_result.pkl (K=3 clustering)
2. Loads debug_data.pkl and matches samples to clusters
3. Calculates Shapley values independently for each cluster
4. Compares pheromone importance across different behavioral situations

Usage:
    PYTHONPATH=. uv run --extra cpu src/interpretation/shapley/cluster_shapley_analysis.py \
        --clustering-result results/.../clustering/clustering_result.pkl \
        --debug-data results/.../debug_data.pkl \
        --optimization-log results/20251027-024639_7bb53c8b/optimization_log.pkl \
        --generation 499 \
        --n-shapley-samples 1000 \
        --output-dir results/cluster_shapley_analysis
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
import matplotlib.pyplot as plt

from src.analysis_mod.structure.debug_data import DebugData
from src.interpretation.clustering.utils import RobotSensorSample, convert_debug_data_to_dataset_filtered
from src.interpretation.clustering.kmeans_clustering import (
    KMeansResult,
    get_cluster_statistics,
    load_clustering_result
)
from src.interpretation.data_collection.io_sample_definition import ShapleyInputSample
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
        """Mask specified feature groups with baseline values."""
        masked = input_vector.copy()
        for group_name in mask_groups:
            start, end = cls.FEATURE_GROUPS[group_name]
            masked[start:end] = baseline[start:end]
        return masked


class ShapleyCalculator:
    """Calculates Shapley values using neural network model."""

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Run model prediction on input batch."""
        with torch.no_grad():
            inputs_tensor = torch.from_numpy(inputs).float().to(self.device)
            outputs = self.model(inputs_tensor)
            return outputs.cpu().numpy()

    def calculate_shapley_values(self, samples: list[RobotSensorSample],
                                 n_samples: Optional[int] = None) -> ShapleyValues:
        """Calculate Shapley values for feature groups.

        Args:
            samples: List of RobotSensorSample (clustered samples)
            n_samples: Number of samples to use (None = use all)
        """
        # Subsample if requested
        if n_samples is not None and n_samples < len(samples):
            indices = np.random.choice(len(samples), n_samples, replace=False)
            samples = [samples[i] for i in indices]

        print(f"    Calculating Shapley values using {len(samples)} samples...")

        # Build full input vectors (9D) from RobotSensorSample
        all_inputs = []
        for s in samples:
            full_input = np.array([
                s.robot_sensor[0], s.robot_sensor[1],
                s.food_sensor[0], s.food_sensor[1],
                s.direction_sensor[0], s.direction_sensor[1],
                s.pheromone_magnitude,
                s.pheromone_grad_forward,
                s.pheromone_grad_side
            ])
            all_inputs.append(full_input)

        all_inputs = np.array(all_inputs)
        baseline = np.mean(all_inputs, axis=0)

        # Get feature group names
        feature_names = FeatureGroupMasker.get_group_names()
        n_features = len(feature_names)

        # Initialize Shapley values
        shapley_values = {name: 0.0 for name in feature_names}

        # Calculate Shapley value for each feature
        for i, feature in enumerate(feature_names):
            other_features = [f for f in feature_names if f != feature]

            total_contribution = 0.0
            n_coalitions = 0

            for coalition_size in range(len(other_features) + 1):
                for coalition in combinations(other_features, coalition_size):
                    coalition_set = set(coalition)
                    coalition_with_feature = coalition_set | {feature}

                    # Calculate coalition weights
                    weight = 1.0 / (n_features * math.comb(n_features - 1, coalition_size))

                    # Prepare masked inputs
                    masked_without = np.array([
                        FeatureGroupMasker.mask_features(
                            inp,
                            set(feature_names) - coalition_set,
                            baseline
                        ) for inp in all_inputs
                    ])

                    masked_with = np.array([
                        FeatureGroupMasker.mask_features(
                            inp,
                            set(feature_names) - coalition_with_feature,
                            baseline
                        ) for inp in all_inputs
                    ])

                    # Get predictions
                    pred_without = self.predict(masked_without)
                    pred_with = self.predict(masked_with)

                    # Calculate marginal contribution
                    diff = np.linalg.norm(pred_with - pred_without, axis=1)
                    marginal_contribution = np.mean(diff)

                    # Weighted contribution
                    weighted_contribution = weight * marginal_contribution
                    total_contribution += weighted_contribution
                    n_coalitions += 1

            shapley_values[feature] = total_contribution

        return ShapleyValues(**shapley_values)


def load_model(optimization_log_path: Path, generation: int) -> nn.Module:
    """Load trained neural network model."""
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

    # Get best individual
    individual = recorder.recs[generation][0]
    model = Controller()
    model.set_params(individual.param)
    model.eval()

    print(f"  Model loaded (fitness: {individual.get_fitness():.2f})")
    return model


def visualize_cluster_shapley_comparison(
    cluster_shapley_results: dict[int, ShapleyValues],
    cluster_stats: dict,
    output_path: Path
):
    """Visualize Shapley values comparison across clusters."""
    feature_names = ['robot_sensor', 'food_sensor', 'direction_sensor',
                     'pheromone_magnitude', 'pheromone_grad_forward', 'pheromone_grad_side']

    cluster_ids = sorted(cluster_shapley_results.keys())
    n_clusters = len(cluster_ids)

    # Extract values
    shapley_matrix = np.zeros((len(feature_names), n_clusters))
    for i, feature in enumerate(feature_names):
        for j, cluster_id in enumerate(cluster_ids):
            shapley_matrix[i, j] = cluster_shapley_results[cluster_id].to_dict()[feature]

    # Normalize to percentages
    shapley_percentage = 100 * shapley_matrix / shapley_matrix.sum(axis=0, keepdims=True)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))

    # Plot 1: Stacked bar chart (percentage)
    ax1 = plt.subplot(1, 3, 1)
    x = np.arange(n_clusters)
    width = 0.6
    colors = plt.cm.Set3(np.linspace(0, 1, len(feature_names)))

    bottom = np.zeros(n_clusters)
    for i, feature in enumerate(feature_names):
        ax1.bar(x, shapley_percentage[i], width, bottom=bottom,
               label=feature, color=colors[i])
        bottom += shapley_percentage[i]

    ax1.set_xlabel('Cluster ID', fontsize=12)
    ax1.set_ylabel('Feature Importance (%)', fontsize=12)
    ax1.set_title('Feature Importance per Cluster (Normalized)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Cluster {i}' for i in cluster_ids])
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Pheromone features comparison
    ax2 = plt.subplot(1, 3, 2)
    pheromone_features = ['pheromone_magnitude', 'pheromone_grad_forward', 'pheromone_grad_side']
    pheromone_indices = [feature_names.index(f) for f in pheromone_features]

    x_pheromone = np.arange(len(pheromone_features))
    bar_width = 0.8 / n_clusters

    for j, cluster_id in enumerate(cluster_ids):
        values = [shapley_percentage[i, j] for i in pheromone_indices]
        offset = (j - n_clusters/2 + 0.5) * bar_width
        ax2.bar(x_pheromone + offset, values, bar_width,
               label=f'Cluster {cluster_id}', alpha=0.8)

    ax2.set_xlabel('Pheromone Feature', fontsize=12)
    ax2.set_ylabel('Importance (%)', fontsize=12)
    ax2.set_title('Pheromone Feature Importance per Cluster', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pheromone)
    ax2.set_xticklabels(['Magnitude', 'Gradient Fwd', 'Gradient Side'], rotation=15)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Absolute Shapley values (not normalized)
    ax3 = plt.subplot(1, 3, 3)

    # Plot pheromone features only
    for j, cluster_id in enumerate(cluster_ids):
        values = [shapley_matrix[i, j] for i in pheromone_indices]
        offset = (j - n_clusters/2 + 0.5) * bar_width
        ax3.bar(x_pheromone + offset, values, bar_width,
               label=f'Cluster {cluster_id}', alpha=0.8)

    ax3.set_xlabel('Pheromone Feature', fontsize=12)
    ax3.set_ylabel('Absolute Shapley Value', fontsize=12)
    ax3.set_title('Pheromone Absolute Contribution per Cluster', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pheromone)
    ax3.set_xticklabels(['Magnitude', 'Gradient Fwd', 'Gradient Side'], rotation=15)
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved visualization to: {output_path}")
    plt.close()


def save_results(cluster_shapley_results: dict[int, ShapleyValues],
                cluster_stats: dict,
                clustering_result: KMeansResult,
                output_dir: Path,
                experiment_id: str,
                generation: int):
    """Save cluster Shapley analysis results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as pickle
    results = {
        'cluster_shapley_values': {cid: sv.to_dict() for cid, sv in cluster_shapley_results.items()},
        'cluster_statistics': cluster_stats,
        'clustering_metrics': {
            'n_clusters': clustering_result.n_clusters,
            'silhouette': clustering_result.silhouette,
            'davies_bouldin': clustering_result.davies_bouldin,
            'inertia': clustering_result.inertia,
        },
        'metadata': {
            'experiment_id': experiment_id,
            'generation': generation,
            'calculation_timestamp': datetime.now().isoformat(),
        }
    }

    pickle_path = output_dir / "cluster_shapley_results.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  Saved pickle results to: {pickle_path}")

    # Save as text
    text_path = output_dir / "cluster_shapley_results.txt"
    with open(text_path, 'w') as f:
        f.write("Cluster-based Shapley Value Analysis Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Experiment: {experiment_id}\n")
        f.write(f"Generation: {generation}\n")
        f.write(f"Number of clusters: {clustering_result.n_clusters}\n")
        f.write(f"Calculation time: {results['metadata']['calculation_timestamp']}\n\n")

        for cluster_id in sorted(cluster_shapley_results.keys()):
            sv = cluster_shapley_results[cluster_id]
            stats = cluster_stats[cluster_id]

            f.write(f"\nCluster {cluster_id} (n={stats['size']} samples)\n")
            f.write("-" * 70 + "\n")
            f.write("Shapley Values:\n")
            for name, value in sv.to_dict().items():
                f.write(f"  {name:30s}: {value:10.6f}\n")

            total = sum(sv.to_dict().values())
            f.write(f"  {'Total':30s}: {total:10.6f}\n\n")

            f.write("Normalized Contributions (%):\n")
            if total > 0:
                for name, value in sv.to_dict().items():
                    percentage = (value / total) * 100
                    f.write(f"  {name:30s}: {percentage:6.2f}%\n")
            f.write("\n")

    print(f"  Saved text summary to: {text_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Shapley values for each cluster"
    )
    parser.add_argument(
        '--clustering-result',
        type=str,
        required=True,
        help='Path to clustering_result.pkl (K=3)'
    )
    parser.add_argument(
        '--debug-data',
        type=str,
        required=True,
        help='Path to debug_data.pkl'
    )
    parser.add_argument(
        '--optimization-log',
        type=str,
        required=True,
        help='Path to optimization_log.pkl'
    )
    parser.add_argument(
        '--generation',
        type=int,
        required=True,
        help='Generation number to analyze'
    )
    parser.add_argument(
        '--n-shapley-samples',
        type=int,
        default=1000,
        help='Number of samples per cluster for Shapley calculation (default: 1000)'
    )
    parser.add_argument(
        '--pheromone-threshold',
        type=float,
        default=0.1,
        help='Minimum pheromone magnitude to include sample (default: 0.1)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: debug_data directory / cluster_shapley)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for Shapley calculation'
    )

    args = parser.parse_args()

    # Setup paths
    clustering_result_path = Path(args.clustering_result)
    debug_data_path = Path(args.debug_data)
    optimization_log_path = Path(args.optimization_log)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = debug_data_path.parent / "cluster_shapley"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract experiment_id from optimization_log path
    experiment_id = optimization_log_path.parent.name

    print(f"\n{'=' * 70}")
    print("Cluster-based Shapley Value Analysis")
    print(f"{'=' * 70}")
    print(f"Clustering result: {clustering_result_path}")
    print(f"Debug data: {debug_data_path}")
    print(f"Optimization log: {optimization_log_path}")
    print(f"Generation: {args.generation}")
    print(f"Shapley samples per cluster: {args.n_shapley_samples}")
    print(f"Pheromone threshold: {args.pheromone_threshold}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 70}\n")

    # Load clustering result
    print("Loading clustering result...")
    clustering_result = load_clustering_result(clustering_result_path)
    print(f"  Loaded K={clustering_result.n_clusters} clustering")
    print(f"    Silhouette score: {clustering_result.silhouette:.4f}")
    print(f"    Davies-Bouldin index: {clustering_result.davies_bouldin:.4f}")

    # Load debug data
    print("\nLoading debug data...")
    debug_data = DebugData.load(debug_data_path)
    print(f"  Loaded debug data with {len(debug_data.frames)} frames")

    # Convert to sensor samples
    print("\nConverting to sensor samples...")
    dataset = convert_debug_data_to_dataset_filtered(
        debug_data,
        experiment_id=experiment_id,
        generation=args.generation,
        pheromone_threshold=args.pheromone_threshold,
        sample_interval=1
    )
    print(f"  Converted {len(dataset)} samples (pheromone >= {args.pheromone_threshold})")

    # Verify sample count matches clustering result
    if len(dataset) != len(clustering_result.labels):
        raise ValueError(
            f"Sample count mismatch: dataset has {len(dataset)} samples, "
            f"but clustering result has {len(clustering_result.labels)} labels"
        )

    # Get cluster statistics
    cluster_stats = get_cluster_statistics(clustering_result)
    print(f"\n  Cluster sizes:")
    for cluster_id, stats in cluster_stats.items():
        print(f"    Cluster {cluster_id}: {stats['size']} samples")

    # Load model
    print(f"\nLoading neural network model...")
    model = load_model(optimization_log_path, args.generation)

    # Calculate Shapley values for each cluster
    print(f"\n{'=' * 70}")
    print("Calculating Shapley Values per Cluster")
    print(f"{'=' * 70}")

    calculator = ShapleyCalculator(model, device=args.device)
    cluster_shapley_results = {}

    for cluster_id in range(clustering_result.n_clusters):
        print(f"\nCluster {cluster_id}:")

        # Extract samples for this cluster
        cluster_mask = clustering_result.labels == cluster_id
        cluster_samples = [s for s, m in zip(dataset, cluster_mask) if m]
        print(f"  Total samples in cluster: {len(cluster_samples)}")

        # Calculate Shapley values
        shapley_values = calculator.calculate_shapley_values(
            cluster_samples,
            n_samples=min(args.n_shapley_samples, len(cluster_samples))
        )

        cluster_shapley_results[cluster_id] = shapley_values

        # Display results
        print(f"  Shapley values:")
        for name, value in shapley_values.to_dict().items():
            print(f"    {name:30s}: {value:10.6f}")

    # Visualize comparison
    print(f"\n{'=' * 70}")
    print("Generating Visualizations")
    print(f"{'=' * 70}")

    viz_path = output_dir / "cluster_shapley_comparison.png"
    visualize_cluster_shapley_comparison(
        cluster_shapley_results,
        cluster_stats,
        viz_path
    )

    # Save results
    print(f"\nSaving results...")
    save_results(
        cluster_shapley_results,
        cluster_stats,
        clustering_result,
        output_dir,
        experiment_id,
        args.generation
    )

    print(f"\n{'=' * 70}")
    print("Analysis Complete!")
    print(f"{'=' * 70}")
    print(f"Results saved to: {output_dir}")
    print(f"  - cluster_shapley_results.pkl")
    print(f"  - cluster_shapley_results.txt")
    print(f"  - cluster_shapley_comparison.png")


if __name__ == '__main__':
    main()