"""Cluster analysis for Shapley data to separate different behavioral contexts.

This script clusters samples based on non-pheromone sensors to identify different
situations, then analyzes pheromone's role in each cluster.

Features used for clustering (excluding pheromone):
    - robot_sensor (2D): Other robots
    - food_sensor (2D): Food items
    - direction_sensor (2D): Direction to nest

Pheromone features analyzed per cluster:
    - pheromone_magnitude (1D)
    - pheromone_grad_forward (1D)
    - pheromone_grad_side (1D)

Usage:
    PYTHONPATH=. uv run --extra cpu src/cluster_analysis.py \
        --data-dir results/20251027-024639_7bb53c8b/shapley_data_subset_10000 \
        --n-clusters 4
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import shap

from src.interpretation.data_collection.io_sample_definition import ShapleyDataset
from framework.types import IndividualRecorder
from config.controller import Controller
import torch.nn as nn


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

    individual = recorder.recs[generation][0]
    model = Controller()
    model.set_params(individual.param)
    model.eval()

    return model


def extract_features(dataset: ShapleyDataset) -> tuple[np.ndarray, np.ndarray]:
    """Extract features for clustering and pheromone analysis.

    Returns:
        clustering_features: (N, 6) array of non-pheromone sensors
        pheromone_features: (N, 3) array of pheromone sensors
    """
    N = len(dataset)
    clustering_features = np.zeros((N, 6))
    pheromone_features = np.zeros((N, 3))

    for i, sample in enumerate(dataset.samples):
        # Non-pheromone sensors for clustering
        clustering_features[i, 0:2] = sample.robot_sensor
        clustering_features[i, 2:4] = sample.food_sensor
        clustering_features[i, 4:6] = sample.direction_sensor

        # Pheromone features for analysis
        pheromone_features[i, 0] = sample.pheromone_magnitude
        pheromone_features[i, 1] = sample.pheromone_grad_forward
        pheromone_features[i, 2] = sample.pheromone_grad_side

    return clustering_features, pheromone_features


def perform_clustering(features: np.ndarray, n_clusters: int) -> tuple[np.ndarray, KMeans]:
    """Perform K-means clustering on features.

    Returns:
        labels: Cluster assignments
        kmeans: Fitted KMeans model
    """
    print(f"\nPerforming K-means clustering with {n_clusters} clusters...")

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    print(f"Clustering complete. Inertia: {kmeans.inertia_:.2f}")

    return labels, kmeans


def analyze_clusters(dataset: ShapleyDataset, labels: np.ndarray,
                     clustering_features: np.ndarray, pheromone_features: np.ndarray,
                     output_dir: Path):
    """Analyze characteristics of each cluster."""
    n_clusters = len(np.unique(labels))

    print("\n" + "="*70)
    print("CLUSTER ANALYSIS")
    print("="*70)

    cluster_stats = []

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        n_samples = np.sum(mask)

        # Clustering features (non-pheromone)
        cluster_features = clustering_features[mask]
        robot_sensor = cluster_features[:, 0:2]
        food_sensor = cluster_features[:, 2:4]
        direction_sensor = cluster_features[:, 4:6]

        # Pheromone features
        cluster_pheromone = pheromone_features[mask]

        print(f"\nCluster {cluster_id} (n={n_samples}):")
        print("-" * 70)

        # Calculate means and stds
        print("Non-pheromone sensors (used for clustering):")
        print(f"  Robot sensor:     mean={np.mean(np.linalg.norm(robot_sensor, axis=1)):.3f}, "
              f"std={np.std(np.linalg.norm(robot_sensor, axis=1)):.3f}")
        print(f"  Food sensor:      mean={np.mean(np.linalg.norm(food_sensor, axis=1)):.3f}, "
              f"std={np.std(np.linalg.norm(food_sensor, axis=1)):.3f}")
        print(f"  Direction sensor: mean={np.mean(np.linalg.norm(direction_sensor, axis=1)):.3f}, "
              f"std={np.std(np.linalg.norm(direction_sensor, axis=1)):.3f}")

        print("\nPheromone sensors (to be analyzed):")
        print(f"  Magnitude:        mean={np.mean(cluster_pheromone[:, 0]):.4f}, "
              f"std={np.std(cluster_pheromone[:, 0]):.4f}")
        print(f"  Grad forward:     mean={np.mean(cluster_pheromone[:, 1]):.4f}, "
              f"std={np.std(cluster_pheromone[:, 1]):.4f}")
        print(f"  Grad side:        mean={np.mean(cluster_pheromone[:, 2]):.4f}, "
              f"std={np.std(cluster_pheromone[:, 2]):.4f}")

        cluster_stats.append({
            'cluster_id': cluster_id,
            'n_samples': n_samples,
            'robot_sensor_mean': np.mean(np.linalg.norm(robot_sensor, axis=1)),
            'food_sensor_mean': np.mean(np.linalg.norm(food_sensor, axis=1)),
            'direction_sensor_mean': np.mean(np.linalg.norm(direction_sensor, axis=1)),
            'pheromone_magnitude_mean': np.mean(cluster_pheromone[:, 0]),
            'pheromone_grad_forward_mean': np.mean(cluster_pheromone[:, 1]),
            'pheromone_grad_side_mean': np.mean(cluster_pheromone[:, 2]),
        })

    # Save cluster statistics
    stats_path = output_dir / "cluster_statistics.txt"
    with open(stats_path, 'w') as f:
        f.write("Cluster Statistics\n")
        f.write("="*70 + "\n\n")
        for stats in cluster_stats:
            f.write(f"Cluster {stats['cluster_id']} (n={stats['n_samples']})\n")
            f.write(f"  Robot sensor:     {stats['robot_sensor_mean']:.3f}\n")
            f.write(f"  Food sensor:      {stats['food_sensor_mean']:.3f}\n")
            f.write(f"  Direction sensor: {stats['direction_sensor_mean']:.3f}\n")
            f.write(f"  Pheromone mag:    {stats['pheromone_magnitude_mean']:.4f}\n")
            f.write(f"  Pheromone fwd:    {stats['pheromone_grad_forward_mean']:.4f}\n")
            f.write(f"  Pheromone side:   {stats['pheromone_grad_side_mean']:.4f}\n")
            f.write("\n")

    print(f"\nCluster statistics saved to: {stats_path}")

    return cluster_stats


def calculate_shap_per_cluster(dataset: ShapleyDataset, model: nn.Module, labels: np.ndarray,
                                n_background: int, output_dir: Path):
    """Calculate SHAP values for each cluster separately."""
    import torch

    n_clusters = len(np.unique(labels))

    # Prepare model wrapper
    def model_predict(X):
        """Wrapper for SHAP that returns only wheel outputs."""
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            outputs = model(X_tensor)
        # Return only wheel outputs [right, left]
        return outputs[:, :2].numpy()

    print("\n" + "="*70)
    print("SHAP VALUE CALCULATION PER CLUSTER")
    print("="*70)

    all_cluster_results = {}

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        n_samples = np.sum(mask)

        print(f"\nCluster {cluster_id} (n={n_samples}):")
        print("-" * 70)

        # Extract cluster samples
        cluster_samples = [s for s, m in zip(dataset.samples, mask) if m]
        X_cluster = np.array([s.full_input for s in cluster_samples])

        # Prepare background data (pheromone features set to 0)
        if n_background < len(X_cluster):
            background_indices = np.random.choice(len(X_cluster), n_background, replace=False)
            background = X_cluster[background_indices].copy()
        else:
            background = X_cluster.copy()

        # Set pheromone features to 0
        pheromone_indices = [6, 7, 8]
        background[:, pheromone_indices] = 0.0

        print(f"Using {len(background)} background samples with pheromone=0")

        # Calculate SHAP values
        explainer = shap.KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(X_cluster)

        # For multi-output, take mean across outputs
        if isinstance(shap_values, list):
            shap_values_combined = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_values_combined = np.abs(shap_values)

        # Calculate group importance
        feature_groups = {
            'robot_sensor': (0, 2),
            'food_sensor': (2, 4),
            'direction_sensor': (4, 6),
            'pheromone_magnitude': (6, 7),
            'pheromone_grad_forward': (7, 8),
            'pheromone_grad_side': (8, 9),
        }

        group_importance = {}
        for group_name, (start, end) in feature_groups.items():
            group_importance[group_name] = np.mean(shap_values_combined[:, start:end])

        total_importance = sum(group_importance.values())

        print("\nSHAP Values by Feature Group:")
        for group_name, importance in sorted(group_importance.items(),
                                            key=lambda x: x[1], reverse=True):
            percentage = 100 * importance / total_importance
            print(f"  {group_name:25s}: {importance:8.6f} ({percentage:5.2f}%)")

        all_cluster_results[cluster_id] = {
            'shap_values': shap_values_combined,
            'group_importance': group_importance,
            'n_samples': n_samples,
        }

    # Save results
    results_path = output_dir / "cluster_shap_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(all_cluster_results, f)

    print(f"\nCluster SHAP results saved to: {results_path}")

    return all_cluster_results


def visualize_results(cluster_results: dict, output_dir: Path):
    """Visualize cluster analysis results."""
    n_clusters = len(cluster_results)

    # Extract group importance for each cluster
    group_names = list(cluster_results[0]['group_importance'].keys())

    # Prepare data for plotting
    cluster_ids = sorted(cluster_results.keys())
    importance_matrix = np.zeros((len(group_names), n_clusters))

    for i, group_name in enumerate(group_names):
        for j, cluster_id in enumerate(cluster_ids):
            importance_matrix[i, j] = cluster_results[cluster_id]['group_importance'][group_name]

    # Normalize to percentages
    importance_percentage = 100 * importance_matrix / importance_matrix.sum(axis=0, keepdims=True)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Stacked bar chart
    x = np.arange(n_clusters)
    width = 0.6
    colors = plt.cm.Set3(np.linspace(0, 1, len(group_names)))

    bottom = np.zeros(n_clusters)
    for i, group_name in enumerate(group_names):
        ax1.bar(x, importance_percentage[i], width, bottom=bottom,
               label=group_name, color=colors[i])
        bottom += importance_percentage[i]

    ax1.set_xlabel('Cluster ID', fontsize=12)
    ax1.set_ylabel('Feature Importance (%)', fontsize=12)
    ax1.set_title('Feature Group Importance per Cluster', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Cluster {i}' for i in cluster_ids])
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Pheromone importance comparison
    pheromone_groups = ['pheromone_magnitude', 'pheromone_grad_forward', 'pheromone_grad_side']
    pheromone_indices = [group_names.index(g) for g in pheromone_groups]

    x_pheromone = np.arange(len(pheromone_groups))
    bar_width = 0.8 / n_clusters

    for j, cluster_id in enumerate(cluster_ids):
        values = [importance_percentage[i, j] for i in pheromone_indices]
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

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / "cluster_shap_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {fig_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Cluster analysis for Shapley data")
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing Shapley data')
    parser.add_argument('--n-clusters', type=int, default=4,
                       help='Number of clusters (default: 4)')
    parser.add_argument('--n-background', type=int, default=100,
                       help='Number of background samples for SHAP (default: 100)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: same as data-dir)')

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract experiment info from data_dir
    experiment_dir = data_dir.parent

    # Load dataset
    dataset = load_dataset(data_dir)

    # Load metadata to get generation
    metadata_path = data_dir / "metadata.txt"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            for line in f:
                if line.startswith('Generation:'):
                    generation = int(line.split(':')[1].strip())
                    break
    else:
        generation = dataset.generation

    # Load model
    model = load_model(experiment_dir, generation)

    # Extract features
    clustering_features, pheromone_features = extract_features(dataset)

    # Perform clustering
    labels, kmeans = perform_clustering(clustering_features, args.n_clusters)

    # Analyze clusters
    cluster_stats = analyze_clusters(dataset, labels, clustering_features,
                                     pheromone_features, output_dir)

    # Calculate SHAP values per cluster
    cluster_results = calculate_shap_per_cluster(dataset, model, labels,
                                                 args.n_background, output_dir)

    # Visualize results
    visualize_results(cluster_results, output_dir)

    print("\n" + "="*70)
    print("CLUSTER ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()