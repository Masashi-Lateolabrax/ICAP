"""
OPTICS Clustering Analysis for Shapley Input Samples

OPTICS-based clustering of robot sensor states (excluding pheromone features)
for situation-dependent Shapley value analysis.

OPTICS (Ordering Points To Identify the Clustering Structure) automatically
detects clusters of varying densities without requiring eps parameter tuning.

Usage:
    from src.interpretation.clustering.optics_clustering import cluster_sensor_states

    dataset = load_dataset("path/to/samples.pkl")
    result = cluster_sensor_states(dataset, min_samples=5, xi=0.05, min_cluster_size=0.05)
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pickle

from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from src.interpretation.data_collection.io_sample_definition import ShapleyDataset


@dataclass
class ClusteringResult:
    """Results from OPTICS clustering."""

    n_clusters: int              # Number of clusters found (excluding noise)
    labels: np.ndarray           # Cluster labels for each sample (-1 = noise)
    features: np.ndarray         # Normalized feature matrix (n_samples, 6)
    feature_names: list[str]     # Feature names
    reachability: np.ndarray     # Reachability distances for each point
    ordering: np.ndarray         # Cluster ordering indices

    def get_cluster_sizes(self) -> dict[int, int]:
        """Get number of samples in each cluster."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))


def extract_sensor_features(dataset: ShapleyDataset, normalize: bool = True) -> tuple[np.ndarray, list[str]]:
    """
    Extract 6D sensor features (excluding pheromone).

    Args:
        dataset: ShapleyDataset containing samples
        normalize: Apply standardization (zero mean, unit variance)

    Returns:
        features: (n_samples, 6) array
        feature_names: List of 6 feature names
    """
    if len(dataset.samples) == 0:
        raise ValueError("Dataset is empty")

    features_list = []
    for sample in dataset.samples:
        features_list.append([
            sample.robot_sensor[0],      # robot_sensor_x
            sample.robot_sensor[1],      # robot_sensor_y
            sample.food_sensor[0],       # food_sensor_x
            sample.food_sensor[1],       # food_sensor_y
            sample.direction_sensor[0],  # direction_sensor_x
            sample.direction_sensor[1],  # direction_sensor_y
        ])

    feature_names = [
        'robot_sensor_x', 'robot_sensor_y',
        'food_sensor_x', 'food_sensor_y',
        'direction_sensor_x', 'direction_sensor_y'
    ]

    features = np.array(features_list)

    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    return features, feature_names


def cluster_sensor_states(
    dataset: ShapleyDataset,
    min_samples: int = 5,
    xi: float = 0.05,
    min_cluster_size: float = 0.05
) -> ClusteringResult:
    """
    Cluster sensor states using OPTICS.

    Args:
        dataset: ShapleyDataset containing samples
        min_samples: Minimum number of samples in a neighborhood (core point threshold)
        xi: Minimum steepness for cluster extraction (0.0-1.0, default: 0.05)
        min_cluster_size: Minimum fraction of samples for a cluster (0.0-1.0, default: 0.05)

    Returns:
        ClusteringResult with cluster labels, reachability, and ordering metadata
    """
    # Extract features
    features, feature_names = extract_sensor_features(dataset, normalize=True)

    # Run OPTICS
    optics = OPTICS(
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size
    )
    labels = optics.fit_predict(features)

    # Count clusters (excluding noise label -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return ClusteringResult(
        n_clusters=n_clusters,
        labels=labels,
        features=features,
        feature_names=feature_names,
        reachability=optics.reachability_,
        ordering=optics.ordering_,
    )


def visualize_clusters(
    result: ClusteringResult,
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 8),
):
    """
    Visualize clusters in 2D using PCA.

    Args:
        result: ClusteringResult from cluster_sensor_states()
        save_path: Path to save figure (None to display)
        figsize: Figure size
    """
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(result.features)

    # Create plot
    plt.figure(figsize=figsize)

    # Plot each cluster
    unique_labels = np.unique(result.labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            marker = 'x'
            label_name = 'Noise'
            alpha = 0.3
        else:
            marker = 'o'
            label_name = f'Cluster {label}'
            alpha = 0.6

        mask = result.labels == label
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[color],
            marker=marker,
            label=label_name,
            alpha=alpha,
            s=50,
        )

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title(f'OPTICS Clustering Results (n={result.n_clusters} clusters)')
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_reachability_plot(
    result: ClusteringResult,
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (12, 6),
):
    """
    Visualize OPTICS reachability plot.

    The reachability plot shows the cluster structure by ordering points
    and plotting their reachability distances. Valleys indicate clusters.

    Args:
        result: ClusteringResult from cluster_sensor_states()
        save_path: Path to save figure (None to display)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Get colors for each cluster
    unique_labels = np.unique(result.labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: color for label, color in zip(unique_labels, colors)}

    # Reorder labels according to OPTICS ordering
    ordered_labels = result.labels[result.ordering]

    # Plot reachability distances with cluster colors
    for i, (reach, label) in enumerate(zip(result.reachability[result.ordering], ordered_labels)):
        if label == -1:
            color = 'lightgray'
        else:
            color = label_to_color[label]
        plt.bar(i, reach, width=1.0, color=color, edgecolor='none')

    plt.xlabel('Sample (OPTICS ordering)')
    plt.ylabel('Reachability Distance')
    plt.title(f'OPTICS Reachability Plot (n={result.n_clusters} clusters)')
    plt.grid(axis='y', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved reachability plot to: {save_path}")
    else:
        plt.show()

    plt.close()


def get_cluster_statistics(result: ClusteringResult) -> dict:
    """
    Get statistics for each cluster.

    Args:
        result: ClusteringResult from cluster_sensor_states()

    Returns:
        Dictionary with cluster statistics
    """
    stats = {}

    for cluster_id in range(result.n_clusters):
        mask = result.labels == cluster_id
        cluster_features = result.features[mask]

        stats[cluster_id] = {
            'size': int(np.sum(mask)),
            'feature_means': {
                name: float(np.mean(cluster_features[:, idx]))
                for idx, name in enumerate(result.feature_names)
            },
            'feature_stds': {
                name: float(np.std(cluster_features[:, idx]))
                for idx, name in enumerate(result.feature_names)
            },
        }

    return stats


def save_clustering_result(result: ClusteringResult, output_path: Path):
    """Save clustering result to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved clustering result to: {output_path}")


def load_clustering_result(input_path: Path) -> ClusteringResult:
    """Load clustering result from disk."""
    with open(input_path, 'rb') as f:
        result = pickle.load(f)
    print(f"Loaded clustering result from: {input_path}")
    return result