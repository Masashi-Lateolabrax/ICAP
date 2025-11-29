"""
K-means Clustering Analysis for Shapley Input Samples

K-means-based clustering of robot sensor states (excluding pheromone features)
for situation-dependent Shapley value analysis.

K-means requires specifying the number of clusters (k) in advance and partitions
data by minimizing within-cluster variance.

Usage:
    from src.interpretation.clustering.kmeans_clustering import cluster_sensor_states

    dataset = load_dataset("path/to/samples.pkl")
    result = cluster_sensor_states(dataset, n_clusters=10)
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pickle

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt

from src.interpretation.data_collection.io_sample_definition import ShapleyDataset


@dataclass
class KMeansResult:
    """Results from K-means clustering."""

    n_clusters: int              # Number of clusters specified
    labels: np.ndarray           # Cluster labels for each sample (0 to n_clusters-1)
    features: np.ndarray         # Normalized feature matrix (n_samples, 6)
    feature_names: list[str]     # Feature names
    cluster_centers: np.ndarray  # Cluster centroids (n_clusters, 6)
    inertia: float               # Sum of squared distances to closest cluster center
    silhouette: float            # Silhouette coefficient (-1 to 1, higher is better)
    davies_bouldin: float        # Davies-Bouldin index (lower is better)
    calinski_harabasz: float     # Calinski-Harabasz index (higher is better)

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
    n_clusters: int = 10,
    random_state: int = 42,
    max_iter: int = 300,
    n_init: int = 10
) -> KMeansResult:
    """
    Cluster sensor states using K-means.

    Args:
        dataset: ShapleyDataset containing samples
        n_clusters: Number of clusters to form
        random_state: Random seed for reproducibility
        max_iter: Maximum number of iterations
        n_init: Number of times to run k-means with different centroid seeds

    Returns:
        KMeansResult with cluster labels and quality metrics
    """
    # Extract features
    features, feature_names = extract_sensor_features(dataset, normalize=True)

    # Run K-means
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init
    )
    labels = kmeans.fit_predict(features)

    # Calculate quality metrics
    silhouette = silhouette_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)
    calinski_harabasz = calinski_harabasz_score(features, labels)

    return KMeansResult(
        n_clusters=n_clusters,
        labels=labels,
        features=features,
        feature_names=feature_names,
        cluster_centers=kmeans.cluster_centers_,
        inertia=kmeans.inertia_,
        silhouette=silhouette,
        davies_bouldin=davies_bouldin,
        calinski_harabasz=calinski_harabasz,
    )


def find_optimal_k(
    dataset: ShapleyDataset,
    k_range: range = range(2, 21),
    random_state: int = 42
) -> dict[int, dict[str, float]]:
    """
    Find optimal number of clusters using elbow method and quality metrics.

    Args:
        dataset: ShapleyDataset containing samples
        k_range: Range of k values to test
        random_state: Random seed for reproducibility

    Returns:
        Dictionary mapping k to metrics (inertia, silhouette, davies_bouldin, calinski_harabasz)
    """
    features, _ = extract_sensor_features(dataset, normalize=True)

    results = {}
    for k in k_range:
        print(f"Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(features)

        results[k] = {
            'inertia': kmeans.inertia_,
            'silhouette': silhouette_score(features, labels),
            'davies_bouldin': davies_bouldin_score(features, labels),
            'calinski_harabasz': calinski_harabasz_score(features, labels),
        }

    return results


def visualize_elbow_curve(
    k_metrics: dict[int, dict[str, float]],
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 10)
):
    """
    Visualize elbow curve and quality metrics for different k values.

    Args:
        k_metrics: Dictionary from find_optimal_k()
        save_path: Path to save figure (None to display)
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    k_values = sorted(k_metrics.keys())

    # Inertia (Elbow method)
    ax = axes[0, 0]
    inertias = [k_metrics[k]['inertia'] for k in k_values]
    ax.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Inertia (Within-cluster sum of squares)')
    ax.set_title('Elbow Method')
    ax.grid(alpha=0.3)

    # Silhouette Score (higher is better)
    ax = axes[0, 1]
    silhouettes = [k_metrics[k]['silhouette'] for k in k_values]
    ax.plot(k_values, silhouettes, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score (higher is better)')
    ax.grid(alpha=0.3)

    # Davies-Bouldin Index (lower is better)
    ax = axes[1, 0]
    db_scores = [k_metrics[k]['davies_bouldin'] for k in k_values]
    ax.plot(k_values, db_scores, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Davies-Bouldin Index')
    ax.set_title('Davies-Bouldin Index (lower is better)')
    ax.grid(alpha=0.3)

    # Calinski-Harabasz Index (higher is better)
    ax = axes[1, 1]
    ch_scores = [k_metrics[k]['calinski_harabasz'] for k in k_values]
    ax.plot(k_values, ch_scores, 'mo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Calinski-Harabasz Index')
    ax.set_title('Calinski-Harabasz Index (higher is better)')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved elbow curve to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_clusters(
    result: KMeansResult,
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 8),
):
    """
    Visualize clusters in 2D using PCA.

    Args:
        result: KMeansResult from cluster_sensor_states()
        save_path: Path to save figure (None to display)
        figsize: Figure size
    """
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(result.features)
    centers_2d = pca.transform(result.cluster_centers)

    # Create plot
    plt.figure(figsize=figsize)

    # Plot each cluster
    unique_labels = np.unique(result.labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = result.labels == label
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[color],
            marker='o',
            label=f'Cluster {label}',
            alpha=0.6,
            s=30,
        )

    # Plot cluster centers
    plt.scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        c='black',
        marker='X',
        s=200,
        edgecolors='white',
        linewidths=2,
        label='Centroids',
        zorder=10
    )

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title(f'K-means Clustering Results (k={result.n_clusters})\n'
              f'Silhouette: {result.silhouette:.3f}, Davies-Bouldin: {result.davies_bouldin:.3f}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def get_cluster_statistics(result: KMeansResult) -> dict:
    """
    Get statistics for each cluster.

    Args:
        result: KMeansResult from cluster_sensor_states()

    Returns:
        Dictionary with cluster statistics
    """
    stats = {}

    for cluster_id in range(result.n_clusters):
        mask = result.labels == cluster_id
        cluster_features = result.features[mask]

        stats[cluster_id] = {
            'size': int(np.sum(mask)),
            'centroid': {
                name: float(result.cluster_centers[cluster_id, idx])
                for idx, name in enumerate(result.feature_names)
            },
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


def save_clustering_result(result: KMeansResult, output_path: Path):
    """Save clustering result to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved clustering result to: {output_path}")


def load_clustering_result(input_path: Path) -> KMeansResult:
    """Load clustering result from disk."""
    with open(input_path, 'rb') as f:
        result = pickle.load(f)
    print(f"Loaded clustering result from: {input_path}")
    return result