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
        'robot_sensor[magnitude]', 'robot_sensor[angle]',
        'food_sensor[magnitude]', 'food_sensor[angle]',
        'direction_sensor[magnitude]', 'direction_sensor[angle]'
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


def analyze_temporal_continuity(result: KMeansResult, dataset: ShapleyDataset) -> dict:
    """
    Analyze temporal continuity within clusters to determine if clusters
    represent situations or time-periods.

    High temporal continuity (>70%) suggests clusters represent time-periods (bad).
    Low temporal continuity (<30%) suggests clusters represent situations (good).

    Args:
        result: KMeansResult from cluster_sensor_states()
        dataset: Original ShapleyDataset used for clustering

    Returns:
        Dictionary with temporal continuity analysis for each cluster
    """
    # Group samples by robot_index and cluster
    cluster_analysis = {}

    for cluster_id in range(result.n_clusters):
        # Get samples in this cluster
        cluster_mask = (result.labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_samples = [dataset.samples[i] for i in cluster_indices]

        # Group by robot
        robot_groups = {}
        for sample in cluster_samples:
            robot_idx = sample.robot_index
            if robot_idx not in robot_groups:
                robot_groups[robot_idx] = []
            robot_groups[robot_idx].append(sample)

        # Count consecutive timesteps within this cluster
        total_samples = len(cluster_samples)
        consecutive_count = 0

        for robot_idx, samples in robot_groups.items():
            # Sort by timestep
            samples_sorted = sorted(samples, key=lambda s: s.timestep)

            # Count consecutive pairs
            for i in range(len(samples_sorted) - 1):
                timestep_diff = samples_sorted[i + 1].timestep - samples_sorted[i].timestep
                if timestep_diff <= 1:
                    consecutive_count += 1

        # Calculate continuity ratio
        if total_samples > 1:
            continuity_ratio = consecutive_count / (total_samples - 1)
        else:
            continuity_ratio = 0.0

        cluster_analysis[cluster_id] = {
            'size': total_samples,
            'consecutive_pairs': consecutive_count,
            'total_pairs': total_samples - 1,
            'continuity_ratio': continuity_ratio,
        }

    return cluster_analysis


def visualize_temporal_continuity(
    continuity_analysis: dict,
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 6)
):
    """
    Visualize temporal continuity ratios for each cluster.

    Args:
        continuity_analysis: Output from analyze_temporal_continuity()
        save_path: Path to save figure (None to display)
        figsize: Figure size
    """
    cluster_ids = sorted(continuity_analysis.keys())
    ratios = [continuity_analysis[cid]['continuity_ratio'] for cid in cluster_ids]
    sizes = [continuity_analysis[cid]['size'] for cid in cluster_ids]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Bar plot of continuity ratios
    colors = ['red' if r > 0.7 else 'orange' if r > 0.3 else 'green' for r in ratios]
    ax1.bar(cluster_ids, ratios, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0.7, color='red', linestyle='--', linewidth=1, label='Time-period threshold (70%)')
    ax1.axhline(y=0.3, color='green', linestyle='--', linewidth=1, label='Situation threshold (30%)')
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Temporal Continuity Ratio')
    ax1.set_title('Temporal Continuity per Cluster')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1.0)

    # Scatter plot: size vs continuity
    ax2.scatter(sizes, ratios, s=100, alpha=0.6, edgecolors='black')
    for i, cid in enumerate(cluster_ids):
        ax2.annotate(f'C{cid}', (sizes[i], ratios[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax2.axhline(y=0.7, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=0.3, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Cluster Size (samples)')
    ax2.set_ylabel('Temporal Continuity Ratio')
    ax2.set_title('Cluster Size vs Temporal Continuity')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved temporal continuity visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_feature_distributions(
    result: KMeansResult,
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 10)
):
    """
    Visualize feature distributions for each cluster to identify
    if clusters represent distinct situations.

    Args:
        result: KMeansResult from cluster_sensor_states()
        save_path: Path to save figure (None to display)
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for feat_idx, feature_name in enumerate(result.feature_names):
        ax = axes[feat_idx]

        # Plot distribution for each cluster
        for cluster_id in range(result.n_clusters):
            mask = result.labels == cluster_id
            feature_values = result.features[mask, feat_idx]
            ax.hist(feature_values, bins=30, alpha=0.5, label=f'C{cluster_id}', density=True)

        ax.set_xlabel(feature_name)
        ax.set_ylabel('Density')
        ax.set_title(f'{feature_name} Distribution')
        ax.grid(alpha=0.3)
        if feat_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature distributions to: {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_cluster_distances(result: KMeansResult) -> dict:
    """
    Calculate pairwise distances between cluster centroids.

    Args:
        result: KMeansResult from cluster_sensor_states()

    Returns:
        Dictionary with distance matrix and statistics
    """
    n_clusters = result.n_clusters

    # Calculate pairwise distances between centroids
    distance_matrix = np.zeros((n_clusters, n_clusters))

    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(
                    result.cluster_centers[i] - result.cluster_centers[j]
                )

    # Find nearest cluster for each cluster
    nearest_clusters = {}
    for i in range(n_clusters):
        distances_from_i = distance_matrix[i].copy()
        distances_from_i[i] = np.inf  # Exclude self
        nearest_idx = np.argmin(distances_from_i)
        nearest_clusters[i] = {
            'nearest_cluster': int(nearest_idx),
            'distance': float(distances_from_i[nearest_idx])
        }

    # Calculate inter-cluster distance statistics (excluding diagonal)
    off_diagonal_distances = distance_matrix[np.triu_indices(n_clusters, k=1)]

    return {
        'distance_matrix': distance_matrix,
        'nearest_clusters': nearest_clusters,
        'min_distance': float(np.min(off_diagonal_distances)),
        'max_distance': float(np.max(off_diagonal_distances)),
        'mean_distance': float(np.mean(off_diagonal_distances)),
        'std_distance': float(np.std(off_diagonal_distances)),
    }


def analyze_temporal_statistics(result: KMeansResult, dataset: ShapleyDataset) -> dict:
    """
    Analyze temporal statistics for each cluster to check if clusters
    represent temporal segments.

    Args:
        result: KMeansResult from cluster_sensor_states()
        dataset: Original ShapleyDataset used for clustering

    Returns:
        Dictionary with temporal statistics per cluster
    """
    temporal_stats = {}

    for cluster_id in range(result.n_clusters):
        # Get samples in this cluster
        cluster_mask = (result.labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_samples = [dataset.samples[i] for i in cluster_indices]

        # Extract timesteps
        timesteps = np.array([s.timestep for s in cluster_samples])
        time_seconds = np.array([s.time_seconds for s in cluster_samples])

        temporal_stats[cluster_id] = {
            'timestep_mean': float(np.mean(timesteps)),
            'timestep_std': float(np.std(timesteps)),
            'timestep_min': int(np.min(timesteps)),
            'timestep_max': int(np.max(timesteps)),
            'time_seconds_mean': float(np.mean(time_seconds)),
            'time_seconds_std': float(np.std(time_seconds)),
            'time_seconds_min': float(np.min(time_seconds)),
            'time_seconds_max': float(np.max(time_seconds)),
        }

    return temporal_stats


def visualize_cluster_distance_matrix(
    distance_analysis: dict,
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 8)
):
    """
    Visualize cluster centroid distance matrix as heatmap.

    Args:
        distance_analysis: Output from analyze_cluster_distances()
        save_path: Path to save figure (None to display)
        figsize: Figure size
    """
    distance_matrix = distance_analysis['distance_matrix']
    n_clusters = distance_matrix.shape[0]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(distance_matrix, cmap='viridis', aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(n_clusters))
    ax.set_yticks(np.arange(n_clusters))
    ax.set_xticklabels([f'C{i}' for i in range(n_clusters)])
    ax.set_yticklabels([f'C{i}' for i in range(n_clusters)])

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Euclidean Distance', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                text = ax.text(j, i, f'{distance_matrix[i, j]:.2f}',
                             ha="center", va="center", color="white", fontsize=8)

    ax.set_title('Cluster Centroid Distance Matrix')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Cluster ID')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved cluster distance matrix to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_temporal_statistics(
    temporal_stats: dict,
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (12, 5)
):
    """
    Visualize temporal statistics for each cluster.

    Args:
        temporal_stats: Output from analyze_temporal_statistics()
        save_path: Path to save figure (None to display)
        figsize: Figure size
    """
    cluster_ids = sorted(temporal_stats.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Mean timestep with error bars
    means = [temporal_stats[cid]['timestep_mean'] for cid in cluster_ids]
    stds = [temporal_stats[cid]['timestep_std'] for cid in cluster_ids]
    mins = [temporal_stats[cid]['timestep_min'] for cid in cluster_ids]
    maxs = [temporal_stats[cid]['timestep_max'] for cid in cluster_ids]

    ax1.errorbar(cluster_ids, means, yerr=stds, fmt='o-', capsize=5,
                markersize=8, linewidth=2, label='Mean ± Std')
    ax1.scatter(cluster_ids, mins, marker='v', s=50, c='blue', alpha=0.6, label='Min')
    ax1.scatter(cluster_ids, maxs, marker='^', s=50, c='red', alpha=0.6, label='Max')
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Timestep')
    ax1.set_title('Temporal Distribution of Clusters')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Timestep ranges as horizontal bars
    for i, cid in enumerate(cluster_ids):
        min_t = temporal_stats[cid]['timestep_min']
        max_t = temporal_stats[cid]['timestep_max']
        ax2.barh(i, max_t - min_t, left=min_t, height=0.6,
                label=f'C{cid}', alpha=0.7)

    ax2.set_yticks(range(len(cluster_ids)))
    ax2.set_yticklabels([f'C{cid}' for cid in cluster_ids])
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Cluster ID')
    ax2.set_title('Temporal Ranges of Clusters')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved temporal statistics visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_original_sensor_values(result: KMeansResult, dataset: ShapleyDataset) -> dict:
    """
    Analyze original (non-standardized) sensor values for each cluster.

    Args:
        result: KMeansResult from cluster_sensor_states()
        dataset: Original ShapleyDataset used for clustering

    Returns:
        Dictionary with original sensor statistics:
        - 'global': Statistics for all data
        - cluster_id: Statistics for each cluster
          Each contains: mean, std, min, max, median for all 6 features
    """
    feature_names = [
        'robot_sensor[magnitude]',
        'robot_sensor[angle]',
        'food_sensor[magnitude]',
        'food_sensor[angle]',
        'direction_sensor[magnitude]',
        'direction_sensor[angle]'
    ]

    # Extract all original values
    def extract_values(samples):
        return {
            'robot_sensor[magnitude]': np.array([s.robot_sensor[0] for s in samples]),
            'robot_sensor[angle]': np.array([s.robot_sensor[1] for s in samples]),
            'food_sensor[magnitude]': np.array([s.food_sensor[0] for s in samples]),
            'food_sensor[angle]': np.array([s.food_sensor[1] for s in samples]),
            'direction_sensor[magnitude]': np.array([s.direction_sensor[0] for s in samples]),
            'direction_sensor[angle]': np.array([s.direction_sensor[1] for s in samples]),
        }

    def compute_stats(values_dict):
        stats = {}
        for feature_name in feature_names:
            values = values_dict[feature_name]
            stats[feature_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
            }
        return stats

    # Global statistics
    all_values = extract_values(dataset.samples)
    analysis = {
        'global': compute_stats(all_values)
    }

    # Per-cluster statistics
    for cluster_id in range(result.n_clusters):
        cluster_mask = (result.labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_samples = [dataset.samples[i] for i in cluster_indices]

        cluster_values = extract_values(cluster_samples)
        analysis[cluster_id] = compute_stats(cluster_values)

    return analysis


def visualize_original_sensor_distributions(
    original_stats: dict,
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 10)
):
    """
    Visualize original sensor value distributions across clusters.

    Args:
        original_stats: Output from analyze_original_sensor_values()
        save_path: Path to save figure (None to display)
        figsize: Figure size
    """
    feature_names = [
        'robot_sensor[magnitude]',
        'robot_sensor[angle]',
        'food_sensor[magnitude]',
        'food_sensor[angle]',
        'direction_sensor[magnitude]',
        'direction_sensor[angle]'
    ]

    # Get cluster IDs (excluding 'global')
    cluster_ids = sorted([k for k in original_stats.keys() if k != 'global'])

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for feat_idx, feature_name in enumerate(feature_names):
        ax = axes[feat_idx]

        # Prepare data for box plot
        data = []
        labels = []
        for cluster_id in cluster_ids:
            stats = original_stats[cluster_id][feature_name]
            # Create synthetic data points for box plot representation
            # Using mean, std, min, max, median
            data.append([
                stats['min'],
                stats['mean'] - stats['std'],
                stats['median'],
                stats['mean'] + stats['std'],
                stats['max']
            ])
            labels.append(f'C{cluster_id}')

        # Plot box-like representation
        positions = range(len(cluster_ids))
        for i, (cluster_id, cluster_data) in enumerate(zip(cluster_ids, data)):
            stats = original_stats[cluster_id][feature_name]
            # Plot as error bar
            ax.errorbar(i, stats['mean'], yerr=stats['std'],
                       fmt='o', capsize=5, markersize=6, alpha=0.7)
            # Plot min/max range
            ax.plot([i, i], [stats['min'], stats['max']], 'k-', alpha=0.3, linewidth=1)

        # Add global mean as reference line
        global_mean = original_stats['global'][feature_name]['mean']
        ax.axhline(y=global_mean, color='red', linestyle='--', linewidth=1,
                  alpha=0.5, label='Global mean')

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Value')
        ax.set_title(feature_name)
        ax.grid(alpha=0.3, axis='y')
        if feat_idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved original sensor distributions to: {save_path}")
    else:
        plt.show()

    plt.close()