"""
K-means Clustering Analysis for Robot Sensor Data

K-means-based clustering of robot sensor states (excluding pheromone features)
for behavioral pattern analysis.

K-means requires specifying the number of clusters (k) in advance and partitions
data by minimizing within-cluster variance.

Usage:
    # Find optimal number of clusters
    PYTHONPATH=. uv run --extra cpu src/interpretation/clustering/kmeans_clustering.py \
        --data-path results/analysis/debug_data.pkl \
        --find-optimal-k \
        --k-range 2-20 \
        --output-dir results/clustering

    # Run clustering with known k
    PYTHONPATH=. uv run --extra cpu src/interpretation/clustering/kmeans_clustering.py \
        --data-path results/analysis/debug_data.pkl \
        --n-clusters 9 \
        --output-dir results/clustering

    # With video generation
    PYTHONPATH=. uv run --extra cpu src/interpretation/clustering/kmeans_clustering.py \
        --data-path results/analysis/debug_data.pkl \
        --n-clusters 9 \
        --create-video \
        --video-with-stats \
        --output-dir results/clustering

    # With filtering
    PYTHONPATH=. uv run --extra cpu src/interpretation/clustering/kmeans_clustering.py \
        --data-path results/analysis/debug_data.pkl \
        --n-clusters 9 \
        --pheromone-threshold 0.1 \
        --sample-interval 2 \
        --output-dir results/clustering
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pickle
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt

from src.interpretation.clustering.utils import RobotSensorSample
from src.settings import MySettings


# Default clustering parameters
DEFAULT_N_CLUSTERS = 10
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_ITER = 300
DEFAULT_N_INIT = 10
DEFAULT_K_RANGE_MIN = 2
DEFAULT_K_RANGE_MAX = 20

# Visualization parameters
DEFAULT_FIGURE_SIZE_ELBOW = (14, 10)
DEFAULT_FIGURE_SIZE_CLUSTERS = (10, 8)
DEFAULT_FIGURE_SIZE_FEATURES = (14, 10)
DEFAULT_FIGURE_SIZE_TIMELINE = (16, 10)
DEFAULT_DPI = 300
DEFAULT_ALPHA = 0.3
DEFAULT_MARKER_SIZE_SAMPLE = 30
DEFAULT_MARKER_SIZE_CENTROID = 200
DEFAULT_LINE_WIDTH_BORDER = 2
DEFAULT_ZORDER_CENTROID = 10

# Output file names
KMEANS_MODEL_FILENAME = "kmeans_model.joblib"  # joblib format: (KMeans, StandardScaler) tuple
CLUSTERING_METADATA_FILENAME = "clustering_metadata.pkl"  # pickle format
CLUSTER_STATS_FILENAME = "cluster_stats.txt"
CLUSTERS_2D_FILENAME = "clusters_2d.png"
SENSOR_DISTRIBUTIONS_FILENAME = "sensor_distributions.png"
CLUSTER_TIMELINE_FILENAME = "cluster_timeline.png"
CLUSTER_TIMELINE_CSV_FILENAME = "cluster_timeline.csv"
CLUSTER_ANIMATION_FILENAME = "cluster_animation.mp4"
CLUSTER_ANIMATION_WITH_STATS_FILENAME = "cluster_animation_with_stats.mp4"
OPTIMAL_K_METRICS_FILENAME = "optimal_k_metrics.txt"
ELBOW_CURVE_FILENAME = "elbow_curve.png"


@dataclass
class ClusteringMetadata:
    """Metadata for clustering results (separate from KMeans model)."""

    n_clusters: int              # Number of clusters specified
    labels: np.ndarray           # Cluster labels for each sample (0 to n_clusters-1)
    features: np.ndarray         # Normalized feature matrix (n_samples, 6)
    feature_names: list[str]     # Feature names
    inertia: float               # Sum of squared distances to closest cluster center
    silhouette: float            # Silhouette coefficient (-1 to 1, higher is better)
    davies_bouldin: float        # Davies-Bouldin index (lower is better)
    calinski_harabasz: float     # Calinski-Harabasz index (higher is better)
    pheromone_threshold: float   # Pheromone threshold used for clustering

    def get_cluster_sizes(self) -> dict[int, int]:
        """Get number of samples in each cluster."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))


def extract_sensor_features(dataset: list[RobotSensorSample]) -> tuple[np.ndarray, list[str], StandardScaler]:
    """
    Extract and normalize 6D sensor features (excluding pheromone).

    Args:
        dataset: List of RobotSensorSample containing samples

    Returns:
        features: (n_samples, 6) normalized array
        feature_names: List of 6 feature names
        scaler: Fitted StandardScaler for transforming new samples
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    features_list = []
    for sample in dataset:
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

    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    return features_normalized, feature_names, scaler


def cluster_sensor_states(
    dataset: list[RobotSensorSample],
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_iter: int = DEFAULT_MAX_ITER,
    n_init: int = DEFAULT_N_INIT,
    pheromone_threshold: float = 0.0
) -> tuple[KMeans, ClusteringMetadata, StandardScaler]:
    """
    Cluster sensor states using K-means.

    Args:
        dataset: List of RobotSensorSample containing samples
        n_clusters: Number of clusters to form
        random_state: Random seed for reproducibility
        max_iter: Maximum number of iterations
        n_init: Number of times to run k-means with different centroid seeds
        pheromone_threshold: Pheromone threshold used for filtering dataset

    Returns:
        Tuple of (fitted KMeans model, ClusteringMetadata, fitted StandardScaler)
    """
    # Extract and normalize features
    features, feature_names, scaler = extract_sensor_features(dataset)

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

    # Create metadata
    metadata = ClusteringMetadata(
        n_clusters=n_clusters,
        labels=labels,
        features=features,
        feature_names=feature_names,
        inertia=kmeans.inertia_,
        silhouette=silhouette,
        davies_bouldin=davies_bouldin,
        calinski_harabasz=calinski_harabasz,
        pheromone_threshold=pheromone_threshold,
    )

    return kmeans, metadata, scaler


def find_optimal_k(
    dataset: list[RobotSensorSample],
    k_range: range = range(DEFAULT_K_RANGE_MIN, DEFAULT_K_RANGE_MAX + 1),
    random_state: int = DEFAULT_RANDOM_STATE
) -> dict[int, dict[str, float]]:
    """
    Find optimal number of clusters using elbow method and quality metrics.

    Args:
        dataset: List of RobotSensorSample containing samples
        k_range: Range of k values to test
        random_state: Random seed for reproducibility

    Returns:
        Dictionary mapping k to metrics (inertia, silhouette, davies_bouldin, calinski_harabasz)
    """
    features, _, _ = extract_sensor_features(dataset)

    results = {}
    for k in k_range:
        print(f"Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=DEFAULT_N_INIT)
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
    figsize: tuple[int, int] = DEFAULT_FIGURE_SIZE_ELBOW
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
    ax.grid(alpha=DEFAULT_ALPHA)

    # Silhouette Score (higher is better)
    ax = axes[0, 1]
    silhouettes = [k_metrics[k]['silhouette'] for k in k_values]
    ax.plot(k_values, silhouettes, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score (higher is better)')
    ax.grid(alpha=DEFAULT_ALPHA)

    # Davies-Bouldin Index (lower is better)
    ax = axes[1, 0]
    db_scores = [k_metrics[k]['davies_bouldin'] for k in k_values]
    ax.plot(k_values, db_scores, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Davies-Bouldin Index')
    ax.set_title('Davies-Bouldin Index (lower is better)')
    ax.grid(alpha=DEFAULT_ALPHA)

    # Calinski-Harabasz Index (higher is better)
    ax = axes[1, 1]
    ch_scores = [k_metrics[k]['calinski_harabasz'] for k in k_values]
    ax.plot(k_values, ch_scores, 'mo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Calinski-Harabasz Index')
    ax.set_title('Calinski-Harabasz Index (higher is better)')
    ax.grid(alpha=DEFAULT_ALPHA)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"Saved elbow curve to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_clusters(
    kmeans_model: KMeans,
    metadata: ClusteringMetadata,
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = DEFAULT_FIGURE_SIZE_CLUSTERS,
):
    """
    Visualize clusters in 2D using PCA.

    Args:
        kmeans_model: Fitted KMeans model
        metadata: ClusteringMetadata from cluster_sensor_states()
        save_path: Path to save figure (None to display)
        figsize: Figure size
    """
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(metadata.features)
    centers_2d = pca.transform(kmeans_model.cluster_centers_)

    # Create plot
    plt.figure(figsize=figsize)

    # Plot each cluster
    unique_labels = np.unique(metadata.labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = metadata.labels == label
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[color],
            marker='o',
            label=f'Cluster {label}',
            alpha=DEFAULT_ALPHA * 2,  # 0.6
            s=DEFAULT_MARKER_SIZE_SAMPLE,
        )

    # Plot cluster centers
    plt.scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        c='black',
        marker='X',
        s=DEFAULT_MARKER_SIZE_CENTROID,
        edgecolors='white',
        linewidths=DEFAULT_LINE_WIDTH_BORDER,
        label='Centroids',
        zorder=DEFAULT_ZORDER_CENTROID
    )

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title(f'K-means Clustering Results (k={metadata.n_clusters})\n'
              f'Silhouette: {metadata.silhouette:.3f}, Davies-Bouldin: {metadata.davies_bouldin:.3f}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def get_cluster_statistics(kmeans_model: KMeans, metadata: ClusteringMetadata) -> dict:
    """
    Get statistics for each cluster.

    Args:
        kmeans_model: Fitted KMeans model
        metadata: ClusteringMetadata from cluster_sensor_states()

    Returns:
        Dictionary with cluster statistics
    """
    stats = {}

    for cluster_id in range(metadata.n_clusters):
        mask = metadata.labels == cluster_id
        cluster_features = metadata.features[mask]

        stats[cluster_id] = {
            'size': int(np.sum(mask)),
            'centroid': {
                name: float(kmeans_model.cluster_centers_[cluster_id, idx])
                for idx, name in enumerate(metadata.feature_names)
            },
            'feature_means': {
                name: float(np.mean(cluster_features[:, idx]))
                for idx, name in enumerate(metadata.feature_names)
            },
            'feature_stds': {
                name: float(np.std(cluster_features[:, idx]))
                for idx, name in enumerate(metadata.feature_names)
            },
        }

    return stats


def save_clustering_artifacts(
    kmeans_model: KMeans,
    metadata: ClusteringMetadata,
    scaler: StandardScaler,
    output_dir: Path
):
    """Save KMeans model, scaler, and metadata separately.

    Args:
        kmeans_model: Fitted KMeans model
        metadata: ClusteringMetadata object
        scaler: Fitted StandardScaler for feature normalization
        output_dir: Directory to save artifacts

    Saves:
        - kmeans_model.joblib: (KMeans model, StandardScaler) tuple (joblib format)
        - clustering_metadata.pkl: ClusteringMetadata object (pickle format)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save KMeans model and scaler together using joblib
    model_path = output_dir / KMEANS_MODEL_FILENAME
    joblib.dump((kmeans_model, scaler), model_path)
    print(f"Saved KMeans model and scaler to: {model_path}")

    # Save metadata using pickle
    metadata_path = output_dir / CLUSTERING_METADATA_FILENAME
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved clustering metadata to: {metadata_path}")


def load_kmeans_model(model_path: Path) -> tuple[KMeans, StandardScaler]:
    """Load KMeans model and scaler from disk.

    Args:
        model_path: Path to kmeans_model.joblib

    Returns:
        Tuple of (fitted KMeans model, fitted StandardScaler)

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If file is not valid format
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"KMeans model file not found: {model_path}\n"
            f"Expected format: joblib-serialized (KMeans, StandardScaler) tuple (.joblib)"
        )

    try:
        loaded = joblib.load(model_path)
    except Exception as e:
        raise ValueError(
            f"Failed to load from {model_path}\n"
            f"Expected: joblib-serialized (KMeans, StandardScaler) tuple\n"
            f"Error: {e}"
        )

    # Check if it's a tuple (new format)
    if isinstance(loaded, tuple) and len(loaded) == 2:
        model, scaler = loaded
        if not isinstance(model, KMeans):
            raise ValueError(f"First element is not KMeans: {type(model)}")
        if not isinstance(scaler, StandardScaler):
            raise ValueError(f"Second element is not StandardScaler: {type(scaler)}")
        print(f"Loaded KMeans model and scaler from: {model_path}")
        return model, scaler

    # Old format: just KMeans model
    elif isinstance(loaded, KMeans):
        raise ValueError(
            f"Old format detected (KMeans only, no StandardScaler)\n"
            f"Please re-run clustering to generate new format files.\n"
            f"File: {model_path}"
        )

    else:
        raise ValueError(
            f"Unexpected format: {type(loaded)}\n"
            f"Expected: (KMeans, StandardScaler) tuple\n"
            f"File: {model_path}"
        )


def load_clustering_metadata(metadata_path: Path) -> ClusteringMetadata:
    """Load clustering metadata from disk.

    Args:
        metadata_path: Path to clustering_metadata.pkl

    Returns:
        ClusteringMetadata object
    """
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    print(f"Loaded clustering metadata from: {metadata_path}")
    return metadata


def analyze_temporal_continuity(metadata: ClusteringMetadata, dataset: list[RobotSensorSample]) -> dict:
    """
    Analyze temporal continuity within clusters to determine if clusters
    represent situations or time-periods.

    High temporal continuity (>70%) suggests clusters represent time-periods (bad).
    Low temporal continuity (<30%) suggests clusters represent situations (good).

    Args:
        metadata: ClusteringMetadata from cluster_sensor_states()
        dataset: Original list of RobotSensorSample used for clustering

    Returns:
        Dictionary with temporal continuity analysis for each cluster
    """
    # Group samples by robot_index and cluster
    cluster_analysis = {}

    for cluster_id in range(metadata.n_clusters):
        # Get samples in this cluster
        cluster_mask = (metadata.labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_samples = [dataset[i] for i in cluster_indices]

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
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"Saved temporal continuity visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_feature_distributions(
    metadata: ClusteringMetadata,
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = DEFAULT_FIGURE_SIZE_FEATURES
):
    """
    Visualize feature distributions for each cluster to identify
    if clusters represent distinct situations.

    Args:
        metadata: ClusteringMetadata from cluster_sensor_states()
        save_path: Path to save figure (None to display)
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for feat_idx, feature_name in enumerate(metadata.feature_names):
        ax = axes[feat_idx]

        # Plot distribution for each cluster
        for cluster_id in range(metadata.n_clusters):
            mask = metadata.labels == cluster_id
            feature_values = metadata.features[mask, feat_idx]
            ax.hist(feature_values, bins=30, alpha=0.5, label=f'C{cluster_id}', density=True)

        ax.set_xlabel(feature_name)
        ax.set_ylabel('Density')
        ax.set_title(f'{feature_name} Distribution')
        ax.grid(alpha=DEFAULT_ALPHA)
        if feat_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"Saved feature distributions to: {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_cluster_distances(kmeans_model: KMeans, metadata: ClusteringMetadata) -> dict:
    """
    Calculate pairwise distances between cluster centroids.

    Args:
        kmeans_model: Fitted KMeans model
        metadata: ClusteringMetadata from cluster_sensor_states()

    Returns:
        Dictionary with distance matrix and statistics
    """
    n_clusters = metadata.n_clusters

    # Calculate pairwise distances between centroids
    distance_matrix = np.zeros((n_clusters, n_clusters))

    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(
                    kmeans_model.cluster_centers_[i] - kmeans_model.cluster_centers_[j]
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


def analyze_temporal_statistics(metadata: ClusteringMetadata, dataset: list[RobotSensorSample]) -> dict:
    """
    Analyze temporal statistics for each cluster to check if clusters
    represent temporal segments.

    Args:
        metadata: ClusteringMetadata from cluster_sensor_states()
        dataset: Original list of RobotSensorSample used for clustering

    Returns:
        Dictionary with temporal statistics per cluster
    """
    temporal_stats = {}

    for cluster_id in range(metadata.n_clusters):
        # Get samples in this cluster
        cluster_mask = (metadata.labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_samples = [dataset[i] for i in cluster_indices]

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
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
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
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"Saved temporal statistics visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_original_sensor_values(metadata: ClusteringMetadata, dataset: list[RobotSensorSample]) -> dict[str | int, dict]:
    """
    Analyze original (non-standardized) sensor values for each cluster.

    Args:
        metadata: ClusteringMetadata from cluster_sensor_states()
        dataset: Original list of RobotSensorSample used for clustering

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
    all_values = extract_values(dataset)
    analysis = {
        'global': compute_stats(all_values)
    }

    # Per-cluster statistics
    for cluster_id in range(metadata.n_clusters):
        cluster_mask = (metadata.labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_samples = [dataset[i] for i in cluster_indices]

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
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"Saved original sensor distributions to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_cluster_timeline(
    metadata: ClusteringMetadata,
    dataset: list[RobotSensorSample],
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = DEFAULT_FIGURE_SIZE_TIMELINE
):
    """
    Visualize which cluster each robot belongs to over time.

    Creates a timeline plot showing cluster membership for each robot
    throughout the simulation.

    Args:
        metadata: ClusteringMetadata from cluster_sensor_states()
        dataset: Original list of RobotSensorSample used for clustering
        save_path: Path to save figure (None to display)
        figsize: Figure size
    """
    # Extract data for each sample
    data = []
    for idx, sample in enumerate(dataset):
        data.append({
            'robot_index': sample.robot_index,
            'timestep': sample.timestep,
            'time_seconds': sample.time_seconds,
            'cluster': metadata.labels[idx]
        })

    # Get unique robot indices
    robot_indices = sorted(set(d['robot_index'] for d in data))
    n_robots = len(robot_indices)

    # Create color map for clusters
    n_clusters = metadata.n_clusters
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])

    # Plot 1: Timeline for each robot
    for robot_idx in robot_indices:
        robot_data = [d for d in data if d['robot_index'] == robot_idx]
        robot_data_sorted = sorted(robot_data, key=lambda x: x['timestep'])

        timesteps = [d['timestep'] for d in robot_data_sorted]
        clusters = [d['cluster'] for d in robot_data_sorted]

        # Plot as scatter points colored by cluster
        for cluster_id in range(n_clusters):
            cluster_mask = np.array(clusters) == cluster_id
            if np.any(cluster_mask):
                ax1.scatter(
                    np.array(timesteps)[cluster_mask],
                    [robot_idx] * np.sum(cluster_mask),
                    c=[colors[cluster_id]],
                    marker='s',
                    s=10,
                    alpha=0.8,
                    label=f'C{cluster_id}' if robot_idx == robot_indices[0] else None
                )

    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Robot Index')
    ax1.set_title('Cluster Membership Timeline per Robot')
    ax1.set_yticks(robot_indices)
    ax1.grid(alpha=0.3, axis='x')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, fontsize=8)

    # Plot 2: Cluster distribution over time (stacked area)
    # Group by timestep and count cluster membership
    timesteps_unique = sorted(set(d['timestep'] for d in data))
    cluster_counts = {cid: [] for cid in range(n_clusters)}

    for ts in timesteps_unique:
        ts_data = [d for d in data if d['timestep'] == ts]
        ts_clusters = [d['cluster'] for d in ts_data]

        for cid in range(n_clusters):
            count = sum(1 for c in ts_clusters if c == cid)
            cluster_counts[cid].append(count)

    # Create stacked area plot
    ax2.stackplot(
        timesteps_unique,
        *[cluster_counts[cid] for cid in range(n_clusters)],
        colors=colors,
        labels=[f'C{cid}' for cid in range(n_clusters)],
        alpha=0.8
    )

    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Number of Robots')
    ax2.set_title('Cluster Distribution Over Time')
    ax2.grid(alpha=0.3, axis='both')
    ax2.set_ylim(0, n_robots)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        print(f"Saved cluster timeline visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def export_cluster_timeline_data(
    metadata: ClusteringMetadata,
    dataset: list[RobotSensorSample],
    output_path: Path
):
    """
    Export cluster membership timeline data to CSV file.

    Args:
        metadata: ClusteringMetadata from cluster_sensor_states()
        dataset: Original list of RobotSensorSample used for clustering
        output_path: Path to save CSV file
    """
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_index', 'robot_index', 'timestep', 'time_seconds', 'cluster_id'])

        for idx, sample in enumerate(dataset):
            writer.writerow([
                idx,
                sample.robot_index,
                sample.timestep,
                sample.time_seconds,
                metadata.labels[idx]
            ])

    print(f"Saved cluster timeline data to: {output_path}")


# ==============================================================================
# CLI Interface
# ==============================================================================

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from src.analysis_mod.structure.debug_data import DebugData
    from src.interpretation.clustering.utils import convert_debug_data_to_dataset_filtered
    from src.interpretation.clustering.cluster_animation import (
        create_cluster_animation,
        create_cluster_animation_with_stats,
    )

    parser = argparse.ArgumentParser(
        description='Run K-means clustering on sensor states from DebugData'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to DebugData pickle file'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=None,
        help='Number of clusters for K-means (required unless --find-optimal-k is used)'
    )
    parser.add_argument(
        '--find-optimal-k',
        action='store_true',
        help='Find optimal number of clusters instead of clustering'
    )
    parser.add_argument(
        '--k-range',
        type=str,
        default='2-20',
        help='Range of k values to test for optimal k (format: "min-max", default: 2-20)'
    )
    parser.add_argument(
        '--experiment-id',
        type=str,
        default='debug_analysis',
        help='Experiment identifier (default: debug_analysis)'
    )
    parser.add_argument(
        '--generation',
        type=int,
        default=0,
        help='Generation number (default: 0)'
    )
    parser.add_argument(
        '--pheromone-threshold',
        type=float,
        default=0.0,
        help='Only include samples with pheromone >= threshold (default: 0.0)'
    )
    parser.add_argument(
        '--sample-interval',
        type=int,
        default=1,
        help='Only include every Nth timestep (default: 1 = all frames)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: same as data-path directory)'
    )
    parser.add_argument(
        '--create-video',
        action='store_true',
        help='Create cluster animation video'
    )
    parser.add_argument(
        '--video-with-stats',
        action='store_true',
        help='Create video with statistics panel'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.find_optimal_k and args.n_clusters is None:
        parser.error("--n-clusters is required unless --find-optimal-k is used")

    # Load DebugData and convert to samples list
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"\n{'=' * 60}")
    print("Loading and Converting DebugData to Sensor Samples")
    print(f"{'=' * 60}")

    debug_data = DebugData.load(data_path)

    # Create filtered dataset for clustering (only pheromone >= threshold)
    dataset = convert_debug_data_to_dataset_filtered(
        debug_data,
        experiment_id=args.experiment_id,
        generation=args.generation,
        pheromone_threshold=args.pheromone_threshold,
        sample_interval=args.sample_interval
    )

    print(f"Conversion complete!")
    print(f"  Filtered samples (for clustering): {len(dataset)}")

    # Set output directory
    if args.output_dir is None:
        output_dir = data_path.parent / "clustering"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find optimal k if requested
    if args.find_optimal_k:
        print(f"\n{'=' * 60}")
        print("Finding Optimal Number of Clusters")
        print(f"{'=' * 60}")

        # Parse k range
        k_min, k_max = map(int, args.k_range.split('-'))
        k_range = range(k_min, k_max + 1)
        print(f"Testing k from {k_min} to {k_max}")
        print(f"{'=' * 60}\n")

        k_metrics = find_optimal_k(dataset, k_range=k_range)

        # Save metrics
        metrics_path = output_dir / OPTIMAL_K_METRICS_FILENAME
        with open(metrics_path, 'w') as f:
            f.write("Optimal K Search Results\n")
            f.write("========================\n\n")
            for k in sorted(k_metrics.keys()):
                f.write(f"k={k}:\n")
                f.write(f"  Inertia: {k_metrics[k]['inertia']:.4f}\n")
                f.write(f"  Silhouette: {k_metrics[k]['silhouette']:.4f}\n")
                f.write(f"  Davies-Bouldin: {k_metrics[k]['davies_bouldin']:.4f}\n")
                f.write(f"  Calinski-Harabasz: {k_metrics[k]['calinski_harabasz']:.4f}\n\n")

        print(f"\nSaved metrics to: {metrics_path}")

        # Visualize elbow curve
        elbow_path = output_dir / ELBOW_CURVE_FILENAME
        visualize_elbow_curve(k_metrics, save_path=elbow_path)

        print(f"\n{'=' * 60}")
        print("Optimal K Search Complete!")
        print(f"{'=' * 60}")
        print(f"Results saved to: {output_dir}")
        print(f"  Metrics: {metrics_path.name}")
        print(f"  Elbow curve: {elbow_path.name}")
        print("\nRecommendations:")
        print("  - Check elbow_curve.png for visual analysis")
        print("  - Look for the 'elbow' in the inertia plot")
        print("  - Higher silhouette score is better (closer to 1)")
        print("  - Lower Davies-Bouldin index is better")
        print("  - Higher Calinski-Harabasz index is better")
        exit(0)

    # Validate n_clusters
    if args.n_clusters is None:
        raise ValueError("--n-clusters is required unless --find-optimal-k is used")

    print(f"\n{'=' * 60}")
    print("K-means Clustering Configuration")
    print(f"{'=' * 60}")
    print(f"Number of clusters: {args.n_clusters}")
    print(f"Features: 6D sensor states (robot, food, direction)")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")

    # Perform clustering
    print("\nRunning K-means clustering...")
    kmeans_model, metadata, scaler = cluster_sensor_states(
        dataset,
        n_clusters=args.n_clusters,
        pheromone_threshold=args.pheromone_threshold
    )

    print(f"\nClustering complete!")
    print(f"  Inertia: {metadata.inertia:.4f}")
    print(f"  Silhouette score: {metadata.silhouette:.4f}")

    # Print cluster sizes
    unique, counts = np.unique(metadata.labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    print("\nCluster sizes:")
    for cluster_id, size in sorted(cluster_sizes.items()):
        print(f"  Cluster {cluster_id}: {size} samples ({100*size/len(metadata.labels):.1f}%)")

    # Get cluster statistics
    print("\nComputing cluster statistics...")
    stats = get_cluster_statistics(kmeans_model, metadata)

    # Save statistics
    stats_path = output_dir / CLUSTER_STATS_FILENAME
    with open(stats_path, 'w') as f:
        f.write(f"K-means Clustering Results\n")
        f.write(f"==========================\n")
        f.write(f"Number of clusters: {args.n_clusters}\n")
        f.write(f"Inertia: {metadata.inertia:.4f}\n")
        f.write(f"Silhouette score: {metadata.silhouette:.4f}\n\n")

        for cluster_id, cluster_stats in stats.items():
            f.write(f"\nCluster {cluster_id}:\n")
            f.write(f"  Size: {cluster_stats['size']}\n")
            f.write(f"  Feature means:\n")
            for feature_name, mean_val in cluster_stats['feature_means'].items():
                f.write(f"    {feature_name}: {mean_val:.4f}\n")

    print(f"Saved statistics to: {stats_path}")

    # Visualize clusters
    print("\nGenerating visualizations...")
    viz_2d_path = output_dir / CLUSTERS_2D_FILENAME
    visualize_clusters(kmeans_model, metadata, save_path=viz_2d_path)

    # Analyze and visualize original sensor distributions
    sensor_dist_path = output_dir / SENSOR_DISTRIBUTIONS_FILENAME
    original_stats = analyze_original_sensor_values(metadata, dataset)
    visualize_original_sensor_distributions(original_stats, save_path=sensor_dist_path)

    # Visualize cluster timeline
    timeline_path = output_dir / CLUSTER_TIMELINE_FILENAME
    visualize_cluster_timeline(metadata, dataset, save_path=timeline_path)

    # Export cluster timeline data
    timeline_csv_path = output_dir / CLUSTER_TIMELINE_CSV_FILENAME
    export_cluster_timeline_data(metadata, dataset, timeline_csv_path)

    # Save clustering results
    save_clustering_artifacts(kmeans_model, metadata, scaler, output_dir)

    # Create video if requested
    video_path = None
    if args.create_video:
        print(f"\n{'=' * 60}")
        print("Generating Cluster Animation Video")
        print(f"{'=' * 60}")

        # Use the scaler from clustering (already fitted on the dataset)
        if args.video_with_stats:
            video_path = output_dir / CLUSTER_ANIMATION_WITH_STATS_FILENAME
            create_cluster_animation_with_stats(
                kmeans_model, scaler, metadata, debug_data, video_path,
                world_width=MySettings.Simulation.WORLD_WIDTH,
                world_height=MySettings.Simulation.WORLD_HEIGHT
            )
        else:
            video_path = output_dir / CLUSTER_ANIMATION_FILENAME
            create_cluster_animation(
                kmeans_model, scaler, metadata, debug_data, video_path,
                world_width=MySettings.Simulation.WORLD_WIDTH,
                world_height=MySettings.Simulation.WORLD_HEIGHT
            )

    print(f"\n{'=' * 60}")
    print("Clustering analysis complete!")
    print(f"{'=' * 60}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  Statistics: {CLUSTER_STATS_FILENAME}")
    print(f"  2D Visualization: {CLUSTERS_2D_FILENAME}")
    print(f"  Sensor Distributions: {SENSOR_DISTRIBUTIONS_FILENAME}")
    print(f"  Cluster Timeline: {CLUSTER_TIMELINE_FILENAME}")
    print(f"  Timeline CSV: {CLUSTER_TIMELINE_CSV_FILENAME}")
    print(f"  KMeans model: {KMEANS_MODEL_FILENAME}")
    print(f"  Clustering metadata: {CLUSTERING_METADATA_FILENAME}")
    if video_path is not None:
        print(f"  Video: {video_path.name}")