"""
Cluster Output Distribution Analysis

Analyzes the distribution of network outputs (wheel speeds and pheromone secretion)
for each cluster obtained from K-means clustering.

Usage:
    PYTHONPATH=. uv run --extra cpu src/interpretation/clustering/cluster_output_analysis.py \
        --clustering-result results/clustering/clustering_metadata.pkl \
        --debug-data results/analysis/debug_data.pkl \
        --sample-interval 1 \
        --output-dir results/clustering/output_analysis
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

from src.interpretation.clustering.utils import RobotSensorSample, convert_debug_data_to_dataset_filtered
from src.interpretation.clustering.kmeans_clustering import ClusteringMetadata
from src.analysis_mod.structure.debug_data import DebugData


@dataclass
class ClusterOutputStats:
    """Statistics for network outputs in a cluster."""
    cluster_id: int
    n_samples: int

    # Wheel speeds (right_wheel, left_wheel)
    right_wheel_mean: float
    right_wheel_std: float
    right_wheel_min: float
    right_wheel_max: float

    left_wheel_mean: float
    left_wheel_std: float
    left_wheel_min: float
    left_wheel_max: float

    # Pheromone secretion
    pheromone_secretion_mean: float
    pheromone_secretion_std: float
    pheromone_secretion_min: float
    pheromone_secretion_max: float

    # Raw outputs for distribution analysis
    right_wheel_outputs: np.ndarray
    left_wheel_outputs: np.ndarray
    pheromone_outputs: np.ndarray


def analyze_cluster_outputs(
    metadata: ClusteringMetadata,
    dataset: list[RobotSensorSample]
) -> dict[int, ClusterOutputStats]:
    """
    Analyze network output distributions for each cluster.

    Args:
        metadata: ClusteringMetadata from K-means clustering
        dataset: Original list of RobotSensorSample used for clustering

    Returns:
        Dictionary mapping cluster_id to ClusterOutputStats
    """
    cluster_stats = {}

    for cluster_id in range(metadata.n_clusters):
        # Get samples in this cluster
        cluster_mask = (metadata.labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_samples = [dataset[i] for i in cluster_indices]

        # Extract network outputs
        # network_output shape: (3,) = [right_wheel, left_wheel, pheromone_secretion]
        right_wheels = np.array([s.network_output[0] for s in cluster_samples])
        left_wheels = np.array([s.network_output[1] for s in cluster_samples])
        pheromones = np.array([s.network_output[2] for s in cluster_samples])

        # Calculate statistics
        stats = ClusterOutputStats(
            cluster_id=cluster_id,
            n_samples=len(cluster_samples),

            right_wheel_mean=float(np.mean(right_wheels)),
            right_wheel_std=float(np.std(right_wheels)),
            right_wheel_min=float(np.min(right_wheels)),
            right_wheel_max=float(np.max(right_wheels)),

            left_wheel_mean=float(np.mean(left_wheels)),
            left_wheel_std=float(np.std(left_wheels)),
            left_wheel_min=float(np.min(left_wheels)),
            left_wheel_max=float(np.max(left_wheels)),

            pheromone_secretion_mean=float(np.mean(pheromones)),
            pheromone_secretion_std=float(np.std(pheromones)),
            pheromone_secretion_min=float(np.min(pheromones)),
            pheromone_secretion_max=float(np.max(pheromones)),

            right_wheel_outputs=right_wheels,
            left_wheel_outputs=left_wheels,
            pheromone_outputs=pheromones,
        )

        cluster_stats[cluster_id] = stats

    return cluster_stats


def visualize_output_distributions(
    cluster_stats: dict[int, ClusterOutputStats],
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (16, 12)
):
    """
    Visualize output distributions for all clusters.

    Args:
        cluster_stats: Output from analyze_cluster_outputs()
        save_path: Path to save figure (None to display)
        figsize: Figure size
    """
    n_clusters = len(cluster_stats)
    fig, axes = plt.subplots(n_clusters, 3, figsize=figsize)

    # Handle single cluster case
    if n_clusters == 1:
        axes = axes.reshape(1, -1)

    for cluster_id in sorted(cluster_stats.keys()):
        stats = cluster_stats[cluster_id]
        row = cluster_id

        # Right wheel distribution
        ax = axes[row, 0]
        ax.hist(stats.right_wheel_outputs, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(stats.right_wheel_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {stats.right_wheel_mean:.3f}')
        ax.set_xlabel('Right Wheel Speed')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Cluster {cluster_id}: Right Wheel (n={stats.n_samples})')
        ax.legend()
        ax.grid(alpha=0.3)

        # Left wheel distribution
        ax = axes[row, 1]
        ax.hist(stats.left_wheel_outputs, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(stats.left_wheel_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {stats.left_wheel_mean:.3f}')
        ax.set_xlabel('Left Wheel Speed')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Cluster {cluster_id}: Left Wheel (n={stats.n_samples})')
        ax.legend()
        ax.grid(alpha=0.3)

        # Pheromone distribution
        ax = axes[row, 2]
        ax.hist(stats.pheromone_outputs, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax.axvline(stats.pheromone_secretion_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {stats.pheromone_secretion_mean:.3f}')
        ax.set_xlabel('Pheromone Secretion')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Cluster {cluster_id}: Pheromone (n={stats.n_samples})')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved output distributions to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_output_comparison(
    cluster_stats: dict[int, ClusterOutputStats],
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 10)
):
    """
    Compare output statistics across clusters using bar plots.

    Args:
        cluster_stats: Output from analyze_cluster_outputs()
        save_path: Path to save figure (None to display)
        figsize: Figure size
    """
    cluster_ids = sorted(cluster_stats.keys())

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Right wheel mean
    ax = axes[0, 0]
    means = [cluster_stats[cid].right_wheel_mean for cid in cluster_ids]
    stds = [cluster_stats[cid].right_wheel_std for cid in cluster_ids]
    ax.bar(cluster_ids, means, yerr=stds, alpha=0.7, color='blue', capsize=5)
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Right Wheel Speed')
    ax.set_title('Right Wheel Speed by Cluster')
    ax.grid(alpha=0.3)

    # Left wheel mean
    ax = axes[0, 1]
    means = [cluster_stats[cid].left_wheel_mean for cid in cluster_ids]
    stds = [cluster_stats[cid].left_wheel_std for cid in cluster_ids]
    ax.bar(cluster_ids, means, yerr=stds, alpha=0.7, color='green', capsize=5)
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Left Wheel Speed')
    ax.set_title('Left Wheel Speed by Cluster')
    ax.grid(alpha=0.3)

    # Pheromone secretion mean
    ax = axes[1, 0]
    means = [cluster_stats[cid].pheromone_secretion_mean for cid in cluster_ids]
    stds = [cluster_stats[cid].pheromone_secretion_std for cid in cluster_ids]
    ax.bar(cluster_ids, means, yerr=stds, alpha=0.7, color='orange', capsize=5)
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Pheromone Secretion')
    ax.set_title('Pheromone Secretion by Cluster')
    ax.grid(alpha=0.3)

    # Sample counts
    ax = axes[1, 1]
    counts = [cluster_stats[cid].n_samples for cid in cluster_ids]
    ax.bar(cluster_ids, counts, alpha=0.7, color='purple')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Cluster Sample Counts')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved output comparison to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_wheel_scatter(
    cluster_stats: dict[int, ClusterOutputStats],
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 8)
):
    """
    Visualize relationship between right and left wheel speeds for each cluster.

    Args:
        cluster_stats: Output from analyze_cluster_outputs()
        save_path: Path to save figure (None to display)
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_stats)))

    for cluster_id, color in zip(sorted(cluster_stats.keys()), colors):
        stats = cluster_stats[cluster_id]
        ax.scatter(
            stats.right_wheel_outputs,
            stats.left_wheel_outputs,
            c=[color],
            alpha=0.5,
            s=20,
            label=f'Cluster {cluster_id} (n={stats.n_samples})'
        )

    ax.set_xlabel('Right Wheel Speed')
    ax.set_ylabel('Left Wheel Speed')
    ax.set_title('Right vs Left Wheel Speed by Cluster')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)

    # Add diagonal line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, label='Right = Left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved wheel scatter plot to: {save_path}")
    else:
        plt.show()

    plt.close()


def save_statistics_report(
    cluster_stats: dict[int, ClusterOutputStats],
    output_path: Path
):
    """
    Save cluster output statistics as text report.

    Args:
        cluster_stats: Output from analyze_cluster_outputs()
        output_path: Path to save text file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("Cluster Output Distribution Analysis\n")
        f.write("=" * 80 + "\n\n")

        for cluster_id in sorted(cluster_stats.keys()):
            stats = cluster_stats[cluster_id]

            f.write(f"Cluster {cluster_id}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of samples: {stats.n_samples}\n\n")

            f.write("Right Wheel Speed:\n")
            f.write(f"  Mean: {stats.right_wheel_mean:.6f}\n")
            f.write(f"  Std:  {stats.right_wheel_std:.6f}\n")
            f.write(f"  Min:  {stats.right_wheel_min:.6f}\n")
            f.write(f"  Max:  {stats.right_wheel_max:.6f}\n\n")

            f.write("Left Wheel Speed:\n")
            f.write(f"  Mean: {stats.left_wheel_mean:.6f}\n")
            f.write(f"  Std:  {stats.left_wheel_std:.6f}\n")
            f.write(f"  Min:  {stats.left_wheel_min:.6f}\n")
            f.write(f"  Max:  {stats.left_wheel_max:.6f}\n\n")

            f.write("Pheromone Secretion:\n")
            f.write(f"  Mean: {stats.pheromone_secretion_mean:.6f}\n")
            f.write(f"  Std:  {stats.pheromone_secretion_std:.6f}\n")
            f.write(f"  Min:  {stats.pheromone_secretion_min:.6f}\n")
            f.write(f"  Max:  {stats.pheromone_secretion_max:.6f}\n\n")

    print(f"Saved statistics report to: {output_path}")


# ==============================================================================
# CLI Interface
# ==============================================================================

if __name__ == '__main__':
    import argparse

    def load_clustering_metadata(path: Path) -> ClusteringMetadata:
        """Load clustering metadata from pickle file."""
        with open(path, 'rb') as f:
            metadata = pickle.load(f)

        if not isinstance(metadata, ClusteringMetadata):
            raise ValueError(f"Expected ClusteringMetadata, got {type(metadata)}")

        print(f"Loaded clustering metadata from: {path}")
        print(f"  Number of clusters: {metadata.n_clusters}")
        print(f"  Number of samples: {len(metadata.labels)}")
        print(f"  Pheromone threshold: {metadata.pheromone_threshold}")
        return metadata

    def load_debug_data(path: Path) -> list[DebugData]:
        """Load debug data from pickle file."""
        with open(path, 'rb') as f:
            debug_data = pickle.load(f)

        print(f"Loaded debug data from: {path}")
        print(f"  Total frames: {len(debug_data)}")
        return debug_data

    parser = argparse.ArgumentParser(
        description="Analyze network output distributions for K-means clusters"
    )

    parser.add_argument(
        "--clustering-result",
        type=str,
        required=True,
        help="Path to clustering metadata pickle file (clustering_metadata.pkl)"
    )
    parser.add_argument(
        "--debug-data",
        type=str,
        required=True,
        help="Path to debug data pickle file (list of DebugData)"
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=1,
        help="Sample interval for dataset generation (default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as clustering-result directory)"
    )

    args = parser.parse_args()

    # Load data
    clustering_path = Path(args.clustering_result)
    debug_data_path = Path(args.debug_data)

    if not clustering_path.exists():
        raise FileNotFoundError(f"Clustering metadata not found: {clustering_path}")
    if not debug_data_path.exists():
        raise FileNotFoundError(f"Debug data not found: {debug_data_path}")

    # Load clustering metadata
    metadata = load_clustering_metadata(clustering_path)

    # Load debug data
    debug_data = load_debug_data(debug_data_path)

    # Regenerate dataset with same filtering as clustering
    print("\nRegenerating dataset with clustering parameters...")
    print(f"  Pheromone threshold: {metadata.pheromone_threshold}")
    print(f"  Sample interval: {args.sample_interval}")

    dataset = convert_debug_data_to_dataset_filtered(
        debug_data,
        pheromone_threshold=metadata.pheromone_threshold,
        sample_interval=args.sample_interval
    )

    print(f"  Generated dataset size: {len(dataset)}")

    # Validate dataset and metadata consistency
    if len(dataset) != len(metadata.labels):
        raise ValueError(
            f"Dataset size ({len(dataset)}) does not match metadata labels size ({len(metadata.labels)})\n"
            f"This may be due to different sample_interval values.\n"
            f"Please ensure you use the same sample_interval as when clustering was performed."
        )

    # Set output directory
    if args.output_dir is None:
        output_dir = clustering_path.parent / "output_analysis"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("Cluster Output Distribution Analysis")
    print("=" * 80)
    print(f"Number of clusters: {metadata.n_clusters}")
    print(f"Total samples: {len(dataset)}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Analyze output distributions
    print("\nAnalyzing cluster output distributions...")
    cluster_stats = analyze_cluster_outputs(metadata, dataset)

    # Print summary
    print("\nCluster Output Summary:")
    print("-" * 80)
    for cluster_id in sorted(cluster_stats.keys()):
        stats = cluster_stats[cluster_id]
        print(f"Cluster {cluster_id} (n={stats.n_samples}):")
        print(f"  Right Wheel:  {stats.right_wheel_mean:.4f} ± {stats.right_wheel_std:.4f}")
        print(f"  Left Wheel:   {stats.left_wheel_mean:.4f} ± {stats.left_wheel_std:.4f}")
        print(f"  Pheromone:    {stats.pheromone_secretion_mean:.4f} ± {stats.pheromone_secretion_std:.4f}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Distribution histograms
    dist_path = output_dir / "output_distributions.png"
    visualize_output_distributions(cluster_stats, save_path=dist_path)

    # Comparison bar plots
    comp_path = output_dir / "output_comparison.png"
    visualize_output_comparison(cluster_stats, save_path=comp_path)

    # Wheel scatter plot
    scatter_path = output_dir / "wheel_scatter.png"
    visualize_wheel_scatter(cluster_stats, save_path=scatter_path)

    # Save statistics report
    report_path = output_dir / "output_statistics.txt"
    save_statistics_report(cluster_stats, report_path)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  Distribution histograms: {dist_path.name}")
    print(f"  Comparison bar plots: {comp_path.name}")
    print(f"  Wheel scatter plot: {scatter_path.name}")
    print(f"  Statistics report: {report_path.name}")
