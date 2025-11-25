"""
Run DBSCAN Clustering on Sensor States

Clusters robot sensor states (excluding pheromone) for Shapley value analysis.

Usage:
    PYTHONPATH=. uv run --extra cpu src/interpretation/clustering/run_dbscan_clustering.py \
        --data-path results/20251027-024639_7bb53c8b/shapley_data/samples.pkl \
        --eps 0.5 \
        --min-samples 5

Parameters:
    --data-path: Path to Shapley dataset pickle file (required)
    --eps: DBSCAN epsilon (maximum distance for neighborhood, default: 0.5)
    --min-samples: DBSCAN minimum samples (minimum neighbors, default: 5)
    --output-dir: Output directory (default: same directory as data-path)
"""

import argparse
import pickle
from pathlib import Path

from src.interpretation.data_collection.io_sample_definition import ShapleyDataset
from src.interpretation.clustering.dbscan_clustering import (
    cluster_sensor_states,
    visualize_clusters,
    get_cluster_statistics,
    save_clustering_result,
)


def load_dataset(data_path: Path) -> ShapleyDataset:
    """Load Shapley dataset from pickle file."""
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Loaded dataset from: {data_path}")
    print(f"  Total samples: {len(dataset.samples)}")
    print(f"  Experiment: {dataset.experiment_id}")
    print(f"  Generation: {dataset.generation}")
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Run DBSCAN clustering on sensor states"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to Shapley dataset pickle file (samples.pkl)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="DBSCAN epsilon parameter (default: 0.5)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="DBSCAN minimum samples parameter (default: 5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as data-path directory)"
    )

    args = parser.parse_args()

    # Load dataset
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    dataset = load_dataset(data_path)

    # Set output directory
    if args.output_dir is None:
        output_dir = data_path.parent / "clustering"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("DBSCAN Clustering Configuration")
    print("=" * 60)
    print(f"Epsilon (eps): {args.eps}")
    print(f"Minimum samples: {args.min_samples}")
    print(f"Features: 6D sensor states (robot, food, direction)")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Perform clustering
    print("\nRunning DBSCAN clustering...")
    result = cluster_sensor_states(
        dataset,
        eps=args.eps,
        min_samples=args.min_samples
    )

    print(f"\nClustering complete!")
    print(f"  Found {result.n_clusters} clusters")

    # Print cluster sizes
    cluster_sizes = result.get_cluster_sizes()
    print("\nCluster sizes:")
    for cluster_id, size in sorted(cluster_sizes.items()):
        if cluster_id == -1:
            print(f"  Noise: {size} samples ({100*size/len(result.labels):.1f}%)")
        else:
            print(f"  Cluster {cluster_id}: {size} samples ({100*size/len(result.labels):.1f}%)")

    # Get cluster statistics
    print("\nComputing cluster statistics...")
    stats = get_cluster_statistics(result)

    # Save statistics
    stats_path = output_dir / "cluster_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"DBSCAN Clustering Results\n")
        f.write(f"========================\n")
        f.write(f"Epsilon: {args.eps}\n")
        f.write(f"Minimum samples: {args.min_samples}\n")
        f.write(f"Number of clusters: {result.n_clusters}\n\n")

        for cluster_id, cluster_stats in stats.items():
            f.write(f"\nCluster {cluster_id}:\n")
            f.write(f"  Size: {cluster_stats['size']}\n")
            f.write(f"  Feature means:\n")
            for feature_name, mean_val in cluster_stats['feature_means'].items():
                f.write(f"    {feature_name}: {mean_val:.4f}\n")

    print(f"Saved statistics to: {stats_path}")

    # Visualize clusters
    print("\nGenerating visualization...")
    viz_path = output_dir / "clusters_2d.png"
    visualize_clusters(result, save_path=viz_path)

    # Save clustering results
    results_path = output_dir / "clustering_result.pkl"
    save_clustering_result(result, results_path)

    print("\n" + "=" * 60)
    print("Clustering analysis complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  Statistics: {stats_path.name}")
    print(f"  Visualization: {viz_path.name}")
    print(f"  Clustering result: {results_path.name}")


if __name__ == "__main__":
    main()