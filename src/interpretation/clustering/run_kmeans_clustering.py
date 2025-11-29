#!/usr/bin/env python3
"""
CLI runner for K-means clustering analysis.

Usage examples:
    # Basic clustering with k=10
    python src/interpretation/clustering/run_kmeans_clustering.py

    # Cluster with k=20
    python src/interpretation/clustering/run_kmeans_clustering.py --n-clusters 20

    # Find optimal k
    python src/interpretation/clustering/run_kmeans_clustering.py --find-optimal --k-min 2 --k-max 30
"""

import argparse
import pickle
from pathlib import Path

from src.interpretation.clustering.kmeans_clustering import (
    cluster_sensor_states,
    find_optimal_k,
    visualize_clusters,
    visualize_elbow_curve,
    get_cluster_statistics,
    save_clustering_result,
)
from src.interpretation.data_collection.io_sample_definition import ShapleyDataset


def main():
    parser = argparse.ArgumentParser(description='K-means clustering for Shapley input samples')
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('results/shapley_samples.pkl'),
        help='Path to input Shapley dataset (default: results/shapley_samples.pkl)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/kmeans_clustering'),
        help='Directory to save clustering results (default: results/kmeans_clustering)'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=10,
        help='Number of clusters (default: 10)'
    )
    parser.add_argument(
        '--find-optimal',
        action='store_true',
        help='Find optimal k using elbow method and quality metrics'
    )
    parser.add_argument(
        '--k-min',
        type=int,
        default=2,
        help='Minimum k for optimal k search (default: 2)'
    )
    parser.add_argument(
        '--k-max',
        type=int,
        default=20,
        help='Maximum k for optimal k search (default: 20)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from: {args.input}")
    with open(args.input, 'rb') as f:
        dataset = pickle.load(f)

    if not isinstance(dataset, ShapleyDataset):
        raise TypeError(f"Expected ShapleyDataset, got {type(dataset)}")

    print(f"\nDataset loaded: {len(dataset)} samples")
    print(dataset.get_summary())

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.find_optimal:
        print(f"\n{'=' * 60}")
        print("Finding Optimal K")
        print(f"{'=' * 60}")

        k_metrics = find_optimal_k(
            dataset,
            k_range=range(args.k_min, args.k_max + 1),
            random_state=args.random_state
        )

        # Save metrics
        metrics_path = args.output_dir / 'optimal_k_metrics.pkl'
        with open(metrics_path, 'wb') as f:
            pickle.dump(k_metrics, f)
        print(f"\nSaved metrics to: {metrics_path}")

        # Print summary
        print("\nMetrics Summary:")
        print(f"{'k':>5} {'Inertia':>12} {'Silhouette':>12} {'Davies-Bouldin':>16} {'Calinski-Harabasz':>18}")
        print('-' * 70)
        for k in sorted(k_metrics.keys()):
            m = k_metrics[k]
            print(f"{k:>5} {m['inertia']:>12.2f} {m['silhouette']:>12.4f} "
                  f"{m['davies_bouldin']:>16.4f} {m['calinski_harabasz']:>18.2f}")

        # Visualize elbow curve
        elbow_path = args.output_dir / 'elbow_curve.png'
        visualize_elbow_curve(k_metrics, save_path=elbow_path)

    else:
        print(f"\n{'=' * 60}")
        print(f"Running K-means Clustering (k={args.n_clusters})")
        print(f"{'=' * 60}")

        # Run clustering
        result = cluster_sensor_states(
            dataset,
            n_clusters=args.n_clusters,
            random_state=args.random_state
        )

        print(f"\nClustering Results:")
        print(f"  Number of clusters: {result.n_clusters}")
        print(f"  Inertia: {result.inertia:.2f}")
        print(f"  Silhouette score: {result.silhouette:.4f}")
        print(f"  Davies-Bouldin index: {result.davies_bouldin:.4f}")
        print(f"  Calinski-Harabasz index: {result.calinski_harabasz:.2f}")

        # Print cluster sizes
        cluster_sizes = result.get_cluster_sizes()
        print(f"\nCluster sizes:")
        for cluster_id in sorted(cluster_sizes.keys()):
            size = cluster_sizes[cluster_id]
            percentage = 100 * size / len(dataset)
            print(f"  Cluster {cluster_id}: {size:>6} samples ({percentage:>5.1f}%)")

        # Get detailed statistics
        stats = get_cluster_statistics(result)

        # Save statistics to file
        stats_path = args.output_dir / 'cluster_stats.txt'
        with open(stats_path, 'w') as f:
            f.write(f"K-means Clustering Results (k={result.n_clusters})\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(f"Overall Metrics:\n")
            f.write(f"  Inertia: {result.inertia:.4f}\n")
            f.write(f"  Silhouette score: {result.silhouette:.4f}\n")
            f.write(f"  Davies-Bouldin index: {result.davies_bouldin:.4f}\n")
            f.write(f"  Calinski-Harabasz index: {result.calinski_harabasz:.4f}\n\n")

            for cluster_id in sorted(stats.keys()):
                s = stats[cluster_id]
                f.write(f"Cluster {cluster_id} (n={s['size']}):\n")
                f.write(f"  Centroid:\n")
                for name, val in s['centroid'].items():
                    f.write(f"    {name}: {val:>8.4f}\n")
                f.write(f"  Feature means:\n")
                for name, val in s['feature_means'].items():
                    f.write(f"    {name}: {val:>8.4f}\n")
                f.write(f"  Feature stds:\n")
                for name, val in s['feature_stds'].items():
                    f.write(f"    {name}: {val:>8.4f}\n")
                f.write(f"\n")

        print(f"\nSaved cluster statistics to: {stats_path}")

        # Visualize clusters
        viz_path = args.output_dir / 'clusters_2d.png'
        visualize_clusters(result, save_path=viz_path)

        # Save clustering result
        result_path = args.output_dir / 'clustering_result.pkl'
        save_clustering_result(result, result_path)

        print(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()