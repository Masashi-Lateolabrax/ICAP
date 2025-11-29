#!/usr/bin/env python3
"""
CLI runner for K-means clustering analysis.

Usage examples:
    # Basic clustering with k=10
    PYTHONPATH=. uv run --extra cpu src/interpretation/clustering/run_kmeans_clustering.py \
        --data-path results/20251027-024639_7bb53c8b/shapley_data/samples.pkl \
        --n-clusters 10

    # Find optimal k
    PYTHONPATH=. uv run --extra cpu src/interpretation/clustering/run_kmeans_clustering.py \
        --data-path results/20251027-024639_7bb53c8b/shapley_data/samples.pkl \
        --find-optimal --k-min 2 --k-max 30

Parameters:
    --data-path: Path to Shapley dataset pickle file (required)
    --n-clusters: Number of clusters (default: 10)
    --find-optimal: Find optimal k using elbow method and quality metrics
    --k-min: Minimum k for optimal k search (default: 2)
    --k-max: Maximum k for optimal k search (default: 20)
    --output-dir: Output directory (default: same directory as data-path)
    --random-state: Random seed for reproducibility (default: 42)
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
    analyze_temporal_continuity,
    visualize_temporal_continuity,
    visualize_feature_distributions,
)
from src.interpretation.data_collection.io_sample_definition import ShapleyDataset


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
    parser = argparse.ArgumentParser(description='K-means clustering for Shapley input samples')
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to Shapley dataset pickle file (samples.pkl)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: same as data-path directory)'
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
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    dataset = load_dataset(data_path)

    # Set output directory
    if args.output_dir is None:
        output_dir = data_path.parent / "kmeans_clustering"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

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
        metrics_path = output_dir / 'optimal_k_metrics.pkl'
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
        elbow_path = output_dir / 'elbow_curve.png'
        visualize_elbow_curve(k_metrics, save_path=elbow_path)

        print(f"\n{'=' * 60}")
        print("Optimal k search complete!")
        print(f"{'=' * 60}")
        print(f"\nResults saved to: {output_dir}")
        print(f"  Metrics: {metrics_path.name}")
        print(f"  Elbow curve: {elbow_path.name}")

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
        stats_path = output_dir / 'cluster_stats.txt'
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

        # Analyze temporal continuity
        print(f"\n{'=' * 60}")
        print("Analyzing Temporal Continuity")
        print(f"{'=' * 60}")

        continuity_analysis = analyze_temporal_continuity(result, dataset)

        # Determine cluster type based on continuity
        situation_based_count = 0
        time_period_count = 0

        print("\nTemporal Continuity per Cluster:")
        print(f"{'Cluster':>8} {'Size':>8} {'Consecutive':>12} {'Ratio':>8} {'Type':>15}")
        print('-' * 60)

        for cluster_id in sorted(continuity_analysis.keys()):
            c = continuity_analysis[cluster_id]
            ratio = c['continuity_ratio']

            if ratio > 0.7:
                cluster_type = "Time-period"
                time_period_count += 1
            elif ratio < 0.3:
                cluster_type = "Situation-based"
                situation_based_count += 1
            else:
                cluster_type = "Mixed"

            print(f"{cluster_id:>8} {c['size']:>8} {c['consecutive_pairs']:>12} "
                  f"{ratio:>7.1%} {cluster_type:>15}")

        print(f"\n{'=' * 60}")
        print("Cluster Type Summary:")
        print(f"  Situation-based clusters (<30% continuity): {situation_based_count}")
        print(f"  Mixed clusters (30-70% continuity): {result.n_clusters - situation_based_count - time_period_count}")
        print(f"  Time-period clusters (>70% continuity): {time_period_count}")

        if time_period_count > result.n_clusters / 2:
            print(f"\n⚠ WARNING: {time_period_count}/{result.n_clusters} clusters are time-period based!")
            print("  This suggests clusters represent temporal segments rather than situations.")
            print("  Consider alternative approaches (e.g., behavioral segmentation).")
        elif situation_based_count > result.n_clusters / 2:
            print(f"\n✓ Good: {situation_based_count}/{result.n_clusters} clusters are situation-based!")
            print("  Clusters appear to represent distinct behavioral situations.")
        else:
            print(f"\n⚠ Mixed: Clusters show both temporal and situational characteristics.")
            print("  Interpretation should be done carefully.")

        # Save temporal continuity analysis
        continuity_path = output_dir / 'temporal_continuity.txt'
        with open(continuity_path, 'w') as f:
            f.write("Temporal Continuity Analysis\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"{'Cluster':>8} {'Size':>8} {'Consecutive':>12} {'Total Pairs':>12} {'Ratio':>8}\n")
            f.write('-' * 60 + '\n')
            for cluster_id in sorted(continuity_analysis.keys()):
                c = continuity_analysis[cluster_id]
                f.write(f"{cluster_id:>8} {c['size']:>8} {c['consecutive_pairs']:>12} "
                       f"{c['total_pairs']:>12} {c['continuity_ratio']:>7.1%}\n")

        print(f"\nSaved temporal continuity analysis to: {continuity_path.name}")

        # Visualize temporal continuity
        continuity_viz_path = output_dir / 'temporal_continuity.png'
        visualize_temporal_continuity(continuity_analysis, save_path=continuity_viz_path)

        # Visualize feature distributions
        print("\nGenerating feature distribution visualizations...")
        feature_dist_path = output_dir / 'feature_distributions.png'
        visualize_feature_distributions(result, save_path=feature_dist_path)

        # Visualize clusters
        viz_path = output_dir / 'clusters_2d.png'
        visualize_clusters(result, save_path=viz_path)

        # Save clustering result
        result_path = output_dir / 'clustering_result.pkl'
        save_clustering_result(result, result_path)

        print(f"\n{'=' * 60}")
        print("Clustering analysis complete!")
        print(f"{'=' * 60}")
        print(f"\nResults saved to: {output_dir}")
        print(f"  Statistics: {stats_path.name}")
        print(f"  Temporal Continuity: {continuity_path.name}")
        print(f"  2D Visualization: {viz_path.name}")
        print(f"  Temporal Continuity Plot: {continuity_viz_path.name}")
        print(f"  Feature Distributions: {feature_dist_path.name}")
        print(f"  Clustering result: {result_path.name}")


if __name__ == '__main__':
    main()