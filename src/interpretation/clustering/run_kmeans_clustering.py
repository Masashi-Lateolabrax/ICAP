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
    analyze_cluster_distances,
    visualize_cluster_distance_matrix,
    analyze_temporal_statistics,
    visualize_temporal_statistics,
    analyze_original_sensor_values,
    visualize_original_sensor_distributions,
    visualize_cluster_timeline,
    export_cluster_timeline_data,
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

        # Analyze cluster distances
        print(f"\n{'=' * 60}")
        print("Analyzing Cluster Distances")
        print(f"{'=' * 60}")

        distance_analysis = analyze_cluster_distances(result)

        print(f"\nInter-cluster Distance Statistics:")
        print(f"  Min distance: {distance_analysis['min_distance']:.4f}")
        print(f"  Max distance: {distance_analysis['max_distance']:.4f}")
        print(f"  Mean distance: {distance_analysis['mean_distance']:.4f}")
        print(f"  Std distance: {distance_analysis['std_distance']:.4f}")

        print(f"\nNearest Cluster for Each Cluster:")
        for cluster_id in sorted(distance_analysis['nearest_clusters'].keys()):
            nearest_info = distance_analysis['nearest_clusters'][cluster_id]
            print(f"  Cluster {cluster_id} → Cluster {nearest_info['nearest_cluster']} "
                  f"(distance: {nearest_info['distance']:.4f})")

        # Save distance matrix
        distance_matrix_path = output_dir / 'cluster_distances.txt'
        with open(distance_matrix_path, 'w') as f:
            f.write("Cluster Distance Matrix\n")
            f.write("=" * 60 + "\n\n")

            # Write distance matrix
            f.write("Distance Matrix:\n")
            n = result.n_clusters
            header = "     " + "".join([f"  C{i:2d}  " for i in range(n)])
            f.write(header + "\n")
            for i in range(n):
                row = f"C{i:2d}  "
                for j in range(n):
                    if i == j:
                        row += "  -    "
                    else:
                        row += f"{distance_analysis['distance_matrix'][i, j]:6.2f} "
                f.write(row + "\n")

            f.write(f"\nStatistics:\n")
            f.write(f"  Min distance: {distance_analysis['min_distance']:.4f}\n")
            f.write(f"  Max distance: {distance_analysis['max_distance']:.4f}\n")
            f.write(f"  Mean distance: {distance_analysis['mean_distance']:.4f}\n")
            f.write(f"  Std distance: {distance_analysis['std_distance']:.4f}\n")

        print(f"\nSaved cluster distance matrix to: {distance_matrix_path.name}")

        # Visualize distance matrix
        distance_viz_path = output_dir / 'cluster_distance_matrix.png'
        visualize_cluster_distance_matrix(distance_analysis, save_path=distance_viz_path)

        # Analyze temporal statistics
        print(f"\n{'=' * 60}")
        print("Analyzing Temporal Statistics")
        print(f"{'=' * 60}")

        temporal_stats = analyze_temporal_statistics(result, dataset)

        print(f"\nTemporal Statistics per Cluster:")
        print(f"{'Cluster':>8} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print('-' * 60)
        for cluster_id in sorted(temporal_stats.keys()):
            ts = temporal_stats[cluster_id]
            print(f"{cluster_id:>8} {ts['timestep_mean']:>10.1f} {ts['timestep_std']:>10.1f} "
                  f"{ts['timestep_min']:>10} {ts['timestep_max']:>10}")

        # Save temporal statistics
        temporal_stats_path = output_dir / 'temporal_statistics.txt'
        with open(temporal_stats_path, 'w') as f:
            f.write("Temporal Statistics per Cluster\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"{'Cluster':>8} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}\n")
            f.write('-' * 60 + '\n')
            for cluster_id in sorted(temporal_stats.keys()):
                ts = temporal_stats[cluster_id]
                f.write(f"{cluster_id:>8} {ts['timestep_mean']:>12.2f} {ts['timestep_std']:>12.2f} "
                       f"{ts['timestep_min']:>12} {ts['timestep_max']:>12}\n")

            f.write(f"\nTime (seconds) Statistics:\n")
            f.write(f"{'Cluster':>8} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}\n")
            f.write('-' * 60 + '\n')
            for cluster_id in sorted(temporal_stats.keys()):
                ts = temporal_stats[cluster_id]
                f.write(f"{cluster_id:>8} {ts['time_seconds_mean']:>12.2f} {ts['time_seconds_std']:>12.2f} "
                       f"{ts['time_seconds_min']:>12.2f} {ts['time_seconds_max']:>12.2f}\n")

        print(f"\nSaved temporal statistics to: {temporal_stats_path.name}")

        # Visualize temporal statistics
        temporal_viz_path = output_dir / 'temporal_statistics.png'
        visualize_temporal_statistics(temporal_stats, save_path=temporal_viz_path)

        # Analyze original sensor values
        print(f"\n{'=' * 60}")
        print("Analyzing Original Sensor Values")
        print(f"{'=' * 60}")

        original_stats = analyze_original_sensor_values(result, dataset)

        # Print global statistics
        print(f"\nGlobal Statistics (all data):")
        for feature_name in ['robot_sensor[magnitude]', 'robot_sensor[angle]', 'food_sensor[magnitude]',
                            'food_sensor[angle]', 'direction_sensor[magnitude]', 'direction_sensor[angle]']:
            stats = original_stats['global'][feature_name]
            print(f"  {feature_name:30s}: mean={stats['mean']:6.3f}, std={stats['std']:6.3f}, "
                  f"min={stats['min']:6.3f}, max={stats['max']:6.3f}")

        # Print key insights for direction_sensor[magnitude]
        print(f"\nKey Insight - direction_sensor[magnitude] (distance to nest):")
        print(f"{'Cluster':>8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Interpretation':>25}")
        print('-' * 80)
        for cluster_id in sorted([k for k in original_stats.keys() if k != 'global']):
            stats = original_stats[cluster_id]['direction_sensor[magnitude]']
            if stats['max'] < 1.0:
                interpretation = "Inside nest"
            elif stats['min'] >= 1.0:
                interpretation = "Outside nest"
            else:
                interpretation = "Mixed (crosses boundary)"
            print(f"{cluster_id:>8} {stats['mean']:>8.3f} {stats['std']:>8.3f} "
                  f"{stats['min']:>8.3f} {stats['max']:>8.3f} {interpretation:>25}")

        # Save original sensor statistics
        original_stats_path = output_dir / 'original_sensor_stats.txt'
        with open(original_stats_path, 'w') as f:
            f.write("Original Sensor Value Statistics\n")
            f.write("=" * 60 + "\n\n")

            # Global statistics
            f.write("Global Statistics (all data):\n")
            f.write('-' * 80 + '\n')
            for feature_name in ['robot_sensor[magnitude]', 'robot_sensor[angle]', 'food_sensor[magnitude]',
                                'food_sensor[angle]', 'direction_sensor[magnitude]', 'direction_sensor[angle]']:
                stats = original_stats['global'][feature_name]
                f.write(f"{feature_name:30s}: mean={stats['mean']:7.4f}, std={stats['std']:7.4f}, "
                       f"min={stats['min']:7.4f}, max={stats['max']:7.4f}, median={stats['median']:7.4f}\n")
            f.write('\n')

            # Per-cluster statistics
            for cluster_id in sorted([k for k in original_stats.keys() if k != 'global']):
                cluster_size = result.get_cluster_sizes()[cluster_id]
                f.write(f"Cluster {cluster_id} (n={cluster_size}):\n")
                f.write('-' * 80 + '\n')
                for feature_name in ['robot_sensor[magnitude]', 'robot_sensor[angle]', 'food_sensor[magnitude]',
                                    'food_sensor[angle]', 'direction_sensor[magnitude]', 'direction_sensor[angle]']:
                    stats = original_stats[cluster_id][feature_name]
                    f.write(f"  {feature_name:30s}: mean={stats['mean']:7.4f}, std={stats['std']:7.4f}, "
                           f"min={stats['min']:7.4f}, max={stats['max']:7.4f}, median={stats['median']:7.4f}\n")
                f.write('\n')

        print(f"\nSaved original sensor statistics to: {original_stats_path.name}")

        # Visualize original sensor distributions
        original_viz_path = output_dir / 'original_sensor_distributions.png'
        visualize_original_sensor_distributions(original_stats, save_path=original_viz_path)

        # Visualize cluster timeline
        print(f"\n{'=' * 60}")
        print("Generating Cluster Timeline")
        print(f"{'=' * 60}")

        timeline_viz_path = output_dir / 'cluster_timeline.png'
        visualize_cluster_timeline(result, dataset, save_path=timeline_viz_path)

        # Export cluster timeline data to CSV
        timeline_csv_path = output_dir / 'cluster_timeline.csv'
        export_cluster_timeline_data(result, dataset, output_path=timeline_csv_path)

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
        print(f"  Cluster Distances: {distance_matrix_path.name}")
        print(f"  Temporal Statistics: {temporal_stats_path.name}")
        print(f"  Original Sensor Stats: {original_stats_path.name}")
        print(f"  Cluster Timeline CSV: {timeline_csv_path.name}")
        print(f"  2D Visualization: {viz_path.name}")
        print(f"  Temporal Continuity Plot: {continuity_viz_path.name}")
        print(f"  Distance Matrix Heatmap: {distance_viz_path.name}")
        print(f"  Temporal Statistics Plot: {temporal_viz_path.name}")
        print(f"  Original Sensor Distributions: {original_viz_path.name}")
        print(f"  Cluster Timeline Plot: {timeline_viz_path.name}")
        print(f"  Feature Distributions: {feature_dist_path.name}")
        print(f"  Clustering result: {result_path.name}")


if __name__ == '__main__':
    main()