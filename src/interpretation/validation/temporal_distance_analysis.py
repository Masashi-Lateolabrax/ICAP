"""
Temporal Distance Analysis for Shapley Input Samples

Validates whether temporally adjacent samples are spatially close in feature space.
This helps determine if time-series structure exists and why OPTICS failed to detect clusters.

Usage:
    PYTHONPATH=. uv run --extra cpu src/interpretation/validation/temporal_distance_analysis.py \
        --data-path results/20251027-024639_7bb53c8b/shapley_data/samples.pkl
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

from src.interpretation.data_collection.io_sample_definition import ShapleyDataset, ShapleyInputSample
from src.interpretation.clustering.pca_clustering import extract_sensor_features


def extract_features_from_sample(sample: ShapleyInputSample) -> np.ndarray:
    """Extract 6D sensor features from a single sample."""
    return np.array([
        sample.robot_sensor[0],
        sample.robot_sensor[1],
        sample.food_sensor[0],
        sample.food_sensor[1],
        sample.direction_sensor[0],
        sample.direction_sensor[1],
    ])


def analyze_temporal_distances(dataset: ShapleyDataset, output_dir: Path):
    """
    Analyze distances between temporally adjacent samples.

    Args:
        dataset: ShapleyDataset containing samples
        output_dir: Directory to save results
    """
    # Group samples by robot
    robot_samples = defaultdict(list)
    for sample in dataset.samples:
        robot_samples[sample.robot_index].append(sample)

    # Sort each robot's samples by timestep
    for robot_id in robot_samples:
        robot_samples[robot_id].sort(key=lambda s: s.timestep)

    print(f"Found {len(robot_samples)} robots")
    for robot_id, samples in robot_samples.items():
        print(f"  Robot {robot_id}: {len(samples)} samples")

    # Analyze temporal distances for each robot
    all_temporal_distances = []
    robot_temporal_stats = {}

    print("\nAnalyzing temporal distances...")
    total_robots = len([r for r in robot_samples.values() if len(r) >= 2])
    processed_robots = 0

    for robot_id, samples in robot_samples.items():
        if len(samples) < 2:
            continue

        temporal_distances = []
        timestep_diffs = []
        skipped_transitions = 0

        for i in range(len(samples) - 1):
            timestep_diff = samples[i + 1].timestep - samples[i].timestep

            if timestep_diff <= 1:
                feat_i = extract_features_from_sample(samples[i])
                feat_j = extract_features_from_sample(samples[i + 1])
                distance = np.linalg.norm(feat_i - feat_j)
                temporal_distances.append(distance)

        all_temporal_distances.extend(temporal_distances)

        if len(temporal_distances) > 0:
            robot_temporal_stats[robot_id] = {
                'mean': np.mean(temporal_distances),
                'std': np.std(temporal_distances),
                'min': np.min(temporal_distances),
                'max': np.max(temporal_distances),
                'median': np.median(temporal_distances),
                'n_samples': len(samples),
                'n_transitions': len(temporal_distances),
                'n_skipped': skipped_transitions,
                'timestep_diffs': timestep_diffs,
            }

        processed_robots += 1
        if processed_robots % 10 == 0 or processed_robots == total_robots:
            print(f"  Processed {processed_robots}/{total_robots} robots...")

    # Print statistics
    print("\n" + "=" * 60)
    print("Temporal Distance Analysis Results")
    print("=" * 60)

    print(f"\nOverall temporal distances (all robots):")
    print(f"  Mean: {np.mean(all_temporal_distances):.4f}")
    print(f"  Std: {np.std(all_temporal_distances):.4f}")
    print(f"  Min: {np.min(all_temporal_distances):.4f}")
    print(f"  Max: {np.max(all_temporal_distances):.4f}")
    print(f"  Median: {np.median(all_temporal_distances):.4f}")
    print(f"  Total transitions: {len(all_temporal_distances)}")

    print(f"\nPer-robot statistics:")
    for robot_id in sorted(robot_temporal_stats.keys()):
        stats = robot_temporal_stats[robot_id]
        print(f"  Robot {robot_id}:")
        print(f"    Mean distance: {stats['mean']:.4f}")
        print(f"    Std: {stats['std']:.4f}")
        print(f"    Samples: {stats['n_samples']}, Transitions: {stats['n_transitions']}, Skipped: {stats['n_skipped']}")

    # Save statistics to file
    stats_path = output_dir / "temporal_distance_stats.txt"
    with open(stats_path, 'w') as f:
        f.write("Temporal Distance Analysis\n")
        f.write("=" * 60 + "\n\n")

        f.write("Overall temporal distances (all robots):\n")
        f.write(f"  Mean: {np.mean(all_temporal_distances):.6f}\n")
        f.write(f"  Std: {np.std(all_temporal_distances):.6f}\n")
        f.write(f"  Min: {np.min(all_temporal_distances):.6f}\n")
        f.write(f"  Max: {np.max(all_temporal_distances):.6f}\n")
        f.write(f"  Median: {np.median(all_temporal_distances):.6f}\n")
        f.write(f"  Total transitions: {len(all_temporal_distances)}\n\n")

        f.write("Per-robot statistics:\n")
        for robot_id in sorted(robot_temporal_stats.keys()):
            stats = robot_temporal_stats[robot_id]
            f.write(f"\n  Robot {robot_id}:\n")
            f.write(f"    Mean: {stats['mean']:.6f}\n")
            f.write(f"    Std: {stats['std']:.6f}\n")
            f.write(f"    Min: {stats['min']:.6f}\n")
            f.write(f"    Max: {stats['max']:.6f}\n")
            f.write(f"    Median: {stats['median']:.6f}\n")
            f.write(f"    Samples: {stats['n_samples']}\n")
            f.write(f"    Transitions: {stats['n_transitions']}\n")
            f.write(f"    Skipped: {stats['n_skipped']}\n")

    print(f"\nSaved statistics to: {stats_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_temporal_distances, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(all_temporal_distances), color='red', linestyle='--',
                label=f'Mean: {np.mean(all_temporal_distances):.3f}')
    plt.axvline(np.median(all_temporal_distances), color='blue', linestyle='--',
                label=f'Median: {np.median(all_temporal_distances):.3f}')
    plt.xlabel('Euclidean Distance (6D space)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Temporal Distances')
    plt.legend()
    plt.grid(alpha=0.3)

    hist_path = output_dir / 'distance_histogram.png'
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved histogram to: {hist_path}")

    # 2. CDF
    sorted_distances = np.sort(all_temporal_distances)
    cumulative_prob = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_distances, cumulative_prob, linewidth=2)
    plt.xlabel('Euclidean Distance (6D space)')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function of Temporal Distances')
    plt.grid(alpha=0.3)

    # Add reference lines for key percentiles
    percentiles = [0.5, 0.9, 0.95]
    for p in percentiles:
        idx = int(p * len(sorted_distances))
        dist_val = sorted_distances[idx]
        plt.axhline(p, color='gray', linestyle=':', alpha=0.5)
        plt.axvline(dist_val, color='gray', linestyle=':', alpha=0.5)
        plt.text(dist_val, p, f'  {int(p*100)}%: {dist_val:.3f}',
                 verticalalignment='bottom')

    cdf_path = output_dir / 'distance_cdf.png'
    plt.savefig(cdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved CDF to: {cdf_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze temporal distances in Shapley input samples"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to Shapley dataset pickle file (samples.pkl)"
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

    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    print(f"Loaded dataset from: {data_path}")
    print(f"  Total samples: {len(dataset.samples)}")
    print(f"  Experiment: {dataset.experiment_id}")
    print(f"  Generation: {dataset.generation}")

    # Set output directory
    if args.output_dir is None:
        output_dir = data_path.parent / "validation"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    analyze_temporal_distances(dataset, output_dir)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
