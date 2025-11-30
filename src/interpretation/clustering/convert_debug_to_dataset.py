#!/usr/bin/env python3
"""
Convert DebugData pickle files to ShapleyDataset format.

This CLI tool converts debug data collected from analysis runs into
the ShapleyDataset format used for clustering analysis.

Usage examples:
    # Basic conversion
    PYTHONPATH=. uv run --extra cpu src/interpretation/clustering/convert_debug_to_dataset.py \
        --debug-data-path results/analysis/debug_data.pkl \
        --output-path results/clustering/samples.pkl

    # Conversion with filtering
    PYTHONPATH=. uv run --extra cpu src/interpretation/clustering/convert_debug_to_dataset.py \
        --debug-data-path results/analysis/debug_data.pkl \
        --output-path results/clustering/samples.pkl \
        --pheromone-threshold 0.1 \
        --sample-interval 5

    # Then run clustering
    PYTHONPATH=. uv run --extra cpu src/interpretation/clustering/run_kmeans_clustering.py \
        --data-path results/clustering/samples.pkl \
        --n-clusters 9 --create-video

Parameters:
    --debug-data-path: Path to DebugData pickle file (required)
    --output-path: Path to save ShapleyDataset pickle (required)
    --experiment-id: Experiment identifier (default: "debug_analysis")
    --generation: Generation number (default: 0)
    --pheromone-threshold: Only include samples with pheromone >= threshold (default: 0.0)
    --sample-interval: Only include every Nth timestep (default: 1 = all frames)
"""

import argparse
import pickle
from pathlib import Path

from src.interpretation.clustering.utils import (
    convert_debug_data_to_dataset_filtered,
)


def main():
    parser = argparse.ArgumentParser(
        description='Convert DebugData to ShapleyDataset for clustering'
    )

    parser.add_argument(
        '--debug-data-path',
        type=str,
        required=True,
        help='Path to DebugData pickle file'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Path to save ShapleyDataset pickle file'
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

    args = parser.parse_args()

    # Load debug data
    debug_data_path = Path(args.debug_data_path)
    if not debug_data_path.exists():
        raise FileNotFoundError(f"Debug data file not found: {debug_data_path}")

    print(f"Loading debug data from: {debug_data_path}")
    with open(debug_data_path, 'rb') as f:
        debug_data = pickle.load(f)

    print(f"Loaded {len(debug_data)} frames")

    # Convert to dataset
    print(f"\n{'=' * 60}")
    print("Converting to ShapleyDataset")
    print(f"{'=' * 60}")
    print(f"Experiment ID: {args.experiment_id}")
    print(f"Generation: {args.generation}")
    print(f"Pheromone threshold: {args.pheromone_threshold}")
    print(f"Sample interval: {args.sample_interval}")

    dataset = convert_debug_data_to_dataset_filtered(
        debug_data,
        experiment_id=args.experiment_id,
        generation=args.generation,
        pheromone_threshold=args.pheromone_threshold,
        sample_interval=args.sample_interval
    )

    print(f"\nConversion complete!")
    print(f"  Total samples: {len(dataset.samples)}")

    # Calculate some statistics
    if len(dataset.samples) > 0:
        unique_timesteps = len(set(s.timestep for s in dataset.samples))
        unique_robots = len(set(s.robot_index for s in dataset.samples))
        avg_pheromone = sum(s.pheromone_magnitude for s in dataset.samples) / len(dataset.samples)

        print(f"  Unique timesteps: {unique_timesteps}")
        print(f"  Unique robots: {unique_robots}")
        print(f"  Average pheromone: {avg_pheromone:.4f}")

    # Save to pickle
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\n{'=' * 60}")
    print("Dataset saved!")
    print(f"{'=' * 60}")
    print(f"Output: {output_path}")
    print(f"\nYou can now run clustering with:")
    print(f"  PYTHONPATH=. uv run --extra cpu src/interpretation/clustering/run_kmeans_clustering.py \\")
    print(f"    --data-path {output_path} \\")
    print(f"    --n-clusters 9 --create-video")


if __name__ == '__main__':
    main()
