"""Create a smaller subset of Shapley data for faster processing.

This script loads a large Shapley dataset and creates a smaller randomly
sampled subset for faster Shapley value calculations.

Usage:
    PYTHONPATH=. uv run --extra cpu src/create_shapley_subset.py \
        --data-dir results/20251027-024639_7bb53c8b/shapley_data \
        --n-samples 10000 \
        --output-dir results/20251027-024639_7bb53c8b/shapley_data_subset

Parameters:
    --data-dir: Directory containing original Shapley data
    --n-samples: Number of samples to include in subset
    --output-dir: Output directory for subset (optional, defaults to data-dir with _subset suffix)
    --seed: Random seed for reproducibility (default: 42)
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
from datetime import datetime

from src.interpretation.data_collection.io_sample_definition import ShapleyDataset


def create_subset(data_dir: Path, n_samples: int, output_dir: Path, seed: int = 42):
    """Create a subset of Shapley data.

    Args:
        data_dir: Directory containing original data
        n_samples: Number of samples to include
        output_dir: Output directory
        seed: Random seed
    """
    # Set random seed
    np.random.seed(seed)

    # Load dataset using dict format (faster)
    dict_path = data_dir / "samples_dict.pkl"
    samples_path = data_dir / "samples.pkl"

    print(f"Loading dataset from: {data_dir}")

    if dict_path.exists():
        print(f"  Using dict format: {dict_path}")
        with open(dict_path, 'rb') as f:
            data_dict = pickle.load(f)

        total_samples = len(data_dict['samples'])
        print(f"  Total samples: {total_samples}")

        if n_samples >= total_samples:
            print(f"  Requested {n_samples} >= {total_samples}, using all samples")
            n_samples = total_samples
            indices = np.arange(total_samples)
        else:
            print(f"  Randomly sampling {n_samples} from {total_samples}...")
            indices = np.random.choice(total_samples, n_samples, replace=False)
            indices.sort()  # Sort for better file access patterns

        # Create subset dict
        subset_samples = [data_dict['samples'][i] for i in indices]
        subset_dict = {
            'experiment_id': data_dict['experiment_id'],
            'generation': data_dict['generation'],
            'individual_id': data_dict['individual_id'],
            'simulation_duration': data_dict['simulation_duration'],
            'pheromone_threshold': data_dict['pheromone_threshold'],
            'collection_timestamp': data_dict['collection_timestamp'],
            'samples': subset_samples,
        }

        dataset = ShapleyDataset.from_dict(subset_dict)

    elif samples_path.exists():
        print(f"  Using object format: {samples_path}")
        with open(samples_path, 'rb') as f:
            dataset = pickle.load(f)

        total_samples = len(dataset.samples)
        print(f"  Total samples: {total_samples}")

        if n_samples >= total_samples:
            print(f"  Requested {n_samples} >= {total_samples}, using all samples")
            n_samples = total_samples
        else:
            print(f"  Randomly sampling {n_samples} from {total_samples}...")
            indices = np.random.choice(total_samples, n_samples, replace=False)
            indices.sort()
            dataset.samples = [dataset.samples[i] for i in indices]
    else:
        raise FileNotFoundError(f"No data files found in: {data_dir}")

    # Save subset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as both formats
    print(f"\nSaving subset to: {output_dir}")

    # Save object format
    samples_out = output_dir / "samples.pkl"
    with open(samples_out, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"  Saved object format: {samples_out} ({samples_out.stat().st_size / 1024 / 1024:.1f} MB)")

    # Save dict format
    dict_out = output_dir / "samples_dict.pkl"
    with open(dict_out, 'wb') as f:
        pickle.dump(dataset.to_dict(), f)
    print(f"  Saved dict format: {dict_out} ({dict_out.stat().st_size / 1024 / 1024:.1f} MB)")

    # Save metadata
    metadata_out = output_dir / "metadata.txt"
    with open(metadata_out, 'w') as f:
        f.write(dataset.get_summary())
        f.write(f"\n\nSubset Creation:\n")
        f.write(f"  Original total: {total_samples}\n")
        f.write(f"  Subset size: {len(dataset)}\n")
        f.write(f"  Random seed: {seed}\n")
        f.write(f"  Created: {datetime.now().isoformat()}\n")
    print(f"  Saved metadata: {metadata_out}")

    print(f"\nSubset creation complete!")
    print(f"  Total samples in subset: {len(dataset)}")

    # Show per-robot distribution
    robot_counts = {}
    for sample in dataset.samples:
        robot_counts[sample.robot_index] = robot_counts.get(sample.robot_index, 0) + 1
    print(f"  Samples per robot: {dict(sorted(robot_counts.items()))}")


def main():
    parser = argparse.ArgumentParser(description="Create subset of Shapley data")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing original Shapley data")
    parser.add_argument("--n-samples", type=int, required=True,
                       help="Number of samples to include in subset")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: <data-dir>_subset)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir.parent / f"{data_dir.name}_subset_{args.n_samples}"

    print(f"Creating Shapley data subset")
    print(f"  Input: {data_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Seed: {args.seed}")
    print()

    create_subset(data_dir, args.n_samples, output_dir, args.seed)


if __name__ == "__main__":
    main()
