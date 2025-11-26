"""
Run PCA Dimensionality Reduction on Sensor States

Reduces high-dimensional sensor states (6D) to 2D or 3D for visualization
and downstream clustering analysis.

Usage:
    PYTHONPATH=. uv run --extra cpu src/interpretation/clustering/run_pca.py \
        --data-path results/20251027-024639_7bb53c8b/shapley_data/samples.pkl \
        --n-components 2

Parameters:
    --data-path: Path to Shapley dataset pickle file (required)
    --n-components: Number of principal components (default: 2)
    --output-dir: Output directory (default: same directory as data-path)
    --find-optimal: Find optimal number of components for variance threshold
    --variance-threshold: Target variance for optimal component search (default: 0.95)
"""

import argparse
import pickle
from pathlib import Path

from src.interpretation.data_collection.io_sample_definition import ShapleyDataset
from src.interpretation.clustering.pca_clustering import (
    reduce_dimensions,
    visualize_pca,
    visualize_variance_explained,
    visualize_component_loadings,
    find_optimal_components,
    save_pca_result,
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
        description="Run PCA dimensionality reduction on sensor states"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to Shapley dataset pickle file (samples.pkl)"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of principal components to retain (default: 2)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as data-path directory)"
    )
    parser.add_argument(
        "--find-optimal",
        action="store_true",
        help="Find optimal number of components for variance threshold"
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.95,
        help="Target cumulative variance for optimal component search (default: 0.95)"
    )

    args = parser.parse_args()

    # Load dataset
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    dataset = load_dataset(data_path)

    # Set output directory
    if args.output_dir is None:
        output_dir = data_path.parent / "pca"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find optimal components if requested
    if args.find_optimal:
        print("\n" + "=" * 60)
        print("Finding Optimal Number of Components")
        print("=" * 60)
        print(f"Variance threshold: {args.variance_threshold:.1%}")
        print("=" * 60)

        optimal_info = find_optimal_components(
            dataset,
            max_components=6,
            variance_threshold=args.variance_threshold
        )

        print(f"\nOptimal number of components: {optimal_info['optimal_n_components']}")
        print(f"Variance explained: {optimal_info['variance_explained']:.1%}")

        print("\nVariance by component:")
        for i, var_ratio in enumerate(optimal_info['all_variance_ratios'], 1):
            cumulative = optimal_info['cumulative_variance'][i-1]
            print(f"  PC{i}: {var_ratio:.1%} (cumulative: {cumulative:.1%})")

        # Save optimal info
        optimal_path = output_dir / "optimal_components.txt"
        with open(optimal_path, 'w') as f:
            f.write(f"Optimal Component Analysis\n")
            f.write(f"=========================\n")
            f.write(f"Variance threshold: {args.variance_threshold:.1%}\n")
            f.write(f"Optimal components: {optimal_info['optimal_n_components']}\n")
            f.write(f"Variance explained: {optimal_info['variance_explained']:.1%}\n\n")
            f.write("Individual variance ratios:\n")
            for i, var_ratio in enumerate(optimal_info['all_variance_ratios'], 1):
                cumulative = optimal_info['cumulative_variance'][i-1]
                f.write(f"  PC{i}: {var_ratio:.4f} (cumulative: {cumulative:.4f})\n")

        print(f"\nSaved optimal component info to: {optimal_path}")

    # Perform PCA
    print("\n" + "=" * 60)
    print("PCA Configuration")
    print("=" * 60)
    print(f"Number of components: {args.n_components}")
    print(f"Features: 6D sensor states (robot, food, direction)")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    print(f"\nRunning PCA with {args.n_components} components...")
    result = reduce_dimensions(dataset, n_components=args.n_components)

    print(f"\nPCA complete!")
    print(f"  Components: {result.n_components}")
    print(f"  Total variance explained: {result.get_total_variance_explained():.1%}")

    print("\nVariance by component:")
    for i, var_ratio in enumerate(result.explained_variance_ratio, 1):
        print(f"  PC{i}: {var_ratio:.1%}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    if result.n_components >= 2:
        viz_path = output_dir / "pca_2d.png"
        visualize_pca(result, save_path=viz_path)

    variance_path = output_dir / "variance_explained.png"
    visualize_variance_explained(result, save_path=variance_path)

    loadings_path = output_dir / "component_loadings.png"
    visualize_component_loadings(result, save_path=loadings_path)

    # Save PCA results
    results_path = output_dir / "pca_result.pkl"
    save_pca_result(result, results_path)

    # Save statistics
    stats_path = output_dir / "pca_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"PCA Results\n")
        f.write(f"===========\n")
        f.write(f"Number of components: {result.n_components}\n")
        f.write(f"Total variance explained: {result.get_total_variance_explained():.4f}\n\n")

        f.write("Variance by component:\n")
        for i, var_ratio in enumerate(result.explained_variance_ratio, 1):
            f.write(f"  PC{i}: {var_ratio:.4f}\n")

        f.write("\nFeature loadings:\n")
        for i in range(min(result.n_components, 4)):
            f.write(f"\n  PC{i+1}:\n")
            for j, feature_name in enumerate(result.feature_names):
                loading = result.components[i, j]
                f.write(f"    {feature_name}: {loading:+.4f}\n")

    print(f"Saved statistics to: {stats_path}")

    print("\n" + "=" * 60)
    print("PCA analysis complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  Statistics: {stats_path.name}")
    if result.n_components >= 2:
        print(f"  2D Visualization: {viz_path.name}")
    print(f"  Variance Plot: {variance_path.name}")
    print(f"  Component Loadings: {loadings_path.name}")
    print(f"  PCA result: {results_path.name}")


if __name__ == "__main__":
    main()
