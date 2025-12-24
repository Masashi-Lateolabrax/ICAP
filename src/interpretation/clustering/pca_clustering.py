"""
PCA-based Dimensionality Reduction for Shapley Input Samples

PCA (Principal Component Analysis) reduces high-dimensional sensor states
(6D: robot_sensor, food_sensor, direction_sensor) to 2D or 3D representations
for visualization and clustering analysis.

Usage:
    from src.interpretation.clustering.pca_clustering import reduce_dimensions

    dataset = load_dataset("path/to/samples.pkl")
    result = reduce_dimensions(dataset, n_components=2)
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from src.interpretation.data_collection.io_sample_definition import ShapleyDataset


@dataclass
class PCAResult:
    """Results from PCA dimensionality reduction."""

    n_components: int                # Number of principal components
    reduced_features: np.ndarray     # Reduced feature matrix (n_samples, n_components)
    original_features: np.ndarray    # Original normalized features (n_samples, 6)
    feature_names: list[str]         # Original feature names
    explained_variance_ratio: np.ndarray  # Variance explained by each component
    components: np.ndarray           # Principal component vectors

    def get_total_variance_explained(self) -> float:
        """Get total variance explained by selected components."""
        return float(np.sum(self.explained_variance_ratio))


def extract_sensor_features(dataset: ShapleyDataset, normalize: bool = True) -> tuple[np.ndarray, list[str]]:
    """
    Extract 6D sensor features (excluding pheromone).

    Args:
        dataset: ShapleyDataset containing samples
        normalize: Apply standardization (zero mean, unit variance)

    Returns:
        features: (n_samples, 6) array
        feature_names: List of 6 feature names
    """
    if len(dataset.samples) == 0:
        raise ValueError("Dataset is empty")

    features_list = []
    for sample in dataset.samples:
        features_list.append([
            sample.robot_sensor[0],      # robot_sensor_x
            sample.robot_sensor[1],      # robot_sensor_y
            sample.food_sensor[0],       # food_sensor_x
            sample.food_sensor[1],       # food_sensor_y
            sample.direction_sensor[0],  # direction_sensor_x
            sample.direction_sensor[1],  # direction_sensor_y
        ])

    feature_names = [
        'robot_sensor_x', 'robot_sensor_y',
        'food_sensor_x', 'food_sensor_y',
        'direction_sensor_x', 'direction_sensor_y'
    ]

    features = np.array(features_list)

    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    return features, feature_names


def reduce_dimensions(
    dataset: ShapleyDataset,
    n_components: int = 2,
) -> PCAResult:
    """
    Reduce sensor state dimensions using PCA.

    Args:
        dataset: ShapleyDataset containing samples
        n_components: Number of principal components to retain (default: 2)

    Returns:
        PCAResult with reduced features and variance information
    """
    # Extract and normalize features
    features, feature_names = extract_sensor_features(dataset, normalize=True)

    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)

    return PCAResult(
        n_components=n_components,
        reduced_features=reduced_features,
        original_features=features,
        feature_names=feature_names,
        explained_variance_ratio=pca.explained_variance_ratio_,
        components=pca.components_,
    )


def visualize_pca(
    result: PCAResult,
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 8),
):
    """
    Visualize PCA-reduced features in 2D.

    Args:
        result: PCAResult from reduce_dimensions()
        save_path: Path to save figure (None to display)
        figsize: Figure size
    """
    if result.n_components < 2:
        raise ValueError("PCA result must have at least 2 components for visualization")

    plt.figure(figsize=figsize)

    # Scatter plot of first two components
    plt.scatter(
        result.reduced_features[:, 0],
        result.reduced_features[:, 1],
        alpha=0.3,
        s=10,
        c='blue',
    )

    var1 = result.explained_variance_ratio[0]
    var2 = result.explained_variance_ratio[1]

    plt.xlabel(f'PC1 ({var1:.1%} variance)')
    plt.ylabel(f'PC2 ({var2:.1%} variance)')
    plt.title(f'PCA Visualization (Total variance: {result.get_total_variance_explained():.1%})')
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PCA visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_variance_explained(
    result: PCAResult,
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 6),
):
    """
    Visualize variance explained by each principal component.

    Args:
        result: PCAResult from reduce_dimensions()
        save_path: Path to save figure (None to display)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Bar plot
    components = np.arange(1, result.n_components + 1)
    plt.bar(components, result.explained_variance_ratio, alpha=0.6, label='Individual')

    # Cumulative line plot
    cumulative = np.cumsum(result.explained_variance_ratio)
    plt.plot(components, cumulative, 'ro-', label='Cumulative')

    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained Ratio')
    plt.title('Variance Explained by Principal Components')
    plt.xticks(components)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved variance plot to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_component_loadings(
    result: PCAResult,
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 6),
):
    """
    Visualize feature contributions to principal components.

    Args:
        result: PCAResult from reduce_dimensions()
        save_path: Path to save figure (None to display)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Heatmap of component loadings
    components_to_show = min(result.n_components, 4)

    im = plt.imshow(
        result.components[:components_to_show],
        cmap='coolwarm',
        aspect='auto',
        interpolation='nearest',
    )

    plt.colorbar(im, label='Loading')
    plt.yticks(range(components_to_show), [f'PC{i+1}' for i in range(components_to_show)])
    plt.xticks(range(len(result.feature_names)), result.feature_names, rotation=45, ha='right')
    plt.title('Feature Loadings on Principal Components')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved component loadings to: {save_path}")
    else:
        plt.show()

    plt.close()


def find_optimal_components(
    dataset: ShapleyDataset,
    max_components: int = 6,
    variance_threshold: float = 0.95,
) -> dict:
    """
    Find optimal number of PCA components to retain target variance.

    Args:
        dataset: ShapleyDataset containing samples
        max_components: Maximum components to test (default: 6)
        variance_threshold: Target cumulative variance (default: 0.95)

    Returns:
        Dictionary with optimal n_components and variance information
    """
    features, _ = extract_sensor_features(dataset, normalize=True)

    pca = PCA(n_components=max_components)
    pca.fit(features)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    optimal_n = int(np.argmax(cumulative_variance >= variance_threshold) + 1)

    return {
        'optimal_n_components': optimal_n,
        'variance_explained': float(cumulative_variance[optimal_n - 1]),
        'all_variance_ratios': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance': cumulative_variance.tolist(),
    }


def save_pca_result(result: PCAResult, output_path: Path):
    """Save PCA result to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved PCA result to: {output_path}")


def load_pca_result(input_path: Path) -> PCAResult:
    """Load PCA result from disk."""
    with open(input_path, 'rb') as f:
        result = pickle.load(f)
    print(f"Loaded PCA result from: {input_path}")
    return result


# ==============================================================================
# CLI Interface
# ==============================================================================

if __name__ == '__main__':
    import argparse
    from pathlib import Path
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
