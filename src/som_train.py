"""Train Self-Organizing Map (SOM) for Shapley data.

This script trains SOM to organize samples based on non-pheromone sensors.

Features used for SOM (excluding pheromone):
    - robot_sensor (2D): Other robots
    - food_sensor (2D): Food items
    - direction_sensor (2D): Direction to nest

Usage:
    PYTHONPATH=. uv run --extra cpu src/som_train.py \
        --data-file results/20251027-024639_7bb53c8b/shapley_data_subset_10000/samples_dict.pkl \
        --grid-size 10
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

from shapley_data import ShapleyDataset

try:
    from minisom import MiniSom

    MINISOM_AVAILABLE = True
except ImportError:
    MINISOM_AVAILABLE = False
    print("WARNING: minisom not installed. Install with: uv pip install minisom")


def load_dataset(data_path: Path) -> ShapleyDataset:
    """Load Shapley dataset from file.

    Args:
        data_path: Path to .pkl file (either samples_dict.pkl or samples.pkl)

    Returns:
        ShapleyDataset object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    print(f"Loading dataset from: {data_path}")

    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        # Handle both dict format and direct ShapleyDataset format
        if isinstance(data, dict):
            dataset = ShapleyDataset.from_dict(data)
        elif isinstance(data, ShapleyDataset):
            dataset = data
        else:
            raise ValueError(f"Invalid data format: expected dict or ShapleyDataset, got {type(data)}")

    except Exception as e:
        raise ValueError(f"Failed to load dataset from {data_path}: {e}")

    print(f"Loaded {len(dataset)} samples")
    return dataset


def extract_features(dataset: ShapleyDataset) -> np.ndarray:
    """Extract features for SOM.

    Returns:
        som_features: (N, 6) array of non-pheromone sensors
    """
    N = len(dataset)
    som_features = np.zeros((N, 6))

    for i, sample in enumerate(dataset.samples):
        # Non-pheromone sensors for SOM
        som_features[i, 0:2] = sample.robot_sensor
        som_features[i, 2:4] = sample.food_sensor
        som_features[i, 4:6] = sample.direction_sensor

    return som_features


def train_som(features: np.ndarray, grid_size: int, sigma: float = 1.0,
              learning_rate: float = 0.5, num_iterations: int = 10000) -> MiniSom:
    """Train Self-Organizing Map.

    Args:
        features: (N, D) array of input features
        grid_size: Size of SOM grid (grid_size x grid_size)
        sigma: Initial neighborhood radius
        learning_rate: Initial learning rate
        num_iterations: Number of training iterations

    Returns:
        Trained MiniSom object
    """
    if not MINISOM_AVAILABLE:
        raise ImportError("minisom is required. Install with: uv install minisom")

    print(f"\nTraining SOM with {grid_size}x{grid_size} grid...")
    print(f"  Sigma: {sigma}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Iterations: {num_iterations}")

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Initialize SOM
    som = MiniSom(x=grid_size, y=grid_size, input_len=features.shape[1],
                  sigma=sigma, learning_rate=learning_rate,
                  neighborhood_function='gaussian', random_seed=42)

    # Initialize weights
    som.random_weights_init(features_scaled)

    # Train
    som.train_random(features_scaled, num_iterations, verbose=True)

    print("SOM training complete")

    return som, scaler


def main():
    parser = argparse.ArgumentParser(description="SOM analysis for Shapley data")
    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to .pkl file containing Shapley data')
    parser.add_argument('--grid-size', type=int, default=10,
                        help='SOM grid size (grid-size x grid-size, default: 10)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as data file directory)')

    args = parser.parse_args()

    sigma: float = args.grid_size * 0.5 * 0.3
    learning_rate: float = 0.9
    num_iter = 100000

    if not MINISOM_AVAILABLE:
        print("ERROR: minisom is not installed.")
        print("Please install with: uv pip install minisom")
        return

    # Setup paths
    data_path = Path(args.data_file)
    output_dir = Path(args.output_dir) if args.output_dir else data_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_dataset(data_path)

    # Extract features
    som_features = extract_features(dataset)

    # Train SOM
    som, scaler = train_som(
        som_features, args.grid_size, sigma=sigma, learning_rate=learning_rate, num_iterations=num_iter
    )

    # Save SOM model for later use
    som_model_path = output_dir / "som_model.pkl"
    with open(som_model_path, 'wb') as f:
        pickle.dump({'som': som, 'scaler': scaler}, f)
    print(f"SOM model saved to: {som_model_path}")

    print("\n" + "=" * 70)
    print("SOM TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
