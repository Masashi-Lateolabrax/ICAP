"""SOM Training for Robot Sensor Data

Self-Organizing Map training to organize sensor states based on non-pheromone sensors
for behavioral pattern analysis.

Features used (6D, excluding pheromone):
    - robot_sensor (2D): Other robots
    - food_sensor (2D): Food items
    - direction_sensor (2D): Direction to nest

Usage:
    # Basic training
    PYTHONPATH=. uv run --extra cpu src/interpretation/som/som_train.py \\
        --data-path results/analysis/debug_data.pkl \\
        --grid-size 10 \\
        --output-dir results/som

    # With filtering
    PYTHONPATH=. uv run --extra cpu src/interpretation/som/som_train.py \\
        --data-path results/analysis/debug_data.pkl \\
        --grid-size 10 \\
        --pheromone-threshold 0.1 \\
        --sample-interval 2 \\
        --output-dir results/som
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.analysis_mod.structure.debug_data import DebugData
from src.interpretation.clustering.utils import RobotSensorSample, convert_debug_data_to_dataset_filtered

try:
    from minisom import MiniSom

    MINISOM_AVAILABLE = True
except ImportError:
    MINISOM_AVAILABLE = False
    print("WARNING: minisom not installed. Install with: uv pip install minisom")


def extract_features(dataset: list[RobotSensorSample]) -> np.ndarray:
    """Extract features for SOM.

    Returns:
        som_features: (N, 6) array of non-pheromone sensors
    """
    N = len(dataset)
    som_features = np.zeros((N, 6))

    for i, sample in enumerate(dataset):
        # Non-pheromone sensors for SOM
        som_features[i, 0:2] = sample.robot_sensor
        som_features[i, 2:4] = sample.food_sensor
        som_features[i, 4:6] = sample.direction_sensor

    return som_features


def train_som(features: np.ndarray, grid_size: int, sigma: float = 1.0,
              learning_rate: float = 0.5, num_iterations: int = 10000) -> tuple[MiniSom, StandardScaler]:
    """Train Self-Organizing Map.

    Args:
        features: (N, D) array of input features
        grid_size: Size of SOM grid (grid_size x grid_size)
        sigma: Initial neighborhood radius
        learning_rate: Initial learning rate
        num_iterations: Number of training iterations

    Returns:
        Tuple of (trained MiniSom object, fitted StandardScaler)
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
    import sys
    print("About to start training...", file=sys.stderr, flush=True)
    try:
        # Use verbose=False to avoid potential issues with progress bar
        som.train_random(features_scaled, num_iterations, verbose=False)
        print("train_random() returned successfully", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"EXCEPTION in train_random(): {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise
    except:
        print("UNKNOWN ERROR in train_random()", file=sys.stderr, flush=True)
        raise

    print("SOM training complete", file=sys.stderr, flush=True)

    return som, scaler


def main():
    import sys
    print("START: som_train.py", file=sys.stderr, flush=True)

    parser = argparse.ArgumentParser(description="SOM training for robot sensor data")
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to DebugData pickle file')
    parser.add_argument('--grid-size', type=int, default=10,
                        help='SOM grid size (grid-size x grid-size, default: 10)')
    parser.add_argument('--experiment-id', type=str, default='debug_analysis',
                        help='Experiment identifier (default: debug_analysis)')
    parser.add_argument('--generation', type=int, default=0,
                        help='Generation number (default: 0)')
    parser.add_argument('--pheromone-threshold', type=float, default=0.0,
                        help='Only include samples with pheromone >= threshold (default: 0.0)')
    parser.add_argument('--sample-interval', type=int, default=1,
                        help='Only include every Nth timestep (default: 1 = all frames)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as data-path directory)')

    args = parser.parse_args()
    print(f"ARGS: {args}", file=sys.stderr, flush=True)

    sigma: float = args.grid_size * 0.5 * 0.3
    learning_rate: float = 0.9
    num_iter = 100000

    print(f"MINISOM_AVAILABLE: {MINISOM_AVAILABLE}", file=sys.stderr, flush=True)

    if not MINISOM_AVAILABLE:
        print("ERROR: minisom is not installed.", file=sys.stderr, flush=True)
        print("Please install with: uv pip install minisom", file=sys.stderr, flush=True)
        return

    # Setup paths
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir) if args.output_dir else data_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"OUTPUT_DIR: {output_dir}", file=sys.stderr, flush=True)

    # Load DebugData and convert to samples list
    print("Loading DebugData...", file=sys.stderr, flush=True)
    debug_data = DebugData.load(data_path)

    print("Converting to sensor samples...", file=sys.stderr, flush=True)
    dataset = convert_debug_data_to_dataset_filtered(
        debug_data,
        experiment_id=args.experiment_id,
        generation=args.generation,
        pheromone_threshold=args.pheromone_threshold,
        sample_interval=args.sample_interval
    )
    print(f"Total samples: {len(dataset)}", file=sys.stderr, flush=True)

    # Extract features
    print("Extracting features...", file=sys.stderr, flush=True)
    som_features = extract_features(dataset)
    print(f"Features shape: {som_features.shape}", file=sys.stderr, flush=True)

    # Train SOM
    print("Training SOM...", file=sys.stderr, flush=True)
    som, scaler = train_som(
        som_features, args.grid_size, sigma=sigma, learning_rate=learning_rate, num_iterations=num_iter
    )

    # Save SOM model for later use
    print("Saving model...", file=sys.stderr, flush=True)
    som_model_path = output_dir / "som_model.pkl"
    with open(som_model_path, 'wb') as f:
        pickle.dump({'som': som, 'scaler': scaler}, f)
    print(f"SOM model saved to: {som_model_path}")

    print("\n" + "=" * 70)
    print("SOM TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
