"""Visualize SOM U-Matrix

This script loads a trained SOM model and visualizes its U-Matrix (distance map).

Usage:
    PYTHONPATH=. uv run --extra cpu src/interpretation/som/som_visualize_umatrix.py \\
        --output-dir results/som
"""

import argparse
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

try:
    from minisom import MiniSom
    MINISOM_AVAILABLE = True
except ImportError:
    MINISOM_AVAILABLE = False
    print("ERROR: minisom not installed")


def visualize_umatrix(som: MiniSom, output_dir: Path):
    """Visualize U-Matrix only."""
    # Create figure
    plt.figure(figsize=(8, 6))

    # U-Matrix (Unified Distance Matrix)
    umatrix = som.distance_map()
    im = plt.imshow(umatrix.T, cmap='bone_r', interpolation='nearest')
    plt.title('U-Matrix (Distance Map)', fontsize=14, fontweight='bold')
    plt.xlabel('SOM X')
    plt.ylabel('SOM Y')
    plt.colorbar(im, label='Distance')

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / "som_umatrix.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"U-Matrix visualization saved to: {fig_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize SOM U-Matrix")
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory containing SOM model (som_model.pkl)')

    args = parser.parse_args()

    if not MINISOM_AVAILABLE:
        print("ERROR: minisom is not installed.")
        return

    output_dir = Path(args.output_dir)

    # Load SOM model from previous training
    som_model_path = output_dir / "som_model.pkl"
    if not som_model_path.exists():
        print(f"ERROR: SOM model not found at {som_model_path}")
        print("Please run src/interpretation/som/som_train.py first to train the SOM.")
        return

    print(f"Loading SOM model from: {som_model_path}")
    with open(som_model_path, 'rb') as f:
        model_data = pickle.load(f)
        som = model_data['som']
    print("SOM model loaded")

    # Visualize U-Matrix
    visualize_umatrix(som, output_dir)


if __name__ == '__main__':
    main()