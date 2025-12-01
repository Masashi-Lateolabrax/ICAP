"""
Generate animation showing cluster membership over time.

This module creates videos visualizing how robots transition between clusters
during the simulation, similar to input_anime.py but colored by cluster ID.

Usage:
    # Basic animation
    PYTHONPATH=. uv run --extra cpu src/interpretation/clustering/cluster_animation.py \\
        --clustering-result results/clustering/clustering_result.pkl \\
        --dataset results/clustering/dataset.pkl \\
        --output cluster_animation.mp4

    # With statistics panel
    PYTHONPATH=. uv run --extra cpu src/interpretation/clustering/cluster_animation.py \\
        --clustering-result results/clustering/clustering_result.pkl \\
        --dataset results/clustering/dataset.pkl \\
        --output cluster_animation.mp4 \\
        --with-stats --fps 60

    # Custom video settings
    PYTHONPATH=. uv run --extra cpu src/interpretation/clustering/cluster_animation.py \\
        --clustering-result results/clustering/clustering_result.pkl \\
        --dataset results/clustering/dataset.pkl \\
        --output cluster_animation.mp4 \\
        --width 1920 --height 1080 --fps 60 \\
        --no-arrows --no-cluster-info
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from src.interpretation.clustering.kmeans_clustering import KMeansResult
from src.interpretation.clustering.utils import RobotSensorSample


# Default rendering settings (can be overridden)
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 800
DEFAULT_FPS = 30
DEFAULT_WORLD_WIDTH = 3.0
DEFAULT_WORLD_HEIGHT = 3.0


def rotate_vector_2d(vector: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate 2D vector by angle in radians."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    rotation = np.array([[c, -s], [s, c]])
    return rotation @ vector


def draw_arrowed_line(
    img: np.ndarray,
    pos: tuple[int, int],
    direction: np.ndarray,
    length: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    tip_length: float = 0.1
):
    """Draw an arrowed line on the image."""
    end = (
        pos[0] + int(length * direction[0]),
        pos[1] - int(length * direction[1])
    )
    cv2.arrowedLine(img, pos, end, color, thickness, tipLength=tip_length)


def get_cluster_color(cluster_id: int, n_clusters: int) -> tuple[int, int, int]:
    """
    Get BGR color for cluster ID using tab20 colormap.

    Args:
        cluster_id: Cluster ID (0 to n_clusters-1)
        n_clusters: Total number of clusters

    Returns:
        BGR color tuple (0-255 range)
    """
    import matplotlib.pyplot as plt

    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    rgb = colors[cluster_id][:3]  # Get RGB, ignore alpha
    # Convert RGB [0,1] to BGR [0,255]
    bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
    return bgr


def create_cluster_animation(
    result: KMeansResult,
    full_dataset: list[RobotSensorSample],
    pheromone_threshold: float,
    output_path: Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    world_width: float = DEFAULT_WORLD_WIDTH,
    world_height: float = DEFAULT_WORLD_HEIGHT,
    show_arrows: bool = True,
    show_cluster_info: bool = True
):
    """
    Create animation showing cluster membership over time.

    Args:
        result: KMeansResult from cluster_sensor_states()
        full_dataset: All RobotSensorSample including below threshold
        pheromone_threshold: Threshold used for clustering
        output_path: Path to save video file (mp4)
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second
        world_width: Simulation world width
        world_height: Simulation world height
        show_arrows: Whether to show direction arrows
        show_cluster_info: Whether to show cluster distribution info
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    # Add margin for world boundary
    margin = 40  # pixels
    render_width = width - 2 * margin
    render_height = height - 2 * margin

    def world_to_pixel(world_pos: np.ndarray) -> tuple[int, int]:
        """Convert world coordinates to pixel coordinates with margin."""
        world_half = np.array([world_width, world_height]) / 2.0
        world_size = np.array([world_width, world_height])
        render_size = np.array([render_width, render_height])

        normalized = (world_pos[:2] + world_half) / world_size
        pixel_pos = normalized * render_size + np.array([margin, margin])
        pixel_pos[1] = height - pixel_pos[1] + margin  # Flip Y

        pixel_pos = np.clip(pixel_pos, [margin, margin], [width - margin - 1, height - margin - 1])
        return int(pixel_pos[0]), int(pixel_pos[1])

    # Build cluster assignment map from samples above threshold
    print("Building cluster assignment map...")
    cluster_map = {}  # (timestep, robot_index) -> cluster_id
    filtered_idx = 0
    for sample in full_dataset:
        if sample.pheromone_magnitude >= pheromone_threshold:
            key = (sample.timestep, sample.robot_index)
            cluster_map[key] = result.labels[filtered_idx]
            filtered_idx += 1

    # Organize all data by timestep
    print("Organizing data by timestep...")
    timestep_data = {}
    for sample in full_dataset:
        ts = sample.timestep
        if ts not in timestep_data:
            timestep_data[ts] = []

        # Check if this sample has a cluster assignment
        key = (ts, sample.robot_index)
        cluster_id = cluster_map.get(key, None)  # None if below threshold

        timestep_data[ts].append({
            'robot_index': sample.robot_index,
            'position': sample.robot_position,
            'direction': sample.robot_direction,
            'cluster': cluster_id  # None for below threshold
        })

    # Sort timesteps
    timesteps = sorted(timestep_data.keys())

    # Create frames
    buffer = np.zeros((height, width, 3), dtype=np.uint8)
    timer = time.time()

    print(f"Generating animation with {len(timesteps)} frames...")

    for frame_idx, ts in enumerate(timesteps):
        buffer.fill(240)  # Light gray background for margin area

        # Draw world boundary rectangle
        cv2.rectangle(
            buffer,
            (margin, margin),
            (width - margin, height - margin),
            (255, 255, 255),  # White
            -1  # Filled
        )
        # Draw world boundary border
        cv2.rectangle(
            buffer,
            (margin, margin),
            (width - margin, height - margin),
            (0, 0, 0),  # Black border
            2  # Border thickness
        )

        frame_data = timestep_data[ts]

        # Count cluster distribution at this timestep (only clustered robots)
        cluster_counts = {}
        for data in frame_data:
            cid = data['cluster']
            if cid is not None:  # Only count robots above threshold
                cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

        # Draw robots
        for data in frame_data:
            pos = world_to_pixel(data['position'])
            cluster_id = data['cluster']

            # Choose color: cluster color if above threshold, black otherwise
            if cluster_id is not None:
                color = get_cluster_color(cluster_id, result.n_clusters)
            else:
                color = (0, 0, 0)  # Black for below threshold

            # Draw robot as filled circle
            cv2.circle(buffer, pos, 8, color, -1)

            # Draw robot index
            robot_idx = data['robot_index']
            cv2.putText(
                buffer, str(robot_idx),
                (pos[0] - 5, pos[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )

            if show_arrows:
                # Draw robot direction arrow
                direction = data['direction']
                draw_arrowed_line(
                    buffer, pos, direction, 25, color,
                    thickness=2, tip_length=0.3
                )

        # Add timestep info
        cv2.putText(
            buffer, f"Timestep: {ts}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2
        )

        # Add cluster distribution info
        if show_cluster_info:
            y_offset = 60
            cv2.putText(
                buffer, "Cluster Distribution:",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
            )

            y_offset += 25
            for cid in sorted(cluster_counts.keys()):
                count = cluster_counts[cid]
                color = get_cluster_color(cid, result.n_clusters)

                # Draw color box
                cv2.rectangle(
                    buffer,
                    (10, y_offset - 12),
                    (30, y_offset + 2),
                    color, -1
                )

                # Draw text
                cv2.putText(
                    buffer, f"C{cid}: {count}",
                    (35, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
                )

                y_offset += 20

        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
        writer.write(img)

        # Progress reporting
        if time.time() - timer > 1.0:
            print(f"Recording frame {frame_idx + 1}/{len(timesteps)}")
            timer = time.time()

    writer.release()
    print(f"Saved cluster animation to: {output_path}")


def create_cluster_animation_with_stats(
    result: KMeansResult,
    full_dataset: list[RobotSensorSample],
    pheromone_threshold: float,
    output_path: Path,
    width: int = 1200,
    height: int = 800,
    fps: int = DEFAULT_FPS,
    world_width: float = DEFAULT_WORLD_WIDTH,
    world_height: float = DEFAULT_WORLD_HEIGHT
):
    """
    Create animation with cluster statistics panel on the right side.

    This version shows the simulation on the left and cluster statistics
    (centroids, sizes, etc.) on the right panel.

    Args:
        result: KMeansResult from cluster_sensor_states()
        full_dataset: All RobotSensorSample including below threshold
        pheromone_threshold: Threshold used for clustering
        output_path: Path to save video file (mp4)
        width: Total video width (simulation + stats panel)
        height: Video height
        fps: Frames per second
        world_width: Simulation world width
        world_height: Simulation world height
    """
    # Use 2/3 for simulation, 1/3 for stats
    sim_width = int(width * 2 / 3)
    stats_width = width - sim_width

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    # Add margin for world boundary
    margin = 40  # pixels
    render_width = sim_width - 2 * margin
    render_height = height - 2 * margin

    def world_to_pixel(world_pos: np.ndarray) -> tuple[int, int]:
        """Convert world coordinates to pixel coordinates with margin."""
        world_half = np.array([world_width, world_height]) / 2.0
        world_size = np.array([world_width, world_height])
        render_size = np.array([render_width, render_height])

        normalized = (world_pos[:2] + world_half) / world_size
        pixel_pos = normalized * render_size + np.array([margin, margin])
        pixel_pos[1] = height - pixel_pos[1] + margin  # Flip Y

        pixel_pos = np.clip(pixel_pos, [margin, margin], [sim_width - margin - 1, height - margin - 1])
        return int(pixel_pos[0]), int(pixel_pos[1])

    # Build cluster assignment map from samples above threshold
    print("Building cluster assignment map...")
    cluster_map = {}  # (timestep, robot_index) -> cluster_id
    filtered_idx = 0
    for sample in full_dataset:
        if sample.pheromone_magnitude >= pheromone_threshold:
            key = (sample.timestep, sample.robot_index)
            cluster_map[key] = result.labels[filtered_idx]
            filtered_idx += 1

    # Organize all data by timestep
    print("Organizing data by timestep...")
    timestep_data = {}
    for sample in full_dataset:
        ts = sample.timestep
        if ts not in timestep_data:
            timestep_data[ts] = []

        # Check if this sample has a cluster assignment
        key = (ts, sample.robot_index)
        cluster_id = cluster_map.get(key, None)  # None if below threshold

        timestep_data[ts].append({
            'robot_index': sample.robot_index,
            'position': sample.robot_position,
            'direction': sample.robot_direction,
            'cluster': cluster_id  # None for below threshold
        })

    # Get cluster sizes (total)
    cluster_sizes = result.get_cluster_sizes()

    # Sort timesteps
    timesteps = sorted(timestep_data.keys())

    # Create frames
    buffer = np.zeros((height, width, 3), dtype=np.uint8)
    timer = time.time()

    print(f"Generating animation with stats panel ({len(timesteps)} frames)...")

    for frame_idx, ts in enumerate(timesteps):
        buffer.fill(240)  # Light gray background for margin area

        # Draw world boundary rectangle in simulation area
        cv2.rectangle(
            buffer,
            (margin, margin),
            (sim_width - margin, height - margin),
            (255, 255, 255),  # White
            -1  # Filled
        )
        # Draw world boundary border
        cv2.rectangle(
            buffer,
            (margin, margin),
            (sim_width - margin, height - margin),
            (0, 0, 0),  # Black border
            2  # Border thickness
        )

        # Fill stats panel with white
        cv2.rectangle(
            buffer,
            (sim_width, 0),
            (width, height),
            (255, 255, 255),
            -1
        )

        frame_data = timestep_data[ts]

        # Count cluster distribution at this timestep (only clustered robots)
        cluster_counts = {}
        for data in frame_data:
            cid = data['cluster']
            if cid is not None:  # Only count robots above threshold
                cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

        # === LEFT PANEL: Simulation ===
        # Draw robots
        for data in frame_data:
            pos = world_to_pixel(data['position'])
            cluster_id = data['cluster']

            # Choose color: cluster color if above threshold, black otherwise
            if cluster_id is not None:
                color = get_cluster_color(cluster_id, result.n_clusters)
            else:
                color = (0, 0, 0)  # Black for below threshold

            # Draw robot as filled circle
            cv2.circle(buffer, pos, 8, color, -1)

            # Draw robot index
            robot_idx = data['robot_index']
            cv2.putText(
                buffer, str(robot_idx),
                (pos[0] - 5, pos[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )

            # Draw robot direction arrow
            direction = data['direction']
            draw_arrowed_line(
                buffer, pos, direction, 25, color,
                thickness=2, tip_length=0.3
            )

        # Add timestep info
        cv2.putText(
            buffer, f"Timestep: {ts}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2
        )

        # === RIGHT PANEL: Statistics ===
        stats_x = sim_width + 10
        y_offset = 30

        cv2.putText(
            buffer, "Cluster Stats",
            (stats_x, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )

        y_offset += 30
        cv2.putText(
            buffer, f"Total: {len(full_dataset)} samples",
            (stats_x, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )

        y_offset += 30
        cv2.line(buffer, (stats_x, y_offset), (width - 10, y_offset), (0, 0, 0), 1)
        y_offset += 20

        # Current distribution
        cv2.putText(
            buffer, "Current Frame:",
            (stats_x, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )
        y_offset += 25

        for cid in range(result.n_clusters):
            count = cluster_counts.get(cid, 0)
            color = get_cluster_color(cid, result.n_clusters)

            # Draw color box
            cv2.rectangle(
                buffer,
                (stats_x, y_offset - 12),
                (stats_x + 20, y_offset + 2),
                color, -1
            )

            # Draw text
            cv2.putText(
                buffer, f"C{cid}: {count}/9",
                (stats_x + 25, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1
            )

            y_offset += 20

        y_offset += 10
        cv2.line(buffer, (stats_x, y_offset), (width - 10, y_offset), (0, 0, 0), 1)
        y_offset += 20

        # Overall cluster sizes
        cv2.putText(
            buffer, "Overall Distribution:",
            (stats_x, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )
        y_offset += 25

        # Calculate total clustered samples
        total_clustered = sum(cluster_sizes.values())

        for cid in range(result.n_clusters):
            size = cluster_sizes[cid]
            percentage = 100 * size / total_clustered
            color = get_cluster_color(cid, result.n_clusters)

            # Draw color box
            cv2.rectangle(
                buffer,
                (stats_x, y_offset - 12),
                (stats_x + 20, y_offset + 2),
                color, -1
            )

            # Draw text
            cv2.putText(
                buffer, f"C{cid}: {percentage:.1f}%",
                (stats_x + 25, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1
            )

            y_offset += 20

        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
        writer.write(img)

        # Progress reporting
        if time.time() - timer > 1.0:
            print(f"Recording frame {frame_idx + 1}/{len(timesteps)}")
            timer = time.time()

    writer.release()
    print(f"Saved cluster animation with stats to: {output_path}")


# ==============================================================================
# CLI Interface
# ==============================================================================

if __name__ == '__main__':
    import argparse
    import pickle
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='Generate cluster animation video from clustering results'
    )

    parser.add_argument(
        '--clustering-result',
        type=str,
        required=True,
        help='Path to clustering result pickle file (clustering_result.pkl)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to RobotSensorSample list pickle file used for clustering'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output video path (mp4)'
    )
    parser.add_argument(
        '--with-stats',
        action='store_true',
        help='Create video with statistics panel'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=DEFAULT_FPS,
        help=f'Frame rate (default: {DEFAULT_FPS})'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=DEFAULT_WIDTH,
        help=f'Video width (default: {DEFAULT_WIDTH})'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=DEFAULT_HEIGHT,
        help=f'Video height (default: {DEFAULT_HEIGHT})'
    )
    parser.add_argument(
        '--world-width',
        type=float,
        default=DEFAULT_WORLD_WIDTH,
        help=f'Simulation world width (default: {DEFAULT_WORLD_WIDTH})'
    )
    parser.add_argument(
        '--world-height',
        type=float,
        default=DEFAULT_WORLD_HEIGHT,
        help=f'Simulation world height (default: {DEFAULT_WORLD_HEIGHT})'
    )
    parser.add_argument(
        '--no-arrows',
        action='store_true',
        help='Hide direction arrows'
    )
    parser.add_argument(
        '--no-cluster-info',
        action='store_true',
        help='Hide cluster distribution info'
    )

    args = parser.parse_args()

    # Load clustering result
    result_path = Path(args.clustering_result)
    if not result_path.exists():
        raise FileNotFoundError(f"Clustering result not found: {result_path}")

    print(f"Loading clustering result from: {result_path}")
    with open(result_path, 'rb') as f:
        result = pickle.load(f)

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    # Generate animation
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Generating Cluster Animation")
    print(f"{'=' * 60}")
    print(f"Clusters: {result.n_clusters}")
    print(f"Samples: {len(dataset)}")
    print(f"Output: {output_path}")
    print(f"FPS: {args.fps}")
    print(f"With stats: {args.with_stats}")
    print(f"{'=' * 60}\n")

    if args.with_stats:
        create_cluster_animation_with_stats(
            result,
            dataset,
            output_path,
            width=args.width,
            height=args.height,
            fps=args.fps,
            world_width=args.world_width,
            world_height=args.world_height
        )
    else:
        create_cluster_animation(
            result,
            dataset,
            output_path,
            width=args.width,
            height=args.height,
            fps=args.fps,
            world_width=args.world_width,
            world_height=args.world_height,
            show_arrows=not args.no_arrows,
            show_cluster_info=not args.no_cluster_info
        )

    print(f"\n{'=' * 60}")
    print("Animation generation complete!")
    print(f"{'=' * 60}")
    print(f"Saved to: {output_path}")
