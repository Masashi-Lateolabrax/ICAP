"""
Analyze pheromone impact on cluster transitions using Shapley values.

This script calculates Shapley values to quantify pheromone's marginal contribution
to cluster transitions, comparing actual behavior (with pheromones) against
counterfactual baseline (without pheromones).

Shapley Value Interpretation:
    - Positive Shapley value: Pheromone promotes the transition
    - Negative Shapley value: Pheromone inhibits the transition
    - Zero Shapley value: Pheromone has no effect on the transition

Usage:
    PYTHONPATH=. uv run --extra cpu src/interpretation/transition_shapley/analyze_pheromone_shapley.py \
        results/20251027-024639_7bb53c8b/analysis_461_564ffb5d_100/kmeans-k3/transition_dataset_p0.pkl \
        --output-dir results/20251027-024639_7bb53c8b/analysis_461_564ffb5d_100/kmeans-k3/
"""

import argparse
import pickle
import sys
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

# Import data structures from the original collection script
from src.interpretation.transition_shapley.collect_transition_data import (
    TransitionDataset,
    TransitionMoment,
)


@dataclass
class TransitionShapleyValue:
    """Shapley value for a specific cluster transition."""
    from_cluster: int
    to_cluster: int
    shapley_value: float
    n_samples: int  # Number of transitions used for calculation


def calculate_transition_probability(
    moments: List[TransitionMoment],
    from_cluster: int,
    to_cluster: int,
    use_baseline: bool = False
) -> float:
    """
    Calculate transition probability from one cluster to another.

    Args:
        moments: List of transition moments
        from_cluster: Source cluster ID
        to_cluster: Target cluster ID
        use_baseline: If True, use baseline transitions; otherwise use actual

    Returns:
        Transition probability
    """
    total_from_cluster = 0
    transitions_to_target = 0

    for moment in moments:
        for i in range(len(moment.robot_indices)):
            current = moment.current_clusters[i]

            if current == from_cluster:
                total_from_cluster += 1

                if use_baseline:
                    next_cluster = moment.baseline_next_clusters[i]
                else:
                    next_cluster = moment.actual_next_clusters[i]

                if next_cluster == to_cluster:
                    transitions_to_target += 1

    if total_from_cluster == 0:
        return 0.0

    return transitions_to_target / total_from_cluster


def calculate_shapley_value_for_transition(
    dataset: TransitionDataset,
    from_cluster: int,
    to_cluster: int
) -> TransitionShapleyValue:
    """
    Calculate Shapley value for pheromone's contribution to a specific transition.

    The Shapley value measures the average marginal contribution of pheromones
    to the transition probability.

    Args:
        dataset: Transition dataset
        from_cluster: Source cluster ID
        to_cluster: Target cluster ID

    Returns:
        TransitionShapleyValue containing the calculated Shapley value
    """
    # Calculate transition probabilities
    p_actual = calculate_transition_probability(
        dataset.moments, from_cluster, to_cluster, use_baseline=False
    )
    p_baseline = calculate_transition_probability(
        dataset.moments, from_cluster, to_cluster, use_baseline=True
    )

    # Shapley value for binary feature (pheromone present/absent)
    # is simply the difference in outcomes
    shapley_value = p_actual - p_baseline

    # Count number of samples
    n_samples = sum(
        1 for moment in dataset.moments
        for i in range(len(moment.robot_indices))
        if moment.current_clusters[i] == from_cluster
    )

    return TransitionShapleyValue(
        from_cluster=from_cluster,
        to_cluster=to_cluster,
        shapley_value=shapley_value,
        n_samples=n_samples
    )


def analyze_all_transitions(dataset: TransitionDataset) -> Dict[Tuple[int, int], TransitionShapleyValue]:
    """
    Calculate Shapley values for all possible cluster transitions.

    Args:
        dataset: Transition dataset

    Returns:
        Dictionary mapping (from_cluster, to_cluster) to ShapleyValue
    """
    results = {}

    for from_cluster in range(dataset.n_clusters):
        for to_cluster in range(dataset.n_clusters):
            shapley_val = calculate_shapley_value_for_transition(
                dataset, from_cluster, to_cluster
            )
            results[(from_cluster, to_cluster)] = shapley_val

    return results


def print_shapley_analysis(
    dataset: TransitionDataset,
    shapley_values: Dict[Tuple[int, int], TransitionShapleyValue],
    dataset_path: Path
):
    """Print Shapley value analysis results."""
    print("=" * 80)
    print(f"Pheromone Shapley Value Analysis: {dataset_path.name}")
    print("=" * 80)
    print()

    print("## Dataset Information")
    print(f"Total transition moments: {len(dataset.moments)}")
    print(f"Number of robots: {dataset.n_robots}")
    print(f"Number of clusters: {dataset.n_clusters}")
    print(f"Total simulation steps: {dataset.total_steps}")
    print()

    print("## Shapley Value Interpretation")
    print("Shapley value represents pheromone's marginal contribution to transition:")
    print("  Positive value: Pheromone PROMOTES the transition")
    print("  Negative value: Pheromone INHIBITS the transition")
    print("  Zero value:     Pheromone has NO EFFECT on the transition")
    print()

    print("## Shapley Values by Transition")
    print(f"{'From':>6} {'To':>6} {'Shapley':>10} {'Samples':>8} {'Effect':>12}")
    print("-" * 80)

    for from_cluster in range(dataset.n_clusters):
        for to_cluster in range(dataset.n_clusters):
            sv = shapley_values[(from_cluster, to_cluster)]

            # Determine effect type
            if abs(sv.shapley_value) < 0.01:
                effect = "No effect"
            elif sv.shapley_value > 0:
                effect = "Promotes"
            else:
                effect = "Inhibits"

            print(f"C{from_cluster:>5} C{to_cluster:>5} {sv.shapley_value:>10.4f} {sv.n_samples:>8} {effect:>12}")

    print()

    # Summary statistics
    print("## Summary Statistics")

    # Calculate average absolute Shapley value
    non_zero_shapley = [
        sv.shapley_value for sv in shapley_values.values()
        if sv.n_samples > 0
    ]

    if non_zero_shapley:
        print(f"Average Shapley value magnitude: {np.mean(np.abs(non_zero_shapley)):.4f}")
        print(f"Max positive contribution: {max(non_zero_shapley):.4f}")
        print(f"Max negative contribution: {min(non_zero_shapley):.4f}")

        # Identify most affected transitions
        print()
        print("## Most Affected Transitions (by absolute Shapley value)")

        sorted_transitions = sorted(
            [(k, v) for k, v in shapley_values.items() if v.n_samples > 0],
            key=lambda x: abs(x[1].shapley_value),
            reverse=True
        )

        for i, ((from_c, to_c), sv) in enumerate(sorted_transitions[:5], 1):
            effect = "promotes" if sv.shapley_value > 0 else "inhibits"
            print(f"{i}. C{from_c} → C{to_c}: Shapley = {sv.shapley_value:+.4f} "
                  f"(pheromone {effect} this transition)")

    print()

    # Shapley value matrix
    print("## Shapley Value Matrix")
    print("Rows: From Cluster, Columns: To Cluster")
    print("     ", end="")
    for j in range(dataset.n_clusters):
        print(f"C{j:>6} ", end="")
    print()

    for i in range(dataset.n_clusters):
        print(f"C{i:>3}  ", end="")
        for j in range(dataset.n_clusters):
            sv = shapley_values[(i, j)]
            print(f"{sv.shapley_value:>7.3f} ", end="")
        print()
    print()


def plot_shapley_heatmap(
    dataset: TransitionDataset,
    shapley_values: Dict[Tuple[int, int], TransitionShapleyValue],
    output_path: Path
):
    """Create and save heatmap visualization of Shapley values."""
    # Create matrix
    shapley_matrix = np.zeros((dataset.n_clusters, dataset.n_clusters))
    for i in range(dataset.n_clusters):
        for j in range(dataset.n_clusters):
            shapley_matrix[i, j] = shapley_values[(i, j)].shapley_value

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Determine color scale limits
    vmax = max(abs(shapley_matrix.min()), abs(shapley_matrix.max()))
    vmin = -vmax

    # Create heatmap using imshow
    im = ax.imshow(shapley_matrix, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Shapley Value', fontsize=12)

    # Set ticks and labels
    ax.set_xticks(np.arange(dataset.n_clusters))
    ax.set_yticks(np.arange(dataset.n_clusters))
    ax.set_xticklabels([f'C{i}' for i in range(dataset.n_clusters)])
    ax.set_yticklabels([f'C{i}' for i in range(dataset.n_clusters)])

    # Add text annotations
    for i in range(dataset.n_clusters):
        for j in range(dataset.n_clusters):
            text = ax.text(j, i, f'{shapley_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=12)

    ax.set_xlabel('To Cluster', fontsize=12)
    ax.set_ylabel('From Cluster', fontsize=12)
    ax.set_title('Pheromone Shapley Values for Cluster Transitions\n'
                 '(Positive=Promotes, Negative=Inhibits)', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved heatmap: {output_path}")


def plot_transition_comparison(
    dataset: TransitionDataset,
    shapley_values: Dict[Tuple[int, int], TransitionShapleyValue],
    output_path: Path
):
    """Create bar plot comparing actual vs baseline transition probabilities."""
    n_transitions = dataset.n_clusters * dataset.n_clusters
    transitions = []
    actual_probs = []
    baseline_probs = []

    for i in range(dataset.n_clusters):
        for j in range(dataset.n_clusters):
            transitions.append(f'C{i}→C{j}')

            p_actual = calculate_transition_probability(
                dataset.moments, i, j, use_baseline=False
            )
            p_baseline = calculate_transition_probability(
                dataset.moments, i, j, use_baseline=True
            )

            actual_probs.append(p_actual)
            baseline_probs.append(p_baseline)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(transitions))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_probs, width, label='Without Pheromone', alpha=0.8)
    bars2 = ax.bar(x + width/2, actual_probs, width, label='With Pheromone', alpha=0.8)

    ax.set_xlabel('Transition', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Cluster Transition Probabilities: With vs Without Pheromone', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(transitions, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison plot: {output_path}")


def plot_shapley_bar(
    dataset: TransitionDataset,
    shapley_values: Dict[Tuple[int, int], TransitionShapleyValue],
    output_path: Path
):
    """Create bar plot of Shapley values sorted by magnitude."""
    transitions = []
    values = []
    colors = []

    for i in range(dataset.n_clusters):
        for j in range(dataset.n_clusters):
            sv = shapley_values[(i, j)]
            if sv.n_samples > 0:
                transitions.append(f'C{i}→C{j}')
                values.append(sv.shapley_value)
                colors.append('red' if sv.shapley_value < 0 else 'blue')

    # Sort by absolute value
    sorted_indices = np.argsort(np.abs(values))[::-1]
    transitions = [transitions[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(transitions, values, color=colors, alpha=0.7)

    ax.set_xlabel('Shapley Value', fontsize=12)
    ax.set_ylabel('Transition', fontsize=12)
    ax.set_title('Pheromone Shapley Values by Transition\n'
                 '(Blue=Promotes, Red=Inhibits)', fontsize=14)
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved bar plot: {output_path}")


def save_shapley_csv(
    dataset: TransitionDataset,
    shapley_values: Dict[Tuple[int, int], TransitionShapleyValue],
    output_path: Path
):
    """Save Shapley values and transition probabilities to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'from_cluster',
            'to_cluster',
            'shapley_value',
            'prob_with_pheromone',
            'prob_without_pheromone',
            'n_samples',
            'effect'
        ])

        # Write data
        for from_cluster in range(dataset.n_clusters):
            for to_cluster in range(dataset.n_clusters):
                sv = shapley_values[(from_cluster, to_cluster)]

                p_actual = calculate_transition_probability(
                    dataset.moments, from_cluster, to_cluster, use_baseline=False
                )
                p_baseline = calculate_transition_probability(
                    dataset.moments, from_cluster, to_cluster, use_baseline=True
                )

                # Determine effect type
                if abs(sv.shapley_value) < 0.01:
                    effect = "no_effect"
                elif sv.shapley_value > 0:
                    effect = "promotes"
                else:
                    effect = "inhibits"

                writer.writerow([
                    from_cluster,
                    to_cluster,
                    f'{sv.shapley_value:.6f}',
                    f'{p_actual:.6f}',
                    f'{p_baseline:.6f}',
                    sv.n_samples,
                    effect
                ])

    print(f"Saved CSV data: {output_path}")


def save_transition_matrix_csv(
    dataset: TransitionDataset,
    shapley_values: Dict[Tuple[int, int], TransitionShapleyValue],
    output_path: Path
):
    """Save Shapley value matrix in wide format CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        header = ['from_cluster'] + [f'to_C{j}' for j in range(dataset.n_clusters)]
        writer.writerow(header)

        # Write matrix rows
        for i in range(dataset.n_clusters):
            row = [f'C{i}']
            for j in range(dataset.n_clusters):
                sv = shapley_values[(i, j)]
                row.append(f'{sv.shapley_value:.6f}')
            writer.writerow(row)

    print(f"Saved matrix CSV: {output_path}")


def load_transition_dataset(filepath: Path):
    """Load transition dataset from pickle file."""
    with open(filepath, "rb") as f:
        dataset = pickle.load(f)
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Analyze pheromone impact using Shapley values')
    parser.add_argument('dataset_path', type=Path,
                        help='Path to transition dataset (pkl)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Directory to save plots (default: same as dataset)')

    args = parser.parse_args()

    if not args.dataset_path.exists():
        print(f"Error: File not found: {args.dataset_path}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir is None:
        output_dir = args.dataset_path.parent
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset_path}")
    dataset = load_transition_dataset(args.dataset_path)

    print(f"Dataset loaded: {len(dataset.moments)} transition moments")
    print(f"Calculating Shapley values...")
    print()

    shapley_values = analyze_all_transitions(dataset)
    print_shapley_analysis(dataset, shapley_values, args.dataset_path)

    # Generate outputs
    print()
    print("Generating visualizations and data files...")

    # CSV files
    csv_path = output_dir / "pheromone_shapley_data.csv"
    save_shapley_csv(dataset, shapley_values, csv_path)

    matrix_csv_path = output_dir / "pheromone_shapley_matrix.csv"
    save_transition_matrix_csv(dataset, shapley_values, matrix_csv_path)

    # Heatmap
    heatmap_path = output_dir / "pheromone_shapley_heatmap.png"
    plot_shapley_heatmap(dataset, shapley_values, heatmap_path)

    # Transition comparison
    comparison_path = output_dir / "pheromone_transition_comparison.png"
    plot_transition_comparison(dataset, shapley_values, comparison_path)

    # Shapley bar plot
    bar_path = output_dir / "pheromone_shapley_bars.png"
    plot_shapley_bar(dataset, shapley_values, bar_path)

    print()
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()