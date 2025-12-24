"""
Analyze both transition dataset files to understand the difference.
"""

import pickle
from pathlib import Path
import sys

sys.path.insert(0, '/home/masashi/Workspace/study.d/ICAP-analysis')
from src.interpretation.transition_shapley.collect_transition_data import TransitionDataset, TransitionMoment

# Load both datasets
datasets = {}

for filename in ['transition_dataset.pkl', 'transition_dataset_p0.pkl']:
    path = Path(f'results/{filename}')
    if path.exists():
        with open(path, 'rb') as f:
            datasets[filename] = pickle.load(f)
        print(f"=== {filename} ===")
        ds = datasets[filename]
        print(f"Number of moments: {len(ds.moments)}")
        print(f"Total steps: {ds.total_steps}")
        print(f"n_robots: {ds.n_robots}")
        print(f"n_clusters: {ds.n_clusters}")

        # Count total transitions
        total_transitions = sum(len(m.robot_indices) for m in ds.moments)
        print(f"Total robot transitions: {total_transitions}")

        # Check timestep range
        if len(ds.moments) > 0:
            timesteps = [m.time_step for m in ds.moments]
            print(f"Timestep range: {min(timesteps)} to {max(timesteps)}")
            print(f"Time range: {min(timesteps)*0.01:.2f}s to {max(timesteps)*0.01:.2f}s")

        print()
