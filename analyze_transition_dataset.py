import pickle
from pathlib import Path
import sys

# Add current directory to path so pickle can find the classes
sys.path.insert(0, '/home/masashi/Workspace/study.d/ICAP-analysis')

from src.interpretation.transition_shapley.collect_transition_data import TransitionDataset, TransitionMoment

# Load the transition dataset
dataset_path = Path('results/20251027-024639_7bb53c8b/analysis_461_564ffb5d_100/kmeans-k3/transition_dataset_p0.pkl')
with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)

print("=== TransitionDataset Summary ===")
print(f"Number of transition moments: {len(dataset.moments)}")
print(f"Number of robots: {dataset.n_robots}")
print(f"Number of clusters: {dataset.n_clusters}")
print(f"Total steps: {dataset.total_steps}")
print(f"Total time: {dataset.total_steps * 0.01:.1f}s")
print()

# Check actual timestep range in moments
if len(dataset.moments) > 0:
    first_timestep = min(m.time_step for m in dataset.moments)
    last_timestep = max(m.time_step for m in dataset.moments)
    print(f"Actual timestep range in data: {first_timestep} to {last_timestep}")
    print(f"Time range: {first_timestep * 0.01:.1f}s to {last_timestep * 0.01:.1f}s")
    print()

# Analyze transitions
total_robot_transitions = 0
cluster_transition_counts = {}

for moment in dataset.moments:
    n_transitions = len(moment.robot_indices)
    total_robot_transitions += n_transitions

    # Count cluster transitions
    for i in range(n_transitions):
        from_cluster = moment.current_clusters[i]
        to_cluster_actual = moment.actual_next_clusters[i]
        to_cluster_baseline = moment.baseline_next_clusters[i]

        key_actual = (from_cluster, to_cluster_actual)
        key_baseline = (from_cluster, to_cluster_baseline)

        cluster_transition_counts[key_actual] = cluster_transition_counts.get(key_actual, 0) + 1

print(f"Total robot transitions across all moments: {total_robot_transitions}")
print()

print("=== Cluster Transition Counts (actual) ===")
for (from_c, to_c), count in sorted(cluster_transition_counts.items()):
    print(f"  Cluster {from_c} → {to_c}: {count} times")
print()

# Show first 5 transition moments
print("=== First 5 Transition Moments ===")
for i, moment in enumerate(dataset.moments[:5]):
    print(f"\nMoment {i+1} (timestep={moment.time_step}):")
    print(f"  Robots that transitioned: {moment.robot_indices}")
    print(f"  Current clusters: {moment.current_clusters}")
    print(f"  Actual next clusters: {moment.actual_next_clusters}")
    print(f"  Baseline next clusters: {moment.baseline_next_clusters}")

    # Check if features are identical
    features_identical = []
    for j in range(len(moment.robot_indices)):
        actual_feat = moment.actual_next_features[j]
        baseline_feat = moment.baseline_next_features[j]
        is_identical = (actual_feat == baseline_feat).all()
        features_identical.append(is_identical)

    print(f"  Features identical (actual vs baseline): {features_identical}")
