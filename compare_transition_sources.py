"""
Compare transitions between cluster_timeline.csv and transition_dataset_p0.pkl
to identify the source of the 13-transition discrepancy.
"""

import pandas as pd
import pickle
from pathlib import Path
import sys

# Add current directory to path so pickle can find the classes
sys.path.insert(0, '/home/masashi/Workspace/study.d/ICAP-analysis')

from src.interpretation.transition_shapley.collect_transition_data import TransitionDataset, TransitionMoment

# Load cluster_timeline.csv and extract transitions
print("=== Loading cluster_timeline.csv ===")
df = pd.read_csv('results/20251027-024639_7bb53c8b/analysis_461_564ffb5d_100/kmeans-k3/cluster_timeline.csv')
df = df[df['timestep'] < 6000]  # Filter to 60s

csv_transitions = []
for robot_idx in df['robot_index'].unique():
    robot_data = df[df['robot_index'] == robot_idx].sort_values('timestep')
    cluster_sequence = robot_data['cluster_id'].values

    for i in range(1, len(cluster_sequence)):
        if cluster_sequence[i] != cluster_sequence[i-1]:
            csv_transitions.append({
                'robot': robot_idx,
                'timestep': int(robot_data.iloc[i]['timestep']),
                'from_cluster': cluster_sequence[i-1],
                'to_cluster': cluster_sequence[i]
            })

print(f"CSV transitions: {len(csv_transitions)}")

# Load transition_dataset_p0.pkl and extract transitions
print("\n=== Loading transition_dataset_p0.pkl ===")
dataset_path = Path('results/20251027-024639_7bb53c8b/analysis_461_564ffb5d_100/kmeans-k3/transition_dataset_p0.pkl')
with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)

pkl_transitions = []
for moment in dataset.moments:
    for i in range(len(moment.robot_indices)):
        pkl_transitions.append({
            'robot': moment.robot_indices[i],
            'timestep': moment.time_step,
            'from_cluster': moment.current_clusters[i],
            'to_cluster': moment.actual_next_clusters[i]
        })

print(f"PKL transitions: {len(pkl_transitions)}")

# Convert to sets for comparison (using (robot, timestep) as key)
csv_set = set((t['robot'], t['timestep']) for t in csv_transitions)
pkl_set = set((t['robot'], t['timestep']) for t in pkl_transitions)

# Find differences
only_in_csv = csv_set - pkl_set
only_in_pkl = pkl_set - csv_set

print(f"\n=== Comparison Results ===")
print(f"Transitions only in CSV: {len(only_in_csv)}")
print(f"Transitions only in PKL: {len(only_in_pkl)}")
print(f"Common transitions: {len(csv_set & pkl_set)}")

# Show transitions only in CSV
if only_in_csv:
    print(f"\n=== Transitions in CSV but NOT in PKL ({len(only_in_csv)} transitions) ===")
    csv_dict = {(t['robot'], t['timestep']): t for t in csv_transitions}
    for robot, timestep in sorted(only_in_csv)[:20]:  # Show first 20
        trans = csv_dict[(robot, timestep)]
        print(f"  Robot {robot} at timestep {timestep} (t={timestep*0.01:.2f}s): "
              f"cluster {trans['from_cluster']} → {trans['to_cluster']}")

# Show transitions only in PKL
if only_in_pkl:
    print(f"\n=== Transitions in PKL but NOT in CSV ({len(only_in_pkl)} transitions) ===")
    pkl_dict = {(t['robot'], t['timestep']): t for t in pkl_transitions}
    for robot, timestep in sorted(only_in_pkl)[:20]:  # Show first 20
        trans = pkl_dict[(robot, timestep)]
        print(f"  Robot {robot} at timestep {timestep} (t={timestep*0.01:.2f}s): "
              f"cluster {trans['from_cluster']} → {trans['to_cluster']}")

# Check for cluster mismatches in common transitions
print(f"\n=== Checking cluster label consistency in common transitions ===")
csv_dict = {(t['robot'], t['timestep']): t for t in csv_transitions}
pkl_dict = {(t['robot'], t['timestep']): t for t in pkl_transitions}

mismatches = []
for robot, timestep in (csv_set & pkl_set):
    csv_trans = csv_dict[(robot, timestep)]
    pkl_trans = pkl_dict[(robot, timestep)]

    if (csv_trans['from_cluster'] != pkl_trans['from_cluster'] or
        csv_trans['to_cluster'] != pkl_trans['to_cluster']):
        mismatches.append({
            'robot': robot,
            'timestep': timestep,
            'csv': (csv_trans['from_cluster'], csv_trans['to_cluster']),
            'pkl': (pkl_trans['from_cluster'], pkl_trans['to_cluster'])
        })

if mismatches:
    print(f"Found {len(mismatches)} cluster label mismatches!")
    for m in mismatches[:10]:  # Show first 10
        print(f"  Robot {m['robot']} at timestep {m['timestep']}: "
              f"CSV={m['csv'][0]}→{m['csv'][1]}, PKL={m['pkl'][0]}→{m['pkl'][1]}")
else:
    print("All common transitions have matching cluster labels ✓")
