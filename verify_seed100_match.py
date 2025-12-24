"""
Verify that transition_dataset_seed100.pkl matches cluster_timeline.csv
"""

import pandas as pd
import pickle
from pathlib import Path
import sys

sys.path.insert(0, '/home/masashi/Workspace/study.d/ICAP-analysis')
from src.interpretation.transition_shapley.collect_transition_data import TransitionDataset, TransitionMoment

# Load cluster_timeline.csv
csv_path = Path('results/20251027-024639_7bb53c8b/analysis_461_564ffb5d_100/kmeans-k3/cluster_timeline.csv')
df = pd.read_csv(csv_path)
df = df[df['timestep'] < 6000]

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

print(f"=== cluster_timeline.csv ===")
print(f"Total transitions: {len(csv_transitions)}")

# Load transition_dataset_seed100.pkl
pkl_path = Path('results/transition_dataset_seed100.pkl')
with open(pkl_path, 'rb') as f:
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

print(f"\n=== transition_dataset_seed100.pkl ===")
print(f"Total transitions: {len(pkl_transitions)}")

# Compare
csv_set = set((t['robot'], t['timestep']) for t in csv_transitions)
pkl_set = set((t['robot'], t['timestep']) for t in pkl_transitions)

only_in_csv = csv_set - pkl_set
only_in_pkl = pkl_set - csv_set
common = csv_set & pkl_set

print(f"\n=== Comparison ===")
print(f"Common transitions: {len(common)}")
print(f"Only in CSV: {len(only_in_csv)}")
print(f"Only in PKL: {len(only_in_pkl)}")

if len(only_in_csv) == 0 and len(only_in_pkl) == 0:
    print("\n✅ PERFECT MATCH! All transitions match between CSV and PKL!")
else:
    print(f"\n⚠️  Mismatch detected")

    if only_in_csv:
        print(f"\nFirst 10 transitions only in CSV:")
        csv_dict = {(t['robot'], t['timestep']): t for t in csv_transitions}
        for robot, timestep in sorted(only_in_csv)[:10]:
            trans = csv_dict[(robot, timestep)]
            print(f"  Robot {robot} at timestep {timestep}: {trans['from_cluster']} → {trans['to_cluster']}")

    if only_in_pkl:
        print(f"\nFirst 10 transitions only in PKL:")
        pkl_dict = {(t['robot'], t['timestep']): t for t in pkl_transitions}
        for robot, timestep in sorted(only_in_pkl)[:10]:
            trans = pkl_dict[(robot, timestep)]
            print(f"  Robot {robot} at timestep {timestep}: {trans['from_cluster']} → {trans['to_cluster']}")

# Check cluster label consistency
if common:
    csv_dict = {(t['robot'], t['timestep']): t for t in csv_transitions}
    pkl_dict = {(t['robot'], t['timestep']): t for t in pkl_transitions}

    mismatches = []
    for robot, timestep in common:
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
        print(f"\n❌ Found {len(mismatches)} cluster label mismatches!")
        for m in mismatches[:5]:
            print(f"  Robot {m['robot']} at timestep {m['timestep']}: CSV={m['csv'][0]}→{m['csv'][1]}, PKL={m['pkl'][0]}→{m['pkl'][1]}")
    else:
        print(f"\n✓ All {len(common)} common transitions have matching cluster labels")
