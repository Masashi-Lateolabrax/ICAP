"""
Check if cluster_timeline.csv was created with pheromone filtering.
We need to examine the original debug_data.pkl to see pheromone values.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, '/home/masashi/Workspace/study.d/ICAP-analysis')

from src.analysis_mod.structure.debug_data import DebugData

# Load the debug_data that was used to create cluster_timeline.csv
debug_data_path = Path('results/20251027-024639_7bb53c8b/analysis_499_5ae3a28c_499/debug_data.pkl')
print(f"Loading debug_data from: {debug_data_path}")
debug_data = DebugData.load(debug_data_path)

print(f"\nDebugData info:")
print(f"  Total frames: {len(debug_data)}")
print(f"  Frame 0 robots: {len(debug_data[0].robot_inputs)}")

# Check first 10 frames for pheromone values
print(f"\n=== Pheromone values in first 10 frames ===")
for frame_idx in range(min(10, len(debug_data))):
    frame = debug_data[frame_idx]
    print(f"\nFrame {frame_idx} (t={frame_idx*0.01:.2f}s):")
    for robot_idx in range(len(frame.robot_inputs)):
        # robot_inputs should be shape (9,): [sensor_features(6), pheromone(3)]
        robot_input = frame.robot_inputs[robot_idx]
        pheromone_input = robot_input[6:9]  # Last 3 values are pheromone
        pheromone_magnitude = (pheromone_input[0]**2 + pheromone_input[1]**2)**0.5
        print(f"  Robot {robot_idx}: pheromone_mag={pheromone_magnitude:.4f}, pheromone_vec={pheromone_input}")

# Load cluster_timeline.csv
csv_df = pd.read_csv('results/cluster_timeline.csv')
csv_df = csv_df[csv_df['timestep'] < 6000]

print(f"\n=== Cluster timeline CSV info ===")
print(f"Total rows (60s): {len(csv_df)}")
print(f"Expected rows (9 robots × 6000 timesteps): {9 * 6000}")
print(f"Unique timesteps: {csv_df['timestep'].nunique()}")

# Check if there are missing timesteps or robots
timesteps_with_all_robots = csv_df.groupby('timestep')['robot_index'].count()
incomplete_timesteps = timesteps_with_all_robots[timesteps_with_all_robots < 9]

if len(incomplete_timesteps) > 0:
    print(f"\n⚠️  Found {len(incomplete_timesteps)} timesteps with < 9 robots!")
    print("This indicates pheromone filtering was applied!")
    print(f"First 10 incomplete timesteps:")
    for ts, count in list(incomplete_timesteps.items())[:10]:
        print(f"  Timestep {ts}: {count} robots (missing {9-count})")
else:
    print("\n✓ All timesteps have 9 robots - no pheromone filtering detected")

# Count transitions with and without considering missing data points
print(f"\n=== Transition analysis ===")

transitions_total = 0
transitions_with_gaps = 0

for robot_idx in range(9):
    robot_data = csv_df[csv_df['robot_index'] == robot_idx].sort_values('timestep')
    cluster_sequence = robot_data['cluster_id'].values
    timesteps = robot_data['timestep'].values

    robot_transitions = 0
    robot_gap_transitions = 0

    for i in range(1, len(cluster_sequence)):
        if cluster_sequence[i] != cluster_sequence[i-1]:
            robot_transitions += 1
            transitions_total += 1

            # Check if there's a gap in timesteps
            if timesteps[i] - timesteps[i-1] > 1:
                robot_gap_transitions += 1
                transitions_with_gaps += 1

    if robot_gap_transitions > 0:
        print(f"Robot {robot_idx}: {robot_transitions} transitions ({robot_gap_transitions} after timestep gaps)")

print(f"\nTotal transitions: {transitions_total}")
print(f"Transitions following timestep gaps: {transitions_with_gaps}")
