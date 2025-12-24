import pandas as pd

# Load the CSV
df = pd.read_csv('results/cluster_timeline.csv')

# Filter to 60s range (6000 timesteps)
df = df[df['timestep'] < 6000]

print(f"\nDataframe shape after filtering: {df.shape}")
print(f"Unique timesteps: {df['timestep'].nunique()}")
print(f"Min timestep: {df['timestep'].min()}, Max timestep: {df['timestep'].max()}")

# Count total transitions
transitions_count = 0
transition_details = []

# Group by robot_index to analyze each robot separately
for robot_idx in df['robot_index'].unique():
    robot_data = df[df['robot_index'] == robot_idx].sort_values('timestep')
    cluster_sequence = robot_data['cluster_id'].values

    # Count transitions for this robot
    robot_transitions = 0
    for i in range(1, len(cluster_sequence)):
        if cluster_sequence[i] != cluster_sequence[i-1]:
            robot_transitions += 1
            transitions_count += 1
            transition_details.append({
                'robot_index': robot_idx,
                'timestep': robot_data.iloc[i]['timestep'],
                'time_seconds': robot_data.iloc[i]['time_seconds'],
                'from_cluster': cluster_sequence[i-1],
                'to_cluster': cluster_sequence[i]
            })

    print(f"Robot {robot_idx}: {robot_transitions} transitions")

print(f"\n総遷移回数: {transitions_count}")
print(f"データポイント数: {len(df)}")
print(f"ロボット数: {df['robot_index'].nunique()}")
print(f"タイムステップ数: {df['timestep'].nunique()}")

# Show first 10 transitions
print(f"\n最初の10個の遷移:")
for i, trans in enumerate(transition_details[:10]):
    print(f"  {i+1}. Robot {trans['robot_index']} at timestep {trans['timestep']} (t={trans['time_seconds']:.2f}s): cluster {trans['from_cluster']} → {trans['to_cluster']}")
