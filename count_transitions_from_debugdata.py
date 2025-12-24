"""
Count transitions directly from debug_data.pkl to verify cluster_timeline.csv.
This will be the ground truth.
"""

import sys
from pathlib import Path
import joblib
import numpy as np

sys.path.insert(0, '/home/masashi/Workspace/study.d/ICAP-analysis')

from src.analysis_mod.structure.debug_data import DebugData

# Load debug_data
debug_data_path = Path('results/20251027-024639_7bb53c8b/analysis_499_5ae3a28c_499/debug_data.pkl')
print(f"Loading debug_data from: {debug_data_path}")
debug_data = DebugData.load(debug_data_path)
print(f"Total frames: {len(debug_data)}")

# Load KMeans model and scaler
kmeans_path = Path('results/clustering/kmeans_model.joblib')
if not kmeans_path.exists():
    print(f"\nERROR: KMeans model not found at {kmeans_path}")
    print("Looking for alternative locations...")
    import glob
    models = glob.glob('results/**/kmeans_model.joblib', recursive=True)
    if models:
        kmeans_path = Path(models[0])
        print(f"Found model at: {kmeans_path}")
    else:
        print("No KMeans model found. Exiting.")
        sys.exit(1)

print(f"\nLoading KMeans model from: {kmeans_path}")
kmeans, scaler = joblib.load(kmeans_path)
print(f"Number of clusters: {kmeans.n_clusters}")

# Extract sensor features and predict clusters for all frames (first 6000 = 60s)
max_frames = 6000
n_robots = 9

print(f"\nProcessing first {max_frames} frames (60 seconds)...")

# Store all cluster predictions
all_clusters = np.zeros((max_frames, n_robots), dtype=int)

for frame_idx in range(min(max_frames, len(debug_data))):
    frame = debug_data[frame_idx]

    for robot_idx in range(n_robots):
        # Extract sensor features (first 6 elements of robot_input)
        sensor_features = frame.robot_inputs[robot_idx][0:6]  # (6,)

        # Scale and predict
        sensor_scaled = scaler.transform(sensor_features.reshape(1, -1))  # (1, 6)
        cluster_id = kmeans.predict(sensor_scaled)[0]  # scalar

        all_clusters[frame_idx, robot_idx] = cluster_id

    if (frame_idx + 1) % 1000 == 0:
        print(f"  Processed {frame_idx + 1}/{max_frames} frames")

print("✓ Cluster prediction complete")

# Count transitions
print(f"\n=== Counting Transitions ===")

transitions_total = 0
transitions_by_robot = {}

for robot_idx in range(n_robots):
    robot_clusters = all_clusters[:, robot_idx]  # (max_frames,)
    robot_transitions = 0

    for i in range(1, len(robot_clusters)):
        if robot_clusters[i] != robot_clusters[i-1]:
            robot_transitions += 1
            transitions_total += 1

    transitions_by_robot[robot_idx] = robot_transitions
    print(f"Robot {robot_idx}: {robot_transitions} transitions")

print(f"\n総遷移回数 (from debug_data.pkl): {transitions_total}")

# Compare with cluster_timeline.csv
print(f"\n=== Comparison ===")
print(f"cluster_timeline.csv: 138 transitions")
print(f"debug_data.pkl (direct): {transitions_total} transitions")

if transitions_total == 138:
    print("✓ MATCH! cluster_timeline.csv is correct")
else:
    print(f"✗ MISMATCH! Difference: {abs(transitions_total - 138)} transitions")
