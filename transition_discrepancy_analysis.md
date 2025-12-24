# Transition Count Discrepancy Analysis

## Summary

The discrepancy between `cluster_timeline.csv` (138 transitions) and `transition_dataset_p0.pkl` (125 transitions) is **NOT a bug**. The datasets come from **different simulation runs** of the same Individual.

## Root Cause

### Dataset Origins

1. **cluster_timeline.csv** (138 transitions)
   - Created: Dec 3 05:49
   - Source: `results/20251027-024639_7bb53c8b/analysis_499_5ae3a28c_499/debug_data.pkl`
   - Original simulation: Oct 28 09:57
   - Method: Exported from saved `DebugData` via `kmeans_clustering.py`
   - Data flow: Single deterministic analysis of saved simulation data

2. **transition_dataset_p0.pkl** (125 transitions)
   - Created: Dec 3 12:23
   - Source: Fresh simulation run
   - Method: Created via `collect_transition_data.py`
   - Data flow: New simulation executed with same Individual

### Why They Differ

The simulations are **stochastic** due to:
- Random food item placement
- MuJoCo physics simulation variability
- Different random seeds
- Simulation timing differences

Even with the **same Individual** (neural network weights from generation 499), different simulation runs produce different robot behaviors and cluster transitions.

## Evidence

### Comparison Results
```
Total transitions:
  cluster_timeline.csv: 138
  transition_dataset_p0.pkl: 125

Overlap analysis:
  Common transitions: 26 (18.8% of CSV, 20.8% of PKL)
  Only in CSV: 112 (81.2%)
  Only in PKL: 99 (79.2%)
```

The minimal overlap (26 common transitions) confirms these are **independent simulation runs**, not different filtering methods on the same data.

### Cluster Label Consistency
All 26 common transitions have matching cluster labels (from_cluster and to_cluster), confirming:
- Both datasets use the same KMeans model
- Both datasets use the same StandardScaler
- Cluster prediction is deterministic and consistent

## Data Collection Methods

### cluster_timeline.csv Creation Flow
```
debug_data.pkl (Oct 28)
  ↓
load DebugData
  ↓
convert_debug_data_to_dataset_filtered()
  - pheromone_threshold filtering
  - sample_interval filtering
  ↓
cluster_sensor_states()
  ↓
export_cluster_timeline_data()
  ↓
cluster_timeline.csv (Dec 3)
```

### transition_dataset_p0.pkl Creation Flow
```
Fresh simulation (Dec 3)
  ↓
TransitionCollector
  - Dual simulator (actual + baseline)
  - Real-time transition detection
  - pheromone_threshold=0.0 filtering
  ↓
collect_transitions()
  ↓
TransitionDataset
  ↓
transition_dataset_p0.pkl (Dec 3)
```

## Key Differences

| Aspect | cluster_timeline.csv | transition_dataset_p0.pkl |
|--------|---------------------|---------------------------|
| **Data source** | Saved debug_data.pkl | Fresh simulation |
| **Simulation date** | Oct 28 09:57 | Dec 3 12:23 |
| **Analysis date** | Dec 3 05:49 | Dec 3 12:23 |
| **Random seed** | Unknown (saved) | Different |
| **Food placement** | Deterministic (saved) | New random placement |
| **Physics outcomes** | Deterministic (saved) | New physics simulation |
| **Transitions (60s)** | 138 | 125 |

## Conclusion

This is **expected behavior**, not a bug. To get matching transition counts, both datasets must be created from the **same DebugData** file:

### Option 1: Create both from saved data
```bash
# Create cluster_timeline.csv from debug_data.pkl
PYTHONPATH=. uv run --extra cpu src/interpretation/clustering/kmeans_clustering.py \
    --data-path results/.../debug_data.pkl \
    --n-clusters 3

# Create transition_dataset from debug_data.pkl (requires implementation)
# Currently, collect_transition_data.py runs fresh simulation
```

### Option 2: Save debug data when collecting transitions
```bash
# Modify collect_transition_data.py to save DebugData
# Then both can analyze the same simulation run
```

## Recommendations

1. **For reproducible analysis**: Always work from saved `DebugData` files
2. **For comparison**: Use the same source data for all analyses
3. **Document which debug_data.pkl was used** for each derived dataset
4. **Consider adding random seed control** to collect_transition_data.py for reproducibility

## File Locations

- `cluster_timeline.csv`: `/home/masashi/Workspace/study.d/ICAP-analysis/results/cluster_timeline.csv`
- `transition_dataset_p0.pkl`: `/home/masashi/Workspace/study.d/ICAP-analysis/results/transition_dataset_p0.pkl`
- Source debug_data.pkl: `/home/masashi/Workspace/study.d/ICAP-analysis/results/20251027-024639_7bb53c8b/analysis_499_5ae3a28c_499/debug_data.pkl`
- Clustering code: `src/interpretation/clustering/kmeans_clustering.py:1388-1389`
- Transition collection code: `src/interpretation/transition_shapley/collect_transition_data.py:94-142`
