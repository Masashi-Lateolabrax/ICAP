"""
Check clustering metadata to understand which debug_data was used.
"""

import pickle
from pathlib import Path
import sys

sys.path.insert(0, '/home/masashi/Workspace/study.d/ICAP-analysis')
from src.interpretation.clustering.kmeans_clustering import ClusteringMetadata

metadata_path = Path('results/20251027-024639_7bb53c8b/analysis_461_564ffb5d_100/kmeans-k3/clustering_metadata.pkl')

with open(metadata_path, 'rb') as f:
    metadata = pickle.load(f)

print("=== Clustering Metadata ===")
print(f"Type: {type(metadata)}")
print(f"n_clusters: {metadata.n_clusters}")
print(f"n_samples: {len(metadata.labels)}")
print(f"pheromone_threshold: {metadata.pheromone_threshold}")

# Check if metadata has information about source data
if hasattr(metadata, '__dict__'):
    print("\n=== All attributes ===")
    for key, value in metadata.__dict__.items():
        if not key.startswith('_'):
            print(f"{key}: {value if not isinstance(value, (list, tuple)) or len(value) < 10 else f'{type(value)} of length {len(value)}'}")
