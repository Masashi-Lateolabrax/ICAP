"""
Verify that simulations with same Individual and generation produce identical results.
"""

import sys
from pathlib import Path
import joblib

sys.path.insert(0, '/home/masashi/Workspace/study.d/ICAP-analysis')

from src.settings import MySettings
from src.config import Simulator
from framework.types.utils import IndividualRecorder

# Load Individual from optimization log
log_path = Path('results/20251027-024639_7bb53c8b/optimization_log.pkl')
recorder = IndividualRecorder.load(str(log_path))

generation = 461
rec = recorder[generation]
individual = rec.best_individual

print(f"Individual info:")
print(f"  Generation: {individual.generation}")
print(f"  Fitness: {individual.get_fitness()}")

# Create two simulators with the same Individual
settings = MySettings()
sim1 = Simulator(settings, individual, render=False)
sim2 = Simulator(settings, individual, render=False)

print(f"\nSimulator RNG seeds:")
print(f"  sim1.rng state: {sim1.rng.bit_generator.state}")
print(f"  sim2.rng state: {sim2.rng.bit_generator.state}")

# Run 10 steps and compare
print(f"\nRunning 10 steps to verify determinism...")

for step in range(10):
    input1_before = sim1.input_ndarray.copy()
    input2_before = sim2.input_ndarray.copy()

    sim1.step()
    sim2.step()

    input1_after = sim1.input_ndarray.copy()
    input2_after = sim2.input_ndarray.copy()

    if not (input1_after == input2_after).all():
        print(f"❌ Step {step}: Simulators diverged!")
        print(f"  Max diff: {abs(input1_after - input2_after).max()}")
        break
else:
    print(f"✓ All 10 steps identical - determinism confirmed")

print(f"\n🔍 Checking if sim1 and sim2 produce same cluster transitions...")

# Load KMeans
kmeans_path = Path('results/20251027-024639_7bb53c8b/analysis_461_564ffb5d_100/kmeans-k3/kmeans_model.joblib')
kmeans, scaler = joblib.load(kmeans_path)

# Predict clusters for both
features1 = sim1.input_ndarray[:, 0:6]
features2 = sim2.input_ndarray[:, 0:6]

scaled1 = scaler.transform(features1)
scaled2 = scaler.transform(features2)

clusters1 = kmeans.predict(scaled1)
clusters2 = kmeans.predict(scaled2)

print(f"Clusters after 10 steps:")
print(f"  sim1: {clusters1}")
print(f"  sim2: {clusters2}")
print(f"  Match: {(clusters1 == clusters2).all()}")
