"""Test if Simulator can be deep copied."""

import copy
from src.settings import MySettings
from src.config import Simulator
from framework.prelude import Individual
import numpy as np

# Create a test individual
settings = MySettings()
individual = Individual(np.random.randn(settings.Optimization.dimension), generation=0)

# Create simulator
simulator = Simulator(settings, individual, render=False)

# Run a few steps
for _ in range(10):
    simulator.step()

print("Original simulator state:")
print(f"  timer.time: {simulator.timer.time}")
print(f"  output_ndarray[0]: {simulator.output_ndarray[0]}")

# Try deep copy
try:
    simulator_copy = copy.deepcopy(simulator)
    print("\n✓ Deep copy succeeded!")
    print(f"  Copy timer.time: {simulator_copy.timer.time}")
    print(f"  Copy output_ndarray[0]: {simulator_copy.output_ndarray[0]}")

    # Verify independence
    simulator.step()
    print(f"\nAfter original steps:")
    print(f"  Original timer.time: {simulator.timer.time}")
    print(f"  Copy timer.time: {simulator_copy.timer.time}")

    if simulator.timer.time != simulator_copy.timer.time:
        print("✓ Copy is independent!")
    else:
        print("✗ Copy is not independent")

except Exception as e:
    print(f"\n✗ Deep copy failed: {e}")
    print(f"  Error type: {type(e).__name__}")
