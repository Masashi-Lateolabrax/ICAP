#!/usr/bin/env python3
"""
GPU performance test for pheromone field calculations.
Uses JAX profiler and proper timing like opt_gpu_example.py
"""

import time
from functools import partial

import jax
import jax.numpy as jnp

from framework.prelude import *
from framework.pheromone import PheromoneField
from framework.utils import configure_gpu_optimization, monitor_gpu_memory


def test_pheromone_field_gpu():
    """Test pheromone field calculations on GPU using proper JAX profiling."""

    nx, ny = 1000, 1000
    dt = 0.01

    print("=== Pheromone Field GPU Performance Test ===\n")

    # Configure GPU optimization
    gpu_available = configure_gpu_optimization()

    # Check JAX devices
    gpu_devices = jax.devices("gpu")
    print(f"GPU devices: {gpu_devices}")

    if not gpu_devices:
        print("ERROR: No GPU devices found in JAX")
        return

    device = gpu_devices[0]
    print(f"Using GPU device: {device}")

    # Create pheromone field for testing
    print(f"\nCreating pheromone field: {nx}x{ny} grid")

    pheromone_field = PheromoneField.new(
        nx=nx,
        ny=ny,
        dx=dt,
        temperature=298.15,
        material=ETHANOL,
        evaporation_rate=0.02,
        decrease_rate=-0.001,
    )

    pheromone_field = pheromone_field.add_liquid(int(nx / 2), int(ny / 2), 1.0)

    # Move to GPU
    pheromone_field = jax.device_put(pheromone_field, device)

    print("\nGPU status after initialization:")
    if gpu_available:
        monitor_gpu_memory()

    @jax.jit
    def jit_pheromone_update(field):
        return field.update(dt=dt)

    @partial(jax.jit, static_argnames=['steps'])
    def jit_pheromone_multiple_updates(field, steps):
        return jax.lax.scan(
            lambda c, _x: (c.update(dt=dt), None),
            init=field,
            length=steps,
            unroll=True,
        )[0]

    @partial(jax.jit, static_argnames=['steps'])
    def jit_pheromone_multiple_updates_with_for(field, steps):
        for _ in range(steps):
            field = field.update(dt=dt)
        return field

    # Test 1: Single update with JAX profiler
    print("\n=== Test 1: Single Pheromone Update with Profiler ===")

    # Warmup
    pheromone_field: PheromoneField = jit_pheromone_update(pheromone_field)
    pheromone_field.values_liquid.block_until_ready()

    # Timed run with profiler
    warmup_start = time.perf_counter()
    with jax.profiler.trace("pheromone_single_trace", create_perfetto_link=True):
        pheromone_field: PheromoneField = jit_pheromone_update(pheromone_field)
        pheromone_field.values_liquid.block_until_ready()
        warmup_time = time.perf_counter() - warmup_start
    print(f"Single update with profiler: {warmup_time:.4f}s")

    print("\nGPU status after single update:")
    if gpu_available:
        monitor_gpu_memory()

    # Test 2: Multiple updates with timing
    print("\n=== Test 2: Multiple Pheromone Updates with Timing ===")
    num_steps = 100

    # Warmup
    pheromone_field: PheromoneField = jit_pheromone_multiple_updates(pheromone_field, steps=num_steps)
    pheromone_field.values_liquid.block_until_ready()

    # Timed run with profiler
    start_time = time.perf_counter()
    with jax.profiler.trace("pheromone_multiple_trace", create_perfetto_link=True):
        pheromone_field: PheromoneField = jit_pheromone_multiple_updates(pheromone_field, steps=num_steps)
        pheromone_field.values_liquid.block_until_ready()
        total_time = time.perf_counter() - start_time
    print(f"{num_steps} updates took {total_time:.4f}s, avg {total_time / num_steps:.4f}s per update")

    print("\nGPU status after multiple updates:")
    if gpu_available:
        monitor_gpu_memory()

    # Test 3: Multiple updates with for loop and timing
    print("\n=== Test 3: Multiple Pheromone Updates with For Loop and Timing ===")
    num_steps = 100

    # Warmup
    pheromone_field: PheromoneField = jit_pheromone_multiple_updates_with_for(pheromone_field, steps=num_steps)
    pheromone_field.values_liquid.block_until_ready()

    # Timed run with profiler
    start_time_for = time.perf_counter()
    with jax.profiler.trace("pheromone_multiple_for_trace", create_perfetto_link=True):
        pheromone_field: PheromoneField = jit_pheromone_multiple_updates_with_for(pheromone_field, steps=num_steps)
        pheromone_field.values_liquid.block_until_ready()
        total_time = time.perf_counter() - start_time_for
    print(f"{num_steps} updates with 'for' took {total_time:.4f}s, avg {total_time / num_steps:.4f}s per update")


if __name__ == "__main__":
    test_pheromone_field_gpu()
