import os

NUM_CPU = 10
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={NUM_CPU}"

import time

import numpy as np
from cmaes import CMA
from icecream import ic

import jax
import jax.numpy as jnp

ic(jax.devices("cpu"))

from framework.prelude import Settings
from framework.utils import monitor_gpu_memory, configure_gpu_optimization

from config import PracticalController, PracticalSimulator


def opt_cpu_example():
    print("Initializing simulation...")

    gpu_available = configure_gpu_optimization()
    settings = Settings()

    population_size = 10
    episode_length = int(90 / settings.Simulation.TIME_STEP)
    batch_steps = 1000  # Number of steps to batch together

    optimizer = CMA(
        mean=np.zeros((PracticalController.dim(),), dtype=np.float32),
        sigma=0.1,
        population_size=population_size,
    )

    cpu_devices = jax.devices("cpu")[0:NUM_CPU]
    parameters = jnp.array([optimizer.ask() for _ in range(NUM_CPU)])
    rngs: jax.Array = jax.random.split(jax.random.PRNGKey(0), NUM_CPU)

    print("Creating simulator...")
    init_start = time.perf_counter()

    def sim_initializer(s, p, r):
        controller = PracticalController(p)
        mj_model, sim = PracticalSimulator.new(s, controller, r)
        return sim

    simulators = jax.vmap(sim_initializer, in_axes=(None, 0, 0))(settings, parameters, rngs)
    init_time = time.perf_counter() - init_start
    print(f"Simulator initialization: {init_time:.2f}s")

    if gpu_available:
        print("\nGPU status after initialization:")
        monitor_gpu_memory()

    # Warmup run to compile JIT functions
    print("\nPerforming JIT warmup...")
    warmup_start = time.perf_counter()
    # simulators = jax.pmap(lambda sim: sim.step(), devices=cpu_devices)(simulators)
    simulators = jax.vmap(lambda sim: sim.step())(simulators)
    warmup_time = time.perf_counter() - warmup_start
    print(f"JIT warmup completed: {warmup_time:.2f}s")

    if gpu_available:
        print("\nGPU status after JIT warmup:")
        monitor_gpu_memory()

    # Main simulation loop using multi-step batching for better performance
    print(f"\nStarting main simulation ({episode_length} steps, {batch_steps} steps per batch)...")
    print_timer = time.perf_counter()
    sim_start = time.perf_counter()

    completed_steps = 0
    jax.profiler.start_trace(f"./jax_trace")
    while completed_steps < episode_length:
        step_start = time.perf_counter()

        steps_to_run = min(batch_steps, episode_length - completed_steps)
        simulators = jax.pmap(lambda sim: sim.step_n(steps_to_run), devices=cpu_devices)(simulators)
        completed_steps += steps_to_run

        step_end = time.perf_counter()

        if completed_steps == episode_length or (step_end - print_timer) > 1.0:
            print_timer = step_end

            sim_time = step_end - step_start
            steps_per_sec = steps_to_run / sim_time
            print(f"\nStep {completed_steps}/{episode_length} - {steps_per_sec:.1f} steps/sec")

            if gpu_available:
                monitor_gpu_memory()
    jax.profiler.stop_trace()

    sim_time = time.perf_counter() - sim_start
    total_steps_per_sec = episode_length / sim_time
    print(f"\nSimulation completed: {sim_time:.2f}s ({total_steps_per_sec:.1f} steps/sec)")

    if gpu_available:
        print("\nFinal GPU status:")
        monitor_gpu_memory()

    def extract_loss(sim: PracticalSimulator) -> jax.Array:
        return sim.evaluate()["loss"]

    losses = jax.vmap(extract_loss)(simulators)
    _losses = np.array(losses)

    # Performance summary
    total_time = init_time + warmup_time + sim_time
    print(f"\n{'=' * 60}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'=' * 60}")
    print(f"\nTiming:")
    print(f"  Initialization: {init_time:.2f}s")
    print(f"  JIT warmup: {warmup_time:.2f}s")
    print(f"  Simulation: {sim_time:.2f}s")
    print(f"  Total: {total_time:.2f}s")
    print(f"\nThroughput:")
    print(f"  Steps per second: {total_steps_per_sec:.1f}")
    print(f"  Efficiency: {((NUM_CPU * episode_length) / sim_time) / 1000:.3f}k individual-steps/sec")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    opt_cpu_example()
