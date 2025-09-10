from functools import partial
import time

import jax
import jax.numpy as jnp
from flax import nnx
from mujoco import mjx

from framework.prelude import *
from framework.utils import configure_gpu_optimization, monitor_gpu_memory

from config import PracticalController, PracticalSimulator


def check_usage_of_gpu():
    if not configure_gpu_optimization():
        print("No GPU detected. Exiting.")
        return

    settings = Settings()

    settings.Simulation.TIME_LENGTH = int(90 / settings.Simulation.TIME_STEP)
    batch_steps = 100  # Number of steps to batch together
    dim = PracticalController.dim()

    print("Creating simulator...")
    start_time = time.perf_counter()

    mj_model, simulator = PracticalSimulator.new(
        settings,
        PracticalController(jnp.zeros(dim)),
        jax.random.PRNGKey(0)
    )
    model = mjx.put_model(mj_model)

    @partial(nnx.jit, static_argnames=("n",))
    def jit_step_n(sim, n: int):
        return sim.step_n(model, n)

    @nnx.jit
    def jit_reset(sim):
        return sim.reset(model)

    # DON'T REMOVE: For some reason, calculations don't work on GPU without this first reset
    simulator = jit_reset(simulator)

    init_time = time.perf_counter() - start_time
    print(f"Simulator initialization: {init_time:.2f}s")

    print("\nGPU status after initialization:")
    monitor_gpu_memory()

    # Warmup run to compile JIT functions
    print("\nPerforming JIT warmup...")
    start_time = time.perf_counter()
    simulator = jit_step_n(simulator, batch_steps)
    warmup_time = time.perf_counter() - start_time
    print(f"JIT warmup completed: {warmup_time:.2f}s")

    print("\nGPU status after JIT warmup:")
    monitor_gpu_memory()

    # Reset simulators before main simulation
    print("\nResetting simulators...")
    start_time = time.perf_counter()
    simulator = jit_reset(simulator)
    reset_time = time.perf_counter() - start_time
    print(f"Simulator reset completed: {reset_time:.2f}s")

    print("\nGPU status after simulator reset:")
    monitor_gpu_memory()

    print(
        f"\nStarting main simulation ({settings.Simulation.TIME_LENGTH} steps, "
        f"{batch_steps} steps per batch)..."
    )
    start_time = end_time = time.perf_counter()

    completed_steps = 0
    with jax.profiler.trace("jax_trace", create_perfetto_link=True):
        while completed_steps < settings.Simulation.TIME_LENGTH:
            step_start_time = time.perf_counter()

            steps_to_run = min(batch_steps, settings.Simulation.TIME_LENGTH - completed_steps)
            simulator = jit_step_n(simulator, steps_to_run)
            completed_steps += steps_to_run

            simulator.block_until_ready()
            end_time = time.perf_counter()

            d_time = end_time - step_start_time
            steps_per_sec = steps_to_run / d_time
            print(
                f"\n[{d_time:.2f}s] Step {completed_steps}/{settings.Simulation.TIME_LENGTH}, "
                f"{steps_per_sec:.1f} steps/s"
            )

            monitor_gpu_memory()

    sim_time = end_time - start_time
    total_steps_per_sec = settings.Simulation.TIME_LENGTH / sim_time
    print(f"\nSimulation completed: {sim_time:.2f}s ({total_steps_per_sec:.1f} steps/sec)")

    print("\nFinal GPU status:")
    monitor_gpu_memory()

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
    print(f"{'=' * 60}")


if __name__ == '__main__':
    check_usage_of_gpu()
