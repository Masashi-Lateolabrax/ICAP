from functools import partial
import time

import numpy as np
from cmaes import CMA

import jax
import jax.numpy as jnp
from flax import nnx

from framework.prelude import *
from framework.utils import configure_gpu_optimization, monitor_gpu_memory

from config import PracticalController, PracticalSimulator


def opt_gpu_example():
    gpu_available = configure_gpu_optimization()

    print("Initializing GPU-optimized simulation...")

    settings = Settings()

    population_size = 10
    batch_size = population_size
    episode_length = int(90 / settings.Simulation.TIME_STEP)
    batch_steps = 10  # Number of steps to batch together

    optimizer = CMA(
        mean=np.zeros((PracticalController.dim(),), dtype=np.float32),
        sigma=0.1,
        population_size=population_size,
    )

    parameters = jnp.array([optimizer.ask() for _ in range(batch_size)])
    rngs: jax.Array = jax.random.split(jax.random.PRNGKey(0), batch_size)

    print("Creating simulator...")
    init_start = time.perf_counter()

    def sim_initializer(s, p, r):
        controller = PracticalController(p)
        mj_model, sim = PracticalSimulator.new(s, controller, r)
        return sim

    @partial(nnx.jit, static_argnames=("n",))
    def jit_step_n(sims, n: int):
        return jax.vmap(lambda s: s.step_n(n))(sims)

    @nnx.jit
    def jit_reset(sims):
        return jax.vmap(lambda sim: sim.reset())(sims)

    simulators = jax.vmap(sim_initializer, in_axes=(None, 0, 0))(settings, parameters, rngs)
    init_time = time.perf_counter() - init_start
    print(f"Simulator initialization: {init_time:.2f}s")

    print("\nGPU status after initialization:")
    monitor_gpu_memory()

    # Warmup run to compile JIT functions
    print("\nPerforming JIT warmup...")
    warmup_start = time.perf_counter()
    with jax.profiler.trace("jax_trace", create_perfetto_link=True):
        simulators = jit_step_n(simulators, batch_steps)
        simulators.block_until_ready()
    warmup_time = time.perf_counter() - warmup_start
    print(f"JIT warmup completed: {warmup_time:.2f}s")

    print("\nGPU status after JIT warmup:")
    monitor_gpu_memory()

    # Reset simulators before main simulation
    reset_start = time.perf_counter()
    simulators = jit_reset(simulators)
    reset_time = time.perf_counter() - reset_start
    print(f"Simulator reset completed: {reset_time:.2f}s")

    print("\nGPU status after simulator reset:")
    monitor_gpu_memory()

    # Main simulation loop using multistep batching for better performance
    print(f"\nStarting main simulation ({episode_length} steps, {batch_steps} steps per batch)...")
    sim_start = time.perf_counter()

    completed_steps = 0
    while completed_steps < episode_length:
        step_start = time.perf_counter()

        steps_to_run = min(batch_steps, episode_length - completed_steps)
        simulators: PracticalSimulator = jit_step_n(simulators, steps_to_run)
        completed_steps += steps_to_run

        simulators.block_until_ready()

        step_end = time.perf_counter()
        d_time = step_end - step_start
        steps_per_sec = steps_to_run / d_time
        print(f"\n[{step_end - step_start:.2f}s] Step {completed_steps}/{episode_length}, {steps_per_sec:.1f} steps/s")

        if gpu_available:
            monitor_gpu_memory()

    sim_time = time.perf_counter() - sim_start
    total_steps_per_sec = episode_length / sim_time
    print(f"\nSimulation completed: {sim_time:.2f}s ({total_steps_per_sec:.1f} steps/sec)")

    if gpu_available:
        print("\nFinal GPU status:")
        monitor_gpu_memory()

    def extract_loss(sim: PracticalSimulator) -> jax.Array:
        return sim.evaluate()["loss"]

    losses = jax.vmap(extract_loss)(simulators)
    losses = np.array(losses)
    parameters = np.array(parameters)

    if population_size == batch_size:
        results = [(para, loss) for para, loss in zip(parameters, losses)]
        optimizer.tell(results)

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
    print(f"  Efficiency: {((batch_size * episode_length) / sim_time) / 1000:.3f}k individual-steps/sec")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    opt_gpu_example()
