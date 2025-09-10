# import os

# These settings degrade performance in some case.
# os.environ['XLA_FLAGS'] = " ".join([
#     os.environ.get('XLA_FLAGS', ''),
#     "--xla_gpu_triton_gemm_any=True",
#     "--xla_gpu_enable_latency_hiding_scheduler=true"
# ])

# os.environ['JAX_LOG_COMPILES'] = '1'

from functools import partial
import time

import numpy as np
from cmaes import CMA

import jax

print(jax.devices())

import jax.numpy as jnp
from flax import nnx
from mujoco import mjx

from framework.prelude import *
from framework.utils import (
    configure_gpu_optimization,
    monitor_gpu_memory,
    monitor_gpu_health,
    force_garbage_collection
)

from config import PracticalController, PracticalSimulator


def opt_gpu_example():
    if not configure_gpu_optimization():
        print("No GPU detected. Exiting.")
        return

    print("Initializing GPU-optimized simulation...")

    settings = Settings()

    population_size = 100
    batch_size = 3
    episode_length = int(90 / settings.Simulation.TIME_STEP)
    batch_steps = 100  # Number of steps to batch together
    unroll = 4
    dim = PracticalController.dim()

    optimizer = CMA(
        mean=np.zeros((dim,), dtype=np.float32),
        sigma=0.1,
        population_size=population_size,
    )

    parameters = jnp.array([optimizer.ask() for _ in range(batch_size)])

    print("Creating simulator...")
    init_start = time.perf_counter()

    mj_model, simulators = PracticalSimulator.new(
        settings,
        PracticalController(jnp.zeros(dim)),
        jax.random.PRNGKey(0)
    )
    model = mjx.put_model(mj_model)

    @partial(nnx.jit, static_argnames=("n",))
    def jit_duplicate_sim(sim: PracticalSimulator, n: int):
        _c, sims = jax.lax.scan(
            lambda c, _x: (c, c.reset(model)),
            init=sim,
            xs=jnp.ones((n,), dtype=jnp.int32),
        )
        return sims

    @nnx.jit
    def jit_set_params(sims, params):
        return jax.vmap(lambda s, p: s.update(controller=PracticalController(p)))(sims, params)

    @partial(nnx.jit, static_argnames=("n",))
    def jit_step_n(sims, n: int):
        return jax.vmap(lambda s: s.step_n(model, n, unroll))(sims)

    @nnx.jit
    def jit_reset(sims):
        return jax.vmap(lambda sim: sim.reset(model))(sims)

    simulators = jit_duplicate_sim(simulators, batch_size)
    simulators = jit_set_params(simulators, parameters)

    init_time = time.perf_counter() - init_start
    print(f"Simulator initialization: {init_time:.2f}s")

    print("\nGPU status after initialization:")
    monitor_gpu_memory()

    # Warmup run to compile JIT functions
    print("\nPerforming JIT warmup...")
    warmup_start = time.perf_counter()
    simulators = jit_step_n(simulators, batch_steps)
    warmup_time = time.perf_counter() - warmup_start
    print(f"JIT warmup completed: {warmup_time:.2f}s")

    print("\nGPU status after JIT warmup:")
    monitor_gpu_memory()

    # Reset simulators before main simulation
    print("\nResetting simulators...")
    reset_start = time.perf_counter()
    simulators = jit_reset(simulators)
    reset_time = time.perf_counter() - reset_start
    print(f"Simulator reset completed: {reset_time:.2f}s")

    print("\nGPU status after simulator reset:")
    monitor_gpu_memory()

    # Main simulation loop using multistep batching for better performance
    print(f"\nStarting main simulation ({episode_length} steps, {batch_steps} steps per batch)...")
    sim_start = step_end = time.perf_counter()

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

        print(f"\n[{d_time:.2f}s] Step {completed_steps}/{episode_length}, {steps_per_sec:.1f} steps/s")

        # Monitor GPU health with centralized utility
        monitor_gpu_health()

        # Force garbage collection every 200 steps to prevent memory fragmentation
        # This is essential for long-running simulations. If you run without this,
        # you may observe degraded performance over time due to GPU memory fragmentation.
        if completed_steps % 500 == 0:
            force_garbage_collection()
            print(f"GC triggered at step {completed_steps}")

        monitor_gpu_memory()

    sim_time = step_end - sim_start
    total_steps_per_sec = episode_length / sim_time
    print(f"\nSimulation completed: {sim_time:.2f}s ({total_steps_per_sec:.1f} steps/sec)")

    print("\nFinal GPU status:")
    monitor_gpu_memory()

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
