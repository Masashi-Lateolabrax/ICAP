import os

NUM_CPU = 3
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={NUM_CPU}"

import time
from functools import partial

import numpy as np
from cmaes import CMA
from icecream import ic

import jax
import jax.numpy as jnp
from flax import nnx

ic(jax.devices("cpu"))

from mujoco import mjx

from framework.prelude import Settings

from config import PracticalController, PracticalSimulator


def opt_cpu_example():
    print("Initializing CPU-optimized simulation...")

    devices = jax.devices("cpu")
    settings = Settings()

    # Program Settings Summary
    population_size = 100
    batch_size = NUM_CPU  # Use number of CPU devices as batch size
    episode_length = int(90 / settings.Simulation.TIME_STEP)
    cma_sigma = 0.1
    random_seed = 0

    print(f"\n{'=' * 60}")
    print(f"PROGRAM SETTINGS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Population size: {population_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Episode length: {episode_length} steps")
    print(f"  Time step: {settings.Simulation.TIME_STEP}s")
    print(f"  CPU devices: {NUM_CPU}")
    print(f"{'=' * 60}")

    dim = PracticalController.dim()

    optimizer = CMA(
        mean=np.zeros((dim,), dtype=np.float32),
        sigma=cma_sigma,
        population_size=population_size,
    )

    parameters = jnp.array([optimizer.ask() for _ in range(batch_size)])

    print("Creating simulator...")
    init_start = time.perf_counter()

    mj_model, simulators = PracticalSimulator.new(
        settings,
        PracticalController(jnp.zeros(dim)),
        jax.random.PRNGKey(random_seed)
    )
    model = mjx.put_model(mj_model)

    @partial(nnx.jit, static_argnames=("n",))
    def jit_duplicate_sim(sim: PracticalSimulator, n: int) -> PracticalSimulator:
        _c, sims = jax.lax.scan(
            lambda c, _x: (c, c.reset(model)),
            init=sim,
            xs=jnp.ones((n,), dtype=jnp.int32),
        )
        return sims

    # @partial(nnx.jit, donate_argnames=("sims",))
    def pmap_step(sims):
        return jax.pmap(lambda s: s.step(model), devices=devices, backend="cpu")(sims)

    @partial(nnx.jit, donate_argnames=("sims",))
    def jit_set_params(sims, params) -> PracticalSimulator:
        return jax.vmap(lambda s, p: s.update(controller=PracticalController(p)))(sims, params)

    @partial(nnx.jit, static_argnames=("n",), donate_argnames=("sims",))
    def jit_step_n(sims, n: int):
        return jax.vmap(lambda s: s.step_n(model, n, unroll))(sims)

    @partial(nnx.jit, donate_argnames=("sims",))
    def jit_reset(sims) -> PracticalSimulator:
        return jax.vmap(lambda sim: sim.reset(model))(sims)

    init_time = time.perf_counter() - init_start
    print(f"Simulator initialization: {init_time:.2f}s")

    # Warmup run to compile JIT functions
    print("\nPerforming JIT warmup...")
    warmup_start = time.perf_counter()
    simulators = simulators.step_n(model, batch_steps)
    warmup_time = time.perf_counter() - warmup_start
    print(f"JIT warmup completed: {warmup_time:.2f}s")

    # Reset simulators before main simulation
    print("\nResetting simulators...")
    reset_start = time.perf_counter()
    simulators = simulators.reset(model)
    reset_time = time.perf_counter() - reset_start
    print(f"Simulator reset completed: {reset_time:.2f}s")

    # Set up batched simulators
    print("\nSetting up batched simulators...")
    simulators = jit_duplicate_sim(simulators, batch_size)
    simulators = jit_set_params(simulators, parameters)
    print(f"Batched simulators ready: {simulators.shape[0]} instances")

    # Main simulation loop using multistep batching for better performance
    print(f"\nStarting main simulation ({episode_length} steps, {batch_steps} steps per batch)...")
    sim_start = step_end = time.perf_counter()

    for step in range(episode_length):
        step_start = time.perf_counter()
        simulators = pmap_step(simulators)
        step_end = time.perf_counter()
        steps_per_sec = 1 / (step_end - step_start)

        if int((step_end - sim_start) * 10) % 10 == 0:
            print(f"\nStep {step}/{episode_length}, {steps_per_sec:.1f} steps/s")

    sim_time = step_end - sim_start
    total_steps_per_sec = episode_length / sim_time
    print(f"\nSimulation completed: {sim_time:.2f}s ({total_steps_per_sec:.1f} steps/sec)")

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
    opt_cpu_example()
