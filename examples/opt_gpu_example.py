import time
from typing import Self

import numpy as np
from cmaes import CMA

import mujoco
from mujoco import mjx

import jax
import jax.numpy as jnp
from flax import nnx
from flax.struct import dataclass as jax_dataclass

from framework.prelude import *
from framework.utils import ParaStock
from framework.utils import configure_gpu_optimization, monitor_gpu_memory

from framework.backends import SimulatorWithCtrl, ControllerInterface


class Controller(ControllerInterface):
    def __init__(self, parameter: jax.Array):
        self.l2 = jnp.linalg.norm(parameter)

        rngs = nnx.Rngs(0)
        parameter = ParaStock(parameter)

        self.layer1 = nnx.Linear(
            in_features=16,
            out_features=8,
            kernel_init=parameter.gen_initializer(16 * 8),
            bias_init=parameter.gen_initializer(8),
            rngs=rngs,
        )
        self.layer2 = nnx.Linear(
            in_features=8,
            out_features=2,
            kernel_init=parameter.gen_initializer(8 * 2),
            bias_init=parameter.gen_initializer(2),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.relu(self.layer1(x))
        x = jnp.clip(self.layer2(x), -0.3, 1.0)
        return x

    def forward(self, x: RobotInputs) -> RobotOutputs:
        x = self.__call__(x.ray)
        return RobotOutputs(
            left_wheel=x[:, 0],
            right_wheel=x[:, 1],
            pheromone=jnp.zeros((x.shape[0],), dtype=jnp.float32),
        )

    def reset(self) -> Self:
        return self

    @staticmethod
    def dim():
        return (16 * 8 + 8) + (8 * 2 + 2)


@jax_dataclass
class Simulator(SimEvaluateTrait):
    _parent_sim: SimulatorWithCtrl

    @property
    def data(self) -> mjx.Data:
        return self._parent_sim.data

    @property
    def controller(self) -> Controller:
        return self._parent_sim.controller

    def _update_parent(self, **kwargs: dict) -> Self:
        return self.replace(_parent_sim=self._parent_sim.update(**kwargs))

    def update(
            self,
            data: mjx.Data = None,
            controller: Controller = None,
            **kwargs
    ) -> Self:
        kwargs["data"] = data
        kwargs["controller"] = controller
        return self._update(**kwargs)

    @classmethod
    def new(cls, settings: Settings, controller: Controller, rngs: jax.Array) -> tuple[mujoco.MjModel, 'Simulator']:
        mj_model, sim = SimulatorWithCtrl.new(settings, controller, rngs)
        return mj_model, cls(_parent_sim=sim)

    @staticmethod
    @nnx.jit
    def _step(this: 'Simulator') -> 'Simulator':
        this: "Simulator" = this.update(_parent_sim=this._parent_sim.step())
        return this

    def step(self) -> Self:
        return Simulator._step(self)

    @staticmethod
    @nnx.jit
    def _step_n(this: "Simulator", n: int) -> "Simulator":
        def body_fn(_i, sim: "Simulator"):
            return Simulator._step(sim)

        this = jax.lax.fori_loop(0, n, body_fn, this)
        return this

    def step_n(self, n: int) -> Self:
        return Simulator._step_n(self, n)

    def reset(self) -> Self:
        return self.update(
            _parent_sim=self._parent_sim
        )

    def evaluate(self) -> dict:
        result = self._parent_sim.evaluate()
        result["l2"] = self.controller.l2
        return result


def opt_gpu_example():
    gpu_available = configure_gpu_optimization()

    print("Initializing GPU-optimized simulation...")

    settings = Settings()

    population_size = 100
    simulation_steps = int(90 / settings.Simulation.TIME_STEP)
    batch_steps = 100  # Number of steps to batch together

    settings.Robot.NUM = 1
    settings.Food.NUM = 1

    optimizer = CMA(
        mean=np.zeros((Controller.dim(),), dtype=np.float32),
        sigma=0.1,
        population_size=population_size,
    )

    parameters = jnp.array([optimizer.ask() for _ in range(population_size)])
    rngs: jax.Array = jax.random.split(jax.random.PRNGKey(0), population_size)

    print("Creating simulator...")
    init_start = time.perf_counter()
    simulators = jax.vmap(Simulator.new, in_axes=(None, 0, 0))(settings, parameters, rngs)
    init_time = time.perf_counter() - init_start
    print(f"Simulator initialization: {init_time:.2f}s")

    print("\nGPU status after initialization:")
    monitor_gpu_memory()

    # Warmup run to compile JIT functions
    print("\nPerforming JIT warmup...")
    warmup_start = time.perf_counter()
    simulators = jax.vmap(lambda sim: sim.step())(simulators)
    warmup_time = time.perf_counter() - warmup_start
    print(f"JIT warmup completed: {warmup_time:.2f}s")

    print("\nGPU status after JIT warmup:")
    monitor_gpu_memory()

    # Main simulation loop using multi-step batching for better performance
    print(f"\nStarting main simulation ({simulation_steps} steps, {batch_steps} steps per batch)...")
    sim_start = time.perf_counter()

    completed_steps = 0
    while completed_steps < simulation_steps:
        steps_to_run = min(batch_steps, simulation_steps - completed_steps)
        simulators = jax.vmap(lambda sim: sim.step_n(steps_to_run))(simulators)
        completed_steps += steps_to_run

        if completed_steps % 100 == 0 or completed_steps == simulation_steps:
            elapsed = time.perf_counter() - sim_start
            steps_per_sec = completed_steps / elapsed
            print(f"\nStep {completed_steps}/{simulation_steps} - {steps_per_sec:.1f} steps/sec")
            if gpu_available:
                monitor_gpu_memory()

    sim_time = time.perf_counter() - sim_start
    total_steps_per_sec = simulation_steps / sim_time
    print(f"\nSimulation completed: {sim_time:.2f}s ({total_steps_per_sec:.1f} steps/sec)")

    if gpu_available:
        print("\nFinal GPU status:")
        monitor_gpu_memory()

    losses = jax.vmap(lambda sim: sim.loss)(simulators)
    losses = np.array(losses)
    parameters = np.array(parameters)

    results = [(para, float(loss)) for para, loss in zip(parameters, losses)]
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
    print(f"  Efficiency: {((population_size * simulation_steps) / sim_time) / 1000:.1f}k individual-steps/sec")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    opt_gpu_example()
