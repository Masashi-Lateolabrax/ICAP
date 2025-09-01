import time

import numpy as np
import mujoco
from cmaes import CMA
from mujoco import mjx

import jax
import jax.numpy as jnp
from flax import nnx
from flax.struct import dataclass as jax_dataclass

from framework.prelude import *
from framework.utils import ParaStock
from framework.backends import BasicSimulatorWithEnv
from framework.utils import configure_gpu_optimization, monitor_gpu_memory


class Controller(nnx.Module):
    def __init__(self, parameter: jax.Array):
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

    @staticmethod
    def dim():
        return (16 * 8 + 8) + (8 * 2 + 2)


@jax_dataclass
class Simulator:
    _env_sim: BasicSimulatorWithEnv

    individual: jax.Array
    controller: Controller

    @property
    def data(self) -> mjx.Data:
        return self._env_sim.data

    @property
    def robots(self) -> BatchedRobots:
        return self._env_sim.robots

    @property
    def robot_inputs(self) -> jax.Array:
        return self._env_sim.robot_inputs

    @property
    def loss(self) -> jax.Array:
        return self._env_sim.loss

    def update(
            self,
            data: mjx.Data = None,
            robots: BatchedRobots = None,
            robot_inputs: jax.Array = None,
            loss: jax.Array = None,

            individual: jax.Array = None,
            controller: Controller = None
    ) -> 'Simulator':
        parent_kwargs = {
            "data": data,
            "robots": robots,
            "robot_inputs": robot_inputs,
            "loss": loss,
        }
        env_sim = self._env_sim.update(**parent_kwargs)

        this_kwargs = {
            "_env_sim": env_sim,
            "individual": individual,
            "controller": controller,
        }
        kwargs = {k: v for k, v in this_kwargs.items() if v is not None}
        if not kwargs:
            return self

        return self.replace(**kwargs)

    @classmethod
    def new(cls, settings: Settings, individual: jax.Array, rngs: jax.Array) -> 'Simulator':
        sim = BasicSimulatorWithEnv.new(settings, rngs)
        controller = Controller(individual)
        return cls(
            _env_sim=sim,
            individual=individual,
            controller=controller,
        )

    def add_pheromone(self, positions: jax.Array, amounts: jax.Array) -> 'Simulator':
        new_env_sim = self._env_sim.add_pheromone(positions, amounts)
        return self.replace(_env_sim=new_env_sim)

    @staticmethod
    @nnx.jit
    def _step(this: 'Simulator', dt: float) -> 'Simulator':
        this: "Simulator" = this.replace(_env_sim=this._env_sim.step(dt))

        output = this.controller(this.robot_inputs)
        new_data = this.robots.set_ctrl(this.data, output)

        this = this.add_pheromone(
            this.robots.positions,
            jnp.ones((this.robots.num_robots,), dtype=jnp.float32)
        )

        return this.update(data=new_data)

    def step(self, dt: float) -> 'Simulator':
        return Simulator._step(self, dt)

    @staticmethod
    @nnx.jit
    def _step_n(simulator: "Simulator", n: int, dt: float) -> "Simulator":
        def body_fn(_i, sim: "Simulator"):
            return Simulator._step(sim, dt)

        new_simulator = jax.lax.fori_loop(0, n, body_fn, simulator)
        return new_simulator

    def step_n(self, n: int, dt: float) -> 'Simulator':
        return Simulator._step_n(self, n, dt)

    def reset(self, individual: jax.Array = None, rngs: jax.Array = None) -> 'Simulator':
        new_env_sim = self._env_sim.reset(rngs)
        this = self.replace(_env_sim=new_env_sim)

        controller = Controller(individual) if individual is not None else None
        return this.update(
            individual=individual,
            controller=controller
        )


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
    simulators = jax.vmap(lambda sim: sim.step(settings.Simulation.TIME_STEP))(simulators)
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
        simulators = jax.vmap(lambda sim: sim.step_n(steps_to_run, settings.Simulation.TIME_STEP))(simulators)
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

    def extract_loss(sim: Simulator) -> tuple[jax.Array, jax.Array]:
        total_loss = sim.loss + settings.Loss.REGULARIZATION_COEFFICIENT * jnp.sum(jnp.square(sim.individual))
        return sim.individual, total_loss

    parameters, losses = jax.vmap(extract_loss)(simulators)
    parameters = np.array(parameters)
    losses = np.array(losses)

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
