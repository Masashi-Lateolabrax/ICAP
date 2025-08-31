import numpy as np
import mujoco
from mujoco import mjx

import jax
import jax.numpy as jnp
from flax import nnx
from flax.struct import dataclass as jax_dataclass

from framework.prelude import *
from framework.utils import ParaStock
from framework.backends import BasicSimulatorWithEnv


class Controller(nnx.Module):
    def __init__(self, parameter: jax.Array):
        rngs = nnx.Rngs(0)
        parameter = ParaStock(parameter)

        self.layer1 = nnx.Linear(
            in_features=8,
            out_features=8,
            kernel_init=parameter.gen_initializer(8 * 8),
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
        return (8 * 8 + 8) + (8 * 2 + 2)


@jax_dataclass
class Simulator:
    _env_sim: BasicSimulatorWithEnv

    individual: jax.Array
    controller: Controller

    delta_loss: jax.Array

    OFFSET_ROBOT_AND_FOOD: float
    SIGMA_ROBOT_AND_FOOD: float
    GAIN_ROBOT_AND_FOOD: float

    OFFSET_NEST_AND_FOOD: float
    SIGMA_NEST_AND_FOOD: float
    GAIN_NEST_AND_FOOD: float

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
    def food_items(self) -> BatchedFood:
        return self._env_sim.food_items

    @property
    def NEST_POSITION(self) -> jax.Array:
        return self._env_sim.NEST_POSITION

    def update(
            self,
            rngs: jax.Array = None,
            model: mjx.Model = None,
            data: mjx.Data = None,
            robots: BatchedRobots = None,
            robot_inputs: jax.Array = None,
            food_items: BatchedFood = None,

            individual: jax.Array = None,
            controller: Controller = None,
            delta_loss: float = None,
    ) -> 'Simulator':
        parent_kwargs = {
            "rngs": rngs,
            "model": model,
            "data": data,
            "robots": robots,
            "robot_inputs": robot_inputs,
            "food_items": food_items,
        }
        env_sim = self._env_sim.update(**parent_kwargs)

        this_kwargs = {
            "_env_sim": env_sim,
            "individual": individual,
            "controller": controller,
            "delta_loss": delta_loss,
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
            delta_loss=jnp.zeros((1,), dtype=jnp.float32),

            OFFSET_ROBOT_AND_FOOD=settings.Loss.OFFSET_ROBOT_AND_FOOD,
            SIGMA_ROBOT_AND_FOOD=settings.Loss.SIGMA_ROBOT_AND_FOOD,
            GAIN_ROBOT_AND_FOOD=settings.Loss.GAIN_ROBOT_AND_FOOD,

            OFFSET_NEST_AND_FOOD=settings.Loss.OFFSET_NEST_AND_FOOD,
            SIGMA_NEST_AND_FOOD=settings.Loss.SIGMA_NEST_AND_FOOD,
            GAIN_NEST_AND_FOOD=settings.Loss.GAIN_NEST_AND_FOOD,
        )

    def get_pheromone(self, positions: jax.Array) -> jax.Array:
        return self._env_sim.get_pheromone(positions)

    def add_pheromone(self, positions: jax.Array, amounts: jax.Array) -> 'Simulator':
        new_env_sim = self._env_sim.add_pheromone(positions, amounts)
        return self.replace(_env_sim=new_env_sim)

    @staticmethod
    @nnx.jit
    def _calc_loss_between_robots_and_food(this: "Simulator") -> jax.Array:
        subs = (this.robots.positions[:, None, :2] - this.food_items.positions[None, :, :2]).reshape(-1, 2)
        distance = jnp.clip(
            jnp.linalg.norm(subs, axis=1) - this.OFFSET_ROBOT_AND_FOOD,
            a_min=0
        )
        rf_loss = -jnp.sum(jnp.exp(-(distance ** 2) / this.SIGMA_ROBOT_AND_FOOD))
        rf_loss = rf_loss * this.GAIN_ROBOT_AND_FOOD
        return rf_loss

    @staticmethod
    @nnx.jit
    def _calc_loss_between_food_and_nest(this: "Simulator") -> jax.Array:
        distance_between_food_and_nest = jnp.linalg.norm(
            this.food_items.positions[:, :2] - this.NEST_POSITION,
            axis=1
        )
        distance = jnp.clip(
            distance_between_food_and_nest - this.OFFSET_NEST_AND_FOOD,
            a_min=0
        )
        fn_loss = -jnp.sum(jnp.exp(-(distance ** 2) / this.SIGMA_NEST_AND_FOOD))
        fn_loss = fn_loss * this.GAIN_NEST_AND_FOOD
        return fn_loss

    @staticmethod
    @nnx.jit
    def _step(this: 'Simulator', dt: float) -> 'Simulator':
        this = this.replace(_env_sim=this._env_sim.step(dt))

        output = this.controller(this.robot_inputs)
        new_data = this.robots.set_ctrl(this.data, output)

        this = this.add_pheromone(this.robots.positions, jnp.ones((this.robots.num_robots,), dtype=jnp.float32))

        rf_loss = Simulator._calc_loss_between_robots_and_food(this)
        fn_loss = Simulator._calc_loss_between_food_and_nest(this)
        delta_loss = rf_loss + fn_loss

        return this.update(data=new_data, delta_loss=delta_loss)

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

    def render(
            self,
            mj_model: mujoco.MjModel,
            img_buf: np.ndarray,
            pos: tuple[float, float, float],
            lookat: tuple[float, float, float],
            max_geom=100,
            max_pheromone=1.0
    ):
        self._env_sim.render(mj_model, img_buf, pos, lookat, max_geom, max_pheromone)

    def reset(self, individual: jax.Array = None, rngs: jax.Array = None) -> 'BasicSimulatorWithEnv':
        new_env_sim = self._env_sim.reset(rngs)

        controller = Controller(individual) if individual is not None else None
        this = self.replace(_env_sim=new_env_sim)
        return this.update(
            individual=individual,
            controller=controller
        )

#
#
# def configure_gpu_optimization():
#     """Configure JAX for optimal GPU utilization"""
#     import os
#
#     # Enable XLA optimizations (compatible flags)
#     os.environ['XLA_FLAGS'] = (
#         '--xla_gpu_enable_latency_hiding_scheduler=true '
#         '--xla_gpu_enable_highest_priority_async_stream=true '
#         '--xla_gpu_deterministic_ops=false '
#         '--xla_gpu_autotune_level=4'
#     )
#
#     # JAX configuration for GPU memory management
#     os.environ['JAX_ENABLE_X64'] = 'False'  # Use 32-bit for better GPU performance
#
#     import jax
#
#     try:
#         # First check available platforms
#         available_devices = jax.devices()
#         gpu_devices = [d for d in available_devices if d.platform == 'gpu']
#
#         if gpu_devices:
#             jax.config.update('jax_platform_name', 'gpu')
#             print(f"GPU optimization configured. Available GPUs: {len(gpu_devices)}")
#             return True
#         else:
#             print("Warning: No GPU devices found, using CPU with optimizations")
#             # Apply CPU optimizations instead
#             os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=true'
#             return False
#     except Exception as e:
#         print(f"Platform detection failed: {e}. Using CPU optimizations.")
#         os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=true'
#         return False
#
#
# def monitor_gpu_memory():
#     """Monitor GPU memory usage"""
#     try:
#         import subprocess
#         result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
#                                  '--format=csv,noheader,nounits'],
#                                 capture_output=True, text=True, timeout=5)
#         if result.returncode == 0:
#             lines = result.stdout.strip().split('\n')
#             for i, line in enumerate(lines):
#                 memory_used, memory_total, gpu_util = line.split(', ')
#                 memory_usage_pct = (int(memory_used) / int(memory_total)) * 100
#                 print(f"GPU {i}: {gpu_util}% util, {memory_usage_pct:.1f}% memory ({memory_used}MB/{memory_total}MB)")
#     except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
#         print("GPU monitoring unavailable (nvidia-smi not found or failed)")
#     except Exception as e:
#         print(f"GPU monitoring error: {e}")
#
#
# def jaxable_example():
#     import time
#     from framework.optimization import CMAES
#
#     # Configure GPU optimization first
#     gpu_available = configure_gpu_optimization()
#
#     print("Initializing GPU-optimized simulation...")
#
#     settings = Settings()
#
#     # GPU-optimized batch size - larger batches better utilize GPU parallelization
#     batch_size = 100
#     simulation_steps = int(90 / settings.Simulation.TIME_STEP)
#
#     settings.Render.RENDER_WIDTH = 480
#     settings.Render.RENDER_HEIGHT = 320
#
#     settings.Pheromone.ACTIVE = True
#
#     settings.Robot.NUM = 1
#     settings.Food.NUM = 1
#
#     cmaes = CMAES(
#         dimension=Controller.dim(),
#         max_generation=10,
#         mean=None,
#         sigma=0.1,
#         population_size=batch_size
#     )
#
#     # Display JAX device information
#     print(f"JAX devices: {jax.devices()}")
#     print(f"JAX default backend: {jax.default_backend()}")
#
#     # Monitor initial GPU state
#     print("\nInitial GPU status:")
#     monitor_gpu_memory()
#
#     individuals = cmaes.get_individuals(batch_size)
#     print(f"\nCreated {len(individuals)} individuals for simulation")
#
#     # Initialize simulator
#     print("Creating simulator...")
#     init_start = time.perf_counter()
#     simulator = Simulator(settings, individuals)
#     init_time = time.perf_counter() - init_start
#     print(f"Simulator initialization: {init_time:.2f}s")
#
#     print("\nGPU status after initialization:")
#     monitor_gpu_memory()
#
#     # Warmup run to compile JIT functions
#     print("\nPerforming JIT warmup...")
#     warmup_start = time.perf_counter()
#     simulator.step()  # Single warmup step
#     warmup_time = time.perf_counter() - warmup_start
#     print(f"JIT warmup completed: {warmup_time:.2f}s")
#
#     print("\nGPU status after JIT warmup:")
#     monitor_gpu_memory()
#
#     # Main simulation loop using multi-step batching for better performance
#     batch_steps = 100  # Number of steps to batch together
#     print(f"\nStarting main simulation ({simulation_steps} steps, {batch_steps} steps per batch)...")
#     sim_start = time.perf_counter()
#
#     completed_steps = 0
#     while completed_steps < simulation_steps:
#         steps_to_run = min(batch_steps, simulation_steps - completed_steps)
#         simulator.multi_step(steps_to_run)
#         completed_steps += steps_to_run
#
#         if completed_steps % 100 == 0 or completed_steps == simulation_steps:
#             elapsed = time.perf_counter() - sim_start
#             steps_per_sec = completed_steps / elapsed
#             print(f"\nStep {completed_steps}/{simulation_steps} - {steps_per_sec:.1f} steps/sec")
#             if gpu_available:
#                 monitor_gpu_memory()
#
#     sim_time = time.perf_counter() - sim_start
#     total_steps_per_sec = simulation_steps / sim_time
#     print(f"\nSimulation completed: {sim_time:.2f}s ({total_steps_per_sec:.1f} steps/sec)")
#
#     if gpu_available:
#         print("\nFinal GPU status:")
#         monitor_gpu_memory()
#
#     # Process results
#     parameters, losses = simulator.get_losses()
#     for para, loss in zip(parameters, losses):
#         for ind in individuals:
#             if ind.is_finished:
#                 continue
#             elif np.allclose(ind, para):
#                 ind.set_calculation_state(CalculationState.FINISHED)
#                 ind.set_fitness(float(loss))
#
#     cmaes.update()
#
#     # Performance summary
#     total_time = init_time + warmup_time + sim_time
#     print(f"\n{'=' * 60}")
#     print(f"PERFORMANCE SUMMARY")
#     print(f"{'=' * 60}")
#     print(f"Configuration:")
#     print(f"  Batch size: {batch_size} (5x increase from original 100)")
#     print(f"  Simulation steps: {simulation_steps} (3x increase from ~90)")
#     print(f"  Multi-step batching: {batch_steps} steps per batch")
#     print(f"  GPU optimizations: {'Enabled' if gpu_available else 'Disabled'}")
#     print(f"\nTiming:")
#     print(f"  Initialization: {init_time:.2f}s")
#     print(f"  JIT warmup: {warmup_time:.2f}s")
#     print(f"  Simulation: {sim_time:.2f}s")
#     print(f"  Total: {total_time:.2f}s")
#     print(f"\nThroughput:")
#     print(f"  Steps per second: {total_steps_per_sec:.1f}")
#     print(f"  Total individuals×steps: {batch_size * simulation_steps:,}")
#     print(f"  Individual-steps/sec: {(batch_size * simulation_steps) / sim_time:.0f}")
#     print(f"  Efficiency: {((batch_size * simulation_steps) / sim_time) / 1000:.1f}k individual-steps/sec")
#     print(f"\nOptimizations Applied:")
#     print(f"  ✓ Increased batch size for better GPU parallelization")
#     print(f"  ✓ Extended simulation for sustained GPU utilization")
#     print(f"  ✓ Multi-step batching with JAX scan for reduced Python overhead")
#     print(f"  ✓ JIT warmup to eliminate compilation overhead")
#     print(f"  ✓ XLA compiler optimizations")
#     print(f"  ✓ GPU memory monitoring")
#     print(f"{'=' * 60}")
