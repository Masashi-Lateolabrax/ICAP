import numpy as np
import mujoco
from mujoco import mjx

import jax
import jax.numpy as jnp
from flax import nnx
from flax.struct import dataclass as jax_dataclass

from framework.prelude import *
from framework.backends import BasicSimulatorWithEnv

from .controller import Controller


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
