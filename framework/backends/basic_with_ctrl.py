import abc

import numpy as np
import mujoco
from mujoco import mjx

import jax
import jax.numpy as jnp
from flax import nnx
from flax.struct import dataclass as jax_dataclass

from framework.prelude import *
from framework.backends import BasicSimulatorWithEnv


class ControllerInterface(nnx.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, parameter):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.ones((0, 3), dtype=jnp.float32)

    @staticmethod
    @abc.abstractmethod
    def dim() -> int:
        raise NotImplementedError


@jax_dataclass
class SimulatorWithCtrl:
    _env_sim: BasicSimulatorWithEnv

    controller: ControllerInterface

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
    def loss(self) -> jax.Array:
        return self._env_sim.loss

    def update(
            self,
            data: mjx.Data = None,
            robots: BatchedRobots = None,
            robot_inputs: jax.Array = None,
            loss: jax.Array = None,

            controller: ControllerInterface = None
    ) -> 'SimulatorWithCtrl':
        parent_kwargs = {
            "data": data,
            "robots": robots,
            "robot_inputs": robot_inputs,
            "loss": loss,
        }
        env_sim = self._env_sim.update(**parent_kwargs)

        this_kwargs = {
            "_env_sim": env_sim,
            "controller": controller,
        }
        kwargs = {k: v for k, v in this_kwargs.items() if v is not None}
        if not kwargs:
            return self

        return self.replace(**kwargs)

    @classmethod
    def new(cls, settings: Settings, controller: ControllerInterface, rngs: jax.Array) -> tuple[
        mujoco.MjModel, 'SimulatorWithCtrl']:
        mj_model, sim = BasicSimulatorWithEnv.new(settings, rngs)
        return mj_model, cls(
            _env_sim=sim,
            controller=controller,
        )

    def add_pheromone(self, positions: jax.Array, amounts: jax.Array) -> 'SimulatorWithCtrl':
        new_env_sim = self._env_sim.add_pheromone(positions, amounts)
        return self.replace(_env_sim=new_env_sim)

    @staticmethod
    @nnx.jit
    def _step(this: 'SimulatorWithCtrl') -> 'SimulatorWithCtrl':
        this: "SimulatorWithCtrl" = this.replace(_env_sim=this._env_sim.step())

        output = this.controller(this.robot_inputs)
        new_data = this.robots.set_ctrl(this.data, output[:, :2])

        this = this.add_pheromone(this.robots.positions, output[:, 2])

        return this.update(data=new_data)

    def step(self) -> 'SimulatorWithCtrl':
        return SimulatorWithCtrl._step(self)

    @staticmethod
    @nnx.jit
    def _step_n(this: "SimulatorWithCtrl", n: int) -> "SimulatorWithCtrl":
        def body_fn(_i, sim: "SimulatorWithCtrl"):
            return SimulatorWithCtrl._step(sim)

        new_simulator = jax.lax.fori_loop(0, n, body_fn, this)
        return new_simulator

    def step_n(self, n: int) -> 'SimulatorWithCtrl':
        return SimulatorWithCtrl._step_n(self, n)

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

    def reset(self, controller: ControllerInterface = None, rngs: jax.Array = None) -> 'SimulatorWithCtrl':
        new_env_sim = self._env_sim.reset(rngs)
        this = self.replace(_env_sim=new_env_sim)
        return this.update(
            controller=controller
        )
