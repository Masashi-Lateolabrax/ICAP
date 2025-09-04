import abc
from typing import Self

import numpy as np
import mujoco
from mujoco import mjx

import jax
from flax import nnx
from flax.struct import dataclass as jax_dataclass

from ..prelude import *
from ..pheromone import PheromoneField
from .basic_with_env import RobotOutputs, BasicSimulatorWithEnv


class ControllerInterface(nnx.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, parameter):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, x: jax.Array) -> RobotOutputs:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> Self:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def dim() -> int:
        raise NotImplementedError


@jax_dataclass
class SimulatorWithCtrl(SimRenderTrait, SimEvaluateTrait):
    _parent_sim: BasicSimulatorWithEnv
    controller: ControllerInterface

    @property
    def data(self) -> mjx.Data:
        return self._parent_sim.data

    @property
    def robots(self) -> BatchedRobots:
        return self._parent_sim.robots

    @property
    def food_items(self) -> BatchedFood:
        return self._parent_sim.food_items

    @property
    def robot_inputs(self) -> jax.Array:
        return self._parent_sim.robot_inputs

    @property
    def robot_outputs(self) -> RobotOutputs:
        return self._parent_sim.robot_outputs

    @property
    def loss_offset(self) -> jax.Array:
        return self._parent_sim.loss_offset

    def _update_parent(self, **kwargs: dict) -> Self:
        return self.replace(
            _parent_sim=self._parent_sim.update(**kwargs)
        )

    def update(
            self,
            model: mjx.Model = None,
            data: mjx.Data = None,
            pheromone: PheromoneField = None,
            robots: BatchedRobots = None,
            food_items: BatchedFood = None,
            robot_inputs: jax.Array = None,
            robot_outputs: RobotOutputs = None,
            loss: jax.Array = None,

            controller: ControllerInterface = None,

            **kwargs
    ) -> Self:
        kwargs["model"] = model
        kwargs["data"] = data
        kwargs["pheromone"] = pheromone
        kwargs["robots"] = robots
        kwargs["food_items"] = food_items
        kwargs["robot_inputs"] = robot_inputs
        kwargs["robot_outputs"] = robot_outputs
        kwargs["loss"] = loss
        kwargs["controller"] = controller
        return self._update(**kwargs)

    @classmethod
    def new(cls, settings: Settings, controller: ControllerInterface, rngs: jax.Array) -> tuple[mujoco.MjModel, Self]:
        mj_model, sim = BasicSimulatorWithEnv.new(settings, rngs)
        return mj_model, cls(
            _parent_sim=sim,
            controller=controller,
        )

    def get_pheromone(self, positions: jax.Array) -> jax.Array:
        return self._parent_sim.get_pheromone(positions)

    def add_pheromone(self, positions: jax.Array, amounts: jax.Array) -> PheromoneField:
        return self._parent_sim.add_pheromone(positions, amounts)

    @staticmethod
    @nnx.jit
    def _step(this: 'SimulatorWithCtrl') -> 'SimulatorWithCtrl':
        this = this.update(
            _parent_sim=this._parent_sim.step()
        )
        this = this.update(
            robot_outputs=this.controller(this.robot_inputs)
        )
        return this

    def step(self) -> Self:
        return SimulatorWithCtrl._step(self)

    @staticmethod
    @nnx.jit
    def _step_n(this: "SimulatorWithCtrl", n: int) -> "SimulatorWithCtrl":
        def body_fn(_i, sim: "SimulatorWithCtrl"):
            return SimulatorWithCtrl._step(sim)

        new_simulator = jax.lax.fori_loop(0, n, body_fn, this)
        return new_simulator

    def step_n(self, n: int) -> Self:
        return SimulatorWithCtrl._step_n(self, n)

    def reset(self) -> 'SimulatorWithCtrl':
        parent_sim = self._parent_sim.reset()
        controller = self.controller.reset()
        return self.update(
            _parent_sim=parent_sim,
            controller=controller
        )

    def render(self, img_buf: np.ndarray, camera: mujoco.MjvCamera, renderer: mujoco.Renderer):
        self._parent_sim.render(img_buf, camera, renderer)

    def evaluate(self) -> dict:
        return self._parent_sim.evaluate()

# @jax_dataclass
# class SimulatorWithCtrl:
#     _env_sim: BasicSimulatorWithEnv
#
#     controller: ControllerInterface
#
#     @property
#     def data(self) -> mjx.Data:
#         return self._env_sim.data
#
#     @property
#     def robots(self) -> BatchedRobots:
#         return self._env_sim.robots
#
#     @property
#     def robot_inputs(self) -> jax.Array:
#         return self._env_sim.robot_inputs
#
#     @property
#     def food_items(self) -> BatchedFood:
#         return self._env_sim.food_items
#
#     @property
#     def loss(self) -> jax.Array:
#         return self._env_sim.loss
#
#     def update(
#             self,
#             data: mjx.Data = None,
#             robots: BatchedRobots = None,
#             robot_inputs: jax.Array = None,
#             food_items: BatchedFood = None,
#             loss: jax.Array = None,
#
#             controller: ControllerInterface = None
#     ) -> 'SimulatorWithCtrl':
#         parent_kwargs = {
#             "data": data,
#             "robots": robots,
#             "robot_inputs": robot_inputs,
#             "food_items": food_items,
#             "loss": loss,
#         }
#         env_sim = self._env_sim.update(**parent_kwargs)
#
#         this_kwargs = {
#             "_env_sim": env_sim,
#             "controller": controller,
#         }
#         kwargs = {k: v for k, v in this_kwargs.items() if v is not None}
#         if not kwargs:
#             return self
#
#         return self.replace(**kwargs)
#
#     @classmethod
#     def new(cls, settings: Settings, controller: ControllerInterface, rngs: jax.Array) -> tuple[
#         mujoco.MjModel, 'SimulatorWithCtrl']:
#         mj_model, sim = BasicSimulatorWithEnv.new(settings, rngs)
#         return mj_model, cls(
#             _env_sim=sim,
#             controller=controller,
#         )
#
#     def add_pheromone(self, positions: jax.Array, amounts: jax.Array) -> 'SimulatorWithCtrl':
#         new_env_sim = self._env_sim.add_pheromone(positions, amounts)
#         return self.replace(_env_sim=new_env_sim)
#
#     @staticmethod
#     @nnx.jit
#     def _step(this: 'SimulatorWithCtrl') -> 'SimulatorWithCtrl':
#         this: "SimulatorWithCtrl" = this.replace(_env_sim=this._env_sim.step())
#
#         output = this.controller(this.robot_inputs)
#         new_data = this.robots.set_ctrl(this.data, output[:, :2])
#
#         this = this.add_pheromone(this.robots.positions, output[:, 2])
#
#         return this.update(data=new_data)
#
#     def step(self) -> 'SimulatorWithCtrl':
#         return SimulatorWithCtrl._step(self)
#
#     @staticmethod
#     @nnx.jit
#     def _step_n(this: "SimulatorWithCtrl", n: int) -> "SimulatorWithCtrl":
#         def body_fn(_i, sim: "SimulatorWithCtrl"):
#             return SimulatorWithCtrl._step(sim)
#
#         new_simulator = jax.lax.fori_loop(0, n, body_fn, this)
#         return new_simulator
#
#     def step_n(self, n: int) -> 'SimulatorWithCtrl':
#         return SimulatorWithCtrl._step_n(self, n)
#
#     def render(
#             self,
#             mj_model: mujoco.MjModel,
#             img_buf: np.ndarray,
#             pos: tuple[float, float, float],
#             lookat: tuple[float, float, float],
#             max_geom=100,
#             max_pheromone=1.0
#     ):
#         self._env_sim.render(mj_model, img_buf, pos, lookat, max_geom, max_pheromone)
#
#     def reset(self, controller: ControllerInterface = None, rngs: jax.Array = None) -> 'SimulatorWithCtrl':
#         new_env_sim = self._env_sim.reset(rngs)
#         this = self.replace(_env_sim=new_env_sim)
#         return this.update(
#             controller=controller
#         )
