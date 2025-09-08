from functools import partial
from typing import Self, Generic

import numpy as np
import mujoco
from mujoco import mjx

import jax
from flax import nnx
from flax.struct import dataclass as jax_dataclass

from ..prelude import *
from ..pheromone import PheromoneField
from .basic_with_env import RobotOutputs, BasicSimulatorWithEnv


@jax_dataclass
class SimulatorWithCtrl(SimRenderTrait, SimEvaluateTrait, Generic[ControllerT]):
    _parent_sim: BasicSimulatorWithEnv
    controller: ControllerT

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
    def robot_inputs(self) -> RobotInputs:
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
            rngs_for_relocating_food: jax.Array = None,

            controller: ControllerT = None,

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
        kwargs["rngs_for_relocating_food"] = rngs_for_relocating_food
        kwargs["controller"] = controller
        return self._update(**kwargs)

    @classmethod
    def new(cls, settings: Settings, controller: ControllerT, rngs: jax.Array) -> tuple[mujoco.MjModel, Self]:
        mj_model, sim = BasicSimulatorWithEnv.new(settings, rngs)
        return mj_model, cls(
            _parent_sim=sim,
            controller=controller,
        )

    @staticmethod
    @partial(nnx.jit, inline=True)
    def _step(this: 'SimulatorWithCtrl') -> 'SimulatorWithCtrl':
        this = this.update(
            _parent_sim=this._parent_sim.step()
        )
        this = this.update(
            robot_outputs=this.controller.forward(this.robot_inputs)
        )
        return this

    def step(self) -> Self:
        return SimulatorWithCtrl._step(self)

    @staticmethod
    @partial(nnx.jit, static_argnames=("n",), inline=True)
    def _step_n(this: "SimulatorWithCtrl", n: int) -> "SimulatorWithCtrl":
        def body_fn(_i, sim: "SimulatorWithCtrl"):
            return SimulatorWithCtrl._step(sim)

        new_simulator = jax.lax.fori_loop(0, n, body_fn, this)
        return new_simulator

    def step_n(self, n: int) -> Self:
        return SimulatorWithCtrl._step_n(self, n)

    @partial(nnx.jit, inline=True)
    def reset(self) -> Self:
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
