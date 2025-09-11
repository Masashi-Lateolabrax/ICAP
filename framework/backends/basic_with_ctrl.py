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

    @partial(nnx.jit, inline=True, donate_argnames=("self",))
    def step(self, model: mjx.Model) -> Self:
        this = self.update(
            _parent_sim=self._parent_sim.step(model)
        )
        return this.update(
            robot_outputs=self.controller.forward(this.robot_inputs)
        )

    @partial(nnx.jit, static_argnames=("n", "unroll"), inline=True, donate_argnames=("self",))
    def step_n(self, model: mjx.Model, n: int, unroll: int = 1) -> Self:
        def body_fn(carry: "SimulatorWithCtrl", _x) -> tuple["SimulatorWithCtrl", None]:
            new_carry = carry.step(model)
            return new_carry, None

        return jax.lax.scan(body_fn, self, length=n, unroll=unroll)[0]

    @partial(nnx.jit, inline=True, donate_argnames=("self",))
    def reset(self, model: mjx.Model) -> Self:
        return self.update(
            _parent_sim=self._parent_sim.reset(model),
            controller=self.controller.reset()
        )

    def render(self, img_buf: np.ndarray, camera: mujoco.MjvCamera, renderer: mujoco.Renderer):
        self._parent_sim.render(img_buf, camera, renderer)

    def evaluate(self) -> dict:
        return self._parent_sim.evaluate()
