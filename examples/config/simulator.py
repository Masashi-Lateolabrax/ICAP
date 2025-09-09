from typing import Self
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from flax.struct import dataclass as jax_dataclass

import numpy as np
import mujoco
from mujoco import mjx

from framework.prelude import *
from framework.backends import SimulatorWithCtrl


@jax_dataclass
class PracticalSimulator(SimEvaluateTrait):
    _parent_sim: SimulatorWithCtrl

    @property
    def data(self) -> mjx.Data:
        return self._parent_sim.data

    @property
    def controller(self) -> ControllerT:
        return self._parent_sim.controller

    def _update_parent(self, **kwargs: dict) -> Self:
        return self.replace(_parent_sim=self._parent_sim.update(**kwargs))

    def update(
            self,
            data: mjx.Data = None,
            controller: ControllerT = None,
            **kwargs
    ) -> Self:
        kwargs["data"] = data
        kwargs["controller"] = controller
        return self._update(**kwargs)

    @classmethod
    def new(cls, settings: Settings, controller: ControllerT, rngs: jax.Array) -> tuple[
        mujoco.MjModel, 'PracticalSimulator']:
        mj_model, sim = SimulatorWithCtrl.new(settings, controller, rngs)
        return mj_model, cls(_parent_sim=sim)

    @staticmethod
    @partial(nnx.jit, inline=True)
    def _step(this: 'PracticalSimulator', model: mjx.Model) -> 'PracticalSimulator':
        this: "PracticalSimulator" = this.update(_parent_sim=this._parent_sim.step(model))
        return this

    def step(self, model: mjx.Model) -> Self:
        return PracticalSimulator._step(self, model)

    @staticmethod
    @partial(nnx.jit, static_argnames=("n",), inline=True)
    def _step_n(this: "PracticalSimulator", model: mjx.Model, n: int) -> "PracticalSimulator":
        def body_fn(_i, sim: "PracticalSimulator"):
            return PracticalSimulator._step(sim, model)

        this = jax.lax.fori_loop(0, n, body_fn, this)
        return this

    def step_n(self, model: mjx.Model, n: int) -> Self:
        return PracticalSimulator._step_n(self, model, n)

    @partial(nnx.jit, inline=True)
    def reset(self) -> Self:
        return self.update(
            _parent_sim=self._parent_sim.reset()
        )

    def evaluate(self) -> dict:
        result = self._parent_sim.evaluate()
        return result


@jax_dataclass
class FoodRelocationSimulator(SimRenderTrait):
    _parent_sim: SimulatorWithCtrl

    @property
    def data(self) -> mjx.Data:
        return self._parent_sim.data

    @property
    def food_items(self) -> BatchedFood:
        return self._parent_sim.food_items

    def _update_parent(self, **kwargs: dict) -> Self:
        return self.replace(
            _parent_sim=self._parent_sim.update(**kwargs)
        )

    def update(
            self,
            data: mjx.Data = None,
            food_items: BatchedFood = None,
            controller: ControllerT = None,
            **kwargs
    ) -> Self:
        kwargs["data"] = data
        kwargs["food_items"] = food_items
        kwargs["controller"] = controller
        return self._update(**kwargs)

    @classmethod
    def new(
            cls, settings: Settings, controller: ControllerT, rngs: jax.Array
    ) -> tuple[mujoco.MjModel, 'FoodRelocationSimulator']:
        mj_model, parent_sim = SimulatorWithCtrl.new(settings, controller, rngs)
        return mj_model, cls(_parent_sim=parent_sim)

    @staticmethod
    @partial(nnx.jit, inline=True)
    def _step(this: 'FoodRelocationSimulator', model: mjx.Model) -> 'FoodRelocationSimulator':
        this = this.update(
            _parent_sim=this._parent_sim.step(model)
        )

        nest_dir = -this.food_items.positions
        nest_dir = nest_dir.at[:, 2].set(0.0)
        force = nest_dir / (jnp.linalg.norm(nest_dir, axis=1, keepdims=True) + 1e-6) * 50.0
        new_data = this.food_items.set_force(this.data, jnp.arange(force.shape[0]), force)

        return this.update(data=new_data)

    def step(self, model: mjx.Model) -> Self:
        return FoodRelocationSimulator._step(self, model)

    @staticmethod
    @partial(nnx.jit, static_argnames=("n",), inline=True)
    def _step_n(this: "FoodRelocationSimulator", model: mjx.Model, n: int) -> "FoodRelocationSimulator":
        def body_fn(_i, sim: "FoodRelocationSimulator"):
            return FoodRelocationSimulator._step(sim, model)

        this = jax.lax.fori_loop(0, n, body_fn, this)
        return this

    def step_n(self, model: mjx.Model, n: int) -> Self:
        return FoodRelocationSimulator._step_n(self, model, n)

    def reset(self) -> Self:
        parent_sim = self._parent_sim.reset()
        return self.update(_parent_sim=parent_sim)

    def render(self, img_buf: np.ndarray, camera: mujoco.MjvCamera, renderer: mujoco.Renderer):
        self._parent_sim.render(img_buf, camera, renderer)
