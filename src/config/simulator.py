import mujoco
from mujoco import mjx
from typing import Self

import jax
from flax import nnx
from flax.struct import dataclass as jax_dataclass

from framework.prelude import *
from framework.backends import SimulatorWithCtrl

from .controller import Controller


@jax_dataclass
class Simulator(SimEvaluateTrait):
    _parent_sim: SimulatorWithCtrl

    @property
    def data(self) -> mjx.Data:
        return self._parent_sim.data

    @property
    def controller(self) -> Controller:
        return self._parent_sim.controller

    def _update_parent(self, **kwargs) -> Self:
        return self.replace(_parent_sim=self._parent_sim.update(**kwargs))

    def update(
            self,
            data=None,
            controller=None,
            rngs_for_relocating_food: jax.Array = None,

            **kwargs
    ) -> Self:
        kwargs["data"] = data
        kwargs["controller"] = controller
        kwargs["rngs_for_relocating_food"] = rngs_for_relocating_food
        return self._update_parent(**kwargs)

    @classmethod
    def new(cls, settings: Settings, controller: Controller, rngs: jax.Array) -> tuple[mujoco.MjModel, Self]:
        mj_model, sim = SimulatorWithCtrl.new(settings, controller, rngs)
        return mj_model, cls(_parent_sim=sim)

    @staticmethod
    @nnx.jit
    def _step(this: 'Simulator') -> 'Simulator':
        this = this.update(_parent_sim=this._parent_sim.step())
        return this

    def step(self) -> Self:
        return Simulator._step(self)

    @staticmethod
    @nnx.jit
    def _step_n(this: 'Simulator', n: int) -> 'Simulator':
        def body_fn(_i, sim: 'Simulator') -> 'Simulator':
            return Simulator._step(sim)

        this = jax.lax.fori_loop(0, n, body_fn, this)
        return this

    def step_n(self, n: int) -> Self:
        return Simulator._step_n(self, n)

    def reset(self) -> Self:
        return self.update(_parent_sim=self._parent_sim.reset())

    def evaluate(self) -> dict:
        result = self._parent_sim.evaluate()
        result["l2"] = self.controller.l2
        return result
