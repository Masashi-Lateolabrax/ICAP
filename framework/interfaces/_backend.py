import abc
from typing import Self

import numpy as np

import mujoco
from mujoco import mjx

import jax
from flax.struct import dataclass as jax_dataclass

from ..pheromone import PheromoneField


@jax_dataclass
class SimulatorTrait(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def data(self) -> mjx.Data:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_parent(self, **kwargs: dict) -> Self:
        return self

    def _update(self, **kwargs: dict) -> Self:
        this = self
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        this_param_names = vars(this).keys()

        if not kwargs:
            return this

        parent_kwargs = {k: v for k, v in kwargs.items() if k not in this_param_names}
        this = this._update_parent(**parent_kwargs)

        this_kwargs = {k: v for k, v in kwargs.items() if k in this_param_names}
        this = this.replace(**this_kwargs)

        return this

    @abc.abstractmethod
    def update(self, **kwargs) -> Self:
        return self._update(**kwargs)

    @abc.abstractmethod
    def step(self, model: mjx.Model) -> Self:
        raise NotImplementedError

    @abc.abstractmethod
    def step_n(self, model: mjx.Model, n: int) -> Self:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> Self:
        raise NotImplementedError

    def block_until_ready(self):
        self.data.qpos.block_until_ready()


class SimPheromoneTrait(SimulatorTrait, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_pheromone(self, positions: jax.Array) -> jax.Array:
        raise NotImplementedError

    @abc.abstractmethod
    def add_pheromone(self, positions: jax.Array, values: jax.Array) -> PheromoneField:
        raise NotImplementedError


class SimRenderTrait(SimulatorTrait, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def render(self, img_buf: np.ndarray, camera: mujoco.MjvCamera, renderer: mujoco.Renderer):
        raise NotImplementedError


class SimEvaluateTrait(SimulatorTrait, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(self) -> dict:
        raise NotImplementedError
