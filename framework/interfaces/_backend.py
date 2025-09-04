import abc
from typing import Self

import numpy as np

import jax
from flax.struct import dataclass as jax_dataclass

from ..pheromone import PheromoneField


@jax_dataclass
class SimulatorTrait(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _update_parent(self, **kwargs: dict) -> Self:
        return self

    def update(self, **kwargs) -> Self:
        this = self
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        this_param_names = vars(this).keys()
        this_kwargs = {k: v for k, v in kwargs.items() if k in this_param_names}
        this = this.replace(**this_kwargs)

        parent_kwargs = {k: v for k, v in kwargs.items() if k not in this_param_names}
        return this._update_parent(parent_kwargs)

    @abc.abstractmethod
    def get_pheromone(self, positions: jax.Array) -> jax.Array:
        raise NotImplementedError

    @abc.abstractmethod
    def add_pheromone(self, positions: jax.Array, values: jax.Array) -> PheromoneField:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self) -> Self:
        raise NotImplementedError

    @abc.abstractmethod
    def step_n(self, n: int) -> Self:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError


class SimRenderTrait(SimulatorTrait, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        raise NotImplementedError


class SimEvaluateTrait(SimulatorTrait, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(self) -> dict:
        raise NotImplementedError
