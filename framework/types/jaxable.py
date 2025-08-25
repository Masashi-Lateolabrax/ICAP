import abc

import jax
import flax.nnx as nnx


class JaxableController(nnx.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError