from typing import Self

import jax
import jax.numpy as jnp
from flax import nnx

from framework.prelude import *
from framework.utils import ParaStock


class PracticalController(ControllerInterface):
    @staticmethod
    def dim():
        return 16 * 3 + 3

    def __init__(self, parameter: jax.Array):
        self.l2 = jnp.linalg.norm(parameter)

        rngs = nnx.Rngs(0)
        parameter = ParaStock(parameter)

        self.layer1 = nnx.Linear(
            in_features=16,
            out_features=3,
            kernel_init=parameter.gen_initializer(16 * 3),
            bias_init=parameter.gen_initializer(3),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.tanh(self.layer1(x))
        return x

    def forward(self, x: RobotInputs) -> RobotOutputs:
        x = self.__call__(x.ray)
        x1 = jnp.clip(x[:, :2], -0.3, 1.0)
        x2 = nnx.sigmoid(x[:, 2])
        return RobotOutputs(
            left_wheel=x1[:, 0],
            right_wheel=x1[:, 1],
            pheromone=x2,
        )

    def reset(self) -> Self:
        return self
