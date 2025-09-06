import jax
import jax.numpy as jnp
from flax import nnx
from typing import Self

from framework.utils import ParaStock
from framework.backends import ControllerInterface
from framework.prelude import RobotInputs, RobotOutputs


class Controller(ControllerInterface):
    @staticmethod
    def dim():
        return (16 * 8 + 8) + (8 * 3 + 3)

    def __init__(self, parameter: jax.Array):
        self.l2 = jnp.linalg.norm(parameter)

        rngs = nnx.Rngs(0)
        parameter = ParaStock(parameter)

        self.layer1 = nnx.Linear(
            in_features=16,
            out_features=8,
            kernel_init=parameter.gen_initializer(16 * 8),
            bias_init=parameter.gen_initializer(8),
            rngs=rngs,
        )
        self.layer2 = nnx.Linear(
            in_features=8,
            out_features=3,
            kernel_init=parameter.gen_initializer(8 * 3),
            bias_init=parameter.gen_initializer(3),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.swish(self.layer1(x))
        x = self.layer2(x)
        act = jnp.clip(x[:, 0:2], -0.3, 1.0)
        phe = nnx.sigmoid(x[:, 2])
        x = x.at[:, 0:2].set(act)
        x = x.at[:, 2].set(phe)
        return x

    def forward(self, x: RobotInputs) -> RobotOutputs:
        x = self.__call__(x.ray)
        return RobotOutputs(
            left_wheel=x[:, 0],
            right_wheel=x[:, 1],
            pheromone=x[:, 2],
        )

    def reset(self) -> Self:
        return self
