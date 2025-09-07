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


class RandomPatternController(ControllerInterface):
    def __init__(self, parameter: int):  # parameter is the number of robots
        self.candidates = jnp.array([
            [1., -1.],
            [-1., 1.],
            [1., 1.],
            [-0.5, -0.5],
        ])
        self.rngs = nnx.Rngs(0)
        self.state = jnp.ones((parameter, 3), dtype=jnp.float32)

    def __call__(self, x: jax.Array) -> jax.Array:
        do_update = jax.random.randint(self.rngs(), (1,), minval=0, maxval=100)
        select = jax.random.randint(self.rngs(), (x.shape[0],), minval=0, maxval=4)

        x = jax.lax.cond(
            do_update[0] < 1,
            lambda _: self.candidates[select, :2],
            lambda _: self.state[:, :2],
            operand=None
        )
        self.state = self.state.at[:, :2].set(x)

        return self.state

    def forward(self, x: RobotInputs) -> RobotOutputs:
        x = x.as_matrix()
        x = self.__call__(x)
        return RobotOutputs(
            left_wheel=x[:, 0],
            right_wheel=x[:, 1],
            pheromone=x[:, 2],
        )

    def reset(self) -> Self:
        return self

    @staticmethod
    def dim():
        return 0
