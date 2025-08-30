from functools import partial

import jax
from flax import struct


@struct.dataclass
class ParaStock:
    index: int
    parameter: jax.Array

    def consume(self, n: int) -> jax.Array:
        para = self.parameter[self.index:self.index + n]
        self.index += n
        return para

    @staticmethod
    def _raw_initializer(_key, shape, dtype, default_parameter: jax.Array) -> jax.Array:
        return default_parameter.reshape(shape).astype(dtype)

    def gen_initializer(self, n: int):
        return partial(self._raw_initializer, default_parameter=self.consume(n))
