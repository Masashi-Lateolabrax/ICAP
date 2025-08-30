from functools import partial
import dataclasses

import jax
from flax.typing import Initializer


@dataclasses.dataclass
class ParaStock:
    parameter: jax.Array
    index: int = 0

    def consume(self, n: int) -> jax.Array:
        para = self.parameter[self.index:self.index + n]
        self.index += n
        return para

    @staticmethod
    def _raw_initializer(_key, shape, dtype, default_parameter: jax.Array) -> jax.Array:
        return default_parameter.reshape(shape).astype(dtype)

    def gen_initializer(self, n: int) -> Initializer:
        return partial(self._raw_initializer, default_parameter=self.consume(n))
