import jax
import jax.numpy as jnp
from flax import nnx

from framework.prelude import *
from framework.types.jaxable import JaxableController

modules = [
    (nnx.Linear, {"in_features": 6, "out_features": 3}, {"kernel_init": 6 * 3, "bias_init": 6}),
    (nnx.Linear, {"in_features": 3, "out_features": 2}, {"kernel_init": 3 * 2, "bias_init": 2}),
]


def gen_module(layer_index: int, parameters: Individual, offset: int, rngs: nnx.Rngs):
    module = modules[layer_index][0]
    keywags = modules[layer_index][1]

    divided_parameters = {}
    for name, num_parms in modules[layer_index][2].items():
        start_idx = offset
        end_idx = start_idx + num_parms
        divided_parameters[name] = jnp.array(parameters[start_idx:end_idx])
        offset += num_parms

    inits = {
        name: lambda key, shape, dtype, p=param: p.reshape(shape).astype(dtype)
        for name, param in divided_parameters.items()
    }

    return module(**keywags, **inits, rngs=rngs), offset


class Controller(JaxableController):
    def __init__(self, parameters: Individual):
        dummy_rngs = nnx.Rngs(params=0)
        self.dense1, offset = gen_module(0, parameters, 0, dummy_rngs)
        self.dense2, offset = gen_module(1, parameters, offset, dummy_rngs)
        self._dim = offset

    def forward(self, x: jax.Array) -> jax.Array:
        x = self.dense1(x)
        x = nnx.silu(x)
        x = self.dense2(x)
        x = nnx.sigmoid(x)
        return x

    @property
    def dim(self):
        return self._dim
