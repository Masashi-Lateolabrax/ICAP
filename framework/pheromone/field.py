from functools import partial

import jax
import jax.numpy as jnp

from ..prelude import *
from .cell import PheromoneFieldCell


@jax.jit
def _set_boundary(values, fill):
    values = values.at[0, :].set(fill)
    values = values.at[-1, :].set(fill)
    values = values.at[:, 0].set(fill)
    values = values.at[:, -1].set(fill)
    return values


def _dDistribution_dt(
        gas_values: jnp.ndarray,
        mask: jnp.ndarray,
        diffusion_coefficient: float,
        dx: float,
        padding_value: float,
) -> jnp.ndarray:
    gas_values = _set_boundary(gas_values, padding_value)

    center = gas_values[1:-1, 1:-1]
    d_left = (gas_values[1:-1, 0:-2] - center) * mask[1:-1, 0:-2]
    d_right = (gas_values[1:-1, 2:] - center) * mask[1:-1, 2:]
    d_top = (gas_values[0:-2, 1:-1] - center) * mask[0:-2, 1:-1]
    d_bottom = (gas_values[2:, 1:-1] - center) * mask[2:, 1:-1]
    return diffusion_coefficient * (d_top + d_bottom + d_left + d_right) / (dx * dx)


def _dEvaporation_dt(
        gas_values: jnp.ndarray,
        liquid_values: jnp.ndarray,
        saturation_pressure: float,
        evaporation_rate: float,
) -> jnp.ndarray:
    evaporation = (saturation_pressure - gas_values[1:-1, 1:-1]) * evaporation_rate
    evaporation = jnp.minimum(evaporation, liquid_values)
    return evaporation


def _dDecrease_dt(
        gas_values: jnp.ndarray,
        decrease_rate: float,
) -> jnp.ndarray:
    return gas_values * decrease_rate


def _d_dt(
        liquid_values: jnp.ndarray,
        gas_values: jnp.ndarray,
        mask: jnp.ndarray,
        dx: float,
        saturation_pressure: float,
        diffusion_coefficient: float,
        evaporation_rate: float,
        decrease_rate: float,
        padding_value: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    d_evaporation = _dEvaporation_dt(
        gas_values=gas_values,
        liquid_values=liquid_values,
        saturation_pressure=saturation_pressure,
        evaporation_rate=evaporation_rate
    )
    d_distribution = _dDistribution_dt(
        gas_values=gas_values,
        mask=mask,
        diffusion_coefficient=diffusion_coefficient,
        dx=dx,
        padding_value=padding_value
    )
    d_decrease = _dDecrease_dt(  # Positive values is decrease. Negative values is increase.
        gas_values=gas_values,
        decrease_rate=decrease_rate
    )

    d_gas = (-d_decrease).at[1:-1, 1:-1].add(d_distribution)
    d_gas = d_gas.at[1:-1, 1:-1].add(d_evaporation)
    d_liquid = -d_evaporation

    return d_gas, d_liquid


def _update_with_rk4(
        liquid_values: jnp.ndarray,
        gas_values: jnp.ndarray,
        mask: jnp.ndarray,
        dx: float,
        saturation_pressure: float,
        diffusion_coefficient: float,
        evaporation_rate: float,
        decrease_rate: float,
        dt: float,
        padding_value: float,
        iter_: int = 1
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dt = dt / iter_
    for _ in range(iter_):
        k1_gas, k1_liquid = _d_dt(
            liquid_values=liquid_values,
            gas_values=gas_values,
            mask=mask,
            dx=dx,
            saturation_pressure=saturation_pressure,
            diffusion_coefficient=diffusion_coefficient,
            evaporation_rate=evaporation_rate,
            decrease_rate=decrease_rate,
            padding_value=padding_value,
        )

        k2_gas, k2_liquid = _d_dt(
            liquid_values=liquid_values + 0.5 * dt * k1_liquid,
            gas_values=gas_values + 0.5 * dt * k1_gas,
            mask=mask,
            dx=dx,
            saturation_pressure=saturation_pressure,
            diffusion_coefficient=diffusion_coefficient,
            evaporation_rate=evaporation_rate,
            decrease_rate=decrease_rate,
            padding_value=padding_value
        )

        k3_gas, k3_liquid = _d_dt(
            liquid_values=liquid_values + 0.5 * dt * k2_liquid,
            gas_values=gas_values + 0.5 * dt * k2_gas,
            mask=mask,
            dx=dx,
            saturation_pressure=saturation_pressure,
            diffusion_coefficient=diffusion_coefficient,
            evaporation_rate=evaporation_rate,
            decrease_rate=decrease_rate,
            padding_value=padding_value
        )

        k4_gas, k4_liquid = _d_dt(
            liquid_values=liquid_values + dt * k3_liquid,
            gas_values=gas_values + dt * k3_gas,
            mask=mask,
            dx=dx,
            saturation_pressure=saturation_pressure,
            diffusion_coefficient=diffusion_coefficient,
            evaporation_rate=evaporation_rate,
            decrease_rate=decrease_rate,
            padding_value=padding_value
        )

        gas_values = jnp.maximum(0.0, gas_values + (k1_gas + 2 * k2_gas + 2 * k3_gas + k4_gas) / 6)
        liquid_values = jnp.maximum(0.0, liquid_values + (k1_liquid + 2 * k2_liquid + 2 * k3_liquid + k4_liquid) / 6)

    return gas_values, liquid_values


class PheromoneField:
    def tree_flatten(self):
        aux_data = {
            'update_func': self._update,
            'reset_func': self._reset,
            'get_func': self._get,
            'add_func': self._add
        }
        return (self.values_liquid, self._values_gas, self.mask), aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        values_liquid, values_gas, mask = leaves

        # Create uninitialized instance
        obj = cls.__new__(cls)

        # Set state directly
        obj.values_liquid = values_liquid
        obj._values_gas = values_gas
        obj.mask = mask

        # Restore JIT functions
        obj._update = aux_data['update_func']
        obj._reset = aux_data['reset_func']
        obj._get = aux_data['get_func']
        obj._add = aux_data['add_func']

        return obj

    def __init__(
            self,
            nx: int,
            ny: int,
            dx: float,
            temperature: float,
            material: Material,
            evaporation_rate: float,
            decrease_rate: float,
            padding_value: float = 0.0,
            iter_: int = 1
    ):
        if nx <= 0 or ny <= 0:
            raise ValueError("Grid dimensions must be positive")
        if dx <= 0:
            raise ValueError("Grid spacing dx must be positive")
        if evaporation_rate < 0 or decrease_rate < 0:
            raise ValueError("Rates must be non-negative")
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if iter_ <= 0:
            raise ValueError("Iteration count must be positive")

        shape = jnp.array((ny, nx), dtype=jnp.int32)
        saturation_pressure = material.saturation_pressure(temperature)
        diffusion_coefficient = material.diffusion_coefficient(temperature)

        self._update = jax.jit(partial(
            _update_with_rk4,
            dx=dx,
            saturation_pressure=saturation_pressure,
            diffusion_coefficient=diffusion_coefficient,
            evaporation_rate=evaporation_rate,
            decrease_rate=decrease_rate,
            padding_value=padding_value,
            iter_=iter_,
        ))

        self.values_liquid = jnp.zeros(shape, dtype=jnp.float32)
        self._values_gas = jnp.zeros(shape + 2, dtype=jnp.float32)
        self.mask = jnp.ones(shape + 2, dtype=jnp.bool_)

    def update(self, dt: float):
        self._values_gas, self.values_liquid = self._update(
            liquid_values=self.values_liquid,
            gas_values=self._values_gas,
            mask=self.mask,
            dt=dt,
        )

    @property
    def shape(self):
        return self.values_liquid.shape

    @property
    def values_gas(self) -> jnp.ndarray:
        return self._values_gas[1:-1, 1:-1]

    @staticmethod
    @jax.jit
    def _reset(values_liquid, values_gas):
        return jnp.zeros_like(values_liquid), jnp.zeros_like(values_gas)

    def reset(self):
        self.values_liquid, self._values_gas = self._reset(
            self.values_liquid, self._values_gas
        )

    def set_neumann_boundary(self):
        self.mask = _set_boundary(self.mask, 0)

    def set_dirichlet_boundary(self):
        self.mask = _set_boundary(self.mask, 1)

    @staticmethod
    @jax.jit
    def _get(values, xs, ys) -> jnp.ndarray:
        shape = values.shape
        xs = jnp.clip(xs.astype(jnp.int32), 0, shape[1])
        ys = jnp.clip(ys.astype(jnp.int32), 0, shape[0])
        return values[ys, xs]

    def get_gas(self, xs, ys) -> jnp.ndarray:
        return self._get(self._values_gas[1:-1, 1:-1], xs, ys)

    def get_liquid(self, xs, ys) -> jnp.ndarray:
        return self._get(self.values_liquid, xs, ys)

    @staticmethod
    @jax.jit
    def _add(values: jax.Array, xs: jax.Array, ys: jax.Array, additions: jax.Array):
        shape = values.shape
        xs = jnp.clip(xs.astype(jnp.int32), 0, shape[1])
        ys = jnp.clip(ys.astype(jnp.int32), 0, shape[0])
        return values.at[ys, xs].add(additions)

    def add_liquid(self, xs, ys, additions):
        self.values_liquid = self._add(self.values_liquid, xs, ys, additions)

    def add_liquid_by_cell(self, cell: list[PheromoneFieldCell]):
        indexes = []
        values = []
        for c in cell:
            if c.add_value > 0:
                indexes.append((c.index_x, c.index_y))
                values.append(c.add_value)

        indexes = jnp.array(indexes, dtype=jnp.int32)
        xs = indexes[:, 0]
        ys = indexes[:, 1]
        vs = jnp.array(values, dtype=jnp.float32)

        if xs.size == 0 or ys.size == 0 or vs.size == 0:
            return

        self.add_liquid(xs, ys, vs)

        for c in cell:
            c.add_value = 0.0


jax.tree_util.register_pytree_node(
    PheromoneField,
    PheromoneField.tree_flatten,
    PheromoneField.tree_unflatten
)
