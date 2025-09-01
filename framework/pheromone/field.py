import jax
import jax.numpy as jnp
from flax.struct import dataclass as jax_dataclass

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


@jax.jit
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
) -> tuple[jnp.ndarray, jnp.ndarray]:
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

    gas_values = gas_values + (k1_gas + 2 * k2_gas + 2 * k3_gas + k4_gas) / 6
    liquid_values = liquid_values + (k1_liquid + 2 * k2_liquid + 2 * k3_liquid + k4_liquid) / 6

    return gas_values, liquid_values


@jax.jit
def _iter_update_with_rk4(
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
        iter_: int = 1,
):
    def body_fn(_i, val):
        g, l = _update_with_rk4(
            liquid_values=val["liquid"],
            gas_values=val["gas"],
            mask=mask,
            dx=dx,
            saturation_pressure=saturation_pressure,
            diffusion_coefficient=diffusion_coefficient,
            evaporation_rate=evaporation_rate,
            decrease_rate=decrease_rate,
            dt=dt,
            padding_value=padding_value,
        )
        return {"gas": g, "liquid": l}

    result = jax.lax.fori_loop(0, iter_, body_fn, {"gas": gas_values, "liquid": liquid_values})
    return result["gas"], result["liquid"]


@jax_dataclass
class PheromoneField:
    nx: int
    ny: int
    dx: float

    saturation_pressure: float
    diffusion_coefficient: float
    evaporation_rate: float
    decrease_rate: float

    values_liquid: jnp.ndarray  # Shape: (x, y)
    _values_gas: jnp.ndarray  # Shape: (x+2, y+2)
    mask: jnp.ndarray  # Shape: (x+2, y+2)

    padding_value: float = 0.0
    iter_: int = 1

    @property
    def values_gas(self) -> jnp.ndarray:
        return self._values_gas[1:-1, 1:-1]

    @classmethod
    def new(
            cls,
            nx: int,
            ny: int,
            dx: float,
            temperature: float,
            material: Material,
            evaporation_rate: float,
            decrease_rate: float,
            padding_value: float = 0.0,
            iter_: int = 1,
    ) -> "PheromoneField":
        shape = jnp.array([ny, nx], dtype=jnp.int32)
        saturation_pressure = material.saturation_pressure(temperature)
        diffusion_coefficient = material.diffusion_coefficient(temperature)

        return cls(
            nx=nx,
            ny=ny,
            dx=dx,

            saturation_pressure=saturation_pressure,
            diffusion_coefficient=diffusion_coefficient,
            evaporation_rate=evaporation_rate,
            decrease_rate=decrease_rate,

            values_liquid=jnp.zeros(shape, dtype=jnp.float32),
            _values_gas=jnp.zeros(shape + 2, dtype=jnp.float32),
            mask=jnp.ones(shape + 2, dtype=jnp.bool_),

            padding_value=padding_value,
            iter_=iter_
        )

    def update(self, dt: float) -> "PheromoneField":
        dt = dt / self.iter_
        new_gas, new_liquid = _iter_update_with_rk4(
            liquid_values=self.values_liquid,
            gas_values=self._values_gas,
            mask=self.mask,
            dx=self.dx,
            saturation_pressure=self.saturation_pressure,
            diffusion_coefficient=self.diffusion_coefficient,
            evaporation_rate=self.evaporation_rate,
            decrease_rate=self.decrease_rate,
            dt=dt,
            padding_value=self.padding_value,
            iter_=self.iter_,
        )
        return self.replace(
            values_liquid=new_liquid,
            _values_gas=new_gas,
        )

    @staticmethod
    @jax.jit
    def _get(values, xs, ys) -> jnp.ndarray:
        shape = values.shape
        xs = jnp.clip(xs.astype(jnp.int32), 0, shape[1] - 1)
        ys = jnp.clip(ys.astype(jnp.int32), 0, shape[0] - 1)
        return values[ys, xs]

    def get_gas(self, xs, ys) -> jnp.ndarray:
        return self._get(self.values_gas, xs, ys)

    def get_liquid(self, xs, ys) -> jnp.ndarray:
        return self._get(self.values_liquid, xs, ys)

    @staticmethod
    @jax.jit
    def _add(values: jax.Array, xs: jax.Array, ys: jax.Array, additions: jax.Array):
        shape = values.shape
        xs = jnp.clip(xs.astype(jnp.int32), 0, shape[1] - 1)
        ys = jnp.clip(ys.astype(jnp.int32), 0, shape[0] - 1)
        return values.at[ys, xs].add(additions)

    def add_liquid(self, xs, ys, additions) -> "PheromoneField":
        new_liquid = self._add(self.values_liquid, xs, ys, additions)
        return self.replace(values_liquid=new_liquid)

    def add_liquid_by_cell(self, cell: list[PheromoneFieldCell]) -> "PheromoneField":
        indexes = []
        values = []
        for c in cell:
            if c.add_value > 0:
                indexes.append((c.index_x, c.index_y))
                values.append(c.add_value)

        if not indexes:
            return self

        indexes = jnp.array(indexes, dtype=jnp.int32)
        xs = indexes[:, 0]
        ys = indexes[:, 1]
        vs = jnp.array(values, dtype=jnp.float32)

        new_field = self.add_liquid(xs, ys, vs)

        for c in cell:
            c.add_value = 0.0

        return new_field

    def reset(self) -> "PheromoneField":
        shape = jnp.array([self.ny, self.nx], dtype=jnp.int32)
        return self.replace(
            values_liquid=jnp.zeros(shape, dtype=jnp.float32),
            _values_gas=jnp.zeros(shape + 2, dtype=jnp.float32),
        )

    def set_neumann_boundary(self) -> "PheromoneField":
        new_mask = _set_boundary(self.mask, 0)
        return self.replace(mask=new_mask)

    def set_dirichlet_boundary(self) -> "PheromoneField":
        new_mask = _set_boundary(self.mask, 1)
        return self.replace(mask=new_mask)

    @property
    def shape(self):
        return self.values_liquid.shape
