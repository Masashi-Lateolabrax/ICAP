import numpy as np
import jax.numpy as jnp
from jax import jit

from ..prelude import *
from .cell import PheromoneFieldCell


@jit
def dDistribution_dt(
        gas_values: jnp.ndarray,
        mask: jnp.ndarray,
        diffusion_coefficient: float,
        dx: float,
        padding_value: float,
) -> jnp.ndarray:
    gas_values = gas_values.at[0, :].set(padding_value)
    gas_values = gas_values.at[-1, :].set(padding_value)
    gas_values = gas_values.at[:, 0].set(padding_value)
    gas_values = gas_values.at[:, -1].set(padding_value)

    center = gas_values[1:-1, 1:-1]
    d_left = (gas_values[1:-1, 0:-2] - center) * mask[1:-1, 0:-2]
    d_right = (gas_values[1:-1, 2:] - center) * mask[1:-1, 2:]
    d_top = (gas_values[0:-2, 1:-1] - center) * mask[0:-2, 1:-1]
    d_bottom = (gas_values[2:, 1:-1] - center) * mask[2:, 1:-1]
    return diffusion_coefficient * (d_top + d_bottom + d_left + d_right) / (dx * dx)


@jit
def dEvaporation_dt(
        material: Material,
        gas_values: jnp.ndarray,
        liquid_values: jnp.ndarray,
        temperature: float,
        evaporation_rate: float,
) -> jnp.ndarray:
    c_sat = material.saturation_pressure(temperature)
    evaporation = (c_sat - gas_values) * evaporation_rate
    evaporation = jnp.minimum(evaporation, liquid_values)
    return evaporation


@jit
def dDecrease_dt(
        gas_values: jnp.ndarray,
        decrease_rate: float,
) -> jnp.ndarray:
    return gas_values * decrease_rate


@jit
def d_dt(
        material: Material,
        liquid_values: jnp.ndarray,
        gas_values: jnp.ndarray,
        mask: jnp.ndarray,
        dx: float,
        temperature: float,
        diffusion_coefficient: float,
        evaporation_rate: float,
        decrease_rate: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    d_evaporation = dEvaporation_dt(
        material=material,
        gas_values=gas_values,
        liquid_values=liquid_values,
        temperature=temperature,
        evaporation_rate=evaporation_rate
    )
    d_distribution = dDistribution_dt(
        gas_values=gas_values,
        mask=mask,
        diffusion_coefficient=diffusion_coefficient,
        dx=dx
    )
    d_decrease = dDecrease_dt(
        gas_values=gas_values,
        decrease_rate=decrease_rate
    )

    d_gas = d_evaporation + d_distribution - d_decrease
    d_liquid = -d_evaporation

    return d_gas, d_liquid


@jit
def update_with_rk4(
        material: Material,
        liquid_values: jnp.ndarray,
        gas_values: jnp.ndarray,
        mask: jnp.ndarray,
        dx: float,
        temperature: float,
        diffusion_coefficient: float,
        evaporation_rate: float,
        decrease_rate: float,
        dt: float,
        padding_value: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    k1_gas, k1_liquid = d_dt(
        material=material,
        liquid_values=liquid_values,
        gas_values=gas_values,
        mask=mask,
        dx=dx,
        temperature=temperature,
        diffusion_coefficient=diffusion_coefficient,
        evaporation_rate=evaporation_rate,
        decrease_rate=decrease_rate,
        padding_value=padding_value,
    )

    k2_gas, k2_liquid = d_dt(
        material=material,
        liquid_values=liquid_values + 0.5 * dt * k1_liquid,
        gas_values=gas_values + 0.5 * dt * k1_gas,
        mask=mask,
        dx=dx,
        temperature=temperature,
        diffusion_coefficient=diffusion_coefficient,
        evaporation_rate=evaporation_rate,
        decrease_rate=decrease_rate,
        padding_value=padding_value
    )

    k3_gas, k3_liquid = d_dt(
        material=material,
        liquid_values=liquid_values + 0.5 * dt * k2_liquid,
        gas_values=gas_values + 0.5 * dt * k2_gas,
        mask=mask,
        dx=dx,
        temperature=temperature,
        diffusion_coefficient=diffusion_coefficient,
        evaporation_rate=evaporation_rate,
        decrease_rate=decrease_rate,
        padding_value=padding_value
    )

    k4_gas, k4_liquid = d_dt(
        material=material,
        liquid_values=liquid_values + dt * k3_liquid,
        gas_values=gas_values + dt * k3_gas,
        mask=mask,
        dx=dx,
        temperature=temperature,
        diffusion_coefficient=diffusion_coefficient,
        evaporation_rate=evaporation_rate,
        decrease_rate=decrease_rate,
        padding_value=padding_value
    )

    gas_values = jnp.maximum(0.0, gas_values + (k1_gas + 2 * k2_gas + 2 * k3_gas + k4_gas) / 6)
    liquid_values = jnp.maximum(0.0, liquid_values + (k1_liquid + 2 * k2_liquid + 2 * k3_liquid + k4_liquid) / 6)

    return gas_values, liquid_values


class PheromoneField:
    def __init__(
            self,
            nx: int,
            ny: int,
            dx: float,
            material: Material,
            diffusion_coefficient: float,
            evaporation_rate: float,
            decrease_rate: float,
            temperature: float,
            iter_: int = 1,
    ):
        # Parameter validation
        if nx <= 0 or ny <= 0:
            raise ValueError("Grid dimensions must be positive")
        if dx <= 0:
            raise ValueError("Grid spacing dx must be positive")
        if diffusion_coefficient < 0:
            raise ValueError("Diffusion coefficient must be non-negative")
        if evaporation_rate < 0 or decrease_rate < 0:
            raise ValueError("Rates must be non-negative")
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if iter_ <= 0:
            raise ValueError("Iteration count must be positive")

        self.shape = jnp.array((ny, nx), dtype=jnp.int32)
        self.dx = dx

        self.material = material
        self.diffusion_coefficient = diffusion_coefficient
        self.evaporation_rate = evaporation_rate
        self.decrease_rate = decrease_rate
        self.temperature = temperature
        self.padding_value = 0.0

        self.iter_ = iter_

        self._values_liquid = jnp.zeros(self.shape, dtype=jnp.float32)
        self._values_gas = jnp.zeros(self.shape, dtype=jnp.float32)
        self.mask = jnp.ones(self.shape, dtype=jnp.bool_)

    def get_gas(self, xs, ys) -> np.ndarray:
        xs = jnp.clip(xs, 0, self.shape[1] - 1)
        ys = jnp.clip(ys, 0, self.shape[0] - 1)
        return np.array(self._values_gas[ys, xs])

    def get_liquid(self, xs, ys) -> np.ndarray:
        xs = jnp.clip(xs, 0, self.shape[1] - 1)
        ys = jnp.clip(ys, 0, self.shape[0] - 1)
        return np.array(self._values_liquid[ys, xs])

    def get_gas_all(self) -> np.ndarray:
        return np.array(self._values_gas)

    def get_liquid_all(self) -> np.ndarray:
        return np.array(self._values_liquid)

    def add_liquid(self, xs, ys, values):
        xs = jnp.clip(xs, 0, self.shape[1] - 1)
        ys = jnp.clip(ys, 0, self.shape[0] - 1)
        self._values_liquid = self._values_liquid.at[ys, xs].add(values)

    def add_liquid_by_cell(self, cell: list[PheromoneFieldCell]):
        xs = jnp.array([c.index_x for c in cell if c.add_value > 0])
        ys = jnp.array([c.index_y for c in cell if c.add_value > 0])
        vs = jnp.array([c.add_value for c in cell if c.add_value > 0])

        if xs.size == 0 or ys.size == 0 or vs.size == 0:
            return

        xs = jnp.clip(xs, 0, self.shape[1] - 1)
        ys = jnp.clip(ys, 0, self.shape[0] - 1)

        self._values_liquid = self._values_liquid.at[ys, xs].add(vs)

        for c in cell:
            c.add_value = 0.0

    def _update_with_rk4(self, dt: float):
        dt = dt / self.iter_
        for _ in range(self.iter_):
            self._values_gas, self._values_liquid = update_with_rk4(
                material=self.material,
                liquid_values=self._values_liquid,
                gas_values=self._values_gas,
                mask=self.mask,
                dx=self.dx,
                temperature=self.temperature,
                diffusion_coefficient=self.diffusion_coefficient,
                evaporation_rate=self.evaporation_rate,
                decrease_rate=self.decrease_rate,
                dt=dt,
                padding_value=self.padding_value,
            )

    def update(self, dt: float):
        self._update_with_rk4(dt)
