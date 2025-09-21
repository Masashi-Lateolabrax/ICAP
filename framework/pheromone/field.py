import numpy as np
import jax.numpy as jnp
from jax import jit

from ..prelude import *
from .cell import PheromoneFieldCell


def dDistribution_dt(
        gas_values: jnp.ndarray,
        mask: jnp.ndarray,
        diffusion_coefficient: float,
        h: float,
        padding_value: float,
) -> jnp.ndarray:
    """
    Calculate time derivative of gas concentration distribution using 3D diffusion equation.
    
    Uses finite difference method with:
    - Standard 5-point stencil for horizontal (x,y) directions
    - Non-uniform asymmetric 3-point stencil for vertical (z) direction
    
    The z-direction uses exponentially spaced layers with intervals:
    z=0 to z=1: h, z=1 to z=2: 2*h, z=2 to z=3: 4*h, etc.
    """
    # Set boundary conditions: padding for x,y boundaries
    gas_values = gas_values.at[0, :, :].set(padding_value)   # x=0 boundary
    gas_values = gas_values.at[-1, :, :].set(padding_value)  # x=max boundary
    gas_values = gas_values.at[:, 0, :].set(padding_value)   # y=0 boundary
    gas_values = gas_values.at[:, -1, :].set(padding_value)  # y=max boundary
    
    # Z-direction boundaries: Neumann (copy) at bottom, Dirichlet (zero) at top
    gas_values = gas_values.at[:, :, 0].set(gas_values[:, :, 1])  # z=0: copy from z=1
    gas_values = gas_values.at[:, :, -1].set(0)                   # z=max: zero concentration

    # Interior points for finite difference calculation
    center = gas_values[1:-1, 1:-1, 1:-1]

    # Horizontal diffusion: standard centered difference (∇²c in x,y)
    # ∂²c/∂x² + ∂²c/∂y² = (c_{i+1,j} + c_{i-1,j} + c_{i,j+1} + c_{i,j-1} - 4c_{i,j}) / h²
    d_left = (gas_values[1:-1, 0:-2, 1:-1] - center) * mask[1:-1, 0:-2, None]
    d_right = (gas_values[1:-1, 2:, 1:-1] - center) * mask[1:-1, 2:, None]
    d_top = (gas_values[0:-2, 1:-1, 1:-1] - center) * mask[0:-2, 1:-1, None]
    d_bottom = (gas_values[2:, 1:-1, 1:-1] - center) * mask[2:, 1:-1, None]
    horizontal = (d_top + d_bottom + d_left + d_right) / (h * h)

    # Vertical diffusion: non-uniform asymmetric 3-point stencil (∇²c in z)
    # For non-uniform grid with spacing h below and 2h above current point:
    # ∂²c/∂z² = (c(z+2h) - 3c(z) + 2c(z-h)) / (3h²)
    # where h = z_weights[i] * h for each layer i
    d_upper = gas_values[1:-1, 1:-1, 2:] - center    # c(z+2h) - c(z)
    d_lower = gas_values[1:-1, 1:-1, 0:-2] - center  # c(z-h) - c(z)
    
    # Generate exponential spacing weights: [1, 2, 4, 8, 16, ...] for each z-layer
    z_weights = [2 ** i for i in range(gas_values.shape[2] - 2)]
    z_weights = jnp.array(z_weights, dtype=jnp.float32)
    
    # Apply asymmetric difference formula: (d_upper + 2*d_lower) / (3*h²)
    vertical = (d_upper + 2 * d_lower) / (3 * (z_weights[None, None, :] * h) ** 2)

    # Total diffusion: D * (∇²c_horizontal + ∇²c_vertical)
    return diffusion_coefficient * (horizontal + vertical)
def dEvaporation_dt(
        gas_values: jnp.ndarray,
        liquid_values: jnp.ndarray,
        saturation_pressure: float,
        evaporation_rate: float,
) -> jnp.ndarray:
    evaporation = (saturation_pressure - gas_values[1:-1, 1:-1]) * evaporation_rate
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
    d_evaporation = dEvaporation_dt(
        gas_values=gas_values,
        liquid_values=liquid_values,
        saturation_pressure=saturation_pressure,
        evaporation_rate=evaporation_rate
    )
    d_distribution = dDistribution_dt(
        gas_values=gas_values,
        mask=mask,
        diffusion_coefficient=diffusion_coefficient,
        dx=dx,
        padding_value=padding_value
    )
    d_decrease = dDecrease_dt(
        gas_values=gas_values,
        decrease_rate=decrease_rate
    )

    d_gas = (-d_decrease).at[1:-1, 1:-1].add(d_distribution)
    d_gas = d_gas.at[1:-1, 1:-1].add(d_evaporation)
    d_liquid = -d_evaporation

    return d_gas, d_liquid


@jit
def update_with_rk4(
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
    k1_gas, k1_liquid = d_dt(
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

    k2_gas, k2_liquid = d_dt(
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

    k3_gas, k3_liquid = d_dt(
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

    k4_gas, k4_liquid = d_dt(
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
    def __init__(
            self,
            nx: int,
            ny: int,
            dx: float,  # [m]
            material: Material,
            evaporation_rate: float,
            decrease_rate: float,
            temperature: float,  # [K]
            iter_: int = 1,
    ):
        # Parameter validation
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

        self.shape = jnp.array((ny, nx), dtype=jnp.int32)
        self.nz = 5  # Z-dimension size
        self.dx = dx

        self.saturation_pressure = material.saturation_pressure(temperature)  # [kPa]
        self.diffusion_coefficient = material.diffusion_coefficient(temperature)  # [m^2/s]
        self.evaporation_rate = evaporation_rate
        self.decrease_rate = decrease_rate
        self.temperature = temperature
        self.padding_value = 0.0

        self.iter_ = iter_

        self._values_liquid = jnp.zeros(self.shape, dtype=jnp.float32)  # [molecules]
        self._values_gas = jnp.zeros((ny + 2, nx + 2, self.nz + 2), dtype=jnp.float32)  # [molecules]
        self.mask = jnp.ones(self.shape, dtype=jnp.bool_)

    def reset(self):
        ny, nx = self.shape
        self._values_liquid = jnp.zeros(self.shape, dtype=jnp.float32)
        self._values_gas = jnp.zeros((ny + 2, nx + 2, self.nz + 2), dtype=jnp.float32)

    def set_neumann_boundary(self):
        self.mask = self.mask.at[0, :].set(0)
        self.mask = self.mask.at[-1, :].set(0)
        self.mask = self.mask.at[:, 0].set(0)
        self.mask = self.mask.at[:, -1].set(0)

    def set_dirichlet_boundary(self, value: float):
        self.padding_value = value
        self.mask = self.mask.at[0, :].set(1)
        self.mask = self.mask.at[-1, :].set(1)
        self.mask = self.mask.at[:, 0].set(1)
        self.mask = self.mask.at[:, -1].set(1)

    def get_gas(self, xs, ys, zs=0) -> np.ndarray:
        xs = jnp.clip(xs, 0, self.shape[1]) + 1
        ys = jnp.clip(ys, 0, self.shape[0]) + 1
        zs = jnp.clip(zs, 0, self.nz - 1) + 1
        return np.array(self._values_gas[ys, xs, zs])

    def get_liquid(self, xs, ys, zs=0) -> np.ndarray:
        xs = jnp.clip(xs, 0, self.shape[1] - 1)
        ys = jnp.clip(ys, 0, self.shape[0] - 1)
        return np.array(self._values_liquid[ys, xs])

    def get_gas_all(self) -> np.ndarray:
        return np.array(self._values_gas[1:-1, 1:-1, 0])

    def get_liquid_all(self) -> np.ndarray:
        return np.array(self._values_liquid)

    def add_liquid(self, xs, ys, values):  # values: [molecules]
        xs = jnp.clip(xs, 0, self.shape[1] - 1)
        ys = jnp.clip(ys, 0, self.shape[0] - 1)
        self._values_liquid = self._values_liquid.at[ys, xs, 0].add(values)

    def add_liquid_by_cell(self, cell: list[PheromoneFieldCell]):
        xs = jnp.array([c.index_x for c in cell if c.add_value > 0])
        ys = jnp.array([c.index_y for c in cell if c.add_value > 0])
        vs = jnp.array([c.add_value for c in cell if c.add_value > 0])

        if xs.size == 0 or ys.size == 0 or vs.size == 0:
            return

        xs = jnp.clip(xs, 0, self.shape[1] - 1)
        ys = jnp.clip(ys, 0, self.shape[0] - 1)

        self._values_liquid = self._values_liquid.at[ys, xs, 0].add(vs)

        for c in cell:
            c.add_value = 0.0

    def _update_with_rk4(self, dt: float):
        dt = dt / self.iter_
        for _ in range(self.iter_):
            self._values_gas, self._values_liquid = update_with_rk4(
                liquid_values=self._values_liquid,
                gas_values=self._values_gas,
                mask=self.mask,
                dx=self.dx,
                saturation_pressure=self.saturation_pressure,
                diffusion_coefficient=self.diffusion_coefficient,
                evaporation_rate=self.evaporation_rate,
                decrease_rate=self.decrease_rate,
                dt=dt,
                padding_value=self.padding_value,
            )

    def update(self, dt: float):
        self._update_with_rk4(dt)
