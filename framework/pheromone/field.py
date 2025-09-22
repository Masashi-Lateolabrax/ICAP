from functools import partial
from typing import Self

import jax
import jax.numpy as jnp
from flax.struct import field, dataclass as jax_dataclass

from ..prelude import *
from .cell import PheromoneFieldCell


@partial(jax.jit, inline=True)
def _set_boundary(values, fill):
    values = values.at[0, :].set(fill)
    values = values.at[-1, :].set(fill)
    values = values.at[:, 0].set(fill)
    values = values.at[:, -1].set(fill)
    return values


def _dDiffusion_dt(
        gas_values: jnp.ndarray,
        mask: jnp.ndarray,
        diffusion_coefficient: float,
        h: float,
        padding_value: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calculate time derivative of gas concentration distribution using 3D diffusion equation.

    Uses finite difference method with:
    - Standard 5-point stencil for horizontal (x,y) directions
    - Non-uniform asymmetric 3-point stencil for vertical (z) direction

    The z-direction uses exponentially spaced layers with intervals:
    z=0 to z=1: h, z=1 to z=2: 2*h, z=2 to z=3: 4*h, etc.
    """

    # Set boundary conditions: padding for x,y boundaries
    # IMPORTANT: Array indexing is [y, x, z] order (row, column, depth)
    gas_values = _set_boundary(gas_values, padding_value)

    # Z-direction boundaries: Neumann (copy) at bottom, Dirichlet (zero) at top
    gas_values = gas_values.at[:, :, 0].set(gas_values[:, :, 1])  # z=0: copy from z=1
    gas_values = gas_values.at[:, :, -1].set(0)  # z=max: zero concentration

    # Interior points for finite difference calculation
    center = gas_values[1:-1, 1:-1, 1:-1]

    # Horizontal diffusion: standard centered difference (∇²c in x,y)
    # ∂²c/∂x² + ∂²c/∂y² = (c_{i+1,j} + c_{i-1,j} + c_{i,j+1} + c_{i,j-1} - 4c_{i,j}) / h²
    d_left = (gas_values[1:-1, 0:-2, 1:-1] - center) * mask[1:-1, 0:-2, None]
    d_right = (gas_values[1:-1, 2:, 1:-1] - center) * mask[1:-1, 2:, None]
    d_top = (gas_values[0:-2, 1:-1, 1:-1] - center) * mask[0:-2, 1:-1, None]
    d_bottom = (gas_values[2:, 1:-1, 1:-1] - center) * mask[2:, 1:-1, None]

    d_dx = (d_left[:, :, 0] - d_right[:, :, 0]) * 0.5 / h  # ∂c/∂x
    d_dy = (d_top[:, :, 0] - d_bottom[:, :, 0]) * 0.5 / h  # ∂c/∂y
    horizontal = (d_top + d_bottom + d_left + d_right) / (h * h)

    # Vertical diffusion: non-uniform asymmetric 3-point stencil (∇²c in z)
    # For non-uniform grid with spacing h below and 2h above current point:
    # ∂²c/∂z² = (c(z+2h) - 3c(z) + 2c(z-h)) / (3h²)
    # where h = z_weights[i] * h for each layer i
    #
    # NOTE: The following implementation is mathematically correct.
    # Expanding: (d_upper + 2*d_lower) / (3h²) where:
    # d_upper = c(z+2h) - c(z)
    # d_lower = c(z-h) - c(z)
    # Results in: (c(z+2h) - c(z) + 2*(c(z-h) - c(z))) / (3h²)
    #           = (c(z+2h) - 3*c(z) + 2*c(z-h)) / (3h²) ✓
    d_upper = gas_values[1:-1, 1:-1, 2:] - center  # c(z+2h) - c(z)
    d_lower = gas_values[1:-1, 1:-1, 0:-2] - center  # c(z-h) - c(z)

    # Generate exponential spacing weights: [1, 2, 4, 8, 16, ...] for each z-layer
    z_weights = [2 ** i for i in range(gas_values.shape[2] - 2)]
    z_weights = jnp.array(z_weights, dtype=jnp.float32)

    # Apply asymmetric difference formula: (d_upper + 2*d_lower) / (3*h²)
    vertical = (d_upper + 2 * d_lower) / (3 * (z_weights[None, None, :] * h) ** 2)

    # Total diffusion: D * (∇²c_horizontal + ∇²c_vertical)
    return diffusion_coefficient * (horizontal + vertical), d_dx, d_dy


def _d_dt(
        gas_values: jnp.ndarray,
        mask: jnp.ndarray,
        h: float,
        diffusion_coefficient: float,
        padding_value: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    d_diffusion, d_dx, d_dy = _dDiffusion_dt(  # Unit: mol/(m^3·s)
        gas_values=gas_values,
        mask=mask,
        diffusion_coefficient=diffusion_coefficient,
        h=h,
        padding_value=padding_value
    )

    d_gas = d_diffusion  # Unit: mol/(m^3·s)

    return d_gas, jnp.stack([d_dx, d_dy], axis=2)


@partial(jax.jit, static_argnames=(
        "saturation_pressure", "diffusion_coefficient", "evaporation_rate", "decrease_rate", "padding_value",
), inline=True)
def _step_with_rk4(
        liquid_values: jnp.ndarray,
        gas_values: jnp.ndarray,
        mask: jnp.ndarray,
        h: float,
        saturation_concentration: float,
        diffusion_coefficient: float,
        dt: float,
        padding_value: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    k1_gas, k1_grad = _d_dt(
        gas_values=gas_values,
        mask=mask,
        h=h,
        diffusion_coefficient=diffusion_coefficient,
        padding_value=padding_value,
    )

    k2_gas, k2_grad = _d_dt(
        gas_values=gas_values.at[1:-1, 1:-1, 1:-1].add(0.5 * dt * k1_gas),
        mask=mask,
        h=h,
        diffusion_coefficient=diffusion_coefficient,
        padding_value=padding_value
    )

    k3_gas, k3_grad = _d_dt(
        gas_values=gas_values.at[1:-1, 1:-1, 1:-1].add(0.5 * dt * k2_gas),
        mask=mask,
        h=h,
        diffusion_coefficient=diffusion_coefficient,
        padding_value=padding_value
    )

    k4_gas, k4_grad = _d_dt(
        gas_values=gas_values.at[1:-1, 1:-1, 1:-1].add(dt * k3_gas),
        mask=mask,
        h=h,
        diffusion_coefficient=diffusion_coefficient,
        padding_value=padding_value
    )

    grad = (k1_grad + 2 * k2_grad + 2 * k3_grad + k4_grad) / 6

    evaporation_mol = (saturation_concentration - gas_values[1:-1, 1:-1, 1]) * (liquid_values > 0) * (h ** 3)
    evaporation_mol = jnp.minimum(evaporation_mol, liquid_values)
    evaporation_con = evaporation_mol / (h ** 3)

    d_gas_values = ((k1_gas + 2 * k2_gas + 2 * k3_gas + k4_gas) / 6).at[:, :, 0].add(evaporation_con)
    gas_values = jnp.maximum(
        0.0,
        gas_values.at[1:-1, 1:-1, 1:-1].add(d_gas_values)
    )

    liquid_values = jnp.maximum(
        0.0,
        liquid_values - evaporation_mol
    )

    return gas_values, liquid_values, grad


@partial(
    jax.jit,
    static_argnames=(
            "saturation_pressure", "diffusion_coefficient", "evaporation_rate", "decrease_rate", "padding_value",
            "iter_"
    ),
    donate_argnames=("liquid_values", "gas_values"),
    inline=True,
)
def _iter_update_with_rk4(
        liquid_values: jnp.ndarray,
        gas_values: jnp.ndarray,
        mask: jnp.ndarray,
        dx: float,
        saturation_concentration: float,
        diffusion_coefficient: float,
        dt: float,
        padding_value: float,
        iter_: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    dt = dt / iter_

    def body_fn(carry: tuple[jax.Array, jax.Array, jax.Array], _x):
        g_values, l_values, _grad = carry
        g_values, l_values, grad, = _step_with_rk4(
            liquid_values=l_values,
            gas_values=g_values,
            mask=mask,
            h=dx,
            saturation_concentration=saturation_concentration,
            diffusion_coefficient=diffusion_coefficient,
            dt=dt,
            padding_value=padding_value,
        )
        return (g_values, l_values, grad), None

    return jax.lax.scan(
        body_fn,
        (gas_values, liquid_values, liquid_values),
        length=iter_,
    )[0]


@jax_dataclass
class PheromoneField:
    dt: float = field(pytree_node=False)

    nx: int = field(pytree_node=False)
    ny: int = field(pytree_node=False)
    nz: int = field(pytree_node=False)
    dx: float = field(pytree_node=False)

    saturation_concentration: float = field(pytree_node=False)
    diffusion_coefficient: float = field(pytree_node=False)

    values_liquid: jnp.ndarray  # Shape: (x, y)
    _values_gas: jnp.ndarray  # Shape: (x+2, y+2)
    grad: jnp.ndarray
    mask: jnp.ndarray  # Shape: (x+2, y+2)

    padding_value: float = field(pytree_node=False)
    iter_: int = field(pytree_node=False)

    @property
    def values_gas(self) -> jnp.ndarray:
        return self._values_gas[1:-1, 1:-1]

    @classmethod
    def new(
            cls,
            dt: float,  # [s] - time step
            nx: int,
            ny: int,
            dx: float,
            temperature: float,
            material: Material,
            padding_value: float = 0.0,
            iter_: int = 1,
            nz: int = 5,
    ) -> Self:
        saturation_pressure = material.saturation_pressure(temperature)
        saturation_concentration = saturation_pressure / (Material.GAS_CONSTANT * temperature)  # [mol/m^3]
        diffusion_coefficient = material.diffusion_coefficient(temperature)

        return cls(
            dt=dt,
            nx=nx,
            ny=ny,
            nz=nz,
            dx=dx,

            saturation_concentration=saturation_concentration,
            diffusion_coefficient=diffusion_coefficient,

            values_liquid=jnp.zeros((ny, nx), dtype=jnp.float32),
            _values_gas=jnp.zeros((ny + 2, nx + 2, nz + 2), dtype=jnp.float32),
            grad=jnp.zeros((ny, nx, 2), dtype=jnp.float32),
            mask=jnp.ones((ny + 2, nx + 2), dtype=jnp.bool_),

            padding_value=padding_value,
            iter_=iter_
        )

    def step(self) -> Self:
        new_gas, new_liquid, new_grad = _iter_update_with_rk4(
            liquid_values=self.values_liquid,
            gas_values=self._values_gas,
            mask=self.mask,
            dx=self.dx,
            saturation_concentration=self.saturation_concentration,
            diffusion_coefficient=self.diffusion_coefficient,
            dt=self.dt,
            padding_value=self.padding_value,
            iter_=self.iter_,
        )
        return self.replace(
            values_liquid=new_liquid,
            _values_gas=new_gas,
        )

    @staticmethod
    @partial(jax.jit, inline=True)
    def _clip_indices(xs: jax.Array, ys: jax.Array, shape: tuple) -> tuple[jax.Array, jax.Array]:
        xs = jnp.clip(xs.astype(jnp.int32), 0, shape[1] - 1)
        ys = jnp.clip(ys.astype(jnp.int32), 0, shape[0] - 1)
        return xs, ys

    @staticmethod
    @partial(jax.jit, inline=True)
    def _get(values, xs, ys) -> jnp.ndarray:
        xs, ys = PheromoneField._clip_indices(xs, ys, values.shape)
        return values[ys, xs]

    def get_gas(self, xs, ys) -> jnp.ndarray:
        return self._get(self.values_gas[1:-1, 1:-1, 0], xs, ys)

    def get_liquid(self, xs, ys) -> jnp.ndarray:
        return self._get(self.values_liquid, xs, ys)

    @staticmethod
    @partial(jax.jit, inline=True)
    def _add(values: jax.Array, xs: jax.Array, ys: jax.Array, additions: jax.Array):
        xs, ys = PheromoneField._clip_indices(xs, ys, values.shape)
        return values.at[ys, xs].add(additions)

    def add_liquid(self, xs, ys, additions) -> Self:
        new_liquid = self._add(self.values_liquid, xs, ys, additions)
        return self.replace(values_liquid=new_liquid)

    def add_liquid_by_cell(self, cell: list[PheromoneFieldCell]) -> Self:
        indexes = []
        values = []
        for c in cell:
            if c.add_value > 0:
                indexes.append((c.index_x, c.index_y))
                values.append(c.add_value)
                c.add_value = 0

        if not indexes:
            return self

        indexes = jnp.array(indexes, dtype=jnp.int32)
        xs = indexes[:, 0]
        ys = indexes[:, 1]
        vs = jnp.array(values, dtype=jnp.float32)

        new_field = self.add_liquid(xs, ys, vs)

        return new_field

    @partial(jax.jit, inline=True)
    def reset(self) -> Self:
        return self.replace(
            values_liquid=jnp.zeros((self.ny, self.nx), dtype=jnp.float32),
            _values_gas=jnp.zeros((self.ny + 2, self.nx + 2, self.nz + 2), dtype=jnp.float32),
            grad=jnp.zeros((self.ny, self.nx, 2), dtype=jnp.float32),
        )

    def set_neumann_boundary(self) -> Self:
        new_mask = _set_boundary(self.mask, 0)
        return self.replace(mask=new_mask)

    def set_dirichlet_boundary(self) -> "PheromoneField":
        new_mask = _set_boundary(self.mask, 1)
        return self.replace(mask=new_mask)

    @property
    def shape(self):
        return self.values_liquid.shape
