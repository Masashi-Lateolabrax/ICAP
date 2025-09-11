from functools import partial
from typing import Self

import mujoco
import mujoco.mjx as mjx

import numpy as np
import jax
import jax.numpy as jnp
from flax.struct import field, dataclass as jax_dataclass

from ..prelude import Settings, SimRenderTrait, SimPheromoneTrait
from ..pheromone import PheromoneField, PheromoneFieldCellSpec, add_pheromone_cells_to_mjspec


@jax_dataclass
class Consts:
    dt: float = field(pytree=False)
    PHEROMONE_CELL_SIZE: float = field(pytree=False)
    PHEROMONE_WIDTH: int = field(pytree=False)
    PHEROMONE_HEIGHT: int = field(pytree=False)


@jax_dataclass
class BasicSimulator(SimRenderTrait, SimPheromoneTrait):
    consts: Consts

    _data: mjx.Data
    _pheromone: PheromoneField

    _pheromone_cell_pos: jax.Array
    _pheromone_cell_site_ids: jax.Array

    @property
    def data(self) -> mjx.Data:
        return self._data

    @property
    def pheromone(self) -> PheromoneField:
        return self._pheromone

    def _update_parent(self, **kwargs: dict) -> Self:
        return self

    def update(
            self,
            data: mjx.Data = None,
            pheromone: PheromoneField = None,
            **kwargs
    ) -> Self:
        kwargs["_data"] = data
        kwargs["_pheromone"] = pheromone
        return self._update(**kwargs)

    @classmethod
    def new(cls, spec: mujoco.MjSpec, settings: Settings) -> tuple[mujoco.MjModel, Self]:
        p_cell_specs: list[PheromoneFieldCellSpec] = add_pheromone_cells_to_mjspec(
            spec, settings.Pheromone.WIDTH_NUM, settings.Pheromone.HEIGHT_NUM, settings.Pheromone.CELL_SIZE
        )

        mj_model: mujoco.MjModel = spec.compile()
        mj_data: mujoco.MjData = mujoco.MjData(mj_model)
        data = mjx.put_data(mj_model, mj_data)

        pheromone = PheromoneField.new(
            nx=settings.Pheromone.WIDTH_NUM,
            ny=settings.Pheromone.HEIGHT_NUM,
            dx=settings.Pheromone.CELL_SIZE,
            material=settings.Pheromone.MATERIAL,
            evaporation_rate=settings.Pheromone.EVAPORATION_RATE,
            decrease_rate=settings.Pheromone.DECREASE_RATE,
            temperature=settings.Simulation.TEMPERATURE,
            iter_=settings.Pheromone.ITERATIONS_PER_STEP
        )

        pheromone_cells = [s.get_cell(mj_model) for s in p_cell_specs]
        pheromone_cell_pos = np.zeros(
            (settings.Pheromone.HEIGHT_NUM, settings.Pheromone.WIDTH_NUM, 2), dtype=jnp.float32
        )
        pheromone_cell_site_ids = np.zeros(
            (settings.Pheromone.HEIGHT_NUM, settings.Pheromone.WIDTH_NUM), dtype=jnp.int32
        )
        for c in pheromone_cells:
            pheromone_cell_pos[c.index_y, c.index_x, :2] = c.pos[:2]
            pheromone_cell_site_ids[c.index_y, c.index_x] = c.id

        return mj_model, cls(
            consts=Consts(
                dt=settings.Simulation.TIME_STEP,
                PHEROMONE_CELL_SIZE=settings.Pheromone.CELL_SIZE,
                PHEROMONE_WIDTH=settings.Pheromone.WIDTH_NUM,
                PHEROMONE_HEIGHT=settings.Pheromone.HEIGHT_NUM
            ),

            _data=data,
            _pheromone=pheromone,

            _pheromone_cell_pos=jnp.array(pheromone_cell_pos),
            _pheromone_cell_site_ids=jnp.array(pheromone_cell_site_ids)
        )

    @partial(jax.jit, inline=True)
    def calc_nearest_pheromone_cell_indices(
            self, positions: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        # Use precomputed constants instead of runtime shape extraction
        width_num = self.consts.PHEROMONE_WIDTH
        height_num = self.consts.PHEROMONE_HEIGHT
        cell_size = self.consts.PHEROMONE_CELL_SIZE

        # Precomputed offsets
        width_offset = (width_num - 1) * 0.5
        height_offset = (height_num - 1) * 0.5

        # Direct coordinate conversion without intermediate clipping
        pos_xy = positions[:, :2]

        # Convert to grid indices with single clamp operation
        xi = jnp.clip(
            jnp.round(pos_xy[:, 0] / cell_size + width_offset).astype(jnp.int32),
            min=0, max=width_num - 1
        )
        yi = jnp.clip(
            jnp.round(height_offset - pos_xy[:, 1] / cell_size).astype(jnp.int32),
            min=0, max=height_num - 1
        )

        return xi, yi

    @partial(jax.jit, inline=True, donate_argnames=("self",))
    def step(self, model: mjx.Model) -> Self:
        return self.update(
            data=mjx.step(model, self.data),
            pheromone=self._pheromone.update(self.consts.dt)
        )

    @partial(jax.jit, static_argnames=("n", "unroll"), inline=True, donate_argnames=("self",))
    def step_n(self, model: mjx.Model, n: int, unroll: int = 1) -> Self:
        def body_fn(carry: "BasicSimulator", _x) -> tuple["BasicSimulator", None]:
            return carry.step(model), None

        return jax.lax.scan(body_fn, self, length=n, unroll=unroll)[0]

    @partial(jax.jit, inline=True, donate_argnames=("self",))
    def reset(self, model: mjx.Model) -> Self:
        return self.update(
            data=mjx.make_data(model),
            pheromone=self._pheromone.reset()
        )

    def render(self, img_buf: np.ndarray, camera: mujoco.MjvCamera, renderer: mujoco.Renderer):
        mj_model = renderer.model
        mj_data = mjx.get_data(mj_model, self.data)

        pheromone = np.array(self._pheromone.values_gas)
        max_pheromone = np.max(pheromone)
        total_pheromone = np.sum(pheromone)

        normalized_pheromone = pheromone / max_pheromone
        colored_pheromone = np.stack([
            normalized_pheromone,
            np.zeros_like(normalized_pheromone),
            1 - normalized_pheromone,
            np.full_like(normalized_pheromone, 0.5)
        ], axis=-1).astype(np.float32)

        mj_model.site_rgba[self._pheromone_cell_site_ids, :] = colored_pheromone

        renderer.update_scene(mj_data, camera)
        renderer.render(out=img_buf)

        return {"max": max_pheromone, "total": total_pheromone}
