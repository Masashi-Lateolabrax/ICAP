from typing import Self

import mujoco
import mujoco.mjx as mjx

import numpy as np
import jax
import jax.numpy as jnp
from flax.struct import dataclass as jax_dataclass

from ..prelude import Settings, SimRenderTrait
from ..pheromone import PheromoneField, PheromoneFieldCellSpec, add_pheromone_cells_to_mjspec


@jax_dataclass
class Consts:
    dt: float


@jax_dataclass
class BasicSimulator(SimRenderTrait):
    consts: Consts

    model: mjx.Model
    data: mjx.Data
    pheromone: PheromoneField

    _pheromone_cell_pos: jax.Array
    _pheromone_cell_site_ids: jax.Array

    def _update_parent(self, **kwargs: dict) -> Self:
        return self

    def update(
            self,
            model: mjx.Model = None,
            data: mjx.Data = None,
            pheromone: PheromoneField = None,
            **kwargs
    ) -> Self:
        kwargs["model"] = model
        kwargs["data"] = data
        kwargs["pheromone"] = pheromone
        return self._update(**kwargs)

    @classmethod
    def new(cls, spec: mujoco.MjSpec, settings: Settings) -> tuple[mujoco.MjModel, Self]:
        p_cell_specs: list[PheromoneFieldCellSpec] = add_pheromone_cells_to_mjspec(
            spec, settings.Pheromone.WIDTH_NUM, settings.Pheromone.HEIGHT_NUM, settings.Pheromone.CELL_SIZE
        )

        mj_model: mujoco.MjModel = spec.compile()
        model = mjx.put_model(mj_model)
        data = mjx.make_data(model)

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
            ),

            model=model,
            data=data,
            pheromone=pheromone,

            _pheromone_cell_pos=jnp.array(pheromone_cell_pos),
            _pheromone_cell_site_ids=jnp.array(pheromone_cell_site_ids)
        )

    @staticmethod
    @jax.jit
    def _calc_nearest_pheromone_cell_indices(
            this: "BasicSimulator", positions: jax.Array
    ) -> jax.Array:
        dists = jnp.linalg.norm(positions[:, None, :2] - this._pheromone_cell_pos[None, :, :2], axis=2)
        return jnp.array(jnp.unravel_index(jnp.argmin(dists), dists.shape))

    def get_pheromone(self, positions: jax.Array) -> jax.Array:
        nearest_indices = BasicSimulator._calc_nearest_pheromone_cell_indices(self, positions)
        return self.pheromone.get_gas(nearest_indices[:, 0], nearest_indices[:, 1])

    def add_pheromone(self, positions: jax.Array, values: jax.Array) -> PheromoneField:
        nearest_indices = BasicSimulator._calc_nearest_pheromone_cell_indices(self, positions)
        new_pheromone = self.pheromone.add_liquid(nearest_indices[:, 0], nearest_indices[:, 1], values)
        return new_pheromone

    @staticmethod
    @jax.jit
    def _step(this: "BasicSimulator"):
        new_data = mjx.step(this.model, this.data)
        new_pheromone = this.pheromone.update(this.consts.dt)
        new_simulator = this.update(data=new_data, pheromone=new_pheromone)
        return new_simulator

    def step(self) -> Self:
        return BasicSimulator._step(self)

    @staticmethod
    @jax.jit
    def _step_n(simulator: "BasicSimulator", n: int) -> "BasicSimulator":
        def body_fn(_i, sim: "BasicSimulator"):
            return BasicSimulator._step(sim)

        new_simulator = jax.lax.fori_loop(0, n, body_fn, simulator)
        return new_simulator

    def step_n(self, n: int) -> Self:
        return BasicSimulator._step_n(self, n)

    def reset(self) -> Self:
        new_data = mjx.make_data(self.model)
        new_pheromone = self.pheromone.reset()
        return self.update(data=new_data, pheromone=new_pheromone)

    def render(self, img_buf: np.ndarray, camera: mujoco.MjvCamera, renderer: mujoco.Renderer):
        mj_model = renderer.model
        mj_data = mjx.get_data(mj_model, self.data)

        pheromone = np.array(self.pheromone.values_gas)
        max_pheromone = np.max(pheromone) + 1e-12
        total_pheromone = np.sum(pheromone)

        def paint(value):
            return np.array(
                [value / max_pheromone, 0.0, 1 - value / max_pheromone, 0.5],
                dtype=np.float64
            )

        colored_pheromone = np.vectorize(paint)(pheromone)

        mj_model.site_rgba[self._pheromone_cell_site_ids, :] = colored_pheromone

        renderer.update_scene(mj_data, camera)
        renderer.render(out=img_buf)

        return {"max": max_pheromone, "total": total_pheromone}
