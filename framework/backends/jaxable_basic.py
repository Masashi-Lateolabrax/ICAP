import mujoco
import mujoco.mjx as mjx

import numpy as np
import jax
import jax.numpy as jnp
from flax.struct import dataclass as jax_dataclass

from ..prelude import Settings
from ..pheromone import PheromoneField, PheromoneFieldCellSpec
from .basic_environment import add_pheromone_cells_in_mjspec


@jax_dataclass
class BasicSimulator:
    model: mjx.Model
    data: mjx.Data
    pheromone: PheromoneField

    pheromone_cell_ind: jax.Array
    pheromone_cell_pos: jax.Array

    def update(
            self,
            model: mjx.Model = None,
            data: mjx.Data = None,
            pheromone: PheromoneField = None,
            pheromone_cell_ind: jax.Array = None,
            pheromone_cell_pos: jax.Array = None
    ) -> 'BasicSimulator':
        kwargs = {
            "model": model,
            "data": data,
            "pheromone": pheromone,
            "pheromone_cell_ind": pheromone_cell_ind,
            "pheromone_cell_pos": pheromone_cell_pos
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if not kwargs:
            return self
        return self.replace(**kwargs)

    @classmethod
    def new(cls, spec: mujoco.MjSpec, settings: Settings) -> 'BasicSimulator':
        p_cell_specs: list[PheromoneFieldCellSpec] = add_pheromone_cells_in_mjspec(spec, settings)

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
        pheromone_cell_ind = jnp.array(
            [(cell.index_x, cell.index_y) for cell in pheromone_cells], dtype=jnp.int32
        )
        pheromone_cell_pos = jnp.array(
            [(cell.pos[0], cell.pos[1]) for cell in pheromone_cells], dtype=jnp.float32
        )

        return cls(
            model=model,
            data=data,
            pheromone=pheromone,
            pheromone_cell_ind=pheromone_cell_ind,
            pheromone_cell_pos=pheromone_cell_pos,
        )

    @staticmethod
    @jax.jit
    def _calc_nearest_pheromone_cell_indices(
            simulator: "BasicSimulator", positions: jax.Array
    ) -> jax.Array:
        dists = jnp.linalg.norm(positions[:, None, :2] - simulator.pheromone_cell_pos[None, :, :2], axis=2)
        nearest_indices = jnp.argmin(dists, axis=1, keepdims=True)
        return nearest_indices

    def calc_nearest_pheromone_cell_indices(self, positions: jax.Array) -> jax.Array:
        return BasicSimulator._calc_nearest_pheromone_cell_indices(self, positions)

    def get_pheromone(self, positions: jax.Array) -> jax.Array:
        nearest_indices = self.calc_nearest_pheromone_cell_indices(positions)
        return self.pheromone.get_gas(nearest_indices[:, 0], nearest_indices[:, 1])

    def add_pheromone(self, positions: jax.Array, values: jax.Array) -> 'BasicSimulator':
        nearest_indices = self.calc_nearest_pheromone_cell_indices(positions)
        new_pheromone = self.pheromone.add_liquid(nearest_indices[:, 0], nearest_indices[:, 1], values)
        new_simulator = self.update(pheromone=new_pheromone)
        return new_simulator

    @staticmethod
    @jax.jit
    def _step(simulator: "BasicSimulator", dt: float):
        new_data = mjx.step(simulator.model, simulator.data)
        new_pheromone = simulator.pheromone.update(dt)
        new_simulator = simulator.update(data=new_data, pheromone=new_pheromone)
        return new_simulator

    def step(self, dt: float) -> 'BasicSimulator':
        return BasicSimulator._step(self, dt)

    @staticmethod
    @jax.jit
    def _step_n(simulator: "BasicSimulator", n: int, dt: float) -> "BasicSimulator":
        def body_fn(_i, sim: "BasicSimulator"):
            return BasicSimulator._step(sim, dt)

        new_simulator = jax.lax.fori_loop(0, n, body_fn, simulator)
        return new_simulator

    def step_n(self, n: int, dt: float) -> 'BasicSimulator':
        return BasicSimulator._step_n(self, n, dt)

    def render(
            self,
            mj_model: mujoco.MjModel,
            img_buf: np.ndarray,
            pos: tuple[float, float, float],
            lookat: tuple[float, float, float],
            max_geom=100,
            max_pheromone=1.0
    ):
        from framework.backends.utils import render

        if img_buf is None:
            return

        mj_data = mjx.get_data(mj_model, self.data)

        pheromone: jnp.ndarray = self.pheromone.values_gas
        for (ix, iy) in zip(self.pheromone_cell_ind[:, 0], self.pheromone_cell_ind[:, 1]):
            pheromone_value = float(pheromone[iy, ix])
            rgba: tuple[float, float, float] = (
                pheromone_value / max_pheromone, 0.0, 1 - pheromone_value / max_pheromone
            )
            mj_model.geom(f"pheromone_cell_{ix}_{iy}").rgba = np.array((*rgba, 0.5), dtype=np.float64)

        render(mj_model, mj_data, mj_model.cam_resolution, max_geom, img_buf, pos, lookat)

    def reset(self) -> 'BasicSimulator':
        new_data = mjx.make_data(self.model)
        new_data = mjx.step(self.model, new_data)

        new_pheromone = self.pheromone.reset()

        return self.update(data=new_data, pheromone=new_pheromone)
