from functools import partial
from typing import Self

import mujoco
import mujoco.mjx as mjx

import numpy as np
import jax
import jax.numpy as jnp
from flax.struct import field, dataclass as jax_dataclass

from ..prelude import Settings, SimRenderTrait, SimPheromoneTrait
from ..pheromone import PheromoneField
from ..mkenv import add_texture, add_material, add_geom


@jax_dataclass
class Consts:
    dt: float = field(pytree_node=False)
    PHEROMONE_CELL_SIZE: float = field(pytree_node=False)
    PHEROMONE_WIDTH: int = field(pytree_node=False)
    PHEROMONE_HEIGHT: int = field(pytree_node=False)


@jax_dataclass
class BasicSimulator(SimRenderTrait, SimPheromoneTrait):
    consts: Consts

    _data: mjx.Data
    _pheromone: PheromoneField

    _pheromone_cell_pos: jax.Array
    _pheromone_texture_id: int = field(pytree_node=False)  # MuJoCo texture ID for runtime updates

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

    @staticmethod
    def _add_pheromone_visualization(spec: mujoco.MjSpec, settings: Settings) -> None:
        """Add runtime-updatable texture for pheromone visualization."""
        # Use wrapper for custom data texture
        add_texture(
            spec,
            name="pheromone_viz",
            type_=mujoco.mjtTexture.mjTEXTURE_2D,
            width=settings.Pheromone.WIDTH_NUM,
            height=settings.Pheromone.HEIGHT_NUM,
            data=np.zeros((settings.Pheromone.HEIGHT_NUM, settings.Pheromone.WIDTH_NUM, 3), dtype=np.uint8)
        )

        # Use wrapper for material
        add_material(spec, name="pheromone_mat", texture="pheromone_viz", texrepeat=(1, 1))

        # Use thin box geometry (not plane) positioned above ground
        add_geom(
            spec.worldbody,
            geom_type=mujoco.mjtGeom.mjGEOM_BOX,
            name="pheromone_overlay",
            size=(settings.Simulation.WORLD_WIDTH * 0.5, settings.Simulation.WORLD_HEIGHT * 0.5, 0.01),
            pos=(0, 0, 0.01),
            material="pheromone_mat",
            rgba=(1, 1, 1, 0.7),
            condim=0  # Disable collision
        )

    @classmethod
    def new(cls, spec: mujoco.MjSpec, settings: Settings) -> tuple[mujoco.MjModel, Self]:
        cls._add_pheromone_visualization(spec, settings)

        mj_model: mujoco.MjModel = spec.compile()
        mj_data: mujoco.MjData = mujoco.MjData(mj_model)
        data = mjx.put_data(mj_model, mj_data)

        pheromone_texture_id = mj_model.texture("pheromone_viz").id

        pheromone = PheromoneField.new(
            dt=settings.Simulation.TIME_STEP,
            nx=settings.Pheromone.WIDTH_NUM,
            ny=settings.Pheromone.HEIGHT_NUM,
            dx=settings.Pheromone.CELL_SIZE,
            temperature=settings.Simulation.TEMPERATURE,
            material=settings.Pheromone.MATERIAL,
            iter_=settings.Pheromone.ITERATIONS_PER_STEP
        )

        # Calculate pheromone cell world positions (for reference)
        pheromone_cell_pos = np.zeros(
            (settings.Pheromone.HEIGHT_NUM, settings.Pheromone.WIDTH_NUM, 2), dtype=np.float32
        )
        for x in range(settings.Pheromone.WIDTH_NUM):
            for y in range(settings.Pheromone.HEIGHT_NUM):
                pos_x = settings.Pheromone.CELL_SIZE * (x - (settings.Pheromone.WIDTH_NUM - 1) * 0.5)
                pos_y = settings.Pheromone.CELL_SIZE * (-y + (settings.Pheromone.HEIGHT_NUM - 1) * 0.5)
                pheromone_cell_pos[y, x, :] = [pos_x, pos_y]

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
            _pheromone_texture_id=pheromone_texture_id
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
            pheromone=self._pheromone.step()
        )

    @partial(jax.jit, static_argnames=("n", "unroll"), inline=True, donate_argnames=("self",))
    def step_n(self, model: mjx.Model, n: int, unroll: int = 1) -> Self:
        def body_fn(carry: "BasicSimulator", _x) -> tuple["BasicSimulator", None]:
            return carry.step(model), None

        return jax.lax.scan(body_fn, self, length=n, unroll=unroll)[0]

    @partial(jax.jit, inline=True)
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

        normalized_pheromone = pheromone / (max_pheromone + 1e-6)
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
