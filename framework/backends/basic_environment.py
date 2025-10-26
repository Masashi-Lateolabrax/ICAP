from abc import ABC
import os
from typing import Optional
import logging

import mujoco
import numpy as np

from ..prelude import *
from ..environment import (
    add_geom, add_texture, add_material,
    setup_option, setup_visual, setup_textures, add_nest, add_wall,
    rand_robot_pos, rand_food_pos, add_mesh_in_asset, MeshContentType,
    add_food_object_with_mesh, add_robot_with_mesh
)
from ..pheromone import PheromoneField
from .basic import BasicMuJoCoSimulator


def add_pheromone_visualization(spec: mujoco.MjSpec, settings: Settings) -> None:
    """Add runtime-updatable texture for pheromone visualization."""
    if not settings.Pheromone.ACTIVE:
        return

    # Create custom data texture for pheromone field
    add_texture(
        spec,
        name="pheromone_viz",
        type_=mujoco.mjtTexture.mjTEXTURE_2D,
        width=settings.Pheromone.WIDTH_NUM,
        height=settings.Pheromone.HEIGHT_NUM,
        data=np.zeros((settings.Pheromone.HEIGHT_NUM, settings.Pheromone.WIDTH_NUM, 3), dtype=np.uint8)
    )

    # Create material using the pheromone texture
    add_material(spec, name="pheromone_mat", texture="pheromone_viz", texrepeat=(1, 1))

    # Add thin box geometry for pheromone overlay (positioned above ground)
    add_geom(
        spec.worldbody,
        geom_type=mujoco.mjtGeom.mjGEOM_BOX,
        name="pheromone_overlay",
        size=(settings.Simulation.WORLD_WIDTH * 0.5, settings.Simulation.WORLD_HEIGHT * 0.5, 0.01),
        pos=(0, 0, 0.01),
        material="pheromone_mat",
        rgba=(1, 1, 1, 0.7),
        contype=0,  # Disable collision detection
        conaffinity=0  # Disable collision affinity
    )


def generate_mjspec(
        settings: Settings
) -> tuple[
    mujoco.MjSpec,
    mujoco._specs.MjsSite,
    list[RobotSpec],
    list[FoodSpec]
]:
    spec = mujoco.MjSpec()

    setup_option(spec, settings)
    setup_visual(spec, settings)
    setup_textures(spec, settings)

    add_wall(spec, settings)
    add_pheromone_visualization(spec, settings)

    add_geom(
        spec.worldbody,
        geom_type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0, 0, 0),
        size=(
            settings.Simulation.WORLD_WIDTH * 0.5,
            settings.Simulation.WORLD_HEIGHT * 0.5,
            1
        ),
        material="ground",
        rgba=GROUND_COLOR,
        condim=GROUND_COLLISION_CONDIM
    )

    nest_spec = add_nest(spec, settings)

    invalid_area: list[tuple[Position, float]] = []

    # Create food objects
    food_specs = []
    if settings.Food.NUM > 0:
        food_mesh = add_mesh_in_asset(
            spec,
            name="food_mesh",
            file=os.path.abspath(os.path.join(settings.Storage.ASSET_DIRECTORY, "food-object.stl")),
            content_type=MeshContentType.STL,
            inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
        )

        for i in range(settings.Food.NUM):
            if i < len(settings.Food.INITIAL_POSITION):
                position: Position = settings.Food.INITIAL_POSITION[i]
            else:
                position: Position = rand_food_pos(settings, invalid_area)
            invalid_area.append(
                (position, settings.Food.RADIUS)
            )
            food_specs.append(
                add_food_object_with_mesh(spec, settings, food_mesh, i, position)
            )

    # Create robots
    robot_specs = []
    if settings.Robot.NUM > 0:
        robot_mesh = add_mesh_in_asset(
            spec,
            name="robot_mesh",
            file=os.path.abspath(os.path.join(settings.Storage.ASSET_DIRECTORY, "robot-object.stl")),
            content_type=MeshContentType.STL,
            inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
        )

        for i in range(settings.Robot.NUM):
            if i < len(settings.Robot.INITIAL_POSITION):
                position: RobotLocation = settings.Robot.INITIAL_POSITION[i]
            else:
                position: RobotLocation = rand_robot_pos(settings, invalid_area)
            invalid_area.append(
                (position.position, settings.Robot.RADIUS)
            )
            robot_specs.append(
                add_robot_with_mesh(spec, settings, robot_mesh, i, position)
            )

    return spec, nest_spec, robot_specs, food_specs


class BasicEnvironment(BasicMuJoCoSimulator, ABC):
    def __init__(self, settings, render: bool = False):
        mj_spec, nest_spec, robot_specs, food_specs = generate_mjspec(settings)
        super().__init__(settings, mj_spec, render)

        self.nest_site = self.data.site(nest_spec.name)

        self.robot_specs = robot_specs
        self.food_specs = food_specs

        self._shadow = []

        self._pheromone_field: Optional[PheromoneField] = None
        self._pheromone_texture_id: Optional[int] = None

        if settings.Pheromone.ACTIVE:
            self._pheromone_field = PheromoneField(
                nx=settings.Pheromone.WIDTH_NUM,
                ny=settings.Pheromone.HEIGHT_NUM,
                dx=settings.Pheromone.CELL_SIZE,
                material=settings.Pheromone.MATERIAL,
                temperature=settings.Pheromone.TEMPERATURE,
                dt=settings.Simulation.TIME_STEP,
                iter_=settings.Pheromone.ITERATIONS_PER_STEP,
            )
            self._pheromone_texture_id = self.model.texture("pheromone_viz").id

    def _calc_pheromone_indices(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculate pheromone grid indices from world positions."""
        if self._pheromone_field is None:
            return np.array([]), np.array([])

        cell_size = self._pheromone_field.dx
        width = self._pheromone_field.shape[1]
        height = self._pheromone_field.shape[0]

        # Convert world coordinates to grid indices
        x_idx = np.round(positions[:, 0] / cell_size + (width - 1) * 0.5).astype(np.int32)
        y_idx = np.round((height - 1) * 0.5 - positions[:, 1] / cell_size).astype(np.int32)

        # Clip to valid range
        x_idx = np.clip(x_idx, 0, width - 1)
        y_idx = np.clip(y_idx, 0, height - 1)

        return x_idx, y_idx

    def add_pheromone(self, positions: np.ndarray, values: np.ndarray):
        if self._pheromone_field is None:
            return
        x_idx, y_idx = self._calc_pheromone_indices(positions)
        self._pheromone_field.add_liquid(x_idx, y_idx, values)

    def get_pheromone(self, positions: np.ndarray) -> np.ndarray:
        if self._pheromone_field is None:
            return np.zeros(len(positions))
        x_idx, y_idx = self._calc_pheromone_indices(positions)
        return self._pheromone_field.get_gas(x_idx, y_idx)

    def get_pheromone_grad(self, positions: np.ndarray) -> np.ndarray:
        if self._pheromone_field is None:
            return np.zeros((len(positions), 2))
        x_idx, y_idx = self._calc_pheromone_indices(positions)
        return self._pheromone_field.get_grad(x_idx, y_idx)

    def get_total_liquid_pheromone(self) -> float:
        if self._pheromone_field is None:
            return 0.0
        return float(np.sum(self._pheromone_field.get_liquid_all()))

    def _update_pheromone_texture(self):
        """Update pheromone texture with current field values."""
        if self._pheromone_field is None or self._pheromone_texture_id is None:
            return

        pheromone = self._pheromone_field.get_gas_all()
        max_pheromone = 0.1
        normalized_pheromone = np.clip(pheromone / (max_pheromone + 1e-6), a_min=0, a_max=1)

        # Create RGB texture data (Red-Blue gradient)
        texture_data = np.stack([
            (normalized_pheromone * 255).astype(np.uint8),
            np.zeros_like(normalized_pheromone, dtype=np.uint8),
            ((1 - normalized_pheromone) * 255).astype(np.uint8),
        ], axis=-1)

        # Update MuJoCo texture
        tex_id = self._pheromone_texture_id
        tex_start = self.model.tex_adr[tex_id]
        tex_size = self.model.tex_height[tex_id] * self.model.tex_width[tex_id] * 3
        self.model.tex_data[tex_start:tex_start + tex_size] = texture_data.flatten()

    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        if not self._do_render:
            return

        if self._pheromone_field:
            self._update_pheromone_texture()

        super().render(img_buf, pos, lookat)

        # while len(self._shadow) > 150:
        #     self._shadow.pop(0)
        #
        # robot_mask = (img_buf[:, :, 0] > 126) * (img_buf[:, :, 1] > 126) * (img_buf[:, :, 2] < 126)
        # self._shadow.append(img_buf[:, :, :] * robot_mask[:, :, None])
        #
        # shadow = np.zeros_like(img_buf)
        # for s in self._shadow:
        #     shadow = np.maximum(shadow * 0.99, s)
        #
        # img_buf *= np.logical_not(np.sum(shadow, axis=2) > 0)[:, :, None]
        # img_buf += shadow.astype(np.uint8)

    def reset(self):
        if self._pheromone_field:
            self._pheromone_field.reset()

        super().reset()
