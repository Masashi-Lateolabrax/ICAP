from abc import ABC
import os
from typing import Optional
import logging

import mujoco
import numpy as np

from ..prelude import *
from ..environment import (
    add_geom,
    setup_option, setup_visual, setup_textures, add_nest, add_wall,
    rand_robot_pos, rand_food_pos, add_mesh_in_asset, MeshContentType,
    add_food_object_with_mesh, add_robot_with_mesh
)
from ..pheromone import PheromoneFieldCellSpec, PheromoneField, PheromoneFieldCell
from .basic import BasicMuJoCoSimulator


def add_pheromone_cells_in_mjspec(
        spec: mujoco.MjSpec,
        settings: Settings
) -> list[PheromoneFieldCellSpec]:
    from ..pheromone import add_pheromone_cell

    if settings.Pheromone.ACTIVE is False:
        return []

    sites = []
    for x in range(settings.Pheromone.WIDTH_NUM):
        for y in range(settings.Pheromone.HEIGHT_NUM):
            pos_x = settings.Pheromone.CELL_SIZE * (x - (settings.Pheromone.WIDTH_NUM - 1) * 0.5)
            pos_y = settings.Pheromone.CELL_SIZE * (-y + (settings.Pheromone.HEIGHT_NUM - 1) * 0.5)

            sites.append(
                add_pheromone_cell(
                    spec,
                    index_x=x,
                    index_y=y,
                    size=settings.Pheromone.CELL_SIZE * 0.5,
                    pos=(pos_x, pos_y, 0),
                )
            )
    return sites


def generate_mjspec(
        settings: Settings
) -> tuple[
    mujoco.MjSpec,
    mujoco._specs.MjsSite,
    list[RobotSpec],
    list[FoodSpec],
    list[PheromoneFieldCellSpec]
]:
    spec = mujoco.MjSpec()

    setup_option(spec, settings)
    setup_visual(spec, settings)
    setup_textures(spec, settings)

    add_wall(spec, settings)
    pheromone_cell_specs = add_pheromone_cells_in_mjspec(spec, settings)

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

    return spec, nest_spec, robot_specs, food_specs, pheromone_cell_specs


class BasicEnvironment(BasicMuJoCoSimulator, ABC):
    def __init__(self, settings, render: bool = False):
        mj_spec, nest_spec, robot_specs, food_specs, pheromone_cell_specs = generate_mjspec(settings)
        super().__init__(settings, mj_spec, render)

        self.nest_site = self.data.site(nest_spec.name)

        self.robot_specs = robot_specs
        self.food_specs = food_specs

        self._pheromone_field: Optional[PheromoneField] = None
        self._pheromone_cells: list[PheromoneFieldCell] = []
        self._pheromone_cell_pos: Optional[np.ndarray] = None
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
            self._pheromone_cells = [s.get_cell(self.model) for s in pheromone_cell_specs]
            self._pheromone_cell_pos = np.array(
                [cell.pos[:2] for cell in self._pheromone_cells], dtype=np.float32
            )

    def _get_pheromone_cells(self, positions: np.ndarray) -> list[PheromoneFieldCell]:
        if positions.ndim != 2 or positions.shape[1] < 2:
            logging.warning(f"Invalid position shape: expected (N, >=2), got {positions.shape}")
        if self._pheromone_cell_pos is None:
            logging.warning("Pheromone cell positions are not initialized.")
            return []

        distance = np.linalg.norm(
            positions[:, None, :2] - self._pheromone_cell_pos[None, :, :2],
            axis=2
        )
        closest_indices = np.argmin(distance, axis=1)
        return [self._pheromone_cells[i] for i in closest_indices]

    def add_pheromone(self, positions: np.ndarray, values: np.ndarray):
        for cell, v in zip(self._get_pheromone_cells(positions), values):
            cell.add_value += v

    def get_pheromone(self, positions: np.ndarray) -> np.ndarray:
        indexes = np.array([(cell.index_x, cell.index_y) for cell in self._get_pheromone_cells(positions)])
        return self._pheromone_field.get_gas(indexes[:, 0], indexes[:, 1])

    def get_total_liquid_pheromone(self) -> float:
        if self._pheromone_field is None:
            return 0.0
        return float(np.sum(self._pheromone_field.get_liquid_all()))

    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        if not self._do_render:
            return

        if self._pheromone_field:
            color_max = 1.0
            pheromone: np.ndarray = self._pheromone_field.get_gas_all()
            for cell in self._pheromone_cells:
                pheromone_value = float(pheromone[cell.index_y, cell.index_x])
                rgba: tuple[float, float, float] = (pheromone_value / color_max, 0.0, 1 - pheromone_value / color_max)
                cell.set_color(*rgba, 0.5)

        super().render(img_buf, pos, lookat)

    def reset(self):
        if self._pheromone_field:
            self._pheromone_field.reset()
            for cell in self._pheromone_cells:
                cell.add_value = 0.0

        super().reset()
