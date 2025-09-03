from abc import ABC
import os
from typing import Optional
import logging

import mujoco
import numpy as np
import jax.numpy as jnp

from ..prelude import *
from ..mkenv import (
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




class BasicMuJoCoSimulatorWithEnv(BasicMuJoCoSimulator, ABC):
    def __init__(self, settings, render: bool = False):
        mj_spec, nest_spec, robot_specs, food_specs = generate_mjspec(settings)
        pheromone_cell_specs: list[PheromoneFieldCellSpec] = add_pheromone_cells_in_mjspec(mj_spec, settings)
        super().__init__(settings, mj_spec, render)

        self.nest_id: int = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, nest_spec.name)

        self.batched_robot_id: BatchedRobotIDs = BatchedRobotIDs.from_specs(self.model, robot_specs)
        self.batched_food_id: BatchedFoodIDs = BatchedFoodIDs.from_specs(self.model, food_specs)

        distance_between_wheels = settings.Robot.DISTANCE_BETWEEN_WHEELS
        max_speed = settings.Robot.MAX_SPEED
        self.robots = BatchedRobots(self.data, self.batched_robot_id, distance_between_wheels, max_speed)

        self.food_items: BatchedFood = BatchedFood(self.data, self.batched_food_id)

        self._pheromone_field: Optional[PheromoneField] = None
        self._pheromone_cells: list[PheromoneFieldCell] = []
        self._pheromone_cell_pos: Optional[jnp.ndarray] = None
        if settings.Pheromone.ACTIVE:
            self._pheromone_field = PheromoneField(
                nx=settings.Pheromone.WIDTH_NUM,
                ny=settings.Pheromone.HEIGHT_NUM,
                dx=settings.Pheromone.CELL_SIZE,
                material=settings.Pheromone.MATERIAL,
                evaporation_rate=settings.Pheromone.EVAPORATION_RATE,
                decrease_rate=settings.Pheromone.DECREASE_RATE,
                temperature=settings.Simulation.TEMPERATURE,
                iter_=settings.Pheromone.ITERATIONS_PER_STEP,
            )
            self._pheromone_cells = [s.get_cell(self.model) for s in pheromone_cell_specs]
            self._pheromone_cell_pos = jnp.array(
                [cell.pos[:2] for cell in self._pheromone_cells], dtype=jnp.float32
            )

    def _get_pheromone_cells(self, positions: jnp.ndarray) -> list[PheromoneFieldCell]:
        if positions.ndim != 2 or positions.shape[1] < 2:
            logging.warning(f"Invalid position shape: expected (N, >=2), got {positions.shape}")
        if self._pheromone_cell_pos is None:
            logging.warning("Pheromone cell positions are not initialized.")
            return []

        distance = jnp.linalg.norm(
            positions[:, None, :2] - self._pheromone_cell_pos[None, :, :2],
            axis=2
        )
        closest_indices = jnp.argmin(distance, axis=1)
        return [self._pheromone_cells[i] for i in closest_indices]

    def add_pheromone(self, positions: jnp.ndarray, values: jnp.ndarray):
        for cell, v in zip(self._get_pheromone_cells(positions), values):
            cell.add_value += v

    def get_pheromone(self, positions: jnp.ndarray) -> jnp.ndarray:
        indexes = jnp.array([(cell.index_x, cell.index_y) for cell in self._get_pheromone_cells(positions)])
        return self._pheromone_field.get_gas(indexes[:, 0], indexes[:, 1])

    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        if not self._do_render:
            return

        if self._pheromone_field:
            color_max = 1.0
            pheromone: jnp.ndarray = self._pheromone_field.values_gas
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
